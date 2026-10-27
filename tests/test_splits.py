import json
import warnings

import numpy as np
import pandas as pd
import pytest

from rsna_knee_abnormality_detection.constants import STUDY_ID, TARGETS
from rsna_knee_abnormality_detection.splits import make_splits


def labels_frame(n=32):
    rng = np.random.default_rng(12)
    labels = pd.DataFrame(rng.integers(0, 2, size=(n, len(TARGETS))).astype(float), columns=TARGETS)
    labels.insert(0, STUDY_ID, [f"study-{i:03d}" for i in range(n)])
    labels["label_source"] = "report"
    labels.loc[0, "label_source"] = "expert"
    return labels


def run_splits(*args, **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        return make_splits(*args, **kwargs)


def test_expert_patient_group_reserved_and_other_groups_stay_together(tmp_path):
    labels = labels_frame()
    labels.loc[31, TARGETS] = np.nan
    labels.loc[31, "label_source"] = "missing"
    labels.to_csv(tmp_path / "labels.csv", index=False)
    groups = labels[[STUDY_ID]].copy()
    groups["group_id"] = [f"patient-{i // 2:02d}" for i in range(len(labels))]
    groups.to_csv(tmp_path / "groups.csv", index=False)
    path = run_splits(tmp_path / "labels.csv", tmp_path / "out", n_folds=5, groups_csv=tmp_path / "groups.csv")
    result = pd.read_csv(path).set_index(STUDY_ID)
    assert result.loc["study-000", "fold"] == -1
    assert result.loc["study-001", "fold"] == -1
    assert result.loc["study-001", "split"] == "expert_holdout"
    assert result.loc["study-031", "fold"] == result.loc["study-030", "fold"]
    assert result.loc["study-031", "split"] == "unlabeled"
    train = result[result["split"] == "train"]
    assert train.groupby("group_id")["fold"].nunique().max() == 1
    assert result.groupby("group_id")["fold"].nunique().max() == 1
    assert set(train["fold"]) == set(range(5))
    assert not set(train["group_id"]) & set(result.loc[result["split"] == "expert_holdout", "group_id"])
    summary = json.loads((path.parent / "split_summary.json").read_text())
    assert summary["expert_studies"] == 1
    assert summary["reserved_expert_group_studies"] == 2
    counts = pd.read_csv(path.parent / "fold_target_counts.csv")
    assert len(counts) == 6 * len(TARGETS)


def test_unknown_groupmates_inherit_patient_fold_but_not_supervised_status(tmp_path):
    labels = labels_frame()
    unknown_indices = [1, 28, 29, 31]
    labels.loc[unknown_indices, TARGETS] = np.nan
    labels.loc[unknown_indices, "label_source"] = "missing"
    labels_path = tmp_path / "labels.csv"
    labels.to_csv(labels_path, index=False)
    original_labels = labels_path.read_bytes()
    groups = labels[[STUDY_ID]].copy()
    groups["group_id"] = [f"patient-{i // 2:02d}" for i in range(len(labels))]
    groups.to_csv(tmp_path / "groups.csv", index=False)
    path = run_splits(labels_path, tmp_path / "out", groups_csv=tmp_path / "groups.csv")
    result = pd.read_csv(path).set_index(STUDY_ID)
    # Expert patients stay entirely outside all model fitting, including their
    # unknown studies, which would otherwise be tempting self-training inputs.
    assert result.loc[["study-000", "study-001"], "fold"].eq(-1).all()
    assert result.loc[["study-000", "study-001"], "split"].eq("expert_holdout").all()
    # The labeled and unknown studies of another patient share a fold, but only
    # the former participates in supervised training or validation.
    assert result.loc["study-030", "fold"] >= 0
    assert result.loc["study-031", "fold"] == result.loc["study-030", "fold"]
    assert result.loc["study-030", "split"] == "train"
    assert result.loc["study-031", "split"] == "unlabeled"
    # A patient without any expert or report targets has no assigned fold.
    assert result.loc[["study-028", "study-029"], "fold"].eq(-2).all()
    assert result.loc[["study-028", "study-029"], "split"].eq("unlabeled").all()
    assert result.groupby("group_id")["fold"].nunique().eq(1).all()
    assert labels_path.read_bytes() == original_labels
    summary = json.loads((path.parent / "split_summary.json").read_text())
    assert summary["unlabeled_studies_with_assigned_fold"] == 1


def test_deterministic_when_input_rows_change_order(tmp_path):
    labels = labels_frame()
    labels.to_csv(tmp_path / "labels.csv", index=False)
    first = pd.read_csv(run_splits(tmp_path / "labels.csv", tmp_path / "out-a", seed=99))
    labels.sample(frac=1, random_state=5).to_csv(tmp_path / "shuffled.csv", index=False)
    second = pd.read_csv(run_splits(tmp_path / "shuffled.csv", tmp_path / "out-b", seed=99))
    pd.testing.assert_frame_equal(first, second)


def test_soft_targets_are_not_changed_and_missing_stays_unlabeled(tmp_path):
    labels = labels_frame()
    labels.loc[1:, "ACL"] = 0.7
    labels.loc[3, TARGETS] = np.nan
    labels.loc[3, "label_source"] = "missing"
    path = tmp_path / "labels.csv"
    labels.to_csv(path, index=False)
    original_bytes = path.read_bytes()
    result = pd.read_csv(run_splits(path, tmp_path / "out"))
    assert path.read_bytes() == original_bytes
    assert result.loc[result[STUDY_ID] == "study-003", "fold"].item() == -2
    assert result.loc[result[STUDY_ID] == "study-003", "split"].item() == "unlabeled"


@pytest.mark.parametrize("mutation,match", [("missing_study", "exactly"), ("extra_study", "exactly"), ("duplicate_study", "duplicate"), ("blank_group", "blank")])
def test_invalid_group_mapping_is_rejected(tmp_path, mutation, match):
    labels = labels_frame()
    labels.to_csv(tmp_path / "labels.csv", index=False)
    groups = labels[[STUDY_ID]].copy()
    groups["group_id"] = labels[STUDY_ID]
    if mutation == "missing_study":
        groups = groups.iloc[1:]
    elif mutation == "extra_study":
        groups.loc[len(groups)] = ["unknown", "unknown"]
    elif mutation == "duplicate_study":
        groups.loc[1, STUDY_ID] = groups.loc[0, STUDY_ID]
    elif mutation == "blank_group":
        groups.loc[0, "group_id"] = ""
    groups.to_csv(tmp_path / "groups.csv", index=False)
    with pytest.raises(ValueError, match=match):
        make_splits(tmp_path / "labels.csv", tmp_path / "out", groups_csv=tmp_path / "groups.csv")


def test_too_few_groups_after_expert_reservation_fails(tmp_path):
    labels = labels_frame(4)
    labels.to_csv(tmp_path / "labels.csv", index=False)
    with pytest.raises(ValueError, match="after reserving the expert holdout"):
        make_splits(tmp_path / "labels.csv", tmp_path / "out", n_folds=5)


def test_study_group_fallback_is_explicit(tmp_path):
    labels_frame().to_csv(tmp_path / "labels.csv", index=False)
    with pytest.warns(UserWarning, match="repeated patients cannot be ruled out"):
        make_splits(tmp_path / "labels.csv", tmp_path / "out")


def test_missing_source_and_inconsistent_masks_are_rejected(tmp_path):
    labels = labels_frame()
    labels.drop(columns="label_source").to_csv(tmp_path / "labels.csv", index=False)
    with pytest.raises(ValueError, match="label_source"):
        make_splits(tmp_path / "labels.csv", tmp_path / "out")
    labels.loc[2, "label_source"] = "missing"
    labels.to_csv(tmp_path / "labels.csv", index=False)
    with pytest.raises(ValueError, match="disagrees"):
        make_splits(tmp_path / "labels.csv", tmp_path / "out")
