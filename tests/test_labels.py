import json

import numpy as np
import pandas as pd
import pytest

from rsna_knee_abnormality_detection.constants import STUDY_ID, TARGETS
from rsna_knee_abnormality_detection.labels import prepare_labels, read_study_csv


def frame(ids, value=np.nan):
    rows = pd.DataFrame(value, index=range(len(ids)), columns=TARGETS)
    rows.insert(0, STUDY_ID, ids)
    return rows


def test_expert_override_preserves_soft_and_unknown_targets(tmp_path):
    train = frame(["001", "002", "003"])
    train.loc[0, TARGETS] = 0
    train.loc[0, "ACL"] = 1
    train["Report"] = ["a report", "not mentioned", "another report"]
    train_csv = tmp_path / "train.csv"
    train.to_csv(train_csv, index=False)
    weak = frame(["002", "001"], 0.25)
    weak.loc[weak[STUDY_ID] == "002", "Synovitis"] = np.nan
    weak_csv = tmp_path / "weak.csv"
    weak.to_csv(weak_csv, index=False)
    path = prepare_labels(train_csv, tmp_path / "out", weak_csv, source="test-source-v1")
    labels = read_study_csv(path).set_index(STUDY_ID)
    assert labels.loc["001", "ACL"] == 1
    assert labels.loc["001", "MCL"] == 0
    assert labels.loc["002", "ACL"] == 0.25
    assert np.isnan(labels.loc["002", "Synovitis"])
    assert labels.loc["003", TARGETS].isna().all()
    assert labels["label_source"].to_dict() == {"001": "expert", "002": "report", "003": "missing"}
    masks = pd.read_csv(path.parent / "label_masks.csv", dtype={STUDY_ID: str}).set_index(STUDY_ID)
    assert masks.loc["002", "Synovitis"] == 0
    assert masks.loc["002", "ACL"] == 1
    sources = pd.read_csv(path.parent / "label_sources.csv", dtype={STUDY_ID: str}).set_index(STUDY_ID)
    assert sources.loc["001", "ACL"] == "expert"
    assert sources.loc["002", "ACL"] == "report"
    assert sources.loc["002", "Synovitis"] == "missing"
    audit = json.loads((path.parent / "label_provenance.json").read_text())
    assert audit["report_label_source"] == "test-source-v1"
    assert audit["report_vs_expert_diagnostics"]["ACL"]["shared_expert_studies"] == 1
    assert audit["coverage"]["Synovitis"]["missing"] == 2
    assert (path.parent / "original_train.csv").read_bytes() == train_csv.read_bytes()
    assert (path.parent / "original_report_labels.csv").read_bytes() == weak_csv.read_bytes()


def test_no_report_import_does_not_use_report_text(tmp_path):
    train = frame(["a", "b"])
    train["Report"] = ["ACL is normal", "an ACL tear"]
    train.to_csv(tmp_path / "train.csv", index=False)
    with pytest.warns(UserWarning, match="remain unlabeled"):
        path = prepare_labels(tmp_path / "train.csv", tmp_path / "labels")
    labels = read_study_csv(path)
    assert labels[TARGETS].isna().all().all()
    assert labels["label_source"].eq("missing").all()


@pytest.mark.parametrize("invalid", [-0.01, 1.01, np.inf, -np.inf, "positive"])
def test_invalid_report_targets_fail_before_writing(tmp_path, invalid):
    train = frame(["a"])
    train.to_csv(tmp_path / "train.csv", index=False)
    weak = frame(["a"]).astype(object)
    weak.loc[0, "ACL"] = invalid
    weak.to_csv(tmp_path / "weak.csv", index=False)
    with pytest.raises(ValueError, match="ACL"):
        prepare_labels(tmp_path / "train.csv", tmp_path / "out", tmp_path / "weak.csv")
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize("ids,match", [(["a", "a"], "duplicate"), (["b"], "absent"), ([" a"], "whitespace"), ([""], "blank")])
def test_invalid_report_study_ids_are_rejected(tmp_path, ids, match):
    frame(["a"]).to_csv(tmp_path / "train.csv", index=False)
    frame(ids, 0).to_csv(tmp_path / "weak.csv", index=False)
    with pytest.raises(ValueError, match=match):
        prepare_labels(tmp_path / "train.csv", tmp_path / "out", tmp_path / "weak.csv")


def test_expert_labels_must_be_binary(tmp_path):
    frame(["a"], 0.7).to_csv(tmp_path / "train.csv", index=False)
    with pytest.raises(ValueError, match="Expert targets"):
        prepare_labels(tmp_path / "train.csv", tmp_path / "out")


def test_duplicate_target_columns_are_rejected(tmp_path):
    path = tmp_path / "invalid.csv"
    path.write_text(
        ",".join([STUDY_ID, *TARGETS, "ACL"]) + "\n" + ",".join(["a", *(["0"] * 13)]) + "\n"
    )
    with pytest.raises(ValueError, match="duplicate columns"):
        read_study_csv(path)


def test_partial_expert_labels_override_only_known_entries(tmp_path):
    train = frame(["a"])
    train.loc[0, "ACL"] = 1
    train.to_csv(tmp_path / "train.csv", index=False)
    frame(["a"], 0.25).to_csv(tmp_path / "weak.csv", index=False)
    path = prepare_labels(tmp_path / "train.csv", tmp_path / "out", tmp_path / "weak.csv")
    labels = read_study_csv(path)
    assert labels.loc[0, "ACL"] == 1
    assert labels.loc[0, "MCL"] == 0.25
    assert labels.loc[0, "label_source"] == "expert"
