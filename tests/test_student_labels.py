"""Student labels change only unmentioned cells, using predictions from models that never trained on the study."""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from rsna_knee_abnormality_detection.constants import STUDY_ID, TARGETS

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("prepare_student_labels", ROOT / "scripts/prepare_student_labels.py")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def build(tmp_path):
    rng = np.random.default_rng(0)
    folds = {f"s{fold}{index}": fold for fold in range(5) for index in range(2)}
    ids = [*folds, "expert", "unlabeled"]
    splits = pd.DataFrame({STUDY_ID: ids, "fold": [*folds.values(), -1, -2],
                           "split": ["train"] * 10 + ["expert_holdout", "unlabeled"], "group_id": ids})
    raptor = pd.DataFrame(rng.choice([0.05, 0.2, 0.35, 0.8, 0.95], size=(12, 12)), columns=TARGETS)
    raptor.loc[10, TARGETS] = rng.integers(0, 2, 12)
    raptor.insert(0, STUDY_ID, ids)
    raptor["label_source"] = ["report"] * 10 + ["expert", "report"]
    baseline = pd.DataFrame(rng.integers(0, 2, size=(12, 12)).astype(float), columns=TARGETS)
    baseline[baseline.lt(1) & pd.DataFrame(rng.random((12, 12)) < 0.4, columns=TARGETS)] = np.nan
    baseline.loc[10, TARGETS] = raptor.loc[10, TARGETS].to_numpy()
    baseline.loc[11, TARGETS] = np.nan
    baseline.insert(0, STUDY_ID, ids)
    baseline["label_source"] = ["report"] * 10 + ["expert", "missing"]
    paths = {name: tmp_path / name / "labels.csv" for name in ("raptor", "baseline")}
    for name, frame in (("raptor", raptor), ("baseline", baseline)):
        paths[name].parent.mkdir()
        frame.to_csv(paths[name], index=False)
    paths["splits"] = tmp_path / "splits.csv"
    splits.to_csv(paths["splits"], index=False)
    runs = []
    for fold in range(5):
        run = tmp_path / f"teacher{fold}"
        run.mkdir()
        validation = [study for study, value in folds.items() if value == fold]
        (run / "config.yaml").write_text(yaml.safe_dump({"data": {"fold": fold}}))
        weights = {name: 1.0 + fold + index / 4 for index, name in enumerate(TARGETS)}
        (run / "provenance.json").write_text(json.dumps({
            "training_objective": {"positive_term_weights": weights}, "validation_ids": validation,
            "training_ids": [study for study in folds if study not in validation]}))
        predictions = pd.DataFrame(rng.uniform(0.01, 0.99, size=(2, 12)), columns=TARGETS)
        predictions.insert(0, STUDY_ID, validation)
        predictions.to_csv(run / MODULE.PREDICTIONS_FILE, index=False)
        runs.append(run)
    return paths, runs


def prepare(paths, runs, tmp_path):
    return MODULE.prepare_student_labels(paths["raptor"], paths["baseline"], paths["splits"], runs, tmp_path / "student")


def test_label_scale_conversion_undoes_the_positive_weight():
    assert MODULE.to_label_scale(0.5, 3.0) == pytest.approx(0.25)
    assert MODULE.to_label_scale(0.8, 1.0) == pytest.approx(0.8)


def test_only_unmentioned_cells_change_and_follow_the_fixed_mix(tmp_path):
    paths, runs = build(tmp_path)
    result = pd.read_csv(prepare(paths, runs, tmp_path), dtype={STUDY_ID: str})
    raptor = pd.read_csv(paths["raptor"], dtype={STUDY_ID: str})
    baseline = pd.read_csv(paths["baseline"], dtype={STUDY_ID: str})
    pd.testing.assert_frame_equal(result.iloc[10:], raptor.iloc[10:])
    assert result["label_source"].equals(raptor["label_source"])
    changed = 0
    for run in runs:
        weights = json.loads((run / "provenance.json").read_text())["training_objective"]["positive_term_weights"]
        predictions = pd.read_csv(run / MODULE.PREDICTIONS_FILE, dtype={STUDY_ID: str})
        for _, row in predictions.iterrows():
            index = raptor.index[raptor[STUDY_ID].eq(row[STUDY_ID])][0]
            for name in TARGETS:
                if np.isnan(baseline.loc[index, name]):
                    expected = 0.5 * raptor.loc[index, name] + 0.5 * MODULE.to_label_scale(row[name], weights[name])
                    changed += 1
                else:
                    expected = raptor.loc[index, name]
                assert result.loc[index, name] == pytest.approx(expected)
    record = json.loads((tmp_path / "student/label_provenance.json").read_text())
    assert record["changed_cells"] == changed > 0 and record["outcome_labels_read"] is False


def test_refuses_a_prediction_from_a_model_that_trained_on_the_study(tmp_path):
    paths, runs = build(tmp_path)
    provenance = json.loads((runs[2] / "provenance.json").read_text())
    provenance["training_ids"].append(provenance["validation_ids"][0])
    (runs[2] / "provenance.json").write_text(json.dumps(provenance))
    with pytest.raises(ValueError, match="used to train"):
        prepare(paths, runs, tmp_path)


@pytest.mark.parametrize("mistake", ["missing_row", "duplicate_fold", "swapped_folds"])
def test_refuses_missing_duplicate_or_misassigned_predictions(tmp_path, mistake):
    paths, runs = build(tmp_path)
    if mistake == "missing_row":
        path = runs[1] / MODULE.PREDICTIONS_FILE
        pd.read_csv(path).iloc[1:].to_csv(path, index=False)
    elif mistake == "duplicate_fold":
        runs[4] = runs[3]
    else:
        for run, fold in ((runs[2], 3), (runs[3], 2)):
            (run / "config.yaml").write_text(yaml.safe_dump({"data": {"fold": fold}}))
    with pytest.raises(ValueError):
        prepare(paths, runs, tmp_path)


def test_refuses_to_write_next_to_the_input_labels(tmp_path):
    paths, runs = build(tmp_path)
    with pytest.raises(ValueError, match="output directory"):
        MODULE.prepare_student_labels(paths["raptor"], paths["baseline"], paths["splits"], runs, paths["raptor"].parent)
