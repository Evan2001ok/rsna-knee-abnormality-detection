"""Round-3 labels mix in out-of-fold predictions with a fixed share set by how reliable each label is."""

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from test_student_labels import MODULE as STUDENT
from test_student_labels import build

from rsna_knee_abnormality_detection.constants import STUDY_ID, TARGETS

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
SPEC = importlib.util.spec_from_file_location("prepare_round3_labels", ROOT / "scripts/prepare_round3_labels.py")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def setup(tmp_path):
    paths, runs = build(tmp_path)
    rng = np.random.default_rng(3)
    raptor = pd.read_csv(paths["raptor"], dtype={STUDY_ID: str})
    gpt = raptor[[STUDY_ID]].copy()
    for name in TARGETS:
        gpt[name] = rng.uniform(0, 1, len(gpt))
    gpt.to_csv(tmp_path / "gpt.csv", index=False)
    return paths, runs


def test_shares_follow_reliability_and_other_rows_stay(tmp_path):
    paths, runs = setup(tmp_path)
    out = MODULE.prepare_round3_labels(paths["raptor"], paths["baseline"], paths["raptor"], tmp_path / "gpt.csv", paths["splits"],
                                       runs, runs, tmp_path / "round3")
    result = pd.read_csv(out, dtype={STUDY_ID: str}).set_index(STUDY_ID)
    current = pd.read_csv(paths["raptor"], dtype={STUDY_ID: str}).set_index(STUDY_ID)
    baseline = pd.read_csv(paths["baseline"], dtype={STUDY_ID: str}).set_index(STUDY_ID)
    gpt = pd.read_csv(tmp_path / "gpt.csv", dtype={STUDY_ID: str}).set_index(STUDY_ID)
    seen = {0.35: 0, 0.65: 0}
    for run in runs:
        weights = json.loads((run / "provenance.json").read_text())["training_objective"]["positive_term_weights"]
        predictions = pd.read_csv(run / STUDENT.PREDICTIONS_FILE, dtype={STUDY_ID: str}).set_index(STUDY_ID)
        for study, row in predictions.iterrows():
            for name in TARGETS:
                model = STUDENT.to_label_scale(row[name], weights[name])
                votes = {current.loc[study, name] >= 0.5, gpt.loc[study, name] >= 0.5, model >= 0.5}
                share = 0.35 if (not np.isnan(baseline.loc[study, name]) and len(votes) == 1) else 0.65
                seen[share] += 1
                assert result.loc[study, name] == pytest.approx((1 - share) * current.loc[study, name] + share * model)
    assert seen[0.35] > 0 and seen[0.65] > 0
    pd.testing.assert_frame_equal(result.loc[["expert", "unlabeled"]], current.loc[["expert", "unlabeled"]])


def test_missing_gpt_labels_are_refused(tmp_path):
    paths, runs = setup(tmp_path)
    gpt = pd.read_csv(tmp_path / "gpt.csv", dtype={STUDY_ID: str}).iloc[1:]
    gpt.to_csv(tmp_path / "gpt.csv", index=False)
    with pytest.raises(ValueError, match="GPT labels are missing"):
        MODULE.prepare_round3_labels(paths["raptor"], paths["baseline"], paths["raptor"], tmp_path / "gpt.csv", paths["splits"],
                                     runs, runs, tmp_path / "round3")
