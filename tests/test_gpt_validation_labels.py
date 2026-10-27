"""Checkpoint-selection labels (the "agreement labels" of docs/WRITEUP.md) keep report cells GPT agrees with and fill
silent cells only from confident GPT reads."""

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from rsna_knee_abnormality_detection.constants import STUDY_ID, TARGETS

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("prepare_gpt_validation_labels", ROOT / "scripts/prepare_gpt_validation_labels.py")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def build(tmp_path, gpt_ids=("s0", "s1")):
    ids = ["s0", "s1", "expert", "unlabeled"]
    pd.DataFrame({STUDY_ID: ids, "fold": [0, 1, -1, -2], "split": ["train", "train", "expert_holdout", "unlabeled"],
                  "group_id": ids}).to_csv(tmp_path / "splits.csv", index=False)
    severity = pd.DataFrame(np.nan, index=range(4), columns=TARGETS)
    severity.loc[0, ["ACL", "MCL", "Effusion"]] = [1.0, 0.0, 1.0]   # stated cells for s0
    severity.loc[2, TARGETS] = 1.0                                     # expert row
    severity.insert(0, STUDY_ID, ids)
    severity["label_source"] = ["report", "report", "expert", "missing"]
    (tmp_path / "severity").mkdir()
    severity.to_csv(tmp_path / "severity" / "labels.csv", index=False)
    gpt = pd.DataFrame(0.5, index=range(len(gpt_ids)), columns=TARGETS)
    gpt.loc[0, ["ACL", "MCL", "Effusion", "Fracture", "Contusion"]] = [0.8, 0.7, 0.2, 0.95, 0.05]
    gpt.insert(0, STUDY_ID, list(gpt_ids))
    gpt.to_csv(tmp_path / "gpt.csv", index=False)
    return tmp_path / "severity" / "labels.csv", tmp_path / "gpt.csv", tmp_path / "splits.csv"


def test_agreement_kept_disagreement_dropped_and_confident_silent_cells_filled(tmp_path):
    severity, gpt, splits = build(tmp_path)
    labels = pd.read_csv(MODULE.prepare_gpt_validation_labels(severity, gpt, splits, tmp_path / "out"),
                         dtype={STUDY_ID: str}).set_index(STUDY_ID)
    row = labels.loc["s0"]
    assert row["ACL"] == 1.0                      # report 1, GPT 0.8: agree
    assert np.isnan(row["MCL"]) and np.isnan(row["Effusion"])  # report 0 vs GPT 0.7; report 1 vs GPT 0.2
    assert row["Fracture"] == 1.0 and row["Contusion"] == 0.0  # silent, GPT confident
    assert np.isnan(row["Synovitis"])             # silent, GPT 0.5: stays blank
    assert labels.loc["s1", TARGETS].isna().all()
    assert (labels.loc["expert", TARGETS] == 1.0).all() and labels.loc["unlabeled", TARGETS].isna().all()


def test_missing_gpt_labels_for_a_validation_study_are_refused(tmp_path):
    severity, gpt, splits = build(tmp_path, gpt_ids=("s0",))
    with pytest.raises(ValueError, match="missing for 1"):
        MODULE.prepare_gpt_validation_labels(severity, gpt, splits, tmp_path / "out")
