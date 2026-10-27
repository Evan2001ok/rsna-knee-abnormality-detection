"""Severity-aware validation labels drop only mild mentions of threshold findings."""

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from rsna_knee_abnormality_detection.constants import STUDY_ID, TARGETS

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("prepare_severity_validation_labels",
                                              ROOT / "scripts/prepare_severity_validation_labels.py")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def build(tmp_path):
    rng = np.random.default_rng(2)
    ids = [f"s{index}" for index in range(8)] + ["expert", "missing"]
    verdicts = rng.choice(["YES", "NO", "UNK"], size=(10, 12))
    scores = np.where(verdicts == "YES", rng.choice([0.68, 0.82, 0.94], size=(10, 12)),
                      np.where(verdicts == "NO", 0.08, 0.28))
    pilkwang = pd.DataFrame({STUDY_ID: ids})
    for column, name in enumerate(TARGETS):
        pilkwang[name] = scores[:, column]
        pilkwang[f"{name}__verdict"] = verdicts[:, column]
    baseline = pd.DataFrame({name: [MODULE.VERDICT_VALUES.get(value, np.nan) for value in verdicts[:, column]]
                             for column, name in enumerate(TARGETS)})
    baseline.loc[8, TARGETS] = rng.integers(0, 2, 12)
    baseline.loc[9, TARGETS] = np.nan
    baseline.insert(0, STUDY_ID, ids)
    baseline["label_source"] = ["report"] * 8 + ["expert", "missing"]
    paths = {"baseline": tmp_path / "baseline/labels.csv", "pilkwang": tmp_path / "pilkwang/report_labels_v2.csv"}
    for name, frame in (("baseline", baseline), ("pilkwang", pilkwang)):
        paths[name].parent.mkdir()
        frame.to_csv(paths[name], index=False)
    return paths, verdicts, scores


def test_only_mild_mentions_of_threshold_findings_become_absent(tmp_path):
    paths, verdicts, scores = build(tmp_path)
    result = pd.read_csv(MODULE.prepare_severity_labels(paths["baseline"], paths["pilkwang"], tmp_path / "out"),
                         dtype={STUDY_ID: str})
    baseline = pd.read_csv(paths["baseline"], dtype={STUDY_ID: str})
    pd.testing.assert_frame_equal(result.iloc[8:], baseline.iloc[8:])
    for row in range(8):
        for column, name in enumerate(TARGETS):
            mild = verdicts[row, column] == "YES" and scores[row, column] == 0.68
            expected = 0.0 if mild and name in MODULE.THRESHOLD_FINDINGS else baseline.loc[row, name]
            assert (np.isnan(expected) and np.isnan(result.loc[row, name])) or result.loc[row, name] == expected


def test_refuses_labels_that_do_not_match_the_verdicts(tmp_path):
    paths, verdicts, _ = build(tmp_path)
    baseline = pd.read_csv(paths["baseline"])
    row = int(np.flatnonzero(verdicts[:8, 0] == "NO")[0])
    baseline.loc[row, "ACL"] = 1.0
    baseline.to_csv(paths["baseline"], index=False)
    with pytest.raises(ValueError, match="do not match"):
        MODULE.prepare_severity_labels(paths["baseline"], paths["pilkwang"], tmp_path / "out")
