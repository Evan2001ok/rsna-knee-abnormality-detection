"""GPT image labels change only the cross-validation studies GPT has read, by a fixed share."""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from rsna_knee_abnormality_detection.constants import STUDY_ID, TARGETS

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("prepare_gpt_mixed_labels", ROOT / "scripts/prepare_gpt_mixed_labels.py")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def build(tmp_path, gpt_ids=("s0", "s3")):
    rng = np.random.default_rng(0)
    ids = ["s0", "s1", "s2", "s3", "expert", "unlabeled"]
    splits = pd.DataFrame({STUDY_ID: ids, "fold": [0, 1, 2, 3, -1, -2], "split": ["train"] * 4 + ["expert_holdout", "unlabeled"],
                           "group_id": ids})
    student = pd.DataFrame(rng.uniform(0.05, 0.95, size=(6, 12)), columns=TARGETS)
    student.insert(0, STUDY_ID, ids)
    student["label_source"] = ["report"] * 4 + ["expert", "report"]
    gpt = pd.DataFrame(rng.uniform(0, 1, size=(len(gpt_ids), 12)), columns=TARGETS)
    gpt.insert(0, STUDY_ID, list(gpt_ids))
    gpt["ACL__assessable"] = True
    paths = {"student": tmp_path / "student" / "labels.csv", "gpt": tmp_path / "gpt.csv", "splits": tmp_path / "splits.csv"}
    paths["student"].parent.mkdir(parents=True)
    student.to_csv(paths["student"], index=False)
    gpt.to_csv(paths["gpt"], index=False)
    splits.to_csv(paths["splits"], index=False)
    return paths, student.set_index(STUDY_ID), gpt.set_index(STUDY_ID)


def test_read_studies_get_the_fixed_share_and_everything_else_is_unchanged(tmp_path):
    paths, student, gpt = build(tmp_path)
    path = MODULE.prepare_gpt_mixed_labels(paths["student"], paths["gpt"], paths["splits"], tmp_path / "out")
    labels = pd.read_csv(path, dtype={STUDY_ID: str}).set_index(STUDY_ID)
    for study in ("s0", "s3"):
        np.testing.assert_allclose(labels.loc[study, TARGETS].to_numpy(dtype=float),
                                   (2 / 3) * student.loc[study, TARGETS].to_numpy(dtype=float) + (1 / 3) * gpt.loc[study, TARGETS].to_numpy(dtype=float))
    for study in ("s1", "s2", "expert", "unlabeled"):
        np.testing.assert_allclose(labels.loc[study, TARGETS].to_numpy(dtype=float), student.loc[study, TARGETS].to_numpy(dtype=float))
    assert labels["label_source"].tolist() == student["label_source"].tolist()
    record = json.loads((tmp_path / "out" / "label_provenance.json").read_text())
    assert record["studies_mixed"] == 2 and record["studies_mixed_per_fold"] == {"0": 1, "3": 1}


@pytest.mark.parametrize("gpt_ids", [("s0", "expert"), ("s0", "unlabeled")])
def test_expert_and_unlabeled_studies_are_refused(tmp_path, gpt_ids):
    paths, _, _ = build(tmp_path, gpt_ids)
    with pytest.raises(ValueError, match="cross-validation"):
        MODULE.prepare_gpt_mixed_labels(paths["student"], paths["gpt"], paths["splits"], tmp_path / "out")


def test_unknown_studies_and_bad_probabilities_are_refused(tmp_path):
    paths, _, _ = build(tmp_path, ("s0", "nowhere"))
    with pytest.raises(ValueError, match="not in the student labels"):
        MODULE.prepare_gpt_mixed_labels(paths["student"], paths["gpt"], paths["splits"], tmp_path / "out")
    paths, _, gpt = build(tmp_path / "second")
    broken = gpt.reset_index()
    broken.loc[0, "Fracture"] = np.nan  # the file reader allows blanks; mixed labels must not
    broken.to_csv(paths["gpt"], index=False)
    with pytest.raises(ValueError, match="complete"):
        MODULE.prepare_gpt_mixed_labels(paths["student"], paths["gpt"], paths["splits"], tmp_path / "second" / "out")
