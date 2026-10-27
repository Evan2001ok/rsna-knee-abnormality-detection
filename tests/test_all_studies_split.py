"""The all-studies split moves only whole patient groups to the placeholder fold and covers every finding."""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from rsna_knee_abnormality_detection.constants import STUDY_ID, TARGETS

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
SPEC = importlib.util.spec_from_file_location("make_all_studies_split", ROOT / "scripts/make_all_studies_split.py")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_pick_groups_keeps_groups_whole_and_covers_findings():
    rng = np.random.default_rng(0)
    n = 400
    ids = [f"s{i}" for i in range(n)]
    splits = pd.DataFrame({STUDY_ID: ids, "group_id": [f"g{i // 2}" for i in range(n)],
                           "fold": [(i // 2) % 5 for i in range(n)], "split": "train"})
    splits.loc[splits.index[-10:], ["fold", "split"]] = [5, "train"]  # training-only rows (like OAI) stay out
    labels = pd.DataFrame(rng.integers(0, 2, (n, len(TARGETS))).astype(float), columns=TARGETS, index=ids)
    groups = MODULE.pick_groups(splits, labels, studies=40, minimum=3, seed=0)
    moved = splits[splits["group_id"].isin(groups)]
    assert moved["fold"].between(0, 4).all()
    assert len(moved) >= 40 and moved.groupby("group_id").size().eq(2).all()
    known = labels.loc[moved[STUDY_ID]]
    assert ((known == 1).sum() >= 3).all() and ((known == 0).sum() >= 3).all()
