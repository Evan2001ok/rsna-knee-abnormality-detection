"""Consensus label helpers (scripts/prepare_consensus_labels.py). The consensus labels are our final training labels:
for each cell, the plain mean of the label sources present."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from prepare_consensus_labels import mean_present, ours_cv_sources, quantile_map  # noqa: E402


def test_mean_present_averages_only_available_sources() -> None:
    idx, cols = ["a", "b"], ["x", "y"]
    s1 = pd.DataFrame([[0.2, np.nan], [0.4, 0.6]], index=idx, columns=cols)
    s2 = pd.DataFrame([[0.4, 0.8], [np.nan, 0.0]], index=idx, columns=cols)
    s3 = pd.DataFrame([[0.6, np.nan], [0.1, 0.3]], index=["b", "a"], columns=cols)  # other row order: aligned by index
    out = mean_present([s1, s2, s3])
    assert np.allclose(out.loc["a"], [(0.2 + 0.4 + 0.1) / 3, (0.8 + 0.3) / 2])
    assert np.allclose(out.loc["b"], [(0.4 + 0.6) / 2, (0.6 + 0.0) / 2])


def test_mean_present_leaves_cells_without_sources_missing() -> None:
    s = pd.DataFrame([[np.nan]], index=["a"], columns=["x"])
    assert np.isnan(mean_present([s, s]).iloc[0, 0])


def test_quantile_map_keeps_order_and_takes_reference_values() -> None:
    scores = pd.Series([0.9, 0.1, 0.5, 0.3])
    reference = np.array([0.0, np.nan, 1.0, 0.25, 0.5])
    mapped = quantile_map(scores, reference)
    assert list(np.argsort(mapped)) == list(np.argsort(scores.to_numpy()))
    assert mapped.min() == 0.0 and mapped.max() == 1.0
    assert np.allclose(np.sort(mapped), [0.0, 0.25, 0.5, 1.0])


def test_round_two_takes_the_three_consensus_seeds_for_our_group() -> None:
    round2 = [str(p) for p in ours_cv_sources(True)]
    assert len(round2) == 12 and all("consensus58" in p and p.endswith("validation-averaged.csv") for p in round2)
    assert {p.split("/")[1].rsplit("_fold", 1)[0] for p in round2} == {
        "coatnet_384_consensus58", "coatnet_384_consensus58s43", "coatnet_384_consensus58s44"}
    assert all("consensus" not in str(p) for p in ours_cv_sources(False))
