"""Windows spread within each series (Oct 17, 2026): an option that takes one window from each stretch of every series
instead of anywhere in it."""

from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest

from rsna_knee_abnormality_detection.config import DEFAULTS, validate_config
from rsna_knee_abnormality_detection.raptor_data import RaptorDataset


def sampler(spread, dropout=0.0, windows=12):
    return SimpleNamespace(windows=windows, sampling="stratified", series_dropout=dropout, window_spread=spread)


def record():
    # Five series: sagittal 0/1, coronal 2/3, axial 4, with 16/12/10/6/10 windows.
    return {"slots": np.repeat([0, 1, 2, 3, 4], [16, 12, 10, 6, 10]),
            "planes": np.repeat([0, 0, 1, 1, 2], [16, 12, 10, 6, 10])}


def old_draw(self, record, rng):
    """The stratified sampler before the change, copied verbatim, to show draws without spread are unchanged."""
    slots, planes = record["slots"], record["planes"]
    present = np.unique(slots)
    kept = present[rng.random(len(present)) >= self.series_dropout]
    for plane in np.unique(planes):
        plane_slots = np.unique(slots[planes == plane])
        if not np.isin(plane_slots, kept).any():
            kept = np.append(kept, rng.choice(plane_slots))
    candidates = np.flatnonzero(np.isin(slots, kept))
    groups = [candidates[slots[candidates] == slot] for slot in np.sort(kept)]
    sizes = np.array([len(group) for group in groups], dtype=float)
    quotas = sizes / sizes.sum() * self.windows
    counts = np.floor(quotas).astype(int)
    for position in np.argsort(-(quotas - counts), kind="stable")[: self.windows - counts.sum()]:
        counts[position] += 1
    return np.concatenate([rng.choice(group, size=number, replace=len(group) < number)
                           for group, number in zip(groups, counts, strict=True) if number > 0])


@pytest.mark.parametrize("dropout", [0.0, 0.15])
def test_draws_without_spread_are_unchanged(dropout):
    for seed in range(30):
        new = RaptorDataset._sample_windows(sampler(False, dropout), record(), np.random.default_rng(seed))
        old = old_draw(sampler(False, dropout), record(), np.random.default_rng(seed))
        assert new.tolist() == old.tolist()


def test_spread_takes_one_window_from_each_stretch_of_every_series():
    data = record()
    quotas = {0: 4, 1: 3, 2: 2, 3: 1, 4: 2}  # 12 windows split 16:12:10:6:10, as the stratified test fixes
    seen = set()
    for seed in range(40):
        choices = RaptorDataset._sample_windows(sampler(True), data, np.random.default_rng(seed))
        assert len(choices) == 12 and len(set(choices.tolist())) == 12
        for slot, quota in quotas.items():
            group = np.flatnonzero(data["slots"] == slot)
            picked = np.sort(choices[data["slots"][choices] == slot])
            assert len(picked) == quota
            stretches = np.array_split(group, quota)
            assert all(picked[i] in stretches[i] for i in range(quota))
        seen.add(tuple(choices.tolist()))
    assert len(seen) > 30  # still random


def test_config_needs_stratified_sampling():
    config = deepcopy(DEFAULTS)
    config["data"].update(recipe="raptor", raptor_variant="raptor-dense-distinct-v1", cache_size=336,
                          raptor_window_sampling="stratified", raptor_window_spread=True)
    config["model"].update(pooling="class_attention", use_metadata=False, use_positions=False)
    validate_config(config)
    config["data"]["raptor_window_sampling"] = "random"
    with pytest.raises(ValueError, match="raptor_window_spread"):
        validate_config(config)
