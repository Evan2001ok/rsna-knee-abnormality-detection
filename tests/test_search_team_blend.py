"""Team blend search over the team's models (scripts/search_team_blend.py): "theirs" are the teammates' model sets,
"ours" our models and "r7" R7, heliosli's public ConvNeXt-small."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from search_team_blend import FIXED_HOURS, Half, configs, hours, search, their_oof  # noqa: E402

from rsna_knee_abnormality_detection.constants import STUDY_ID, TARGETS  # noqa: E402


def test_hours_add_up_and_the_budget_drops_slow_configurations() -> None:
    set_hours = {"a": 0.2, "b": 0.3, "c": 3.0}
    ours = {"x": ((42,), 1.0)}
    cfg = (("a", "b"), "x", True)
    assert hours(cfg, set_hours, ours) == pytest.approx(FIXED_HOURS["setup"] + FIXED_HOURS["theirs_cache"] + 0.5 + 1.0
                                                        + FIXED_HOURS["r7"])
    kept = configs(set_hours, ours, sets=("a", "b", "c"), limit=4.5)
    assert cfg in kept and all("c" not in c[0] or not c[2] for c in kept)
    assert all(hours(c, set_hours, ours) <= 4.5 for c in kept)


def test_search_recovers_the_informative_sets() -> None:
    rng = np.random.default_rng(0)
    n = 1500
    y = (rng.random((n, 12)) < 0.3).astype(float)
    signal = lambda s: y + rng.normal(0, s, y.shape)  # noqa: E731
    arrays = {"theirs": {"a": signal(0.8), "b": signal(0.8), "c": rng.normal(0, 1, y.shape)},
              "ours": {"x": signal(1.0)}, "r7": rng.normal(0, 1, y.shape)}
    judges = [(y, np.ones(y.shape, bool))]
    cfgs = configs({"a": 0.1, "b": 0.1, "c": 0.1}, {"x": ((42,), 0.1)}, sets=("a", "b", "c"), limit=10)
    r = search(Half(arrays, judges, np.arange(n)), cfgs)
    assert r["winner"][0] == ("a", "b")
    assert not r["winner"][2] or r["weights"][2] <= 0.05


def test_their_out_of_fold_rows_for_expert_studies_are_dropped(tmp_path: Path) -> None:
    rows = pd.DataFrame({"fold": [0, 1], STUDY_ID: ["s1", "s2"], "is_gold": [0, 1], **{t: [0.2, 0.9] for t in TARGETS}})
    rows.to_csv(tmp_path / "oof.csv", index=False)
    frame = their_oof(tmp_path / "oof.csv")
    assert list(frame.index) == ["s1"] and list(frame.columns) == TARGETS
