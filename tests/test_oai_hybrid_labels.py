"""The hybrid OAI label rule (scripts/oai/prepare_oai_hybrid_labels.py) on synthetic knees: "ours" are our OAI labels
(MOAKS grades mapped by fixed rules), "theirs" the teammates' readings of the same knees."""

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

SPEC = importlib.util.spec_from_file_location(
    "hybrid", Path(__file__).parents[1] / "scripts" / "oai" / "prepare_oai_hybrid_labels.py")
hybrid_module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(hybrid_module)
SHARED = hybrid_module.SHARED
TARGETS = ["ACL", "MCL", *SHARED[:5], "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]


def frames():
    ids = [f"OAI_{9000000 + i}_LEFT_V00" for i in range(8)]
    rng = np.random.default_rng(0)
    ours = pd.DataFrame(rng.choice([0.0, 0.1, 0.35, 0.7, 1.0], size=(8, len(TARGETS))), index=ids, columns=TARGETS)
    ours[["MCL", "Contusion", "Fracture"]] = np.nan
    ours.loc[ids[:3], "Effusion"] = np.nan  # our readings leave effusion blank for three knees
    ours.loc[ids[0], "Synovitis"] = np.nan
    theirs = pd.DataFrame(rng.random((8, len(SHARED))), index=ids, columns=SHARED)
    theirs.loc[ids[5], "Lateral OA"] = np.nan  # masked on their side
    theirs.loc[ids[0], "Synovitis"] = np.nan
    return ours, theirs


def test_average_where_both_and_ours_where_only_ours():
    ours, theirs = frames()
    out, counts = hybrid_module.hybrid(ours, theirs)
    both = ours["Medial OA"].notna() & theirs["Medial OA"].notna()
    assert np.allclose(out.loc[both, "Medial OA"], 0.5 * ours.loc[both, "Medial OA"] + 0.5 * theirs.loc[both, "Medial OA"])
    assert out.loc[theirs.index[5], "Lateral OA"] == ours.loc[theirs.index[5], "Lateral OA"]
    assert counts["Lateral OA"]["only_ours"] == 1
    assert np.isnan(out.loc[theirs.index[0], "Synovitis"])  # blank on both sides stays blank
    for f in ("ACL", "Baker's"):
        assert out[f].equals(ours[f])
    assert out[["MCL", "Contusion", "Fracture"]].isna().all().all()


def test_cells_only_theirs_cover_stay_blank():
    ours, theirs = frames()
    out, counts = hybrid_module.hybrid(ours, theirs)
    assert out.loc[ours.index[:3], "Effusion"].isna().all()
    assert counts["Effusion"]["only_theirs_left_blank"] == 3


def test_knee_sets_must_match():
    ours, theirs = frames()
    with pytest.raises(ValueError, match="must match"):
        hybrid_module.hybrid(ours, theirs.iloc[1:])
