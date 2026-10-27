import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "oai"))
from merge_oai_manifests import merge  # noqa: E402


def frame(ids):
    return pd.DataFrame({"StudyInstanceUID": ids, "cache_path": [f"{i}.npy" for i in ids]})


BASE = frame(["OAI_1_LEFT_V00", "OAI_1_RIGHT_V00"])
V01 = frame(["OAI_1_LEFT_V01", "OAI_2_LEFT_V01"])
V03 = frame(["OAI_2_LEFT_V03", "OAI_3_RIGHT_V03", "OAI_4_LEFT_V03"])
LABELLED = {"OAI_1_LEFT_V01", "OAI_2_LEFT_V01", "OAI_2_LEFT_V03", "OAI_3_RIGHT_V03"}


def test_all_visits_keeps_everything():
    out = merge([BASE, V01, V03], LABELLED, new_knees_only=False)
    assert len(out) == 7


def test_new_knees_only_takes_first_labelled_visit_of_knees_missing_from_base():
    out = merge([BASE, V03, V01], LABELLED, new_knees_only=True)
    assert sorted(out["StudyInstanceUID"]) == ["OAI_1_LEFT_V00", "OAI_1_RIGHT_V00", "OAI_2_LEFT_V01", "OAI_3_RIGHT_V03"]


def test_duplicates_refused():
    with pytest.raises(ValueError):
        merge([BASE, BASE], set(), new_knees_only=False)
