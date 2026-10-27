"""Averaged labels change only stated cells of cross-validation studies, skipping each source's 'not mentioned' code."""

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from rsna_knee_abnormality_detection.constants import STUDY_ID, TARGETS

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("prepare_averaged_labels", ROOT / "scripts/prepare_averaged_labels.py")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def build(tmp_path):
    rng = np.random.default_rng(1)
    ids = [f"s{index}" for index in range(10)] + ["expert", "unlabeled"]
    splits = pd.DataFrame({STUDY_ID: ids, "fold": [index % 5 for index in range(10)] + [-1, -2],
                           "split": ["train"] * 10 + ["expert_holdout", "unlabeled"], "group_id": ids})
    raptor = pd.DataFrame(rng.choice([0.05, 0.2, 0.35, 0.8, 0.95], size=(12, 12)), columns=TARGETS)
    raptor.loc[10, TARGETS] = rng.integers(0, 2, 12)
    raptor.insert(0, STUDY_ID, ids)
    raptor["label_source"] = ["report"] * 10 + ["expert", "report"]
    baseline = pd.DataFrame(rng.integers(0, 2, size=(12, 12)).astype(float), columns=TARGETS)
    baseline[pd.DataFrame(rng.random((12, 12)) < 0.4, columns=TARGETS)] = np.nan
    baseline.loc[10, TARGETS] = raptor.loc[10, TARGETS].to_numpy()
    baseline.loc[11, TARGETS] = np.nan
    baseline.insert(0, STUDY_ID, ids)
    baseline["label_source"] = ["report"] * 10 + ["expert", "missing"]
    steven = pd.DataFrame(rng.choice([0.5, 0.1, 0.9], size=(12, 12)), columns=TARGETS)
    steven.insert(0, STUDY_ID, ids)
    gpt = pd.DataFrame(rng.choice([0.0, 1.0], size=(12, 12)), columns=TARGETS)
    gpt.insert(0, STUDY_ID, ids)
    paths = {}
    for name, frame in (("raptor", raptor), ("baseline", baseline), ("steven", steven), ("gpt", gpt)):
        paths[name] = tmp_path / name / "labels.csv"
        paths[name].parent.mkdir()
        frame.to_csv(paths[name], index=False)
    paths["splits"] = tmp_path / "splits.csv"
    splits.to_csv(paths["splits"], index=False)
    return paths


def prepare(paths, output):
    sources = [(paths["steven"], 0.5), (paths["gpt"], None)]
    return MODULE.prepare_averaged_labels(paths["raptor"], paths["baseline"], paths["splits"], sources, output)


def test_stated_cells_average_available_readings_and_silent_cells_keep_raptor(tmp_path):
    paths = build(tmp_path)
    result = pd.read_csv(prepare(paths, tmp_path / "out"), dtype={STUDY_ID: str})
    raptor, baseline = (pd.read_csv(paths[name], dtype={STUDY_ID: str}) for name in ("raptor", "baseline"))
    steven, gpt = (pd.read_csv(paths[name], dtype={STUDY_ID: str}) for name in ("steven", "gpt"))
    pd.testing.assert_frame_equal(result.iloc[10:], raptor.iloc[10:])
    for row in range(10):
        for name in TARGETS:
            if np.isnan(baseline.loc[row, name]):
                expected = raptor.loc[row, name]
            else:
                readings = [raptor.loc[row, name], gpt.loc[row, name]]
                if steven.loc[row, name] != 0.5:
                    readings.append(steven.loc[row, name])
                expected = np.mean(readings)
            assert result.loc[row, name] == pytest.approx(expected)


def test_refuses_out_of_range_readings_and_missing_studies(tmp_path):
    paths = build(tmp_path)
    gpt = pd.read_csv(paths["gpt"])
    gpt.loc[0, "ACL"] = 1.5
    gpt.to_csv(paths["gpt"], index=False)
    with pytest.raises(ValueError, match="between"):
        prepare(paths, tmp_path / "out")
    gpt.loc[0, "ACL"] = 1.0
    gpt.iloc[1:].to_csv(paths["gpt"], index=False)
    with pytest.raises(ValueError, match="missing"):
        prepare(paths, tmp_path / "out")


def test_refuses_to_write_next_to_an_input(tmp_path):
    paths = build(tmp_path)
    with pytest.raises(ValueError, match="output directory"):
        prepare(paths, paths["steven"].parent)


def test_findings_outside_the_list_keep_their_raptor_values(tmp_path):
    paths = build(tmp_path)
    sources = [(paths["steven"], 0.5), (paths["gpt"], None)]
    averaged = ["Medial Meniscus", "Synovitis"]
    result = pd.read_csv(MODULE.prepare_averaged_labels(paths["raptor"], paths["baseline"], paths["splits"], sources,
                                                        tmp_path / "out", averaged), dtype={STUDY_ID: str})
    raptor = pd.read_csv(paths["raptor"], dtype={STUDY_ID: str})
    others = [name for name in TARGETS if name not in averaged]
    pd.testing.assert_frame_equal(result[others], raptor[others])
    assert not result[averaged].equals(raptor[averaged])
