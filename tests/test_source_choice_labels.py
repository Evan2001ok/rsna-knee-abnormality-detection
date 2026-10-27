"""Source-choice labels: bounded source weights and the out-of-fold CoAtNet readings used as a source."""

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from rsna_knee_abnormality_detection.constants import STUDY_ID, TARGETS

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
SPEC = importlib.util.spec_from_file_location("prepare_source_choice_labels",
                                              ROOT / "scripts/prepare_source_choice_labels.py")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


@pytest.mark.parametrize("lowest, highest", [(0.1, 0.7), (0.2, 0.6), (0.0, 1.0)])
def test_bounded_weights_sum_to_one_within_bounds(lowest, highest):
    for shares in ({"report": 0.0, "gpt": 0.02, "coatnet": 0.98}, {"report": 0.7, "gpt": 0.2, "coatnet": 0.1},
                   {"report": 0.4, "gpt": 0.3, "coatnet": 0.3}):
        weights = MODULE.bounded_weights(shares, lowest, highest)
        assert sum(weights.values()) == pytest.approx(1.0)
        assert all(lowest - 1e-12 <= w <= highest + 1e-12 for w in weights.values())


def test_bounded_weights_with_a_winner_and_twenty_sixty_bounds():
    weights = MODULE.bounded_weights({"report": 0.0, "gpt": 0.02, "coatnet": 0.98}, 0.2, 0.6)
    assert weights == pytest.approx({"report": 0.2, "gpt": 0.2, "coatnet": 0.6})


def write_run(folder: Path, validation: list[str], training: list[str], weight: float = 2.0) -> Path:
    folder.mkdir(parents=True)
    weights = dict.fromkeys(TARGETS, weight)
    (folder / "provenance.json").write_text(json.dumps({
        "training_objective": {"positive_term_weights": weights}, "validation_ids": validation,
        "training_ids": training}))
    frame = pd.DataFrame({STUDY_ID: validation, **{name: 0.5 for name in TARGETS}})
    frame.to_csv(folder / MODULE.PREDICTIONS_FILE, index=False)
    return folder


def test_load_run_converts_to_the_label_scale(tmp_path):
    run = write_run(tmp_path / "run", ["a", "b"], ["c"], weight=3.0)
    loaded = MODULE.load_run(run)
    assert np.allclose(loaded["predictions"].to_numpy(), 0.5 / (0.5 + 3.0 * 0.5))


def test_load_run_refuses_predictions_on_its_own_training_studies(tmp_path):
    run = write_run(tmp_path / "run", ["a", "b"], ["b", "c"])
    with pytest.raises(ValueError, match="used to train"):
        MODULE.load_run(run)
