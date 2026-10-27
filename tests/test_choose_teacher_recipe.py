"""The teacher recipe rule and the fold configs it writes (scripts/choose_teacher_recipe.py). The teachers are the fold
models whose out-of-fold predictions go into the student labels; students are then trained on those labels."""

import importlib.util
from pathlib import Path

import pytest

from rsna_knee_abnormality_detection.config import load_config

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("choose_teacher_recipe", ROOT / "scripts/choose_teacher_recipe.py")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_candidate_kept_within_margin_and_replaced_when_clearly_worse():
    alternatives = [{"name": "a", "auc": 0.8653}, {"name": "b", "auc": 0.8630}]
    assert MODULE.choose({"name": "new", "auc": 0.8640}, alternatives)["name"] == "new"
    assert MODULE.choose({"name": "new", "auc": 0.8700}, alternatives)["name"] == "new"
    assert MODULE.choose({"name": "new", "auc": 0.8620}, alternatives)["name"] == "a"


def test_fold_config_changes_only_fold_and_output_directory(tmp_path):
    source = ROOT / "configs/followup/dense_combined_long.yaml"
    path = tmp_path / "fold3.yaml"
    path.write_text(MODULE.fold_config(source.read_text(), source, 3, "runs/teacher_x_fold3"))
    changed, expected = load_config(path), load_config(source)
    expected["data"]["fold"], expected["training"]["output_dir"] = 3, "runs/teacher_x_fold3"
    assert changed == expected


def test_fold_config_refuses_ambiguous_lines():
    with pytest.raises(ValueError):
        MODULE.replace_once("  fold: 0\n  fold: 0\n", "  fold: 0\n", "  fold: 1\n")
