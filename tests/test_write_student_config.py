"""The student config differs from its teacher config only in labels, fixed loss weights, output folder and,
optionally, the checkpoint-selection labels. (A student trains on labels that include its teachers' out-of-fold
predictions.)"""

import importlib.util
from pathlib import Path

import pytest

from rsna_knee_abnormality_detection.config import load_config
from rsna_knee_abnormality_detection.constants import TARGETS

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("write_student_config", ROOT / "scripts/write_student_config.py")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_student_config_changes_only_labels_weights_and_output(tmp_path):
    source = ROOT / "configs/followup/dense_combined_long.yaml"
    weights = {name: 1.0 + index / 3 for index, name in enumerate(TARGETS)}
    path = tmp_path / "student.yaml"
    path.write_text(MODULE.student_config(source, weights, "data/labels/student_v1/labels.csv", "runs/student_x"))
    expected = load_config(source)
    expected["data"]["labels"], expected["training"]["output_dir"] = "data/labels/student_v1/labels.csv", "runs/student_x"
    expected["training"]["positive_weighting"], expected["training"]["fixed_positive_weights"] = "fixed", weights
    assert load_config(path) == expected


def test_fold_k_config_can_also_change_the_checkpoint_selection_labels(tmp_path):
    source = ROOT / "configs/teachers/dense_combined_long_fold3.yaml"
    weights = {name: 2.0 for name in TARGETS}
    path = tmp_path / "student.yaml"
    path.write_text(MODULE.student_config(source, weights, "data/labels/student_v2/labels.csv", "runs/final_fold3",
                                          "Final student fold 3", "data/labels/baseline_severity/labels.csv"))
    expected = load_config(source)
    expected["data"]["labels"], expected["training"]["output_dir"] = "data/labels/student_v2/labels.csv", "runs/final_fold3"
    expected["data"]["validation_labels"] = "data/labels/baseline_severity/labels.csv"
    expected["training"]["positive_weighting"], expected["training"]["fixed_positive_weights"] = "fixed", weights
    result = load_config(path)
    assert result == expected and result["data"]["fold"] == 3


def test_student_config_requires_a_teacher_with_label_based_weights():
    with pytest.raises(ValueError):
        MODULE.student_config(ROOT / "configs/raptor_resnet34_silence_policy.yaml", {}, "x.csv", "runs/x")
