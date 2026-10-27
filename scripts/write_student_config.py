#!/usr/bin/env python3
"""Write a student config from the teacher recipe chosen by scripts/choose_teacher_recipe.py.

A student is a new model trained from scratch on labels refined with the teachers' out-of-fold predictions
(prepare_student_labels.py).

The teacher config for the requested fold is copied with these changes only: the training labels,
the loss weights (fixed to that fold's teacher positive-term weights, so the labels are the only
training difference), the output folder, and optionally the labels used to select checkpoints.
"""

import argparse
import json
from pathlib import Path

import yaml

from rsna_knee_abnormality_detection.config import load_config
from rsna_knee_abnormality_detection.constants import TARGETS


def replace_once(text: str, old: str, new: str) -> str:
    if text.count(old) != 1:
        raise ValueError(f"Expected exactly one {old.strip()!r} line in the config")
    return text.replace(old, new)


def student_config(source: Path, weights: dict, labels: str, output_dir: str,
                   purpose: str = "Student fold 0 for the teacher-student round",
                   validation_labels: str | None = None) -> str:
    settings = load_config(source)
    if settings["training"]["positive_weighting"] != "raptor":
        raise ValueError("Expected a teacher config that computes its weights from the labels")
    text = source.read_text()
    text = replace_once(text, f"  labels: {settings['data']['labels']}\n", f"  labels: {labels}\n")
    text = replace_once(text, f"  output_dir: {settings['training']['output_dir']}\n", f"  output_dir: {output_dir}\n")
    changed = "the labels, the (fixed) loss weights and the output folder"
    if validation_labels is not None:
        text = replace_once(text, f"  validation_labels: {settings['data']['validation_labels']}\n",
                            f"  validation_labels: {validation_labels}\n")
        changed = "the labels, the (fixed) loss weights, the output folder and the checkpoint-selection labels"
    block = "".join(f"    {name}: {float(weights[name])!r}\n" for name in TARGETS)
    text = replace_once(text, "  positive_weighting: raptor\n",
                        f"  positive_weighting: fixed\n  fixed_positive_weights:\n{block}")
    note = (f"# {purpose}, written by scripts/write_student_config.py from {source}.\n"
            f"# Only {changed} differ from that teacher config.\n")
    return note + text


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--decision", type=Path, default=Path("artifacts/teacher-recipe-decision.json"))
    parser.add_argument("--fold", type=int, default=0, help="Teacher fold whose config and loss weights to use")
    parser.add_argument("--labels", default="data/labels/student_v1/labels.csv")
    parser.add_argument("--validation-labels", help="Replace the labels used to select checkpoints")
    parser.add_argument("--output", type=Path, default=Path("configs/student/student_v1_fold0.yaml"))
    parser.add_argument("--output-dir", default="runs/student_v1_fold0")
    parser.add_argument("--purpose", default="Student fold 0 for the teacher-student round",
                        help="First line of the header comment")
    args = parser.parse_args()
    decision = json.loads(args.decision.read_text())
    chosen = next(item for item in [decision["candidate"], *decision["alternatives"]] if item["name"] == decision["chosen"])
    if decision["teacher_runs"][0] != chosen["run"]:
        raise ValueError("The decision's fold-0 teacher run does not match the chosen recipe")
    if not 0 <= args.fold < len(decision["teacher_runs"]):
        raise ValueError(f"Fold must be between 0 and {len(decision['teacher_runs']) - 1}")
    source = Path(chosen["config"]) if args.fold == 0 else Path(decision["fold_configs"][args.fold - 1])
    teacher = Path(decision["teacher_runs"][args.fold])
    if load_config(source)["data"]["fold"] != args.fold:
        raise ValueError(f"{source} is not the fold-{args.fold} teacher config")
    if yaml.safe_load((teacher / "config.yaml").read_text())["data"]["fold"] != args.fold:
        raise ValueError(f"{teacher} is not the fold-{args.fold} teacher run")
    weights = json.loads((teacher / "provenance.json").read_text())["training_objective"]["positive_term_weights"]
    content = student_config(source, weights, args.labels, args.output_dir, args.purpose, args.validation_labels)
    if args.output.exists() and args.output.read_text() != content:
        raise FileExistsError(f"{args.output} exists with different content")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(content)
    expected = load_config(source)
    expected["data"]["labels"], expected["training"]["output_dir"] = args.labels, args.output_dir
    if args.validation_labels is not None:
        expected["data"]["validation_labels"] = args.validation_labels
    expected["training"]["positive_weighting"] = "fixed"
    expected["training"]["fixed_positive_weights"] = {name: float(weights[name]) for name in TARGETS}
    if load_config(args.output) != expected:
        raise AssertionError("The student config differs from the teacher config in more than the intended settings")
    print(args.output)


if __name__ == "__main__":
    main()
