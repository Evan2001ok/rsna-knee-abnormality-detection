#!/usr/bin/env python3
"""Build second-round ("student") training labels from the Raptor soft labels and fold-model predictions.

The "teachers" are five first-round fold models (recipe chosen by choose_teacher_recipe.py); a "student" is a new
model trained from scratch on labels refined with the teachers' out-of-fold predictions. In the README both steps
together are round 1, where --raptor-labels was data/labels/avg_v2/labels.csv (the averaged report labels from
prepare_averaged_labels.py) and the output was data/labels/student_v2.

Only cells the radiology report does not mention change; they are blank in the baseline report
labels. Each such cell becomes MIX * converted prediction + (1 - MIX) * Raptor value. The prediction
comes from the fold model whose validation fold holds the study, so that model never trained on it.

Predictions are first converted back to the label scale. Training multiplies each finding's
positive loss term by a weight w, which raises the predicted odds by about w, so a prediction p
becomes p / (p + w * (1 - p)) using that fold model's own weights.

Cells the report states keep their Raptor value. Expert-holdout and unlabeled rows are copied
unchanged. Nothing here reads expert labels, scores predictions, or tunes the mix.

Run: uv run python scripts/prepare_student_labels.py --teacher-runs <fold 0 run> ... <fold 4 run>
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from rsna_knee_abnormality_detection.constants import STUDY_ID, TARGETS
from rsna_knee_abnormality_detection.labels import file_record, read_study_csv

MIX = 0.5
PREDICTIONS_FILE = "validation-averaged-model-mirror_tta.csv"


def to_label_scale(probability, weight):
    """Undo the odds increase caused by a positive-term loss weight."""
    return probability / (probability + weight * (1 - probability))


def load_teacher(run: Path) -> dict:
    run = Path(run)
    fold = yaml.safe_load((run / "config.yaml").read_text())["data"]["fold"]
    provenance = json.loads((run / "provenance.json").read_text())
    weights = provenance["training_objective"]["positive_term_weights"]
    if sorted(weights) != sorted(TARGETS) or not all(np.isfinite(weights[name]) and weights[name] >= 1 for name in TARGETS):
        raise ValueError(f"{run}: positive-term weights must cover all findings and be at least 1")
    predictions = read_study_csv(run / PREDICTIONS_FILE).set_index(STUDY_ID)[TARGETS]
    values = predictions.to_numpy()
    if not np.isfinite(values).all() or (values < 0).any() or (values > 1).any():
        raise ValueError(f"{run}: predictions must be finite and between zero and one")
    if set(predictions.index) != set(provenance["validation_ids"]):
        raise ValueError(f"{run}: predictions must cover exactly the run's validation studies")
    if set(predictions.index) & set(provenance["training_ids"]):
        raise ValueError(f"{run}: some predicted studies were used to train this model")
    converted = predictions.copy()
    for name in TARGETS:
        converted[name] = to_label_scale(predictions[name], weights[name])
    return {"run": str(run), "fold": int(fold), "weights": weights, "predictions": converted,
            "predictions_file": file_record(run / PREDICTIONS_FILE), "provenance_file": file_record(run / "provenance.json")}


def prepare_student_labels(raptor_csv: Path, baseline_csv: Path, splits_csv: Path, teacher_runs: list[Path],
                           output_dir: Path, mix: float = MIX) -> Path:
    if not 0 < mix <= 1:
        raise ValueError("The prediction share must be above zero and at most one")
    raptor_csv, baseline_csv, splits_csv, output_dir = map(Path, (raptor_csv, baseline_csv, splits_csv, output_dir))
    input_paths = [raptor_csv, baseline_csv, splits_csv]
    if any(output_dir.resolve() == path.parent.resolve() for path in input_paths):
        raise ValueError("The output directory must not hold the input labels or splits")
    raptor = read_study_csv(raptor_csv)
    baseline = read_study_csv(baseline_csv).set_index(STUDY_ID)
    splits = read_study_csv(splits_csv, require_targets=False).set_index(STUDY_ID)
    ids = raptor[STUDY_ID]
    if set(ids) != set(baseline.index) or set(ids) != set(splits.index):
        raise ValueError("Raptor labels, baseline labels and splits must contain the same studies")
    if "label_source" not in raptor:
        raise ValueError("Raptor labels need a label_source column")
    fold = splits.loc[ids, "fold"].astype(int).to_numpy()
    cross_validation = fold >= 0
    if not (raptor.loc[cross_validation, "label_source"].eq("report").all()
            and baseline.loc[ids[cross_validation], "label_source"].eq("report").all()):
        raise ValueError("Every cross-validation study must have report-derived labels")

    teachers = [load_teacher(run) for run in teacher_runs]
    expected_folds = sorted(set(fold[cross_validation].tolist()))
    if sorted(teacher["fold"] for teacher in teachers) != expected_folds:
        raise ValueError(f"Give exactly one model per fold {expected_folds}")
    for teacher in teachers:
        if set(teacher["predictions"].index) != set(ids[fold == teacher["fold"]]):
            raise ValueError(f"{teacher['run']}: predictions must cover exactly the studies of fold {teacher['fold']}")
    converted = pd.concat([teacher["predictions"] for teacher in teachers]).loc[ids[cross_validation]]

    labels = raptor.copy()
    before = raptor.loc[cross_validation, TARGETS].to_numpy()
    silent = baseline.loc[ids[cross_validation], TARGETS].isna().to_numpy()
    after = np.where(silent, mix * converted.to_numpy() + (1 - mix) * before, before)
    labels.loc[cross_validation, TARGETS] = after
    values = labels[TARGETS].to_numpy()
    if not np.isfinite(values).all() or (values < 0).any() or (values > 1).any():
        raise AssertionError("Student labels must be finite and between zero and one")
    if not labels.loc[~cross_validation].equals(raptor.loc[~cross_validation]):
        raise AssertionError("Expert-holdout and unlabeled rows must stay unchanged")
    if not np.array_equal(after[~silent], before[~silent]):
        raise AssertionError("Cells the report mentions must keep their Raptor value")

    summary = {}
    prediction_values = converted.to_numpy()
    for column, name in enumerate(TARGETS):
        quiet, said = silent[:, column], ~silent[:, column]
        summary[name] = {
            "silent_cells": int(quiet.sum()),
            "silent_mean_before": round(float(before[quiet, column].mean()), 4) if quiet.any() else None,
            "silent_mean_after": round(float(after[quiet, column].mean()), 4) if quiet.any() else None,
            "mentioned_mean_raptor": round(float(before[said, column].mean()), 4) if said.any() else None,
            "mentioned_mean_converted_prediction": round(float(prediction_values[said, column].mean()), 4) if said.any() else None,
        }
    inputs_before = {str(path): file_record(path) for path in input_paths}
    output_dir.mkdir(parents=True, exist_ok=True)
    labels_path = output_dir / "labels.csv"
    labels.to_csv(labels_path, index=False)
    provenance = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "rule": (f"Cells blank in the baseline report labels (not mentioned) become {mix} x converted fold-model prediction "
                 f"+ {1 - mix} x Raptor value; mentioned cells keep the Raptor value; expert-holdout and unlabeled rows "
                 "are unchanged."),
        "prediction_conversion": "p / (p + w * (1 - p)) with each fold model's own positive-term weight w",
        "mix": mix,
        "inputs": {"raptor_labels": inputs_before[str(raptor_csv)], "baseline_labels": inputs_before[str(baseline_csv)],
                   "splits": inputs_before[str(splits_csv)]},
        "teachers": [{key: teacher[key] for key in ("run", "fold", "weights", "predictions_file", "provenance_file")}
                     for teacher in sorted(teachers, key=lambda item: item["fold"])],
        "studies": len(labels),
        "cross_validation_studies_refined": int(cross_validation.sum()),
        "changed_cells": int(silent.sum()),
        "per_finding": summary,
        "configuration": {"data.labels": str(labels_path), "data.validation_labels": str(baseline_csv),
                          "data.splits": str(splits_csv)},
        "limitations": [
            "Fold models 1-4 trained on fold-0 studies (and so on for each fold), so a student validated on any fold "
            "has indirectly seen information from that fold's labels; its validation score is somewhat optimistic.",
            "The 50/50 share is fixed in advance, not tuned.",
            "Predictions come from models trained on report-derived labels, not expert image labels.",
        ],
        "outcome_labels_read": False,
        "labels_csv": file_record(labels_path),
        "reproduction": "uv run python scripts/prepare_student_labels.py --teacher-runs " + " ".join(
            teacher["run"] for teacher in sorted(teachers, key=lambda item: item["fold"])),
    }
    if any(file_record(path)["sha256"] != inputs_before[str(path)]["sha256"] for path in input_paths):
        raise RuntimeError("An input changed during preparation; the generated labels must not be used")
    (output_dir / "label_provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    return labels_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raptor-labels", type=Path, default=Path("data/labels/raptor/labels.csv"))
    parser.add_argument("--baseline-labels", type=Path, default=Path("data/labels/baseline/labels.csv"))
    parser.add_argument("--splits", type=Path, default=Path("data/splits/baseline/splits.csv"))
    parser.add_argument("--teacher-runs", type=Path, nargs="+", required=True, help="One finished run per fold")
    parser.add_argument("--output-dir", type=Path, default=Path("data/labels/student_v1"))
    parser.add_argument("--mix", type=float, default=MIX, help="Share of the converted prediction in silent cells")
    args = parser.parse_args()
    path = prepare_student_labels(args.raptor_labels, args.baseline_labels, args.splits, args.teacher_runs,
                                  args.output_dir, args.mix)
    record = json.loads((path.parent / "label_provenance.json").read_text())
    print(json.dumps({"labels": str(path), "changed_cells": record["changed_cells"], "per_finding": record["per_finding"]}))


if __name__ == "__main__":
    main()
