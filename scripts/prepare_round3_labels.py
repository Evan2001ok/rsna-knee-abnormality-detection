#!/usr/bin/env python3
"""Round-3 training labels: mix the GPT-label folds' out-of-fold predictions into the current labels.

The GPT-label folds (--new-runs) are the five fold models trained on the round-2 labels (data/labels/student_v2_gpt_full,
from prepare_gpt_mixed_labels.py). The earlier teachers (--earlier-runs) are the first-round fold models listed in
artifacts/teacher-recipe-decision.json, written by choose_teacher_recipe.py; the script reads that file at start-up
for the default, so it must exist even when --earlier-runs is given.
Each cross-validation cell becomes (1 - s) x current label + s x converted prediction, where the prediction
comes from the fold model that never trained on the study. The share is fixed in advance by how reliable
the label is:
- RELIABLE_SHARE where the report states the finding and the report value (avg_v2), the GPT probability and
  the earlier teachers' converted out-of-fold prediction all fall on the same side of 0.5;
- UNRELIABLE_SHARE everywhere else (report silent, or the sources disagree).
Expert-holdout and unlabeled rows are copied unchanged. No expert label is read and nothing is tuned.

Run: uv run python scripts/prepare_round3_labels.py
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from prepare_student_labels import load_teacher

from rsna_knee_abnormality_detection.constants import STUDY_ID, TARGETS
from rsna_knee_abnormality_detection.labels import file_record, read_study_csv

RELIABLE_SHARE, UNRELIABLE_SHARE = 0.35, 0.65


def fold_predictions(runs, ids, fold):
    """Converted out-of-fold predictions, each study predicted by the model of its own validation fold."""
    teachers = [load_teacher(run) for run in runs]
    expected = sorted(set(fold.tolist()))
    if sorted(teacher["fold"] for teacher in teachers) != expected:
        raise ValueError(f"Give exactly one run per fold {expected}")
    for teacher in teachers:
        if set(teacher["predictions"].index) != set(ids[fold == teacher["fold"]]):
            raise ValueError(f"{teacher['run']}: predictions must cover exactly the studies of fold {teacher['fold']}")
    return pd.concat([teacher["predictions"] for teacher in teachers]).loc[ids].to_numpy(dtype=float), teachers


def prepare_round3_labels(current_csv, baseline_csv, report_csv, gpt_csv, splits_csv, earlier_runs, new_runs,
                          output_dir, reliable_share=RELIABLE_SHARE, unreliable_share=UNRELIABLE_SHARE) -> Path:
    paths = [Path(p) for p in (current_csv, baseline_csv, report_csv, gpt_csv, splits_csv)]
    output_dir = Path(output_dir)
    if any(output_dir.resolve() == path.parent.resolve() for path in paths):
        raise ValueError("The output directory must not hold an input")
    current, baseline, report, gpt, splits = (read_study_csv(paths[0]), read_study_csv(paths[1]).set_index(STUDY_ID),
                                              read_study_csv(paths[2]).set_index(STUDY_ID),
                                              read_study_csv(paths[3]).set_index(STUDY_ID),
                                              read_study_csv(paths[4], require_targets=False).set_index(STUDY_ID))
    ids_all = current[STUDY_ID]
    fold_all = splits.loc[ids_all, "fold"].astype(int).to_numpy()
    cross_validation = fold_all >= 0
    ids, fold = ids_all[cross_validation].reset_index(drop=True), fold_all[cross_validation]
    missing = set(ids) - set(gpt.index)
    if missing:
        raise ValueError(f"GPT labels are missing for {len(missing)} cross-validation studies")
    before = current.loc[cross_validation, TARGETS].to_numpy(dtype=float)
    model, new_teachers = fold_predictions(new_runs, ids, fold)
    earlier, _ = fold_predictions(earlier_runs, ids, fold)
    report_value = report.loc[ids, TARGETS].to_numpy(dtype=float)
    gpt_value = gpt.loc[ids, TARGETS].to_numpy(dtype=float)
    stated = baseline.loc[ids, TARGETS].notna().to_numpy()
    votes = np.stack([report_value >= 0.5, gpt_value >= 0.5, earlier >= 0.5])
    agree = votes.all(axis=0) | (~votes).all(axis=0)
    reliable = stated & agree
    share = np.where(reliable, reliable_share, unreliable_share)
    after = (1 - share) * before + share * model
    if not np.isfinite(after).all() or (after < 0).any() or (after > 1).any():
        raise AssertionError("Round-3 labels must be finite and between zero and one")
    labels = current.copy()
    labels.loc[cross_validation, TARGETS] = after
    if not labels.loc[~cross_validation].equals(current.loc[~cross_validation]):
        raise AssertionError("Expert-holdout and unlabeled rows must stay unchanged")

    summary = {name: {"reliable_cells": int(reliable[:, j].sum()), "other_cells": int((~reliable[:, j]).sum()),
                      "mean_before": round(float(before[:, j].mean()), 4), "mean_model": round(float(model[:, j].mean()), 4),
                      "mean_after": round(float(after[:, j].mean()), 4)} for j, name in enumerate(TARGETS)}
    records = {str(path): file_record(path) for path in paths}
    output_dir.mkdir(parents=True, exist_ok=True)
    labels_path = output_dir / "labels.csv"
    labels.to_csv(labels_path, index=False)
    provenance = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "rule": (f"cross-validation cells become (1-s) x current label + s x converted out-of-fold prediction; s = "
                 f"{reliable_share} where the report states the finding and report, GPT and earlier teachers agree on the "
                 f"side of 0.5, else {unreliable_share}; other rows unchanged"),
        "reliable_share": reliable_share, "unreliable_share": unreliable_share,
        "reliable_cell_fraction": round(float(reliable.mean()), 4), "mean_share": round(float(share.mean()), 4),
        "inputs": records, "new_runs": [t["run"] for t in sorted(new_teachers, key=lambda t: t["fold"])],
        "earlier_runs": [str(run) for run in earlier_runs], "per_finding": summary, "outcome_labels_read": False,
        "labels_csv": file_record(labels_path),
    }
    if any(file_record(path)["sha256"] != records[str(path)]["sha256"] for path in paths):
        raise RuntimeError("An input changed during preparation; the generated labels must not be used")
    (output_dir / "label_provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    return labels_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--current-labels", type=Path, default=Path("data/labels/student_v2_gpt_full/labels.csv"))
    parser.add_argument("--baseline-labels", type=Path, default=Path("data/labels/baseline/labels.csv"))
    parser.add_argument("--report-labels", type=Path, default=Path("data/labels/avg_v2/labels.csv"))
    parser.add_argument("--gpt-labels", type=Path, default=Path("artifacts/gpt-image-labels/v1/labels-gpt-cv-medium.csv"))
    parser.add_argument("--splits", type=Path, default=Path("data/splits/baseline/splits.csv"))
    parser.add_argument("--earlier-runs", type=Path, nargs="+",
                        default=[Path(run) for run in json.loads(Path("artifacts/teacher-recipe-decision.json").read_text())["teacher_runs"]])
    parser.add_argument("--new-runs", type=Path, nargs="+", default=[Path(f"runs/student_v2_gpt_full_fold{k}") for k in range(5)])
    parser.add_argument("--output-dir", type=Path, default=Path("data/labels/student_v3_round3"))
    args = parser.parse_args()
    path = prepare_round3_labels(args.current_labels, args.baseline_labels, args.report_labels, args.gpt_labels, args.splits,
                                 args.earlier_runs, args.new_runs, args.output_dir)
    record = json.loads((path.parent / "label_provenance.json").read_text())
    print(json.dumps({k: record[k] for k in ("reliable_cell_fraction", "mean_share")} | {"labels": str(path)}))


if __name__ == "__main__":
    main()
