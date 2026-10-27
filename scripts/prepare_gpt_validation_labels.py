#!/usr/bin/env python3
"""Build checkpoint-selection labels from the severity-aware report labels and GPT-6 Astra image labels.

The inputs come from prepare_severity_validation_labels.py and gpt_image_labels.py.

Scoring counts only exact 0/1 labels, so the two sources are combined into plain yes/no cells:
- where the report states a finding, keep its 0/1 only if GPT agrees (GPT >= 0.5 for 1, < 0.5 for 0);
  where they disagree, leave the cell blank;
- where the report is silent, use GPT only when it is confident: >= HIGH gives 1, <= LOW gives 0.
Only cross-validation studies change. Expert-holdout rows (the clean check that tracks the
leaderboard) and unlabeled rows are copied unchanged, and no expert label is read.

Run: uv run python scripts/prepare_gpt_validation_labels.py --gpt-labels <labels csv>
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from rsna_knee_abnormality_detection.constants import STUDY_ID, TARGETS
from rsna_knee_abnormality_detection.labels import file_record, read_study_csv

HIGH, LOW = 0.9, 0.1


def prepare_gpt_validation_labels(severity_csv: Path, gpt_csv: Path, splits_csv: Path, output_dir: Path,
                                  high: float = HIGH, low: float = LOW) -> Path:
    if not 0 <= low < 0.5 <= high <= 1:
        raise ValueError("Need 0 <= low < 0.5 <= high <= 1")
    severity_csv, gpt_csv, splits_csv, output_dir = map(Path, (severity_csv, gpt_csv, splits_csv, output_dir))
    input_paths = [severity_csv, gpt_csv, splits_csv]
    if any(output_dir.resolve() == path.parent.resolve() for path in input_paths):
        raise ValueError("The output directory must not hold the input labels or splits")
    severity = read_study_csv(severity_csv)
    gpt = read_study_csv(gpt_csv).set_index(STUDY_ID)[TARGETS]
    splits = read_study_csv(splits_csv, require_targets=False).set_index(STUDY_ID)
    if set(severity[STUDY_ID]) != set(splits.index):
        raise ValueError("Severity labels and splits must contain the same studies")
    cross_validation = splits.loc[severity[STUDY_ID], "fold"].astype(int).to_numpy() >= 0
    ids = severity.loc[cross_validation, STUDY_ID]
    missing = sorted(set(ids) - set(gpt.index))
    if missing:
        raise ValueError(f"GPT labels are missing for {len(missing)} cross-validation studies")
    report = severity.loc[cross_validation, TARGETS].to_numpy(dtype=float)
    stated = np.isfinite(report)
    if not np.isin(report[stated], (0.0, 1.0)).all():
        raise ValueError("Severity labels must be 0, 1 or blank")
    probability = gpt.loc[ids, TARGETS].to_numpy(dtype=float)
    if not np.isfinite(probability).all() or (probability < 0).any() or (probability > 1).any():
        raise ValueError("GPT probabilities must be complete and between zero and one")
    agree = stated & ((probability >= 0.5) == (report == 1))
    after = np.full(report.shape, np.nan)
    after[agree] = report[agree]
    after[~stated & (probability >= high)] = 1.0
    after[~stated & (probability <= low)] = 0.0
    labels = severity.copy()
    labels.loc[cross_validation, TARGETS] = after
    if not labels.loc[~cross_validation].equals(severity.loc[~cross_validation]):
        raise AssertionError("Expert-holdout and unlabeled rows must stay unchanged")

    summary = {}
    for column, name in enumerate(TARGETS):
        said, quiet = stated[:, column], ~stated[:, column]
        summary[name] = {
            "report_cells": int(said.sum()), "kept_where_gpt_agrees": int(agree[:, column].sum()),
            "dropped_disagreements": int((said & ~agree[:, column]).sum()),
            "silent_cells": int(quiet.sum()),
            "silent_filled_positive": int((quiet & (probability[:, column] >= high)).sum()),
            "silent_filled_negative": int((quiet & (probability[:, column] <= low)).sum()),
            "positives_after": int((after[:, column] == 1).sum()), "negatives_after": int((after[:, column] == 0).sum()),
        }
    inputs_before = {str(path): file_record(path) for path in input_paths}
    output_dir.mkdir(parents=True, exist_ok=True)
    labels_path = output_dir / "labels.csv"
    labels.to_csv(labels_path, index=False)
    provenance = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "rule": (f"Stated report cells kept only where GPT agrees (>= 0.5 for 1, < 0.5 for 0), otherwise blank; silent "
                 f"cells become 1 when GPT >= {high} and 0 when GPT <= {low}; expert-holdout and unlabeled rows unchanged."),
        "high": high, "low": low,
        "inputs": {"severity_labels": inputs_before[str(severity_csv)], "gpt_labels": inputs_before[str(gpt_csv)],
                   "splits": inputs_before[str(splits_csv)]},
        "cross_validation_studies": int(cross_validation.sum()), "per_finding": summary,
        "purpose": "checkpoint selection only; the 58 expert studies stay untouched for the final check",
        "outcome_labels_read": False, "labels_csv": file_record(labels_path),
    }
    if any(file_record(path)["sha256"] != inputs_before[str(path)]["sha256"] for path in input_paths):
        raise RuntimeError("An input changed during preparation; the generated labels must not be used")
    (output_dir / "label_provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    return labels_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--severity-labels", type=Path, default=Path("data/labels/baseline_severity/labels.csv"))
    parser.add_argument("--gpt-labels", type=Path, required=True)
    parser.add_argument("--splits", type=Path, default=Path("data/splits/baseline/splits.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/labels/severity_gpt_agreement"))
    args = parser.parse_args()
    path = prepare_gpt_validation_labels(args.severity_labels, args.gpt_labels, args.splits, args.output_dir)
    record = json.loads((path.parent / "label_provenance.json").read_text())
    print(json.dumps({"labels": str(path), "per_finding": record["per_finding"]}))


if __name__ == "__main__":
    main()
