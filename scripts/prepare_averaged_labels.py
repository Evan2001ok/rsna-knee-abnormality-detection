#!/usr/bin/env python3
"""Build training labels that average several public report readings where the report states a finding.

Cells the report states (non-blank in the baseline report labels, data/labels/baseline/labels.csv, made by
`knee labels` from Kaggle user pilkwang's yes/no report readings, see fetch_public_labels.py) become the mean of the
available readings: the Raptor value (dreaddevelopment's public report labels, see prepare_raptor_labels.py) plus
each extra source whose value there is not that source's own "not mentioned" code. Cells the report does not mention keep the Raptor value, whose per-finding
defaults already match how often experts found each finding when reports were silent.
--findings limits averaging to some findings; the rest keep the Raptor value. The severity-aware
version averages only findings without a severity threshold, because only Raptor scores mild
mentions low and the host counts only moderate or high-grade ACL, MCL, OA, effusion and Baker's.
Expert-holdout and unlabeled rows are copied unchanged. Pilkwang readings are not averaged in,
because they are the validation labels. No expert labels are read and nothing is tuned.

The two extra readings are public Kaggle datasets (CC0). Download them by hand from Kaggle and put the files at
the fixed paths in DEFAULT_SOURCES (there is no option to change them):
  stevenleehans v2:     https://www.kaggle.com/datasets/stevenleehans/rsna-knee-llm-report-labels
                        -> artifacts/public-training-review/labels/stevenleehans_rsna-knee-llm-report-labels/llm_labels_v2.csv
                        (0.5 is its "not mentioned" code)
  lixin73 GPT-5.6-Sol:  https://www.kaggle.com/datasets/lixin73/rsna-knee-llm-report-labels-sol56
                        -> artifacts/public-training-review/labels/lixin73_rsna-knee-llm-report-labels-sol56/labels_llm_gpt56sol.csv
                        (no such code; only blank cells are skipped)
We downloaded both on October 5, 2026; later versions on Kaggle may differ. label_provenance.json records the
SHA-256 of every input.

Run: uv run python scripts/prepare_averaged_labels.py   (all findings, written to data/labels/avg_v1)
Our standard labels (avg_v2, used in round 1 of the README) average only the five findings without a severity
threshold:
  uv run python scripts/prepare_averaged_labels.py --findings "Medial Meniscus" "Lateral Meniscus" Synovitis \
      Contusion Fracture --output-dir data/labels/avg_v2
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from rsna_knee_abnormality_detection.constants import STUDY_ID, TARGETS
from rsna_knee_abnormality_detection.labels import file_record, read_study_csv

PUBLIC = Path("artifacts/public-training-review/labels")
DEFAULT_SOURCES = [
    (PUBLIC / "stevenleehans_rsna-knee-llm-report-labels/llm_labels_v2.csv", 0.5),
    (PUBLIC / "lixin73_rsna-knee-llm-report-labels-sol56/labels_llm_gpt56sol.csv", None),
]


def prepare_averaged_labels(raptor_csv: Path, baseline_csv: Path, splits_csv: Path,
                            sources: list[tuple[Path, float | None]], output_dir: Path,
                            findings: list[str] | None = None) -> Path:
    raptor_csv, baseline_csv, splits_csv, output_dir = map(Path, (raptor_csv, baseline_csv, splits_csv, output_dir))
    input_paths = [raptor_csv, baseline_csv, splits_csv, *(Path(path) for path, _ in sources)]
    if any(output_dir.resolve() == path.parent.resolve() for path in input_paths):
        raise ValueError("The output directory must not hold any input")
    if not sources:
        raise ValueError("Give at least one extra reading to average")
    findings = list(TARGETS if findings is None else findings)
    if not findings or not set(findings) <= set(TARGETS):
        raise ValueError(f"Findings must be a non-empty subset of {TARGETS}")
    raptor = read_study_csv(raptor_csv)
    baseline = read_study_csv(baseline_csv).set_index(STUDY_ID)
    splits = read_study_csv(splits_csv, require_targets=False).set_index(STUDY_ID)
    ids = raptor[STUDY_ID]
    if set(ids) != set(baseline.index) or set(ids) != set(splits.index):
        raise ValueError("Raptor labels, baseline labels and splits must contain the same studies")
    cross_validation = splits.loc[ids, "fold"].astype(int).to_numpy() >= 0
    if not (raptor.loc[cross_validation, "label_source"].eq("report").all()
            and baseline.loc[ids[cross_validation], "label_source"].eq("report").all()):
        raise ValueError("Every cross-validation study must have report-derived labels")
    cv_ids = ids[cross_validation]
    before = raptor.loc[cross_validation, TARGETS].to_numpy(dtype=float)
    stated = baseline.loc[cv_ids, TARGETS].notna().to_numpy() & np.isin(TARGETS, findings)[None, :]

    total, count = before.copy(), np.ones_like(before)
    readings = {}
    for path, missing_code in sources:
        frame = read_study_csv(path).set_index(STUDY_ID)
        if not set(cv_ids) <= set(frame.index):
            raise ValueError(f"{path}: some cross-validation studies are missing")
        values = frame.loc[cv_ids, TARGETS].to_numpy(dtype=float)
        usable = np.isfinite(values)
        if missing_code is not None:
            usable &= ~np.isclose(values, missing_code)
        if ((values[usable] < 0) | (values[usable] > 1)).any():
            raise ValueError(f"{path}: readings must be between zero and one")
        used = usable & stated
        total[used] += values[used]
        count[used] += 1
        readings[str(path)] = {"missing_code": missing_code, "cells_used": int(used.sum())}
    after = np.where(stated, total / count, before)

    labels = raptor.copy()
    labels.loc[cross_validation, TARGETS] = after
    values = labels[TARGETS].to_numpy(dtype=float)
    if not np.isfinite(values).all() or (values < 0).any() or (values > 1).any():
        raise AssertionError("Averaged labels must be finite and between zero and one")
    if not labels.loc[~cross_validation].equals(raptor.loc[~cross_validation]):
        raise AssertionError("Expert-holdout and unlabeled rows must stay unchanged")
    if not np.array_equal(after[~stated], before[~stated]):
        raise AssertionError("Cells the report does not mention must keep their Raptor value")

    summary = {}
    for column, name in enumerate(TARGETS):
        cells = stated[:, column]
        summary[name] = {
            "stated_cells": int(cells.sum()),
            "stated_mean_before": round(float(before[cells, column].mean()), 4) if cells.any() else None,
            "stated_mean_after": round(float(after[cells, column].mean()), 4) if cells.any() else None,
            "mean_readings_per_stated_cell": round(float(count[cells, column].mean()), 3) if cells.any() else None,
        }
    inputs_before = {str(path): file_record(path) for path in input_paths}
    output_dir.mkdir(parents=True, exist_ok=True)
    labels_path = output_dir / "labels.csv"
    labels.to_csv(labels_path, index=False)
    provenance = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "rule": ("For the averaged findings, cells the baseline report labels state become the mean of the Raptor value "
                 "and each extra reading that is not that source's 'not mentioned' code; unmentioned cells and other "
                 "findings keep the Raptor value; expert-holdout and unlabeled rows are unchanged."),
        "averaged_findings": findings,
        "inputs": inputs_before,
        "extra_readings": readings,
        "studies": len(labels),
        "changed_cells": int((~np.isclose(after, before)).sum()),
        "per_finding": summary,
        "configuration": {"data.labels": str(labels_path), "data.validation_labels": str(baseline_csv),
                          "data.splits": str(splits_csv)},
        "limitations": [
            "Which cells count as stated comes from the Pilkwang reading; other readers may disagree about mentions.",
            "Equal weights are fixed in advance, not tuned.",
            "Pilkwang readings are left out of the average because they are the validation labels.",
        ],
        "outcome_labels_read": False,
        "labels_csv": file_record(labels_path),
        "reproduction": "uv run python scripts/prepare_averaged_labels.py",
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
    parser.add_argument("--output-dir", type=Path, default=Path("data/labels/avg_v1"))
    parser.add_argument("--findings", nargs="+", choices=TARGETS, help="Average only these findings (default: all)")
    args = parser.parse_args()
    path = prepare_averaged_labels(args.raptor_labels, args.baseline_labels, args.splits, DEFAULT_SOURCES,
                                   args.output_dir, args.findings)
    record = json.loads((path.parent / "label_provenance.json").read_text())
    print(json.dumps({"labels": str(path), "changed_cells": record["changed_cells"], "per_finding": record["per_finding"]}))


if __name__ == "__main__":
    main()
