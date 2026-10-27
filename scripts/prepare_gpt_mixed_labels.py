#!/usr/bin/env python3
"""Mix image-only GPT-6 Astra labels into the student training labels for the studies GPT has read.

The student labels come from prepare_student_labels.py and the GPT labels from gpt_image_labels.py.
Every finding of each such study becomes SHARE x GPT probability + (1 - SHARE) x student label.
The student labels are already about half report reading and half fold-model prediction where the
report is silent, so a one-third share weighs the three sources about equally. On the 57
expert-labeled studies a mix like this scored 0.942 macro AUC against 0.929 without GPT; shares from
0.25 to 0.67 all scored 0.941-0.943, so the share was fixed at one third rather than tuned.

Only cross-validation studies change. Expert-holdout and unlabeled rows, and studies GPT has not read,
are copied unchanged. GPT never saw the radiology report. Round 2 in the README mixed in GPT readings of all
cross-validation studies (output data/labels/student_v2_gpt_full, which later rounds start from).

Run: uv run python scripts/prepare_gpt_mixed_labels.py --gpt-labels <labels csv from gpt_image_labels.py parse>
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from rsna_knee_abnormality_detection.constants import STUDY_ID, TARGETS
from rsna_knee_abnormality_detection.labels import file_record, read_study_csv

SHARE = 1 / 3


def prepare_gpt_mixed_labels(student_csv: Path, gpt_csv: Path, splits_csv: Path, output_dir: Path,
                             share: float = SHARE) -> Path:
    if not 0 < share <= 1:
        raise ValueError("The GPT share must be above zero and at most one")
    student_csv, gpt_csv, splits_csv, output_dir = map(Path, (student_csv, gpt_csv, splits_csv, output_dir))
    input_paths = [student_csv, gpt_csv, splits_csv]
    if any(output_dir.resolve() == path.parent.resolve() for path in input_paths):
        raise ValueError("The output directory must not hold the input labels or splits")
    student = read_study_csv(student_csv)
    gpt = read_study_csv(gpt_csv).set_index(STUDY_ID)[TARGETS]
    splits = read_study_csv(splits_csv, require_targets=False).set_index(STUDY_ID)
    if set(student[STUDY_ID]) != set(splits.index):
        raise ValueError("Student labels and splits must contain the same studies")
    unknown = sorted(set(gpt.index) - set(student[STUDY_ID]))
    if unknown:
        raise ValueError(f"GPT labels name {len(unknown)} studies that are not in the student labels")
    if (splits.loc[gpt.index, "fold"].astype(int) < 0).any():
        raise ValueError("GPT labels may only change cross-validation studies, never expert-holdout or unlabeled ones")
    gpt_values = gpt.to_numpy(dtype=float)
    if not np.isfinite(gpt_values).all() or (gpt_values < 0).any() or (gpt_values > 1).any():
        raise ValueError("GPT probabilities must be complete and between zero and one")

    labels = student.copy()
    mixed = labels[STUDY_ID].isin(gpt.index).to_numpy()
    before = student.loc[mixed, TARGETS].to_numpy(dtype=float)
    after = (1 - share) * before + share * gpt.loc[labels.loc[mixed, STUDY_ID], TARGETS].to_numpy(dtype=float)
    labels.loc[mixed, TARGETS] = after
    values = labels[TARGETS].to_numpy(dtype=float)
    if not np.isfinite(values).all() or (values < 0).any() or (values > 1).any():
        raise AssertionError("Mixed labels must be finite and between zero and one")
    if not labels.loc[~mixed].equals(student.loc[~mixed]):
        raise AssertionError("Studies GPT has not read must stay unchanged")

    folds = splits.loc[labels.loc[mixed, STUDY_ID], "fold"].astype(int)
    summary = {name: {"mean_before": round(float(before[:, column].mean()), 4),
                      "mean_gpt": round(float(gpt[name].mean()), 4),
                      "mean_after": round(float(after[:, column].mean()), 4)} for column, name in enumerate(TARGETS)}
    inputs_before = {str(path): file_record(path) for path in input_paths}
    output_dir.mkdir(parents=True, exist_ok=True)
    labels_path = output_dir / "labels.csv"
    labels.to_csv(labels_path, index=False)
    provenance = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "rule": (f"For each study with GPT-6 Astra image labels, every finding becomes {share:.4f} x GPT probability + "
                 f"{1 - share:.4f} x student label; all other studies are unchanged."),
        "share": share,
        "inputs": {"student_labels": inputs_before[str(student_csv)], "gpt_labels": inputs_before[str(gpt_csv)],
                   "splits": inputs_before[str(splits_csv)]},
        "studies": len(labels),
        "studies_mixed": int(mixed.sum()),
        "studies_mixed_per_fold": {str(fold): int(count) for fold, count in folds.value_counts().sort_index().items()},
        "per_finding": summary,
        "limitations": [
            "GPT read only the images, at 512 px, with medium reasoning effort; its labels are estimates, not expert reads.",
            "The one-third share was checked, not tuned, on only 57 expert-labeled studies.",
        ],
        "outcome_labels_read": False,
        "labels_csv": file_record(labels_path),
        "reproduction": f"uv run python scripts/prepare_gpt_mixed_labels.py --gpt-labels {gpt_csv} --output-dir {output_dir}",
    }
    if any(file_record(path)["sha256"] != inputs_before[str(path)]["sha256"] for path in input_paths):
        raise RuntimeError("An input changed during preparation; the generated labels must not be used")
    (output_dir / "label_provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    return labels_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--student-labels", type=Path, default=Path("data/labels/student_v2/labels.csv"))
    parser.add_argument("--gpt-labels", type=Path, required=True)
    parser.add_argument("--splits", type=Path, default=Path("data/splits/baseline/splits.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/labels/student_v2_gpt_pilot"))
    parser.add_argument("--share", type=float, default=SHARE, help="Share of the GPT probability in each mixed cell")
    args = parser.parse_args()
    path = prepare_gpt_mixed_labels(args.student_labels, args.gpt_labels, args.splits, args.output_dir, args.share)
    record = json.loads((path.parent / "label_provenance.json").read_text())
    print(json.dumps({"labels": str(path), "studies_mixed": record["studies_mixed"],
                      "per_fold": record["studies_mixed_per_fold"]}))


if __name__ == "__main__":
    main()
