"""Import report-derived targets without silently inventing negative findings.

This module imports an existing report-label CSV; it does not extract diagnoses
from report text. Blank labels remain unknown. Image-model predictions belong in
a separate, fold-aware teacher/student workflow, not in this report-label input.
"""

from __future__ import annotations

import csv
import hashlib
import json
import shutil
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from .constants import STUDY_ID, TARGETS


def file_record(path: Path) -> dict:
    """Describe a local input without including reports or credentials."""
    path = Path(path)
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return {"path": str(path.resolve()), "sha256": digest.hexdigest(), "bytes": path.stat().st_size}


def read_study_csv(path: Path, *, require_targets: bool = True) -> pd.DataFrame:
    """Read study IDs exactly and reject ambiguous IDs or duplicate columns."""
    path = Path(path)
    with path.open(newline="", encoding="utf-8-sig") as stream:
        header = next(csv.reader(stream), [])
    duplicates = sorted({name for name in header if header.count(name) > 1})
    if duplicates:
        raise ValueError(f"{path}: duplicate columns: {duplicates}")
    required = [STUDY_ID, *TARGETS] if require_targets else [STUDY_ID]
    missing = [name for name in required if name not in header]
    if missing:
        raise ValueError(f"{path}: missing required columns: {missing}")
    frame = pd.read_csv(path, dtype={STUDY_ID: str, "group_id": str})
    ids = frame[STUDY_ID]
    if ids.isna().any() or ids.str.strip().eq("").any():
        raise ValueError(f"{path}: blank study IDs are not allowed")
    if ids.ne(ids.str.strip()).any():
        raise ValueError(f"{path}: study IDs contain leading or trailing whitespace")
    if ids.duplicated().any():
        raise ValueError(f"{path}: duplicate study IDs are not allowed")
    if require_targets:
        for target in TARGETS:
            try:
                values = pd.to_numeric(frame[target], errors="raise").astype(float)
            except (TypeError, ValueError) as error:
                raise ValueError(f"{path}: {target} must be numeric or blank") from error
            known = values.dropna()
            if not np.isfinite(known).all() or not known.between(0, 1).all():
                raise ValueError(f"{path}: {target} must be between 0 and 1, or blank")
            frame[target] = values
    return frame


def _copy_original(source: Path, destination: Path) -> None:
    if source.resolve() != destination.resolve():
        shutil.copyfile(source, destination)


def prepare_labels(
    train_csv: Path,
    output_dir: Path,
    weak_csv: Path | None = None,
    source: str | None = None,
) -> Path:
    """Write canonical labels and a reproducible account of their origin.

    ``weak_csv`` must contain the exact competition target names and study IDs.
    It may cover a subset of training studies and may contain soft probabilities.
    It must describe labels extracted from reports, rather than predictions from
    an image model. Supplied expert labels always win on a per-target basis.
    Unknown entries stay NaN and must be masked out of the training loss.

    Expert comparisons are diagnostics, not an instruction to select a label
    source using the same expert holdout that will judge the final image model.
    """
    train_csv, output_dir = Path(train_csv), Path(output_dir)
    train = read_study_csv(train_csv)
    if train.empty:
        raise ValueError("The training CSV has no studies")
    expert_values = train[TARGETS].to_numpy()
    if np.any(np.isfinite(expert_values) & ~np.isin(expert_values, [0.0, 1.0])):
        raise ValueError("Expert targets in train.csv must be binary or blank")
    expert = train.set_index(STUDY_ID)[TARGETS]
    report = pd.DataFrame(np.nan, index=expert.index, columns=TARGETS)
    weak_record = None
    if weak_csv is not None:
        weak_csv = Path(weak_csv)
        weak = read_study_csv(weak_csv)
        extra_ids = set(weak[STUDY_ID]) - set(train[STUDY_ID])
        if extra_ids:
            raise ValueError(f"Report labels contain {len(extra_ids)} study IDs absent from train.csv")
        report.loc[weak[STUDY_ID], TARGETS] = weak.set_index(STUDY_ID)[TARGETS]
        weak_record = file_record(weak_csv)
    else:
        warnings.warn(
            "No report-label CSV was supplied. Studies without expert labels remain unlabeled.",
            UserWarning,
            stacklevel=2,
        )

    # DataFrame.combine_first fills ONLY missing values. It never rounds or
    # thresholds a soft report target and never overwrites an expert judgment.
    combined = expert.combine_first(report).reindex(expert.index)[TARGETS]
    target_sources = pd.DataFrame("missing", index=expert.index, columns=TARGETS)
    target_sources[report.notna()] = "report"
    target_sources[expert.notna()] = "expert"
    has_expert = expert.notna().any(axis=1)
    has_report = report.notna().any(axis=1)
    labels = combined.reset_index()
    labels["label_source"] = np.select([has_expert, has_report], ["expert", "report"], default="missing")

    output_dir.mkdir(parents=True, exist_ok=True)
    _copy_original(train_csv, output_dir / "original_train.csv")
    if weak_csv is not None:
        _copy_original(weak_csv, output_dir / "original_report_labels.csv")
    expert.loc[has_expert].reset_index().to_csv(output_dir / "expert_labels.csv", index=False)
    report.reset_index().to_csv(output_dir / "report_labels.csv", index=False)
    combined.notna().astype("uint8").reset_index().to_csv(output_dir / "label_masks.csv", index=False)
    target_sources.reset_index().to_csv(output_dir / "label_sources.csv", index=False)
    labels_path = output_dir / "labels.csv"
    labels.to_csv(labels_path, index=False)

    diagnostics = {}
    for target in TARGETS:
        shared = expert[target].notna() & report[target].notna()
        truth, predicted = expert.loc[shared, target], report.loc[shared, target]
        diagnostics[target] = {
            "shared_expert_studies": int(shared.sum()),
            "agreement_at_0_5": float((truth == (predicted >= 0.5)).mean()) if shared.any() else None,
            "mean_absolute_difference": float((truth - predicted).abs().mean()) if shared.any() else None,
            "roc_auc": float(roc_auc_score(truth, predicted)) if truth.nunique() == 2 else None,
        }
    provenance = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "train_csv": file_record(train_csv),
        "report_labels_csv": weak_record,
        "report_label_source": source or (str(weak_csv.resolve()) if weak_csv is not None else None),
        "policy": {
            "expert_labels_override_report_labels": True,
            "blank_targets": "unknown; excluded from loss",
            "unmentioned_findings": "no automatic conversion to negative",
            "soft_targets": "preserved without rounding",
            "expert_comparisons": "diagnostic only; the 58-study holdout is small",
            "input_contract": "report-derived labels; no image-teacher predictions",
        },
        "studies": len(labels),
        "study_sources": {key: int(value) for key, value in labels["label_source"].value_counts().items()},
        "coverage": {
            target: {
                "expert": int(expert[target].notna().sum()),
                "report": int(report[target].notna().sum()),
                "combined": int(combined[target].notna().sum()),
                "missing": int(combined[target].isna().sum()),
                "soft": int((combined[target].notna() & ~combined[target].isin([0.0, 1.0])).sum()),
            }
            for target in TARGETS
        },
        "report_vs_expert_diagnostics": diagnostics,
        "labels_csv": file_record(labels_path),
    }
    (output_dir / "label_provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    return labels_path
