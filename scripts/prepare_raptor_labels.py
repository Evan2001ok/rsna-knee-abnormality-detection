#!/usr/bin/env python3
"""Prepare the public Raptor soft targets without changing our baseline split.

Raptor labels are dreaddevelopment's public report readings (Kaggle dataset rsna-knee-labels, version 3, CC0).
The baseline labels (data/labels/baseline) come from `knee labels`, and the baseline split (data/splits/baseline)
from `knee split` with its default 5 folds.

Run from the repository with ``uv run python scripts/prepare_raptor_labels.py``.
The local source snapshot is used by default. On a fresh checkout, add
``--download-source`` to fetch the pinned public dataset version if it is absent.

Set data.labels to the generated labels.csv, data.validation_labels to the
original baseline labels.csv, and keep data.splits unchanged. Public soft
targets are for fitting the model; baseline hard targets and their missing-value
masks remain the evaluation reference. This script never scores predictions or
chooses a threshold, and never calls a language model.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import shutil
import urllib.request
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from rsna_knee_abnormality_detection.constants import STUDY_ID, TARGETS
from rsna_knee_abnormality_detection.labels import file_record, read_study_csv

SOURCE_URL = "https://www.kaggle.com/datasets/dreaddevelopment/rsna-knee-labels"
DOWNLOAD_URL = (
    "https://www.kaggle.com/api/v1/datasets/download/dreaddevelopment/rsna-knee-labels"
    "?datasetVersionNumber=3"
)
EXPECTED_SOURCE_SHA256 = "8cec477fb555e88bbb4196197fd3f806cbafed8cc2f6b2d48d26a8d49fbd86a1"
SOURCE_FILENAME = "labels_llm_soft.csv"
EXPECTED_STUDIES = 4407
EXPECTED_EXPERT_STUDIES = 58
EXPECTED_PARTITIONS = {"training": 3443, "validation": 853, "expert_holdout": 58, "excluded": 53}
MAX_ARCHIVE_BYTES = 10 * 1024 * 1024


def download_source(path: Path) -> Path:
    """Fetch only an existing, checksum-pinned label CSV; do not execute code."""
    path = Path(path)
    if path.exists():
        if file_record(path)["sha256"] != EXPECTED_SOURCE_SHA256:
            raise ValueError("Existing source CSV does not match the pinned Raptor snapshot")
        return path
    request = urllib.request.Request(DOWNLOAD_URL, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(request, timeout=60) as response:
        archive_bytes = response.read(MAX_ARCHIVE_BYTES + 1)
    if len(archive_bytes) > MAX_ARCHIVE_BYTES:
        raise ValueError("The Raptor label archive exceeded the expected size limit")
    with zipfile.ZipFile(io.BytesIO(archive_bytes)) as archive:
        member = archive.getinfo(SOURCE_FILENAME)
        if member.file_size > MAX_ARCHIVE_BYTES:
            raise ValueError("The Raptor label CSV exceeded the expected size limit")
        source_bytes = archive.read(SOURCE_FILENAME)
    if hashlib.sha256(source_bytes).hexdigest() != EXPECTED_SOURCE_SHA256:
        raise ValueError("Downloaded source CSV does not match the pinned Raptor snapshot")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(source_bytes)
    return path


def _partition_record(ids) -> dict:
    """Hash sorted, exact IDs so subsequent runs can verify split membership."""
    values = sorted(map(str, ids))
    encoded = json.dumps(values, ensure_ascii=False, separators=(",", ":")).encode()
    return {"studies": len(values), "sorted_ids_sha256": hashlib.sha256(encoded).hexdigest()}


def prepare_raptor_labels(
    source_csv: Path,
    train_csv: Path,
    baseline_labels_csv: Path,
    splits_csv: Path,
    output_dir: Path,
) -> Path:
    """Join the V3 public snapshot onto our fixed baseline study population.

    The fixed population checks are intentional: this is a controlled change to
    the existing fold-0 experiment, not a new split or a general label importer.
    Soft source values remain unchanged, including values assigned by the public
    author when a report is uncertain. The author has not published enough
    information to independently reconstruct those uncertainty judgments.
    """
    source_csv, train_csv = Path(source_csv), Path(train_csv)
    baseline_labels_csv, splits_csv = Path(baseline_labels_csv), Path(splits_csv)
    output_dir = Path(output_dir)
    input_paths = [source_csv, train_csv, baseline_labels_csv, splits_csv]
    output_paths = [output_dir / name for name in ["labels.csv", "original_raptor_labels.csv", "label_provenance.json"]]
    if output_dir.resolve() == baseline_labels_csv.parent.resolve() or any(
        destination.resolve() == source.resolve() for destination in output_paths for source in input_paths
    ):
        raise ValueError("The output must not overwrite baseline labels, splits, or source inputs")

    train = read_study_csv(train_csv)
    baseline = read_study_csv(baseline_labels_csv)
    source = read_study_csv(source_csv)
    splits = read_study_csv(splits_csv, require_targets=False)
    if len(train) != EXPECTED_STUDIES:
        raise ValueError(f"Expected {EXPECTED_STUDIES} competition training studies")
    train_ids = set(train[STUDY_ID])
    if set(baseline[STUDY_ID]) != train_ids or set(splits[STUDY_ID]) != train_ids:
        raise ValueError("Baseline labels and splits must contain exactly the competition training study IDs")
    expert_mask = train[TARGETS].notna().any(axis=1)
    expert = train.loc[expert_mask].set_index(STUDY_ID)[TARGETS]
    if len(expert) != EXPECTED_EXPERT_STUDIES or not expert.isin([0.0, 1.0]).all().all():
        raise ValueError(f"Expected {EXPECTED_EXPERT_STUDIES} completely labeled binary expert studies")
    expert_ids = set(expert.index)
    if set(source[STUDY_ID]) != train_ids - expert_ids:
        raise ValueError("Raptor source IDs must be exactly training studies minus the 58 expert studies")
    if not np.isfinite(source[TARGETS].to_numpy()).all():
        raise ValueError("Every Raptor source finding must be finite; missing soft targets are not allowed")
    if file_record(source_csv)["sha256"] != EXPECTED_SOURCE_SHA256:
        raise ValueError("Source CSV does not match the pinned Raptor dataset V3 snapshot")

    if "label_source" not in baseline or not baseline["label_source"].isin(["expert", "report", "missing"]).all():
        raise ValueError("Baseline labels need valid label_source values")
    if set(baseline.loc[baseline["label_source"].eq("expert"), STUDY_ID]) != expert_ids:
        raise ValueError("Baseline expert study IDs do not match the original competition labels")
    baseline_expert = baseline.set_index(STUDY_ID).loc[expert.index, TARGETS]
    if not np.array_equal(baseline_expert.to_numpy(), expert.to_numpy()):
        raise ValueError("Baseline expert target values differ from the original competition labels")
    baseline_values = baseline[TARGETS].to_numpy()
    if not np.isin(baseline_values[np.isfinite(baseline_values)], [0.0, 1.0]).all():
        raise ValueError("Baseline evaluation targets must stay binary or missing")
    if not baseline["label_source"].eq("missing").equals(baseline[TARGETS].isna().all(axis=1)):
        raise ValueError("Baseline missing-label metadata disagrees with its target values")

    if not {"fold", "split", "group_id"}.issubset(splits.columns):
        raise ValueError("Baseline splits require fold, split, and group_id columns")
    fold = pd.to_numeric(splits["fold"], errors="raise")
    if fold.isna().any() or not np.isfinite(fold).all() or (fold != np.floor(fold)).any():
        raise ValueError("Baseline fold values must be finite integers")
    if not splits["split"].isin(["train", "expert_holdout", "unlabeled"]).all():
        raise ValueError("Unexpected baseline split category")
    if (splits["split"].eq("train") & ~fold.between(0, 4)).any():
        raise ValueError("Baseline training folds must be between 0 and 4")
    if (splits["split"].eq("expert_holdout") & fold.ne(-1)).any():
        raise ValueError("Expert holdout studies must remain in fold -1")
    if (splits["split"].eq("unlabeled") & fold.ne(-2)).any():
        raise ValueError("Baseline excluded studies must remain in fold -2")
    if splits["group_id"].isna().any() or splits["group_id"].str.strip().eq("").any():
        raise ValueError("Baseline group IDs must not be blank")
    if splits.assign(fold=fold).groupby("group_id")["fold"].nunique().gt(1).any():
        raise ValueError("A baseline patient group crosses split partitions")
    holdout_ids = set(splits.loc[splits["split"].eq("expert_holdout"), STUDY_ID])
    if holdout_ids != expert_ids:
        raise ValueError("The original 58 expert studies must remain exactly the expert holdout")

    partitions = {
        "training": splits.loc[splits["split"].eq("train") & fold.ne(0), STUDY_ID],
        "validation": splits.loc[splits["split"].eq("train") & fold.eq(0), STUDY_ID],
        "expert_holdout": splits.loc[splits["split"].eq("expert_holdout"), STUDY_ID],
        "excluded": splits.loc[splits["split"].eq("unlabeled"), STUDY_ID],
    }
    if {name: len(ids) for name, ids in partitions.items()} != EXPECTED_PARTITIONS:
        raise ValueError(f"Baseline fold-0 partition sizes changed; expected {EXPECTED_PARTITIONS}")
    if set(partitions["excluded"]) != set(baseline.loc[baseline["label_source"].eq("missing"), STUDY_ID]):
        raise ValueError("The original 53 missing-label studies must remain excluded")

    labels = source.set_index(STUDY_ID)[TARGETS].reindex(baseline[STUDY_ID]).copy()
    labels.loc[expert.index, TARGETS] = expert
    labels["label_source"] = np.where(labels.index.isin(expert_ids), "expert", "report")
    labels = labels.reset_index()
    # Membership depends exclusively on the existing split file. In particular,
    # gaining a source target does not admit the 53 excluded studies to fitting.
    inputs_before = {str(path): file_record(path) for path in input_paths}
    output_dir.mkdir(parents=True, exist_ok=True)
    labels_path = output_dir / "labels.csv"
    labels.to_csv(labels_path, index=False)
    shutil.copyfile(source_csv, output_dir / "original_raptor_labels.csv")
    provenance = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": {
            "url": SOURCE_URL,
            "download_url": DOWNLOAD_URL,
            "owner": "dreaddevelopment",
            "author_display_name": "Dread Development / Johnathan Wagner",
            "dataset_id": 11640568,
            "dataset_version": 3,
            "license": "CC0: Public Domain",
            "kind": "Public LLM readings of competition radiology reports, not image-model predictions",
            "file": inputs_before[str(source_csv)],
            "pinned_csv_sha256": EXPECTED_SOURCE_SHA256,
        },
        "original_train_csv": inputs_before[str(train_csv)],
        "validation_labels_csv": inputs_before[str(baseline_labels_csv)],
        "unchanged_splits_csv": inputs_before[str(splits_csv)],
        "fold": 0,
        "studies": len(labels),
        "study_sources": {"report": len(source), "expert": len(expert)},
        "partitions": {name: _partition_record(ids) for name, ids in partitions.items()},
        "policy": {
            "training_targets": "Use public soft values unchanged; original expert targets override and remain reserved",
            "evaluation_targets": "Use original baseline hard labels and missing-value masks, not these soft labels",
            "split_membership": "Existing baseline splits are unchanged, including all 53 excluded studies",
            "threshold_selection": "None; no target thresholding or scoring is performed",
            "external_model_calls": "None",
        },
        "configuration": {
            "data.labels": str(labels_path),
            "data.validation_labels": str(baseline_labels_csv),
            "data.splits": str(splits_csv),
            "data.fold": 0,
        },
        "limitations": [
            "These are report-derived training targets, not expert image annotations or proven calibrated probabilities.",
            "The exact public author's LLM and prompt are not established by the dataset description.",
            "Excluding expert IDs from this file does not prove the author never used their labels to develop the recipe.",
            "Source uncertainty values are preserved as a deliberate alternative training recipe; they are not newly observed negative findings.",
            "Changing the label source together with the image model means this experiment cannot isolate architecture gains alone.",
        ],
        "labels_csv": file_record(labels_path),
        "source_copy": file_record(output_dir / "original_raptor_labels.csv"),
        "reproduction": "uv run python scripts/prepare_raptor_labels.py --download-source",
    }
    if any(file_record(path)["sha256"] != inputs_before[str(path)]["sha256"] for path in input_paths):
        raise RuntimeError("An input changed during preparation; the generated labels must not be used")
    (output_dir / "label_provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    return labels_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-csv", type=Path, default=Path("artifacts/public-training-review/raptor/labels_llm_soft.csv"))
    parser.add_argument("--train-csv", type=Path, default=Path("data/rsna-knee-abnormality-detection/train.csv"))
    parser.add_argument("--baseline-labels", type=Path, default=Path("data/labels/baseline/labels.csv"))
    parser.add_argument("--splits", type=Path, default=Path("data/splits/baseline/splits.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/labels/raptor"))
    parser.add_argument("--download-source", action="store_true", help="Download the pinned CSV only if it is absent")
    args = parser.parse_args()
    if args.download_source:
        download_source(args.source_csv)
    path = prepare_raptor_labels(args.source_csv, args.train_csv, args.baseline_labels, args.splits, args.output_dir)
    print(f"Prepared Raptor soft training labels: {path}")
    print(f"Keep evaluation labels at {args.baseline_labels} and splits at {args.splits}")


if __name__ == "__main__":
    main()
