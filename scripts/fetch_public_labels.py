#!/usr/bin/env python3
"""Fetch a pinned public report-label file and retain explicit unknown findings.

Run from the repository with:
    uv run python scripts/fetch_public_labels.py

The source is Kaggle user pilkwang's dataset rsna-knee-llm-labels, version 1 (CC0), called "Pilkwang" in
the other scripts. report_labels_binary.csv (1 = present, 0 = absent, blank = unknown) is the yes/no file
for `knee labels --weak-csv`; the raw report_labels_v2.csv is read by prepare_severity_validation_labels.py.

This downloads existing public labels. It never runs the bundled labeling code,
sends a report to an external model, or uses API credentials.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import urllib.request
import zipfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from rsna_knee_abnormality_detection.constants import STUDY_ID, TARGETS

SOURCE_URL = "https://www.kaggle.com/datasets/pilkwang/rsna-knee-llm-labels"
DOWNLOAD_URL = "https://www.kaggle.com/api/v1/datasets/download/pilkwang/rsna-knee-llm-labels?datasetVersionNumber=1"
RAW_FILENAME = "report_labels_v2.csv"
EXPECTED_CSV_SHA256 = "6f704a7bdb2f894cc49445b19ba7c4378c3f548d3449e00361e10044bee40920"
MAX_ARCHIVE_BYTES = 10 * 1024 * 1024


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def fetch_public_labels(output_dir: Path, train_csv: Path | None = None) -> Path:
    """Download version 1, verify its content, and write explicit report targets."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    archive_path = output_dir / "rsna-knee-llm-labels-v1.zip"
    if archive_path.is_file():
        archive_bytes = archive_path.read_bytes()
    else:
        request = urllib.request.Request(DOWNLOAD_URL, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(request, timeout=60) as response:
            archive_bytes = response.read(MAX_ARCHIVE_BYTES + 1)
        if len(archive_bytes) > MAX_ARCHIVE_BYTES:
            raise ValueError("The public label archive exceeded the expected size limit")
    with zipfile.ZipFile(io.BytesIO(archive_bytes)) as archive:
        member = archive.getinfo(RAW_FILENAME)
        if member.file_size > MAX_ARCHIVE_BYTES:
            raise ValueError("The public label CSV exceeded the expected size limit")
        raw_bytes = archive.read(RAW_FILENAME)
        if sha256(raw_bytes) != EXPECTED_CSV_SHA256:
            raise ValueError("The pinned public-label CSV has changed; inspect the source before importing it")
        # Retain the author's extraction code as evidence; do not import or run it.
        author_code = archive.read("api_labeler.py")

    reader = csv.DictReader(io.StringIO(raw_bytes.decode("utf-8-sig")))
    rows = list(reader)
    ids = [row[STUDY_ID] for row in rows]
    if len(ids) != len(set(ids)) or any(not uid.strip() for uid in ids):
        raise ValueError("The public-label CSV has duplicate or blank study IDs")
    missing_training_studies = None
    if train_csv is not None:
        with Path(train_csv).open(newline="", encoding="utf-8-sig") as stream:
            training_ids = {row[STUDY_ID] for row in csv.DictReader(stream)}
        if set(ids) - training_ids:
            raise ValueError("The public-label CSV includes studies absent from the provided training CSV")
        missing_training_studies = len(training_ids - set(ids))

    binary_rows, score_rows, counts = [], [], {target: Counter() for target in TARGETS}
    for row in rows:
        binary = {STUDY_ID: row[STUDY_ID]}
        scored = {STUDY_ID: row[STUDY_ID]}
        for target in TARGETS:
            verdict = row[f"{target}__verdict"]
            if verdict not in {"YES", "NO", "UNK"}:
                raise ValueError(f"Unexpected verdict for {target}: {verdict}")
            counts[target][verdict] += 1
            binary[target] = {"YES": 1, "NO": 0, "UNK": ""}[verdict]
            # The author's UNK score is 0.28, a prior, not an observed negative.
            # Neither output treats that number as evidence about the finding.
            scored[target] = row[target] if verdict != "UNK" else ""
        binary_rows.append(binary)
        score_rows.append(scored)

    archive_path.write_bytes(archive_bytes)
    (output_dir / RAW_FILENAME).write_bytes(raw_bytes)
    (output_dir / "api_labeler.py").write_bytes(author_code)
    binary_path = output_dir / "report_labels_binary.csv"
    score_path = output_dir / "report_scores_masked.csv"
    for path, records in [(binary_path, binary_rows), (score_path, score_rows)]:
        with path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=[STUDY_ID, *TARGETS])
            writer.writeheader()
            writer.writerows(records)
    manifest = {
        "downloaded_or_verified_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_url": SOURCE_URL,
        "download_url": DOWNLOAD_URL,
        "owner": "pilkwang",
        "dataset_id": 11529209,
        "dataset_version": 1,
        "license": "CC0: Public Domain",
        "source_kind": "existing language-model readings of competition training reports",
        "rows": len(rows),
        "missing_training_studies": missing_training_studies,
        "source_csv_sha256": sha256(raw_bytes),
        "archive_sha256": sha256(archive_bytes),
        "author_labeling_code_sha256": sha256(author_code),
        "outputs": {
            binary_path.name: {
                "sha256": sha256(binary_path.read_bytes()),
                "mapping": {"YES": 1, "NO": 0, "UNK": "blank (unknown, excluded from loss)"},
            },
            score_path.name: {
                "sha256": sha256(score_path.read_bytes()),
                "mapping": "preserve original scores for YES/NO; make UNK blank",
                "limitation": "Author describes scores as useful for ranking, not calibrated probabilities.",
            },
        },
        "verdict_counts": {target: dict(value) for target, value in counts.items()},
        "limitations": [
            "Report-derived labels are imperfect training targets, not expert annotations.",
            "Reports may omit real findings. Unknown findings remain blank.",
            "YES means reported as present; this mapping does not reproduce every clinical severity threshold.",
            "The public author's use of the 58 expert studies while developing prompts cannot be ruled out.",
            "The reserved expert set is protected from our image-model fitting, but is not a proven untouched public-label development set.",
        ],
        "reproduction": "uv run python scripts/fetch_public_labels.py --train-csv data/rsna-knee-abnormality-detection/train.csv",
        "external_model_calls": "none; the bundled labeling code was not executed",
    }
    (output_dir / "source_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return binary_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("data/labels/public/pilkwang"))
    parser.add_argument("--train-csv", type=Path, help="Optionally verify all public-label IDs against train.csv")
    args = parser.parse_args()
    path = fetch_public_labels(args.output_dir, args.train_csv)
    print(f"Prepared public report labels: {path}")
    print(f"Original data and the source/mapping record are saved in {path.parent}")


if __name__ == "__main__":
    main()
