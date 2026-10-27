#!/usr/bin/env python3
"""Prepare a Raptor image cache (the layout adapted from dreaddevelopment's public Raptor notebooks;
64-slice scored layout, or the dense variant at 336, 384 or 448 px, or knee-centred) from the DICOM files."""

import argparse
from pathlib import Path

import pandas as pd

from rsna_knee_abnormality_detection.constants import STUDY_ID
from rsna_knee_abnormality_detection.raptor_data import (
    RAPTOR_DENSE_160MM_384_VERSION,
    RAPTOR_DENSE_200MM_384_VERSION,
    RAPTOR_DENSE_384_VERSION,
    RAPTOR_DENSE_448_VERSION,
    RAPTOR_DENSE_KNEE_384_VERSION,
    RAPTOR_DENSE_KNEE_CLOSEUP_384_VERSION,
    RAPTOR_DENSE_VERSION,
    RAPTOR_PREPROCESS_VERSION,
    RAPTOR_VARIANTS,
    prepare_raptor_data,
    validate_raptor_manifest,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data/rsna-knee-abnormality-detection"))
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--split", choices=["train", "test"], default="train")
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--studies", type=Path, help="CSV containing the StudyInstanceUIDs to prepare")
    parser.add_argument("--limit", type=int, help="Prepare only the first N selected study IDs")
    parser.add_argument("--verify-checksums", action="store_true", help="Also hash all prepared image arrays during validation")
    parser.add_argument("--variant", choices=sorted(RAPTOR_VARIANTS), default=RAPTOR_PREPROCESS_VERSION,
                        help="Cache layout; the dense variant keeps every slice of the span, up to 32 per series")
    parser.add_argument("--anchors-dir", type=Path,
                        help="Knee-centred variants: folder of <StudyInstanceUID>.json anchor records (joint_lps)")
    args = parser.parse_args()
    studies = None
    if args.studies:
        studies = pd.read_csv(args.studies, usecols=[STUDY_ID], dtype={STUDY_ID: str})[STUDY_ID].tolist()
    default_root = {RAPTOR_DENSE_VERSION: "data/processed/raptor-dense",
                    RAPTOR_DENSE_384_VERSION: "data/processed/raptor-dense-384",
                    RAPTOR_DENSE_448_VERSION: "data/processed/raptor-dense-448",
                    RAPTOR_DENSE_160MM_384_VERSION: "data/processed/raptor-dense-160mm-384",
                    RAPTOR_DENSE_200MM_384_VERSION: "data/processed/raptor-dense-200mm-384",
                    RAPTOR_DENSE_KNEE_384_VERSION: "data/processed/raptor-dense-knee-384",
                    RAPTOR_DENSE_KNEE_CLOSEUP_384_VERSION: "data/processed/raptor-dense-knee-closeup-384"}.get(args.variant, "data/processed/raptor")
    output = args.output_dir or Path(default_root) / args.split
    manifest = prepare_raptor_data(args.data_root, output, args.split, args.workers, studies, args.limit, args.variant,
                                   args.anchors_dir)
    result = validate_raptor_manifest(manifest, verify_checksums=args.verify_checksums, version=args.variant)
    print(f"Manifest: {manifest}", flush=True)
    print(f"Validated: {result['studies']} studies, {result['valid_windows']} valid windows, "
          f"{result['cache_bytes'] / 1e9:.2f} GB", flush=True)


if __name__ == "__main__":
    main()
