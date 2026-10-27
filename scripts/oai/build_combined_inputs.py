#!/usr/bin/env python3
"""Add OAI knees to the training inputs as extra training-only studies.

Writes, under --output-dir: labels.csv (the base training labels plus OAI rows, label_source "oai"),
validation_labels.csv (the base validation labels plus the same OAI rows, never used because OAI knees never
validate), splits.csv (OAI knees in fold 5, which no run validates on, so every fold trains on them; both knees of a
person share a group) and manifest.csv (the base image list plus the OAI cache, all paths absolute because the
combined list lives in a different folder). Only OAI knees
that have both labels and a prepared image cache are added.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from rsna_knee_abnormality_detection.constants import STUDY_ID, TARGETS
from rsna_knee_abnormality_detection.utils import file_hash, write_json

OAI_FOLD = 5


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--labels", type=Path, required=True)
    p.add_argument("--validation-labels", type=Path, required=True)
    p.add_argument("--splits", type=Path, required=True)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--oai-labels", type=Path, required=True)
    p.add_argument("--oai-manifest", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    args = p.parse_args()
    labels = pd.read_csv(args.labels, dtype={STUDY_ID: str})
    validation = pd.read_csv(args.validation_labels, dtype={STUDY_ID: str})
    splits = pd.read_csv(args.splits, dtype={STUDY_ID: str, "group_id": str})
    manifest = pd.read_csv(args.manifest, dtype={STUDY_ID: str})
    oai = pd.read_csv(args.oai_labels, dtype={STUDY_ID: str, "oai_id": str})
    oai_manifest = pd.read_csv(args.oai_manifest, dtype={STUDY_ID: str})
    oai = oai[oai[STUDY_ID].isin(set(oai_manifest[STUDY_ID]))].copy()
    if oai.empty or set(oai[STUDY_ID]) & set(labels[STUDY_ID]):
        raise ValueError("No OAI knees to add, or OAI IDs clash with base studies")
    rows = oai[[STUDY_ID, *TARGETS]].assign(label_source="oai")
    for column in labels.columns.difference(rows.columns):
        rows[column] = pd.NA
    out_labels = pd.concat([labels, rows[labels.columns]], ignore_index=True)
    val_rows = rows.reindex(columns=validation.columns)
    out_validation = pd.concat([validation, val_rows], ignore_index=True)
    out_splits = pd.concat([splits, pd.DataFrame({STUDY_ID: oai[STUDY_ID], "fold": OAI_FOLD, "split": "train",
                                                  "group_id": "OAI_" + oai["oai_id"]})], ignore_index=True)
    extra = oai_manifest[oai_manifest[STUDY_ID].isin(set(oai[STUDY_ID]))].copy()
    for frame, folder in ((extra, args.oai_manifest.resolve().parent), (manifest, args.manifest.resolve().parent)):
        for column in ("cache_path", "metadata_path"):
            frame[column] = [str(folder / value) for value in frame[column]]
    if set(manifest.columns) != set(extra.columns):
        raise ValueError("Base and OAI manifests have different columns (different preparation versions?)")
    out_manifest = pd.concat([manifest, extra[manifest.columns]], ignore_index=True)
    if fold_clash := set(splits["fold"]) & {OAI_FOLD}:
        raise ValueError(f"Base splits already use fold {fold_clash}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {"labels.csv": out_labels, "validation_labels.csv": out_validation, "splits.csv": out_splits,
               "manifest.csv": out_manifest}
    for name, frame in outputs.items():
        frame.to_csv(args.output_dir / name, index=False)
    write_json(args.output_dir / "provenance.json", {
        "inputs": {str(path): file_hash(path) for path in (args.labels, args.validation_labels, args.splits,
                                                            args.manifest, args.oai_labels, args.oai_manifest)},
        "oai_knee_visits": len(oai), "oai_people": int(oai["oai_id"].nunique()), "oai_fold": OAI_FOLD,
        "outputs": {name: file_hash(args.output_dir / name) for name in outputs}})
    print(f"added {len(oai)} OAI knee-visits ({oai['oai_id'].nunique()} people) -> {args.output_dir}")


if __name__ == "__main__":
    main()
