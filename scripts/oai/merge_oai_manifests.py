#!/usr/bin/env python3
"""Merge per-visit OAI cache manifests into one list for build_combined_inputs.py.

The first manifest is the base visit (baseline, V00) and is kept whole. With --new-knees-only, a later visit adds only
knees (person and side) that are not in the base, each at its earliest visit that has both an image cache and a label;
otherwise every visit of every knee is kept. Study IDs are OAI_<person>_<SIDE>_<visit>. Paths become absolute.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from rsna_knee_abnormality_detection.constants import STUDY_ID
from rsna_knee_abnormality_detection.utils import file_hash, write_json


def knee(ids: pd.Series) -> pd.Series:
    return ids.str.rsplit("_", n=1).str[0]


def merge(manifests: list[pd.DataFrame], labelled: set[str], new_knees_only: bool) -> pd.DataFrame:
    base, later = manifests[0], pd.concat(manifests[1:], ignore_index=True)
    if new_knees_only:
        later = later[later[STUDY_ID].isin(labelled) & ~knee(later[STUDY_ID]).isin(set(knee(base[STUDY_ID])))]
        later = later.assign(_knee=knee(later[STUDY_ID]), _visit=later[STUDY_ID].str.rsplit("_", n=1).str[1])
        later = later.sort_values(["_knee", "_visit"]).drop_duplicates("_knee").drop(columns=["_knee", "_visit"])
    merged = pd.concat([base, later], ignore_index=True)
    if merged[STUDY_ID].duplicated().any():
        raise ValueError("A knee-visit appears in more than one manifest")
    return merged


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifests", type=Path, nargs="+", required=True, help="Base visit first")
    p.add_argument("--labels", type=Path, required=True, help="OAI labels (used to pick each new knee's first visit)")
    p.add_argument("--new-knees-only", action="store_true")
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    frames = []
    for path in a.manifests:
        m = pd.read_csv(path, dtype={STUDY_ID: str})
        for column in ("cache_path", "metadata_path"):
            m[column] = [str(path.resolve().parent / v) for v in m[column]]
        frames.append(m)
    if any(list(f.columns) != list(frames[0].columns) for f in frames):
        raise ValueError("Manifests have different columns (different preparation versions?)")
    labelled = set(pd.read_csv(a.labels, dtype={STUDY_ID: str}, usecols=[STUDY_ID])[STUDY_ID])
    merged = merge(frames, labelled, a.new_knees_only)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(a.output, index=False)
    visits = merged[STUDY_ID].str.rsplit("_", n=1).str[1].value_counts().sort_index()
    write_json(a.output.with_suffix(".json"), {
        "rule": __doc__.split("\n\n", 1)[1].strip(), "new_knees_only": a.new_knees_only,
        "inputs": {str(path): file_hash(path) for path in [*a.manifests, a.labels]},
        "knee_visits": len(merged), "knees": int(knee(merged[STUDY_ID]).nunique()),
        "per_visit": {k: int(v) for k, v in visits.items()}, "output_sha256": file_hash(a.output)})
    print(f"{len(merged)} knee-visits ({knee(merged[STUDY_ID]).nunique()} knees) -> {a.output}: {visits.to_dict()}")


if __name__ == "__main__":
    main()
