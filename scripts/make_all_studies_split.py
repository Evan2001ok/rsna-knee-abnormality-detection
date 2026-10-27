#!/usr/bin/env python3
"""Splits for training on (almost) all studies: a small placeholder validation set, the rest trains.

Checkpoint averaging needs a validation set on which all twelve findings have positives and negatives (exact 0/1
selection labels). This picks whole patient groups (here each study is its own group, since we had no patient mapping) from the
cross-validation studies, at random but making sure each
finding has at least --minimum positives and negatives, until about --studies studies are reached, and moves them to a
new fold (--fold). Every other cross-validation study, and any training-only rows (such as OAI knees in fold 5), keep a
fold other than --fold and so all train. Runs using it set data.fold to --fold and validate only from epoch 22 with
average_top_k 3, so the averaged epochs are 22-24 whatever the scores: in base training this set never chooses
anything. The all-window fine-tunes validate from epoch 1 and use it to pick their best 3 of 4 epochs.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from rsna_knee_abnormality_detection.constants import STUDY_ID, TARGETS
from rsna_knee_abnormality_detection.utils import file_hash, write_json


def pick_groups(splits: pd.DataFrame, labels: pd.DataFrame, studies: int, minimum: int, seed: int) -> list[str]:
    """Whole groups from folds 0-4 so that every finding has `minimum` exact positives and negatives."""
    pool = splits[(splits["split"] == "train") & splits["fold"].between(0, 4)]
    known = labels.reindex(pool[STUDY_ID].to_numpy())[TARGETS]
    known.index = pool["group_id"].to_numpy()
    positives = (known == 1).groupby(level=0).sum()
    negatives = (known == 0).groupby(level=0).sum()
    sizes = pool.groupby("group_id").size()
    order = list(np.random.default_rng(seed).permutation(sorted(sizes.index)))
    chosen: list[str] = []
    have_pos, have_neg = pd.Series(0, index=TARGETS), pd.Series(0, index=TARGETS)
    for name in sorted(TARGETS, key=lambda t: positives[t].sum()):  # rarest finding first
        for group in order:
            if have_pos[name] >= minimum:
                break
            if group not in chosen and positives.loc[group, name] > 0:
                chosen.append(group)
                have_pos += positives.loc[group]
                have_neg += negatives.loc[group]
    for group in order:
        if sizes[chosen].sum() >= studies and (have_neg >= minimum).all():
            break
        if group not in chosen:
            chosen.append(group)
            have_pos += positives.loc[group]
            have_neg += negatives.loc[group]
    if (have_pos < minimum).any() or (have_neg < minimum).any():
        raise ValueError("Could not cover every finding with the requested minimum")
    return chosen


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--splits", type=Path, required=True)
    parser.add_argument("--validation-labels", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fold", type=int, default=6)
    parser.add_argument("--studies", type=int, default=100)
    parser.add_argument("--minimum", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    splits = pd.read_csv(args.splits, dtype={STUDY_ID: str, "group_id": str})
    labels = pd.read_csv(args.validation_labels, dtype={STUDY_ID: str}).set_index(STUDY_ID)
    if (splits["fold"] == args.fold).any():
        raise ValueError(f"Fold {args.fold} is already in use")
    groups = pick_groups(splits, labels, args.studies, args.minimum, args.seed)
    out = splits.copy()
    out.loc[out["group_id"].isin(groups) & (out["split"] == "train"), "fold"] = args.fold
    moved = out[out["fold"] == args.fold]
    out.to_csv(args.output, index=False)
    known = labels.reindex(moved[STUDY_ID])[TARGETS]
    write_json(args.output.with_suffix(".json"), {
        "rule": __doc__.split("\n\n", 1)[1].strip(), "inputs": {str(args.splits): file_hash(args.splits),
        str(args.validation_labels): file_hash(args.validation_labels)}, "fold": args.fold, "seed": args.seed,
        "placeholder_studies": len(moved), "groups": len(groups),
        "training_rows": int(((out["split"] == "train") & (out["fold"] >= 0) & (out["fold"] != args.fold)).sum()),
        "positives": {t: int((known[t] == 1).sum()) for t in TARGETS},
        "negatives": {t: int((known[t] == 0).sum()) for t in TARGETS}, "output_sha256": file_hash(args.output)})
    print(f"{len(moved)} placeholder validation studies in fold {args.fold}; wrote {args.output}")


if __name__ == "__main__":
    main()
