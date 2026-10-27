#!/usr/bin/env python3
"""Information only (Oct 17, 2026): the consensus58 seeds out of fold on the 4,296 cross-validation studies, to show
whether a new seed's four folds are broken and what averaging seeds does, alone and in the team blend (0.5 / 0.3 / 0.2,
ten held-out halves) against seed 42, the seed already scored on the leaderboard. consensus58 = the 4-fold CoAtNet-2
runs on the consensus labels (scripts/prepare_consensus_labels.py) that also train on the 58 expert studies.

Every seed uses its training run's own averaged-model validation file (validation-averaged.csv), so the seeds are
compared like for like (the later seeds' runs were training only and have no separate plain scoring; for seed 42 the
two files differ by at most 0.004). The numbers are flattered, as all out-of-fold scores on the consensus labels are
(their model part comes from models that used other splits): they can show a broken run, not a gain.

Run: CUDA_VISIBLE_DEVICES="" PYTHONPATH=scripts .venv/bin/python scripts/consensus_seed_oof_info.py 42 43 [44]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from compare_team_blend import R7, read, their_frames
from evaluate_fast320 import JUDGES, their_labels_judge
from refit_blend_compare import SEEDS, macro_auc, pct_ranks, scores

from rsna_knee_abnormality_detection.constants import STUDY_ID, TARGETS

THEIRS = ("dino", "eff", "dino3", "conv")


def run_prefix(seed: int) -> str:
    return "runs/coatnet_384_consensus58_fold" if seed == 42 else f"runs/coatnet_384_consensus58s{seed}_fold"


def seed_frame(seed: int) -> pd.DataFrame:
    frame = pd.concat([read(Path(f"{run_prefix(seed)}{k}") / "validation-averaged.csv") for k in range(4)]).sort_index()
    if not frame.index.is_unique:
        raise ValueError(f"seed {seed}: folds overlap")
    return frame


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("seeds", type=int, nargs="+")
    args = parser.parse_args()
    if args.seeds[0] != 42:
        raise ValueError("seed 42 (the board-tested one) comes first")
    seeds = {s: seed_frame(s) for s in args.seeds}
    frames = their_frames()
    r7 = pd.concat([read(p) for p in R7])
    ids = seeds[42].index
    for f in [*seeds.values(), r7, *frames.values()]:
        ids = ids.intersection(f.index)
    ids = ids.sort_values()
    judges = []
    for path in [*JUDGES.values(), their_labels_judge()]:
        y = pd.read_csv(path, dtype={STUDY_ID: str}).set_index(STUDY_ID)[TARGETS].reindex(ids).to_numpy(float)
        known = (y == 0) | (y == 1)
        judges.append((np.where(known, y, 0), known))
    units = {f"seed {s}": f.loc[ids] for s, f in seeds.items()}
    if len(seeds) > 1:
        units["mean of seeds " + "+".join(map(str, seeds))] = sum(f.loc[ids] for f in seeds.values()) / len(seeds)
    alone = {k: [round(macro_auc(v.to_numpy(float), j), 4) for j in judges] for k, v in units.items()}
    theirs = (sum(frames[k].loc[ids] for k in THEIRS) / len(THEIRS)).to_numpy(float)
    rows = {k: [] for k in units}
    for seed in SEEDS:
        order = np.random.default_rng(seed).permutation(len(ids))
        halves = [np.sort(order[: len(ids) // 2]), np.sort(order[len(ids) // 2:])]
        for fit_half in (0, 1):
            t = halves[1 - fit_half]
            jt = [(y[t], k[t]) for y, k in judges]
            for name, frame in units.items():
                parts = [pct_ranks(theirs[t]), pct_ranks(frame.to_numpy(float)[t]), pct_ranks(r7.loc[ids].to_numpy(float)[t])]
                rows[name].append(scores(parts, (0.5, 0.3, 0.2), jt))
    team = {}
    for name in list(units)[1:]:
        diff = np.array(rows[name]) - np.array(rows["seed 42"])
        mean = diff.mean(1)
        team[name] = {"mean_of_judges": round(float(mean.mean()), 5), "positive": f"{int((mean > 0).sum())} of 10",
                      "severity_agreement_their_labels": [round(float(x), 4) for x in diff.mean(0)]}
    result = {"studies": len(ids), "flattered": True, "file_per_seed": "validation-averaged.csv",
              "alone_severity_agreement_their_labels": alone, "team_vs_seed_42_in_our_slot_0.5_0.3_0.2": team}
    out = Path("artifacts/consensus/seed-oof-info.json")
    out.write_text(json.dumps(result, indent=1) + "\n")
    print(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
