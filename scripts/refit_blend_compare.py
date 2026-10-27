#!/usr/bin/env python3
"""Compare candidates at their own refitted team-blend weights (Oct 16, 2026; rules fixed before results).

Each configuration names theirs (equal mean of the teammates' chosen sets), ours (equal mean of our chosen five-fold
sets: cur = v7 + OAI, D = our round-6 labels, H = v7 with the hybrid OAI labels) and R7 (heliosli's public model, plain
or with mirrored test-time predictions, r7flip). Three group weights on per-finding percentile ranks are searched on a
0.05 grid. In each of 10 held-out evaluations (5 random halves x both directions, ranks computed inside each half) every
configuration gets the grid point with the best mean of the three judges (severity-aware report labels, report + GPT
agreement labels, the teammates' labels cut at 0.5) on the fitting half, scored on the other half; a candidate is
compared with the reference on the same half.

Run: CUDA_VISIBLE_DEVICES="" PYTHONPATH=scripts .venv/bin/python scripts/refit_blend_compare.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from compare_team_blend import R7, R7_EXPERT, read, their_frames
from evaluate_fast320 import BASE384, EXPERT_LABELS, FIVE_FOLD, JUDGES, pooled, their_labels_judge
from scipy.stats import rankdata

from rsna_knee_abnormality_detection.constants import STUDY_ID, TARGETS

R7_FLIP = [Path(f"artifacts/s6r7-results/r7flip_oof_f{k}/predictions.csv") for k in range(4)]
R7_FLIP_EXPERT = Path("artifacts/s6r7-results/r7flip_gold/predictions.csv")
SUBMITTED = ("eff", "dino", "dino3")
OURS_SETS = {"cur": BASE384, "D": FIVE_FOLD["D"], "H": FIVE_FOLD["H"]}
REFERENCE = {"theirs": SUBMITTED, "ours": ("cur",), "r7": "r7"}
CANDIDATES = {
    "ours_D": {"ours": ("D",)}, "ours_H": {"ours": ("H",)}, "ours_cur_D": {"ours": ("cur", "D")},
    "ours_cur_H": {"ours": ("cur", "H")}, "ours_D_H": {"ours": ("D", "H")}, "ours_cur_D_H": {"ours": ("cur", "D", "H")},
    "r7_flip": {"r7": "r7flip"},
    "theirs_conv": {"theirs": (*SUBMITTED, "conv")}, "theirs_conv_new": {"theirs": (*SUBMITTED, "conv_new")},
    "theirs_dino_new": {"theirs": (*SUBMITTED, "dino_new")},
    "theirs_conv__ours_cur_D_H": {"theirs": (*SUBMITTED, "conv"), "ours": ("cur", "D", "H")},
    "theirs_conv__ours_D_H": {"theirs": (*SUBMITTED, "conv"), "ours": ("D", "H")},
}
GRID = [(t / 20, o / 20, (20 - t - o) / 20) for t in range(21) for o in range(21 - t)]
SEEDS = (0, 1, 2, 3, 4)


def macro_auc(scores: np.ndarray, judge: tuple[np.ndarray, np.ndarray]) -> float:
    """Mean over findings of the tie-aware AUC on cells labelled exactly 0 or 1 (judge = labels, known mask)."""
    y, known = judge
    values = []
    for j in range(y.shape[1]):
        k = known[:, j]
        yy, ss = y[k, j], scores[k, j]
        pos = int(yy.sum())
        neg = len(yy) - pos
        if pos == 0 or neg == 0:
            continue
        r = rankdata(ss)
        values.append((r[yy == 1].sum() - pos * (pos + 1) / 2) / (pos * neg))
    return float(np.mean(values))


def pct_ranks(values: np.ndarray) -> np.ndarray:
    return np.column_stack([rankdata(values[:, j]) / len(values) for j in range(values.shape[1])])


def config(base: dict, change: dict) -> dict:
    return {**base, **change}


def parts(cfg: dict, sources: dict, idx: np.ndarray) -> list[np.ndarray]:
    theirs = sum(sources["theirs"][k][idx] for k in cfg["theirs"]) / len(cfg["theirs"])
    ours = sum(sources["ours"][k][idx] for k in cfg["ours"]) / len(cfg["ours"])
    return [pct_ranks(theirs), pct_ranks(ours), pct_ranks(sources["r7"][cfg["r7"]][idx])]


def scores(ranked: list[np.ndarray], w: tuple, judges: list) -> list[float]:
    blend = w[0] * ranked[0] + w[1] * ranked[1] + w[2] * ranked[2]
    return [macro_auc(blend, j) for j in judges]


def best(ranked: list[np.ndarray], judges: list) -> tuple:
    results = [(np.mean(scores(ranked, w, judges)), w) for w in GRID]
    return max(results, key=lambda x: x[0])[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("artifacts/refit-blend/result.json"))
    args = parser.parse_args()
    frames = their_frames()
    ours = {k: pooled(p) for k, p in OURS_SETS.items()}
    r7 = {"r7": pd.concat([read(p) for p in R7]), "r7flip": pd.concat([read(p) for p in R7_FLIP])}
    judge_paths = {**JUDGES, "theirs_labels": their_labels_judge()}
    labels = {n: pd.read_csv(p, dtype={STUDY_ID: str}).set_index(STUDY_ID)[TARGETS] for n, p in judge_paths.items()}
    ids = ours["cur"].index
    for f in [*frames.values(), *ours.values(), *r7.values()]:
        ids = ids.intersection(f.index)
    ids = ids.sort_values()
    sources = {"theirs": {k: f.loc[ids].to_numpy(float) for k, f in frames.items()},
               "ours": {k: f.loc[ids].to_numpy(float) for k, f in ours.items()},
               "r7": {k: f.loc[ids].to_numpy(float) for k, f in r7.items()}}
    full_judges = []
    for name in judge_paths:
        y = labels[name].reindex(ids).to_numpy(float)
        known = (y == 0) | (y == 1)
        full_judges.append((np.where(known, y, 0), known))
    configs = {"reference": REFERENCE, **{k: config(REFERENCE, v) for k, v in CANDIDATES.items()}}
    names = list(judge_paths)

    def judges_at(idx):
        return [(y[idx], k[idx]) for y, k in full_judges]

    evaluations = []
    for seed in SEEDS:
        order = np.random.default_rng(seed).permutation(len(ids))
        halves = [np.sort(order[: len(ids) // 2]), np.sort(order[len(ids) // 2:])]
        for fit_half in (0, 1):
            fit_idx, test_idx = halves[fit_half], halves[1 - fit_half]
            jf, jt = judges_at(fit_idx), judges_at(test_idx)
            row = {"seed": seed, "fit_half": fit_half, "configs": {}}
            for name, cfg in configs.items():
                w = best(parts(cfg, sources, fit_idx), jf)
                s = scores(parts(cfg, sources, test_idx), w, jt)
                row["configs"][name] = {"weights": w, "held_out": dict(zip(names, s)), "mean": float(np.mean(s))}
            evaluations.append(row)
            print(seed, fit_half, {k: v["weights"] for k, v in row["configs"].items() if k in ("reference", "ours_cur_D_H")}, flush=True)
    every = np.arange(len(ids))
    full = {}
    for name, cfg in configs.items():
        w = best(parts(cfg, sources, every), full_judges)
        full[name] = {"weights_fitted_on_all": w,
                      "at_fitted": dict(zip(names, scores(parts(cfg, sources, every), w, full_judges))),
                      "at_fixed_50_30_20": dict(zip(names, scores(parts(cfg, sources, every), (0.5, 0.3, 0.2), full_judges)))}
    # The 58 expert studies at the weights fitted on all studies (information only).
    expert = pd.read_csv(EXPERT_LABELS, dtype={STUDY_ID: str}).set_index(STUDY_ID)[TARGETS]
    eids = expert.index.sort_values()
    ey = expert.loc[eids].to_numpy(float)
    eknown = (ey == 0) | (ey == 1)
    ejudge = [(np.where(eknown, ey, 0), eknown)]
    e_ours = {k: sum(read(Path(f"{p}{f}") / "expert58-plain.csv").loc[eids] for f in range(5)).to_numpy(float) / 5
              for k, p in OURS_SETS.items()}
    e_sources = {"theirs": {k: f.loc[eids].to_numpy(float) for k, f in frames.items()}, "ours": e_ours,
                 "r7": {"r7": read(R7_EXPERT).loc[eids].to_numpy(float), "r7flip": read(R7_FLIP_EXPERT).loc[eids].to_numpy(float)}}
    allidx = np.arange(len(eids))
    for name, cfg in configs.items():
        full[name]["experts_at_fitted"] = round(scores(parts(cfg, e_sources, allidx), full[name]["weights_fitted_on_all"], ejudge)[0], 4)
        full[name]["experts_at_fixed"] = round(scores(parts(cfg, e_sources, allidx), (0.5, 0.3, 0.2), ejudge)[0], 4)
    summary = {}
    for name in CANDIDATES:
        diffs = np.array([e["configs"][name]["mean"] - e["configs"]["reference"]["mean"] for e in evaluations])
        per_judge = {j: float(np.mean([e["configs"][name]["held_out"][j] - e["configs"]["reference"]["held_out"][j]
                                       for e in evaluations])) for j in names}
        fixed = {j: round(full[name]["at_fixed_50_30_20"][j] - full["reference"]["at_fixed_50_30_20"][j], 4) for j in names}
        improves = bool(diffs.mean() > 0 and (diffs > 0).sum() >= 8 and min(per_judge.values()) >= -0.0005)
        summary[name] = {"held_out_mean_of_judges": round(float(diffs.mean()), 5), "positive": f"{int((diffs > 0).sum())} of 10",
                         "held_out_per_judge": {j: round(v, 5) for j, v in per_judge.items()}, "improves": improves,
                         "fixed_50_30_20_per_judge": fixed, "weights_fitted_on_all": full[name]["weights_fitted_on_all"],
                         "experts_at_fitted_vs_reference": round(full[name]["experts_at_fitted"] - full["reference"]["experts_at_fitted"], 4)}
    result = {"studies": len(ids), "grid_points": len(GRID), "reference": full["reference"], "summary": summary,
              "full": full, "evaluations": evaluations}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=1) + "\n")
    print(json.dumps({"studies": len(ids), "reference": full["reference"], "summary": summary}, indent=1))


if __name__ == "__main__":
    main()
