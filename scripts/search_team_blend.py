#!/usr/bin/env python3
"""Search the team blend over all the team's models already scored on the leaderboard (Oct 17, 2026; rules fixed before
results).

A configuration is a subset of their sets (equal probability mean, their rule), one of our consensus options (equal mean
of the chosen consensus58 seeds), R7 in or out, and three group weights on per-finding percentile ranks. Configurations
over the time limit (run time x 1.25 above 7.7 h; Kaggle allows 9 h) are dropped first. Weights are searched on a 0.1
grid, then on a 0.05 grid around the best three (configuration, weights) pairs; the score is the mean of the three
judges' macro AUC (severity-aware report labels, report + GPT agreement labels, the teammates' labels cut at 0.5).

Nested estimate: in each of 10 held-out evaluations (5 random halves x both directions, ranks computed inside each half)
the whole search runs on the fitting half and its winner is scored on the other half, next to the consensus package
dhcons (their dino, eff, dino3 and conv; our consensus seed 42; R7) at refitted weights, and dh10-conv as submitted (an
earlier team package: the same four sets, ours = the D and H folds at 56 windows, 0.5 / 0.3 / 0.2). The winner on all
studies is the candidate to submit.

Run: CUDA_VISIBLE_DEVICES="" PYTHONPATH=scripts nice .venv/bin/python scripts/search_team_blend.py --newconv-hours H
"""

from __future__ import annotations

import argparse
import itertools
import json
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from compare_team_blend import R7, THEIR_SETS, THEIRS, read
from consensus_seed_oof_info import seed_frame
from evaluate_fast320 import FIVE_FOLD, JUDGES, pooled, their_labels_judge
from refit_blend_compare import GRID, SEEDS, macro_auc, pct_ranks

from rsna_knee_abnormality_detection.constants import STUDY_ID, TARGETS

SETS = ("dino", "eff", "dino3", "conv", "tiny", "rad", "newconv")
NEWCONV_OOF = Path("artifacts/teammates/newconv-20261016/oof")
S6 = [Path(f"artifacts/s6r7-results/s6_oof_f{k}/predictions.csv") for k in range(4)]
# Raw hours at 1,320 studies: measured in Kaggle timing runs of the dh10-conv and dhcons packages, except the
# estimated ones, which need a Kaggle timing run before a configuration using them is submitted.
FIXED_HOURS = {"setup": 0.15, "theirs_cache": 0.40, "r7": 2.32}
SET_HOURS = {"dino": 0.18, "eff": 0.32, "dino3": 0.18, "conv": 0.34, "tiny": 0.19, "rad": 0.06}
OURS = {"s42": ((42,), 1.34), "s42+43": ((42, 43), 1.9), "s42+43+44": ((42, 43, 44), 2.8)}
ESTIMATED = {"tiny", "rad", "s42+43", "s42+43+44"}
LIMIT_RAW = 7.7 / 1.25
DHCONS = (("dino", "eff", "dino3", "conv"), "s42", True)
GRID10 = [(t / 10, o / 10, (10 - t - o) / 10) for t in range(11) for o in range(11 - t)]
GRID10_NO_R7 = [(t / 10, (10 - t) / 10, 0.0) for t in range(11)]


def hours(cfg: tuple, set_hours: dict, ours: dict = OURS) -> float:
    sets, ours_key, r7 = cfg
    h = FIXED_HOURS["setup"] + ours[ours_key][1] + (FIXED_HOURS["r7"] if r7 else 0.0)
    if sets:
        h += FIXED_HOURS["theirs_cache"] + sum(set_hours[k] for k in sets)
    return h


def configs(set_hours: dict, ours: dict = OURS, sets: tuple = SETS, limit: float = LIMIT_RAW) -> list[tuple]:
    out = []
    for n in range(2, len(sets) + 1):
        for subset in itertools.combinations(sets, n):
            for ours_key in ours:
                for r7 in (True, False):
                    cfg = (subset, ours_key, r7)
                    if hours(cfg, set_hours, ours) <= limit + 1e-9:
                        out.append(cfg)
    return out


def their_oof(path: Path) -> pd.DataFrame:
    """Their out-of-fold file without the expert (gold) rows, which are never read."""
    frame = pd.read_csv(path, dtype={STUDY_ID: str})
    frame = frame[frame["is_gold"] == 0]
    if frame[STUDY_ID].duplicated().any():
        raise ValueError(f"duplicate studies in {path}")
    values = frame[TARGETS].to_numpy(float)
    if not np.isfinite(values).all() or (values < 0).any() or (values > 1).any():
        raise ValueError(f"predictions must be probabilities: {path}")
    return frame.set_index(STUDY_ID)[TARGETS].astype(float)


def load() -> tuple[pd.Index, dict, list]:
    theirs = {}
    for key in SETS[:-1]:
        found = sorted((THEIRS / THEIR_SETS[key]).glob("*/oof_predictions.csv"))
        if len(found) != 1:
            raise ValueError(f"{key}: expected one oof_predictions.csv, found {found}")
        theirs[key] = their_oof(found[0])
    theirs["newconv"] = pd.concat([their_oof(NEWCONV_OOF / p / "oof_predictions.csv") for p in ("fold12", "fold23")])
    if not theirs["newconv"].index.is_unique:
        raise ValueError("newconv folds overlap")
    seeds = {s: seed_frame(s) for s in (42, 43, 44)}
    extra = {"D": pooled(FIVE_FOLD["D"], "validation-averaged-model-w56.csv"),
             "H": pooled(FIVE_FOLD["H"], "validation-averaged-model-w56.csv"),
             "r7": pd.concat([read(p) for p in R7]), "s6": pd.concat([read(p) for p in S6])}
    ids = seeds[42].index
    for f in [*theirs.values(), *seeds.values(), *extra.values()]:
        ids = ids.intersection(f.index)
    ids = ids.sort_values()
    if len(ids) != 4296:
        raise ValueError(f"expected the 4,296 cross-validation studies, found {len(ids)}")
    arrays = {"theirs": {k: f.loc[ids].to_numpy(float) for k, f in theirs.items()},
              "ours": {k: sum(seeds[s].loc[ids] for s in v[0]).to_numpy(float) / len(v[0]) for k, v in OURS.items()},
              **{k: f.loc[ids].to_numpy(float) for k, f in extra.items()}}
    judges = []
    for path in [*JUDGES.values(), their_labels_judge()]:
        y = pd.read_csv(path, dtype={STUDY_ID: str}).set_index(STUDY_ID)[TARGETS].reindex(ids).to_numpy(float)
        known = (y == 0) | (y == 1)
        judges.append((np.where(known, y, 0), known))
    return ids, arrays, judges


class Half:
    """Ranked parts on one set of studies, computed once per subset or option."""

    def __init__(self, arrays: dict, judges: list, idx: np.ndarray) -> None:
        self.arrays, self.idx = arrays, idx
        self.judges = [(y[idx], k[idx]) for y, k in judges]
        self.cache = {}

    def theirs(self, subset: tuple) -> np.ndarray:
        key = ("t", subset)
        if key not in self.cache:
            self.cache[key] = pct_ranks(sum(self.arrays["theirs"][k][self.idx] for k in subset) / len(subset))
        return self.cache[key]

    def part(self, name: str, ours_key: str | None = None) -> np.ndarray:
        key = (name, ours_key)
        if key not in self.cache:
            values = self.arrays["ours"][ours_key] if name == "ours" else self.arrays[name]
            self.cache[key] = pct_ranks(values[self.idx])
        return self.cache[key]

    def scores(self, cfg: tuple, w: tuple, ours: np.ndarray | None = None, r7: np.ndarray | None = None) -> list[float]:
        sets, ours_key, use_r7 = cfg
        o = self.part("ours", ours_key) if ours is None else ours
        blend = w[0] * self.theirs(sets) + w[1] * o
        if use_r7:
            blend = blend + w[2] * (self.part("r7") if r7 is None else r7)
        return [macro_auc(blend, j) for j in self.judges]


def neighbours(w: tuple, use_r7: bool) -> list[tuple]:
    if not use_r7:
        return [(round(t, 2), round(1 - t, 2), 0.0) for t in np.arange(w[0] - 0.05, w[0] + 0.051, 0.05) if -1e-9 <= t <= 1 + 1e-9]
    out = []
    for t in np.arange(w[0] - 0.05, w[0] + 0.051, 0.05):
        for o in np.arange(w[1] - 0.05, w[1] + 0.051, 0.05):
            r = 1 - t - o
            if t > -1e-9 and o > -1e-9 and r > -1e-9:
                out.append((round(t, 2), round(o, 2), round(r, 2)))
    return out


def search(half: Half, cfgs: list[tuple], top: int = 3) -> dict:
    """Best configuration and weights on this half: 0.1 grid for all, 0.05 around the best `top` pairs.
    Returns the best per configuration (for the table) and the overall winner."""
    per_cfg = {}
    for cfg in cfgs:
        for w in (GRID10 if cfg[2] else GRID10_NO_R7):
            s = float(np.mean(half.scores(cfg, w)))
            if cfg not in per_cfg or s > per_cfg[cfg][0]:
                per_cfg[cfg] = (s, w)
    for cfg, (_, w) in sorted(per_cfg.items(), key=lambda kv: -kv[1][0])[:top]:
        for v in neighbours(w, cfg[2]):
            s = float(np.mean(half.scores(cfg, v)))
            if s > per_cfg[cfg][0]:
                per_cfg[cfg] = (s, v)
    winner = max(per_cfg, key=lambda c: per_cfg[c][0])
    return {"winner": winner, "weights": per_cfg[winner][1], "score": per_cfg[winner][0], "per_cfg": per_cfg}


def refit(half: Half, cfg: tuple) -> tuple:
    grid = GRID if cfg[2] else [(t / 20, 1 - t / 20, 0.0) for t in range(21)]
    return max(grid, key=lambda w: np.mean(half.scores(cfg, w)))


def nested_task(task: tuple) -> dict:
    arrays, judges, cfgs, seed, fit_half = task
    n = len(judges[0][0])
    order = np.random.default_rng(seed).permutation(n)
    halves = [np.sort(order[: n // 2]), np.sort(order[n // 2:])]
    fit, test = Half(arrays, judges, halves[fit_half]), Half(arrays, judges, halves[1 - fit_half])
    full = search(fit, cfgs)
    only42 = search(fit, [c for c in cfgs if c[1] == "s42"])
    ref_w = refit(fit, DHCONS)
    dh10_ours = pct_ranks(((arrays["D"] + arrays["H"]) / 2)[test.idx])
    dh10 = test.scores((("dino", "eff", "dino3", "conv"), "s42", True), (0.5, 0.3, 0.2), ours=dh10_ours)
    return {"seed": seed, "fit_half": fit_half,
            "winner": [list(full["winner"][0]), full["winner"][1], full["winner"][2]], "weights": full["weights"],
            "winner_seed42_only": [list(only42["winner"][0]), only42["winner"][1], only42["winner"][2]],
            "test": {"winner": test.scores(full["winner"], full["weights"]),
                     "winner_seed42_only": test.scores(only42["winner"], only42["weights"]),
                     "dhcons_refitted": test.scores(DHCONS, ref_w), "dhcons_as_built": test.scores(DHCONS, (0.5, 0.3, 0.2)),
                     "dh10_conv_as_submitted": dh10}}


def compare(rows: list[dict], a: str, b: str) -> dict:
    d = np.array([r["test"][a] for r in rows]) - np.array([r["test"][b] for r in rows])
    mean = d.mean(1)
    return {"mean_of_judges": round(float(mean.mean()), 5), "positive": f"{int((mean > 0).sum())} of {len(rows)}",
            "per_judge": [round(float(x), 5) for x in d.mean(0)]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--newconv-hours", type=float, required=True, help="raw hours at 1,320 studies (timing run)")
    parser.add_argument("--out", type=Path, default=Path("artifacts/team-blend-search/result.json"))
    args = parser.parse_args()
    set_hours = {**SET_HOURS, "newconv": args.newconv_hours}
    ids, arrays, judges = load()
    cfgs = configs(set_hours)
    print(len(ids), "studies;", len(cfgs), "configurations within the time budget", flush=True)
    tasks = [(arrays, judges, cfgs, seed, fit_half) for seed in SEEDS for fit_half in (0, 1)]
    with Pool(10) as pool:
        rows = pool.map(nested_task, tasks)
    allh = Half(arrays, judges, np.arange(len(ids)))
    full = search(allh, cfgs)
    only42 = search(allh, [c for c in cfgs if c[1] == "s42"])

    def describe(cfg: tuple, w: tuple, s: float | None = None) -> dict:
        per = allh.scores(cfg, w)
        return {"theirs": list(cfg[0]), "ours": cfg[1], "r7": cfg[2], "weights": list(w),
                "raw_hours": round(hours(cfg, set_hours), 2), "hours_x1_25": round(1.25 * hours(cfg, set_hours), 2),
                "needs_timing": sorted({*cfg[0], cfg[1]} & ESTIMATED), "mean_of_judges": round(float(np.mean(per)), 5),
                "per_judge": [round(x, 5) for x in per]}

    table = sorted(full["per_cfg"].items(), key=lambda kv: -kv[1][0])
    ref_w = refit(allh, DHCONS)
    winner = describe(full["winner"], full["weights"])
    info = {}
    w = full["weights"]
    if full["winner"][2]:
        mix = pct_ranks(((arrays["r7"] + arrays["s6"]) / 2))
        info["winner_with_r7_and_s6"] = round(float(np.mean(allh.scores(full["winner"], w, r7=mix))) - winner["mean_of_judges"], 5)
    dh = pct_ranks((arrays["ours"][full["winner"][1]] + (arrays["D"] + arrays["H"]) / 2) / 2)
    info["winner_with_ours_plus_D_and_H"] = round(float(np.mean(allh.scores(full["winner"], w, ours=dh))) - winner["mean_of_judges"], 5)
    result = {
        "studies": len(ids), "configurations": len(cfgs), "set_hours": set_hours, "limit_raw_hours": round(LIMIT_RAW, 3),
        "winner_all_studies": winner, "winner_seed42_only_all_studies": describe(only42["winner"], only42["weights"]),
        "dhcons_refitted_all_studies": describe(DHCONS, ref_w),
        "top10_all_studies": [describe(c, v[1]) for c, v in table[:10]],
        "nested": {"winner_vs_dhcons_refitted": compare(rows, "winner", "dhcons_refitted"),
                   "winner_vs_dhcons_as_built_0.5_0.3_0.2": compare(rows, "winner", "dhcons_as_built"),
                   "seed42_only_winner_vs_dhcons_refitted": compare(rows, "winner_seed42_only", "dhcons_refitted"),
                   "winner_vs_seed42_only_winner": compare(rows, "winner", "winner_seed42_only"),
                   "winner_vs_dh10_conv_as_submitted": compare(rows, "winner", "dh10_conv_as_submitted"),
                   "halves": [{k: r[k] for k in ("seed", "fit_half", "winner", "weights", "winner_seed42_only")} for r in rows]},
        "information_only": info}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=1) + "\n")
    print(json.dumps({k: result[k] for k in ("winner_all_studies", "winner_seed42_only_all_studies", "dhcons_refitted_all_studies",
                                             "nested", "information_only")}, indent=1))


if __name__ == "__main__":
    main()
