#!/usr/bin/env python3
"""Per-model ensemble weights for the merged team, checked on held-out halves (Oct 15, 2026; rule fixed before
results).

Members: the teammates' eight sets, our v7 + OAI folds and R7 (heliosli's public model), all out-of-fold on the 4,296
cross-validation studies. Weights are fitted by greedy forward selection with replacement on one half of the studies
(objective: mean macro AUC of three judges: severity-aware report labels, report + GPT agreement labels, the teammates'
labels cut at 0.5) and scored on the other half against the fixed 50/30/20 blend (theirs / ours / R7). Ranks are
computed inside each half, as Kaggle ranks inside the test set.

Run: .venv/bin/python scripts/fit_team_member_weights.py [--out artifacts/team-blend/member-weights.json]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from compare_team_blend import (
    EXPERT_LABELS,
    GROUP_A,
    GROUP_B,
    JUDGES,
    OURS,
    OURS_EXPERT,
    R7,
    R7_EXPERT,
    SPLITS,
    THEIR_SETS,
    THEIRS,
    read,
    their_frames,
    theirs_ensemble,
)

from rsna_knee_abnormality_detection.constants import STUDY_ID, TARGETS

STEPS = 40
SEEDS = (0, 1, 2, 3, 4)
FIXED = {"theirs": 0.5, "ours": 0.3, "r7": 0.2}


def auc_columns(y: np.ndarray, s: np.ndarray) -> float:
    """Mean over findings of the tie-aware AUC on cells labelled exactly 0 or 1."""
    values = []
    for j in range(y.shape[1]):
        known = (y[:, j] == 0) | (y[:, j] == 1)
        yy, ss = y[known, j], s[known, j]
        pos, neg = (yy == 1).sum(), (yy == 0).sum()
        if pos == 0 or neg == 0:
            continue
        r = pd.Series(ss).rank(method="average").to_numpy()
        values.append((r[yy == 1].sum() - pos * (pos + 1) / 2) / (pos * neg))
    return float(np.mean(values))


def score(judges: list[np.ndarray], s: np.ndarray) -> list[float]:
    return [auc_columns(y, s) for y in judges]


def ranks(frame: pd.DataFrame) -> np.ndarray:
    return frame.rank(method="average", pct=True).to_numpy()


def greedy(members: dict[str, np.ndarray], judges: list[np.ndarray]) -> dict[str, float]:
    names, counts, total = list(members), {n: 0 for n in members}, None
    for step in range(1, STEPS + 1):
        best, best_value = None, -1.0
        for n in names:
            trial = members[n] if total is None else total + members[n]
            value = float(np.mean(score(judges, trial / step)))
            if value > best_value:
                best, best_value = n, value
        counts[best] += 1
        total = members[best] if total is None else total + members[best]
    return {n: c / STEPS for n, c in counts.items() if c}


def combine(members: dict[str, np.ndarray], weights: dict[str, float]) -> np.ndarray:
    return sum(w * members[n] for n, w in weights.items()) / sum(weights.values())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("artifacts/team-blend/member-weights.json"))
    args = parser.parse_args()
    splits = pd.read_csv(SPLITS, dtype={STUDY_ID: str}).set_index(STUDY_ID)["fold"]
    frames = their_frames()
    ours, r7 = read(OURS), pd.concat([read(p) for p in R7])
    ids = splits.index[splits >= 0].intersection(ours.index).intersection(r7.index)
    for f in frames.values():
        ids = ids.intersection(f.index)
    ids = ids.sort_values()
    found = sorted((THEIRS / THEIR_SETS["conv"]).glob("*/labels_used_vlm07_llm03.csv"))
    their_labels = (pd.read_csv(found[0], dtype={STUDY_ID: str}).set_index(STUDY_ID)[TARGETS] >= 0.5).astype(float)
    judge_frames = [pd.read_csv(p, dtype={STUDY_ID: str}).set_index(STUDY_ID)[TARGETS] for p in JUDGES.values()]
    judge_frames.append(their_labels)

    def parts(sel):
        per = {k: frames[k].loc[sel] for k in THEIR_SETS}
        per["ours"], per["r7"] = ours.loc[sel], r7.loc[sel]
        group = {"theirs": theirs_ensemble(frames, sel), "ours": ours.loc[sel], "r7": r7.loc[sel]}
        return ({k: ranks(v) for k, v in per.items()}, {k: ranks(v) for k, v in group.items()},
                [j.loc[sel].to_numpy(float) for j in judge_frames])

    evaluations = []
    for seed in SEEDS:
        order = np.random.default_rng(seed).permutation(len(ids))
        halves = [ids[np.sort(order[: len(ids) // 2])], ids[np.sort(order[len(ids) // 2:])]]
        for fit_half, test_half in ((0, 1), (1, 0)):
            per_fit, group_fit, judges_fit = parts(halves[fit_half])
            per_test, group_test, judges_test = parts(halves[test_half])
            w_per, w_group = greedy(per_fit, judges_fit), greedy(group_fit, judges_fit)
            fixed = score(judges_test, combine(group_test, FIXED))
            evaluations.append({
                "seed": seed, "fit_half": fit_half, "per_model_weights": w_per, "group_weights": w_group,
                "fixed_50_30_20": fixed,
                "per_model": score(judges_test, combine(per_test, w_per)),
                "fitted_group": score(judges_test, combine(group_test, w_group))})

    names = ["severity_aware", "agreement", "theirs_labels"]
    summary = {}
    for kind in ("per_model", "fitted_group"):
        diffs = np.array([[e[kind][i] - e["fixed_50_30_20"][i] for i in range(3)] for e in evaluations])
        mean_diff = diffs.mean(axis=1)
        summary[kind] = {"mean_of_judges_difference": round(float(mean_diff.mean()), 5),
                         "positive_evaluations": int((mean_diff > 0).sum()), "evaluations": len(evaluations),
                         "per_judge_difference": {n: round(float(diffs[:, i].mean()), 5) for i, n in enumerate(names)}}
    rule = summary["per_model"]
    summary["rule_passed"] = bool(rule["mean_of_judges_difference"] >= 0.001 and rule["positive_evaluations"] >= 8
                                  and min(rule["per_judge_difference"].values()) >= -0.0005)

    # Weights fitted on all studies (information only), and their expert-study score.
    per_all, group_all, judges_all = parts(ids)
    w_all = greedy(per_all, judges_all)
    expert = pd.read_csv(EXPERT_LABELS, dtype={STUDY_ID: str}).set_index(STUDY_ID)[TARGETS]
    eids = expert.index.sort_values()
    e_per = {k: ranks(frames[k].loc[eids]) for k in THEIR_SETS}
    e_per["ours"], e_per["r7"] = ranks(read(OURS_EXPERT).loc[eids]), ranks(read(R7_EXPERT).loc[eids])
    e_group = {"theirs": ranks(theirs_ensemble(frames, eids)), "ours": e_per["ours"], "r7": e_per["r7"]}
    y = expert.loc[eids].to_numpy(float)
    summary["all_studies_per_model_weights"] = w_all
    summary["experts"] = {"fixed_50_30_20": round(auc_columns(y, combine(e_group, FIXED)), 4),
                          "per_model_all_studies_weights": round(auc_columns(y, combine(e_per, w_all)), 4)}
    summary["groups"] = {"A": list(GROUP_A), "B": list(GROUP_B)}
    args.out.write_text(json.dumps({"summary": summary, "evaluations": evaluations}, indent=1) + "\n")
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
