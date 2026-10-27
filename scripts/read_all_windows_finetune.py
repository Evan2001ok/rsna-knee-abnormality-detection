#!/usr/bin/env python3
"""Reading of the all-window fine-tune (Oct 17, 2026; rules fixed beforehand).

All-window runs (coatnet_384_v7r7mix30_ftall_fold0-3: fine-tuned on every window of each study) vs the 12-window
controls (coatnet_384_v7r7mix30_ft12_fold0-3: the same fine-tune with 12 windows), both fine-tuned from D's averaged
models (D = the folds trained on our round-6 labels, 0.7 x v7 + 0.3 x R7): pooled over the validation studies of folds
0-3 (validation-averaged.csv, paired), macro AUC on the three judges (severity-aware report labels, report + GPT
agreement labels, the teammates' labels cut at 0.5) and their mean, with a paired study resampling interval; the 58
expert studies: mean of the four folds' expert58-plain.csv. Per finding for information, and both against D's folds 0-3
on the same studies. Bug check per fold: severity-aware AUC no more than 0.005 below D's.

Run: CUDA_VISIBLE_DEVICES="" PYTHONPATH=scripts:src nice .venv/bin/python scripts/read_all_windows_finetune.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from evaluate_fast320 import EXPERT_LABELS, JUDGES, read, their_labels_judge
from scipy.stats import rankdata

from rsna_knee_abnormality_detection.constants import STUDY_ID, TARGETS

ARMS = {"ftall": "runs/coatnet_384_v7r7mix30_ftall_fold", "ft12": "runs/coatnet_384_v7r7mix30_ft12_fold",
        "D": "runs/coatnet_384_v7oai_r7mix30_fold"}
FOLDS = (0, 1, 2, 3)
WEAK = ("PF OA", "Lateral OA", "Synovitis")
OUT = Path("artifacts/all-windows-finetune/reading.json")
REPS = 2000


def aucs(y: np.ndarray, s: np.ndarray) -> np.ndarray:
    """Per-finding tie-aware AUC on cells labelled exactly 0 or 1 (NaN where a finding has one class)."""
    out = np.full(y.shape[1], np.nan)
    for j in range(y.shape[1]):
        known = (y[:, j] == 0) | (y[:, j] == 1)
        yy, ss = y[known, j], s[known, j]
        pos = int((yy == 1).sum())
        neg = len(yy) - pos
        if pos and neg:
            r = rankdata(ss)
            out[j] = (r[yy == 1].sum() - pos * (pos + 1) / 2) / (pos * neg)
    return out


def labels(path: Path, ids: pd.Index) -> np.ndarray:
    frame = pd.read_csv(path, dtype={STUDY_ID: str}).set_index(STUDY_ID)[TARGETS]
    return frame.reindex(ids).to_numpy(float)


def main() -> None:
    judges = {**{k: v for k, v in JUDGES.items()}, "their_labels": their_labels_judge()}
    frames = {arm: {k: read(Path(f"{stem}{k}") / "validation-averaged.csv") for k in FOLDS} for arm, stem in ARMS.items()}
    result = {"rules": "experiments/all-windows-finetune-20261017.json", "bug_check": {}}
    for k in FOLDS:
        ids = frames["D"][k].index
        for arm in ("ftall", "ft12"):
            if sorted(frames[arm][k].index) != sorted(ids):
                raise ValueError(f"fold {k}: {arm} validates on other studies than D")
        y = labels(judges["severity_aware"], ids)
        sev = {arm: float(np.nanmean(aucs(y, frames[arm][k].loc[ids, TARGETS].to_numpy(float)))) for arm in ARMS}
        result["bug_check"][f"fold{k}"] = {**{a: round(v, 5) for a, v in sev.items()},
                                           "pass": all(sev[a] >= sev["D"] - 0.005 for a in ("ftall", "ft12"))}
    ids = pd.Index(sorted(set().union(*[frames["D"][k].index for k in FOLDS])))
    pooled = {arm: pd.concat(frames[arm].values()).loc[ids, TARGETS].to_numpy(float) for arm in ARMS}
    ys = {name: labels(path, ids) for name, path in judges.items()}
    per_finding = {name: {arm: aucs(y, pooled[arm]) for arm in ARMS} for name, y in ys.items()}
    macro = {name: {arm: float(np.nanmean(v)) for arm, v in d.items()} for name, d in per_finding.items()}
    rng = np.random.default_rng(0)
    boot = {pair: [] for pair in (("ftall", "ft12"), ("ftall", "D"), ("ft12", "D"))}
    for _ in range(REPS):
        idx = rng.integers(0, len(ids), len(ids))
        m = {arm: np.mean([np.nanmean(aucs(y[idx], pooled[arm][idx])) for y in ys.values()]) for arm in ARMS}
        for a, b in boot:
            boot[(a, b)].append(m[a] - m[b])
    mean_of_judges = {arm: float(np.mean([macro[j][arm] for j in ys])) for arm in ARMS}
    expert_y = pd.read_csv(EXPERT_LABELS, dtype={STUDY_ID: str}).set_index(STUDY_ID)[TARGETS]
    experts = {}
    for arm, stem in ARMS.items():
        mean = sum(read(Path(f"{stem}{k}") / "expert58-plain.csv").loc[expert_y.index, TARGETS] for k in FOLDS) / len(FOLDS)
        experts[arm] = aucs(expert_y.to_numpy(float), mean.to_numpy(float))
    expert_macro = {arm: float(np.nanmean(v)) for arm, v in experts.items()}

    def diff(a: str, b: str) -> dict:
        lo, hi = np.percentile(boot[(a, b)], [2.5, 97.5])
        return {"mean_of_judges": round(mean_of_judges[a] - mean_of_judges[b], 5),
                "interval_95": [round(float(lo), 5), round(float(hi), 5)],
                "per_judge": {j: round(macro[j][a] - macro[j][b], 5) for j in ys},
                "experts58": round(expert_macro[a] - expert_macro[b], 4),
                "weak_findings_mean_of_judges": {f: round(float(np.mean([per_finding[j][a][TARGETS.index(f)]
                                                                          - per_finding[j][b][TARGETS.index(f)] for j in ys])), 4)
                                                 for f in WEAK},
                "weak_findings_experts58": {f: round(float(experts[a][TARGETS.index(f)] - experts[b][TARGETS.index(f)]), 4)
                                            for f in WEAK}}

    main_diff = diff("ftall", "ft12")
    helps = (main_diff["experts58"] >= -0.005 and main_diff["mean_of_judges"] >= -0.001
             and (main_diff["experts58"] >= 0.003 or main_diff["mean_of_judges"] >= 0.0005))
    result.update({
        "studies": len(ids),
        "scores": {arm: {"mean_of_judges": round(mean_of_judges[arm], 5), **{j: round(macro[j][arm], 5) for j in ys},
                         "experts58": round(expert_macro[arm], 4)} for arm in ARMS},
        "all_window_vs_control": main_diff,
        "information_vs_D": {"ftall": diff("ftall", "D"), "ft12": diff("ft12", "D")},
        "all_windows_help": bool(helps),
    })
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=1) + "\n")
    print(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
