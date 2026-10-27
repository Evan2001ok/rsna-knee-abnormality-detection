#!/usr/bin/env python3
"""Reading of the consensus all-window fine-tunes (Oct 18, 2026; rules fixed beforehand).

The four fine-tuned folds (coatnet_384_consensus58_ftall_fold0-3: fine-tuned on every window of each study) vs the four
consensus folds they start from (coatnet_384_consensus58_fold0-3: 4-fold runs on the consensus labels that also train on
the 58 expert studies), pooled over their validation studies (validation-averaged.csv, paired): macro AUC on the three
judges (severity-aware report labels, report + GPT agreement labels, the teammates' labels cut at 0.5) and their mean,
with a paired study resampling interval; per finding for information. Bug check per fold: severity-aware AUC no more
than 0.005 below the consensus fold's. Use them if the mean of judges gains at least +0.0005 and no judge drops below
-0.001. The 58 experts are information only (this arm trained on them).

Run: CUDA_VISIBLE_DEVICES="" PYTHONPATH=scripts:src nice .venv/bin/python scripts/read_consensus_all_windows.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from evaluate_fast320 import EXPERT_LABELS, JUDGES, read, their_labels_judge
from read_all_windows_finetune import aucs, labels

from rsna_knee_abnormality_detection.constants import STUDY_ID, TARGETS

ARMS = {"ftall": "runs/coatnet_384_consensus58_ftall_fold", "consensus": "runs/coatnet_384_consensus58_fold"}
FOLDS = (0, 1, 2, 3)
OUT = Path("artifacts/consensus-all-windows/reading.json")
REPS = 2000


def main() -> None:
    codes = {k: json.loads((Path(f"{ARMS['ftall']}{k}") / "provenance.json").read_text())["code_sha256"][:8] for k in FOLDS}
    judges = {**JUDGES, "their_labels": their_labels_judge()}
    frames = {arm: {k: read(Path(f"{stem}{k}") / "validation-averaged.csv") for k in FOLDS} for arm, stem in ARMS.items()}
    result = {"rules": "experiments/consensus-all-windows-20261017.json", "code_sha256": codes, "bug_check": {}}
    for k in FOLDS:
        ids = frames["consensus"][k].index
        if sorted(frames["ftall"][k].index) != sorted(ids):
            raise ValueError(f"fold {k}: the fine-tune validates on other studies")
        y = labels(judges["severity_aware"], ids)
        sev = {arm: float(np.nanmean(aucs(y, frames[arm][k].loc[ids, TARGETS].to_numpy(float)))) for arm in ARMS}
        result["bug_check"][f"fold{k}"] = {**{a: round(v, 5) for a, v in sev.items()}, "pass": sev["ftall"] >= sev["consensus"] - 0.005}
    ids = pd.Index(sorted(set().union(*[frames["consensus"][k].index for k in FOLDS])))
    pooled = {arm: pd.concat(frames[arm].values()).loc[ids, TARGETS].to_numpy(float) for arm in ARMS}
    ys = {name: labels(path, ids) for name, path in judges.items()}
    per = {name: {arm: aucs(y, pooled[arm]) for arm in ARMS} for name, y in ys.items()}
    macro = {name: {arm: float(np.nanmean(v)) for arm, v in d.items()} for name, d in per.items()}
    mean = {arm: float(np.mean([macro[j][arm] for j in ys])) for arm in ARMS}
    rng = np.random.default_rng(0)
    boot = []
    for _ in range(REPS):
        i = rng.integers(0, len(ids), len(ids))
        boot.append(np.mean([np.nanmean(aucs(y[i], pooled["ftall"][i])) - np.nanmean(aucs(y[i], pooled["consensus"][i]))
                             for y in ys.values()]))
    # Experts (information only; this arm trained on the 58). Only consensus fold 3 has 58-study predictions, so the
    # side-by-side uses fold 3 alone; the four fine-tunes' average is reported on its own.
    ey = pd.read_csv(EXPERT_LABELS, dtype={STUDY_ID: str}).set_index(STUDY_ID)[TARGETS]

    def expert_auc(stem: str, folds: tuple[int, ...]) -> float:
        mean = sum(read(Path(f"{stem}{k}") / "expert58-plain.csv").loc[ey.index, TARGETS] for k in folds) / len(folds)
        return float(np.nanmean(aucs(ey.to_numpy(float), mean.to_numpy(float))))

    experts = {arm: expert_auc(stem, (3,)) for arm, stem in ARMS.items()}
    experts_ftall_four = expert_auc(ARMS["ftall"], FOLDS)
    d_mean = mean["ftall"] - mean["consensus"]
    per_judge = {j: macro[j]["ftall"] - macro[j]["consensus"] for j in ys}
    use = d_mean >= 0.0005 and min(per_judge.values()) >= -0.001
    result.update({
        "studies": len(ids),
        "scores": {arm: {"mean_of_judges": round(mean[arm], 5), **{j: round(macro[j][arm], 5) for j in ys},
                         "experts58_fold3_information": round(experts[arm], 4)} for arm in ARMS},
        "experts58_ftall_four_folds_information": round(experts_ftall_four, 4),
        "ftall_minus_consensus": {
            "mean_of_judges": round(d_mean, 5), "interval_95": [round(float(x), 5) for x in np.percentile(boot, [2.5, 97.5])],
            "per_judge": {j: round(v, 5) for j, v in per_judge.items()},
            "experts58_fold3_information": round(experts["ftall"] - experts["consensus"], 4),
            "per_finding_mean_of_judges": {t: round(float(np.mean([per[j]["ftall"][i] - per[j]["consensus"][i] for j in ys])), 4)
                                           for i, t in enumerate(TARGETS)}},
        "use_the_fine_tunes": bool(use),
    })
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=1) + "\n")
    print(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
