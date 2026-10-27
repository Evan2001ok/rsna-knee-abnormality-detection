#!/usr/bin/env python3
"""Bug check of the consensus full-data seeds (Oct 17, 2026; rule fixed before training). Seeds 42-45 train on every
study except 100 held-out cross-validation studies, called placeholder studies here (kept out only as a small check
set). Rule: each seed's macro AUC on those 100 studies, on every judge (severity-aware report labels, report + GPT
agreement labels, the teammates' labels cut at 0.5), is no more than 0.02 below the out-of-fold AUC of the consensus58
seed-42 fold models (4-fold runs on the same labels) on the same 100 studies. Information only beyond that: 100 studies
cannot rank the seeds.

Run: CUDA_VISIBLE_DEVICES="" PYTHONPATH=scripts .venv/bin/python scripts/check_consensus_fulldata.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from compare_team_blend import read
from consensus_seed_oof_info import seed_frame
from evaluate_fast320 import JUDGES, their_labels_judge
from refit_blend_compare import macro_auc

from rsna_knee_abnormality_detection.constants import STUDY_ID, TARGETS

SEEDS = (42, 43, 44, 45)
LIMIT = 0.02


def main() -> None:
    folds = seed_frame(42)
    judges = {**{k: Path(v) for k, v in JUDGES.items()}, "their_labels": their_labels_judge()}
    result, failures = {"rule": f"each judge no more than {LIMIT} below the seed-42 fold models", "seeds": {}}, []
    reference_ids = None
    for seed in SEEDS:
        frame = read(Path(f"runs/coatnet_384_consensus58_all_s{seed}") / "validation-averaged.csv")
        ids = frame.index.sort_values()
        if len(ids) != 100 or not ids.isin(folds.index).all():
            raise SystemExit(f"seed {seed}: expected the 100 placeholder studies, got {len(ids)}")
        if reference_ids is not None and not ids.equals(reference_ids):
            raise SystemExit(f"seed {seed}: different placeholder studies")
        reference_ids = ids
        row = {}
        for name, path in judges.items():
            y = pd.read_csv(path, dtype={STUDY_ID: str}).set_index(STUDY_ID)[TARGETS].reindex(ids).to_numpy(float)
            known = (y == 0) | (y == 1)
            judge = (np.where(known, y, 0), known)
            ours, reference = macro_auc(frame.loc[ids].to_numpy(float), judge), macro_auc(folds.loc[ids].to_numpy(float), judge)
            row[name] = {"seed": round(ours, 4), "fold_models": round(reference, 4), "difference": round(ours - reference, 4)}
            if ours < reference - LIMIT:
                failures.append(f"seed {seed} {name}: {ours:.4f} vs {reference:.4f}")
        result["seeds"][seed] = row
    result["failures"], result["pass"] = failures, not failures
    out = Path("artifacts/consensus-fulldata/bug-check.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=1) + "\n")
    print(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
