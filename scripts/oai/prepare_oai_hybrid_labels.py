#!/usr/bin/env python3
"""Hybrid OAI labels (Oct 16, 2026; rules fixed before building): our v1 OAI labels (MOAKS grades mapped by fixed
rules: scripts/oai/prepare_oai_labels.py without --fixed) averaged with the teammates' v3 OAI labels (a vision-language
model reading each knee's images with its MOAKS gradings) on the OAI baseline knees.

- The seven findings both cover (both menisci, the three OA findings, effusion, synovitis): 0.5 x ours + 0.5 x theirs
  where both exist; ours where only ours exists.
- Cells only theirs cover stay blank, as in v1. (A planned fill of the 739 effusion gaps from theirs failed its check: on
  those knees their reading had no MOAKS effusion grade and its values sit mid-scale, so the fit learned where both exist
  turned 97.6% of them negative, against 80% elsewhere and about 70% by the OAI-free models.)
- ACL and Baker's: ours. MCL, contusion, fracture: blank, as in v1. Other visits: unchanged.

Their file (THEIRS below: where we kept the teammates' file; point it to your copy) is read as plain CSV only. The
provenance file also holds a "judge check", made as in an earlier comparison script (not included): how well the labels
agree with predictions on the OAI knees (JUDGE below) from our round-4 CoAtNet folds (0.957 public), which never trained
on OAI.

Run: .venv/bin/python scripts/oai/prepare_oai_hybrid_labels.py --output-dir data/labels/oai_hybrid_v1v3
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

ID = "StudyInstanceUID"
SHARED = ["Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA", "PF OA", "Effusion", "Synovitis"]
OURS = Path("data/labels/oai_moaks_v1/labels.csv")
THEIRS = Path("data/teammates/labels/oai/OAI_soft_labels_2399_v3.csv")
JUDGE = Path("artifacts/oai-check/pred957_oai.csv")


def read_theirs(path: Path) -> pd.DataFrame:
    """Their values with masked cells as NaN, indexed like ours (OAI_<id>_<side>_<visit>)."""
    raw = pd.read_csv(path, dtype=str, encoding="utf-8-sig")
    out = pd.DataFrame(index="OAI_" + raw["knee_id"] + "_" + raw["visit"])
    for f in SHARED:
        values = pd.to_numeric(raw[f]).where(raw[f"{f}__mask"].astype(int) == 1)
        out[f] = values.to_numpy()
    if not out.index.is_unique:
        raise ValueError("Duplicate knees in their labels")
    return out


def hybrid(ours: pd.DataFrame, theirs: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """ours: our rows for the knees to change (all 12 targets); theirs: read_theirs() rows. Returns the new rows and counts."""
    if set(ours.index) != set(theirs.index):
        raise ValueError("Our and their knees must match")
    theirs = theirs.loc[ours.index]
    out = ours.copy()
    counts = {}
    for f in SHARED:
        o, t = ours[f].astype(float), theirs[f].astype(float)
        both, only_ours, only_theirs = o.notna() & t.notna(), o.notna() & t.isna(), o.isna() & t.notna()
        value = pd.Series(np.nan, index=ours.index)
        value[both] = 0.5 * o[both] + 0.5 * t[both]
        value[only_ours] = o[only_ours]
        out[f] = value
        counts[f] = {"both": int(both.sum()), "only_ours": int(only_ours.sum()), "only_theirs_left_blank": int(only_theirs.sum()),
                     "blank": int(value.isna().sum())}
    return out, counts


def judge_check(labels: pd.DataFrame) -> dict:
    judge = pd.read_csv(JUDGE, dtype={ID: str}).set_index(ID)
    result = {}
    for f in SHARED:
        frame = pd.DataFrame({"label": labels[f]}).join(judge[f].rename("judge"), how="inner").dropna()
        y = frame["label"].to_numpy() >= 0.5
        result[f] = {"knees": len(frame), "rank_agreement": round(float(spearmanr(frame["label"], frame["judge"])[0]), 3),
                     "judge_auc_at_05": round(float(roc_auc_score(y, frame["judge"])), 3), "positive_share": round(float(y.mean()), 3)}
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"{args.output_dir} already exists")
    ours_all = pd.read_csv(OURS, dtype={ID: str, "oai_id": str})
    base = ours_all.set_index(ID)
    baseline = base.index[base["visit"] == "V00"]
    theirs = read_theirs(THEIRS)
    targets = [c for c in base.columns if c not in ("oai_id", "side", "visit")]
    new_rows, counts = hybrid(base.loc[baseline, targets], theirs)
    labels = base.copy()
    labels.loc[baseline, targets] = new_rows[targets].to_numpy()
    values = labels.loc[baseline, SHARED].to_numpy(float)
    if not np.all(np.isnan(values) | ((values >= 0) & (values <= 1))):
        raise ValueError("Labels must be within [0, 1]")
    for f in ("ACL", "Baker's", "MCL", "Contusion", "Fracture"):
        if not labels[f].equals(base[f]):
            raise ValueError(f"{f} must stay as in v1")
    if not labels.drop(index=baseline).equals(base.drop(index=baseline)):
        raise ValueError("Other visits must stay unchanged")
    args.output_dir.mkdir(parents=True)
    output = args.output_dir / "labels.csv"
    labels.reset_index().to_csv(output, index=False)
    digest = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()  # noqa: E731
    provenance = {
        "rule": __doc__.split("\n\n")[1].strip(), "baseline_knees": len(baseline), "counts": counts,
        "judge_check_hybrid": judge_check(labels.loc[baseline]), "judge_check_ours_v1": judge_check(base.loc[baseline]),
        "inputs": {str(p): digest(p) for p in (OURS, THEIRS, JUDGE)}, "output_sha256": digest(output),
        "reproduction": f".venv/bin/python scripts/oai/prepare_oai_hybrid_labels.py --output-dir {args.output_dir}"}
    (args.output_dir / "label_provenance.json").write_text(json.dumps(provenance, indent=1) + "\n")
    print(json.dumps({"counts": counts, "judge_check_hybrid": provenance["judge_check_hybrid"]}, indent=1))


if __name__ == "__main__":
    main()
