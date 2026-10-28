#!/usr/bin/env python3
"""Consensus training labels (Oct 16, 2026; the rules were fixed before the labels were built).

Every cross-validation study and the 53 unlabeled studies get the equal-weight mean of the sources present:
  S1, ours: the v7 labels (our round-5 labels, made with scripts/prepare_source_choice_labels.py); for the 53, which
      have no report labels, the GPT image readings where GPT marked the finding assessable;
  S2, theirs: the teammates' 0.7 VLM + 0.3 LLM readings (rows marked vlm07_llm03 only; their 58 expert rows are never
      read);
  S3, all models: per-finding percentile ranks of three groups (ours = mean probability of our three five-fold CoAtNet
      sets v7 + OAI, D and H; theirs = equal mean of the teammates' nine sets; R7, heliosli's public model), weighted by
      group weights fitted on the 4,296 cross-validation studies by the mean of the three judges, then mapped onto the
      label scale (the sorted values of the mean of S1 and S2), as teacher() does in scripts/prepare_packed_labels.py.
      D = trained on our round-6 labels (0.7 x v7 + 0.3 x R7's out-of-fold predictions); H = v7 with the hybrid OAI
      labels (scripts/oai/prepare_oai_hybrid_labels.py). The three judges are our checking label sets: severity-aware
      report labels, report + GPT agreement labels and the teammates' labels cut at 0.5.
All model predictions are out of fold; for the 53, which no fold model trained on, our sets' five fold models and R7's
four models are averaged (artifacts/consensus/pred53). The 58 expert rows are copied unchanged from the v7 file, which
holds the host's expert labels.

Inputs: the teammates' labels file is public on their branch multibackbone-pipeline at labels/rsna/conv_labels_used.csv;
put it at data/teammates/labels/rsna/conv_labels_used.csv. Its 58 expert rows are 0.5 placeholders; this script never
reads them and copies the host's expert labels instead. The teammates' out-of-fold predictions, R7's out-of-fold
predictions and our GPT readings are not public.

--round2 (Oct 17, 2026): S3's ours group becomes the mean of the consensus58 seeds 42, 43 and 44 out of fold (each run's
validation-averaged.csv, four-fold split) on the 4,296; for the 53, which the consensus runs trained on, it keeps the
round-1 predictions above. consensus58 = the 4-fold CoAtNet-2 runs on these labels that also train on the 58 expert
studies. Output data/labels/consensus_v2.

Run: CUDA_VISIBLE_DEVICES="" PYTHONPATH=scripts .venv/bin/python scripts/prepare_consensus_labels.py [--round2]
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from compare_team_blend import R7, read, their_frames
from consensus_seed_oof_info import run_prefix, seed_frame
from evaluate_fast320 import JUDGES, pooled, their_labels_judge
from refit_blend_compare import GRID, SEEDS, macro_auc, pct_ranks, scores

from rsna_knee_abnormality_detection.constants import STUDY_ID, TARGETS

V7 = Path("data/labels/source_choice_v7_own_coatnet/labels.csv")
GPT_ALL = Path("artifacts/gpt-image-labels/v1/labels-gpt-all-medium.csv")
THEIR_LABELS = Path("data/teammates/labels/rsna/conv_labels_used.csv")
NEWCONV = Path("artifacts/teammates/newconv-20261016/oof")
PRED53 = Path("artifacts/consensus/pred53")
SPLITS = Path("data/splits/baseline/splits.csv")
OURS_SETS = {"coatnet_384_srcchoice7own_oai": "runs/coatnet_384_srcchoice7own_oai_fold",
             "coatnet_384_v7oai_r7mix30": "runs/coatnet_384_v7oai_r7mix30_fold",
             "coatnet_384_v7oai_oaihybrid": "runs/coatnet_384_v7oai_oaihybrid_fold"}
CONSENSUS_SEEDS = (42, 43, 44)
THEIR_KEYS = ("conv", "eff", "dino", "dino3", "rad", "tiny", "conv_new", "dino_new", "newconv")
GROUPS = ("theirs", "ours", "r7")


def mean_present(frames: list[pd.DataFrame]) -> pd.DataFrame:
    """Equal-weight mean over the sources that have a value for each cell."""
    index, columns = frames[0].index, frames[0].columns
    stack = np.stack([f.reindex(index=index, columns=columns).to_numpy(float) for f in frames])
    count = np.isfinite(stack).sum(0)
    total = np.nansum(stack, 0)
    out = np.where(count > 0, total / np.maximum(count, 1), np.nan)
    return pd.DataFrame(out, index=index, columns=columns)


def quantile_map(scores_: pd.Series, reference: np.ndarray) -> np.ndarray:
    """Keep the order of `scores_` and give it the value distribution of `reference` (missing values ignored)."""
    reference = np.sort(reference[np.isfinite(reference)])
    position = (scores_.rank(method="average") - 1) / max(len(scores_) - 1, 1)
    return np.interp(position.to_numpy(), np.linspace(0, 1, len(reference)), reference)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_s1(cv: pd.Index, unlabeled: pd.Index) -> pd.DataFrame:
    v7 = read(V7)
    gpt = pd.read_csv(GPT_ALL, dtype={STUDY_ID: str}).set_index(STUDY_ID)
    rows = {t: pd.to_numeric(gpt[t]).where(gpt[f"{t}__assessable"].astype(str) == "True") for t in TARGETS}
    gpt = pd.DataFrame(rows).loc[unlabeled]
    return pd.concat([v7.loc[cv], gpt])


def load_s2(ids: pd.Index) -> pd.DataFrame:
    frame = pd.read_csv(THEIR_LABELS, dtype={STUDY_ID: str}).set_index(STUDY_ID)
    frame = frame[frame["label_source"] == "vlm07_llm03"]  # their 58 expert rows (0.5 placeholders) are never read
    if not set(ids) <= set(frame.index):
        raise ValueError("their labels must cover every study being relabelled")
    return frame.loc[ids, TARGETS].astype(float)


def ours_cv_sources(round2: bool) -> list[Path]:
    """The out-of-fold files whose mean is S3's ours group on the 4,296 studies."""
    if round2:
        return [Path(f"{run_prefix(s)}{k}") / "validation-averaged.csv" for s in CONSENSUS_SEEDS for k in range(4)]
    return [Path(f"{p}{k}") / "validation-averaged-model-plain.csv" for p in OURS_SETS.values() for k in range(5)]


def model_groups(cv: pd.Index, unlabeled: pd.Index, round2: bool = False) -> dict[str, pd.DataFrame]:
    if round2:
        ours_cv = sum(seed_frame(s) for s in CONSENSUS_SEEDS).loc[cv] / len(CONSENSUS_SEEDS)
    else:
        ours_cv = sum(pooled(p) for p in OURS_SETS.values()).loc[cv] / len(OURS_SETS)
    ours_53 = sum(read(PRED53 / f"{name}.csv") for name in OURS_SETS).loc[unlabeled] / len(OURS_SETS)
    frames = their_frames()
    frames["newconv"] = pd.concat([pd.read_csv(NEWCONV / p / "oof_predictions.csv", dtype={STUDY_ID: str})
                                   for p in ("fold12", "fold23")]).set_index(STUDY_ID)[TARGETS].astype(float)
    ids = cv.append(unlabeled)
    theirs = sum(frames[k].loc[ids] for k in THEIR_KEYS) / len(THEIR_KEYS)
    r7 = pd.concat([pd.concat([read(p) for p in R7]).loc[cv], read(PRED53 / "r7.csv").loc[unlabeled]])
    return {"theirs": theirs, "ours": pd.concat([ours_cv, ours_53]), "r7": r7}


def judges(ids: pd.Index) -> list:
    out = []
    for path in [*JUDGES.values(), their_labels_judge()]:
        y = pd.read_csv(path, dtype={STUDY_ID: str}).set_index(STUDY_ID)[TARGETS].reindex(ids).to_numpy(float)
        known = (y == 0) | (y == 1)
        out.append((np.where(known, y, 0), known))
    return out


def fit_weights(groups: dict[str, pd.DataFrame], cv: pd.Index) -> tuple[tuple, dict]:
    """Group weights by the mean of the three judges on the 4,296 studies, and a held-out check against equal weights."""
    arrays = [groups[g].loc[cv].to_numpy(float) for g in GROUPS]
    js = judges(cv)

    def best_at(idx):
        ranked = [pct_ranks(a[idx]) for a in arrays]
        jj = [(y[idx], k[idx]) for y, k in js]
        return max(GRID, key=lambda w: np.mean(scores(ranked, w, jj))), ranked, jj

    weights, _, _ = best_at(np.arange(len(cv)))
    held = []
    for seed in SEEDS:
        order = np.random.default_rng(seed).permutation(len(cv))
        halves = [np.sort(order[: len(cv) // 2]), np.sort(order[len(cv) // 2:])]
        for fit_half in (0, 1):
            w, _, _ = best_at(halves[fit_half])
            test = halves[1 - fit_half]
            ranked = [pct_ranks(a[test]) for a in arrays]
            jj = [(y[test], k[test]) for y, k in js]
            held.append({"weights": w, "fitted": float(np.mean(scores(ranked, w, jj))),
                         "equal": float(np.mean(scores(ranked, (1 / 3, 1 / 3, 1 / 3), jj)))})
    diff = np.array([h["fitted"] - h["equal"] for h in held])
    check = {"held_out_fitted_minus_equal_mean_of_judges": round(float(diff.mean()), 5),
             "positive": f"{int((diff > 0).sum())} of {len(diff)}", "weights_per_half": [h["weights"] for h in held]}
    return weights, check


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--round2", action="store_true", help="S3's ours group = the three consensus seeds (Oct 17)")
    args = parser.parse_args()
    args.output_dir = args.output_dir or Path("data/labels/consensus_v2" if args.round2 else "data/labels/consensus_v1")
    splits = pd.read_csv(SPLITS, dtype={STUDY_ID: str}).set_index(STUDY_ID)
    cv = splits.index[splits["fold"] >= 0].sort_values()
    unlabeled = splits.index[splits["fold"] == -2].sort_values()
    if len(cv) != 4296 or len(unlabeled) != 53:
        raise ValueError(f"expected 4,296 cross-validation and 53 unlabeled studies, found {len(cv)} and {len(unlabeled)}")
    ids = cv.append(unlabeled)
    s1, s2 = load_s1(cv, unlabeled), load_s2(ids)
    groups = model_groups(cv, unlabeled, args.round2)
    weights, check = fit_weights(groups, cv)
    blend = sum(w * groups[g].rank(pct=True) for w, g in zip(weights, GROUPS, strict=True))
    reference = mean_present([s1, s2])
    s3 = pd.DataFrame({t: quantile_map(blend[t], reference[t].to_numpy()) for t in TARGETS}, index=ids)
    mixed = mean_present([s1, s2, s3])
    full = pd.read_csv(V7, dtype={STUDY_ID: str}).set_index(STUDY_ID)
    labels = full.copy()
    labels.loc[ids, TARGETS] = mixed.loc[ids, TARGETS].to_numpy()
    tag = "consensus_v2" if args.round2 else "consensus"
    labels.loc[unlabeled, "label_source"] = f"{tag}_unlabeled"
    labels.loc[cv, "label_source"] = tag
    values = labels.loc[ids, TARGETS].to_numpy(float)
    experts = splits.index[splits["fold"] == -1]
    if len(labels) != 4407 or not np.isfinite(values).all() or (values < 0).any() or (values > 1).any():
        raise ValueError("consensus labels must cover 4,407 studies with finite values in [0, 1]")
    if not labels.loc[experts].equals(full.loc[experts]):
        raise ValueError("expert rows must stay unchanged")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    labels.reset_index().to_csv(args.output_dir / "labels.csv", index=False)
    per_finding = {}
    for t in TARGETS:
        cols = {"S1": s1[t], "S2": s2[t], "S3": s3[t], "mixed": mixed[t]}
        per_finding[t] = {"mean": {k: round(float(v.loc[cv].mean()), 4) for k, v in cols.items()},
                          "spearman_cv": {f"{a}-{b}": round(float(cols[a].loc[cv].corr(cols[b].loc[cv], method="spearman")), 3)
                                          for a, b in (("S1", "S2"), ("S1", "S3"), ("S2", "S3"))},
                          "mean_53": round(float(mixed[t].loc[unlabeled].mean()), 4)}
    js = judges(cv)
    s3_alone = [round(macro_auc(s3.loc[cv].to_numpy(float), j), 4) for j in js]
    provenance = {
        "rules": "experiments/consensus-round2-20261017.json" if args.round2 else "experiments/consensus-labels-20261016.json",
        "s3_ours_group_on_cv": [str(p) for p in ours_cv_sources(args.round2)],
        "studies_relabelled": {"cv": len(cv), "unlabeled": len(unlabeled)},
        "s3_group_weights_theirs_ours_r7": weights, "s3_weight_check": check,
        "s3_alone_on_cv_severity_agreement_their_labels": s3_alone,
        "gpt_missing_cells_on_53": int(s1.loc[unlabeled].isna().to_numpy().sum()),
        "inputs": {str(p): sha(p) for p in (V7, GPT_ALL, THEIR_LABELS, SPLITS, *sorted(PRED53.glob("*.csv")),
                                            *(ours_cv_sources(True) if args.round2 else []))},
        "labels_sha256": sha(args.output_dir / "labels.csv"), "per_finding": per_finding}
    (args.output_dir / "label_provenance.json").write_text(json.dumps(provenance, indent=1) + "\n")
    print(json.dumps({k: provenance[k] for k in ("s3_group_weights_theirs_ours_r7", "s3_weight_check", "s3_alone_on_cv_severity_agreement_their_labels",
                                                  "gpt_missing_cells_on_53", "labels_sha256")}, indent=1))


if __name__ == "__main__":
    main()
