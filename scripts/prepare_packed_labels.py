#!/usr/bin/env python3
"""Packed labels (Oct 15, 2026): the v7 source-choice labels with only the model source replaced by a blend of our
best out-of-fold models.

v7 (our round-5 labels, data/labels/source_choice_v7_own_coatnet, from prepare_source_choice_labels.py) mixes, per
finding, the report, the GPT image reading and CoAtNet out-of-fold predictions (the v6 five folds: CoAtNet trained on
the version-6 labels, on the label scale), with weights from the expert-study win shares kept
between 10% and 70%. Here the weights are computed exactly as for v7 (same v6 source, same resamples), so they are v7's
weights; with --mode v7 the script reproduces v7's labels.csv byte for byte. With --mode packed the model source becomes
the per-finding rank blend of our v7 + OAI five folds (CoAtNet trained on v7 labels plus OAI knees; 0.7, label
scale) and R7 (0.3, its four hash folds), out of fold,
mapped back onto the value distribution of our own folds' label-scale readings (so its scale matches the source it
replaces). Expert and unlabelled rows are unchanged; no expert label is read except through the v7 weights.

--mode r7mix (Oct 15, 2026; round 6 in the README): a stronger dose of R7 for every finding. Each report-labelled
study's label is (1 - share) x its v7 label + share x R7's out-of-fold prediction, R7 mapped by rank onto our folds'
label-scale values as above (our share 0). Default share 0.3, as in the best ours/R7 blend weight out of fold (70/30).

R7 is heliosli's public ConvNeXt-small model (see the README). Its out-of-fold predictions (R7_OOF) come from
scripts/run_public_compact_oof.py (one run per fold); its four folds are set by a hash of the study ID.

Run: .venv/bin/python scripts/prepare_packed_labels.py --mode packed --output-dir data/labels/v7_packed_ours_r7
     .venv/bin/python scripts/prepare_packed_labels.py --mode r7mix --r7-share 0.3 --output-dir data/labels/v7_r7mix30
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from prepare_source_choice_labels import (
    GPT_TRAINING,
    REPORT_TRAINING,
    SOURCES,
    UNCHANGED_SOURCE,
    coatnet_from_sets,
    numeric,
    source_weights,
)

from rsna_knee_abnormality_detection.constants import STUDY_ID, TARGETS
from rsna_knee_abnormality_detection.utils import file_hash, write_json

V6_SET = "runs/coatnet_384native_srcchoice6_fold"  # v7's model source
OURS_SET = "runs/coatnet_384_srcchoice7own_oai_fold"  # v7 + OAI five folds (in a bundle that scored 0.964 public)
R7_OOF = [Path(f"artifacts/s6r7-results/r7_oof_f{k}/predictions.csv") for k in range(4)]
OURS_SHARE = 0.7


def mix(v7: pd.DataFrame, r7_mapped: pd.DataFrame, share: float) -> pd.DataFrame:
    """--mode r7mix: (1 - share) x v7 label + share x R7 on the label scale, cell by cell."""
    if not 0 < share < 1:
        raise ValueError("The R7 share must be between 0 and 1")
    return (1 - share) * v7 + share * r7_mapped


def teacher(ids: pd.Index, ours_share: float = OURS_SHARE) -> pd.DataFrame:
    ours, _ = coatnet_from_sets([OURS_SET], ids)
    r7 = pd.concat([pd.read_csv(p, dtype={STUDY_ID: str}).set_index(STUDY_ID)[TARGETS] for p in R7_OOF])
    if not r7.index.is_unique or not set(ids) <= set(r7.index):
        raise ValueError("R7 out-of-fold predictions must cover every cross-validation study once")
    r7 = r7.loc[ids]
    out = {}
    for target in TARGETS:
        blended = ours_share * ours[target].rank(pct=True) + (1 - ours_share) * r7[target].rank(pct=True)
        position = (blended.rank(method="average") - 1) / (len(ids) - 1)
        reference = np.sort(ours[target].to_numpy())
        out[target] = np.interp(position.to_numpy(), np.linspace(0, 1, len(ids)), reference)
    return pd.DataFrame(out, index=ids)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["v7", "packed", "r7mix"], required=True)
    parser.add_argument("--r7-share", type=float, default=0.3, help="--mode r7mix only")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"{args.output_dir} already exists")
    base = pd.read_csv(UNCHANGED_SOURCE, dtype={STUDY_ID: str}).set_index(STUDY_ID)
    gpt_frame = pd.read_csv(GPT_TRAINING, dtype={STUDY_ID: str}).set_index(STUDY_ID)
    ids = gpt_frame.index
    if not (base.loc[ids, "label_source"] == "report").all():
        raise ValueError("GPT studies must be the report-labelled cross-validation studies")
    report = numeric(REPORT_TRAINING).loc[ids]
    v6, v6_expert = coatnet_from_sets([V6_SET], ids)
    weights = source_weights(0.10, 0.70, v6_expert, SOURCES)  # v7's weights
    model = v6 if args.mode in ("v7", "r7mix") else teacher(ids)
    labels = base.copy()
    for target in TARGETS:
        gpt = pd.to_numeric(gpt_frame[target]).where(gpt_frame[f"{target}__assessable"].astype(str) == "True")
        values = {"report": report[target], "gpt": gpt, "coatnet": model[target]}
        w = weights[target]["weights"]
        total = sum(w[s] * values[s].fillna(0) for s in SOURCES)
        weight = sum(w[s] * values[s].notna() for s in SOURCES)
        fallback = (report[target] + model[target]) / 2
        labels.loc[ids, target] = (total / weight).where(weight > 0, fallback).to_numpy()
    if args.mode == "r7mix":  # the labels above are exactly v7's (checked below); add R7 to every finding
        v7_cells = labels.loc[ids, TARGETS].astype(float)
        saved_v7 = numeric(Path("data/labels/source_choice_v7_own_coatnet/labels.csv")).loc[ids, TARGETS]
        if not np.allclose(v7_cells.to_numpy(), saved_v7.to_numpy(), atol=1e-9, rtol=0):
            raise ValueError("The recomputed labels differ from v7")
        labels.loc[ids, TARGETS] = mix(v7_cells, teacher(ids, ours_share=0.0), args.r7_share).to_numpy()
    values = labels.loc[ids, TARGETS].to_numpy(dtype=float)
    if not np.isfinite(values).all() or (values < 0).any() or (values > 1).any():
        raise ValueError("Labels must be finite and between zero and one")
    others = labels.index.difference(ids)
    if not labels.loc[others].equals(base.loc[others]):
        raise ValueError("Expert and unlabeled rows must stay unchanged")
    args.output_dir.mkdir(parents=True)
    output = args.output_dir / "labels.csv"
    labels.reset_index().to_csv(output, index=False)
    v7 = numeric(Path("data/labels/source_choice_v7_own_coatnet/labels.csv")).loc[ids, TARGETS]
    rule = {"v7": "unchanged (v6 CoAtNet folds)",
            "packed": (f"replaced by the rank blend of our v7 + OAI folds ({OURS_SHARE}) and R7 ({1 - OURS_SHARE:.1f}), "
                       "out of fold, mapped onto our folds' label-scale distribution"),
            "r7mix": (f"unchanged, then every report-labelled cell mixed {1 - args.r7_share:.2f} v7 + {args.r7_share:.2f} "
                      "R7 out of fold (R7 mapped by rank onto our folds' label-scale distribution)")}[args.mode]
    new = labels.loc[ids, TARGETS].astype(float)
    write_json(args.output_dir / "label_provenance.json", {
        "mode": args.mode,
        "r7_share": args.r7_share if args.mode == "r7mix" else None,
        "rule": "v7 source-choice labels (weights recomputed exactly as for v7) with the model source " + rule,
        "weights": {t: weights[t]["weights"] for t in TARGETS},
        "mean_abs_change_vs_v7": {t: round(float((new[t] - v7[t]).abs().mean()), 4) for t in TARGETS},
        "share_crossing_half_vs_v7": {t: round(float(((new[t] >= 0.5) != (v7[t] >= 0.5)).mean()), 4) for t in TARGETS},
        "studies_relabelled": len(ids), "output_sha256": file_hash(output),
        "inputs": {str(p): file_hash(p) for p in [REPORT_TRAINING, GPT_TRAINING, UNCHANGED_SOURCE, *R7_OOF]},
        "reproduction": f".venv/bin/python scripts/prepare_packed_labels.py --mode {args.mode} --output-dir {args.output_dir}",
    })
    print(f"wrote {output} ({file_hash(output)[:16]})")


if __name__ == "__main__":
    main()
