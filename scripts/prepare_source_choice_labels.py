#!/usr/bin/env python3
"""Labels that use, for each finding, the label source that reads it best, or a mix weighted by how sure we are.

Sources: the report (graded report labels, avg_v2 from prepare_averaged_labels.py), GPT (the GPT-6 Astra image
reading from gpt_image_labels.py) and CoAtNet (384 px CoAtNet predictions from the fold models that never trained on
the study; by default the round-3 models, average of the normal and 448-crop sets, converted back to the label
scale). How well each source reads each finding is measured on the expert studies: the AUC of the report readings
(Pilkwang report labels, silent = 0.5), the medium-effort GPT reading and the five CoAtNet folds against the expert
labels, and how often each source is best over 2,000 resamples of those studies (fixed seed).
Rule (fixed on October 10, 2026, before training): the three sources are mixed, each weighted by the
share of resamples in which it was best, but no source gets more than 70% or less than 10%. (Version 1 used a clear
winner, best in at least 85% of resamples, alone; that left effusion, medial OA and synovitis as CoAtNet's own
predictions, so CoAtNet would mostly learn what it already thinks, and it gave the report nothing on findings where
the report measured here, with silence as 0.5, understates the graded report labels used in training.)
Version 2 is this rule with the default options (round 4 in the README).
Version 3 (--uncapped, October 10, 2026): the plain win shares, with no 10%/70% bounds, to test
whether the cap matters.
Version 4 (--coatnet-set, --lowest, --highest; October 11, 2026): the CoAtNet source becomes the
average of the 0.957 and 0.958 five-fold sets (named by their public leaderboard scores: CoAtNet folds trained on the
version-2 labels, without and with OAI knees), each prediction converted back to the label scale with its run's
positive-term weights, on both the cross-validation and the expert studies; and each source is kept between 20% and 60%, the best region of an October 10 test of 56 bound
settings on held-out halves of the expert studies (our records, not public), fixed in advance and not re-chosen with
the new source.
Version 5 (October 12, 2026): as version 4, but the CoAtNet source is the 0.958 five-fold set
alone (version-2 CoAtNet folds with OAI knees): --coatnet-set runs/coatnet_384_srcchoice2_oai_fold.
Version 6 (--sources report gpt; October 12, 2026): only the report and the GPT image reading,
no model predictions at all, to train other model families on labels that are not partly CoAtNet's own predictions.
Same rule and bounds as version 2 (10%-70%, which with two sources keeps each between 30% and 70%).
Version 7 ("v7", round 5 in the README): the version-2 rule with the CoAtNet source being five CoAtNet folds trained
on the version-6 labels: --coatnet-set runs/coatnet_384native_srcchoice6_fold
--output-dir data/labels/source_choice_v7_own_coatnet.
Because the expert studies chose the weights, their scores for runs on these labels are partly flattering; the
leaderboard is the check. Expert-holdout and unlabeled rows keep their GPT-mixed labels (round 2, from
prepare_gpt_mixed_labels.py) unchanged.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from prepare_reader_model_labels import coatnet_predictions  # noqa: E402
from prepare_student_labels import PREDICTIONS_FILE, to_label_scale  # noqa: E402
from sklearn.metrics import roc_auc_score

from rsna_knee_abnormality_detection.constants import STUDY_ID, TARGETS
from rsna_knee_abnormality_detection.utils import file_hash, write_json

REPORT_TRAINING = Path("data/labels/avg_v2/labels.csv")
GPT_TRAINING = Path("artifacts/gpt-image-labels/v1/labels-gpt-cv-medium.csv")
UNCHANGED_SOURCE = Path("data/labels/student_v2_gpt_full/labels.csv")
EXPERT = Path("data/labels/baseline/expert_labels.csv")
REPORT_EXPERT = Path("data/labels/pilkwang_scores/original_report_labels.csv")
GPT_EXPERT = Path("artifacts/gpt-image-labels/v1/labels-replies-medium.csv")
COATNET_EXPERT = Path("artifacts/expert58-coatnet5-normal-round3.csv")
LOWEST, HIGHEST = 0.10, 0.70
RESAMPLES, SEED = 2000, 0
SOURCES = ("report", "gpt", "coatnet")


def numeric(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, dtype={STUDY_ID: str}).set_index(STUDY_ID)[TARGETS].apply(pd.to_numeric, errors="coerce")


def bounded_weights(shares: dict[str, float], lowest: float = LOWEST, highest: float = HIGHEST) -> dict[str, float]:
    """Weights proportional to the win shares, each kept between LOWEST and HIGHEST and summing to one."""
    fixed: dict[str, float] = {}
    while True:
        free = [s for s in shares if s not in fixed]
        remaining = 1 - sum(fixed.values())
        total = sum(shares[s] for s in free)
        weights = {**fixed, **{s: remaining * (shares[s] / total if total > 0 else 1 / len(free)) for s in free}}
        outside = {s: max(weights[s] - highest, lowest - weights[s]) for s in free
                   if not lowest <= weights[s] <= highest}
        if not outside:
            return weights
        worst = max(outside, key=outside.get)  # pin one source at a time, so the rest still sums to one
        fixed[worst] = min(max(weights[worst], lowest), highest)


def set_runs(prefix: str) -> list[Path]:
    """A five-fold set named by its run-folder prefix, e.g. runs/coatnet_384_srcchoice2_fold -> folds 0-4."""
    runs = [Path(f"{prefix}{k}") for k in range(5)]
    missing = [str(run) for run in runs if not run.is_dir()]
    if missing:
        raise ValueError(f"Missing runs: {missing}")
    return runs


def load_run(run: Path) -> dict:
    """A fold model's out-of-fold predictions on the label scale, with the same checks as prepare_student_labels'
    load_teacher, except that the fold number is not needed (the 0.957 runs here have no copy of their config)."""
    provenance = json.loads((run / "provenance.json").read_text())
    weights = provenance["training_objective"]["positive_term_weights"]
    if sorted(weights) != sorted(TARGETS) or not all(np.isfinite(weights[name]) and weights[name] >= 1 for name in TARGETS):
        raise ValueError(f"{run}: positive-term weights must cover all findings and be at least 1")
    predictions = numeric(run / PREDICTIONS_FILE)
    values = predictions.to_numpy()
    if not np.isfinite(values).all() or (values < 0).any() or (values > 1).any():
        raise ValueError(f"{run}: predictions must be finite and between zero and one")
    if set(predictions.index) != set(provenance["validation_ids"]):
        raise ValueError(f"{run}: predictions must cover exactly the run's validation studies")
    if set(predictions.index) & set(provenance["training_ids"]):
        raise ValueError(f"{run}: some predicted studies were used to train this model")
    converted = pd.DataFrame({name: to_label_scale(predictions[name], weights[name]) for name in TARGETS})
    return {"run": str(run), "weights": weights, "predictions": converted}


def coatnet_from_sets(prefixes: list[str], ids: pd.Index) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Cross-validation and expert-study CoAtNet readings: each set's out-of-fold predictions (and each run's expert
    predictions) on the label scale, averaged over the sets (and over every run for the expert studies)."""
    per_set, expert = [], []
    for prefix in prefixes:
        teachers = [load_run(run) for run in set_runs(prefix)]
        frame = pd.concat([teacher["predictions"] for teacher in teachers])
        if not frame.index.is_unique or set(frame.index) != set(ids):
            raise ValueError(f"{prefix}: each CoAtNet set must cover every cross-validation study exactly once")
        per_set.append(frame.loc[ids, TARGETS])
        for teacher in teachers:
            predictions = numeric(Path(teacher["run"]) / "expert58-plain.csv")
            if set(predictions.index) & set(ids):
                raise ValueError(f"{teacher['run']}: expert predictions include cross-validation studies")
            expert.append(pd.DataFrame({name: to_label_scale(predictions[name], teacher["weights"][name])
                                        for name in TARGETS}))
    index = expert[0].index
    if any(set(frame.index) != set(index) for frame in expert):
        raise ValueError("Every run must predict the same expert studies")
    return sum(per_set) / len(per_set), sum(frame.loc[index] for frame in expert) / len(expert)


def source_weights(lowest: float = LOWEST, highest: float = HIGHEST, coatnet_expert: pd.DataFrame | None = None,
                   sources: tuple[str, ...] = SOURCES) -> dict:
    expert = numeric(EXPERT)
    report = numeric(REPORT_EXPERT)
    ids = expert.index.intersection(report.index)
    readers = {"report": report.loc[ids].fillna(0.5), "gpt": numeric(GPT_EXPERT).loc[ids].fillna(0.5)}
    if "coatnet" in sources:
        coatnet_expert = numeric(COATNET_EXPERT) if coatnet_expert is None else coatnet_expert
        if not set(ids) <= set(coatnet_expert.index):
            raise ValueError("The CoAtNet expert readings miss some expert studies")
        readers["coatnet"] = coatnet_expert.loc[ids]
    rng = np.random.default_rng(SEED)
    draws = [rng.integers(0, len(ids), len(ids)) for _ in range(RESAMPLES)]
    result = {}
    for target in TARGETS:
        y = expert.loc[ids, target].to_numpy()
        wins, counted = dict.fromkeys(sources, 0), 0
        for draw in draws:
            if y[draw].min() == y[draw].max():
                continue
            aucs = {s: roc_auc_score(y[draw], readers[s][target].to_numpy()[draw]) for s in sources}
            wins[max(aucs, key=aucs.get)] += 1
            counted += 1
        shares = {s: wins[s] / counted for s in sources}
        weights = bounded_weights(shares, lowest, highest)
        result[target] = {"auc": {s: round(roc_auc_score(y, readers[s][target].to_numpy()), 4) for s in sources},
                          "best_share": {s: round(v, 4) for s, v in shares.items()},
                          "weights": {s: round(v, 4) for s, v in weights.items()},
                          "expert_studies": len(ids), "positives": int(y.sum())}
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("data/labels/source_choice_v2"))
    parser.add_argument("--uncapped", action="store_true", help="use the plain win shares, without the 10%%/70%% bounds")
    parser.add_argument("--lowest", type=float, default=LOWEST, help="smallest weight of any source")
    parser.add_argument("--highest", type=float, default=HIGHEST, help="largest weight of any source")
    parser.add_argument("--coatnet-set", action="append", default=[],
                        help="run-folder prefix of a five-fold CoAtNet set (repeat to average sets); default: round 3")
    parser.add_argument("--sources", nargs="+", choices=SOURCES, default=list(SOURCES),
                        help="label sources to mix (version 6: report gpt)")
    args = parser.parse_args()
    sources = tuple(args.sources)
    if "report" not in sources or len(set(sources)) != len(sources) or len(sources) < 2:
        raise ValueError("Mix the report with at least one other source, each named once")
    lowest, highest = (0.0, 1.0) if args.uncapped else (args.lowest, args.highest)
    if not 0 <= lowest <= 1 / len(sources) <= highest <= 1:
        raise ValueError("Bounds must allow weights that sum to one")
    base = pd.read_csv(UNCHANGED_SOURCE, dtype={STUDY_ID: str}).set_index(STUDY_ID)
    gpt_frame = pd.read_csv(GPT_TRAINING, dtype={STUDY_ID: str}).set_index(STUDY_ID)
    ids = gpt_frame.index
    if not (base.loc[ids, "label_source"] == "report").all():
        raise ValueError("GPT studies must be the report-labelled cross-validation studies")
    report = numeric(REPORT_TRAINING).loc[ids]
    if "coatnet" not in sources:
        coatnet, coatnet_expert = None, None
    elif args.coatnet_set:
        coatnet, coatnet_expert = coatnet_from_sets(args.coatnet_set, ids)
    else:
        coatnet, coatnet_expert = coatnet_predictions(ids), None
    weights = source_weights(lowest, highest, coatnet_expert, sources)
    labels = base.copy()
    for target in TARGETS:
        gpt = pd.to_numeric(gpt_frame[target]).where(gpt_frame[f"{target}__assessable"].astype(str) == "True")
        values = {"report": report[target], "gpt": gpt, **({"coatnet": coatnet[target]} if coatnet is not None else {})}
        w = weights[target]["weights"]
        total = sum(w[s] * values[s].fillna(0) for s in sources)
        weight = sum(w[s] * values[s].notna() for s in sources)
        mixed = total / weight
        # only if the chosen sources are all missing
        fallback = (report[target] + coatnet[target]) / 2 if coatnet is not None else report[target]
        labels.loc[ids, target] = mixed.where(weight > 0, fallback).to_numpy()
        weights[target]["mean_label_before"] = round(float(base.loc[ids, target].mean()), 4)
        weights[target]["mean_label_after"] = round(float(labels.loc[ids, target].mean()), 4)
        weights[target]["mean_abs_change"] = round(float((labels.loc[ids, target] - base.loc[ids, target]).abs().mean()), 4)
    values = labels.loc[ids, TARGETS].to_numpy(dtype=float)
    if not np.isfinite(values).all() or (values < 0).any() or (values > 1).any():
        raise ValueError("Labels must be finite and between zero and one")
    others = labels.index.difference(ids)
    if not labels.loc[others].equals(base.loc[others]):
        raise ValueError("Expert and unlabeled rows must stay unchanged")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / "labels.csv"
    labels.reset_index().to_csv(output, index=False)
    inputs = [REPORT_TRAINING, GPT_TRAINING, UNCHANGED_SOURCE, EXPERT, REPORT_EXPERT, GPT_EXPERT]
    if "coatnet" not in sources:
        pass
    elif args.coatnet_set:
        runs = [run for prefix in args.coatnet_set for run in set_runs(prefix)]
        inputs += [run / name for run in runs for name in ("validation-averaged-model-mirror_tta.csv",
                                                           "expert58-plain.csv", "provenance.json")]
    else:
        inputs.append(COATNET_EXPERT)
    write_json(args.output_dir / "label_provenance.json", {
        "rule": "mix weighted by best-share" + (" (uncapped)" if args.uncapped else
                                                f", each source kept between {lowest:.0%} and {highest:.0%}"),
        "sources": list(sources),
        "coatnet_source": None if "coatnet" not in sources else
        ("average of the five-fold sets " + ", ".join(args.coatnet_set)) if args.coatnet_set
        else "round-3 CoAtNet normal and 448-crop sets (expert studies: normal set)",
        "inputs": {str(p): file_hash(p) for p in inputs},
        "resamples": RESAMPLES, "seed": SEED, "studies_relabelled": len(ids), "output_sha256": file_hash(output),
        "per_finding": weights,
    })
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
