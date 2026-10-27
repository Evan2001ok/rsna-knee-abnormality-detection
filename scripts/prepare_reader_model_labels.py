#!/usr/bin/env python3
"""Combine three readers into one label per study and finding by estimating how reliable each reader is.

Readers, for every cross-validation study:
  report   the severity-aware report label (data/labels/baseline_severity, from prepare_severity_validation_labels.py):
           yes, no, or silent;
  gpt      the GPT-6 Astra image reading (artifacts/gpt-image-labels/v1/labels-gpt-cv-medium.csv, written by
           gpt_image_labels.py parse --replies gpt-cv-medium), a probability;
  coatnet  the 384 px CoAtNet prediction (models trained on the round-3 labels) from the fold models that never
           trained on the study (average of the normal and 448-crop sets), converted back to the label scale.
For each finding separately, a latent-class model (Dawid-Skene) treats the true answer as unknown and estimates, from
how the readers agree and disagree across all 4,296 studies, how each reader's answers are distributed when the
finding is present and when it is absent (the report's "silent" is one of its answers; the probabilities are split
into eight bins by quantile). The label is then each study's estimated chance that the finding is present. CoAtNet
learned from labels built from the reports and GPT, so its mistakes partly copy theirs; it counts as half a reader.
The 58 expert studies are not read. Expert-holdout and unlabeled rows keep their GPT-mixed labels (round 2, from
prepare_gpt_mixed_labels.py) unchanged.
We did not use these labels: they came out almost all 0 or 1 and drifted from the host's severity definitions (MCL
prevalence 0.09 -> 0.26). prepare_source_choice_labels.py still uses coatnet_predictions from this file.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from prepare_student_labels import load_teacher  # noqa: E402

from rsna_knee_abnormality_detection.constants import STUDY_ID, TARGETS
from rsna_knee_abnormality_detection.utils import file_hash, write_json

SEVERITY = Path("data/labels/baseline_severity/labels.csv")
GPT = Path("artifacts/gpt-image-labels/v1/labels-gpt-cv-medium.csv")
UNCHANGED_SOURCE = Path("data/labels/student_v2_gpt_full/labels.csv")
CLOUD = Path("artifacts/cloud-results")
COATNET_RUNS = ([CLOUD / f"coatnet_384_round3_fold{k}" for k in range(4)] + [Path("runs/coatnet_384_round3_fold4")]
                + [CLOUD / f"coatnet_384crop448_round3_fold{k}" for k in range(5)])
READER_WEIGHTS = {"report": 1.0, "gpt": 1.0, "coatnet": 0.5}
BINS = 8


def quantile_bins(values: pd.Series) -> pd.Series:
    """Eight equal-count bins per finding (missing values stay missing)."""
    ranks = values.rank(method="average", pct=True)
    return np.minimum((ranks * BINS).fillna(-1).astype(int), BINS - 1).where(values.notna())


def fit_reader_model(answers: dict[str, pd.Series], start: pd.Series, weights: dict[str, float],
                     iterations: int = 500) -> tuple[pd.Series, dict]:
    """EM for one finding. `answers` maps reader -> categorical answers (NaN = missing); `start` is an initial
    chance of presence per study. Returns the final chance of presence and each reader's answer distributions."""
    posterior = start.clip(0.01, 0.99)
    for _ in range(iterations):
        prior = float(posterior.mean())
        tables, log_odds = {}, np.full(len(posterior), np.log(prior / (1 - prior)))
        for reader, answer in answers.items():
            seen = answer.notna()
            categories = sorted(answer[seen].unique())
            present = np.array([posterior[seen & (answer == c)].sum() + 1.0 for c in categories])
            absent = np.array([(1 - posterior[seen & (answer == c)]).sum() + 1.0 for c in categories])
            present, absent = present / present.sum(), absent / absent.sum()
            tables[reader] = {str(c): {"if_present": round(float(p), 4), "if_absent": round(float(a), 4)}
                              for c, p, a in zip(categories, present, absent, strict=True)}
            ratio = dict(zip(categories, np.log(present / absent), strict=True))
            log_odds = log_odds + weights[reader] * answer.map(ratio).fillna(0.0).to_numpy()
        updated = pd.Series(1 / (1 + np.exp(-log_odds)), index=posterior.index)
        change = float((updated - posterior).abs().max())
        posterior = updated
        if change < 1e-7:
            break
    return posterior, {"prevalence": round(float(posterior.mean()), 4), "readers": tables}


def coatnet_predictions(ids: pd.Index) -> pd.DataFrame:
    runs = [load_teacher(run) for run in COATNET_RUNS]
    per_set = []
    for runs_of_set in (runs[:5], runs[5:]):
        frame = pd.concat([run["predictions"] for run in runs_of_set])
        if not frame.index.is_unique or set(frame.index) != set(ids):
            raise ValueError("Each CoAtNet set must cover every cross-validation study exactly once")
        per_set.append(frame.loc[ids, TARGETS])
    return (per_set[0] + per_set[1]) / 2


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("data/labels/reader_model_v1"))
    args = parser.parse_args()
    base = pd.read_csv(UNCHANGED_SOURCE, dtype={STUDY_ID: str}).set_index(STUDY_ID)
    severity = pd.read_csv(SEVERITY, dtype={STUDY_ID: str}).set_index(STUDY_ID)
    gpt = pd.read_csv(GPT, dtype={STUDY_ID: str}).set_index(STUDY_ID)
    ids = gpt.index
    if not (severity.loc[ids, "label_source"] == "report").all() or not (base.loc[ids, "label_source"] == "report").all():
        raise ValueError("GPT studies must be the report-labelled cross-validation studies")
    coatnet = coatnet_predictions(ids)
    labels = base.copy()
    summary = {}
    for target in TARGETS:
        report = severity.loc[ids, target].astype(float).map({0.0: "no", 1.0: "yes"}).fillna("silent")
        gpt_value = pd.to_numeric(gpt[target]).where(gpt[f"{target}__assessable"].astype(str) == "True")
        answers = {"report": report, "gpt": quantile_bins(gpt_value), "coatnet": quantile_bins(coatnet[target])}
        start = pd.concat([report.map({"yes": 1.0, "no": 0.0}), gpt_value, coatnet[target]], axis=1).mean(axis=1)
        posterior, fitted = fit_reader_model(answers, start, READER_WEIGHTS)
        labels.loc[ids, target] = posterior.to_numpy()
        fitted["gpt_bin_upper_edges"] = [round(float(x), 4) for x in gpt_value.quantile(np.arange(1, BINS + 1) / BINS)]
        fitted["coatnet_bin_upper_edges"] = [round(float(x), 4) for x in coatnet[target].quantile(np.arange(1, BINS + 1) / BINS)]
        fitted["mean_label_before"] = round(float(base.loc[ids, target].mean()), 4)
        fitted["mean_label_after"] = round(float(posterior.mean()), 4)
        fitted["mean_abs_change"] = round(float((posterior - base.loc[ids, target]).abs().mean()), 4)
        fitted["share_near_certain"] = round(float(((posterior < 0.05) | (posterior > 0.95)).mean()), 4)
        summary[target] = fitted
    values = labels.loc[ids, TARGETS].to_numpy(dtype=float)
    if not np.isfinite(values).all() or (values < 0).any() or (values > 1).any():
        raise ValueError("Labels must be finite and between zero and one")
    others = labels.index.difference(ids)
    if not labels.loc[others].equals(base.loc[others]):
        raise ValueError("Expert and unlabeled rows must stay unchanged")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / "labels.csv"
    labels.reset_index().to_csv(output, index=False)
    write_json(args.output_dir / "label_provenance.json", {
        "rule": __doc__.strip().splitlines()[0], "reader_weights": READER_WEIGHTS, "bins": BINS,
        "inputs": {str(path): file_hash(path) for path in (SEVERITY, GPT, UNCHANGED_SOURCE)},
        "coatnet_runs": [str(run) for run in COATNET_RUNS], "studies_relabelled": len(ids),
        "output_sha256": file_hash(output), "per_finding": summary,
    })
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
