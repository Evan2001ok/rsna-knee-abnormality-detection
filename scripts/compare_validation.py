#!/usr/bin/env python3
"""Compare predictions using the same resampled validation studies for every model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from rsna_knee_abnormality_detection.constants import STUDY_ID, TARGETS
from rsna_knee_abnormality_detection.metrics import read_predictions, score_predictions
from rsna_knee_abnormality_detection.utils import file_hash, write_json


def resampled_auc(targets, probabilities, counts):
    """Weighted pair counts, including half credit for tied predictions."""
    values = []
    for column in range(len(TARGETS)):
        y, p = targets[:, column], probabilities[:, column]
        known = np.flatnonzero((y == 0) | (y == 1))
        order = known[np.argsort(p[known], kind="stable")]
        starts = np.r_[0, np.flatnonzero(np.diff(p[order])) + 1]
        positive = np.add.reduceat(counts[:, order] * (y[order] == 1), starts, axis=1)
        negative = np.add.reduceat(counts[:, order] * (y[order] == 0), starts, axis=1)
        numerator = (positive * (negative.cumsum(axis=1) - 0.5 * negative)).sum(axis=1)
        denominator = positive.sum(axis=1) * negative.sum(axis=1)
        auc = np.full(len(counts), np.nan)
        np.divide(numerator, denominator, out=auc, where=denominator > 0)
        values.append(auc)
    return np.mean(values, axis=0)


def compare(labels_path, reference_path, candidates, output, samples=2000, seed=42):
    if samples < 100:
        raise ValueError("Use at least 100 resamples")
    inputs = [Path(labels_path), Path(reference_path), *map(Path, candidates)]
    if Path(output).resolve() in {path.resolve() for path in inputs}:
        raise ValueError("Comparison output must not overwrite an input")
    reference = read_predictions(reference_path)
    labels = pd.read_csv(labels_path, dtype={STUDY_ID: str})
    if labels[STUDY_ID].duplicated().any():
        raise ValueError("Duplicate reference-label study IDs")
    targets = labels.set_index(STUDY_ID).loc[reference[STUDY_ID], TARGETS].to_numpy()
    ids = set(reference[STUDY_ID])
    n = len(reference)
    counts = np.random.default_rng(seed).multinomial(n, np.full(n, 1 / n), size=samples)
    baseline = reference[TARGETS].to_numpy()
    base_metrics = score_predictions(targets, baseline)
    if base_metrics["auc_targets"] != len(TARGETS):
        raise ValueError("Every finding must have both known classes in the original validation set")
    base_auc = resampled_auc(targets, baseline, counts)
    # Check the optimized tie-aware calculation against the standard metric on
    # actual weighted resamples before using its intervals.
    for sample in range(min(5, samples)):
        expected = []
        for column in range(len(TARGETS)):
            known = (targets[:, column] == 0) | (targets[:, column] == 1)
            y, weights = targets[known, column], counts[sample, known]
            if weights[y == 0].sum() == 0 or weights[y == 1].sum() == 0:
                expected.append(np.nan)
            else:
                expected.append(roc_auc_score(y, baseline[known, column], sample_weight=weights))
        np.testing.assert_allclose(base_auc[sample], np.mean(expected), atol=1e-12, rtol=0, equal_nan=True)
    report = {
        "reference": str(reference_path), "reference_auc": base_metrics["macro_auc"],
        "labels_sha256": file_hash(labels_path), "reference_sha256": file_hash(reference_path),
        "method": "Paired percentile bootstrap of validation studies; same resampling counts for every model and finding.",
        "samples": samples, "seed": seed, "studies": n,
        "limitations": "Intervals assume studies are independent and are conditional on these trained checkpoints. They do not account for selecting epochs or variants using this fold, label errors, or site shift.",
        "comparisons": [],
    }
    for path in candidates:
        frame = read_predictions(path).set_index(STUDY_ID)
        if set(frame.index) != ids:
            raise ValueError(f"Candidate studies differ from the reference: {path}")
        values = frame.loc[reference[STUDY_ID], TARGETS].to_numpy()
        metrics = score_predictions(targets, values)
        deltas = resampled_auc(targets, values, counts) - base_auc
        finite = np.isfinite(deltas)
        if finite.sum() < samples * 0.9:
            raise ValueError("Too many resamples lack positive or negative examples for a finding")
        report["comparisons"].append({
            "predictions": str(path), "sha256": file_hash(path), "auc": metrics["macro_auc"],
            "auc_difference": metrics["macro_auc"] - base_metrics["macro_auc"],
            "paired_95_percent_interval": np.quantile(deltas[finite], [0.025, 0.975]).tolist(),
            "valid_resamples": int(finite.sum()),
            "per_finding_difference": {name: metrics["per_target"][name]["auc"] - base_metrics["per_target"][name]["auc"] for name in TARGETS},
        })
    write_json(output, report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", type=Path, default=Path("data/labels/baseline/labels.csv"))
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=2000)
    args = parser.parse_args()
    report = compare(args.labels, args.reference, args.predictions, args.output, args.samples)
    print(json.dumps({"reference_auc": report["reference_auc"], "comparisons": [
        {key: value for key, value in row.items() if key not in {"per_finding_difference", "sha256"}}
        for row in report["comparisons"]]}, indent=2))


if __name__ == "__main__":
    main()
