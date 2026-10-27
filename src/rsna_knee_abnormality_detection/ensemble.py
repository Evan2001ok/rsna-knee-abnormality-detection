"""Blend aligned predictions without fitting weights on the public leaderboard."""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from .constants import STUDY_ID, TARGETS
from .metrics import evaluate_files, read_predictions
from .utils import file_hash, write_json


def ensemble_predictions(inputs, output_csv, weights=None, method="mean", labels_csv=None) -> Path:
    if not inputs or method not in {"mean", "rank"}:
        raise ValueError("Supply predictions and choose mean or rank blending")
    output = Path(output_csv)
    if output.resolve() in {Path(path).resolve() for path in inputs}:
        raise ValueError("Ensemble output must not overwrite an input")
    if labels_csv and output.resolve() == Path(labels_csv).resolve():
        raise ValueError("Ensemble output must not overwrite reference labels")
    frames = [read_predictions(path) for path in inputs]
    ids = frames[0][STUDY_ID].tolist()
    arrays = []
    for frame in frames:
        if set(frame[STUDY_ID]) != set(ids):
            raise ValueError("Every prediction file must contain exactly the same study IDs")
        values = frame.set_index(STUDY_ID).loc[ids, TARGETS].to_numpy()
        if method == "rank":
            values = rankdata(values, axis=0, method="average") / (len(ids) + 1)
        arrays.append(values)
    w = np.asarray(weights if weights is not None else np.ones(len(arrays)), dtype=float)
    if w.shape == (len(arrays),):
        w = np.repeat(w[:, None], len(TARGETS), axis=1)
    if w.shape != (len(arrays), len(TARGETS)) or not np.isfinite(w).all() or (w < 0).any() or (w.sum(axis=0) <= 0).any():
        raise ValueError("Weights must be nonnegative, with one weight per model or model/target and positive sums")
    w /= w.sum(axis=0, keepdims=True)
    blend = (np.stack(arrays) * w[:, None, :]).sum(axis=0)
    frame = pd.DataFrame(blend, columns=TARGETS)
    frame.insert(0, STUDY_ID, ids)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False)
    report = {
        "method": method, "targets": TARGETS, "weights": w.tolist(),
        "sources": [{"path": str(path), "sha256": file_hash(path)} for path in inputs],
    }
    if labels_csv:
        report["validation"] = evaluate_files(labels_csv, output)
        report["individual_validation"] = [evaluate_files(labels_csv, path) for path in inputs]
    write_json(output.with_suffix(".provenance.json"), report)
    return output
