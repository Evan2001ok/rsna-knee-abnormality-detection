"""Study-level metrics with explicit handling of missing and soft targets."""

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from torch.nn import functional as F

from .config import validate_positive_weighting
from .constants import STUDY_ID, TARGETS


def training_loss_weights(targets: np.ndarray, mode: str = "observed") -> tuple[np.ndarray, np.ndarray | None]:
    """Derive fixed finding weights using only the studies that will be fitted.

    For equal_finding, dividing the weighted sum over the entire training set by
    its number of studies gives the mean of the twelve per-finding mean losses.
    The same weights and study denominator apply to every minibatch. Recomputing
    weights or normalizing by weighted target counts inside a minibatch would
    optimize a different objective.
    """
    values = np.asarray(targets, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] == 0 or values.shape[1] != len(TARGETS):
        raise ValueError("Training targets must have shape (nonempty studies, 12)")
    if mode not in {"observed", "equal_finding"}:
        raise ValueError("target_weighting must be observed or equal_finding")
    known = ~np.isnan(values)
    if not np.isfinite(values[known]).all() or ((values[known] < 0) | (values[known] > 1)).any():
        raise ValueError("Known training targets must be finite values in [0,1]")
    counts = known.sum(axis=0)
    if mode == "observed":
        return counts, None
    absent = [target for target, count in zip(TARGETS, counts, strict=True) if count == 0]
    if absent:
        raise ValueError(
            "equal_finding needs observed training labels for every finding; "
            f"no training labels for: {', '.join(absent)}"
        )
    return counts, len(values) / (len(TARGETS) * counts)


def training_positive_weights(
    targets: np.ndarray, mode: str = "none", fixed_weights: dict | None = None,
) -> np.ndarray | None:
    """Use named fixed weights or derive Raptor weights from training targets."""
    validate_positive_weighting(mode, fixed_weights)
    if mode == "none":
        return None
    if mode == "fixed":
        return np.asarray([fixed_weights[target] for target in TARGETS], dtype=np.float64)
    values = np.asarray(targets, dtype=np.float64)
    counts, _ = training_loss_weights(values)
    if (counts == 0).any():
        raise ValueError("Positive weighting needs training labels for every finding")
    prevalence = np.clip(np.nanmean(values, axis=0), 0.03, 0.70)
    return np.clip((1 - prevalence) / prevalence, 1, 10)


def masked_bce_sum(logits, targets, mask, target_weights=None, positive_weights=None):
    """Return a loss sum and its denominator for exact gradient accumulation.

    Without weights, the denominator is the number of observed targets, matching
    the original objective. With fixed finding weights, it is the batch's study
    count. Add numerators and denominators over an accumulation window before
    dividing gradients; do not average individual minibatch means.
    """
    if logits.shape != targets.shape or logits.shape != mask.shape:
        raise ValueError("Logits, targets, and masks must have matching shapes")
    if mask.dtype != torch.bool:
        raise ValueError("Target masks must be boolean")
    count = int(mask.sum().item())
    if count == 0 and target_weights is None:
        raise ValueError("Batch contains no observed targets")
    observed = targets[mask]
    if not torch.isfinite(observed).all() or ((observed < 0) | (observed > 1)).any():
        raise ValueError("Observed targets must be finite values in [0,1]")
    positive = None
    if positive_weights is not None:
        positive = torch.as_tensor(positive_weights, dtype=torch.float32, device=logits.device)
        if logits.ndim != 2 or positive.shape != (logits.shape[1],):
            raise ValueError("Positive weights need one value per finding")
        if not torch.isfinite(positive).all() or (positive <= 0).any():
            raise ValueError("Positive weights must be finite and positive")
        positive = positive.expand_as(logits)[mask]
    if target_weights is None:
        loss = F.binary_cross_entropy_with_logits(
            logits[mask].float(), observed.float(), pos_weight=positive, reduction="sum",
        )
        return loss, count
    weights = torch.as_tensor(target_weights, dtype=torch.float32, device=logits.device)
    if logits.ndim != 2 or logits.shape[0] == 0 or weights.shape != (logits.shape[1],):
        raise ValueError("Weighted loss needs nonempty (studies, findings) tensors and one weight per finding")
    if not torch.isfinite(weights).all() or (weights <= 0).any():
        raise ValueError("Finding weights must be finite and positive")
    losses = F.binary_cross_entropy_with_logits(
        logits[mask].float(), observed.float(), pos_weight=positive, reduction="none",
    )
    loss = (losses * weights.expand_as(logits)[mask]).sum()
    return loss, logits.shape[0]


def pairwise_rank_loss(logits, targets, mask):
    """In-batch pairwise logistic ranking loss per finding (as in the teammates' scripts, Oct 2026).

    For every finding, each (positive, negative) pair of studies in the batch adds softplus(negative logit - positive
    logit); positives are known targets >= 0.5 and negatives known targets < 0.5. Returns the mean over all pairs, or
    zero when the batch has none.
    """
    if logits.shape != targets.shape or logits.shape != mask.shape or logits.ndim != 2:
        raise ValueError("Logits, targets, and masks must be matching (studies, findings) tensors")
    if mask.dtype != torch.bool:
        raise ValueError("Target masks must be boolean")
    positive = (mask & (targets >= 0.5)).float()
    negative = (mask & (targets < 0.5)).float()
    pairs = positive[:, None, :] * negative[None, :, :]
    if not pairs.any():
        return logits.float().sum() * 0.0
    values = logits.float()
    difference = values[:, None, :] - values[None, :, :]
    return (F.softplus(-difference) * pairs).sum() / pairs.sum()


def score_predictions(targets: np.ndarray, probabilities: np.ndarray) -> dict:
    targets = np.asarray(targets, dtype=np.float64)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    if targets.shape != probabilities.shape or targets.ndim != 2 or targets.shape[1] != len(TARGETS):
        raise ValueError("Targets and probabilities must both have shape (studies, 12)")
    if not np.isfinite(probabilities).all() or ((probabilities < 0) | (probabilities > 1)).any():
        raise ValueError("Predictions must be finite probabilities in [0,1]")
    if np.isinf(targets).any():
        raise ValueError("Infinite reference targets are invalid; only NaN represents a missing label")
    observed = np.isfinite(targets)
    if ((targets[observed] < 0) | (targets[observed] > 1)).any():
        raise ValueError("Observed targets must lie in [0,1]")
    aucs = []
    by_target = {}
    for i, name in enumerate(TARGETS):
        y = targets[:, i]
        binary = (y == 0) | (y == 1)
        positives, negatives = int((y == 1).sum()), int((y == 0).sum())
        auc = float(roc_auc_score(y[binary], probabilities[binary, i])) if positives and negatives else None
        if auc is not None:
            aucs.append(auc)
        by_target[name] = {
            "auc": auc, "positive": positives, "negative": negatives,
            "soft": int((np.isfinite(y) & ~binary).sum()), "missing": int((~np.isfinite(y)).sum()),
        }
    p = np.clip(probabilities, 1e-7, 1 - 1e-7)
    loss = -(targets[observed] * np.log(p[observed]) + (1 - targets[observed]) * np.log1p(-p[observed]))
    available = float(np.mean(aucs)) if aucs else None
    return {
        "studies": len(targets), "loss": float(loss.mean()) if loss.size else None,
        "macro_auc": available if len(aucs) == len(TARGETS) else None,
        "available_auc": available, "auc_targets": len(aucs), "per_target": by_target,
    }


def read_predictions(path) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype={STUDY_ID: str})
    if list(frame.columns) != [STUDY_ID, *TARGETS]:
        raise ValueError(f"Prediction columns must be {[STUDY_ID, *TARGETS]}")
    if frame.empty or frame[STUDY_ID].isna().any() or frame[STUDY_ID].duplicated().any():
        raise ValueError("Predictions require nonempty, unique study IDs")
    values = frame[TARGETS].to_numpy(dtype=float)
    if not np.isfinite(values).all() or ((values < 0) | (values > 1)).any():
        raise ValueError("Predictions must be finite probabilities in [0,1]")
    return frame


def evaluate_files(labels_csv, predictions_csv) -> dict:
    labels = pd.read_csv(labels_csv, dtype={STUDY_ID: str})
    predictions = read_predictions(predictions_csv)
    if labels[STUDY_ID].duplicated().any():
        raise ValueError("Duplicate label study IDs")
    missing = set(predictions[STUDY_ID]) - set(labels[STUDY_ID])
    if missing:
        raise ValueError(f"Predictions contain {len(missing)} studies without reference rows")
    reference = labels.set_index(STUDY_ID).loc[predictions[STUDY_ID], TARGETS]
    return score_predictions(reference.to_numpy(), predictions[TARGETS].to_numpy())
