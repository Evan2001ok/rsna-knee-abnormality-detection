"""Keep missing targets, study alignment, and reference labels unambiguous."""

import numpy as np
import pandas as pd
import pytest
import torch
from torch.nn import functional as F

from rsna_knee_abnormality_detection.constants import STUDY_ID, TARGETS
from rsna_knee_abnormality_detection.ensemble import ensemble_predictions
from rsna_knee_abnormality_detection.metrics import (
    evaluate_files,
    masked_bce_sum,
    read_predictions,
    score_predictions,
    training_loss_weights,
)


def prediction_file(path, ids, values):
    values = np.asarray(values)
    if values.ndim == 1:
        values = np.repeat(values[:, None], len(TARGETS), axis=1)
    frame = pd.DataFrame(values, columns=TARGETS)
    frame.insert(0, STUDY_ID, ids)
    frame.to_csv(path, index=False)
    return path


def test_partial_target_mask_excludes_unknown_values_and_gradients():
    logits = torch.tensor([[0.1, -0.2, 100.0], [0.5, 9.0, -0.7]], requires_grad=True)
    targets = torch.tensor([[1.0, 0.25, float("nan")], [0.0, float("nan"), 0.8]])
    mask = torch.isfinite(targets)
    loss, count = masked_bce_sum(logits, targets, mask)
    assert count == 4
    expected = F.binary_cross_entropy_with_logits(logits[mask], targets[mask], reduction="mean")
    assert torch.allclose(loss / count, expected)
    (loss / count).backward()
    assert torch.equal(logits.grad[~mask], torch.zeros(2))
    assert torch.isfinite(logits.grad).all()


def test_accumulating_unequal_label_counts_matches_one_mean_loss():
    logits = torch.tensor([0.2, -0.3, 0.7, -0.5], requires_grad=True)
    targets = torch.tensor([0.0, 0.4, 1.0, float("nan")])
    mask = torch.isfinite(targets)
    for start, end in [(0, 1), (1, 4)]:
        loss, _ = masked_bce_sum(logits[start:end], targets[start:end], mask[start:end])
        loss.backward()
    accumulated = logits.grad.clone() / mask.sum()
    logits.grad.zero_()
    whole, count = masked_bce_sum(logits, targets, mask)
    (whole / count).backward()
    torch.testing.assert_close(accumulated, logits.grad)


def partial_training_targets():
    values = np.tile(np.linspace(0.0, 1.0, len(TARGETS)), (5, 1))
    values[0, 2:] = np.nan
    values[1, 4:] = np.nan
    values[2, 7:] = np.nan
    values[3, 1:3] = np.nan
    return values


def test_equal_finding_loss_and_gradients_match_mean_of_per_finding_means():
    targets = torch.tensor(partial_training_targets(), dtype=torch.float32)
    mask = torch.isfinite(targets)
    logits = torch.linspace(-1.3, 1.1, targets.numel()).reshape_as(targets).requires_grad_()
    counts, weights = training_loss_weights(targets.numpy(), "equal_finding")
    expected_counts = mask.sum(0).numpy()
    np.testing.assert_array_equal(counts, expected_counts)
    np.testing.assert_allclose(weights, len(targets) / (len(TARGETS) * expected_counts))
    weighted, denominator = masked_bce_sum(logits, targets, mask, weights)
    assert denominator == len(targets)
    (weighted / denominator).backward()
    weighted_gradients = logits.grad.clone()
    logits.grad.zero_()
    # This independent reference averages each finding's observed-label loss,
    # rather than reproducing the weighted-sum implementation.
    reference = torch.stack([
        F.binary_cross_entropy_with_logits(logits[mask[:, i], i], targets[mask[:, i], i])
        for i in range(len(TARGETS))
    ]).mean()
    reference.backward()
    torch.testing.assert_close(weighted / denominator, reference)
    torch.testing.assert_close(weighted_gradients, logits.grad)
    assert torch.equal(weighted_gradients[~mask], torch.zeros_like(weighted_gradients[~mask]))


def test_equal_finding_accumulation_with_partial_batches_uses_study_counts():
    targets = torch.tensor(partial_training_targets(), dtype=torch.float32)
    mask = torch.isfinite(targets)
    _, weights = training_loss_weights(targets.numpy(), "equal_finding")
    features = torch.tensor([[1., 2.], [3., -1.], [0.5, 0.], [-1., 1.], [0.2, -0.5]])
    parameters = torch.linspace(-0.3, 0.3, 2 * len(TARGETS)).reshape(2, len(TARGETS)).requires_grad_()
    total_studies = 0
    for start, end in [(0, 2), (2, 4), (4, 5)]:
        loss_sum, denominator = masked_bce_sum(
            features[start:end] @ parameters, targets[start:end], mask[start:end], weights,
        )
        assert denominator == end - start
        total_studies += denominator
        loss_sum.backward()
    accumulated = parameters.grad.clone() / total_studies
    parameters.grad.zero_()
    full_logits = features @ parameters
    expected = torch.stack([
        F.binary_cross_entropy_with_logits(full_logits[mask[:, i], i], targets[mask[:, i], i])
        for i in range(len(TARGETS))
    ]).mean()
    expected.backward()
    torch.testing.assert_close(accumulated, parameters.grad)


def test_weighted_minibatch_is_not_renormalized_by_its_observed_weights():
    # The rare label appears only in the last training study. Its minibatch must
    # retain its larger fixed contribution, despite a different missing-label mix.
    training = np.zeros((4, len(TARGETS)))
    training[:3, -1] = np.nan
    _, weights = training_loss_weights(training, "equal_finding")
    targets = torch.tensor(training[[0]], dtype=torch.float32)
    mask = torch.isfinite(targets)
    loss, denominator = masked_bce_sum(torch.zeros_like(targets), targets, mask, weights)
    assert denominator == 1
    torch.testing.assert_close(loss / denominator, torch.tensor(np.log(2) * 11 / 12, dtype=torch.float32))


def test_weighted_mask_ignores_unknown_logits_and_can_represent_empty_evidence():
    logits = torch.tensor([[float("nan"), 0.4], [0.8, float("nan")]], requires_grad=True)
    targets = torch.tensor([[float("nan"), 0.25], [0.7, 999.0]])
    mask = torch.tensor([[False, True], [True, False]])
    loss, denominator = masked_bce_sum(logits, targets, mask, torch.tensor([0.2, 0.8]))
    assert denominator == 2 and torch.isfinite(loss)
    (loss / denominator).backward()
    assert torch.isfinite(logits.grad).all()
    assert torch.equal(logits.grad[~mask], torch.zeros(2))
    logits.grad.zero_()
    empty_loss, denominator = masked_bce_sum(logits, targets, torch.zeros_like(mask), torch.ones(2))
    assert denominator == 2 and empty_loss.item() == 0
    empty_loss.backward()
    assert torch.equal(logits.grad, torch.zeros_like(logits))


def test_equal_finding_rejects_missing_training_finding_but_observed_keeps_old_behavior():
    targets = partial_training_targets()
    targets[:, -1] = np.nan
    counts, weights = training_loss_weights(targets, "observed")
    assert counts[-1] == 0 and weights is None
    with pytest.raises(ValueError, match="no training labels for: Fracture"):
        training_loss_weights(targets, "equal_finding")
    targets[0, -1] = float("inf")
    with pytest.raises(ValueError, match="finite"):
        training_loss_weights(targets, "equal_finding")


@pytest.mark.parametrize("weights", [[1.0], [0.0, 1.0], [-1.0, 1.0], [float("inf"), 1.0]])
def test_weighted_loss_rejects_invalid_finding_weights(weights):
    with pytest.raises(ValueError, match="weight|positive"):
        masked_bce_sum(torch.zeros(2, 2), torch.zeros(2, 2), torch.ones(2, 2, dtype=torch.bool), weights)


def test_rejects_no_observed_targets_and_invalid_observed_probabilities():
    with pytest.raises(ValueError, match="no observed"):
        masked_bce_sum(torch.zeros(2), torch.zeros(2), torch.zeros(2, dtype=torch.bool))
    with pytest.raises(ValueError, match="finite"):
        masked_bce_sum(torch.zeros(2), torch.tensor([0.0, float("nan")]), torch.ones(2, dtype=torch.bool))


def test_soft_targets_have_loss_but_are_not_rounded_for_auc():
    targets = np.full((4, len(TARGETS)), 0.35)
    probabilities = np.full_like(targets, 0.6)
    targets[:, 0] = [0.0, 1.0, 0.35, np.nan]
    probabilities[:, 0] = [0.1, 0.9, 0.95, 0.3]
    scores = score_predictions(targets, probabilities)
    assert scores["macro_auc"] is None
    assert scores["available_auc"] == 1.0
    assert scores["auc_targets"] == 1
    assert scores["per_target"]["ACL"] == {"auc": 1.0, "positive": 1, "negative": 1, "soft": 1, "missing": 1}
    assert scores["per_target"]["MCL"]["auc"] is None
    assert np.isfinite(scores["loss"])
    targets[:] = 0.35
    scores = score_predictions(targets, probabilities)
    assert scores["available_auc"] is None and scores["macro_auc"] is None


def test_macro_auc_requires_both_classes_for_every_target():
    targets = np.tile([[0], [1]], (1, len(TARGETS))).astype(float)
    probabilities = np.tile([[0.1], [0.9]], (1, len(TARGETS)))
    assert score_predictions(targets, probabilities)["macro_auc"] == 1.0
    targets[:, 4] = 0
    scores = score_predictions(targets, probabilities)
    assert scores["macro_auc"] is None
    assert scores["auc_targets"] == 11


@pytest.mark.parametrize("invalid", [float("inf"), -float("inf")])
def test_infinite_reference_labels_are_not_silently_missing(invalid):
    targets = np.zeros((2, len(TARGETS)))
    targets[0, 0] = invalid
    with pytest.raises(ValueError, match="finite|[Ii]nfinite|[Ii]nvalid"):
        score_predictions(targets, np.full_like(targets, 0.5))


def test_ensemble_aligns_ids_before_weighting_and_evaluation(tmp_path):
    first = prediction_file(tmp_path / "first.csv", ["a", "b", "c"], [0.1, 0.8, 0.4])
    second = prediction_file(tmp_path / "second.csv", ["c", "a", "b"], [0.9, 0.3, 0.6])
    labels = prediction_file(tmp_path / "labels.csv", ["b", "c", "a"], [1, 0, 0])
    output = ensemble_predictions([first, second], tmp_path / "blend.csv", weights=[3, 1], labels_csv=labels)
    result = read_predictions(output)
    assert result[STUDY_ID].tolist() == ["a", "b", "c"]
    np.testing.assert_allclose(result["ACL"], [0.15, 0.75, 0.525])
    assert evaluate_files(labels, output)["macro_auc"] == 1.0
    assert output.with_suffix(".provenance.json").exists()


def test_ensemble_target_weights_and_rank_blend_handle_ties(tmp_path):
    first = prediction_file(tmp_path / "first.csv", ["a", "b", "c"], [0.5, 0.5, 0.9])
    second = prediction_file(tmp_path / "second.csv", ["a", "b", "c"], [0.9, 0.1, 0.3])
    weights = np.zeros((2, len(TARGETS)))
    weights[0, :] = 1
    weights[:, 1] = [0, 1]
    output = ensemble_predictions([first, second], tmp_path / "rank.csv", weights=weights, method="rank")
    result = read_predictions(output)
    np.testing.assert_allclose(result["ACL"], [0.375, 0.375, 0.75])
    np.testing.assert_allclose(result["MCL"], [0.75, 0.25, 0.5])


def test_ensemble_rejects_missing_ids_duplicates_and_invalid_probabilities(tmp_path):
    first = prediction_file(tmp_path / "first.csv", ["a", "b"], [0.1, 0.8])
    missing = prediction_file(tmp_path / "missing.csv", ["a"], [0.2])
    with pytest.raises(ValueError, match="same study IDs"):
        ensemble_predictions([first, missing], tmp_path / "blend.csv")
    duplicate = prediction_file(tmp_path / "duplicate.csv", ["a", "a"], [0.1, 0.8])
    with pytest.raises(ValueError, match="unique"):
        read_predictions(duplicate)
    invalid = prediction_file(tmp_path / "invalid.csv", ["a", "b"], [0.1, 1.8])
    with pytest.raises(ValueError, match="probabilities"):
        read_predictions(invalid)


def test_ensemble_cannot_overwrite_reference_labels(tmp_path):
    first = prediction_file(tmp_path / "first.csv", ["a", "b"], [0.1, 0.8])
    labels = prediction_file(tmp_path / "labels.csv", ["a", "b"], [0, 1])
    original = labels.read_bytes()
    with pytest.raises(ValueError, match="overwrite|[Ii]nput|reference"):
        ensemble_predictions([first], labels, labels_csv=labels)
    assert labels.read_bytes() == original
