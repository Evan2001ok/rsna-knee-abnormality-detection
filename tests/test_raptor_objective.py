"""Check the public soft-label objective independently of a model architecture."""

from copy import deepcopy

import numpy as np
import pytest
import torch
from torch.nn import functional as F

from rsna_knee_abnormality_detection.config import DEFAULTS, load_config, validate_config, write_config
from rsna_knee_abnormality_detection.constants import TARGETS
from rsna_knee_abnormality_detection.metrics import masked_bce_sum, training_positive_weights


def test_positive_weighting_uses_soft_prevalence_and_public_caps():
    targets = np.tile([0.01, 0.05, 0.2, 0.8] * 3, (5, 1))
    expected = np.tile([10.0, 10.0, 4.0, 1.0], 3)
    np.testing.assert_allclose(training_positive_weights(targets, "raptor"), expected)
    targets[:, 0] = np.nan
    with pytest.raises(ValueError, match="every finding"):
        training_positive_weights(targets, "raptor")


def test_masked_positive_bce_and_partial_accumulation_match_full_objective():
    generator = torch.Generator().manual_seed(83)
    logits = torch.randn(5, 12, generator=generator, requires_grad=True)
    targets = torch.rand(5, 12, generator=generator)
    mask = torch.rand(5, 12, generator=generator) > 0.25
    weights = torch.linspace(1, 10, 12)
    expected = F.binary_cross_entropy_with_logits(logits, targets, pos_weight=weights, reduction="none")[mask].mean()
    numerator, denominator = 0, 0
    for start in range(0, 5, 2):
        n, d = masked_bce_sum(logits[start:start + 2], targets[start:start + 2], mask[start:start + 2], positive_weights=weights)
        numerator, denominator = numerator + n, denominator + d
    actual = numerator / denominator
    torch.testing.assert_close(actual, expected)
    expected_grad, = torch.autograd.grad(expected, logits, retain_graph=True)
    actual_grad, = torch.autograd.grad(actual, logits)
    torch.testing.assert_close(actual_grad, expected_grad)
    assert torch.equal(actual_grad[~mask], torch.zeros_like(actual_grad[~mask]))


def test_fixed_positive_weights_roundtrip_and_ignore_current_prevalence(tmp_path):
    path = tmp_path / "weights.yaml"
    path.write_text("{}\n")
    config = load_config(path)
    assert config["training"]["fixed_positive_weights"] is None
    assert training_positive_weights(np.ones((3, len(TARGETS))), "none") is None
    # The insertion order intentionally differs from the model's target order.
    expected = np.arange(len(TARGETS), dtype=float) / 4 + 0.75
    weights = dict(reversed(list(zip(TARGETS, expected.tolist(), strict=True))))
    config["training"].update(positive_weighting="fixed", fixed_positive_weights=weights)
    write_config(config, path)
    loaded = load_config(path)
    assert loaded["training"]["fixed_positive_weights"] == weights
    for targets in [np.full((3, len(TARGETS)), 0.05), np.full((3, len(TARGETS)), 0.95)]:
        np.testing.assert_array_equal(training_positive_weights(targets, "fixed", weights), expected)
    # Missing observations cannot cause fixed weights to be estimated from other rows.
    np.testing.assert_array_equal(
        training_positive_weights(np.full((3, len(TARGETS)), np.nan), "fixed", weights), expected,
    )


@pytest.mark.parametrize("weights", [None, {}, [1.0] * 12, {"ACL": 1.0}, {**dict.fromkeys(TARGETS, 1), "extra": 1}])
def test_fixed_positive_weights_require_exactly_all_named_targets(weights):
    config = deepcopy(DEFAULTS)
    config["training"].update(positive_weighting="fixed", fixed_positive_weights=weights)
    with pytest.raises(ValueError, match="fixed_positive_weights"):
        validate_config(config)
    with pytest.raises(ValueError, match="fixed_positive_weights"):
        training_positive_weights(np.ones((2, len(TARGETS))), "fixed", weights)


@pytest.mark.parametrize("value", [True, False, "1", None, [], {}, 1 + 2j, 0, -1, np.nan, np.inf, -np.inf, 10 ** 1000])
def test_fixed_positive_weights_reject_invalid_values(value):
    weights = dict.fromkeys(TARGETS, 1.0)
    weights["ACL"] = value
    config = deepcopy(DEFAULTS)
    config["training"].update(positive_weighting="fixed", fixed_positive_weights=weights)
    with pytest.raises(ValueError, match="fixed_positive_weights"):
        validate_config(config)
    with pytest.raises(ValueError, match="fixed_positive_weights"):
        training_positive_weights(np.ones((2, len(TARGETS))), "fixed", weights)


@pytest.mark.parametrize("mode", ["none", "raptor"])
def test_fixed_positive_weights_cannot_be_silently_ignored_by_other_modes(mode):
    weights = dict.fromkeys(TARGETS, 1)
    config = deepcopy(DEFAULTS)
    config["training"].update(positive_weighting=mode, fixed_positive_weights=weights)
    with pytest.raises(ValueError, match="requires positive_weighting fixed"):
        validate_config(config)
    with pytest.raises(ValueError, match="requires positive_weighting fixed"):
        training_positive_weights(np.ones((2, len(TARGETS))), mode, weights)


def test_named_fixed_positive_weights_match_known_loss_and_gradient():
    logits = torch.zeros(3, len(TARGETS), requires_grad=True)
    targets = torch.linspace(0, 1, logits.numel()).reshape_as(logits)
    mask = torch.ones_like(targets, dtype=torch.bool)
    mask[0, ::3] = False
    mask[2, 1::3] = False
    targets[~mask] = float("nan")
    weights = dict(reversed([(target, (index + 1) / 2) for index, target in enumerate(TARGETS)]))
    resolved = torch.tensor(training_positive_weights(targets.numpy(), "fixed", weights), dtype=torch.float32)
    numerator, denominator = masked_bce_sum(logits, targets, mask, positive_weights=resolved)
    loss = numerator / denominator
    # At zero logits, BCE is log(2) times ((1-y) + weight*y).
    positive = resolved.expand_as(logits)
    expected_loss = (torch.log(torch.tensor(2.0)) * ((1 - targets[mask]) + positive[mask] * targets[mask])).mean()
    expected_gradient = torch.zeros_like(logits)
    expected_gradient[mask] = 0.5 * ((1 - targets[mask]) - positive[mask] * targets[mask]) / int(mask.sum())
    torch.testing.assert_close(loss, expected_loss)
    loss.backward()
    torch.testing.assert_close(logits.grad, expected_gradient)
    assert torch.equal(logits.grad[~mask], torch.zeros_like(logits.grad[~mask]))
