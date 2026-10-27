import copy

import pytest
import torch
from torch.nn import functional as F

from rsna_knee_abnormality_detection.config import DEFAULTS, validate_config
from rsna_knee_abnormality_detection.metrics import pairwise_rank_loss


def test_matches_hand_computation():
    logits = torch.tensor([[2.0, 0.0], [0.5, 1.0], [-1.0, 3.0]])
    targets = torch.tensor([[1.0, 0.2], [0.0, 0.9], [0.4, 0.6]])
    mask = torch.ones_like(targets, dtype=torch.bool)
    # Finding 0: positive study 0 against negatives 1 and 2; finding 1: positives 1 and 2 against negative 0.
    pairs = [(0.5 - 2.0), (-1.0 - 2.0), (0.0 - 1.0), (0.0 - 3.0)]
    expected = torch.stack([F.softplus(torch.tensor(d)) for d in pairs]).mean()
    assert torch.allclose(pairwise_rank_loss(logits, targets, mask), expected)


def test_masked_cells_and_no_pairs():
    logits = torch.tensor([[2.0, 0.0], [0.5, 1.0]])
    targets = torch.tensor([[1.0, 1.0], [0.0, 1.0]])
    mask = torch.tensor([[True, True], [False, True]])
    loss = pairwise_rank_loss(logits, targets, mask)  # finding 0's negative is masked; finding 1 has no negatives
    assert loss.item() == 0.0 and loss.requires_grad is False


def test_gradient_pushes_positive_above_negative():
    logits = torch.zeros(2, 1, requires_grad=True)
    pairwise_rank_loss(logits, torch.tensor([[1.0], [0.0]]), torch.ones(2, 1, dtype=torch.bool)).backward()
    assert logits.grad[0, 0] < 0 < logits.grad[1, 0]


def test_config_weight_validation():
    config = copy.deepcopy(DEFAULTS)
    assert config["training"]["rank_loss_weight"] == 0.0
    validate_config(config)
    for bad in (-0.1, True, "0.1", 10):
        config["training"]["rank_loss_weight"] = bad
        with pytest.raises(ValueError):
            validate_config(config)
