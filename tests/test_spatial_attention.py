"""CPU checks for the optional per-finding attention inside each window (model.spatial_attention)."""

import copy
import sys
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from rsna_knee_abnormality_detection.config import DEFAULTS, validate_config
from rsna_knee_abnormality_detection.constants import TARGETS
from rsna_knee_abnormality_detection.models import RAPTOR_BACKBONE, StudyModel


@pytest.fixture
def tiny_timm(monkeypatch):
    class TinyEncoder(nn.Module):
        num_features = 8

        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(3, self.num_features, 4, stride=4)

        def forward_features(self, images):
            return self.conv(images)  # [n, channels, 8, 8] for 32 px input

        def forward(self, images):
            return self.forward_features(images).mean(dim=(-2, -1))  # timm's average-pool head

    monkeypatch.setitem(sys.modules, "timm", SimpleNamespace(create_model=lambda name, **kwargs: TinyEncoder()))


def config(**overrides):
    return {"backbone": RAPTOR_BACKBONE, "pooling": "class_attention", "input_size": 32, "encoder_chunk_size": 4,
            "dropout": 0.0, **overrides}


def batch():
    torch.manual_seed(0)
    images = torch.rand(2, 3, 2, 3, 32, 32)
    mask = torch.tensor([[True, False, True], [False, True, False]])
    return images, mask, torch.zeros(2, 3, 5)


def test_zero_initialised_attention_matches_plain_model(tiny_timm):
    torch.manual_seed(1)
    plain = StudyModel(config(), pretrained=False).eval()
    torch.manual_seed(1)
    spatial = StudyModel(config(spatial_attention=True), pretrained=False).eval()
    missing = spatial.load_state_dict(plain.state_dict(), strict=False).missing_keys
    assert sorted(missing) == ["spatial_scores.0.bias", "spatial_scores.0.weight", "spatial_scores.2.bias",
                               "spatial_scores.2.weight"]
    images, mask, metadata = batch()
    with torch.no_grad():
        assert torch.allclose(plain(images, mask, metadata), spatial(images, mask, metadata), atol=1e-6)


def test_attention_learns_and_changes_predictions(tiny_timm):
    model = StudyModel(config(spatial_attention=True), pretrained=False).train()
    images, mask, metadata = batch()
    model(images, mask, metadata).sum().backward()
    assert model.spatial_scores[-1].weight.grad.abs().sum() > 0
    before = model.eval()(images, mask, metadata).detach()
    with torch.no_grad():
        model.spatial_scores[-1].weight.normal_()
    after = model(images, mask, metadata).detach()
    assert after.shape == (2, len(TARGETS)) and torch.isfinite(after).all() and not torch.allclose(before, after)


def test_masked_series_cannot_change_prediction(tiny_timm):
    model = StudyModel(config(spatial_attention=True), pretrained=False).eval()
    with torch.no_grad():
        model.spatial_scores[-1].weight.normal_()
    images, mask, metadata = batch()
    changed = images.clone()
    changed[~mask] = torch.rand_like(changed[~mask]) * 100
    with torch.no_grad():
        assert torch.allclose(model(images, mask, metadata), model(changed, mask, metadata), atol=1e-6)


def test_config_validation():
    settings = copy.deepcopy(DEFAULTS)
    assert settings["model"]["spatial_attention"] is False
    validate_config(settings)
    settings["model"]["spatial_attention"] = True  # resnet34 with mean pooling
    with pytest.raises(ValueError):
        validate_config(settings)
    settings["model"]["spatial_attention"] = "yes"
    with pytest.raises(ValueError):
        validate_config(settings)
