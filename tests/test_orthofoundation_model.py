"""OrthoFoundation-L encoder: built empty, loaded strictly from the local release, never downloaded."""

import sys
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from rsna_knee_abnormality_detection import models
from rsna_knee_abnormality_detection.config import DEFAULTS, validate_config
from rsna_knee_abnormality_detection.models import ORTHOFOUNDATION_BACKBONE, ORTHOFOUNDATION_TIMM_NAME, StudyModel


class TinyViT(nn.Module):
    num_features = 8

    def __init__(self):
        super().__init__()
        self.patch = nn.Conv2d(3, self.num_features, 16, stride=16)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.num_features))
        self.checkpointing = False

    def set_grad_checkpointing(self, enabled):
        self.checkpointing = enabled

    def forward(self, images):
        return self.patch(images).mean(dim=(-2, -1)) + self.cls_token[:, 0]


@pytest.fixture
def tiny_timm(monkeypatch):
    calls = []

    def create_model(name, **kwargs):
        calls.append((name, kwargs))
        return TinyViT()

    monkeypatch.setitem(sys.modules, "timm", SimpleNamespace(create_model=create_model))
    monkeypatch.setattr(models, "_dinov3_filter", lambda state, model: state)
    return calls


def config(tmp_path=None, **overrides):
    return {"backbone": ORTHOFOUNDATION_BACKBONE, "pooling": "class_attention", "input_size": 32, "encoder_chunk_size": 4,
            "use_metadata": False, "use_positions": False,
            "pretrained_checkpoint": str(tmp_path / "release.pth") if tmp_path else None, **overrides}


def release(tmp_path, extra=None):
    state = {f"backbone.{key}": torch.full_like(value, 0.5) for key, value in TinyViT().state_dict().items()}
    torch.save({**state, **(extra or {})}, tmp_path / "release.pth")


def test_release_weights_load_strictly_after_removing_the_prefix(tiny_timm, tmp_path):
    release(tmp_path)
    model = StudyModel(config(tmp_path, grad_checkpointing=True), pretrained=True)
    name, kwargs = tiny_timm[0]
    assert name == ORTHOFOUNDATION_TIMM_NAME and kwargs["pretrained"] is False and kwargs["global_pool"] == "token"
    assert kwargs["img_size"] == 32 and kwargs["num_classes"] == 0
    assert torch.all(model.encoder.patch.weight == 0.5) and model.encoder.checkpointing and model.feature_dim == 8
    release(tmp_path, {"backbone.unexpected": torch.zeros(1)})
    with pytest.raises(RuntimeError, match="unexpected"):
        StudyModel(config(tmp_path), pretrained=True)


def test_trained_checkpoints_reload_without_the_release_file(tiny_timm, tmp_path):
    release(tmp_path)
    model = StudyModel(config(tmp_path), pretrained=True)
    (tmp_path / "release.pth").unlink()
    reloaded = StudyModel(config(tmp_path), pretrained=False)
    reloaded.load_state_dict(model.state_dict(), strict=True)
    images = torch.rand(1, 1, 2, 3, 32, 32)
    inputs = (images, torch.ones(1, 1, dtype=torch.bool), torch.zeros(1, 1, 5), torch.zeros(1, 1, 2))
    torch.testing.assert_close(reloaded.eval()(*inputs), model.eval()(*inputs))
    with pytest.raises(ValueError, match="pretrained_checkpoint"):
        StudyModel(config(), pretrained=True)


@pytest.mark.parametrize("size", [0, 30, True])
def test_input_size_must_be_a_positive_multiple_of_16(tiny_timm, size):
    with pytest.raises(ValueError, match="multiple of 16"):
        StudyModel(config(input_size=size), pretrained=False)


def test_config_requires_matching_image_and_input_sizes():
    settings = {**DEFAULTS, "data": {**DEFAULTS["data"], "image_size": 224},
                "model": {**DEFAULTS["model"], "backbone": ORTHOFOUNDATION_BACKBONE, "input_size": 224}}
    validate_config(settings)
    with pytest.raises(ValueError, match="ViT input_size"):
        validate_config({**settings, "model": {**settings["model"], "input_size": 256}})


@pytest.mark.parametrize("name, timm_name", sorted(models.DINOV3_BACKBONES.items()))
def test_dinov3_encoders_use_timm_weights_only_when_pretrained(tiny_timm, name, timm_name):
    StudyModel(config(backbone=name), pretrained=True)
    StudyModel(config(backbone=name), pretrained=False)
    assert [call[0] for call in tiny_timm] == [timm_name, timm_name]
    assert [call[1]["pretrained"] for call in tiny_timm] == [True, False]
    assert all(call[1]["global_pool"] == "token" and call[1]["img_size"] == 32 for call in tiny_timm)


def test_resnet50_is_available_without_downloading_when_not_pretrained():
    model = StudyModel({"backbone": "resnet50", "pooling": "class_attention", "use_metadata": False, "use_positions": False},
                       pretrained=False)
    assert model.feature_dim == 2048
