import json
from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from rsna_knee_abnormality_detection.diagnostics import recalibrate_batchnorm
from rsna_knee_abnormality_detection.models import StudyModel


@pytest.fixture(autouse=True)
def two_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


class TinyStudy(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(nn.BatchNorm2d(3, momentum=0.23), nn.Dropout2d(0.8))
        self.head = nn.Linear(3, 12)
        self.encoder_chunk_size = 32
        self.seen = []

    def forward(self, images, series_mask, metadata, positions=None):
        self.seen.append((self.training, self.encoder[0].training, self.encoder[1].training))
        real = images[series_mask].flatten(0, 1)
        features = torch.cat([self.encoder(chunk) for chunk in real.split(self.encoder_chunk_size)])
        pooled = features.mean(dim=(0, 2, 3)).unsqueeze(0).expand(images.shape[0], -1)
        return self.head(pooled)


def full_batch():
    values = torch.arange(48).reshape(2, 3, 8, 1, 1, 1).float()
    images = values.expand(2, 3, 8, 3, 2, 2).clone()
    images[:, :, :, 1] += 0.5
    images[:, :, :, 2] += 1.0
    return {
        "images": images, "series_mask": torch.ones(2, 3, dtype=torch.bool),
        "metadata": torch.zeros(2, 3, 5), "positions": torch.zeros(2, 3, 8),
        "study_id": ["train-a", "train-b"], "role": ["training", "training"],
    }


def partial_batch():
    images = torch.full((1, 3, 8, 3, 2, 2), float("nan"))
    images[:, 0] = 100
    return {
        "images": images, "series_mask": torch.tensor([[True, False, False]]),
        "metadata": torch.zeros(1, 3, 5), "positions": torch.zeros(1, 3, 8),
        "study_id": ["train-c"], "role": ["training"],
    }


def snapshot(model):
    return {name: value.clone() for name, value in model.state_dict().items()}


def test_recalibration_weights_observations_and_restores_every_training_setting():
    model = TinyStudy().train()
    model.encoder[0].eval()
    model.encoder[0].running_mean.fill_(17)
    model.encoder[0].running_var.fill_(3)
    model.encoder[0].num_batches_tracked.fill_(99)
    model.encoder[0].weight.grad = torch.tensor([0.1, 0.2, 0.3])
    original_parameters = {name: value.clone() for name, value in model.named_parameters()}
    original_gradient = model.encoder[0].weight.grad.clone()
    original_modes = [module.training for module in model.modules()]
    batches = [full_batch(), partial_batch()]
    originals = deepcopy(batches)
    report = recalibrate_batchnorm(model, batches, "cpu", "fp32")
    json.dumps(report, allow_nan=False)
    values = torch.cat([batch["images"][batch["series_mask"]].flatten(0, 1) for batch in batches])
    expected = values.permute(1, 0, 2, 3).reshape(3, -1)
    torch.testing.assert_close(model.encoder[0].running_mean, expected.mean(dim=1))
    torch.testing.assert_close(model.encoder[0].running_var, expected.var(dim=1, correction=1))
    assert model.encoder[0].num_batches_tracked == 2
    assert report["batches"] == 2 and report["studies"] == 3
    assert report["series"] == 7 and report["image_groups"] == 56
    assert report["images_per_batch"] == [48, 8]
    assert report["encoder_chunk_sizes_used"] == [32, 48]
    assert report["batchnorm"]["encoder.0"] == {"forward_calls": 2, "observations_per_channel": 224}
    assert report["study_ids"] == ["train-a", "train-b", "train-c"]
    assert report["batches_with_training_roles"] == 2
    assert report["parameters_unchanged"] and report["gradients_unchanged"]
    assert model.encoder_chunk_size == 32 and model.encoder[0].momentum == 0.23
    assert [module.training for module in model.modules()] == original_modes
    assert model.seen == [(False, True, False), (False, True, False)]
    for name, value in model.named_parameters():
        assert torch.equal(value, original_parameters[name])
    assert torch.equal(model.encoder[0].weight.grad, original_gradient)
    assert model.head.weight.grad is None
    for before, after in zip(originals, batches, strict=True):
        for key in ["images", "series_mask", "metadata", "positions"]:
            torch.testing.assert_close(before[key], after[key], equal_nan=True)


@pytest.mark.parametrize("problem", ["empty", "validation", "nonfinite", "no_series", "duplicate_id"])
def test_invalid_loader_rolls_back_partial_recalibration(problem):
    model = TinyStudy().train()
    model.encoder[0].running_mean.fill_(7)
    model.encoder[0].running_var.fill_(2)
    model.encoder[0].num_batches_tracked.fill_(13)
    before = snapshot(model)
    modes = [module.training for module in model.modules()]
    invalid = partial_batch()
    batches = [full_batch(), invalid]
    if problem == "empty":
        batches = []
    elif problem == "validation":
        invalid["role"] = ["validation"]
    elif problem == "nonfinite":
        invalid["images"][:, 0] = float("nan")
    elif problem == "no_series":
        invalid["series_mask"].fill_(False)
    elif problem == "duplicate_id":
        invalid["study_id"] = ["train-a"]
    with pytest.raises(ValueError):
        recalibrate_batchnorm(model, batches, "cpu", "fp32")
    for name, expected in before.items():
        assert torch.equal(model.state_dict()[name], expected), name
    assert [module.training for module in model.modules()] == modes
    assert model.encoder[0].momentum == 0.23 and model.encoder_chunk_size == 32
    assert not model.encoder[0]._forward_pre_hooks
    assert all(parameter.grad is None for parameter in model.parameters())


def test_rejects_augmentation_and_random_training_selection():
    class Loader(list):
        dataset = SimpleNamespace(training=True)

    with pytest.raises(ValueError, match="training=False"):
        recalibrate_batchnorm(TinyStudy(), Loader([full_batch()]), "cpu", "fp32")


def test_real_resnet_calibration_keeps_parameters_fixed_and_buffers_finite():
    model = StudyModel({"backbone": "resnet18", "pooling": "attention"}, pretrained=False).eval()
    original = {name: parameter.clone() for name, parameter in model.named_parameters()}
    example = full_batch()
    example["images"] = torch.rand(2, 3, 8, 3, 32, 32)
    report = recalibrate_batchnorm(model, [example], "cpu", "bf16")
    assert report["precision_used"] == "fp32"
    assert report["encoder_chunk_sizes_used"] == [48]
    assert len(report["batchnorm"]) == 20
    for name, parameter in model.named_parameters():
        assert torch.equal(parameter, original[name])
        assert parameter.grad is None
    for module in model.modules():
        assert not module.training
        if isinstance(module, nn.BatchNorm2d):
            assert torch.isfinite(module.running_mean).all()
            assert torch.isfinite(module.running_var).all()
            assert (module.running_var >= 0).all()
            assert module.num_batches_tracked == 1
