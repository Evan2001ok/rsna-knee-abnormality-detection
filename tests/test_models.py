import pytest
import torch
from torch import nn

from rsna_knee_abnormality_detection.models import StudyModel


@pytest.fixture(autouse=True)
def single_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def inputs():
    images = torch.rand(2, 3, 2, 3, 32, 32)
    mask = torch.tensor([[True, False, True], [False, True, False]])
    metadata = torch.tensor([
        [[1, 0, 0, 1, 1], [0, 0, 0, -1, -1], [0, 0, 1, 1, 0]],
        [[0, 0, 0, -1, -1], [0, 1, 0, 1, 1], [0, 0, 0, -1, -1]],
    ], dtype=torch.float32)
    positions = torch.tensor([0.25, 0.75]).expand(2, 3, 2).clone()
    return images, mask, metadata, positions


@pytest.mark.parametrize("pooling", ["mean", "max", "attention"])
def test_logits_gradients_and_padding_does_not_reach_encoder(pooling):
    model = StudyModel({"backbone": "resnet18", "pooling": pooling, "encoder_chunk_size": 4}, pretrained=False)
    model.train()
    images, mask, metadata, positions = inputs()
    seen = []
    hook = model.encoder.register_forward_pre_hook(lambda _, args: seen.append(args[0].shape[0]))
    logits = model(images, mask, metadata, positions)
    hook.remove()
    assert logits.shape == (2, 12)
    assert torch.isfinite(logits).all()
    # Three real series with two slice groups each; padded series never enter BN.
    assert seen == [4, 2]
    nn.functional.binary_cross_entropy_with_logits(logits, torch.rand_like(logits)).backward()
    assert model.encoder.conv1.weight.grad is not None
    assert torch.isfinite(model.encoder.conv1.weight.grad).all()
    assert model.head[-1].weight.grad.abs().sum() > 0
    assert model.position_modulation[-1].weight.grad.abs().sum() > 0


@pytest.mark.parametrize("pooling", ["mean", "max", "attention"])
def test_masked_slots_cannot_change_predictions(pooling):
    model = StudyModel({"backbone": "resnet18", "pooling": pooling}, pretrained=False).eval()
    images, mask, metadata, positions = inputs()
    with torch.no_grad():
        expected = model(images, mask, metadata, positions)
        images[~mask] = float("nan")
        metadata[~mask] = float("nan")
        positions[~mask] = float("nan")
        actual = model(images, mask, metadata, positions)
    assert torch.equal(expected, actual)


def test_frozen_encoder_keeps_batch_norm_fixed_and_head_trainable():
    model = StudyModel({"backbone": "resnet18", "freeze_encoder": True}, pretrained=False)
    model.train()
    before = model.encoder.bn1.running_mean.clone()
    assert not model.encoder.training
    logits = model(*inputs())
    logits.square().mean().backward()
    assert torch.equal(before, model.encoder.bn1.running_mean)
    assert model.encoder.conv1.weight.grad is None
    assert model.head[-1].weight.grad.abs().sum() > 0


def test_inference_disables_downloads_and_checkpoint_roundtrip(monkeypatch):
    import torchvision.models._api

    def no_download(*args, **kwargs):
        raise AssertionError("Inference must not download pretrained weights")

    monkeypatch.setattr(torchvision.models._api, "load_state_dict_from_url", no_download)
    config = {"backbone": "resnet18", "pretrained": True, "pooling": "attention"}
    model = StudyModel(config, pretrained=False).eval()
    restored = StudyModel(config, pretrained=False).eval()
    restored.load_state_dict(model.state_dict())
    with torch.no_grad():
        batch = inputs()
        assert torch.equal(model(*batch), restored(*batch))


def test_rejects_entirely_missing_study():
    model = StudyModel({"backbone": "resnet18"}, pretrained=False)
    images, mask, metadata, positions = inputs()
    mask[0] = False
    with pytest.raises(ValueError, match="at least one real series"):
        model(images, mask, metadata, positions)


def test_pooling_comparison_preserves_shared_initial_weights_and_random_stream():
    torch.manual_seed(17)
    mean = StudyModel({"backbone": "resnet18", "pooling": "mean"}, pretrained=False)
    original_rng = torch.get_rng_state().clone()
    torch.manual_seed(17)
    attention = StudyModel({"backbone": "resnet18", "pooling": "attention"}, pretrained=False)
    assert torch.equal(torch.get_rng_state(), original_rng)
    attention_state = attention.state_dict()
    for name, value in mean.state_dict().items():
        assert torch.equal(value, attention_state[name]), name


@pytest.mark.parametrize("pooling", ["mean", "class_attention"])
def test_padding_the_last_encoder_chunk_for_inference_keeps_the_outputs(pooling):
    """Oct 16, 2026 speed-up: the last encoder chunk is padded to the full chunk size so a compiled encoder sees one
    shape; in evaluation mode each window is encoded on its own, so the study logits stay the same."""
    model = StudyModel({"backbone": "resnet18", "pooling": pooling, "encoder_chunk_size": 4}, pretrained=False).eval()
    images, mask, metadata, positions = inputs()
    seen = []
    hook = model.encoder.register_forward_pre_hook(lambda _, args: seen.append(args[0].shape[0]))
    with torch.no_grad():
        plain = model(images, mask, metadata, positions)
        model.pad_last_chunk = True
        padded = model(images, mask, metadata, positions)
    hook.remove()
    assert seen == [4, 2, 4, 4]  # three real series x two windows: chunks of 4 and 2, then 4 and 4 (padded)
    torch.testing.assert_close(padded, plain, rtol=0, atol=1e-6)
