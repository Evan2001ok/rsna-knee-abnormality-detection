"""CPU checks for the public Raptor head and its study-batch adapter."""

import math
import sys
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from rsna_knee_abnormality_detection.models import RAPTOR_BACKBONE, ClassAttentionHead, StudyModel


@pytest.fixture(autouse=True)
def single_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.fixture
def tiny_timm(monkeypatch):
    calls = []

    class TinyEncoder(nn.Module):
        num_features = 8

        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(3, self.num_features, 1)
            self.bn = nn.BatchNorm2d(self.num_features)
            self.checkpointing = False

        def set_grad_checkpointing(self, enabled):
            self.checkpointing = enabled

        def forward(self, images):
            return self.bn(self.conv(images)).mean(dim=(-2, -1))

    def create_model(name, **kwargs):
        calls.append((name, kwargs))
        return TinyEncoder()

    monkeypatch.setitem(sys.modules, "timm", SimpleNamespace(create_model=create_model))
    return calls


def config(**overrides):
    return {
        "backbone": RAPTOR_BACKBONE,
        "pooling": "class_attention",
        "input_size": 32,
        "encoder_chunk_size": 4,
        "dropout": 0.2,
        **overrides,
    }


def batch(groups=2):
    images = torch.rand(2, 3, groups, 3, 32, 32)
    mask = torch.tensor([[True, False, True], [False, True, False]])
    metadata = torch.zeros(2, 3, 5)
    positions = torch.linspace(0, 1, groups).expand(2, 3, groups).clone()
    return images, mask, metadata, positions


def test_coatnet_constructor_and_gradient_checkpointing(tiny_timm):
    model = StudyModel({"backbone": RAPTOR_BACKBONE, "pooling": "class_attention", "grad_checkpointing": True})
    assert tiny_timm == [(RAPTOR_BACKBONE, {
        "pretrained": True, "num_classes": 0, "in_chans": 3, "global_pool": "avg", "img_size": 384,
    })]
    assert model.encoder.checkpointing
    assert model.input_size == 384
    assert not model.use_metadata and not model.use_positions
    assert model.head.att[0].out_features == 256
    assert model.head.att[-1].out_features == 12
    assert model.head.clsW.shape == (12, model.feature_dim)
    assert not hasattr(model, "metadata_embedding")


def test_class_attention_finite_gradients_and_padding_excluded_from_encoder(tiny_timm):
    model = StudyModel(config(), pretrained=False).train()
    images, mask, metadata, positions = batch()
    images[~mask] = float("nan")
    images.requires_grad_(True)
    seen = []

    def inspect_inputs(_, args):
        seen.append(args[0].shape[0])
        assert torch.isfinite(args[0]).all()

    hook = model.encoder.register_forward_pre_hook(inspect_inputs)
    logits = model(images, mask, metadata, positions)
    hook.remove()
    assert seen == [4, 2]
    assert logits.shape == (2, 12)
    assert torch.isfinite(logits).all()
    nn.functional.binary_cross_entropy_with_logits(logits, torch.rand_like(logits)).backward()
    for name, parameter in model.named_parameters():
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name
    assert model.encoder.conv.weight.grad.abs().sum() > 0
    assert model.head.att[0].weight.grad.abs().sum() > 0
    assert model.head.clsW.grad.abs().sum() > 0
    assert torch.isfinite(images.grad).all()
    assert torch.count_nonzero(images.grad[~mask]) == 0


def test_masked_values_metadata_and_positions_cannot_change_raptor_prediction(tiny_timm):
    model = StudyModel(config(), pretrained=False).eval()
    images, mask, metadata, positions = batch()
    with torch.no_grad():
        expected = model(images, mask, metadata, positions)
        images[~mask] = float("nan")
        metadata[:] = float("nan")
        positions[:] = float("nan")
        actual = model(images, mask, metadata, positions)
    assert torch.equal(actual, expected)


def test_adapter_matches_public_head_on_each_studys_real_triplets(tiny_timm):
    model = StudyModel(config(), pretrained=False).eval()
    images, mask, metadata, positions = batch(groups=3)
    with torch.no_grad():
        actual = model(images, mask, metadata, positions)
        reference = []
        for study, valid in zip(images, mask, strict=True):
            features = model._encode(study[valid].flatten(0, 1)).unsqueeze(0)
            # RaptorClassifier.head of dreaddevelopment's public training notebook (script 347608187), same maths.
            h = model.head.norm(features)
            a = torch.softmax(model.head.att(h), dim=1)
            pooled = torch.einsum("bkn,bkf->bnf", a, h)
            reference.append((pooled * model.head.clsW).sum(-1) + model.head.clsb)
    torch.testing.assert_close(actual, torch.cat(reference), rtol=1e-5, atol=1e-6)


def test_findings_have_separate_attention_distributions_and_classifier_vectors():
    head = ClassAttentionHead(2, dropout=0).eval()
    head.norm = nn.Identity()
    head.att = nn.Linear(2, 12, bias=False)
    with torch.no_grad():
        head.att.weight.zero_()
        head.att.weight[0, 0] = math.log(3)
        head.att.weight[1, 1] = math.log(3) / 2
        head.clsW.zero_()
        head.clsW[:2] = torch.tensor([1.0, 2.0])
        head.clsb.zero_()
        head.clsb[0] = 0.2
    features = torch.tensor([[[1.0, 0.0], [0.0, 2.0], [float("nan"), float("nan")]]])
    logits = head(features, torch.tensor([[True, True, False]]))
    # Finding 0 uses weights 3/4,1/4; finding 1 uses 1/4,3/4.
    torch.testing.assert_close(logits[0, :2], torch.tensor([1.95, 3.25]))
    assert torch.count_nonzero(logits[0, 2:]) == 0
    with torch.no_grad():
        head.clsW[0] = 17
        head.att.weight[0] = -11
        changed = head(features, torch.tensor([[True, True, False]]))
    assert torch.equal(logits[:, 1:], changed[:, 1:])


@pytest.mark.parametrize("groups", [1, 12, 42])
def test_raptor_accepts_training_and_validation_bag_sizes(tiny_timm, groups):
    model = StudyModel(config(), pretrained=False).eval()
    images = torch.rand(2, 1, groups, 3, 32, 32)
    with torch.no_grad():
        logits = model(images, torch.ones(2, 1, dtype=torch.bool), torch.zeros(2, 1, 5))
    assert logits.shape == (2, 12)
    assert torch.isfinite(logits).all()


def test_coatnet_checkpoint_roundtrip_constructs_without_pretrained_download(tiny_timm):
    settings = config(pretrained=True)
    model = StudyModel(settings, pretrained=False).eval()
    restored = StudyModel(settings, pretrained=False).eval()
    restored.load_state_dict(model.state_dict(), strict=True)
    assert all(not kwargs["pretrained"] for _, kwargs in tiny_timm)
    with torch.no_grad():
        inputs = batch()
        assert torch.equal(model(*inputs), restored(*inputs))


def test_frozen_coatnet_keeps_batch_norm_fixed_and_head_trainable(tiny_timm):
    model = StudyModel(config(freeze_encoder=True), pretrained=False).train()
    before = model.encoder.bn.running_mean.clone()
    logits = model(*batch())
    logits.square().mean().backward()
    assert not model.encoder.training
    assert torch.equal(before, model.encoder.bn.running_mean)
    assert model.encoder.conv.weight.grad is None
    assert model.head.clsW.grad.abs().sum() > 0


def test_coatnet_applies_imagenet_normalization_once(tiny_timm):
    model = StudyModel(config(), pretrained=False).eval()
    images = torch.tensor([0.0, 0.5, 1.0]).view(1, 1, 1, 3, 1, 1).expand(1, 1, 1, 3, 32, 32)
    seen = []
    hook = model.encoder.register_forward_pre_hook(lambda _, args: seen.append(args[0].clone()))
    with torch.no_grad():
        model(images, torch.ones(1, 1, dtype=torch.bool), torch.zeros(1, 1, 5))
    hook.remove()
    expected = (torch.tensor([0.0, 0.5, 1.0]) - torch.tensor([0.485, 0.456, 0.406]))
    expected /= torch.tensor([0.229, 0.224, 0.225])
    torch.testing.assert_close(seen[0][0, :, 0, 0], expected)


@pytest.mark.parametrize("flag", ["use_metadata", "use_positions"])
def test_raptor_rejects_unimplemented_metadata_changes(tiny_timm, flag):
    with pytest.raises(ValueError, match="requires use_metadata=False"):
        StudyModel(config(**{flag: True}), pretrained=False)


def test_raptor_rejects_missing_study_or_wrong_input_size(tiny_timm):
    model = StudyModel(config(), pretrained=False)
    images, mask, metadata, positions = batch()
    mask[0] = False
    with pytest.raises(ValueError, match="at least one real series"):
        model(images, mask, metadata, positions)
    with pytest.raises(ValueError, match="height and width 32"):
        model(images[..., :16, :16], mask, metadata, positions)


def test_gradient_checkpointing_fails_clearly_for_unsupported_backbone():
    with pytest.raises(ValueError, match="Gradient checkpointing is not supported"):
        StudyModel({"backbone": "resnet18", "grad_checkpointing": True}, pretrained=False)
