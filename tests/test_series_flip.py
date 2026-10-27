"""The per-series slice-order reversal (data.gpu_augmentation.series_flip_prob, Oct 16, 2026)."""

import pytest
import torch

from rsna_knee_abnormality_detection.augment import augment_batch, validate_gpu_augmentation


def batch():
    torch.manual_seed(0)
    images = torch.rand(3, 1, 8, 3, 16, 16)
    planes = torch.tensor([0, 0, 1, 1, 2, 2, 0, 1]).repeat(3, 1, 1)
    slots = torch.tensor([0, 0, 2, 2, 4, 4, 1, 3]).repeat(3, 1, 1)
    return images, planes, slots


def settings(**extra):
    return validate_gpu_augmentation({"enabled": True, "mirror_prob": 0.5, "affine_prob": 0.8, "intensity_prob": 0.8,
                                      "noise_prob": 0.3, **extra})


def run(cfg, slots):
    images, planes, _ = batch()
    generator = torch.Generator().manual_seed(7)
    return augment_batch(images, planes, cfg, generator, None, slots)[0]


def test_off_by_default_and_unchanged():
    images, planes, slots = batch()
    assert validate_gpu_augmentation({})["series_flip_prob"] == 0.0
    a = augment_batch(images, planes, settings(), torch.Generator().manual_seed(7))[0]
    b = run(settings(), slots)
    assert torch.equal(a, b)


def test_flips_whole_series_and_keeps_other_draws():
    _, _, slots = batch()
    plain = run(settings(), slots)
    flipped = run(settings(series_flip_prob=0.5), slots)
    same = (flipped == plain).flatten(3).all(-1)
    reversed_ = (flipped == plain.flip(3)).flatten(3).all(-1)
    assert (same | reversed_).all(), "every window is either unchanged or exactly reversed"
    changed = reversed_ & ~same
    for study in range(3):
        for slot in slots[study, 0].unique():
            windows = slots[study, 0] == slot
            assert changed[study, 0, windows].all() or not changed[study, 0, windows].any(), "a series flips as a whole"
    assert changed.any() and not changed.all()


def test_always_and_needs_slots():
    images, planes, slots = batch()
    plain = run(settings(), slots)
    assert torch.equal(run(settings(series_flip_prob=1.0), slots), plain.flip(3))
    with pytest.raises(ValueError, match="window_slots"):
        augment_batch(images, planes, settings(series_flip_prob=0.5), torch.Generator().manual_seed(7))
    with pytest.raises(ValueError, match="at most one"):
        validate_gpu_augmentation({"series_flip_prob": 1.5})
