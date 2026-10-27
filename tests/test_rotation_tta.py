"""In-plane rotation of test-time views (augment.rotate_windows)."""

from __future__ import annotations

import torch

from rsna_knee_abnormality_detection.augment import rotate_windows


def test_zero_rotation_is_identity() -> None:
    images = torch.rand(2, 1, 3, 3, 16, 16)
    assert torch.equal(rotate_windows(images, 0.0), images)


def test_quarter_turn_matches_rot90_and_keeps_shape_and_dtype() -> None:
    images = torch.rand(1, 1, 2, 3, 8, 8)
    rotated = rotate_windows(images, 90.0)
    assert rotated.shape == images.shape and rotated.dtype == images.dtype
    expected = torch.rot90(images, k=1, dims=(-2, -1))
    other = torch.rot90(images, k=-1, dims=(-2, -1))
    assert torch.allclose(rotated, expected, atol=1e-5) or torch.allclose(rotated, other, atol=1e-5)


def test_small_rotation_keeps_the_centre_and_zero_pads_corners() -> None:
    images = torch.ones(1, 1, 1, 3, 32, 32)
    rotated = rotate_windows(images, 12.0)
    assert torch.allclose(rotated[..., 16, 16], torch.ones(3))
    assert float(rotated[..., 0, 0].max()) < 0.5


def test_opposite_rotations_differ() -> None:
    images = torch.zeros(1, 1, 1, 3, 32, 32)
    images[..., 4:8, 20:28] = 1.0
    assert not torch.allclose(rotate_windows(images, 12.0), rotate_windows(images, -12.0))
