"""Training-time image augmentation on the GPU for bags of three-slice windows.

Images have shape ``[batch, series, windows, 3, height, width]`` with values in
[0, 1]; the three channels are neighbouring MRI slices. ``window_planes`` gives
each window's plane: 0 sagittal, 1 coronal, 2 axial.

Mirroring turns a right knee into a realistic left knee. Coronal and axial images
in this dataset all put the patient's left on the image right, and every
sagittal stack is ordered along the same patient axis, so a horizontal flip of
coronal/axial windows plus reversed slice order in sagittal windows keeps the
anatomical medial/lateral labels correct. Other transforms change each window
independently; the three slices of a window always share one transform.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from .constants import SIDE_FEATURE, WINDOW_FEATURES

SAGITTAL = 0

DEFAULT_GPU_AUGMENTATION = {
    "enabled": False,
    "mirror_prob": 0.0,
    "affine_prob": 0.0,
    "rotation_deg": 15.0,
    "zoom_min": 0.9,
    "zoom_max": 1.15,
    "translate": 0.08,
    "intensity_prob": 0.0,
    "gamma": 1.4,
    "contrast": 0.15,
    "bias_field_prob": 0.0,
    "bias_field_strength": 0.3,
    "noise_prob": 0.0,
    "noise_std": 0.03,
    "blur_prob": 0.0,
    "blur_sigma_max": 1.0,
    "lowres_prob": 0.0,
    "lowres_factor_max": 2.0,
    # Per series, reverse the slice order inside every window of that series (the teammates' random slice-order
    # reversal, which helped their model; Oct 16, 2026). Drawn last, so the other transforms keep their random draws.
    "series_flip_prob": 0.0,
}


def validate_gpu_augmentation(settings: dict) -> dict:
    """Return complete settings, rejecting unknown names and impossible ranges."""
    if not isinstance(settings, dict):
        raise ValueError("data.gpu_augmentation must be a mapping")
    unknown = set(settings) - set(DEFAULT_GPU_AUGMENTATION)
    if unknown:
        raise ValueError(f"Unknown data.gpu_augmentation settings: {sorted(unknown)}")
    merged = DEFAULT_GPU_AUGMENTATION | settings
    if not isinstance(merged["enabled"], bool):
        raise ValueError("data.gpu_augmentation.enabled must be true or false")
    for name, value in merged.items():
        if name == "enabled":
            continue
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            raise ValueError(f"data.gpu_augmentation.{name} must be a finite nonnegative number")
        if name.endswith("_prob") and value > 1:
            raise ValueError(f"data.gpu_augmentation.{name} must be at most one")
    if not 0 < merged["zoom_min"] <= merged["zoom_max"]:
        raise ValueError("data.gpu_augmentation requires 0 < zoom_min <= zoom_max")
    if merged["gamma"] < 1 or merged["lowres_factor_max"] < 1:
        raise ValueError("data.gpu_augmentation gamma and lowres_factor_max must be at least one")
    if merged["contrast"] >= 1 or merged["translate"] >= 1:
        raise ValueError("data.gpu_augmentation contrast and translate must be below one")
    return merged


def _uniform(shape, low, high, generator, device):
    return low + (high - low) * torch.rand(shape, generator=generator, device=device)


def mirror_windows(images: torch.Tensor, window_planes: torch.Tensor, studies: torch.Tensor) -> torch.Tensor:
    """Mirror the selected studies: flip coronal/axial windows, reverse sagittal slice order."""
    if images.ndim != 6 or window_planes.shape != images.shape[:3] or studies.shape != images.shape[:1]:
        raise ValueError("Expected images [B,S,W,3,H,W], window_planes [B,S,W] and a [B] study selection")
    selected = studies.view(-1, 1, 1).to(torch.bool)
    in_plane = (selected & (window_planes != SAGITTAL)).view(*window_planes.shape, 1, 1, 1)
    depth = (selected & (window_planes == SAGITTAL)).view(*window_planes.shape, 1, 1, 1)
    images = torch.where(in_plane, images.flip(-1), images)
    return torch.where(depth, images.flip(3), images)


def rotate_windows(images: torch.Tensor, degrees: float) -> torch.Tensor:
    """Rotate every window image in its own plane by `degrees` about the image centre (test-time views; same
    convention, bilinear sampling and zero padding as the training rotation in `_affine`)."""
    if images.ndim != 6 or images.shape[3] != 3:
        raise ValueError("Expected images [B,S,W,3,H,W]")
    if degrees == 0:
        return images
    shape = images.shape
    flat = images.reshape(-1, 3, *shape[-2:]).float()
    angle = torch.deg2rad(torch.tensor(float(degrees), device=images.device))
    cos, sin = torch.cos(angle), torch.sin(angle)
    zero = torch.zeros((), device=images.device)
    theta = torch.stack([torch.stack([cos, -sin, zero]), torch.stack([sin, cos, zero])]).expand(flat.shape[0], 2, 3)
    grid = F.affine_grid(theta, list(flat.shape), align_corners=False)
    rotated = F.grid_sample(flat, grid, mode="bilinear", padding_mode="zeros", align_corners=False)
    return rotated.reshape(shape).to(images.dtype)


def mirror_window_features(features: torch.Tensor, studies: torch.Tensor) -> torch.Tensor:
    """A mirrored right knee is a left knee: flip the side; anatomical offsets stay."""
    if features.ndim != 4 or features.shape[-1] != len(WINDOW_FEATURES) or studies.shape != features.shape[:1]:
        raise ValueError("Expected window features [B,S,W,7] and a [B] study selection")
    sign = torch.ones_like(features)
    sign[..., SIDE_FEATURE] = torch.where(studies.to(torch.bool), -1.0, 1.0).view(-1, 1, 1).to(features.dtype)
    return features * sign


def _affine(flat: torch.Tensor, settings: dict, generator, chosen: torch.Tensor) -> torch.Tensor:
    count, device = flat.shape[0], flat.device
    angle = torch.deg2rad(_uniform(count, -settings["rotation_deg"], settings["rotation_deg"], generator, device))
    zoom = torch.exp(_uniform(count, math.log(settings["zoom_min"]), math.log(settings["zoom_max"]), generator, device))
    # Normalized coordinates span 2 units across the image.
    shift = _uniform((count, 2), -2 * settings["translate"], 2 * settings["translate"], generator, device)
    cos, sin = torch.cos(angle) / zoom, torch.sin(angle) / zoom
    theta = torch.stack([torch.stack([cos, -sin, shift[:, 0]], 1), torch.stack([sin, cos, shift[:, 1]], 1)], 1)
    identity = torch.tensor([[1., 0., 0.], [0., 1., 0.]], device=device).expand(count, 2, 3)
    theta = torch.where(chosen.view(-1, 1, 1), theta, identity)
    grid = F.affine_grid(theta.to(flat.dtype), list(flat.shape), align_corners=False)
    return F.grid_sample(flat, grid, mode="bilinear", padding_mode="zeros", align_corners=False)


def _gaussian_blur(flat: torch.Tensor, sigma: float) -> torch.Tensor:
    radius = max(1, int(math.ceil(3 * sigma)))
    offsets = torch.arange(-radius, radius + 1, device=flat.device, dtype=flat.dtype)
    kernel = torch.exp(-0.5 * (offsets / sigma) ** 2)
    kernel = kernel / kernel.sum()
    channels = flat.shape[1]
    horizontal = kernel.view(1, 1, 1, -1).repeat(channels, 1, 1, 1)
    vertical = kernel.view(1, 1, -1, 1).repeat(channels, 1, 1, 1)
    flat = F.conv2d(F.pad(flat, (radius, radius, 0, 0), mode="reflect"), horizontal, groups=channels)
    return F.conv2d(F.pad(flat, (0, 0, radius, radius), mode="reflect"), vertical, groups=channels)


def flip_series(images: torch.Tensor, window_slots: torch.Tensor, probability: float,
                generator: torch.Generator) -> torch.Tensor:
    """Per study and series, with the given probability reverse the slice order (the three channels) of all its windows."""
    if window_slots.shape != images.shape[:3]:
        raise ValueError("window_slots must have shape [B,S,W]")
    batch = images.shape[0]
    slots = window_slots.to(images.device).long().reshape(batch, -1)
    draws = torch.rand(batch, int(slots.max()) + 1, generator=generator, device=images.device) < probability
    flip = torch.gather(draws, 1, slots).view(*images.shape[:3], 1, 1, 1)
    return torch.where(flip, images.flip(3), images)


def augment_batch(images: torch.Tensor, window_planes: torch.Tensor, settings: dict,
                  generator: torch.Generator, window_features: torch.Tensor | None = None,
                  window_slots: torch.Tensor | None = None):
    """Apply the configured random transforms.

    Returns new images in [0, 1] and the window features, whose knee side is
    flipped for mirrored studies (``None`` stays ``None``). ``window_slots`` (each
    window's series) is needed only for ``series_flip_prob``.
    """
    if not settings["enabled"]:
        return images, window_features
    if images.ndim != 6 or images.shape[3] != 3 or window_planes.shape != images.shape[:3]:
        raise ValueError("Expected images [B,S,W,3,H,W] and window_planes [B,S,W]")
    device, dtype = images.device, images.dtype
    images = images.float()
    batch = images.shape[0]
    if settings["mirror_prob"] > 0:
        studies = torch.rand(batch, generator=generator, device=device) < settings["mirror_prob"]
        images = mirror_windows(images, window_planes, studies)
        if window_features is not None:
            window_features = mirror_window_features(window_features, studies)
    shape = images.shape
    flat = images.reshape(-1, 3, *shape[-2:])
    count = flat.shape[0]

    def chosen(probability):
        return torch.rand(count, generator=generator, device=device) < probability

    if settings["affine_prob"] > 0:
        flat = _affine(flat, settings, generator, chosen(settings["affine_prob"]))
    if settings["bias_field_prob"] > 0:
        # Smooth multiplicative shading, as from uneven receive-coil sensitivity.
        field = torch.randn((count, 1, 4, 4), generator=generator, device=device) * settings["bias_field_strength"]
        field = F.interpolate(field, size=shape[-2:], mode="bicubic", align_corners=False)
        field = torch.exp(field - field.mean(dim=(2, 3), keepdim=True))
        flat = torch.where(chosen(settings["bias_field_prob"]).view(-1, 1, 1, 1), flat * field, flat)
    if settings["intensity_prob"] > 0:
        log_gamma = math.log(settings["gamma"])
        gamma = torch.exp(_uniform(count, -log_gamma, log_gamma, generator, device)).view(-1, 1, 1, 1)
        contrast = _uniform(count, 1 - settings["contrast"], 1 + settings["contrast"], generator, device).view(-1, 1, 1, 1)
        adjusted = flat.clamp(0, 1).pow(gamma) * contrast
        flat = torch.where(chosen(settings["intensity_prob"]).view(-1, 1, 1, 1), adjusted, flat)
    if settings["blur_prob"] > 0:
        sigma = float(_uniform(1, 0.3, max(settings["blur_sigma_max"], 0.3), generator, device))
        flat = torch.where(chosen(settings["blur_prob"]).view(-1, 1, 1, 1), _gaussian_blur(flat, sigma), flat)
    if settings["lowres_prob"] > 0:
        factor = float(_uniform(1, 1.25, max(settings["lowres_factor_max"], 1.25), generator, device))
        small = [max(8, int(round(size / factor))) for size in shape[-2:]]
        degraded = F.interpolate(F.interpolate(flat, size=small, mode="area"), size=shape[-2:], mode="bilinear",
                                 align_corners=False)
        flat = torch.where(chosen(settings["lowres_prob"]).view(-1, 1, 1, 1), degraded, flat)
    if settings["noise_prob"] > 0:
        level = _uniform(count, 0, settings["noise_std"], generator, device).view(-1, 1, 1, 1)
        noise = torch.randn(flat.shape, generator=generator, device=device) * level
        flat = torch.where(chosen(settings["noise_prob"]).view(-1, 1, 1, 1), flat + noise, flat)
    images = flat.clamp_(0, 1).reshape(shape)
    if settings["series_flip_prob"] > 0:
        if window_slots is None:
            raise ValueError("series_flip_prob needs window_slots")
        images = flip_series(images, window_slots, settings["series_flip_prob"], generator)
    return images.to(dtype), window_features
