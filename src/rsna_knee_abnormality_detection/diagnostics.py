"""Measure fixed-checkpoint behavior without changing learned parameters."""

from __future__ import annotations

import time
from contextlib import nullcontext

import torch
from torch import nn


class _ChannelMoments:
    """Merge batches without giving a small final batch excess weight."""

    def __init__(self) -> None:
        self.count = 0
        self.calls = 0
        self.mean: torch.Tensor | None = None
        self.m2: torch.Tensor | None = None

    def observe(self, module: nn.Module, arguments: tuple) -> None:
        values = arguments[0].detach()
        if values.ndim < 2 or values.shape[1] != module.num_features:
            raise ValueError("Unexpected BatchNorm input shape during recalibration")
        count = values.numel() // values.shape[1]
        if count < 1:
            raise ValueError("BatchNorm received an empty input")
        axes = (0, *range(2, values.ndim))
        # Float32 activation reductions avoid allocating full float64 feature
        # maps. Only the small per-channel vectors use float64 for merging.
        variance, mean = torch.var_mean(values.float(), dim=axes, correction=0)
        mean = mean.to(torch.float64)
        m2 = variance.to(torch.float64) * count
        if self.mean is None:
            self.mean, self.m2 = mean, m2
        else:
            total = self.count + count
            difference = mean - self.mean
            self.m2 = self.m2 + m2 + difference.square() * (self.count * count / total)
            self.mean = self.mean + difference * (count / total)
        self.count += count
        self.calls += 1


def _batch_roles(batch: dict, batch_size: int) -> bool:
    checked = False
    for key in ("role", "partition", "split"):
        if key not in batch:
            continue
        values = batch[key]
        if isinstance(values, str):
            values = [values] * batch_size
        else:
            values = list(values)
        if len(values) != batch_size or any(str(value).strip().lower() not in {"train", "training"} for value in values):
            raise ValueError(f"BatchNorm recalibration only accepts training data; invalid {key}")
        checked = True
    return checked


def _equal_to_snapshot(tensor: torch.Tensor, original: torch.Tensor) -> bool:
    current = tensor.detach().cpu()
    if current.shape != original.shape or current.dtype != original.dtype:
        return False
    if current.is_floating_point() or current.is_complex():
        return bool(torch.all((current == original) | (torch.isnan(current) & torch.isnan(original))))
    return torch.equal(current, original)


def recalibrate_batchnorm(model: nn.Module, loader, device, precision: str) -> dict:
    """Re-estimate BatchNorm statistics using only the supplied training images.

    The caller must select the training partition. Bare images do not identify
    their fold, so this function cannot establish that isolation on its own.
    When batch roles are supplied, every role must be ``train`` or ``training``.
    Returned study IDs let the caller verify the actual images that were used.
    A KneeDataset must have ``training=False``: this disables augmentation and
    uses its fixed slice/series choices without changing which studies it holds.

    The whole model is evaluated with no gradients. Only BatchNorm layers enter
    training mode, with reset running statistics and ``momentum=None``. Encoder
    chunks expand to contain every real image group in a complete study batch,
    preventing a fixed 32/16 split from separating sagittal from other views.

    Hooks collect each BatchNorm layer's input moments. Batch means/variances are
    computed in float32, and per-channel float64 Welford merges weight every
    image/spatial observation equally. The final running variance is the pooled
    unbiased variance ``M2 / (count - 1)``, matching PyTorch's running-variance
    convention. This replaces BatchNorm's temporary equal-batch cumulative
    averages, so a partial final batch is not overweighted. Deeper-layer moments
    are observed while preceding BatchNorm layers use their current batch's
    statistics; this is a single recalibration pass, not an iterative fit.

    Learned parameters, gradients, modes, momentum, and chunk size are preserved.
    On success only BatchNorm running means, variances, and counters change.
    Empty or invalid input rolls back every buffer before raising the error.
    No files are read or written by this function apart from the supplied loader.
    """
    if precision not in {"fp32", "fp16", "bf16"}:
        raise ValueError("precision must be fp32, fp16, or bf16")
    device = torch.device(device)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise ValueError("CUDA is unavailable")
        if device.index is None:
            device = torch.device("cuda", torch.cuda.current_device())
        if precision == "bf16" and not torch.cuda.is_bf16_supported():
            raise ValueError("This GPU does not support bf16 recalibration")
    dataset_training = getattr(getattr(loader, "dataset", None), "training", None)
    if dataset_training:
        raise ValueError("Use a dataset with training=False for fixed, unaugmented recalibration images")
    batchnorm = [(name, module) for name, module in model.named_modules()
                 if isinstance(module, nn.modules.batchnorm._BatchNorm)]
    if not batchnorm:
        raise ValueError("The model has no BatchNorm layers to recalibrate")
    for name, module in batchnorm:
        if not module.track_running_stats or module.running_mean is None or module.running_var is None:
            raise ValueError(f"BatchNorm layer {name} does not track running statistics")
    for tensor in [*model.parameters(), *model.buffers()]:
        if tensor.device != device:
            raise ValueError(f"Move the model to {device} before recalibration")
    chunk_size = getattr(model, "encoder_chunk_size", None)
    if chunk_size is not None and (not isinstance(chunk_size, int) or chunk_size < 1):
        raise ValueError("encoder_chunk_size must be a positive integer")

    started = time.perf_counter()
    modes = [(module, module.training) for module in model.modules()]
    momentums = [(module, module.momentum) for _, module in batchnorm]
    parameters = [(parameter, parameter.detach().to("cpu", copy=True)) for parameter in model.parameters()]
    gradients = [(parameter, parameter.grad, None if parameter.grad is None else parameter.grad.detach().to("cpu", copy=True))
                 for parameter in model.parameters()]
    buffers = [(name, buffer, buffer.detach().to("cpu", copy=True)) for name, buffer in model.named_buffers()]
    bn_buffer_ids = {id(buffer) for _, module in batchnorm for buffer in
                     (module.running_mean, module.running_var, module.num_batches_tracked)}
    moments = {name: _ChannelMoments() for name, _ in batchnorm}
    handles = []
    succeeded = False
    batches, studies, series, image_groups, batches_with_roles = 0, 0, 0, 0, 0
    study_ids: list[str] = []
    seen_ids: set[str] = set()
    has_ids: bool | None = None
    images_per_batch = []
    used_chunk_sizes = set()
    try:
        model.eval()
        for name, module in batchnorm:
            module.reset_running_stats()
            module.momentum = None
            module.train(True)
            handles.append(module.register_forward_pre_hook(moments[name].observe))
        with torch.no_grad():
            for batch in loader:
                if not isinstance(batch, dict):
                    raise ValueError("Recalibration batches must be dictionaries")
                images, mask, metadata = (batch.get(key) for key in ("images", "series_mask", "metadata"))
                if not isinstance(images, torch.Tensor) or images.ndim != 6 or images.shape[3] != 3:
                    raise ValueError("images must have shape [batch, series, groups, 3, height, width]")
                batch_size, slots, groups = images.shape[:3]
                if min(images.shape) < 1 or not images.is_floating_point():
                    raise ValueError("Recalibration images must be nonempty floating-point tensors")
                if not isinstance(mask, torch.Tensor) or mask.shape != (batch_size, slots) or mask.dtype != torch.bool:
                    raise ValueError("series_mask must be a matching boolean matrix")
                if not mask.any(dim=1).all():
                    raise ValueError("Every recalibration study needs at least one real series")
                if not isinstance(metadata, torch.Tensor) or metadata.shape != (batch_size, slots, 5):
                    raise ValueError("metadata must have shape [batch, series, 5]")
                positions = batch.get("positions")
                if positions is not None and (not isinstance(positions, torch.Tensor) or positions.shape != (batch_size, slots, groups)):
                    raise ValueError("positions must have shape [batch, series, groups]")
                batches_with_roles += int(_batch_roles(batch, batch_size))
                current_has_ids = "study_id" in batch
                if has_ids is None:
                    has_ids = current_has_ids
                elif current_has_ids != has_ids:
                    raise ValueError("Supply study IDs consistently in every recalibration batch")
                if current_has_ids:
                    ids = batch["study_id"]
                    ids = [ids] if isinstance(ids, str) else list(ids)
                    if len(ids) != batch_size or any(not isinstance(value, str) or not value.strip() for value in ids):
                        raise ValueError("Recalibration study IDs must be nonempty strings matching the batch")
                    if len(set(ids)) != len(ids) or seen_ids.intersection(ids):
                        raise ValueError("A study appears more than once in the recalibration loader")
                    seen_ids.update(ids)
                    study_ids.extend(ids)
                real_series = int(mask.sum().item())
                real_images = real_series * groups
                if chunk_size is not None:
                    model.encoder_chunk_size = max(chunk_size, real_images)
                    used_chunk_sizes.add(model.encoder_chunk_size)
                context = torch.autocast("cuda", dtype=torch.bfloat16 if precision == "bf16" else torch.float16) \
                    if device.type == "cuda" and precision != "fp32" else nullcontext()
                with context:
                    predictions = model(
                        images=images.to(device), series_mask=mask.to(device), metadata=metadata.to(device),
                        positions=None if positions is None else positions.to(device),
                    )
                if not isinstance(predictions, torch.Tensor) or not torch.isfinite(predictions).all():
                    raise ValueError("The model returned invalid predictions during recalibration")
                batches += 1
                studies += batch_size
                series += real_series
                image_groups += real_images
                images_per_batch.append(real_images)
            if not batches:
                raise ValueError("The recalibration loader is empty")
            for name, module in batchnorm:
                state = moments[name]
                if state.count < 2 or state.mean is None or state.m2 is None:
                    raise ValueError(f"Insufficient observations for BatchNorm layer {name}")
                variance = state.m2 / (state.count - 1)
                if not torch.isfinite(state.mean).all() or not torch.isfinite(variance).all() or (variance < 0).any():
                    raise ValueError(f"Nonfinite or negative BatchNorm statistics in {name}")
                module.running_mean.copy_(state.mean)
                module.running_var.copy_(variance)
                module.num_batches_tracked.fill_(state.calls)
            if any(not _equal_to_snapshot(parameter, original) for parameter, original in parameters):
                raise RuntimeError("A learned parameter changed during BatchNorm recalibration")
            for parameter, gradient, original in gradients:
                if parameter.grad is not gradient or (original is not None and not _equal_to_snapshot(gradient, original)):
                    raise RuntimeError("A parameter gradient changed during BatchNorm recalibration")
            for name, buffer, original in buffers:
                if id(buffer) not in bn_buffer_ids and not _equal_to_snapshot(buffer, original):
                    raise RuntimeError(f"A buffer outside BatchNorm changed during recalibration: {name}")
        succeeded = True
    finally:
        for handle in handles:
            handle.remove()
        if not succeeded:
            with torch.no_grad():
                for _, buffer, original in buffers:
                    buffer.copy_(original)
                for parameter, original in parameters:
                    if not _equal_to_snapshot(parameter, original):
                        parameter.copy_(original)
                for parameter, gradient, original in gradients:
                    parameter.grad = gradient
                    if gradient is not None and not _equal_to_snapshot(gradient, original):
                        gradient.copy_(original)
        for module, momentum in momentums:
            module.momentum = momentum
        # Assign each flag directly: calling train() recursively could overwrite
        # a deliberately frozen submodule's original mode.
        for module, mode in modes:
            module.training = mode
        if chunk_size is not None:
            model.encoder_chunk_size = chunk_size

    return {
        "method": "pooled_batchnorm_input_moments",
        "variance": "pooled unbiased variance over image and spatial observations",
        "batch_moment_dtype": "float32", "merge_dtype": "float64",
        "device": str(device), "precision_requested": precision,
        "precision_used": precision if device.type == "cuda" else "fp32",
        "batches": batches, "studies": studies, "series": series, "image_groups": image_groups,
        "study_ids": study_ids if has_ids else None,
        "batches_with_training_roles": batches_with_roles,
        "dataset_training_mode": None if dataset_training is None else bool(dataset_training),
        "images_per_batch": images_per_batch,
        "encoder_chunk_size_before": chunk_size,
        "encoder_chunk_sizes_used": sorted(used_chunk_sizes),
        "batchnorm": {name: {"forward_calls": state.calls, "observations_per_channel": state.count}
                      for name, state in moments.items()},
        "parameters_unchanged": True, "gradients_unchanged": True,
        "module_modes_restored": True, "momentum_restored": True, "chunk_size_restored": True,
        "seconds": time.perf_counter() - started,
    }
