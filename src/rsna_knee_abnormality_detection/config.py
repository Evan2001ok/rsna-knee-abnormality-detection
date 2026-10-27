"""Explicit, versioned settings shared by training and prediction."""

import math
from copy import deepcopy
from numbers import Real
from pathlib import Path

import yaml

from .augment import DEFAULT_GPU_AUGMENTATION, validate_gpu_augmentation
from .constants import RAPTOR_VARIANT_NAMES, RAPTOR_VARIANT_SIZES, TARGETS

DEFAULTS = {
    "seed": 42,
    "data": {
        "recipe": "standard",
        "manifest": "data/processed/train/manifest.csv",
        "labels": "data/labels/baseline/labels.csv",
        "validation_labels": "",
        "splits": "data/splits/baseline/splits.csv",
        "fold": 0,
        "cache_size": 320,
        "image_size": 224,
        "slices_per_series": 8,
        "series_per_plane": 1,
        "neighbor_stride": 1,
        "raptor_train_windows": 12,
        "raptor_eval_windows": 42,
        "raptor_variant": "raptor-widedense-series-safe-v1",
        "raptor_window_sampling": "random",
        "raptor_series_dropout": 0.0,
        "raptor_view": "resize",
        "raptor_slots": None,
        # Each series' share of training windows spread over its depth (Oct 17, 2026; stratified draws only).
        "raptor_window_spread": False,
        # Fine-tuning on all windows (Oct 17, 2026; needs training.batch_size 1).
        "raptor_train_all_windows": False,
        # Guided training windows (Oct 17, 2026; 0 = off): see raptor_data.load_window_guide.
        "raptor_guide_file": None,
        "raptor_guided_windows": 0,
        "raptor_guide_top": 3,
        "test_time_mirror": False,
        "gpu_augmentation": dict(DEFAULT_GPU_AUGMENTATION),
        "augmentation": {
            "enabled": True,
            "rotation_deg": 7.0,
            "translate": 0.04,
            "scale": 0.08,
            "contrast": 0.15,
            "brightness": 0.05,
            "noise_std": 0.015,
        },
    },
    "model": {
        "backbone": "resnet34",
        "input_size": 224,
        "grad_checkpointing": False,
        "pretrained": True,
        "pretrained_checkpoint": None,
        "pooling": "mean",
        "dropout": 0.2,
        "encoder_chunk_size": 32,
        "freeze_encoder": False,
        "use_metadata": True,
        "use_positions": True,
        "window_features": "none",
        # Per-finding attention over each window's final CoAtNet feature map (False = the plain average).
        "spatial_attention": False,
    },
    "training": {
        "output_dir": "runs/resnet34_fold0",
        "epochs": 15,
        "batch_size": 2,
        "accumulation_steps": 4,
        "encoder_lr": 0.0001,
        "head_lr": 0.0003,
        "weight_decay": 0.0001,
        "warmup_epochs": 1,
        "min_lr_ratio": 0.05,
        "clip_grad_norm": 5.0,
        "num_workers": 4,
        "precision": "bf16",
        "device": "cuda",
        "selection_metric": "macro_auc",
        "target_weighting": "observed",
        "positive_weighting": "none",
        "fixed_positive_weights": None,
        "schedule": "cosine",
        "one_cycle_warmup_fraction": 0.15,
        "average_top_k": 1,
        "ema_decay": 0.0,
        # Weight of the in-batch pairwise ranking loss (0 = off, the original objective).
        "rank_loss_weight": 0.0,
        # Weight of the per-series loss: each series' windows alone also predict the study's findings through the same
        # head (0 = off; training only, no new weights).
        "series_aux_weight": 0.0,
        "deterministic": False,
        # OAI rows drawn per epoch, participants balanced (Oct 17, 2026; 0 = every OAI row every epoch).
        "oai_epoch_draw": 0,
        # Start from this checkpoint's model weights (Oct 17, 2026; None = the usual pretrained encoder).
        "init_checkpoint": None,
        # BatchNorm keeps its statistics from the start checkpoint (Oct 17, 2026; fine-tuning one study per step).
        "freeze_batchnorm": False,
        # Speed settings: skip full validation before this epoch (selection only uses later epochs),
        # store encoder weights channels-last, and compile the encoder. None change the model's weights or outputs.
        "validation_start_epoch": 1,
        "channels_last": False,
        "compile": False,
    },
    "debug": {"max_train_studies": 0, "max_val_studies": 0},
    "monitoring": {
        "enabled": True, "log_every_updates": 10, "flush_seconds": 10, "images": True,
        "validation_every_updates": 0, "validation_studies": 32,
    },
}


def _merge(defaults: dict, values: dict, prefix: str = "") -> dict:
    result = deepcopy(defaults)
    if not isinstance(values, dict):
        raise ValueError(f"{prefix or 'Configuration'} must be a mapping")
    for key, value in values.items():
        name = f"{prefix}{key}"
        if key not in result:
            raise ValueError(f"Unknown setting: {name}")
        if isinstance(result[key], dict):
            result[key] = _merge(result[key], value, name + ".")
        else:
            result[key] = value
    return result


def validate_positive_weighting(mode: str, fixed_weights: dict | None = None) -> None:
    """Require an explicit, complete named weight map only for fixed mode."""
    if mode not in ("none", "raptor", "fixed"):
        raise ValueError("training.positive_weighting must be none, raptor, or fixed")
    if mode != "fixed":
        if fixed_weights is not None:
            raise ValueError("training.fixed_positive_weights requires positive_weighting fixed")
        return
    if not isinstance(fixed_weights, dict) or set(fixed_weights) != set(TARGETS):
        raise ValueError("training.fixed_positive_weights must map exactly the twelve target names to weights")
    for target, value in fixed_weights.items():
        if isinstance(value, bool) or not isinstance(value, Real):
            raise ValueError(f"training.fixed_positive_weights[{target!r}] must be a finite positive number")
        try:
            valid = math.isfinite(value) and value > 0
        except OverflowError:
            valid = False
        if not valid:
            raise ValueError(f"training.fixed_positive_weights[{target!r}] must be a finite positive number")


def validate_config(config: dict) -> dict:
    for section, keys in {
        "data": ["cache_size", "image_size", "slices_per_series", "series_per_plane", "neighbor_stride"],
        "training": ["epochs", "batch_size", "accumulation_steps"],
        "model": ["encoder_chunk_size"],
        "monitoring": ["log_every_updates", "flush_seconds"],
    }.items():
        for key in keys:
            value = config[section][key]
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{section}.{key} must be a positive integer")
    for key in ["encoder_lr", "head_lr", "clip_grad_norm"]:
        if config["training"][key] <= 0:
            raise ValueError(f"training.{key} must be positive")
    if config["training"]["num_workers"] < 0:
        raise ValueError("training.num_workers must be nonnegative")
    if config["data"]["fold"] < 0:
        raise ValueError("data.fold must be nonnegative")
    if config["training"]["precision"] not in {"fp32", "fp16", "bf16"}:
        raise ValueError("training.precision must be fp32, fp16, or bf16")
    if config["training"]["selection_metric"] not in {"macro_auc", "available_auc", "loss"}:
        raise ValueError("selection_metric must be macro_auc, available_auc, or loss")
    if config["training"].get("target_weighting", "observed") not in {"observed", "equal_finding"}:
        raise ValueError("training.target_weighting must be observed or equal_finding")
    validate_positive_weighting(
        config["training"].get("positive_weighting", "none"),
        config["training"].get("fixed_positive_weights"),
    )
    if config["training"].get("schedule", "cosine") not in {"cosine", "one_cycle"}:
        raise ValueError("training.schedule must be cosine or one_cycle")
    if not 0 < config["training"].get("one_cycle_warmup_fraction", 0.15) < 1:
        raise ValueError("one_cycle_warmup_fraction must be between zero and one")
    top_k = config["training"].get("average_top_k", 1)
    if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 1:
        raise ValueError("training.average_top_k must be a positive integer")
    start = config["training"].get("validation_start_epoch", 1)
    if isinstance(start, bool) or not isinstance(start, int) or not 1 <= start <= config["training"]["epochs"]:
        raise ValueError("training.validation_start_epoch must be an integer between 1 and epochs")
    if top_k > config["training"]["epochs"] - start + 1:
        raise ValueError("training.average_top_k cannot exceed the number of validated epochs")
    for key in ["channels_last", "compile"]:
        if not isinstance(config["training"].get(key, False), bool):
            raise ValueError(f"training.{key} must be true or false")
    if top_k > 1 and config["training"]["selection_metric"] != "macro_auc":
        raise ValueError("Checkpoint averaging requires selection_metric macro_auc")
    if config["data"].get("recipe", "standard") not in {"standard", "raptor"}:
        raise ValueError("data.recipe must be standard or raptor")
    if config["model"]["pooling"] not in {"mean", "max", "attention", "class_attention"}:
        raise ValueError("model.pooling must be mean, max, attention, or class_attention")
    if config["data"].get("recipe", "standard") == "raptor":
        for key in ["raptor_train_windows", "raptor_eval_windows"]:
            value = config["data"].get(key, 12 if key == "raptor_train_windows" else 42)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"data.{key} must be a positive integer")
        variant = config["data"].get("raptor_variant", RAPTOR_VARIANT_NAMES[0])
        if variant not in RAPTOR_VARIANT_NAMES:
            raise ValueError(f"data.raptor_variant must be one of {RAPTOR_VARIANT_NAMES}")
        if config["data"]["cache_size"] != RAPTOR_VARIANT_SIZES[variant]:
            raise ValueError(f"Raptor preparation {variant} requires cache_size {RAPTOR_VARIANT_SIZES[variant]}")
        view = config["data"].get("raptor_view", "resize")
        if view not in {"resize", "center_crop"}:
            raise ValueError("data.raptor_view must be resize or center_crop")
        if view == "center_crop" and config["data"]["image_size"] > config["data"]["cache_size"]:
            raise ValueError("data.raptor_view center_crop needs image_size no larger than cache_size")
        if config["data"].get("raptor_window_sampling", "random") not in {"random", "stratified"}:
            raise ValueError("data.raptor_window_sampling must be random or stratified")
        dropout = config["data"].get("raptor_series_dropout", 0.0)
        if isinstance(dropout, bool) or not isinstance(dropout, Real) or not 0 <= dropout < 1:
            raise ValueError("data.raptor_series_dropout must be in [0, 1)")
        chosen = config["data"].get("raptor_slots")
        if chosen is not None and (not isinstance(chosen, list) or not chosen or len(set(chosen)) != len(chosen)
                                   or any(isinstance(v, bool) or not isinstance(v, int) or v < 0 for v in chosen)):
            raise ValueError("data.raptor_slots must be null or a list of distinct slot numbers")
        guided = config["data"].get("raptor_guided_windows", 0)
        top = config["data"].get("raptor_guide_top", 3)
        if (isinstance(guided, bool) or not isinstance(guided, int) or not 0 <= guided <= len(TARGETS)
                or (guided and guided >= config["data"]["raptor_train_windows"])):
            raise ValueError("data.raptor_guided_windows must be 0, or a whole number below raptor_train_windows "
                             "and at most the number of findings")
        if isinstance(top, bool) or not isinstance(top, int) or top < 1:
            raise ValueError("data.raptor_guide_top must be a positive whole number")
        if guided and not config["data"].get("raptor_guide_file"):
            raise ValueError("data.raptor_guided_windows needs data.raptor_guide_file")
    all_windows = config["data"].get("raptor_train_all_windows", False)
    if not isinstance(all_windows, bool):
        raise ValueError("data.raptor_train_all_windows must be true or false")
    if all_windows and (config["data"].get("recipe", "standard") != "raptor" or config["training"]["batch_size"] != 1):
        raise ValueError("data.raptor_train_all_windows needs the raptor recipe and training.batch_size 1")
    initial = config["training"].get("init_checkpoint")
    if initial is not None and (not isinstance(initial, str) or not initial):
        raise ValueError("training.init_checkpoint must be null or a checkpoint path")
    if not isinstance(config["training"].get("freeze_batchnorm", False), bool):
        raise ValueError("training.freeze_batchnorm must be true or false")
    draw = config["training"].get("oai_epoch_draw", 0)
    if isinstance(draw, bool) or not isinstance(draw, int) or draw < 0:
        raise ValueError("training.oai_epoch_draw must be a whole number of 0 or more")
    spread = config["data"].get("raptor_window_spread", False)
    if not isinstance(spread, bool):
        raise ValueError("data.raptor_window_spread must be true or false")
    if spread and (config["data"].get("recipe", "standard") != "raptor"
                   or config["data"].get("raptor_window_sampling", "random") != "stratified"):
        raise ValueError("data.raptor_window_spread needs the raptor recipe with stratified window sampling")
    augmentation = validate_gpu_augmentation(config["data"].get("gpu_augmentation", {}))
    if not isinstance(config["data"].get("test_time_mirror", False), bool):
        raise ValueError("data.test_time_mirror must be true or false")
    if (augmentation["enabled"] or config["data"].get("test_time_mirror", False)) and config["data"].get("recipe", "standard") != "raptor":
        raise ValueError("GPU augmentation and test-time mirroring need the raptor recipe's window planes")
    window_mode = config["model"].get("window_features", "none")
    if window_mode not in {"none", "types", "types_and_positions"}:
        raise ValueError("model.window_features must be none, types, or types_and_positions")
    if window_mode != "none" and (config["data"].get("recipe", "standard") != "raptor"
                                  or config["model"]["pooling"] != "class_attention"):
        raise ValueError("model.window_features need the raptor recipe and class_attention pooling")
    ema_decay = config["training"].get("ema_decay", 0.0)
    if isinstance(ema_decay, bool) or not isinstance(ema_decay, Real) or not 0 <= ema_decay < 1:
        raise ValueError("training.ema_decay must be in [0, 1)")
    rank_weight = config["training"].get("rank_loss_weight", 0.0)
    if isinstance(rank_weight, bool) or not isinstance(rank_weight, Real) or not 0 <= rank_weight < 10:
        raise ValueError("training.rank_loss_weight must be in [0, 10)")
    series_weight = config["training"].get("series_aux_weight", 0.0)
    if isinstance(series_weight, bool) or not isinstance(series_weight, Real) or not 0 <= series_weight < 10:
        raise ValueError("training.series_aux_weight must be in [0, 10)")
    if series_weight and (config["data"].get("recipe", "standard") != "raptor" or config["model"]["pooling"] != "class_attention"):
        raise ValueError("training.series_aux_weight needs the raptor recipe and class_attention pooling")
    spatial =config["model"].get("spatial_attention", False)
    if not isinstance(spatial, bool) or (spatial and (not config["model"]["backbone"].startswith("coatnet_")
                                                       or config["model"]["pooling"] != "class_attention")):
        raise ValueError("model.spatial_attention must be true or false, and true needs CoAtNet with class_attention")
    if config["model"]["backbone"].startswith("coatnet_"):
        if config["model"].get("input_size", 384) != config["data"]["image_size"]:
            raise ValueError("CoAtNet input_size must equal data.image_size")
    if config["model"]["backbone"] in {"orthofoundation_l", "dinov3_small", "dinov3_base"}:
        if config["model"]["input_size"] != config["data"]["image_size"] or config["model"]["input_size"] % 16:
            raise ValueError("ViT input_size must equal data.image_size and be a multiple of 16")
    if not 0 <= config["model"]["dropout"] < 1:
        raise ValueError("model.dropout must be in [0,1)")
    for key in config["debug"]:
        if config["debug"][key] < 0:
            raise ValueError(f"debug.{key} must be nonnegative")
    if not 0 <= config["training"]["min_lr_ratio"] <= 1:
        raise ValueError("training.min_lr_ratio must be in [0,1]")
    if not 0 <= config["training"]["warmup_epochs"] <= config["training"]["epochs"]:
        raise ValueError("warmup_epochs must be between 0 and epochs")
    if config["training"]["weight_decay"] < 0:
        raise ValueError("weight_decay must be nonnegative")
    for key in ["enabled", "images"]:
        if not isinstance(config["monitoring"][key], bool):
            raise ValueError(f"monitoring.{key} must be true or false")
    for key in ["validation_every_updates", "validation_studies"]:
        value = config["monitoring"].get(key, 0 if key == "validation_every_updates" else 32)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"monitoring.{key} must be a nonnegative integer")
    return config


def load_config(path: str | Path) -> dict:
    """Paths are relative to the working directory, just like CLI file paths."""
    with Path(path).open() as handle:
        values = yaml.safe_load(handle)
    return validate_config(_merge(DEFAULTS, values or {}))


def write_config(config: dict, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(yaml.safe_dump(config, sort_keys=False))
