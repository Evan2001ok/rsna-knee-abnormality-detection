"""Training, validation, and resumable checkpoints for study-level models."""

import copy
import json
import math
import os
import pickle
import platform
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset

from .augment import augment_batch, mirror_window_features, mirror_windows, rotate_windows, validate_gpu_augmentation
from .checkpoint_average import average_checkpoints, retain_top_checkpoint
from .config import validate_config, write_config
from .constants import STUDY_ID, TARGETS
from .dataset import make_dataset, validate_dataset_manifest
from .metrics import (
    masked_bce_sum,
    pairwise_rank_loss,
    read_predictions,
    score_predictions,
    training_loss_weights,
    training_positive_weights,
)
from .models import StudyModel
from .monitoring import TrainingMonitor
from .utils import (
    code_hash,
    file_hash,
    git_state,
    restore_rng,
    rng_state,
    save_checkpoint,
    seed_everything,
    write_json,
)
from .validation_sample import preserve_training_state, select_validation_sample


def study_partitions(config: dict, require_manifest: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    data = config["data"]
    labels = pd.read_csv(data["labels"], dtype={STUDY_ID: str})
    splits = pd.read_csv(data["splits"], dtype={STUDY_ID: str, "group_id": str})
    for name, frame in [("labels", labels), ("splits", splits)]:
        if frame[STUDY_ID].isna().any() or frame[STUDY_ID].duplicated().any():
            raise ValueError(f"{name} must contain unique, nonmissing study IDs")
    if set(labels[STUDY_ID]) != set(splits[STUDY_ID]):
        raise ValueError("Labels and splits must cover exactly the same studies")
    if not {"split", "fold", "group_id"}.issubset(splits.columns):
        raise ValueError("Splits require split, fold, and group_id columns")
    if splits["group_id"].isna().any():
        raise ValueError("Splits must have nonmissing group IDs")
    if "label_source" not in labels:
        raise ValueError("Labels need label_source provenance; run the labels command first")
    expert_ids = set(labels.loc[labels["label_source"].eq("expert"), STUDY_ID])
    expert_groups = set(splits.loc[splits[STUDY_ID].isin(expert_ids), "group_id"])
    expert_rows = splits[splits["group_id"].isin(expert_groups)]
    if not (expert_rows["split"].eq("expert_holdout") & expert_rows["fold"].eq(-1)).all():
        raise ValueError("Expert studies and their patient groups must remain in expert_holdout (fold -1)")
    for _, rows in splits.groupby("group_id"):
        if rows["fold"].nunique() != 1:
            raise ValueError("A patient group crosses validation partitions")
    if not splits["split"].isin(["train", "expert_holdout", "unlabeled"]).all():
        raise ValueError("Unknown split category")
    if (splits["split"].eq("train") & splits["fold"].lt(0)).any():
        raise ValueError("Training studies need nonnegative fold numbers")
    if (splits["split"].eq("expert_holdout") & splits["fold"].ne(-1)).any():
        raise ValueError("Expert holdout studies need fold -1")
    joined = labels[[STUDY_ID, *TARGETS]].merge(splits, on=STUDY_ID, validate="one_to_one")
    eligible = joined["split"].eq("train") & joined["fold"].ge(0)
    training = joined[eligible & joined["fold"].ne(data["fold"])].sort_values(STUDY_ID)
    validation = joined[eligible & joined["fold"].eq(data["fold"])].sort_values(STUDY_ID)
    if data.get("validation_labels"):
        reference = pd.read_csv(data["validation_labels"], dtype={STUDY_ID: str})
        if (reference[STUDY_ID].isna().any() or reference[STUDY_ID].duplicated().any()
                or set(reference[STUDY_ID]) != set(labels[STUDY_ID])):
            raise ValueError("Validation labels must cover exactly the same unique study IDs")
        # Evaluation labels never enter training rows, prevalence weights, or
        # gradient updates. Keep the same validation target policy as the baseline.
        validation = validation.drop(columns=TARGETS).merge(
            reference[[STUDY_ID, *TARGETS]], on=STUDY_ID, validate="one_to_one", how="left",
        ).sort_values(STUDY_ID)
    for frame in [training, validation]:
        if frame.empty:
            raise ValueError("Training and validation must both contain studies; check fold and labels")
        if frame[TARGETS].notna().sum(axis=1).eq(0).any():
            raise ValueError("A training or validation study has no observed target")
    for frame in [training, validation]:
        for col in TARGETS:
            values = pd.to_numeric(frame[col], errors="raise").dropna()
            if not np.isfinite(values).all() or not values.between(0, 1).all():
                raise ValueError(f"Invalid target values in {col}")
    if config["debug"]["max_train_studies"]:
        training = training.head(config["debug"]["max_train_studies"])
    if config["debug"]["max_val_studies"]:
        validation = validation.head(config["debug"]["max_val_studies"])
    if require_manifest:
        manifest = pd.read_csv(data["manifest"], dtype={STUDY_ID: str})
        required = set(training[STUDY_ID]) | set(validation[STUDY_ID])
        missing = required - set(manifest[STUDY_ID])
        if missing:
            raise ValueError(f"Prepared images missing for {len(missing)} selected studies. Run prepare first.")
    return training.reset_index(drop=True), validation.reset_index(drop=True)


def get_device(name: str) -> torch.device:
    if name.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; set training.device: cpu for a CPU run")
    return torch.device(name)


def autocast_context(device: torch.device, precision: str):
    if precision == "fp32" or device.type != "cuda":
        return nullcontext()
    return torch.autocast(device_type="cuda", dtype=torch.bfloat16 if precision == "bf16" else torch.float16)


def model_inputs(batch: dict, device: torch.device) -> dict:
    # Window descriptors exist only for the Raptor recipe; models without them ignore them.
    names = ["images", "series_mask", "metadata", "positions", *(["window_features"] if "window_features" in batch else [])]
    return {name: batch[name].to(device, non_blocking=True) for name in names}


def oai_epoch_indices(study_ids: list[str], draw: int, seed: int) -> np.ndarray:
    """OAI follow-up draw (Oct 17, 2026): positions of every non-OAI row plus `draw` OAI rows (OAI_<participant>_<side>_
    <visit>) chosen with participants balanced: participants in a random order each take one random exam in turn, round
    after round. All OAI rows when `draw` reaches their number. Sorted positions; the loader shuffles them."""
    ids = [str(study) for study in study_ids]
    oai = [i for i, study in enumerate(ids) if study.startswith("OAI_")]
    other = [i for i, study in enumerate(ids) if not study.startswith("OAI_")]
    if draw >= len(oai):
        return np.arange(len(ids))
    rng = np.random.default_rng(seed)
    exams: dict[str, list[int]] = {}
    for i in oai:
        exams.setdefault(ids[i].split("_")[1], []).append(i)
    people = sorted(exams)
    order = [people[j] for j in rng.permutation(len(people))]
    pools = {person: [exams[person][j] for j in rng.permutation(len(exams[person]))] for person in order}
    chosen: list[int] = []
    while len(chosen) < draw:
        for person in order:
            if pools[person]:
                chosen.append(pools[person].pop())
                if len(chosen) == draw:
                    break
    return np.sort(np.asarray(other + chosen, dtype=np.int64))


def make_loader(dataset, settings: dict, training: bool, seed: int) -> DataLoader:
    workers = settings["num_workers"]
    kwargs = {"prefetch_factor": 2, "multiprocessing_context": "spawn"} if workers else {}
    return DataLoader(
        dataset, batch_size=settings["batch_size"], shuffle=training, num_workers=workers,
        pin_memory=settings["device"].startswith("cuda"), drop_last=False,
        generator=torch.Generator().manual_seed(seed), **kwargs,
    )


@torch.inference_mode()
def predict_loader(model, loader, device: torch.device, precision: str, first_batch_callback=None,
                   test_time_mirror: bool = False, test_time_rotations: tuple[float, ...] = ()) -> tuple[pd.DataFrame, dict]:
    """Predict each study; optionally average with its left/right mirror image, and/or over in-plane rotations of
    every window (degrees, e.g. (-12, 0, 12); empty = the unrotated images only)."""
    model.eval()
    ids, predictions, targets = [], [], []
    for batch_index, batch in enumerate(loader):
        if batch_index == 0 and first_batch_callback is not None:
            first_batch_callback(batch)
        inputs = model_inputs(batch, device)
        if test_time_rotations:
            views = []
            for degrees in test_time_rotations:
                with autocast_context(device, precision):
                    views.append(model(**{**inputs, "images": rotate_windows(inputs["images"], degrees)}).float().sigmoid())
            probabilities = torch.stack(views).mean(0)
        else:
            with autocast_context(device, precision):
                logits = model(**inputs)
            probabilities = logits.float().sigmoid()
        if test_time_mirror and test_time_rotations:
            raise ValueError("Mirrored and rotated test-time views are not combined")
        if test_time_mirror:
            planes = batch["window_planes"].to(device, non_blocking=True)
            everyone = torch.ones(planes.shape[0], dtype=torch.bool, device=device)
            mirrored = {**inputs, "images": mirror_windows(inputs["images"], planes, everyone)}
            if "window_features" in inputs:
                mirrored["window_features"] = mirror_window_features(inputs["window_features"], everyone)
            with autocast_context(device, precision):
                probabilities = (probabilities + model(**mirrored).float().sigmoid()) / 2
        probabilities = probabilities.cpu().numpy()
        if not np.isfinite(probabilities).all():
            raise FloatingPointError("Nonfinite prediction")
        ids.extend(batch["study_id"])
        predictions.append(probabilities)
        y = batch["targets"].numpy().astype(np.float64)
        y[~batch["target_mask"].numpy()] = np.nan
        targets.append(y)
    if not ids:
        raise ValueError("No studies to predict")
    probabilities = np.concatenate(predictions)
    frame = pd.DataFrame(probabilities, columns=TARGETS)
    frame.insert(0, STUDY_ID, ids)
    return frame, score_predictions(np.concatenate(targets), probabilities)


def _schedule(total_steps: int, warmup_steps: int, min_ratio: float):
    def multiplier(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / max(1, warmup_steps)
        progress = min(1.0, (step - warmup_steps) / max(1, total_steps - warmup_steps))
        return min_ratio + (1 - min_ratio) * (1 + math.cos(math.pi * progress)) / 2
    return multiplier


@torch.no_grad()
def update_ema(average, model, decay: float, step: int) -> None:
    """Move the running weight average toward the current weights.

    The effective decay starts low so early, untrained weights fade quickly.
    Floating-point buffers such as normalization statistics are averaged too.
    """
    rate = 1 - min(decay, (1 + step) / (10 + step))
    for target, source in zip(average.state_dict().values(), model.state_dict().values(), strict=True):
        if target.is_floating_point():
            target.lerp_(source.detach().to(target.dtype), rate)
        else:
            target.copy_(source)


def augmentation_generator(device: torch.device, seed: int, epoch: int, batch_index: int) -> torch.Generator:
    generator = torch.Generator(device=device)
    generator.manual_seed((seed * 1_000_003 + epoch * 100_003 + batch_index) % (2**63 - 1))
    return generator


def _best_artifact_receipt(output: Path, epoch: int) -> None:
    write_json(output / "best-artifacts.json", {
        "epoch": epoch,
        "files": {name: file_hash(output / name) for name in [
            "best-model.pt", "validation-best.csv", "metrics-best.json",
        ]},
    })


def _recover_best_artifacts(output, checkpoint, model, val_data, validation_ids, device, settings, seed,
                            test_time_mirror=False):
    """Repair an interrupted best-artifact write without advancing training state."""
    metric = settings["selection_metric"]
    history = checkpoint["history"]
    if not history:
        raise ValueError("Resume checkpoint has no history to identify its best epoch")

    def rank(record):
        value = record["validation"][metric]
        if value is None or not np.isfinite(value):
            raise ValueError("Resume history contains an undefined selection metric")
        return (value if metric == "loss" else -value, record["epoch"])

    expected = min(history, key=rank)
    expected_score = -rank(expected)[0]
    if checkpoint["best_score"] != expected_score:
        raise ValueError("Resume best_score disagrees with recovered history")

    def matches(candidate):
        if not isinstance(candidate, dict):
            return False
        if candidate.get("epoch") != expected["epoch"] or candidate.get("metrics") != expected["validation"]:
            return False
        if any(candidate.get(key) != checkpoint.get(key) for key in [
            "format_version", "targets", "config", "image_preparation",
        ]):
            return False
        provenance = candidate.get("provenance", {})
        return all(provenance.get(key) == checkpoint["provenance"].get(key) for key in [
            "code_sha256", "input_sha256",
        ])

    def load_if_readable(path):
        try:
            return torch.load(path, map_location="cpu", weights_only=True)
        except (OSError, EOFError, RuntimeError, ValueError, TypeError, pickle.UnpicklingError):
            return None

    best = load_if_readable(output / "best-model.pt")
    agrees = matches(best)
    if agrees:
        try:
            predictions = read_predictions(output / "validation-best.csv")
            agrees = predictions[STUDY_ID].tolist() == validation_ids
            receipt_path = output / "best-artifacts.json"
            if receipt_path.exists():
                receipt = json.loads(receipt_path.read_text())
                agrees = agrees and receipt.get("epoch") == expected["epoch"] and receipt.get("files") == {
                    name: file_hash(output / name) for name in ["best-model.pt", "validation-best.csv", "metrics-best.json"]
                }
            else:
                # Compatibility with runs created before artifact receipts.
                agrees = agrees and json.loads((output / "metrics-best.json").read_text()) == expected["validation"]
        except (OSError, ValueError, KeyError, TypeError):
            agrees = False
    if agrees:
        return

    if matches(checkpoint):
        source = checkpoint
    elif matches(best):
        source = best
    else:
        source = load_if_readable(output / f"candidate-epoch-{expected['epoch']:03d}.pt")
    if not matches(source):
        raise ValueError(
            f"Cannot recover best epoch {expected['epoch']}: its checkpoint is missing or incompatible"
        )

    before_rng = rng_state()
    modes = [(module, module.training) for module in model.modules()]
    try:
        model.load_state_dict(source["model"], strict=True)
        loader = make_loader(val_data, settings, training=False, seed=seed)
        predictions, metrics = predict_loader(model, loader, device, settings["precision"],
                                              test_time_mirror=test_time_mirror)
        # Preserve the original selection metrics in the checkpoint. Replayed
        # low-precision predictions can differ slightly; their measured metrics
        # belong to the regenerated CSV and are tied to it by the receipt.
        inference = {key: source[key] for key in [
            "format_version", "targets", "config", "model", "epoch", "metrics", "provenance", "image_preparation",
        ]}
        save_checkpoint(output / "best-model.pt", inference)
        predictions.to_csv(output / "validation-best.csv", index=False)
        write_json(output / "metrics-best.json", metrics)
        _best_artifact_receipt(output, expected["epoch"])
        print(json.dumps({"event": "recovered_best_artifacts", "epoch": expected["epoch"]}), flush=True)
    finally:
        # With a running weight average, "model" holds the averaged weights.
        model.load_state_dict(checkpoint.get("training_model", checkpoint["model"]), strict=True)
        for module, mode in modes:
            module.training = mode
        restore_rng(before_rng)


def train(config: dict, resume: str | Path | None = None, stop_after_epoch: int | None = None) -> Path:
    """Resume at an epoch boundary with the original split and optimization schedule."""
    validate_config(config)
    settings = config["training"]
    if stop_after_epoch is not None and stop_after_epoch < 1:
        raise ValueError("stop_after_epoch must be positive")
    torch.set_num_threads(min(8, os.cpu_count() or 1))
    seed_everything(config["seed"], settings["deterministic"])
    device = get_device(settings["device"])
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        if settings["precision"] == "bf16" and not torch.cuda.is_bf16_supported():
            raise RuntimeError("This GPU does not support bf16; select fp16 or fp32")
    training, validation = study_partitions(config)
    target_weighting = settings.get("target_weighting", "observed")
    known_counts, finding_weights = training_loss_weights(training[TARGETS].to_numpy(), target_weighting)
    target_weights = None if finding_weights is None else torch.as_tensor(finding_weights, dtype=torch.float32, device=device)
    positive_values = training_positive_weights(
        training[TARGETS].to_numpy(), settings.get("positive_weighting", "none"),
        settings.get("fixed_positive_weights"),
    )
    positive_weights = None if positive_values is None else torch.as_tensor(positive_values, dtype=torch.float32, device=device)
    selected_metric = settings["selection_metric"]
    # Check reference-label support before writing a run or downloading weights.
    support = score_predictions(validation[TARGETS].to_numpy(), np.full((len(validation), len(TARGETS)), 0.5))
    if support[selected_metric] is None:
        raise ValueError(f"Validation {selected_metric} is undefined. Check labels or explicitly select loss/available_auc.")
    image_preparation = validate_dataset_manifest(
        config["data"]["manifest"], config["data"],
        study_ids=training[STUDY_ID].tolist() + validation[STUDY_ID].tolist(),
    )
    output = Path(settings["output_dir"])
    if output.exists() and any(output.iterdir()) and not resume:
        raise FileExistsError(f"{output} is not empty. Choose a new output_dir or use --resume.")
    if resume and Path(resume).resolve().parent != output.resolve():
        raise ValueError("Resume checkpoint must be in the configured output_dir")
    signature = {name: file_hash(config["data"][name]) for name in ["manifest", "labels", "splits"]}
    if settings.get("init_checkpoint"):
        signature["init_checkpoint"] = file_hash(settings["init_checkpoint"])
    if config["data"].get("validation_labels"):
        signature["validation_labels"] = file_hash(config["data"]["validation_labels"])
    provenance = {
        "input_sha256": signature, "code_sha256": code_hash(), "git": git_state(),
        "python": platform.python_version(), "torch": str(torch.__version__), "numpy": str(np.__version__),
        "device": torch.cuda.get_device_name(device) if device.type == "cuda" else str(device),
        "training_studies": len(training), "validation_studies": len(validation),
        "training_ids": training[STUDY_ID].tolist(), "validation_ids": validation[STUDY_ID].tolist(),
        "debug_run": any(config["debug"].values()),
        "training_objective": {
            "target_weighting": target_weighting,
            "weight_source": (
                "finding weights: selected training studies only; positive weights: fixed configuration"
                if settings.get("positive_weighting", "none") == "fixed" else
                "selected training studies only; validation and expert labels excluded"
            ),
            "known_training_counts": {name: int(count) for name, count in zip(TARGETS, known_counts, strict=True)},
            "target_weights": {
                name: float(weight)
                for name, weight in zip(TARGETS, np.ones(len(TARGETS)) if finding_weights is None else finding_weights, strict=True)
            },
            "normalization": "observed targets" if finding_weights is None else "studies",
            "formula": "sum(observed BCE) / observed target count" if finding_weights is None else
                "sum(observed BCE * N_training / (12 * known_training_count)) / study count",
            "positive_weighting": settings.get("positive_weighting", "none"),
            "positive_term_weights": None if positive_values is None else dict(zip(TARGETS, positive_values.tolist(), strict=True)),
            "positive_weight_source": (
                "training.fixed_positive_weights in configuration; not recomputed from target prevalence"
                if settings.get("positive_weighting", "none") == "fixed" else
                "selected training studies only; validation and expert labels excluded" if positive_values is not None else None
            ),
            "positive_weight_formula": (
                "p = clip(mean soft training targets, 0.03, 0.70); positive BCE multiplier = clip((1-p)/p, 1, 10)"
                if settings.get("positive_weighting", "none") == "raptor" else None
            ),
        },
        "validation_labels": config["data"].get("validation_labels") or config["data"]["labels"],
    }
    train_data = make_dataset(config["data"]["manifest"], training, config["data"], training=True, seed=config["seed"])
    val_data = make_dataset(config["data"]["manifest"], validation, config["data"], training=False, seed=config["seed"])
    monitoring = config["monitoring"]
    validation_interval = monitoring.get("validation_every_updates", 0) if monitoring["enabled"] else 0
    sample_rows, sample_loader = None, None
    if validation_interval:
        sample_rows = select_validation_sample(validation, monitoring.get("validation_studies", 32), config["seed"])
        val_indices = {study: index for index, study in enumerate(validation[STUDY_ID])}
        sample_data = Subset(val_data, [val_indices[study] for study in sample_rows[STUDY_ID]])
        # A small fixed sample does not justify starting worker processes at
        # every check. Reuse its loader without consuming the training RNG.
        sample_loader = make_loader(sample_data, {**settings, "num_workers": 0}, training=False, seed=config["seed"])
    provenance["monitoring"] = {
        "validation_every_updates": validation_interval,
        "validation_sample_ids": [] if sample_rows is None else sample_rows[STUDY_ID].tolist(),
        "checkpoint_selection": "full validation after each epoch only",
    }
    initial = settings.get("init_checkpoint")
    model = StudyModel(config["model"], pretrained=False if resume or initial else None).to(device)
    if initial and not resume:
        # Fine-tuning (Oct 17, 2026): the same model settings, every weight loaded.
        start = torch.load(initial, map_location="cpu", weights_only=True)
        if start.get("format_version") != 1 or start.get("targets") != TARGETS:
            raise ValueError("Incompatible init_checkpoint version or output order")
        for key in ("backbone", "input_size", "pooling", "use_metadata", "use_positions", "window_features",
                    "spatial_attention"):
            if start["config"]["model"].get(key) != config["model"].get(key):
                raise ValueError(f"init_checkpoint model.{key} differs from this run's")
        model.load_state_dict(start["model"], strict=True)
    ema_decay = settings.get("ema_decay", 0.0)
    # The running weight average is used for validation, selection, and saved models.
    ema_model = copy.deepcopy(model).requires_grad_(False) if ema_decay > 0 else None
    if settings.get("channels_last", False):
        # Convolutions then run in the faster channels-last layout; saved weights load into any layout.
        model.encoder.to(memory_format=torch.channels_last)
    if settings.get("compile", False):
        # Compiles in place, so parameter names and saved checkpoints are unchanged.
        model.encoder.compile()
        if model.spatial_attention:
            # In-image attention calls forward_features directly, which the module compile above skips; uncompiled, a
            # 320 px batch ran out of memory on a 32 GB GPU (Oct 15, 2026). Not saved: the weights are unchanged.
            model.encoder.forward_features = torch.compile(model.encoder.forward_features)
    validation_start = settings.get("validation_start_epoch", 1)
    augmentation = validate_gpu_augmentation(config["data"].get("gpu_augmentation", {}))
    test_time_mirror = config["data"].get("test_time_mirror", False)
    encoder_ids = {id(p) for p in model.encoder.parameters()}
    groups = [
        {"params": [p for p in model.encoder.parameters() if p.requires_grad], "lr": settings["encoder_lr"], "name": "encoder"},
        {"params": [p for p in model.parameters() if p.requires_grad and id(p) not in encoder_ids], "lr": settings["head_lr"], "name": "head"},
    ]
    optimizer = torch.optim.AdamW([group for group in groups if group["params"]], weight_decay=settings["weight_decay"])
    oai_draw = settings.get("oai_epoch_draw", 0)
    study_ids = training[STUDY_ID].tolist()
    epoch_size = len(oai_epoch_indices(study_ids, oai_draw, 0)) if oai_draw else len(train_data)
    batches_per_epoch = math.ceil(epoch_size / settings["batch_size"])
    updates_per_epoch = math.ceil(batches_per_epoch / settings["accumulation_steps"])
    if settings.get("schedule", "cosine") == "one_cycle":
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=[group["lr"] for group in optimizer.param_groups],
            total_steps=updates_per_epoch * settings["epochs"],
            pct_start=settings.get("one_cycle_warmup_fraction", 0.15),
        )
    else:
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            _schedule(updates_per_epoch * settings["epochs"], updates_per_epoch * settings["warmup_epochs"], settings["min_lr_ratio"]),
        )
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda" and settings["precision"] == "fp16")
    start_epoch, best_score, global_step = 0, None, 0
    history, training_history = [], []
    average_top_k = settings.get("average_top_k", 1)
    average_candidates = []
    if resume:
        checkpoint = torch.load(resume, map_location="cpu", weights_only=True)
        if checkpoint.get("format_version") != 1 or "optimizer" not in checkpoint:
            raise ValueError("Resume needs a full training checkpoint (last.pt)")
        if checkpoint["config"] != config:
            raise ValueError("Resume settings differ from checkpoint. Start a new run for a changed experiment.")
        if checkpoint["provenance"]["input_sha256"] != signature:
            raise ValueError("Data, labels, or splits changed since this checkpoint")
        if checkpoint["provenance"]["code_sha256"] != provenance["code_sha256"]:
            raise ValueError("Pipeline code changed since this checkpoint; resume from the original code revision")
        if ema_model is not None:
            model.load_state_dict(checkpoint["training_model"])
            ema_model.load_state_dict(checkpoint["model"])
        else:
            model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        scaler.load_state_dict(checkpoint["scaler"])
        start_epoch, best_score, global_step = checkpoint["epoch"], checkpoint["best_score"], checkpoint["global_step"]
        history = checkpoint["history"]
        training_history = checkpoint.get("training_history", [])
        provenance = checkpoint["provenance"]
        restore_rng(checkpoint["rng"])
        # A checkpoint from an epoch before validation starts has no scores to select from yet.
        if checkpoint["metrics"] is not None:
            if average_top_k > 1:
                average_candidates = retain_top_checkpoint(output, checkpoint, history, average_top_k)
            write_json(output / "history.json", history)
            _recover_best_artifacts(
                output, checkpoint, model, val_data, validation[STUDY_ID].tolist(), device, settings, config["seed"],
                test_time_mirror,
            )
    output.mkdir(parents=True, exist_ok=True)
    write_config(config, output / "config.yaml")
    write_json(output / "provenance.json", provenance)
    partition_rows = pd.concat([training.assign(role="training"), validation.assign(role="validation")])
    partition_rows[[STUDY_ID, "group_id", "fold", "role"]].to_csv(output / "partitions.csv", index=False)
    if sample_rows is not None:
        sample_rows[[STUDY_ID]].to_csv(output / "monitoring-validation-sample.csv", index=False)
    print(json.dumps({"event": "start", "train": len(training), "validation": len(validation),
                      "parameters": sum(p.numel() for p in model.parameters()), "resume_epoch": start_epoch,
                      "debug_run": provenance["debug_run"]}), flush=True)
    with TrainingMonitor(output, config, resume_step=global_step if resume else None) as monitor:
        monitor.support(support, global_step)
        if sample_rows is not None:
            sample_support = score_predictions(
                sample_rows[TARGETS].to_numpy(), np.full((len(sample_rows), len(TARGETS)), 0.5),
            )
            monitor.sample_support(sample_support, sample_rows[STUDY_ID].tolist(), global_step)
        end_epoch = min(settings["epochs"], stop_after_epoch or settings["epochs"])
        for epoch in range(start_epoch, end_epoch):
            started = time.perf_counter()
            train_data.set_epoch(epoch)
            # OAI follow-up draw (Oct 17, 2026): a fresh participant-balanced OAI subset each epoch.
            epoch_data = (Subset(train_data, oai_epoch_indices(study_ids, oai_draw, config["seed"] * 1000 + epoch).tolist())
                          if oai_draw else train_data)
            loader = make_loader(epoch_data, settings, training=True, seed=config["seed"] + epoch)
            model.train()
            if settings.get("freeze_batchnorm", False):
                # Fine-tuning (Oct 17, 2026): BatchNorm normalizes with the start checkpoint's statistics and does not
                # update them (one study per step would otherwise set them); its scale and shift still train.
                for module in model.modules():
                    if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
                        module.eval()
            optimizer.zero_grad(set_to_none=True)
            total_loss, total_denominator, accumulated_denominator = 0.0, 0, 0
            recent_loss, recent_denominator, recent_studies, recent_updates = 0.0, 0, 0, 0
            recent_started = started
            skipped_updates = 0
            # Keep the scaled backward loss near a mean, even with partial labels.
            # Correct unscaled gradients to the actual window denominator below:
            # observed targets for the original loss, studies for equal_finding.
            backward_normalizer = settings["batch_size"] * settings["accumulation_steps"]
            rank_weight = float(settings.get("rank_loss_weight", 0.0))
            series_weight = float(settings.get("series_aux_weight", 0.0))
            if target_weights is None:
                backward_normalizer *= len(TARGETS)
            for batch_index, batch in enumerate(loader):
                targets = batch["targets"].to(device, non_blocking=True)
                mask = batch["target_mask"].to(device, non_blocking=True)
                inputs = model_inputs(batch, device)
                if augmentation["enabled"]:
                    inputs["images"], features = augment_batch(
                        inputs["images"], batch["window_planes"].to(device, non_blocking=True), augmentation,
                        augmentation_generator(device, config["seed"], epoch, batch_index), inputs.get("window_features"),
                        batch["window_slots"].to(device, non_blocking=True) if "window_slots" in batch else None,
                    )
                    if features is not None:
                        inputs["window_features"] = features
                with autocast_context(device, settings["precision"]):
                    if series_weight:
                        logits, series_logits, rows = model(
                            **inputs, window_slots=batch["window_slots"].to(device, non_blocking=True), return_series=True)
                    else:
                        logits = model(**inputs)
                    loss_sum, denominator = masked_bce_sum(logits, targets, mask, target_weights, positive_weights)
                    if series_weight:
                        # Each series alone against the study labels; its mean enters with series_weight relative to
                        # the mean study loss, as the ranking term below.
                        s_sum, s_denominator = masked_bce_sum(series_logits, targets[rows], mask[rows], target_weights,
                                                              positive_weights)
                        loss_sum = loss_sum + series_weight * (s_sum / max(s_denominator, 1)) * denominator
                    if rank_weight:
                        # Scaled by the denominator so that, after the per-window normalisation below, the ranking
                        # term enters with weight rank_weight relative to the mean BCE.
                        loss_sum = loss_sum + rank_weight * pairwise_rank_loss(logits, targets, mask) * denominator
                if not torch.isfinite(loss_sum):
                    raise FloatingPointError("Nonfinite training loss")
                scaler.scale(loss_sum / backward_normalizer).backward()
                batch_loss = float(loss_sum.detach())
                total_loss += batch_loss
                total_denominator += denominator
                accumulated_denominator += denominator
                recent_loss += batch_loss
                recent_denominator += denominator
                recent_studies += len(batch["study_id"])
                if (batch_index + 1) % 100 == 0:
                    print(json.dumps({"event": "training", "epoch": epoch + 1,
                                      "batch": batch_index + 1, "batches": len(loader),
                                      "loss": total_loss / total_denominator,
                                      "seconds": time.perf_counter() - started}), flush=True)
                if (batch_index + 1) % settings["accumulation_steps"] == 0 or batch_index + 1 == len(loader):
                    scaler.unscale_(optimizer)
                    for parameter in model.parameters():
                        if parameter.grad is not None:
                            parameter.grad.mul_(backward_normalizer / accumulated_denominator)
                    gradient_norm = torch.nn.utils.clip_grad_norm_(
                        model.parameters(), settings["clip_grad_norm"], error_if_nonfinite=not scaler.is_enabled(),
                    )
                    previous_scale = scaler.get_scale()
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)
                    if scaler.get_scale() >= previous_scale:
                        scheduler.step()
                        global_step += 1
                        recent_updates += 1
                        if ema_model is not None:
                            update_ema(ema_model, model, ema_decay, global_step)
                        force_log = batch_index + 1 == len(loader)
                        if monitor.should_log(global_step, force=force_log):
                            now = time.perf_counter()
                            elapsed = max(now - recent_started, 1e-9)
                            monitor.update(
                                global_step, total_loss / total_denominator,
                                {group["name"]: group["lr"] for group in optimizer.param_groups}, force=force_log,
                                measurements={
                                    "Loss/train_recent": recent_loss / recent_denominator,
                                    "Progress/epoch": epoch + (batch_index + 1) / len(loader),
                                    "Performance/training_studies_per_second": recent_studies / elapsed,
                                    "Performance/training_updates_per_second": recent_updates / elapsed,
                                    "Performance/elapsed_epoch_seconds": now - started,
                                    "Performance/GPU_allocated_GiB": torch.cuda.memory_allocated(device) / 1024**3
                                        if device.type == "cuda" else 0.0,
                                    "Performance/peak_GPU_GiB": torch.cuda.max_memory_allocated(device) / 1024**3
                                        if device.type == "cuda" else 0.0,
                                    "Optimization/gradient_norm_before_clip": float(gradient_norm),
                                    "Optimization/skipped_updates": skipped_updates,
                                    "Optimization/loss_scale": scaler.get_scale(),
                                },
                            )
                            preview = batch if not augmentation["enabled"] else {
                                **batch, "images": inputs["images"][:1].detach().float().cpu()}
                            monitor.batch_images(preview, "training_augmented", global_step)
                            recent_loss, recent_denominator, recent_studies, recent_updates = 0.0, 0, 0, 0
                            recent_started = time.perf_counter()
                        if sample_loader is not None and global_step % validation_interval == 0:
                            sample_started = time.perf_counter()
                            with preserve_training_state(model):
                                _, sample_metrics = predict_loader(
                                    model, sample_loader, device, settings["precision"],
                                    first_batch_callback=lambda batch: monitor.batch_images(
                                        batch, "validation_sample", global_step,
                                    ),
                                )
                            sample_seconds = time.perf_counter() - sample_started
                            monitor.validation_sample(sample_metrics, global_step, sample_seconds)
                            # Throughput measures training work, excluding the
                            # extra validation pass, even with different cadences.
                            recent_started += sample_seconds
                    else:
                        # An fp16 overflow reduces the scale and skips this update.
                        # Do not advance the learning rate for an update not taken.
                        skipped_updates += 1
                    accumulated_denominator = 0
            eval_model = model if ema_model is None else ema_model
            if epoch + 1 < validation_start:
                # Early epochs are never among the selected ones; skip their full validation pass.
                record = {
                    "epoch": epoch + 1, "global_step": global_step, "train_loss": total_loss / total_denominator,
                    "skipped_updates": skipped_updates, "seconds": time.perf_counter() - started,
                    "peak_gpu_gb": torch.cuda.max_memory_allocated(device) / 1024**3 if device.type == "cuda" else 0.0,
                    "learning_rates": scheduler.get_last_lr(),
                }
                training_history.append(record)
                save_checkpoint(output / "last.pt", {
                    "format_version": 1, "targets": TARGETS, "config": config, "model": eval_model.state_dict(),
                    "epoch": epoch + 1, "metrics": None, "provenance": provenance, "image_preparation": image_preparation,
                    "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict(),
                    "rng": rng_state(), "global_step": global_step, "best_score": best_score, "history": history,
                    "training_history": training_history,
                    **({"training_model": model.state_dict()} if ema_model is not None else {}),
                })
                write_json(output / "training-history.json", training_history)
                monitor.training_epoch(record)
                print(json.dumps(record | {"validation": "skipped"}), flush=True)
                continue
            val_loader = make_loader(val_data, settings, training=False, seed=config["seed"])
            predictions, metrics = predict_loader(
                eval_model, val_loader, device, settings["precision"],
                first_batch_callback=lambda batch: monitor.batch_images(batch, "validation", global_step),
                test_time_mirror=test_time_mirror,
            )
            value = metrics[selected_metric]
            if value is None or not np.isfinite(value):
                raise FloatingPointError(f"Validation {selected_metric} is undefined")
            score = -value if selected_metric == "loss" else value
            improved = best_score is None or score > best_score
            if improved:
                best_score = score
            record = {
                "epoch": epoch + 1, "global_step": global_step,
                "train_loss": total_loss / total_denominator, "validation": metrics,
                "skipped_updates": skipped_updates,
                "seconds": time.perf_counter() - started,
                "peak_gpu_gb": torch.cuda.max_memory_allocated(device) / 1024**3 if device.type == "cuda" else 0.0,
                "learning_rates": scheduler.get_last_lr(), "best": improved,
            }
            history.append(record)
            inference_checkpoint = {
                "format_version": 1, "targets": TARGETS, "config": config, "model": eval_model.state_dict(),
                "epoch": epoch + 1, "metrics": metrics, "provenance": provenance,
                "image_preparation": image_preparation,
            }
            checkpoint = {
                **inference_checkpoint, "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                "scaler": scaler.state_dict(), "rng": rng_state(), "global_step": global_step,
                "best_score": best_score, "history": history, "training_history": training_history,
            }
            if ema_model is not None:
                checkpoint["training_model"] = model.state_dict()
            predictions.to_csv(output / "validation-last.csv", index=False)
            save_checkpoint(output / "last.pt", checkpoint)
            if average_top_k > 1:
                average_candidates = retain_top_checkpoint(output, inference_checkpoint, history, average_top_k)
            if improved:
                save_checkpoint(output / "best-model.pt", inference_checkpoint)
                predictions.to_csv(output / "validation-best.csv", index=False)
                write_json(output / "metrics-best.json", metrics)
                _best_artifact_receipt(output, epoch + 1)
            write_json(output / "history.json", history)
            monitor.epoch(record)
            print(json.dumps({k: v for k, v in record.items() if k != "validation"} | {
                "val_loss": metrics["loss"], "macro_auc": metrics["macro_auc"], "available_auc": metrics["available_auc"],
            }), flush=True)
        if average_top_k > 1 and max(start_epoch, end_epoch) >= settings["epochs"]:
            averaged = average_checkpoints(
                [output / name for name in average_candidates], output / "averaged-model.pt",
            )
            model.load_state_dict(averaged["model"], strict=True)
            val_loader = make_loader(val_data, settings, training=False, seed=config["seed"])
            predictions, metrics = predict_loader(model, val_loader, device, settings["precision"],
                                                  test_time_mirror=test_time_mirror)
            averaged["metrics"] = metrics
            save_checkpoint(output / "averaged-model.pt", averaged)
            predictions.to_csv(output / "validation-averaged.csv", index=False)
            write_json(output / "metrics-averaged.json", metrics)
            monitor.scalar("AUC/averaged_top_epochs", metrics["macro_auc"], global_step)
            monitor.scalar("Loss/averaged_validation", metrics["loss"], global_step)
            print(json.dumps({"event": "checkpoint_average", "members": len(average_candidates),
                              "macro_auc": metrics["macro_auc"]}), flush=True)
        write_json(output / "status.json", {
            "completed": max(start_epoch, end_epoch) >= settings["epochs"], "epochs_finished": max(start_epoch, end_epoch),
            "planned_epochs": settings["epochs"], "global_step": global_step,
            "best_selection_score": best_score, "debug_run": provenance["debug_run"],
        })
        monitor.status(max(start_epoch, end_epoch) >= settings["epochs"], max(start_epoch, end_epoch), global_step)
    return output / "best-model.pt"
