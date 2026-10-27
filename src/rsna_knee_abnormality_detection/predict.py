"""Use exactly the trained image settings for local or offline predictions."""

import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

from .constants import STUDY_ID, TARGETS
from .dataset import dataset_preprocess_version, make_dataset, validate_dataset_manifest
from .models import StudyModel
from .train import get_device, make_loader, predict_loader
from .utils import file_hash, write_json


class ProbabilityAverage(nn.Module):
    """Models that share one image preparation, run on a single pass over the data.

    Returns the logit of their weighted mean probability, so the usual sigmoid (and the
    average with the mirrored pass) yields exactly the averaged probabilities. Each model has
    one weight, or one weight per finding.
    """

    def __init__(self, models: list[nn.Module], weights) -> None:
        super().__init__()
        self.members = nn.ModuleList(models)
        weights = torch.as_tensor(np.asarray(weights, dtype=np.float32)).reshape(len(models), -1)
        device = next((value.device for value in self.members.parameters()), torch.device("cpu"))
        self.register_buffer("weights", weights.to(device), persistent=False)

    def forward(self, **inputs):
        mean = sum(weight * member(**inputs).float().sigmoid()
                   for weight, member in zip(self.weights, self.members, strict=True))
        return torch.logit(mean, eps=1e-7)


def inference_precision(device: torch.device, configured: str) -> str:
    """GPUs before Ampere (Kaggle's T4) have no native bf16; PyTorch emulates it slowly and still reports it as supported.

    fp16 matched fp32 within 5e-4 on 16 studies and runs on the T4's tensor cores.
    """
    if device.type == "cuda" and configured == "bf16" and torch.cuda.get_device_capability(device) < (8, 0):
        return "fp16"
    return configured


# Data settings that only choose training/validation studies or labels; they never change how a study is read.
SPLIT_ONLY_DATA_KEYS = frozenset({"fold", "labels", "validation_labels", "splits"})


def same_image_inputs(configs: list[dict]) -> bool:
    """True when every model sees identical windows: same reading settings and the same sampling seed."""
    def reading(config):
        # Checkpoints from before data.raptor_view always resized; count that as the same view.
        data = {"raptor_view": "resize", **config["data"]}
        return {key: value for key, value in data.items() if key not in SPLIT_ONLY_DATA_KEYS}, config["seed"]

    return all(reading(config) == reading(configs[0]) for config in configs)


def load_model(checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if checkpoint.get("format_version") != 1 or checkpoint.get("targets") != TARGETS:
        raise ValueError("Incompatible checkpoint version or output order")
    if checkpoint.get("image_preparation", {}).get("preprocess_version") != dataset_preprocess_version(checkpoint["config"]["data"]):
        raise ValueError("Checkpoint was trained with a different or unrecorded image preparation version")
    model = StudyModel(checkpoint["config"]["model"], pretrained=False)
    model.load_state_dict(checkpoint["model"], strict=True)
    return model.to(device).eval(), checkpoint


def blend_weights_by_finding(weights, models: int) -> np.ndarray:
    """One column of weights per finding, each summing to one: from one weight per model (the same for every
    finding) or one weight per model and finding."""
    matrix = np.asarray(weights if weights is not None else np.ones(models), dtype=float)
    if matrix.ndim == 1:
        matrix = np.repeat(matrix[:, None], len(TARGETS), axis=1)
    if (matrix.shape != (models, len(TARGETS)) or not np.isfinite(matrix).all() or (matrix < 0).any()
            or (matrix.sum(axis=0) <= 0).any()):
        raise ValueError("Provide one finite, nonnegative weight per checkpoint (or per checkpoint and finding), "
                         "with a positive sum for every finding")
    return matrix / matrix.sum(axis=0)


class Predictor:
    """Checkpoints loaded (and hashed) once, then used for any number of prepared image caches.

    `predict` is one Predictor used once. A long-running process (the bundle's per-device workers) keeps one Predictor
    for every group of studies, so the models are not reloaded and rehashed for each group; the predictions are the
    same, because every group runs exactly the code below.
    """

    def __init__(self, checkpoints: list[str | Path], device: str = "cuda", weights=None, accelerate: bool = False) -> None:
        """With `accelerate` (Oct 16, 2026; CUDA only): channels_last encoders, cuDNN's fastest-algorithm search,
        the last encoder chunk padded to one shape, and compiled encoders. On a T4 this was 29% faster per model and
        within 0.0002 of the plain autocast predictions in our speed test; compiling takes
        about a minute, so it suits a long-lived process."""
        if not checkpoints:
            raise ValueError("At least one checkpoint is required")
        self.checkpoints = [Path(path) for path in checkpoints]
        self.blend_weights = blend_weights_by_finding(weights, len(self.checkpoints))
        self.by_finding = not np.allclose(self.blend_weights, self.blend_weights[:, :1])
        self.device = device
        self.compute_device = get_device(device)
        if accelerate and self.compute_device.type != "cuda":
            raise ValueError("accelerate needs a CUDA device")
        self.accelerate = accelerate
        self.loaded = [load_model(path, self.compute_device) for path in self.checkpoints]
        if accelerate:
            torch.backends.cudnn.benchmark = True
            for model, _ in self.loaded:
                model.pad_last_chunk = True
                model.encoder.to(memory_format=torch.channels_last)
                model.encoder.compile()
        self.hashes = [file_hash(path) for path in self.checkpoints]
        self.configs = [checkpoint["config"] for _, checkpoint in self.loaded]
        self.shared = same_image_inputs(self.configs) and len({c["training"]["precision"] for c in self.configs}) == 1
        # Models that see identical windows share one pass over the data instead of reloading it per model.
        self.passes = ([(ProbabilityAverage([model for model, _ in self.loaded], self.blend_weights), self.configs[0], 1.0)]
                       if self.shared else [(model, checkpoint["config"], weight)
                                            for (model, checkpoint), weight in zip(self.loaded, self.blend_weights)])

    def run(self, manifest: str | Path, studies_csv: str | Path, output_csv: str | Path, batch_size: int = 2,
            num_workers: int = 4, started: float | None = None) -> Path:
        if batch_size < 1 or num_workers < 0:
            raise ValueError("batch_size must be positive; num_workers must be nonnegative")
        started = time.perf_counter() if started is None else started
        studies = pd.read_csv(studies_csv, dtype={STUDY_ID: str})
        if STUDY_ID not in studies or studies.empty or studies[STUDY_ID].isna().any() or studies[STUDY_ID].duplicated().any():
            raise ValueError("Study file must contain unique, nonmissing study IDs")
        output_csv = Path(output_csv)
        if output_csv.resolve() in {Path(studies_csv).resolve(), Path(manifest).resolve(),
                                    *(path.resolve() for path in self.checkpoints)}:
            raise ValueError("Prediction output must not overwrite an input")
        total = np.zeros((len(studies), len(TARGETS)), dtype=np.float64)
        sources = []
        for (_, checkpoint), path, digest, weight in zip(self.loaded, self.checkpoints, self.hashes, self.blend_weights,
                                                         strict=True):
            validate_dataset_manifest(manifest, checkpoint["config"]["data"], study_ids=studies[STUDY_ID].tolist())
            sources.append({"file": str(path), "sha256": digest,
                            "weight": dict(zip(TARGETS, weight.tolist())) if self.by_finding else float(weight[0])})
        for model, config, weight in self.passes:
            data = make_dataset(manifest, studies[[STUDY_ID]], config["data"], training=False, seed=config["seed"])
            settings = {"batch_size": batch_size, "num_workers": num_workers, "device": self.device}
            loader = make_loader(data, settings, training=False, seed=config["seed"])
            predictions, _ = predict_loader(
                model, loader, self.compute_device,
                inference_precision(self.compute_device, config["training"]["precision"]),
                test_time_mirror=config["data"].get("test_time_mirror", False),
                test_time_rotations=tuple(config["data"].get("test_time_rotations", ())),
            )
            aligned = predictions.set_index(STUDY_ID).loc[studies[STUDY_ID], TARGETS]
            total += weight * aligned.to_numpy()
        if not np.isfinite(total).all() or ((total < 0) | (total > 1)).any():
            raise FloatingPointError("Invalid final prediction probabilities")
        result = pd.DataFrame(total, columns=TARGETS)
        result.insert(0, STUDY_ID, studies[STUDY_ID].tolist())
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        result.to_csv(output_csv, index=False)
        write_json(output_csv.with_suffix(".provenance.json"), {
            "studies": len(studies), "seconds": time.perf_counter() - started,
            "checkpoints": sources, "manifest_sha256": file_hash(manifest),
            "studies_sha256": file_hash(studies_csv), "device": self.device, "shared_pass": self.shared,
            "precision": inference_precision(self.compute_device, self.configs[0]["training"]["precision"]),
            "accelerate": self.accelerate,
        })
        return output_csv


def predict(
    manifest: str | Path, studies_csv: str | Path, checkpoints: list[str | Path],
    output_csv: str | Path, batch_size: int = 2, num_workers: int = 4,
    device: str = "cuda", weights=None,
) -> Path:
    if not checkpoints:
        raise ValueError("At least one checkpoint is required")
    if batch_size < 1 or num_workers < 0:
        raise ValueError("batch_size must be positive; num_workers must be nonnegative")
    torch.set_num_threads(min(8, os.cpu_count() or 1))
    started = time.perf_counter()
    return Predictor(checkpoints, device, weights).run(manifest, studies_csv, output_csv, batch_size, num_workers,
                                                       started=started)
