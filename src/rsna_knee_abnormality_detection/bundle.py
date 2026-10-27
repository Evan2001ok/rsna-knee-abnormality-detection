"""Portable model exports and an offline entry point for Kaggle."""

import gc
import json
import math
import multiprocessing
import os
import queue
import shutil
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .constants import STUDY_ID, TARGETS
from .dataset import dataset_preprocess_version
from .metrics import read_predictions
from .predict import Predictor, blend_weights_by_finding, predict
from .prepare import prepare_data
from .utils import file_hash, save_checkpoint, write_json

RAPTOR_REQUIREMENTS = ["timm==1.0.26", "opencv-python-headless==4.12.0.88"]


def _extra_requirements(recipe: str, loaded: list[dict]) -> list[str]:
    """Raptor image preparation needs OpenCV; timm builds the CoAtNet and OrthoFoundation encoders (ResNets come from torchvision)."""
    if recipe != "raptor":
        return []
    from .models import TIMM_BACKBONES

    if any(item["config"]["model"]["backbone"] in TIMM_BACKBONES for item in loaded):
        return list(RAPTOR_REQUIREMENTS)
    return [requirement for requirement in RAPTOR_REQUIREMENTS if not requirement.startswith("timm==")]


def _recipe(data: dict) -> str:
    recipe = data.get("recipe", "standard")
    if recipe not in {"standard", "raptor"}:
        raise ValueError(f"Unknown bundle image recipe: {recipe}")
    return recipe


def _check_recipe_size(recipe: str, size: int, version: str) -> None:
    if recipe == "raptor":
        from .raptor_data import raptor_variant

        expected = raptor_variant(version)["size"]
        if size != expected:
            raise ValueError(f"Raptor image preparation {version} requires cache size {expected}")


def _finding_weights(weights, models: int) -> list[list[float]] | None:
    """Blend weights as stored in bundle.json: None (equal weights), or for each model either one number or a
    mapping from every finding to its weight; the weights of each finding need not sum to one."""
    if weights is None:
        return None
    if len(weights) != models:
        raise ValueError("Provide one weight entry per checkpoint")
    rows = []
    for entry in weights:
        if isinstance(entry, dict):
            if set(entry) != set(TARGETS):
                raise ValueError(f"Per-finding weights must name exactly these findings: {TARGETS}")
            rows.append([float(entry[name]) for name in TARGETS])
        else:
            rows.append([float(entry)] * len(TARGETS))
    blend_weights_by_finding(rows, models)  # the same checks as at prediction time
    return rows


def export_bundle(checkpoints: list[str | Path], output_dir: str | Path, wheelhouse: str | Path | None = None,
                  test_time_mirror: bool = False, weights=None, eval_windows: int | None = None,
                  test_time_rotations: tuple[float, ...] | None = None) -> Path:
    if not checkpoints:
        raise ValueError("At least one checkpoint is required")
    if test_time_rotations is not None:
        test_time_rotations = tuple(float(angle) for angle in test_time_rotations)
        if not test_time_rotations or not all(math.isfinite(a) and abs(a) <= 45 for a in test_time_rotations):
            raise ValueError("test_time_rotations must be finite angles in degrees within +-45")
        if test_time_mirror:
            raise ValueError("Mirrored and rotated test-time views are not combined")
    if eval_windows is not None and (isinstance(eval_windows, bool) or not isinstance(eval_windows, int) or eval_windows < 1):
        raise ValueError("eval_windows must be a positive integer")
    finding_weights = _finding_weights(weights, len(checkpoints))
    output = Path(output_dir)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Bundle output is not empty: {output}")
    loaded = [torch.load(path, map_location="cpu", weights_only=True) for path in checkpoints]
    if any(item.get("targets") != TARGETS or item.get("format_version") != 1 for item in loaded):
        raise ValueError("Incompatible checkpoint format or target order")
    recipes = {_recipe(item["config"]["data"]) for item in loaded}
    if len(recipes) != 1:
        raise ValueError("All bundled models must use the same image recipe; export standard and Raptor models separately")
    recipe = recipes.pop()
    versions = {dataset_preprocess_version(item["config"]["data"]) for item in loaded}
    if len(versions) != 1:
        raise ValueError("All bundled models must use the same image preparation; export each cache layout separately")
    preprocess_version = versions.pop()
    if any(item.get("image_preparation", {}).get("preprocess_version") != preprocess_version for item in loaded):
        raise ValueError("Checkpoint image preparation version does not match the exported pipeline")
    if any(item["image_preparation"].get("image_size") != item["config"]["data"]["cache_size"] for item in loaded):
        raise ValueError("Checkpoint image size does not match its data configuration")
    sizes = {item["config"]["data"]["cache_size"] for item in loaded}
    if len(sizes) != 1:
        raise ValueError("All bundled models must use the same prepared image size")
    cache_size = sizes.pop()
    _check_recipe_size(recipe, cache_size, preprocess_version)
    if test_time_mirror and recipe != "raptor":
        raise ValueError("Mirrored predictions need the Raptor image recipe")
    if eval_windows is not None and recipe != "raptor":
        raise ValueError("A prediction window count needs the Raptor image recipe")
    if test_time_rotations is not None and recipe != "raptor":
        raise ValueError("Rotated test-time views need the Raptor image recipe")
    source_root = Path(__file__).resolve().parents[2]
    attribution_paths = [Path("third_party/NOTICE"), Path("third_party/licenses/Apache-2.0.txt")]
    if any(not (source_root / path).is_file() for path in attribution_paths):
        raise FileNotFoundError("Third-party attribution files are missing; export from a complete repository checkout")
    output.mkdir(parents=True, exist_ok=True)
    destination = output / "src" / "rsna_knee_abnormality_detection"
    destination.mkdir(parents=True)
    for path in Path(__file__).parent.glob("*.py"):
        shutil.copy2(path, destination / path.name)
    for path in attribution_paths:
        (output / path).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_root / path, output / path)
    exports = []
    for i, checkpoint in enumerate(loaded):
        filename = f"model-{i}.pt"
        # Training optimizer state and study IDs are unnecessary for prediction.
        lean = {key: checkpoint[key] for key in ["format_version", "targets", "config", "model", "epoch"]}
        if test_time_mirror:
            # Average each study with its left/right mirror image, as in scripts/evaluate_mirror_tta.py.
            lean["config"] = {**lean["config"], "data": {**lean["config"]["data"], "test_time_mirror": True}}
        if eval_windows is not None:
            # Fewer windows per study at prediction time (56 matched 112 on all five folds trained on v7, our round-5
            # labels, plus OAI knees; Oct 15, 2026).
            lean["config"] = {**lean["config"], "data": {**lean["config"]["data"], "raptor_eval_windows": eval_windows}}
        if test_time_rotations is not None:
            # In-plane rotated views averaged per study (Oct 16, 2026).
            lean["config"] = {**lean["config"], "data": {**lean["config"]["data"],
                                                         "test_time_rotations": list(test_time_rotations)}}
        lean["image_preparation"] = {
            "preprocess_version": checkpoint["image_preparation"]["preprocess_version"],
            "image_size": checkpoint["image_preparation"]["image_size"],
        }
        save_checkpoint(output / filename, lean)
        exports.append({"file": filename, "sha256": file_hash(output / filename),
                        **({"weights": dict(zip(TARGETS, finding_weights[i]))} if finding_weights else {})})
    if wheelhouse is not None:
        wheels = list(Path(wheelhouse).glob("*.whl"))
        if not wheels:
            raise ValueError("The supplied wheelhouse has no .whl files")
        (output / "wheelhouse").mkdir()
        for path in wheels:
            shutil.copy2(path, output / "wheelhouse" / path.name)
        if (Path(wheelhouse) / "wheelhouse.json").exists():
            shutil.copy2(Path(wheelhouse) / "wheelhouse.json", output / "wheelhouse" / "wheelhouse.json")
    write_json(output / "bundle.json", {
        "format_version": 1, "recipe": recipe, "preprocess_version": preprocess_version,
        "cache_size": cache_size, "targets": TARGETS, "models": exports,
        "extra_inference_requirements": _extra_requirements(recipe, loaded), "test_time_mirror": test_time_mirror,
        **({"test_time_rotations": list(test_time_rotations)} if test_time_rotations is not None else {}),
    })
    (output / "run_inference.py").write_text('''"""Run from any working directory, without downloading model weights."""
import argparse
import sys
from pathlib import Path

bundle = Path(__file__).resolve().parent
sys.path.insert(0, str(bundle / "src"))
from rsna_knee_abnormality_detection.bundle import run_bundle

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--work-dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--studies-per-chunk", type=int, default=16)
    args = parser.parse_args()
    print(run_bundle(bundle, args.data_root, args.work_dir, args.device, args.batch_size, args.workers, args.studies_per_chunk))
''')
    notebook_code = '''from pathlib import Path
import hashlib
import importlib.metadata
import json
import math
import subprocess
import sys

# Add the exported bundle as a Kaggle dataset input before running this notebook.
import os

kaggle_input = Path(os.environ.get("KNEE_KAGGLE_INPUT", "/kaggle/input"))
# The competition data holds hundreds of thousands of DICOM files; search only the attached datasets.
competition_folders = {"competitions", "rsna-knee-abnormality-detection"}
dataset_roots = (sorted(path for path in kaggle_input.iterdir() if path.is_dir() and path.name not in competition_folders)
                 if kaggle_input.is_dir() else [])


def find(pattern):
    return [match for root in dataset_roots for match in root.glob(pattern)]


candidates = find("**/bundle.json")
if not candidates:
    # Kaggle may keep an uploaded archive as one zip file instead of extracting it.
    import tempfile
    import zipfile

    def holds_bundle(path):
        with zipfile.ZipFile(path) as archive:
            return any(Path(name).name == "bundle.json" for name in archive.namelist())

    archives = [path for path in find("**/*.zip") if holds_bundle(path)]
    if len(archives) == 1:
        target = Path(tempfile.mkdtemp(prefix="knee-bundle-"))
        with zipfile.ZipFile(archives[0]) as archive:
            if any(Path(name).is_absolute() or ".." in Path(name).parts for name in archive.namelist()):
                raise RuntimeError("Unsafe path in the bundle archive.")
            archive.extractall(target)
        candidates = list(target.glob("**/bundle.json"))
if len(candidates) != 1:
    raise RuntimeError("Attach exactly one exported knee pipeline bundle.")
bundle = candidates[0].parent
bundle_metadata = json.loads((bundle / "bundle.json").read_text())
recipe = bundle_metadata.get("recipe", "standard")
if recipe not in {"standard", "raptor"}:
    raise RuntimeError(f"Unknown bundle image recipe: {recipe}")
extra_requirements = bundle_metadata.get("extra_inference_requirements", [])
wheelhouse = bundle / "wheelhouse"
if wheelhouse.exists():
    wheel_metadata = wheelhouse / "wheelhouse.json"
    if wheel_metadata.exists():
        record = json.loads(wheel_metadata.read_text())
        current_python = f"{sys.version_info.major}.{sys.version_info.minor}"
        if record["python_version"] != current_python:
            raise RuntimeError(f"Dependency wheels target Python {record['python_version']}; runtime is {current_python}.")
        for item in record["wheels"]:
            if hashlib.sha256((wheelhouse / item["file"]).read_bytes()).hexdigest() != item["sha256"]:
                raise RuntimeError(f"Dependency wheel checksum mismatch: {item['file']}")
    import numpy as np
    if int(np.__version__.split(".")[0]) != 2:
        raise RuntimeError("These decoders require NumPy 2; use a compatible Kaggle runtime before installing them.")
    subprocess.run([sys.executable, "-m", "pip", "install", "--no-index", "--find-links", str(wheelhouse),
                    "pydicom", "pylibjpeg", "pylibjpeg-libjpeg", "pylibjpeg-openjpeg", "pyjpegls", "PyYAML",
                    *extra_requirements], check=True)
for requirement in extra_requirements:
    name, expected = requirement.split("==")
    try:
        installed = importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        installed = "missing"
    if installed != expected:
        raise RuntimeError(f"Raptor inference requires {requirement}; found {installed}. "
                           "Attach an offline wheelhouse containing this version and its dependencies, "
                           "or prepare a compatible Kaggle runtime before running this notebook.")
sys.path.insert(0, str(bundle / "src"))
from rsna_knee_abnormality_detection.bundle import run_bundle
# Newer Kaggle images mount competition data under /kaggle/input/competitions/.
data_roots = [kaggle_input / "competitions" / "rsna-knee-abnormality-detection", kaggle_input / "rsna-knee-abnormality-detection"]
data_root = next((root for root in data_roots if (root / "test.csv").is_file()), data_roots[-1])
print(json.dumps({"bundle": str(bundle), "data_root": str(data_root), "python": sys.version.split()[0]}), flush=True)
# Kaggle stops a run at 9 hours; studies not reached after 8 get the average prediction instead.
try:
    run_bundle(bundle, str(data_root), "/kaggle/working", "cuda", 2, 4, deadline_seconds=8 * 3600,
               separate_processes=True)
except Exception as error:
    print(json.dumps({"event": "run_failed", "error": repr(error)[:1000]}), flush=True)
    output = Path("/kaggle/working/submission.csv")
    if not output.is_file():
        # Last resort: a complete file with every finding at 0.5, so the run ends cleanly and is scored.
        import pandas as pd

        ids = pd.read_csv(data_root / "test.csv", dtype=str)["StudyInstanceUID"]
        pd.DataFrame({"StudyInstanceUID": ids, **{name: 0.5 for name in bundle_metadata["targets"]}}).to_csv(output, index=False)
'''
    write_json(output / "inference.ipynb", {
        "nbformat": 4, "nbformat_minor": 5,
        "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}},
        "cells": [{"cell_type": "code", "id": "offline-inference", "metadata": {}, "execution_count": None,
                   "outputs": [], "source": notebook_code.splitlines(keepends=True)}],
    })
    return output


def prediction_devices(device: str, models: int) -> list[str]:
    """Kaggle has two T4s: with several models on "cuda", every visible GPU predicts its own share of them."""
    count = torch.cuda.device_count() if device == "cuda" and torch.cuda.is_available() else 0
    if models >= 2 and count >= 2:
        return [f"cuda:{index}" for index in range(min(models, count))]
    return [device]


def split_models(checkpoints: list, devices: list[str]) -> list[list]:
    """Deal the models out in turn, so every device gets the same number (give or take one)."""
    return [checkpoints[index::len(devices)] for index in range(len(devices))]


def _share_weights(weights: np.ndarray, devices: list[str]) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Each device's weights (dealt out like `split_models`) and the share of every finding's total weight it holds.

    A device whose models have no weight for a finding still predicts it, with equal weights, but contributes nothing.
    """
    shares = split_models(list(weights), devices)
    mass = [np.sum(share, axis=0) for share in shares]
    total = np.sum(mass, axis=0)
    usable = [np.where(part > 0, np.asarray(share), 1.0) for share, part in zip(shares, mass, strict=True)]
    return usable, [part / total for part in mass]


def _predict_share(manifest, group_path, share, part, batch_size, workers, device, weights=None):
    """Predict one device's share of the models for one group of studies."""
    with torch.cuda.device(device) if device.startswith("cuda") else nullcontext():
        predict(manifest, group_path, share, part, batch_size=batch_size, num_workers=workers, device=device,
                weights=weights)


def _run_in_fresh_processes(jobs):
    """Run each job in its own new process and wait for all of them.

    Memory that GPU prediction leaves behind in a long-running process (about 0.7 GB per 16-study group with 15
    models on Kaggle, enough to exhaust its 32 GB after about 20 groups) is returned to the system when each process
    exits, so memory stays flat however many groups follow.
    """
    context = multiprocessing.get_context("spawn")
    processes = [context.Process(target=_predict_share, args=job) for job in jobs]
    for process in processes:
        process.start()
    for process in processes:
        process.join()
    failed = [process.exitcode for process in processes if process.exitcode != 0]
    if failed:
        raise RuntimeError(f"A prediction process failed (exit codes {failed})")


def _predict_on_devices(manifest, group_path, checkpoints, destination, devices, batch_size, workers,
                        separate_processes=False, weights=None):
    """Each device predicts its share of the models at the same time, on the one prepared image cache.

    `predict` averages its models' probabilities with their weights (equal by default), so weighting each
    share's average by its part of every finding's total weight gives exactly the single-device average.
    """
    weights = blend_weights_by_finding(weights, len(checkpoints))
    if len(devices) == 1:
        job = (manifest, group_path, checkpoints, destination, batch_size, workers, devices[0], weights)
        _run_in_fresh_processes([job]) if separate_processes else _predict_share(*job)
        return read_predictions(destination)
    shares = split_models(checkpoints, devices)
    share_weights, share_mass = _share_weights(weights, devices)
    # Kaggle has only 4 CPU cores; the devices' image loaders share them.
    loader_workers = max(1, workers // len(devices))

    def run(index):
        device = devices[index]
        part = destination.with_name(f"{destination.stem}-device{index}.csv")
        # Keep each thread's own CUDA work (pinned memory, kernels) on its own GPU.
        with torch.cuda.device(device) if device.startswith("cuda") else nullcontext():
            predict(manifest, group_path, shares[index], part, batch_size=batch_size, num_workers=loader_workers,
                    device=device, weights=share_weights[index])
        return read_predictions(part).set_index(STUDY_ID)[TARGETS]

    # Leaving the pool waits for every device, so the image cache outlives all readers; a failure on any
    # device is raised here once all have stopped, and the caller then retries the studies one by one.
    if separate_processes:
        names = [destination.with_name(f"{destination.stem}-device{index}.csv") for index in range(len(devices))]
        _run_in_fresh_processes([(manifest, group_path, shares[index], names[index], batch_size, loader_workers,
                                  devices[index], share_weights[index]) for index in range(len(devices))])
        parts = [read_predictions(name).set_index(STUDY_ID)[TARGETS] for name in names]
    else:
        with ThreadPoolExecutor(max_workers=len(devices)) as pool:
            parts = list(pool.map(run, range(len(devices))))
    order = parts[0].index
    total = sum(part.loc[order] * mass for mass, part in zip(share_mass, parts, strict=True))
    total.reset_index().to_csv(destination, index=False)
    return read_predictions(destination)


def _device_worker(share, device, weights, jobs, results, accelerate=False):
    """One device's long-lived prediction process: loads its share of the models once, then predicts every group sent
    to it with exactly `predict`'s code (`Predictor.run`), writing the same per-device file the short-lived processes
    write."""
    try:
        torch.set_num_threads(min(8, os.cpu_count() or 1))
        with torch.cuda.device(device) if device.startswith("cuda") else nullcontext():
            predictor = Predictor(share, device, weights, accelerate=accelerate)
            results.put(("ready", os.getpid()))
            while True:
                job = jobs.get()
                if job is None:
                    return
                manifest, studies_csv, output, batch_size, workers = job
                try:
                    predictor.run(manifest, studies_csv, output, batch_size, workers)
                    results.put(("ok", output))
                except Exception as error:
                    results.put(("error", repr(error)[:500]))
                gc.collect()
                if device.startswith("cuda"):
                    torch.cuda.empty_cache()
    except Exception as error:
        results.put(("error", "the models could not be loaded: " + repr(error)[:500]))


def _process_rss(pid: int) -> int:
    try:
        with open(f"/proc/{pid}/status") as handle:
            for line in handle:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) * 1024
    except OSError:
        pass
    return 0


def _container_anon_memory() -> int | None:
    """Memory the container's processes hold themselves (page cache excluded: it is given back on demand)."""
    for path, key in (("/sys/fs/cgroup/memory.stat", "anon"), ("/sys/fs/cgroup/memory/memory.stat", "total_rss")):
        try:
            with open(path) as handle:
                for line in handle:
                    name, value = line.split()[:2]
                    if name == key:
                        return int(value)
        except OSError:
            continue
    return None


class _DeviceWorkers:
    """Long-lived prediction processes, one per device, each holding its share of the models (`_device_worker`).

    A group is predicted exactly as by `_predict_on_devices` (the same per-device files, combined with the same
    weights), but the models are loaded and hashed once instead of once per group. A worker that dies or stops answering
    is replaced and the group fails, so the caller retries its studies one by one; a worker whose memory grows past
    `memory_limit_bytes`, or any worker while the container's own memory is past `container_limit_bytes`, is replaced
    between groups (the reason short-lived processes were used before).
    """

    before_job = None  # tests only: called as before_job(workers, device_index) just before a job is sent

    def __init__(self, checkpoints, devices, weights, batch_size, workers, timeout_seconds=1800,
                 memory_limit_bytes=12 * 2**30, container_limit_bytes=27 * 2**30, accelerate=False):
        self.devices = list(devices)
        self.accelerate = accelerate
        weights = blend_weights_by_finding(weights, len(checkpoints))
        if len(self.devices) == 1:
            self.shares, self.share_weights, self.share_mass = [list(checkpoints)], [weights], None
            self.loader_workers = workers
        else:
            self.shares = split_models(list(checkpoints), self.devices)
            self.share_weights, self.share_mass = _share_weights(weights, self.devices)
            # Kaggle has only 4 CPU cores; the devices' image loaders share them.
            self.loader_workers = max(1, workers // len(self.devices))
        self.batch_size, self.timeout = batch_size, timeout_seconds
        self.memory_limit, self.container_limit = memory_limit_bytes, container_limit_bytes
        self.context = multiprocessing.get_context("spawn")
        self.processes, self.jobs, self.results = [None] * len(self.devices), [None] * len(self.devices), [None] * len(self.devices)
        self.restarts = 0
        try:
            for index in range(len(self.devices)):
                self._start(index)
        except BaseException:
            self.close()
            raise

    def _start(self, index):
        jobs, results = self.context.Queue(), self.context.Queue()
        # Not daemonic: each worker's image loader starts processes of its own. run_bundle always closes the workers.
        process = self.context.Process(target=_device_worker, args=(
            self.shares[index], self.devices[index], self.share_weights[index], jobs, results, self.accelerate))
        previous = os.environ.get("MALLOC_ARENA_MAX")
        os.environ["MALLOC_ARENA_MAX"] = "2"  # fewer malloc arenas: less memory left behind by the image loaders
        try:
            process.start()
        finally:
            if previous is None:
                os.environ.pop("MALLOC_ARENA_MAX", None)
            else:
                os.environ["MALLOC_ARENA_MAX"] = previous
        status, detail = self._answer(results, process)
        if status != "ready":
            if process.is_alive():
                process.kill()
            process.join()
            raise RuntimeError(f"The prediction worker on {self.devices[index]} did not start: {detail}")
        self.processes[index], self.jobs[index], self.results[index] = process, jobs, results

    def _answer(self, results, process):
        deadline = time.monotonic() + self.timeout
        while True:
            try:
                return results.get(timeout=2)
            except queue.Empty:
                if not process.is_alive():
                    return "died", f"exit code {process.exitcode}"
                if time.monotonic() > deadline:
                    return "timeout", f"no answer within {self.timeout} s"

    def _stop(self, index, graceful=True):
        process = self.processes[index]
        if process is None:
            return
        if graceful and process.is_alive():
            self.jobs[index].put(None)
            process.join(timeout=60)
        if process.is_alive():
            process.kill()
        process.join()
        self.processes[index] = None

    def _replace(self, index, why):
        print(json.dumps({"event": "prediction_worker_replaced", "device": self.devices[index], "why": why}), flush=True)
        self._stop(index, graceful=why == "memory")
        self.restarts += 1
        self._start(index)

    def predict(self, manifest, group_path, destination):
        destination = Path(destination)
        names = ([destination] if len(self.devices) == 1 else
                 [destination.with_name(f"{destination.stem}-device{index}.csv") for index in range(len(self.devices))])
        for index, name in enumerate(names):
            if type(self).before_job is not None:
                type(self).before_job(self, index)
            self.jobs[index].put((str(manifest), str(group_path), str(name), self.batch_size, self.loader_workers))
        problems = []
        for index in range(len(names)):
            status, detail = self._answer(self.results[index], self.processes[index])
            if status in ("died", "timeout"):
                self._replace(index, status)
            if status != "ok":
                problems.append(f"{self.devices[index]}: {detail}")
        if problems:
            raise RuntimeError("Prediction failed: " + "; ".join(problems))
        self._check_memory()
        if len(self.devices) == 1:
            return read_predictions(destination)
        parts = [read_predictions(name).set_index(STUDY_ID)[TARGETS] for name in names]
        order = parts[0].index
        total = sum(part.loc[order] * mass for mass, part in zip(self.share_mass, parts, strict=True))
        total.reset_index().to_csv(destination, index=False)
        return read_predictions(destination)

    def _check_memory(self):
        for index, process in enumerate(self.processes):
            container = _container_anon_memory()
            if _process_rss(process.pid) > self.memory_limit or (container is not None and container > self.container_limit):
                self._replace(index, "memory")

    def close(self):
        for index in range(len(self.processes)):
            self._stop(index)


def _bundle_weights(metadata: dict) -> list[list[float]] | None:
    """The per-finding blend weights stored in bundle.json, or None for equal weights (older bundles)."""
    stored = ["weights" in item for item in metadata["models"]]
    if not any(stored):
        return None
    if not all(stored):
        raise ValueError("Either every bundled model has blend weights or none does")
    return [[float(item["weights"][name]) for name in TARGETS] for item in metadata["models"]]


def _prepare_group(recipe, metadata, data_root, work, parts, label, study_ids, workers):
    """Write one group's study list and prepare its images in its own temporary cache, which the caller removes."""
    group_path = parts / f"studies-{label}.csv"
    group_path.write_text(pd.DataFrame({STUDY_ID: study_ids}).to_csv(index=False))
    # Each group gets its own owned temporary directory. Only that directory
    # is removed; model files, inputs, predictions, and QC records are retained.
    temporary = tempfile.TemporaryDirectory(prefix="knee-images-", dir=work)
    try:
        qc_path = Path(temporary.name) / "qc.json"
        try:
            if recipe == "raptor":
                from .raptor_data import prepare_raptor_data

                # Bad-series insurance: an unreadable or inconsistent series is dropped (its slot takes the next
                # series of the same plane or stays empty) instead of costing the study its prediction.
                manifest = prepare_raptor_data(Path(data_root), Path(temporary.name), split="test", workers=workers,
                                               study_ids=study_ids, version=metadata["preprocess_version"],
                                               drop_bad_series=True)
            else:
                manifest = prepare_data(Path(data_root), Path(temporary.name), split="test", size=metadata["cache_size"],
                                        workers=workers, study_ids=study_ids)
        finally:
            # Raptor preparation raises after writing QC for a failed study.
            # Retain that explanation before the owned image cache is deleted.
            if qc_path.is_file():
                shutil.copy2(qc_path, parts / f"qc-{label}.json")
        summary = json.loads(qc_path.read_text())["summary"]
    except BaseException:
        temporary.cleanup()
        raise
    return group_path, temporary, manifest, summary


def _predict_group(recipe, metadata, data_root, work, parts, label, study_ids, predict_group, workers):
    """Prepare and predict one group of studies in its own temporary image cache."""
    group_path, temporary, manifest, summary = _prepare_group(recipe, metadata, data_root, work, parts, label,
                                                              study_ids, workers)
    try:
        return predict_group(manifest, group_path, parts / f"predictions-{label}.csv"), summary
    finally:
        temporary.cleanup()


def run_bundle(bundle_dir, data_root, work_dir, device="cuda", batch_size=2, workers=4, studies_per_chunk=16,
               deadline_seconds: float | None = None, devices: list[str] | None = None,
               separate_processes: bool = False, persistent: bool = False, prepare_ahead: bool = False,
               accelerate: bool = False) -> Path:
    """Predict every test study. A study that cannot be read or predicted, or is not reached before the deadline,
    gets the average prediction of the other studies (a neutral middle rank) instead of stopping the run;
    only when no study at all can be predicted does the run fail. With the Raptor recipe one bad series no longer
    makes its study unreadable: that series is dropped (listed under "dropped_series" in the inference summary).

    The models are split across `devices` (by default every visible GPU when `device` is "cuda"). With
    `separate_processes`, every group is predicted in new processes that exit afterwards, so memory does not build up
    over a long run (the Kaggle notebook uses this). With `persistent` (Oct 16, 2026), one long-lived process per
    device keeps its models loaded for every group instead, and is replaced if its memory grows too much; with
    `prepare_ahead`, the next group's images are prepared while the current group is predicted. Both give the same
    predictions; they only remove waiting. `accelerate` (with `persistent` only, as compiling takes about a minute per
    process) compiles the encoders and uses faster GPU layouts: within 0.0002 of the plain predictions on a T4."""
    bundle, work = Path(bundle_dir), Path(work_dir)
    if studies_per_chunk < 1 or workers < 1:
        raise ValueError("studies_per_chunk and workers must be positive")
    if accelerate and not persistent:
        raise ValueError("accelerate needs persistent workers: compiling once per group would cost more than it saves")
    started = time.perf_counter()
    metadata = json.loads((bundle / "bundle.json").read_text())
    recipe = _recipe(metadata)
    expected = dataset_preprocess_version({"recipe": recipe, "raptor_variant": metadata["preprocess_version"]}
                                          if recipe == "raptor" else {"recipe": recipe})
    if metadata["targets"] != TARGETS or metadata["preprocess_version"] != expected:
        raise ValueError("Bundle targets or image preparation do not match the packaged code")
    _check_recipe_size(recipe, metadata["cache_size"], metadata["preprocess_version"])
    checkpoints = []
    for item in metadata["models"]:
        path = bundle / item["file"]
        if file_hash(path) != item["sha256"]:
            raise ValueError(f"Model checksum mismatch: {path.name}")
        checkpoints.append(path)
    blend_weights_by_finding(_bundle_weights(metadata), len(checkpoints))  # reject bad weights before reading images
    devices = list(devices) if devices is not None else prediction_devices(device, len(checkpoints))
    if not devices:
        raise ValueError("At least one prediction device is required")
    devices = devices[:len(checkpoints)]
    studies = pd.read_csv(Path(data_root) / "test.csv", dtype={STUDY_ID: str})
    if studies.empty or studies[STUDY_ID].isna().any() or studies[STUDY_ID].duplicated().any():
        raise ValueError("Test studies must be nonempty and have unique IDs")
    work.mkdir(parents=True, exist_ok=True)
    parts = work / "prediction-parts"
    parts.mkdir(exist_ok=True)
    frames, quality, failures, skipped = [], [], [], []
    group_args = (recipe, metadata, data_root, work, parts)
    weights = _bundle_weights(metadata)
    pool = _DeviceWorkers(checkpoints, devices, weights, batch_size, workers, accelerate=accelerate) if persistent else None

    def predict_group(manifest, group_path, destination):
        if pool is not None:
            return pool.predict(manifest, group_path, destination)
        return _predict_on_devices(manifest, group_path, checkpoints, destination, devices, batch_size, workers,
                                   separate_processes, weights)

    groups = [(start, studies.iloc[start:start + studies_per_chunk][STUDY_ID].tolist())
              for start in range(0, len(studies), studies_per_chunk)]

    def late():
        return deadline_seconds is not None and time.perf_counter() - started > deadline_seconds

    ahead = ThreadPoolExecutor(max_workers=1) if prepare_ahead else None

    def prepare(index):
        start, ids = groups[index]
        return _prepare_group(*group_args, f"{start:06d}", ids, workers)

    def discard(future):
        try:
            future.result()[1].cleanup()
        except Exception:
            pass

    pending = ahead.submit(prepare, 0) if ahead is not None and groups and not late() else None
    try:
        for index, (start, ids) in enumerate(groups):
            if late():
                skipped.extend(ids)
                if pending is not None:
                    discard(pending)
                    pending = None
                continue
            try:
                if ahead is not None:
                    current = pending if pending is not None else ahead.submit(prepare, index)
                    pending = ahead.submit(prepare, index + 1) if index + 1 < len(groups) and not late() else None
                    group_path, temporary, manifest, summary = current.result()
                    try:
                        frame = predict_group(manifest, group_path, parts / f"predictions-{start:06d}.csv")
                    finally:
                        temporary.cleanup()
                else:
                    frame, summary = _predict_group(*group_args, f"{start:06d}", ids, predict_group, workers)
                frames.append(frame)
                quality.append(summary)
            except Exception as error:  # one unreadable study must not cost the whole group: retry each study alone
                print(json.dumps({"event": "group_failed", "first_study": start, "error": repr(error)[:300]}), flush=True)
                for offset, study in enumerate(ids):
                    try:
                        frame, summary = _predict_group(*group_args, f"{start:06d}-{offset:02d}", [study],
                                                        predict_group, workers)
                        frames.append(frame)
                        quality.append(summary)
                    except Exception as single_error:
                        failures.append({STUDY_ID: study, "error": repr(single_error)[:500]})
            print(json.dumps({"event": "prediction", "finished_studies": start + len(ids), "total_studies": len(studies),
                              "failed_studies": len(failures), "seconds": time.perf_counter() - started}), flush=True)
    finally:
        if pending is not None:
            discard(pending)
        if ahead is not None:
            ahead.shutdown(wait=True)
        restarts = pool.restarts if pool is not None else 0
        if pool is not None:
            pool.close()
    if not frames:
        first = failures[0]["error"] if failures else "no study was attempted before the deadline"
        raise RuntimeError(f"No study could be predicted; first error: {first}")
    predicted = pd.concat(frames, ignore_index=True).set_index(STUDY_ID)[TARGETS]
    missing = [study for study in studies[STUDY_ID] if study not in predicted.index]
    if missing:
        fill = pd.DataFrame([predicted.mean().to_numpy()] * len(missing), columns=TARGETS,
                            index=pd.Index(missing, name=STUDY_ID))
        predicted = pd.concat([predicted, fill])
    final = predicted.loc[studies[STUDY_ID]].reset_index()
    output = work / "submission.csv"
    final[[STUDY_ID, *TARGETS]].to_csv(output, index=False)
    read_predictions(output)
    report = {
        "studies": len(studies), "seconds": time.perf_counter() - started,
        "studies_per_chunk": studies_per_chunk, "device": device, "devices_used": len(devices),
        "models_per_device": [{"device": name, "models": [path.name for path in share]}
                              for name, share in zip(devices, split_models(checkpoints, devices), strict=True)],
        "recipe": recipe, "preprocess_version": metadata["preprocess_version"],
        "models": metadata["models"], "quality": quality,
        "failed_studies": failures, "skipped_after_deadline": len(skipped), "filled_with_average": len(missing),
        "filled_study_ids": missing, "persistent_workers": persistent, "prepare_ahead": prepare_ahead, "accelerate": accelerate,
        "worker_restarts": restarts,
    }
    # Series dropped by bad-series insurance (study, slot, series, error, replacement); listed only when there are any.
    dropped = [item for summary in quality for item in summary.get("dropped_series", [])]
    if dropped:
        report["dropped_series"] = dropped
    write_json(work / "inference-summary.json", report)
    return output
