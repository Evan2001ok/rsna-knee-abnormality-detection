"""Offline export routes the trained Raptor recipe through real DICOM preparation. The bundle is what `knee export`
writes for Kaggle: weights, code, offline install files and a notebook."""

import importlib.metadata
import json
import socket
import subprocess
import sys
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch
from test_prepare import make_dataset
from test_raptor_data import make_raw
from torch import nn

from rsna_knee_abnormality_detection import bundle as bundle_module
from rsna_knee_abnormality_detection import raptor_data
from rsna_knee_abnormality_detection.bundle import export_bundle, run_bundle
from rsna_knee_abnormality_detection.config import DEFAULTS
from rsna_knee_abnormality_detection.constants import SERIES_ID, STUDY_ID, TARGETS
from rsna_knee_abnormality_detection.imaging import PREPROCESS_VERSION
from rsna_knee_abnormality_detection.models import RAPTOR_BACKBONE, StudyModel
from rsna_knee_abnormality_detection.predict import load_model, predict
from rsna_knee_abnormality_detection.utils import file_hash, save_checkpoint


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    connect = socket.socket.connect

    def local_only(sock, address):
        if sock.family in {socket.AF_INET, socket.AF_INET6}:
            raise AssertionError("Raptor bundle tests must not use the network")
        return connect(sock, address)

    monkeypatch.setattr(socket.socket, "connect", local_only)
    yield
    torch.set_num_threads(previous_threads)


@pytest.fixture
def checkpoint(tmp_path, monkeypatch):
    """Exercise the real adapter/head with a tiny replacement image encoder."""
    calls = []

    class TinyEncoder(nn.Module):
        num_features = 8

        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(3, self.num_features, 1)

        def forward(self, images):
            return self.conv(images).mean(dim=(-2, -1))

    def create_model(name, **kwargs):
        assert name == RAPTOR_BACKBONE
        assert kwargs["pretrained"] is False
        calls.append(kwargs)
        return TinyEncoder()

    monkeypatch.setitem(sys.modules, "timm", SimpleNamespace(create_model=create_model))
    settings = deepcopy(DEFAULTS)
    settings["data"].update({"recipe": "raptor", "cache_size": raptor_data.RAPTOR_SIZE,
                             "image_size": 32, "raptor_eval_windows": 2})
    settings["model"].update({"backbone": RAPTOR_BACKBONE, "pooling": "class_attention", "pretrained": True,
                              "input_size": 32, "encoder_chunk_size": 4, "use_metadata": False,
                              "use_positions": False, "grad_checkpointing": False})
    settings["training"]["precision"] = "fp32"
    model = StudyModel(settings["model"], pretrained=False)
    path = tmp_path / "raptor.pt"
    save_checkpoint(path, {
        "format_version": 1, "targets": TARGETS, "config": settings, "model": model.state_dict(), "epoch": 2,
        "image_preparation": {"preprocess_version": raptor_data.RAPTOR_PREPROCESS_VERSION,
                              "image_size": raptor_data.RAPTOR_SIZE},
        "optimizer": {"unneeded": torch.ones(1)}, "provenance": {"training_ids": ["must-not-export"]},
    })
    return path, calls


def notebook_source(bundle):
    notebook = json.loads((bundle / "inference.ipynb").read_text())
    return "\n".join("".join(cell["source"]) for cell in notebook["cells"] if cell["cell_type"] == "code")


def execute_notebook_preflight(bundle, monkeypatch):
    # The bundle's parent stands in for /kaggle/input, with the bundle as an attached dataset.
    monkeypatch.setenv("KNEE_KAGGLE_INPUT", str(bundle.parent))
    monkeypatch.setattr(sys, "path", list(sys.path))
    calls = []
    monkeypatch.setattr(bundle_module, "run_bundle", lambda *args, **kwargs: calls.append((args, kwargs)))
    source = notebook_source(bundle)
    exec(compile(source, "inference.ipynb", "exec"), {})
    return calls


def test_raptor_export_retains_recipe_licenses_and_loads_without_download(checkpoint, tmp_path):
    path, constructors = checkpoint
    output = export_bundle([path], tmp_path / "bundle")
    metadata = json.loads((output / "bundle.json").read_text())
    assert metadata["recipe"] == "raptor"
    assert metadata["preprocess_version"] == raptor_data.RAPTOR_PREPROCESS_VERSION
    assert metadata["cache_size"] == 336
    assert metadata["extra_inference_requirements"] == bundle_module.RAPTOR_REQUIREMENTS
    model_path = output / metadata["models"][0]["file"]
    assert file_hash(model_path) == metadata["models"][0]["sha256"]
    original_model, _ = load_model(path, torch.device("cpu"))
    restored, lean = load_model(model_path, torch.device("cpu"))
    assert "optimizer" not in lean and "provenance" not in lean
    assert lean["config"]["data"]["recipe"] == "raptor"
    assert len(constructors) == 3 and all(call["pretrained"] is False for call in constructors)
    images = torch.rand(2, 1, 2, 3, 32, 32)
    mask, info = torch.ones(2, 1, dtype=torch.bool), torch.zeros(2, 1, 5)
    with torch.no_grad():
        torch.testing.assert_close(restored(images, mask, info), original_model(images, mask, info), rtol=0, atol=0)
    source_root = Path(__file__).resolve().parents[1]
    for relative in ["third_party/NOTICE", "third_party/licenses/Apache-2.0.txt"]:
        assert (output / relative).read_bytes() == (source_root / relative).read_bytes()
    compile(notebook_source(output), "inference.ipynb", "exec")


def test_resnet_raptor_export_needs_only_opencv_and_can_mirror(checkpoint, tmp_path):
    path, _ = checkpoint
    stored = torch.load(path, map_location="cpu", weights_only=True)
    stored["config"]["model"]["backbone"] = "resnet18"
    stored["model"] = StudyModel(stored["config"]["model"], pretrained=False).state_dict()
    resnet = tmp_path / "resnet.pt"
    save_checkpoint(resnet, stored)
    output = export_bundle([resnet], tmp_path / "bundle", test_time_mirror=True)
    metadata = json.loads((output / "bundle.json").read_text())
    assert metadata["extra_inference_requirements"] == ["opencv-python-headless==4.12.0.88"]
    assert metadata["test_time_mirror"] is True
    _, lean = load_model(output / metadata["models"][0]["file"], torch.device("cpu"))
    assert lean["config"]["data"]["test_time_mirror"] is True
    assert torch.load(resnet, map_location="cpu", weights_only=True)["config"]["data"].get("test_time_mirror", False) is False
    assert "timm" not in notebook_source(output) or "extra_inference_requirements" in notebook_source(output)


def test_raptor_export_can_store_rotated_views_and_leaves_the_default_unchanged(checkpoint, tmp_path):
    path, _ = checkpoint
    plain = export_bundle([path], tmp_path / "plain")
    assert "test_time_rotations" not in json.loads((plain / "bundle.json").read_text())
    output = export_bundle([path], tmp_path / "bundle", test_time_rotations=(-12, 0, 12))
    metadata = json.loads((output / "bundle.json").read_text())
    assert metadata["test_time_rotations"] == [-12.0, 0.0, 12.0]
    _, lean = load_model(output / metadata["models"][0]["file"], torch.device("cpu"))
    assert lean["config"]["data"]["test_time_rotations"] == [-12.0, 0.0, 12.0]
    with pytest.raises(ValueError):
        export_bundle([path], tmp_path / "both", test_time_mirror=True, test_time_rotations=(0, 12))
    with pytest.raises(ValueError):
        export_bundle([path], tmp_path / "wide", test_time_rotations=(90,))


def test_notebook_extracts_a_bundle_left_as_one_zip_file(checkpoint, tmp_path, monkeypatch):
    import zipfile

    path, _ = checkpoint
    bundle = export_bundle([path], tmp_path / "bundle")
    archive = tmp_path / "input" / "datasets" / "owner" / "bundle" / "knee-bundle.zip"
    archive.parent.mkdir(parents=True)
    with zipfile.ZipFile(archive, "w") as handle:
        for item in bundle.rglob("*"):
            if item.is_file():
                handle.write(item, Path("knee-bundle") / item.relative_to(bundle))
    competition = tmp_path / "input" / "competitions" / "rsna-knee-abnormality-detection"
    competition.mkdir(parents=True)
    (competition / "test.csv").write_text("StudyInstanceUID\n")
    (competition / "bundle.json").write_text("{}")  # must never be found: competition data is not searched
    monkeypatch.setenv("KNEE_KAGGLE_INPUT", str(tmp_path / "input"))
    monkeypatch.setattr(sys, "path", list(sys.path))
    calls = []
    monkeypatch.setattr(bundle_module, "run_bundle", lambda *args, **kwargs: calls.append(args))
    monkeypatch.setattr(importlib.metadata, "version", lambda name: {"timm": "1.0.26", "opencv-python-headless": "4.12.0.88"}[name])
    exec(compile(notebook_source(bundle), "inference.ipynb", "exec"), {})
    extracted, data_root = calls[0][0], calls[0][1]
    assert (extracted / "bundle.json").read_bytes() == (bundle / "bundle.json").read_bytes()
    assert extracted != bundle and extracted.name == "knee-bundle" and data_root == str(competition)


def test_standard_and_raptor_exports_cannot_share_one_preparation(checkpoint, tmp_path):
    path, _ = checkpoint
    standard = torch.load(path, map_location="cpu", weights_only=True)
    standard["config"]["data"]["recipe"] = "standard"
    standard["image_preparation"]["preprocess_version"] = PREPROCESS_VERSION
    other = tmp_path / "standard.pt"
    save_checkpoint(other, standard)
    with pytest.raises(ValueError, match="same image recipe"):
        export_bundle([path, other], tmp_path / "mixed")
    assert not (tmp_path / "mixed").exists()


@pytest.mark.parametrize("field,value,match", [
    ("recipe", "unknown", "Unknown bundle image recipe"),
    ("cache_size", 320, "requires cache size 336"),
    ("preprocess_version", PREPROCESS_VERSION, "preparation version"),
])
def test_incompatible_raptor_preparation_is_rejected(checkpoint, tmp_path, field, value, match):
    path, _ = checkpoint
    changed = torch.load(path, map_location="cpu", weights_only=True)
    if field == "preprocess_version":
        changed["image_preparation"][field] = value
    else:
        changed["config"]["data"][field] = value
        if field == "cache_size":
            changed["image_preparation"]["image_size"] = value
    save_checkpoint(path, changed)
    with pytest.raises(ValueError, match=match):
        export_bundle([path], tmp_path / "wrong")


def test_raptor_bundle_chunks_real_images_and_matches_whole_prediction(checkpoint, tmp_path, monkeypatch):
    path, _ = checkpoint
    bundle = export_bundle([path], tmp_path / "bundle")
    data, work = tmp_path / "data", tmp_path / "work"
    order = ["study-z", "study-a", "study-q"]
    make_dataset(data, [(study, f"series-{study}") for study in order], split="test")
    prepare = raptor_data.prepare_raptor_data
    seen = []

    def observed_prepare(data_root, output_dir, **kwargs):
        seen.append(kwargs["study_ids"])
        assert len(list(work.glob("knee-images-*"))) == 1
        return prepare(data_root, output_dir, **kwargs)

    def wrong_recipe(*args, **kwargs):
        raise AssertionError("Raptor inference must not use standard image preparation")

    monkeypatch.setattr(raptor_data, "prepare_raptor_data", observed_prepare)
    monkeypatch.setattr(bundle_module, "prepare_data", wrong_recipe)
    output = run_bundle(bundle, data, work, device="cpu", batch_size=2, workers=1, studies_per_chunk=2)
    result = pd.read_csv(output)
    assert result[STUDY_ID].tolist() == order
    assert seen == [order[:2], order[2:]]
    assert not list(work.glob("knee-images-*")) and not list(work.rglob("*.npy"))
    records = [json.loads(path.read_text()) for path in sorted((work / "prediction-parts").glob("qc-*.json"))]
    assert [record["summary"]["studies"] for record in records] == [2, 1]
    assert all(record["summary"]["failed_studies"] == 0 for record in records)
    summary = json.loads((work / "inference-summary.json").read_text())
    assert summary["recipe"] == "raptor" and len(summary["quality"]) == 2

    manifest = prepare(data, tmp_path / "reference-cache", split="test", workers=1)
    reference = pd.read_csv(predict(manifest, data / "test.csv", [bundle / "model-0.pt"],
                                   tmp_path / "reference.csv", device="cpu", batch_size=2, num_workers=0))
    assert reference[STUDY_ID].tolist() == order
    np.testing.assert_allclose(result[TARGETS], reference[TARGETS], rtol=0, atol=1e-7)


def test_failed_raptor_preparation_retains_qc_and_removes_only_owned_cache(checkpoint, tmp_path):
    path, _ = checkpoint
    bundle = export_bundle([path], tmp_path / "bundle")
    data, work = tmp_path / "data", tmp_path / "work"
    make_dataset(data, [("study-a", "series-a")], split="test")
    (data / "test_series/study-a/series-a/0.dcm").write_bytes(b"broken dicom")
    work.mkdir()
    preserved = work / "keep.txt"
    preserved.write_text("unrelated file")
    with pytest.raises(RuntimeError, match="No study could be predicted"):
        run_bundle(bundle, data, work, device="cpu", workers=1)
    assert not list(work.glob("knee-images-*"))
    qc_files = list((work / "prediction-parts").glob("qc-*.json"))
    assert len(qc_files) == 2  # the group, then the study on its own
    assert all(json.loads(path.read_text())["summary"]["failed_studies"] == 1 for path in qc_files)
    assert not (work / "submission.csv").exists()
    assert preserved.read_text() == "unrelated file"


def test_an_unreadable_study_gets_the_average_prediction_and_the_others_are_kept(checkpoint, tmp_path):
    path, _ = checkpoint
    bundle = export_bundle([path], tmp_path / "bundle")
    data, work = tmp_path / "data", tmp_path / "work"
    make_dataset(data, [("study-a", "series-a"), ("study-b", "series-b"), ("study-c", "series-c")], split="test")
    (data / "test_series/study-b/series-b/0.dcm").write_bytes(b"broken dicom")
    output = run_bundle(bundle, data, work, device="cpu", workers=1, studies_per_chunk=3)
    submission = pd.read_csv(output, dtype={STUDY_ID: str}).set_index(STUDY_ID)
    assert submission.index.tolist() == ["study-a", "study-b", "study-c"]
    np.testing.assert_allclose(submission.loc["study-b", TARGETS].to_numpy(dtype=float),
                               submission.loc[["study-a", "study-c"], TARGETS].mean().to_numpy(dtype=float))
    summary = json.loads((work / "inference-summary.json").read_text())
    assert [item[STUDY_ID] for item in summary["failed_studies"]] == ["study-b"] and summary["filled_with_average"] == 1
    assert not list(work.glob("knee-images-*"))


def test_a_malformed_series_row_affects_only_its_own_study(checkpoint, tmp_path):
    path, _ = checkpoint
    bundle = export_bundle([path], tmp_path / "bundle")
    data, work = tmp_path / "data", tmp_path / "work"
    make_dataset(data, [("study-a", "series-a"), ("study-b", "series-b"), ("study-c", "series-c")], split="test")
    series = pd.read_csv(data / "test_series.csv", dtype=str)
    series.loc[series[STUDY_ID].eq("study-b"), "Anatomical_Plane"] = "Oblique"
    series.to_csv(data / "test_series.csv", index=False)
    output = run_bundle(bundle, data, work, device="cpu", workers=1, studies_per_chunk=3)
    summary = json.loads((work / "inference-summary.json").read_text())
    assert [item[STUDY_ID] for item in summary["failed_studies"]] == ["study-b"]
    assert pd.read_csv(output, dtype={STUDY_ID: str})[STUDY_ID].tolist() == ["study-a", "study-b", "study-c"]


def test_a_bad_series_is_dropped_and_its_study_keeps_its_own_prediction(checkpoint, tmp_path):
    # Bad-series insurance: study-b's only Axial series is unreadable, so that slot is left empty and study-b is
    # predicted from its other four series instead of getting the average prediction.
    path, _ = checkpoint
    bundle = export_bundle([path], tmp_path / "bundle")
    data, work = tmp_path / "data", tmp_path / "work"
    make_raw(data, studies=("study-a", "study-b"), split="test")
    clean = pd.read_csv(run_bundle(bundle, data, tmp_path / "clean-work", device="cpu", workers=1))
    # Without a bad series the summary has exactly the keys it had before the insurance existed.
    clean_summary = json.loads((tmp_path / "clean-work/inference-summary.json").read_text())
    assert list(clean_summary) == [
        "studies", "seconds", "studies_per_chunk", "device", "devices_used", "models_per_device", "recipe",
        "preprocess_version", "models", "quality", "failed_studies", "skipped_after_deadline", "filled_with_average",
        "filled_study_ids", "persistent_workers", "prepare_ahead", "accelerate", "worker_restarts"]
    assert [list(group) for group in clean_summary["quality"]] == [
        ["studies", "failed_studies", "prepared_studies", "reused_studies", "studies_with_warnings", "seconds"]]
    (data / "test_series/study-b/series-4/0.dcm").write_bytes(b"broken dicom")
    result = pd.read_csv(run_bundle(bundle, data, work, device="cpu", workers=1))
    summary = json.loads((work / "inference-summary.json").read_text())
    assert not summary["failed_studies"] and summary["filled_with_average"] == 0
    assert [(item[STUDY_ID], item["slot"], item[SERIES_ID], item["replacement"])
            for item in summary["dropped_series"]] == [("study-b", 4, "series-4", None)]
    # study-a is unchanged; study-b equals a local prediction on the same insured cache.
    np.testing.assert_allclose(result[TARGETS].iloc[0], clean[TARGETS].iloc[0], rtol=0, atol=0)
    manifest = raptor_data.prepare_raptor_data(data, tmp_path / "reference-cache", split="test", workers=1,
                                               drop_bad_series=True)
    reference = pd.read_csv(predict(manifest, data / "test.csv", [bundle / "model-0.pt"],
                                   tmp_path / "reference.csv", device="cpu", batch_size=2, num_workers=0))
    np.testing.assert_allclose(result[TARGETS], reference[TARGETS], rtol=0, atol=1e-7)
    assert np.abs(result[TARGETS].iloc[1].to_numpy() - clean[TARGETS].iloc[1].to_numpy()).max() > 0


def test_studies_not_reached_before_the_deadline_get_the_average_prediction(checkpoint, tmp_path, monkeypatch):
    path, _ = checkpoint
    bundle = export_bundle([path], tmp_path / "bundle")
    data, work = tmp_path / "data", tmp_path / "work"
    make_dataset(data, [("study-a", "series-a"), ("study-b", "series-b")], split="test")
    now = {"seconds": 0.0}
    monkeypatch.setattr(bundle_module.time, "perf_counter", lambda: now["seconds"])
    real_group = bundle_module._predict_group

    def slow_group(*args, **kwargs):  # every group takes 150 s on the fake clock
        result = real_group(*args, **kwargs)
        now["seconds"] += 150.0
        return result

    monkeypatch.setattr(bundle_module, "_predict_group", slow_group)
    output = run_bundle(bundle, data, work, device="cpu", workers=1, studies_per_chunk=1, deadline_seconds=100)
    submission = pd.read_csv(output, dtype={STUDY_ID: str}).set_index(STUDY_ID)
    np.testing.assert_allclose(submission.loc["study-b", TARGETS].to_numpy(dtype=float),
                               submission.loc["study-a", TARGETS].to_numpy(dtype=float))
    summary = json.loads((work / "inference-summary.json").read_text())
    assert summary["skipped_after_deadline"] == 1 and summary["filled_with_average"] == 1
    assert summary["filled_study_ids"] == ["study-b"]


def test_raptor_notebook_missing_dependencies_fail_without_network(checkpoint, tmp_path, monkeypatch):
    path, _ = checkpoint
    bundle = export_bundle([path], tmp_path / "bundle")

    def missing(name):
        raise importlib.metadata.PackageNotFoundError(name)

    monkeypatch.setattr(importlib.metadata, "version", missing)
    with pytest.raises(RuntimeError, match="Raptor inference requires timm==1.0.26; found missing"):
        execute_notebook_preflight(bundle, monkeypatch)


def test_raptor_notebook_installs_extra_wheels_only_offline(checkpoint, tmp_path, monkeypatch):
    path, _ = checkpoint
    wheelhouse = tmp_path / "wheels"
    wheelhouse.mkdir()
    (wheelhouse / "fixture-1.0-py3-none-any.whl").write_bytes(b"never installed")
    bundle = export_bundle([path], tmp_path / "bundle", wheelhouse=wheelhouse)
    installed = {"timm": "1.0.26", "opencv-python-headless": "4.12.0.88"}
    monkeypatch.setattr(importlib.metadata, "version", lambda name: installed[name])
    installations = []
    monkeypatch.setattr(subprocess, "run", lambda args, **kwargs: installations.append((args, kwargs)))
    calls = execute_notebook_preflight(bundle, monkeypatch)
    assert len(calls) == 1 and calls[0][0][0] == bundle
    assert len(installations) == 1
    arguments, options = installations[0]
    assert "--no-index" in arguments and "--find-links" in arguments
    assert "--upgrade" not in arguments and "--force-reinstall" not in arguments
    assert options["check"] is True
    assert set(bundle_module.RAPTOR_REQUIREMENTS).issubset(arguments)
    assert not any(argument.startswith(("torch", "nvidia", "cuda")) for argument in arguments)


def test_448_cache_with_center_crop_exports_and_predicts_like_a_local_run(checkpoint, tmp_path):
    path, _ = checkpoint
    changed = torch.load(path, map_location="cpu", weights_only=True)
    changed["config"]["data"].update({"cache_size": 448, "raptor_variant": raptor_data.RAPTOR_DENSE_448_VERSION,
                                      "raptor_view": "center_crop"})
    changed["image_preparation"] = {"preprocess_version": raptor_data.RAPTOR_DENSE_448_VERSION, "image_size": 448}
    save_checkpoint(path, changed)
    bundle = export_bundle([path], tmp_path / "bundle", test_time_mirror=True)
    metadata = json.loads((bundle / "bundle.json").read_text())
    assert metadata["cache_size"] == 448 and metadata["preprocess_version"] == raptor_data.RAPTOR_DENSE_448_VERSION
    _, lean = load_model(bundle / "model-0.pt", torch.device("cpu"))
    assert lean["config"]["data"]["raptor_view"] == "center_crop"
    data, work = tmp_path / "data", tmp_path / "work"
    # The dense layouts need several slices per series, which the single-series fixture lacks.
    make_raw(data, studies=("study-a", "study-b"), split="test")
    result = pd.read_csv(run_bundle(bundle, data, work, device="cpu", workers=1))
    assert not json.loads((work / "inference-summary.json").read_text())["failed_studies"]
    manifest = raptor_data.prepare_raptor_data(data, tmp_path / "reference-cache", split="test", workers=1,
                                               version=raptor_data.RAPTOR_DENSE_448_VERSION)
    reference = pd.read_csv(predict(manifest, data / "test.csv", [bundle / "model-0.pt"],
                                   tmp_path / "reference.csv", device="cpu", batch_size=2, num_workers=0))
    np.testing.assert_allclose(result[TARGETS], reference[TARGETS], rtol=0, atol=1e-7)
    changed["config"]["data"]["cache_size"] = changed["image_preparation"]["image_size"] = 336
    save_checkpoint(path, changed)
    with pytest.raises(ValueError, match="requires cache size 448"):
        export_bundle([path], tmp_path / "wrong")


@pytest.fixture
def three_models(checkpoint, tmp_path):
    """Three checkpoints of the same recipe with different weights, as five folds would be."""
    path, _ = checkpoint
    state = torch.load(path, map_location="cpu", weights_only=True)
    paths = []
    for index in range(3):
        generator = torch.Generator().manual_seed(index)
        model = {name: value + 0.1 * torch.randn(value.shape, generator=generator) if value.is_floating_point() else value
                 for name, value in state["model"].items()}
        paths.append(tmp_path / f"fold-{index}.pt")
        save_checkpoint(paths[-1], {**state, "model": model})
    return paths


def test_models_split_across_two_devices_match_one_device(three_models, tmp_path, monkeypatch):
    bundle = export_bundle(three_models, tmp_path / "bundle", test_time_mirror=True)
    data = tmp_path / "data"
    order = ["study-z", "study-a", "study-q"]
    make_dataset(data, [(study, f"series-{study}") for study in order], split="test")
    prepare = raptor_data.prepare_raptor_data
    prepared = []

    def observed_prepare(*args, **kwargs):
        prepared.append(kwargs["study_ids"])
        return prepare(*args, **kwargs)

    monkeypatch.setattr(raptor_data, "prepare_raptor_data", observed_prepare)
    one = pd.read_csv(run_bundle(bundle, data, tmp_path / "one", device="cpu", workers=1, studies_per_chunk=2))
    two = pd.read_csv(run_bundle(bundle, data, tmp_path / "two", device="cpu", workers=1, studies_per_chunk=2,
                                 devices=["cpu", "cpu"]))
    assert one[STUDY_ID].tolist() == two[STUDY_ID].tolist() == order
    np.testing.assert_allclose(two[TARGETS], one[TARGETS], rtol=0, atol=1e-6)
    # The two shares really differ, so a wrong share weighting would show.
    parts = tmp_path / "two" / "prediction-parts"
    shares = [pd.read_csv(parts / f"predictions-000000-device{index}.csv")[TARGETS] for index in range(2)]
    assert np.abs(shares[0].to_numpy() - shares[1].to_numpy()).max() > 1e-3
    assert prepared == [order[:2], order[2:]] * 2  # each chunk is prepared once per run, not once per device
    summary = json.loads((tmp_path / "two" / "inference-summary.json").read_text())
    assert summary["devices_used"] == 2
    assert summary["models_per_device"] == [{"device": "cpu", "models": ["model-0.pt", "model-2.pt"]},
                                            {"device": "cpu", "models": ["model-1.pt"]}]
    assert json.loads((tmp_path / "one" / "inference-summary.json").read_text())["devices_used"] == 1


def test_one_model_or_one_device_keeps_a_single_prediction_pass(checkpoint, tmp_path, monkeypatch):
    path, _ = checkpoint
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    assert bundle_module.prediction_devices("cuda", 3) == ["cuda:0", "cuda:1"]
    assert bundle_module.prediction_devices("cuda", 1) == ["cuda"]
    assert bundle_module.prediction_devices("cuda:0", 3) == ["cuda:0"]
    assert bundle_module.prediction_devices("cpu", 3) == ["cpu"]
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    assert bundle_module.prediction_devices("cuda", 3) == ["cuda"]

    bundle = export_bundle([path], tmp_path / "bundle")
    data, work = tmp_path / "data", tmp_path / "work"
    make_dataset(data, [("study-a", "series-a"), ("study-b", "series-b")], split="test")
    calls = []
    real_predict = bundle_module.predict

    def observed_predict(manifest, studies, checkpoints, output, **kwargs):
        calls.append(([Path(item).name for item in checkpoints], kwargs["device"], kwargs["num_workers"]))
        return real_predict(manifest, studies, checkpoints, output, **kwargs)

    monkeypatch.setattr(bundle_module, "predict", observed_predict)
    run_bundle(bundle, data, work, device="cpu", workers=2, devices=["cpu", "cpu"])
    assert calls == [(["model-0.pt"], "cpu", 2)]
    assert json.loads((work / "inference-summary.json").read_text())["devices_used"] == 1


def test_a_failure_on_one_device_reaches_the_per_study_fallback(three_models, tmp_path, monkeypatch):
    bundle = export_bundle(three_models, tmp_path / "bundle")
    data, work = tmp_path / "data", tmp_path / "work"
    make_dataset(data, [("study-a", "series-a"), ("study-b", "series-b"), ("study-c", "series-c")], split="test")
    real_predict = bundle_module.predict

    def second_device_fails_on_b(manifest, studies, checkpoints, output, **kwargs):
        if [Path(item).name for item in checkpoints] == ["model-1.pt"] and \
                "study-b" in pd.read_csv(studies, dtype=str)[STUDY_ID].tolist():
            raise RuntimeError("simulated GPU failure")
        return real_predict(manifest, studies, checkpoints, output, **kwargs)

    monkeypatch.setattr(bundle_module, "predict", second_device_fails_on_b)
    output = run_bundle(bundle, data, work, device="cpu", workers=1, studies_per_chunk=3, devices=["cpu", "cpu"])
    submission = pd.read_csv(output, dtype={STUDY_ID: str}).set_index(STUDY_ID)
    assert submission.index.tolist() == ["study-a", "study-b", "study-c"]
    np.testing.assert_allclose(submission.loc["study-b", TARGETS].to_numpy(dtype=float),
                               submission.loc[["study-a", "study-c"], TARGETS].mean().to_numpy(dtype=float))
    summary = json.loads((work / "inference-summary.json").read_text())
    assert [item[STUDY_ID] for item in summary["failed_studies"]] == ["study-b"]
    assert "simulated GPU failure" in summary["failed_studies"][0]["error"]
    assert summary["filled_study_ids"] == ["study-b"]
    assert not list(work.glob("knee-images-*"))


def resnet_bundle(tmp_path, by_finding=False, models=2):
    """A bundle of real (torchvision) models that new processes can load, as the Kaggle notebook's processes do (the
    `checkpoint` fixture's stand-in encoder exists only inside the test process)."""
    settings = deepcopy(DEFAULTS)
    settings["data"].update({"recipe": "raptor", "cache_size": raptor_data.RAPTOR_SIZE,
                             "image_size": 32, "raptor_eval_windows": 2})
    settings["model"].update({"backbone": "resnet18", "pooling": "class_attention", "pretrained": False,
                              "encoder_chunk_size": 4, "use_metadata": False, "use_positions": False,
                              "grad_checkpointing": False})
    settings["training"]["precision"] = "fp32"
    paths = []
    for index in range(models):
        torch.manual_seed(index)
        model = StudyModel(settings["model"], pretrained=False)
        paths.append(tmp_path / f"fold-{index}.pt")
        save_checkpoint(paths[-1], {
            "format_version": 1, "targets": TARGETS, "config": settings, "model": model.state_dict(), "epoch": 1,
            "image_preparation": {"preprocess_version": raptor_data.RAPTOR_PREPROCESS_VERSION,
                                  "image_size": raptor_data.RAPTOR_SIZE},
        })
    weights = ([{name: 1.0 for name in TARGETS} | {TARGETS[0]: 0.2, TARGETS[1]: 1.0},
                {name: 1.0 for name in TARGETS} | {TARGETS[0]: 0.8, TARGETS[1]: 0.0}] if by_finding else None)
    return export_bundle(paths, tmp_path / "bundle", weights=weights)


@pytest.mark.parametrize("by_finding", [False, True])
def test_fresh_processes_per_group_give_the_same_predictions(tmp_path, by_finding):
    """The Kaggle notebook predicts every group in new processes (so memory cannot build up over a long run);
    the predictions must match the in-process ones, on one device and with the models split over two, with equal or
    per-finding weights."""
    bundle = resnet_bundle(tmp_path, by_finding)
    data = tmp_path / "data"
    order = ["study-b", "study-a", "study-c"]
    make_dataset(data, [(study, f"series-{study}") for study in order], split="test")
    results = {}
    for devices in (["cpu"], ["cpu", "cpu"]):
        for separate in (False, True):
            work = tmp_path / f"work-{len(devices)}-{separate}"
            results[len(devices), separate] = pd.read_csv(run_bundle(
                bundle, data, work, device="cpu", workers=1, studies_per_chunk=2, devices=devices,
                separate_processes=separate))
            assert not json.loads((work / "inference-summary.json").read_text())["failed_studies"]
    reference = results[1, False]
    assert reference[STUDY_ID].tolist() == order
    for key, result in results.items():
        assert result[STUDY_ID].tolist() == order, key
        np.testing.assert_allclose(result[TARGETS], reference[TARGETS], rtol=0, atol=1e-6, err_msg=str(key))


@pytest.mark.parametrize("by_finding", [False, True])
def test_persistent_workers_and_preparing_ahead_give_exactly_the_fresh_process_predictions(tmp_path, by_finding):
    """Oct 16, 2026 speed-up: models kept loaded in one process per device, and the next group prepared while the
    current one is predicted, must not change a single number."""
    bundle = resnet_bundle(tmp_path, by_finding)
    data = tmp_path / "data"
    order = ["study-b", "study-a", "study-c"]
    make_dataset(data, [(study, f"series-{study}") for study in order], split="test")
    for devices in (["cpu"], ["cpu", "cpu"]):
        reference = pd.read_csv(run_bundle(bundle, data, tmp_path / f"ref-{len(devices)}", device="cpu", workers=1,
                                           studies_per_chunk=2, devices=devices, separate_processes=True))
        for persistent, ahead in ((True, False), (False, True), (True, True)):
            work = tmp_path / f"work-{len(devices)}-{persistent}-{ahead}"
            result = pd.read_csv(run_bundle(bundle, data, work, device="cpu", workers=1, studies_per_chunk=2,
                                            devices=devices, separate_processes=True, persistent=persistent,
                                            prepare_ahead=ahead))
            summary = json.loads((work / "inference-summary.json").read_text())
            key = (len(devices), persistent, ahead)
            assert result[STUDY_ID].tolist() == order, key
            np.testing.assert_array_equal(result[TARGETS].to_numpy(), reference[TARGETS].to_numpy(), err_msg=str(key))
            assert not summary["failed_studies"] and summary["worker_restarts"] == 0, key
            assert summary["persistent_workers"] is persistent and summary["prepare_ahead"] is ahead, key
            assert not list(work.glob("knee-images-*")), key


def test_a_dead_worker_is_replaced_and_its_group_retried_study_by_study(tmp_path, monkeypatch):
    import os
    import signal

    bundle = resnet_bundle(tmp_path)
    data = tmp_path / "data"
    make_dataset(data, [("study-a", "series-a"), ("study-b", "series-b"), ("study-c", "series-c")], split="test")
    reference = pd.read_csv(run_bundle(bundle, data, tmp_path / "ref", device="cpu", workers=1, studies_per_chunk=3,
                                       devices=["cpu", "cpu"], separate_processes=True))
    killed = []

    def kill_second_worker_once(workers, index):
        if index == 1 and not killed:
            process = workers.processes[1]
            os.kill(process.pid, signal.SIGKILL)
            process.join()
            killed.append(process.pid)

    monkeypatch.setattr(bundle_module._DeviceWorkers, "before_job", kill_second_worker_once)
    work = tmp_path / "work"
    result = pd.read_csv(run_bundle(bundle, data, work, device="cpu", workers=1, studies_per_chunk=3,
                                    devices=["cpu", "cpu"], persistent=True))
    summary = json.loads((work / "inference-summary.json").read_text())
    assert len(killed) == 1 and summary["worker_restarts"] == 1 and not summary["failed_studies"]
    assert result[STUDY_ID].tolist() == reference[STUDY_ID].tolist()
    np.testing.assert_allclose(result[TARGETS], reference[TARGETS], rtol=0, atol=1e-6)


def test_workers_past_the_memory_limit_are_replaced_between_groups(tmp_path, monkeypatch):
    bundle = resnet_bundle(tmp_path)
    data = tmp_path / "data"
    make_dataset(data, [("study-a", "series-a"), ("study-b", "series-b"), ("study-c", "series-c")], split="test")
    reference = pd.read_csv(run_bundle(bundle, data, tmp_path / "ref", device="cpu", workers=1, studies_per_chunk=2,
                                       devices=["cpu"], separate_processes=True))
    monkeypatch.setattr(bundle_module, "_process_rss", lambda pid: 2**60)
    work = tmp_path / "work"
    result = pd.read_csv(run_bundle(bundle, data, work, device="cpu", workers=1, studies_per_chunk=2, devices=["cpu"],
                                    persistent=True))
    assert json.loads((work / "inference-summary.json").read_text())["worker_restarts"] == 2  # after each of 2 groups
    np.testing.assert_array_equal(result[TARGETS].to_numpy(), reference[TARGETS].to_numpy())


def test_preparing_ahead_stops_at_the_deadline_and_removes_unused_images(tmp_path, monkeypatch):
    bundle = resnet_bundle(tmp_path, models=1)
    data, work = tmp_path / "data", tmp_path / "work"
    make_dataset(data, [("study-a", "series-a"), ("study-b", "series-b")], split="test")
    now = {"seconds": 0.0}
    monkeypatch.setattr(bundle_module.time, "perf_counter", lambda: now["seconds"])
    real = bundle_module._predict_on_devices

    def slow(*args, **kwargs):  # every group's prediction takes 150 s on the fake clock
        result = real(*args, **kwargs)
        now["seconds"] += 150.0
        return result

    monkeypatch.setattr(bundle_module, "_predict_on_devices", slow)
    output = run_bundle(bundle, data, work, device="cpu", workers=1, studies_per_chunk=1, deadline_seconds=100,
                        devices=["cpu"], prepare_ahead=True)
    submission = pd.read_csv(output, dtype={STUDY_ID: str}).set_index(STUDY_ID)
    np.testing.assert_allclose(submission.loc["study-b", TARGETS].to_numpy(dtype=float),
                               submission.loc["study-a", TARGETS].to_numpy(dtype=float))
    summary = json.loads((work / "inference-summary.json").read_text())
    assert summary["skipped_after_deadline"] == 1 and summary["filled_study_ids"] == ["study-b"]
    assert not list(work.glob("knee-images-*"))  # study-b's images, prepared ahead, were removed unused


def weighted_reference(bundle, data, tmp_path, weights):
    """Each bundled model predicted alone on a local cache, then blended by hand with per-finding weights."""
    manifest = raptor_data.prepare_raptor_data(data, tmp_path / "reference-cache", split="test", workers=1)
    alone = [pd.read_csv(predict(manifest, data / "test.csv", [bundle / f"model-{index}.pt"],
                                 tmp_path / f"alone-{index}.csv", device="cpu", batch_size=2, num_workers=0))
             for index in range(len(weights))]
    matrix = np.array([[entry[name] if isinstance(entry, dict) else entry for name in TARGETS] for entry in weights],
                      dtype=float)
    matrix /= matrix.sum(axis=0)
    return alone[0][[STUDY_ID]].assign(**{name: sum(matrix[i, j] * frame[name] for i, frame in enumerate(alone))
                                          for j, name in enumerate(TARGETS)})


def test_per_finding_weights_give_the_hand_made_blend_on_one_or_two_devices(three_models, tmp_path):
    # Model 1 alone holds the second device's share, and it has no weight for the first finding.
    only_two = {name: 1.0 for name in TARGETS} | {TARGETS[0]: 0.0}
    weights = [{name: 1.0 for name in TARGETS} | {TARGETS[0]: 3.0, TARGETS[1]: 0.0}, only_two, 2.0]
    bundle = export_bundle(three_models, tmp_path / "bundle", weights=weights)
    stored = json.loads((bundle / "bundle.json").read_text())["models"]
    assert stored[1]["weights"][TARGETS[0]] == 0.0 and stored[2]["weights"] == {name: 2.0 for name in TARGETS}
    data = tmp_path / "data"
    order = ["study-z", "study-a", "study-q"]
    make_dataset(data, [(study, f"series-{study}") for study in order], split="test")
    reference = weighted_reference(bundle, data, tmp_path, weights)
    for devices in (["cpu"], ["cpu", "cpu"]):
        result = pd.read_csv(run_bundle(bundle, data, tmp_path / f"work-{len(devices)}", device="cpu", workers=1,
                                        studies_per_chunk=2, devices=devices))
        assert result[STUDY_ID].tolist() == order
        np.testing.assert_allclose(result[TARGETS], reference[TARGETS], rtol=0, atol=1e-6, err_msg=str(devices))
    # The blend differs from equal weights, so ignoring the stored weights would show.
    equal = pd.read_csv(run_bundle(export_bundle(three_models, tmp_path / "equal"), data, tmp_path / "work-equal",
                                   device="cpu", workers=1))
    assert np.abs(equal[TARGETS].to_numpy() - reference[TARGETS].to_numpy()).max() > 1e-4


def test_bad_bundle_weights_are_rejected(three_models, tmp_path):
    partial = {name: 1.0 for name in TARGETS[:-1]}
    zero_for_a_finding = [{name: 1.0 for name in TARGETS} | {TARGETS[4]: 0.0}] * 3
    for index, bad in enumerate(([1.0, 1.0], [1.0, partial, 1.0], [1.0, -1.0, 1.0], zero_for_a_finding)):
        with pytest.raises(ValueError):
            export_bundle(three_models, tmp_path / f"bad-{index}", weights=bad)
        assert not (tmp_path / f"bad-{index}").exists()


def test_export_can_set_the_prediction_window_count(checkpoint, tmp_path):
    path, _ = checkpoint
    original = torch.load(path, map_location="cpu", weights_only=True)["config"]["data"]["raptor_eval_windows"]
    output = export_bundle([path], tmp_path / "bundle", eval_windows=56)
    metadata = json.loads((output / "bundle.json").read_text())
    _, lean = load_model(output / metadata["models"][0]["file"], torch.device("cpu"))
    assert lean["config"]["data"]["raptor_eval_windows"] == 56
    assert torch.load(path, map_location="cpu", weights_only=True)["config"]["data"]["raptor_eval_windows"] == original
    for bad in (0, -3, True, 2.5):
        with pytest.raises(ValueError):
            export_bundle([path], tmp_path / f"bad-{bad}", eval_windows=bad)


def test_accelerate_needs_persistent_workers_and_a_cuda_device(tmp_path):
    from rsna_knee_abnormality_detection.predict import Predictor

    bundle = resnet_bundle(tmp_path, models=1)
    data = tmp_path / "data"
    make_dataset(data, [("study-a", "series-a")], split="test")
    with pytest.raises(ValueError, match="persistent"):
        run_bundle(bundle, data, tmp_path / "work", device="cpu", workers=1, devices=["cpu"], accelerate=True)
    with pytest.raises(ValueError, match="CUDA"):
        Predictor([bundle / "model-0.pt"], device="cpu", accelerate=True)
