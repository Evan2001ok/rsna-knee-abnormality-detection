import json
import socket
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from test_imaging import write_dicom
from test_prepare import make_dataset

from rsna_knee_abnormality_detection import bundle as bundle_module
from rsna_knee_abnormality_detection.bundle import export_bundle, run_bundle
from rsna_knee_abnormality_detection.config import DEFAULTS
from rsna_knee_abnormality_detection.constants import STUDY_ID, TARGETS
from rsna_knee_abnormality_detection.imaging import PREPROCESS_VERSION
from rsna_knee_abnormality_detection.models import StudyModel
from rsna_knee_abnormality_detection.predict import load_model, predict
from rsna_knee_abnormality_detection.prepare import prepare_data
from rsna_knee_abnormality_detection.utils import file_hash, save_checkpoint


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def fail(*args, **kwargs):
        raise AssertionError("Bundle tests must not use the network")
    connect = socket.socket.connect

    def local_only(sock, address):
        if sock.family in {socket.AF_INET, socket.AF_INET6}:
            fail()
        # PyTorch's worker processes exchange tensors over local Unix sockets.
        return connect(sock, address)

    monkeypatch.setattr(socket.socket, "connect", local_only)
    monkeypatch.setattr(torch.hub, "download_url_to_file", fail)


@pytest.fixture(scope="module")
def checkpoint(tmp_path_factory):
    path = tmp_path_factory.mktemp("bundle-checkpoint") / "training.pt"
    config = deepcopy(DEFAULTS)
    config["model"].update({"backbone": "resnet18", "pretrained": True})
    model = StudyModel(config["model"], pretrained=False)
    save_checkpoint(path, {
        "format_version": 1,
        "targets": TARGETS,
        "config": config,
        "model": model.state_dict(),
        "epoch": 2,
        "image_preparation": {
            "preprocess_version": PREPROCESS_VERSION,
            "image_size": config["data"]["cache_size"],
            "source_paths": ["unneeded-training-data-path"],
        },
        "optimizer": {"training_only": torch.ones(3)},
        "scheduler": {"last_epoch": 7},
        "rng": {"torch": torch.get_rng_state()},
        "provenance": {"training_ids": ["should-not-be-exported"]},
        "history": [{"train_loss": 0.5}],
    })
    return path


def test_export_keeps_prediction_state_and_loads_without_pretrained_download(checkpoint, tmp_path):
    output = export_bundle([checkpoint], tmp_path / "bundle")
    metadata = json.loads((output / "bundle.json").read_text())
    assert metadata["targets"] == TARGETS
    assert metadata["preprocess_version"] == PREPROCESS_VERSION
    assert metadata["cache_size"] == DEFAULTS["data"]["cache_size"]
    exported = output / metadata["models"][0]["file"]
    assert file_hash(exported) == metadata["models"][0]["sha256"]
    model, lean = load_model(exported, torch.device("cpu"))
    assert not model.training
    assert lean["epoch"] == 2
    assert set(lean) == {"format_version", "targets", "config", "model", "epoch", "image_preparation"}
    assert lean["image_preparation"] == {
        "preprocess_version": PREPROCESS_VERSION,
        "image_size": DEFAULTS["data"]["cache_size"],
    }
    original = torch.load(checkpoint, map_location="cpu", weights_only=True)
    for name, value in original["model"].items():
        assert torch.equal(value, lean["model"][name]), name
    assert (output / "src" / "rsna_knee_abnormality_detection" / "predict.py").is_file()
    assert (output / "run_inference.py").is_file()
    compile((output / "run_inference.py").read_text(), "run_inference.py", "exec")


def test_exported_notebook_is_python_and_installs_only_offline(checkpoint, tmp_path):
    output = export_bundle([checkpoint], tmp_path / "bundle")
    notebook = json.loads((output / "inference.ipynb").read_text())
    assert notebook["nbformat"] == 4
    source = "\n".join("".join(cell["source"]) for cell in notebook["cells"] if cell["cell_type"] == "code")
    compile(source, "inference.ipynb", "exec")
    assert "--no-index" in source
    assert "--find-links" in source
    assert "--upgrade" not in source and "--force-reinstall" not in source
    assert "/kaggle/input" in source and "/kaggle/working" in source


def test_modified_model_is_rejected_before_reading_images(checkpoint, tmp_path, monkeypatch):
    output = export_bundle([checkpoint], tmp_path / "bundle")
    metadata = json.loads((output / "bundle.json").read_text())
    path = output / metadata["models"][0]["file"]
    with path.open("ab") as stream:
        stream.write(b"unexpected change")
    def fail(*args, **kwargs):
        raise AssertionError("Image preparation must not begin for a damaged model")
    monkeypatch.setattr(bundle_module, "prepare_data", fail)
    with pytest.raises(ValueError, match="checksum"):
        run_bundle(output, tmp_path / "data", tmp_path / "work", device="cpu")


@pytest.mark.parametrize("field,value", [("targets", list(reversed(TARGETS))), ("format_version", 999)])
def test_incompatible_checkpoints_are_rejected(checkpoint, tmp_path, field, value):
    original = torch.load(checkpoint, map_location="cpu", weights_only=True)
    original[field] = value
    invalid = tmp_path / "invalid.pt"
    save_checkpoint(invalid, original)
    with pytest.raises(ValueError, match="Incompatible"):
        export_bundle([invalid], tmp_path / "bundle")


def test_different_preparation_sizes_cannot_share_bundle(checkpoint, tmp_path):
    changed = torch.load(checkpoint, map_location="cpu", weights_only=True)
    changed["config"]["data"]["cache_size"] += 16
    changed["image_preparation"]["image_size"] = changed["config"]["data"]["cache_size"]
    other = tmp_path / "other.pt"
    save_checkpoint(other, changed)
    with pytest.raises(ValueError, match="same prepared image size"):
        export_bundle([checkpoint, other], tmp_path / "bundle")


def test_export_requires_models_and_preserves_existing_output(checkpoint, tmp_path):
    output = tmp_path / "bundle"
    with pytest.raises(ValueError, match="At least one"):
        export_bundle([], output)
    output.mkdir()
    existing = output / "keep.txt"
    existing.write_text("existing result")
    with pytest.raises(FileExistsError, match="not empty"):
        export_bundle([checkpoint], output)
    assert existing.read_text() == "existing result"


def test_wheelhouse_files_are_included_and_empty_wheelhouse_is_rejected(checkpoint, tmp_path):
    wheelhouse = tmp_path / "wheelhouse"
    wheelhouse.mkdir()
    with pytest.raises(ValueError, match="no .whl"):
        export_bundle([checkpoint], tmp_path / "empty-wheel-bundle", wheelhouse)
    wheel = wheelhouse / "example-1.0-py3-none-any.whl"
    wheel.write_bytes(b"a wheel copying fixture; never installed")
    wheel_metadata = {
        "python_version": "3.12", "implementation": "cp", "platform": "manylinux2014_x86_64",
        "architecture": "x86_64", "requirements": ["example==1.0"],
        "wheels": [{"file": wheel.name, "sha256": file_hash(wheel), "bytes": wheel.stat().st_size}],
    }
    (wheelhouse / "wheelhouse.json").write_text(json.dumps(wheel_metadata))
    output = export_bundle([checkpoint], tmp_path / "bundle", wheelhouse)
    assert (output / "wheelhouse" / wheel.name).read_bytes() == wheel.read_bytes()
    assert json.loads((output / "wheelhouse" / "wheelhouse.json").read_text()) == wheel_metadata


@pytest.mark.parametrize("field,value", [("preprocess_version", "an-old-image-pipeline"), ("image_size", 999)])
def test_checkpoint_image_preparation_must_match_export(checkpoint, tmp_path, field, value):
    changed = torch.load(checkpoint, map_location="cpu", weights_only=True)
    changed["image_preparation"][field] = value
    path = tmp_path / "wrong-preparation.pt"
    save_checkpoint(path, changed)
    with pytest.raises(ValueError, match="preparation|size"):
        export_bundle([path], tmp_path / "bundle")


def test_multiple_real_image_chunks_match_full_prediction_and_release_cache(checkpoint, tmp_path, monkeypatch):
    small = torch.load(checkpoint, map_location="cpu", weights_only=True)
    small["config"]["data"].update({"cache_size": 32, "image_size": 32, "slices_per_series": 2})
    small["image_preparation"]["image_size"] = 32
    small["config"]["training"]["precision"] = "fp32"
    small_path = tmp_path / "small.pt"
    save_checkpoint(small_path, small)
    output = export_bundle([small_path], tmp_path / "bundle")
    data, work = tmp_path / "data", tmp_path / "work"
    order = ["study-c", "study-a", "study-e", "study-b", "study-d"]
    make_dataset(data, [(study, f"series-{study}") for study in order], split="test")
    rng = np.random.default_rng(104)
    for study in order:
        for index in range(2):
            write_dicom(
                data / "test_series" / study / f"series-{study}" / f"{index}.dcm",
                rng.integers(0, 1000, size=(8, 10)), position=(0, 0, index * 2),
            )
    work.mkdir()
    preserved = work / "keep.txt"
    preserved.write_text("an unrelated existing file")
    cache_paths, prepared_groups = [], []

    def observed_prepare(data_root, output_dir, **kwargs):
        cache_paths.append(Path(output_dir))
        # At most one batch of prepared volumes may exist at a time.
        assert len(list(work.glob("knee-images-*"))) == 1
        assert all(not path.exists() for path in cache_paths[:-1])
        result = prepare_data(data_root, output_dir, **kwargs)
        prepared_groups.append(set(pd.read_csv(result)[STUDY_ID]))
        return result

    monkeypatch.setattr(bundle_module, "prepare_data", observed_prepare)
    submission = run_bundle(output, data, work, device="cpu", batch_size=2, workers=1, studies_per_chunk=2)
    result = pd.read_csv(submission)
    assert result.columns.tolist() == [STUDY_ID, *TARGETS]
    assert result[STUDY_ID].tolist() == order
    assert [len(group) for group in prepared_groups] == [2, 2, 1]
    assert set.union(*prepared_groups) == set(order)
    assert not list(work.glob("knee-images-*"))
    assert not list(work.rglob("*.npy"))
    assert preserved.read_text() == "an unrelated existing file"
    assert len(list((work / "prediction-parts").glob("predictions-*.csv"))) == 3
    assert len(list((work / "prediction-parts").glob("qc-*.json"))) == 3
    summary = json.loads((work / "inference-summary.json").read_text())
    assert summary["studies"] == len(order) and len(summary["quality"]) == 3

    # Compare against a complete-data prediction, so chunk assembly cannot pass
    # merely by producing valid-looking IDs and probabilities of the right shape.
    manifest = prepare_data(data, tmp_path / "reference-cache", split="test", size=32, workers=1)
    reference = pd.read_csv(predict(
        manifest, data / "test.csv", [output / "model-0.pt"], tmp_path / "reference.csv",
        device="cpu", batch_size=2, num_workers=0,
    ))
    assert reference[STUDY_ID].tolist() == order
    np.testing.assert_allclose(result[TARGETS], reference[TARGETS], rtol=0, atol=1e-6)


def test_temporary_image_cache_is_released_if_prediction_fails(checkpoint, tmp_path, monkeypatch):
    output = export_bundle([checkpoint], tmp_path / "bundle")
    data, work = tmp_path / "data", tmp_path / "work"
    make_dataset(data, [("study-a", "series-a")], split="test")

    def fail(*args, **kwargs):
        raise RuntimeError("deliberate prediction failure")

    monkeypatch.setattr(bundle_module, "predict", fail)
    with pytest.raises(RuntimeError, match="deliberate prediction failure"):
        run_bundle(output, data, work, device="cpu", workers=1, studies_per_chunk=1)
    assert not list(work.glob("knee-images-*"))
    assert list((work / "prediction-parts").glob("qc-*.json"))
