"""Fine-tuning on all windows (Oct 17, 2026): training continues from a checkpoint and sees every window of each kept
series, instead of a small random draw."""

import json
from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch
from test_raptor_data import make_raw

from rsna_knee_abnormality_detection.config import DEFAULTS, validate_config
from rsna_knee_abnormality_detection.constants import STUDY_ID, TARGETS
from rsna_knee_abnormality_detection.raptor_data import RaptorDataset, prepare_raptor_data
from rsna_knee_abnormality_detection.train import train


def record():
    # Five series: sagittal 0/1, coronal 2/3, axial 4, with 16/12/10/6/10 windows.
    return {"slots": np.repeat([0, 1, 2, 3, 4], [16, 12, 10, 6, 10]),
            "planes": np.repeat([0, 0, 1, 1, 2], [16, 12, 10, 6, 10])}


def test_all_window_draw_returns_every_window_of_the_kept_series():
    data = record()
    every = SimpleNamespace(windows=12, sampling="stratified", series_dropout=0.0, all_windows=True)
    assert RaptorDataset._sample_windows(every, data, np.random.default_rng(0)).tolist() == list(range(54))
    hiding = SimpleNamespace(windows=12, sampling="stratified", series_dropout=0.5, all_windows=True)
    for seed in range(30):
        choices = RaptorDataset._sample_windows(hiding, data, np.random.default_rng(seed))
        kept = np.unique(data["slots"][choices])
        assert choices.tolist() == np.flatnonzero(np.isin(data["slots"], kept)).tolist()
        assert set(data["planes"][choices]) == {0, 1, 2}


def raptor_defaults():
    config = deepcopy(DEFAULTS)
    config["data"].update(recipe="raptor", raptor_variant="raptor-dense-distinct-v1", cache_size=336)
    config["model"].update(pooling="class_attention", use_metadata=False, use_positions=False)
    return config


def test_config_checks_all_window_settings():
    config = raptor_defaults()
    config["data"]["raptor_train_all_windows"] = True
    with pytest.raises(ValueError, match="batch_size 1"):
        validate_config(config)
    config["training"]["batch_size"] = 1
    validate_config(config)
    config["training"]["init_checkpoint"] = 5
    with pytest.raises(ValueError, match="init_checkpoint"):
        validate_config(config)


def test_fine_tune_starts_from_the_checkpoint_and_trains_on_all_windows(tmp_path):
    studies = [f"s{index}" for index in range(8)]
    make_raw(tmp_path / "raw", studies=studies)
    manifest = prepare_raptor_data(tmp_path / "raw", tmp_path / "cache", workers=1)
    rng = np.random.default_rng(5)
    pd.DataFrame({STUDY_ID: studies, **{name: rng.random(len(studies)).round() for name in TARGETS},
                  "label_source": "report"}).to_csv(tmp_path / "labels.csv", index=False)
    pd.DataFrame({STUDY_ID: studies, "fold": [0, 0, 0, 1, 1, 1, 1, 1], "split": "train",
                  "group_id": studies}).to_csv(tmp_path / "splits.csv", index=False)
    config = deepcopy(DEFAULTS)
    config["seed"] = 11
    config["data"].update({
        "recipe": "raptor", "manifest": str(manifest), "labels": str(tmp_path / "labels.csv"),
        "splits": str(tmp_path / "splits.csv"), "fold": 0, "cache_size": 336, "image_size": 32,
        "raptor_train_windows": 5, "raptor_eval_windows": 6, "raptor_window_sampling": "stratified",
        "raptor_series_dropout": 0.3})
    config["model"].update({"backbone": "resnet18", "pretrained": False, "pooling": "class_attention",
                            "use_metadata": False, "use_positions": False, "encoder_chunk_size": 8})
    config["training"].update({"output_dir": str(tmp_path / "base"), "epochs": 1, "batch_size": 2,
                               "accumulation_steps": 1, "num_workers": 0, "device": "cpu", "precision": "fp32",
                               "selection_metric": "loss", "deterministic": True})
    train(config)
    start = next(p for p in (tmp_path / "base" / "averaged-model.pt", tmp_path / "base" / "best-model.pt") if p.exists())
    tuned = deepcopy(config)
    tuned["data"]["raptor_train_all_windows"] = True
    tuned["training"].update({"output_dir": str(tmp_path / "tuned"), "batch_size": 1, "accumulation_steps": 2,
                              "init_checkpoint": str(start), "encoder_lr": 1e-12, "head_lr": 1e-12,
                              "freeze_batchnorm": True})
    train(tuned)
    before = torch.load(start, weights_only=True)["model"]
    after = torch.load(tmp_path / "tuned" / "last.pt", weights_only=True)["model"]
    names = [name for name, value in before.items() if value.is_floating_point() and "running" not in name]
    assert names and all(torch.allclose(before[name], after[name], atol=1e-6) for name in names)
    # Frozen BatchNorm: the start checkpoint's statistics are untouched.
    stats = [name for name in before if "running" in name or "num_batches_tracked" in name]
    assert stats and all(torch.equal(before[name], after[name]) for name in stats)
    provenance = json.loads((tmp_path / "tuned" / "provenance.json").read_text())
    assert "init_checkpoint" in provenance["input_sha256"]
    # The training items had every window of the kept series (more than the 5 of a draw).
    data = RaptorDataset(manifest, pd.DataFrame({STUDY_ID: studies}), tuned["data"], training=True, seed=11)
    assert all(data[i]["images"].shape[1] > 5 for i in range(len(studies)))
