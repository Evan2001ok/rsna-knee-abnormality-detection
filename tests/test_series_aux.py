"""CPU checks for the optional per-series training loss (training.series_aux_weight, Oct 15, 2026)."""

import sys
from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch
from test_raptor_data import make_raw
from torch import nn

from rsna_knee_abnormality_detection.config import DEFAULTS, validate_config
from rsna_knee_abnormality_detection.constants import STUDY_ID, TARGETS
from rsna_knee_abnormality_detection.models import RAPTOR_BACKBONE, StudyModel
from rsna_knee_abnormality_detection.raptor_data import RaptorDataset, prepare_raptor_data
from rsna_knee_abnormality_detection.train import train


@pytest.fixture
def tiny_timm(monkeypatch):
    class TinyEncoder(nn.Module):
        num_features = 8

        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(3, self.num_features, 4, stride=4)

        def forward_features(self, images):
            return self.conv(images)

        def forward(self, images):
            return self.forward_features(images).mean(dim=(-2, -1))

    monkeypatch.setitem(sys.modules, "timm", SimpleNamespace(create_model=lambda name, **kwargs: TinyEncoder()))


def model_config(**overrides):
    return {"backbone": RAPTOR_BACKBONE, "pooling": "class_attention", "input_size": 32, "encoder_chunk_size": 4,
            "dropout": 0.0, **overrides}


def raptor_batch():
    """Two studies in the raptor layout: one series axis, 6 windows each, windows from slots 0, 2 and 4."""
    torch.manual_seed(0)
    images = torch.rand(2, 1, 6, 3, 32, 32)
    slots = torch.tensor([[[0, 0, 2, 2, 4, 4]], [[2, 2, 2, 4, 4, 4]]])
    return images, torch.ones(2, 1, dtype=torch.bool), torch.zeros(2, 1, 5), slots


@pytest.mark.parametrize("spatial", [False, True])
def test_series_logits_equal_the_head_on_each_series_alone(tiny_timm, spatial):
    model = StudyModel(model_config(spatial_attention=spatial), pretrained=False).eval()
    images, mask, metadata, slots = raptor_batch()
    with torch.no_grad():
        plain = model(images, mask, metadata)
        logits, series_logits, rows = model(images, mask, metadata, window_slots=slots, return_series=True)
        assert torch.equal(plain, logits)
        expected, expected_rows = [], []
        for slot in (0, 2, 4):
            for study in range(2):
                keep = (slots[study, 0] == slot).nonzero().flatten()
                if len(keep):
                    expected.append(model(images[study:study + 1, :, keep], mask[study:study + 1], metadata[study:study + 1])[0])
                    expected_rows.append(study)
    assert rows.tolist() == expected_rows  # study 1 has no slot-0 windows, so it is skipped for that series
    assert torch.allclose(series_logits, torch.stack(expected), atol=1e-5)


def test_return_series_needs_slots_and_class_attention(tiny_timm):
    model = StudyModel(model_config(), pretrained=False)
    images, mask, metadata, _ = raptor_batch()
    with pytest.raises(ValueError, match="window_slots"):
        model(images, mask, metadata, return_series=True)


def test_config_weight_validation():
    assert DEFAULTS["training"]["series_aux_weight"] == 0.0
    for bad in (-0.1, True, "0.2", 10):
        config = deepcopy(DEFAULTS)
        config["training"]["series_aux_weight"] = bad
        with pytest.raises(ValueError, match="series_aux_weight"):
            validate_config(config)
    config = deepcopy(DEFAULTS)
    config["training"]["series_aux_weight"] = 0.2  # the default recipe is not raptor
    with pytest.raises(ValueError, match="raptor recipe"):
        validate_config(config)


def raptor_training_config(tmp_path, weight):
    studies = [f"s{index}" for index in range(6)]
    if not (tmp_path / "cache").exists():
        make_raw(tmp_path / "raw", studies=studies)
        prepare_raptor_data(tmp_path / "raw", tmp_path / "cache", workers=1)
        rng = np.random.default_rng(9)
        pd.DataFrame({STUDY_ID: studies, **{name: rng.random(len(studies)).round() for name in TARGETS},
                      "label_source": "report"}).to_csv(tmp_path / "labels.csv", index=False)
        pd.DataFrame({STUDY_ID: studies, "fold": [0, 0, 1, 1, 1, 1], "split": "train",
                      "group_id": studies}).to_csv(tmp_path / "splits.csv", index=False)
    config = deepcopy(DEFAULTS)
    config["data"].update({"recipe": "raptor", "manifest": str(tmp_path / "cache" / "manifest.csv"),
                           "labels": str(tmp_path / "labels.csv"), "splits": str(tmp_path / "splits.csv"),
                           "cache_size": 336, "image_size": 32, "raptor_train_windows": 4, "raptor_eval_windows": 5})
    config["model"].update({"backbone": "resnet18", "pretrained": False, "pooling": "class_attention",
                            "use_metadata": False, "use_positions": False, "encoder_chunk_size": 8})
    config["training"].update({"epochs": 2, "batch_size": 2, "accumulation_steps": 1, "num_workers": 0,
                               "device": "cpu", "precision": "fp32", "selection_metric": "loss", "deterministic": True,
                               "series_aux_weight": weight})
    return config


def test_dataset_returns_each_windows_series(tmp_path):
    config = raptor_training_config(tmp_path, 0.0)
    data = RaptorDataset(config["data"]["manifest"], pd.read_csv(config["data"]["labels"]).head(2), config["data"],
                         training=True)
    item = data[0]
    assert item["window_slots"].shape == item["window_planes"].shape == (1, 4)
    assert item["window_slots"].dtype == torch.int64


def test_training_with_series_loss_runs_resumes_and_differs(tmp_path):
    config = raptor_training_config(tmp_path, 0.2)
    config["training"]["output_dir"] = str(tmp_path / "full")
    train(config)
    resumed = deepcopy(config)
    resumed["training"]["output_dir"] = str(tmp_path / "resumed")
    train(resumed, stop_after_epoch=1)
    train(resumed, resume=tmp_path / "resumed" / "last.pt")
    off = raptor_training_config(tmp_path, 0.0)
    off["training"]["output_dir"] = str(tmp_path / "off")
    train(off)
    full = torch.load(tmp_path / "full" / "last.pt", weights_only=True)["model"]
    restarted = torch.load(tmp_path / "resumed" / "last.pt", weights_only=True)["model"]
    without = torch.load(tmp_path / "off" / "last.pt", weights_only=True)["model"]
    for name, value in full.items():
        assert torch.equal(value, restarted[name]), name
    assert set(full) == set(without)  # no new weights
    assert any(not torch.equal(full[name], without[name]) for name in full)
