"""Per-window descriptors: plane, sequence, knee side, and anatomical slice offset."""

import json
from copy import deepcopy

import numpy as np
import pandas as pd
import pytest
import torch
from test_augmentation import DEFAULT_GPU_AUGMENTATION
from test_raptor_data import make_raw

from rsna_knee_abnormality_detection.augment import augment_batch, mirror_window_features, validate_gpu_augmentation
from rsna_knee_abnormality_detection.config import DEFAULTS, validate_config
from rsna_knee_abnormality_detection.constants import SIDE_FEATURE, STUDY_ID, TARGETS, WINDOW_FEATURES
from rsna_knee_abnormality_detection.models import StudyModel
from rsna_knee_abnormality_detection.raptor_data import RaptorDataset, knee_side, prepare_raptor_data
from rsna_knee_abnormality_detection.train import train


@pytest.fixture
def cached(tmp_path):
    make_raw(tmp_path / "raw")
    manifest = prepare_raptor_data(tmp_path / "raw", tmp_path / "cache", workers=1)
    return manifest, json.loads((tmp_path / "cache/studies/study.json").read_text())


def test_descriptors_match_plane_sequence_side_and_anatomical_order(cached):
    manifest, metadata = cached
    dataset = RaptorDataset(manifest, pd.DataFrame({STUDY_ID: ["study"]}), {"image_size": 32}, training=False)
    record = dataset._records["study"]
    features = record["window_features"]
    assert features.shape == (len(record["centers"]), len(WINDOW_FEATURES))
    np.testing.assert_array_equal(features[:, :3].argmax(1), record["planes"])
    assert (features[:, :3].sum(1) == 1).all()
    # make_raw marks the fluid-sensitive series as fat suppressed, too.
    slot_flags = {slot["slot"]: slot["Fluid_Sensitive"] for slot in metadata["slots"]}
    np.testing.assert_array_equal(features[:, 3], [slot_flags[slot] for slot in record["slots"]])
    np.testing.assert_array_equal(features[:, 3], features[:, 4])
    # The synthetic front view lies at positive patient x: a left knee.
    assert knee_side(metadata["slots"]) == 1.0 and (features[:, SIDE_FEATURE] == 1).all()
    sagittal = features[record["slots"] == 0, -1]
    # Synthetic sagittal slices move toward the patient's right (medial for a left knee)
    # as the cached depth increases, so the lateral offset must fall.
    assert (np.diff(sagittal) <= 0).all() and sagittal[0] > 0 > sagittal[-1]
    sample = dataset[0]
    assert sample["window_features"].shape == (1, 42, len(WINDOW_FEATURES))


def test_mirroring_flips_only_the_knee_side():
    features = torch.randn(2, 1, 4, len(WINDOW_FEATURES))
    mirrored = mirror_window_features(features, torch.tensor([True, False]))
    assert torch.equal(mirrored[0, ..., SIDE_FEATURE], -features[0, ..., SIDE_FEATURE])
    others = [index for index in range(len(WINDOW_FEATURES)) if index != SIDE_FEATURE]
    assert torch.equal(mirrored[0, ..., others], features[0, ..., others])
    assert torch.equal(mirrored[1], features[1])
    images = torch.rand(2, 1, 4, 3, 8, 8)
    planes = torch.tensor([[[0, 1, 2, 1]]]).repeat(2, 1, 1)
    settings = validate_gpu_augmentation({"enabled": True, "mirror_prob": 1.0})
    _, changed = augment_batch(images, planes, settings, torch.Generator().manual_seed(0), features)
    assert torch.equal(changed[..., SIDE_FEATURE], -features[..., SIDE_FEATURE])


@pytest.mark.parametrize("mode", ["types", "types_and_positions"])
def test_zero_initialized_descriptors_start_from_identical_predictions(mode):
    config = {"backbone": "resnet18", "pretrained": False, "pooling": "class_attention",
              "use_metadata": False, "use_positions": False}
    torch.manual_seed(3)
    plain = StudyModel(config).eval()
    torch.manual_seed(3)
    described = StudyModel({**config, "window_features": mode}).eval()
    inputs = {"images": torch.rand(2, 1, 4, 3, 32, 32), "series_mask": torch.ones(2, 1, dtype=torch.bool),
              "metadata": torch.zeros(2, 1, 5), "positions": torch.zeros(2, 1, 4)}
    features = torch.randn(2, 1, 4, len(WINDOW_FEATURES))
    with torch.no_grad():
        torch.testing.assert_close(plain(**inputs), described(**inputs, window_features=features))
    with pytest.raises(ValueError, match="window_features"):
        described(**inputs)


def test_descriptors_need_the_raptor_recipe_and_class_attention():
    config = deepcopy(DEFAULTS)
    config["model"]["window_features"] = "types"
    with pytest.raises(ValueError, match="window_features"):
        validate_config(config)
    config["model"]["window_features"] = "everything"
    with pytest.raises(ValueError, match="window_features"):
        validate_config(config)


def test_training_with_descriptors_and_mirroring_resumes_exactly(tmp_path):
    studies = [f"s{index}" for index in range(6)]
    make_raw(tmp_path / "raw", studies=studies)
    manifest = prepare_raptor_data(tmp_path / "raw", tmp_path / "cache", workers=1)
    rng = np.random.default_rng(9)
    pd.DataFrame({STUDY_ID: studies, **{name: rng.random(len(studies)).round() for name in TARGETS},
                  "label_source": "report"}).to_csv(tmp_path / "labels.csv", index=False)
    pd.DataFrame({STUDY_ID: studies, "fold": [0, 0, 1, 1, 1, 1], "split": "train",
                  "group_id": studies}).to_csv(tmp_path / "splits.csv", index=False)
    config = deepcopy(DEFAULTS)
    config["data"].update({
        "recipe": "raptor", "manifest": str(manifest), "labels": str(tmp_path / "labels.csv"),
        "splits": str(tmp_path / "splits.csv"), "cache_size": 336, "image_size": 32, "raptor_train_windows": 4,
        "raptor_eval_windows": 5, "test_time_mirror": True,
        "gpu_augmentation": {**DEFAULT_GPU_AUGMENTATION, "enabled": True, "mirror_prob": 0.5},
    })
    config["model"].update({"backbone": "resnet18", "pretrained": False, "pooling": "class_attention",
                            "use_metadata": False, "use_positions": False, "encoder_chunk_size": 8,
                            "window_features": "types_and_positions"})
    config["training"].update({"output_dir": str(tmp_path / "full"), "epochs": 2, "batch_size": 2,
                               "accumulation_steps": 1, "num_workers": 0, "device": "cpu", "precision": "fp32",
                               "selection_metric": "loss", "deterministic": True})
    train(config)
    resumed = deepcopy(config)
    resumed["training"]["output_dir"] = str(tmp_path / "resumed")
    train(resumed, stop_after_epoch=1)
    train(resumed, resume=tmp_path / "resumed" / "last.pt")
    complete = torch.load(tmp_path / "full" / "last.pt", weights_only=True)
    restarted = torch.load(tmp_path / "resumed" / "last.pt", weights_only=True)
    for name, value in complete["model"].items():
        assert torch.equal(value, restarted["model"][name]), name
    assert complete["model"]["window_feature_projection.weight"].abs().sum() > 0
