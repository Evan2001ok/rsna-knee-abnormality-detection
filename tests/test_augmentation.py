"""GPU-style augmentation, window sampling, weight averaging, and test-time mirroring."""

from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch
from test_raptor_data import make_raw

from rsna_knee_abnormality_detection.augment import (
    DEFAULT_GPU_AUGMENTATION,
    augment_batch,
    mirror_windows,
    validate_gpu_augmentation,
)
from rsna_knee_abnormality_detection.config import DEFAULTS, validate_config
from rsna_knee_abnormality_detection.constants import STUDY_ID, TARGETS
from rsna_knee_abnormality_detection.raptor_data import RaptorDataset, prepare_raptor_data
from rsna_knee_abnormality_detection.train import predict_loader, train, update_ema


def bag(planes=(0, 1, 2), size=8, studies=2):
    generator = torch.Generator().manual_seed(0)
    images = torch.rand((studies, 1, len(planes), 3, size, size), generator=generator)
    window_planes = torch.tensor(planes).repeat(studies, 1).unsqueeze(1)
    return images, window_planes


def settings(**values):
    return validate_gpu_augmentation({"enabled": True, **values})


def test_mirror_flips_coronal_and_axial_and_reverses_sagittal_slice_order():
    images, planes = bag()
    mirrored = mirror_windows(images, planes, torch.tensor([True, False]))
    assert torch.equal(mirrored[0, 0, 0], images[0, 0, 0].flip(0))  # sagittal: slice order reversed
    assert torch.equal(mirrored[0, 0, 1], images[0, 0, 1].flip(-1))  # coronal: left-right flip
    assert torch.equal(mirrored[0, 0, 2], images[0, 0, 2].flip(-1))  # axial: left-right flip
    assert torch.equal(mirrored[1], images[1])
    assert torch.equal(mirror_windows(mirrored, planes, torch.tensor([True, False])), images)


def test_disabled_zero_probability_and_identity_transforms_keep_pixels():
    images, planes = bag()
    generator = torch.Generator().manual_seed(1)
    assert augment_batch(images, planes, validate_gpu_augmentation({}), generator)[0] is images
    torch.testing.assert_close(augment_batch(images, planes, settings(), generator)[0], images)
    identity = settings(affine_prob=1, rotation_deg=0, zoom_min=1, zoom_max=1, translate=0)
    torch.testing.assert_close(augment_batch(images, planes, identity, generator)[0], images, atol=1e-6, rtol=0)


def test_augmentation_is_reproducible_bounded_and_seed_dependent():
    images, planes = bag(size=16)
    every = settings(mirror_prob=.5, affine_prob=1, intensity_prob=1, bias_field_prob=1, noise_prob=1,
                     blur_prob=.5, lowres_prob=.5)
    first = augment_batch(images, planes, every, torch.Generator().manual_seed(3))[0]
    again = augment_batch(images, planes, every, torch.Generator().manual_seed(3))[0]
    other = augment_batch(images, planes, every, torch.Generator().manual_seed(4))[0]
    assert torch.equal(first, again)
    assert not torch.equal(first, other)
    assert first.shape == images.shape and first.dtype == images.dtype
    assert 0 <= first.min() <= first.max() <= 1


@pytest.mark.parametrize("bad", [{"rotation": 5}, {"mirror_prob": 1.5}, {"zoom_min": 1.2, "zoom_max": 1.1},
                                 {"gamma": .5}, {"noise_std": -1}, {"enabled": "yes"}, {"contrast": 1.0}])
def test_invalid_augmentation_settings_are_rejected(bad):
    with pytest.raises(ValueError):
        validate_gpu_augmentation(bad)


def test_gpu_augmentation_and_mirroring_need_raptor_window_planes():
    config = deepcopy(DEFAULTS)
    config["data"]["gpu_augmentation"] = {**DEFAULT_GPU_AUGMENTATION, "enabled": True}
    with pytest.raises(ValueError, match="raptor"):
        validate_config(config)
    config["data"]["gpu_augmentation"]["enabled"] = False
    config["training"]["ema_decay"] = 1.0
    with pytest.raises(ValueError, match="ema_decay"):
        validate_config(config)


def sampler(windows, sampling="stratified", dropout=0.0):
    return SimpleNamespace(windows=windows, sampling=sampling, series_dropout=dropout)


def record():
    # Five series: sagittal 0/1, coronal 2/3, axial 4, with 16/12/10/6/10 windows.
    slots = np.repeat([0, 1, 2, 3, 4], [16, 12, 10, 6, 10])
    planes = np.repeat([0, 0, 1, 1, 2], [16, 12, 10, 6, 10])
    return {"slots": slots, "planes": planes}


def test_stratified_sampling_gives_every_series_a_proportional_share():
    data = record()
    for seed in range(20):
        choices = RaptorDataset._sample_windows(sampler(12), data, np.random.default_rng(seed))
        assert len(choices) == 12 and len(set(choices)) == 12
        # 12 windows split 16:12:10:6:10, rounded down, then largest remainders first.
        assert np.bincount(data["slots"][choices], minlength=5).tolist() == [4, 3, 2, 1, 2]


def test_series_hiding_keeps_at_least_one_series_for_each_plane():
    data = record()
    for seed in range(50):
        choices = RaptorDataset._sample_windows(sampler(12, dropout=.95), data, np.random.default_rng(seed))
        assert len(choices) == 12
        assert set(data["planes"][choices]) == {0, 1, 2}
        kept = set(data["slots"][choices])
        assert len(kept & {0, 1}) >= 1 and len(kept & {2, 3}) >= 1 and 4 in kept


def test_running_average_moves_toward_weights_and_copies_counters():
    model = torch.nn.Sequential(torch.nn.Linear(2, 2), torch.nn.BatchNorm1d(2))
    average = deepcopy(model)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.add_(2.0)
        model[1].num_batches_tracked.fill_(7)
    before = deepcopy(average.state_dict())
    update_ema(average, model, decay=0.5, step=10_000)
    for name, value in average.state_dict().items():
        if value.is_floating_point():
            torch.testing.assert_close(value, (before[name] + model.state_dict()[name]) / 2)
    assert int(average[1].num_batches_tracked) == 7


class LeftRightModel(torch.nn.Module):
    """Scores the left-right brightness difference, which mirroring negates."""

    def forward(self, images, series_mask, metadata, positions):
        width = images.shape[-1]
        difference = images[..., : width // 2].mean(dim=(1, 2, 3, 4, 5)) - images[..., width // 2:].mean(dim=(1, 2, 3, 4, 5))
        return (10 * difference).unsqueeze(1).repeat(1, len(TARGETS))


def test_test_time_mirroring_averages_the_study_and_its_mirror_image():
    images = torch.zeros(2, 1, 3, 3, 8, 8)
    images[..., :4] = 1.0
    batch = {"images": images, "series_mask": torch.ones(2, 1, dtype=torch.bool), "metadata": torch.zeros(2, 1, 5),
             "positions": torch.zeros(2, 1, 3), "window_planes": torch.ones(2, 1, 3, dtype=torch.long),
             "targets": torch.zeros(2, len(TARGETS)), "target_mask": torch.zeros(2, len(TARGETS), dtype=torch.bool),
             "study_id": ["a", "b"]}
    plain, _ = predict_loader(LeftRightModel(), [batch], torch.device("cpu"), "fp32")
    mirrored, _ = predict_loader(LeftRightModel(), [batch], torch.device("cpu"), "fp32", test_time_mirror=True)
    assert (plain[TARGETS].to_numpy() > 0.99).all()
    np.testing.assert_allclose(mirrored[TARGETS].to_numpy(), 0.5)


def test_raptor_training_with_all_new_options_resumes_exactly(tmp_path):
    studies = [f"s{index}" for index in range(8)]
    make_raw(tmp_path / "raw", studies=studies)
    manifest = prepare_raptor_data(tmp_path / "raw", tmp_path / "cache", workers=1)
    rng = np.random.default_rng(5)
    labels = pd.DataFrame({STUDY_ID: studies, **{name: rng.random(len(studies)).round() for name in TARGETS},
                           "label_source": "report"})
    labels.to_csv(tmp_path / "labels.csv", index=False)
    pd.DataFrame({STUDY_ID: studies, "fold": [0, 0, 0, 1, 1, 1, 1, 1], "split": "train",
                  "group_id": studies}).to_csv(tmp_path / "splits.csv", index=False)
    config = deepcopy(DEFAULTS)
    config["seed"] = 11
    config["data"].update({
        "recipe": "raptor", "manifest": str(manifest), "labels": str(tmp_path / "labels.csv"),
        "splits": str(tmp_path / "splits.csv"), "fold": 0, "cache_size": 336, "image_size": 32,
        "raptor_train_windows": 5, "raptor_eval_windows": 6, "raptor_window_sampling": "stratified",
        "raptor_series_dropout": 0.3, "test_time_mirror": True,
        "gpu_augmentation": {**DEFAULT_GPU_AUGMENTATION, "enabled": True, "mirror_prob": .5, "affine_prob": .5,
                             "intensity_prob": .5, "bias_field_prob": .5, "noise_prob": .5, "blur_prob": .5,
                             "lowres_prob": .5},
    })
    config["model"].update({"backbone": "resnet18", "pretrained": False, "pooling": "class_attention",
                            "use_metadata": False, "use_positions": False, "encoder_chunk_size": 8})
    config["training"].update({"output_dir": str(tmp_path / "full"), "epochs": 2, "batch_size": 2,
                               "accumulation_steps": 1, "num_workers": 0, "device": "cpu", "precision": "fp32",
                               "selection_metric": "loss", "deterministic": True, "ema_decay": 0.9})
    config["monitoring"].update({"log_every_updates": 1})
    train(config)
    resumed = deepcopy(config)
    resumed["training"]["output_dir"] = str(tmp_path / "resumed")
    train(resumed, stop_after_epoch=1)
    train(resumed, resume=tmp_path / "resumed" / "last.pt")
    complete = torch.load(tmp_path / "full" / "last.pt", weights_only=True)
    restarted = torch.load(tmp_path / "resumed" / "last.pt", weights_only=True)
    for key in ("model", "training_model"):
        for name, value in complete[key].items():
            assert torch.equal(value, restarted[key][name]), (key, name)
    assert complete["history"][-1]["validation"] == restarted["history"][-1]["validation"]
    assert not all(torch.equal(value, complete["training_model"][name]) for name, value in complete["model"].items())
