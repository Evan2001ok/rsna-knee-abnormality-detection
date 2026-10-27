"""Read real TensorBoard events to check logging and restart behavior."""

import io
import json
import random
import subprocess
import sys
from copy import deepcopy

import numpy as np
import pytest
import torch
import yaml
from PIL import Image
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from rsna_knee_abnormality_detection.config import DEFAULTS
from rsna_knee_abnormality_detection.constants import TARGETS
from rsna_knee_abnormality_detection.metrics import score_predictions
from rsna_knee_abnormality_detection.monitoring import TrainingMonitor, import_histories, input_image


def read_events(directory):
    return EventAccumulator(str(directory), size_guidance={"scalars": 0, "images": 0, "tensors": 0}).Reload()


def text_value(events, tag):
    return events.Tensors(tag + "/text_summary")[-1].tensor_proto.string_val[0].decode()


def configuration():
    config = deepcopy(DEFAULTS)
    config["monitoring"].update({"log_every_updates": 2, "flush_seconds": 1})
    return config


def record(step=2, epoch=1, missing=True):
    targets = np.tile([[0], [1]], (1, len(TARGETS))).astype(float)
    probabilities = np.tile([[0.2], [0.8]], (1, len(TARGETS)))
    if missing:
        targets[:, TARGETS.index("MCL")] = np.nan
    return {
        "global_step": step, "epoch": epoch, "train_loss": 0.7,
        "validation": score_predictions(targets, probabilities), "seconds": 2.5,
        "peak_gpu_gb": 1.25, "learning_rates": [0.0001, 0.0003], "skipped_updates": 0,
    }


def batch():
    images = torch.zeros(1, 3, 5, 3, 16, 24)
    for slot in [0, 2]:
        for group in range(5):
            images[0, slot, group, 1] = (group + 1) / 6
            images[0, slot, group, 2] = 1
    images[:, 1] = float("nan")
    return {
        "images": images,
        "series_mask": torch.tensor([[True, False, True]]),
        "metadata": torch.tensor([[[1, 0, 0, 1, 1], [0, 0, 0, -1, -1], [0, 0, 1, 1, 0]]]),
        "positions": torch.linspace(0, 1, 5).expand(1, 3, 5).clone(),
    }


def test_events_record_real_values_and_skip_undefined_auc(tmp_path):
    first = record()
    with TrainingMonitor(tmp_path, configuration()) as monitor:
        monitor.support(first["validation"], 0)
        monitor.update(1, 1.4, {"encoder": 0.0002, "head": 0.0006})
        monitor.update(2, 0.8, {"encoder": 0.0001, "head": 0.0003})
        monitor.epoch(first)
        monitor.scalar("Invalid/missing", None, 2)
        monitor.scalar("Invalid/nonfinite", float("nan"), 2)
    events = read_events(tmp_path / "tensorboard")
    tags = events.Tags()["scalars"]
    assert "AUC/macro_12_findings" not in tags
    assert "AUC_by_finding/MCL" not in tags
    assert not any(tag.startswith("Invalid/") for tag in tags)
    assert [(event.step, event.value) for event in events.Scalars("Loss/train_running")] == [(2, pytest.approx(0.8))]
    assert events.Scalars("Loss/validation")[0].value == pytest.approx(first["validation"]["loss"])
    assert events.Scalars("Learning_rate/encoder")[0].value == pytest.approx(0.0001)
    assert events.Scalars("Learning_rate/head")[0].value == pytest.approx(0.0003)
    assert events.Scalars("AUC/available_findings_only")[0].value == 1.0
    assert events.Scalars("AUC_by_finding/Medial_Meniscus")[0].value == 1.0
    assert events.Scalars("Progress/auc_findings_available")[0].value == 11
    assert "Validation/label_counts/text_summary" in events.Tags()["tensors"]


def test_update_cadence_measurements_and_flush_are_visible_before_close(tmp_path):
    config = configuration()
    config["monitoring"].update(log_every_updates=10, flush_seconds=3600)
    expected_steps = []
    with TrainingMonitor(tmp_path, config) as monitor:
        assert not monitor.should_log(1)
        assert monitor.should_log(1, force=True)
        for step in range(1, 26):
            force = step == 25
            due = step in [10, 20, 25]
            assert monitor.should_log(step, force=force) is due
            monitor.update(step, 1.0 / step, {"encoder": 0.001, "head": 0.01}, force=force, measurements={
                "Loss/train_recent": 2.0 / step,
                "Optimization/gradient_norm": step / 10,
                "Performance/studies_per_second": 3.5,
                "Invalid/missing": None,
                "Invalid/not_finite": float("inf"),
            })
            if due:
                expected_steps.append(step)
                # The periodic writer flush is one hour away, and this writer
                # is still open. Each logged update must already be readable.
                events = read_events(tmp_path / "tensorboard")
                for tag in ["Loss/train_running", "Loss/train_recent", "Optimization/gradient_norm",
                            "Performance/studies_per_second", "Learning_rate/encoder", "Learning_rate/head"]:
                    assert [event.step for event in events.Scalars(tag)] == expected_steps
    assert [event.value for event in events.Scalars("Loss/train_recent")] == pytest.approx([0.2, 0.1, 0.08])
    assert [event.value for event in events.Scalars("Optimization/gradient_norm")] == pytest.approx([1, 2, 2.5])
    assert not any(tag.startswith("Invalid/") for tag in events.Tags()["scalars"])


def test_sample_validation_is_separate_and_skips_undefined_auc(tmp_path):
    config = configuration()
    config["monitoring"].update(flush_seconds=3600)
    incomplete = record(missing=True)["validation"]
    complete = record(missing=False)["validation"]
    no_auc = score_predictions(np.full((2, len(TARGETS)), 0.5), np.full((2, len(TARGETS)), 0.5))
    with TrainingMonitor(tmp_path, config) as monitor:
        monitor.validation_sample(incomplete, 10, 1.5)
        events = read_events(tmp_path / "tensorboard")
        assert events.Scalars("Validation_sample/loss")[0].value == pytest.approx(incomplete["loss"])
        assert "Validation_sample/macro_12_findings" not in events.Tags()["scalars"]
        assert "Validation_sample/AUC_by_finding/MCL" not in events.Tags()["scalars"]
        assert not any(tag.startswith(("Loss/validation", "AUC/", "AUC_by_finding/")) for tag in events.Tags()["scalars"])
        monitor.validation_sample(complete, 20, 1.25)
        monitor.validation_sample(no_auc, 30, 1.0)
        events = read_events(tmp_path / "tensorboard")
        assert [event.step for event in events.Scalars("Validation_sample/macro_12_findings")] == [20]
        assert [event.step for event in events.Scalars("Validation_sample/available_findings_only")] == [10]
        assert [event.step for event in events.Scalars("Validation_sample/AUC_by_finding/MCL")] == [20]
        assert [event.value for event in events.Scalars("Validation_sample/auc_findings_available")] == [11, 12, 0]
        assert [event.value for event in events.Scalars("Validation_sample/seconds")] == pytest.approx([1.5, 1.25, 1.0])
        assert [event.value for event in events.Scalars("Validation_sample/studies")] == [2, 2, 2]
        monitor.epoch(record(step=40, epoch=1, missing=False))
    events = read_events(tmp_path / "tensorboard")
    assert [event.step for event in events.Scalars("Loss/validation")] == [40]
    assert [event.step for event in events.Scalars("AUC/macro_12_findings")] == [40]
    assert [event.step for event in events.Scalars("Validation_sample/loss")] == [10, 20, 30]


def test_sample_support_records_ids_counts_and_selection_policy(tmp_path):
    config = configuration()
    config["monitoring"].update(validation_every_updates=10, validation_studies=32, flush_seconds=3600)
    metrics = record()["validation"]
    with TrainingMonitor(tmp_path, config) as monitor:
        monitor.sample_support(metrics, ["study-first", "study-second"])
        events = read_events(tmp_path / "tensorboard")
        support = text_value(events, "Validation_sample/label_counts")
        assert "2 studies" in support
        assert "monitoring only" in support
        assert "Full validation at each epoch selects the best checkpoint" in support
        assert "| MCL | 0 | 0 | 0 | 2 |" in support
        assert '"study-first"' in support and '"study-second"' in support
        assert events.Tensors("Validation_sample/label_counts/text_summary")[0].step == 0
        settings = text_value(events, "Run/settings")
        assert "32 validation studies" in settings and "every 10 optimizer updates" in settings
        assert "Loss/train_running is the mean since the start" in settings
        assert "Loss/train_recent covers the current logging interval" in settings


def test_run_settings_describe_disabled_sample_validation(tmp_path):
    config = configuration()
    config["monitoring"].update(validation_every_updates=0, validation_studies=32)
    with TrainingMonitor(tmp_path, config):
        pass
    events = read_events(tmp_path / "tensorboard")
    settings = text_value(events, "Run/settings")
    assert "Validation sampling between epochs is disabled" in settings
    assert "validation_every_updates=0" in settings
    assert "Full validation is recorded after each epoch" in settings
    assert not any(tag.startswith("Validation_sample/") for tag in events.Tags()["scalars"])


def test_zero_sample_size_means_full_validation_and_legacy_interval_means_disabled(tmp_path):
    full_config = configuration()
    full_config["monitoring"].update(validation_every_updates=10, validation_studies=0)
    with TrainingMonitor(tmp_path / "full", full_config) as monitor:
        monitor.sample_support(record()["validation"], ["first", "second"])
    events = read_events(tmp_path / "full" / "tensorboard")
    settings = text_value(events, "Run/settings")
    assert "All validation studies (validation_studies=0) are evaluated every 10 optimizer updates" in settings
    assert "Validation sampling between epochs is disabled" not in settings
    support = text_value(events, "Validation_sample/label_counts")
    assert "Full validation between epochs: **2 studies**" in support
    assert "entire validation partition" in support
    legacy = configuration()
    legacy["monitoring"].pop("validation_every_updates", None)
    with TrainingMonitor(tmp_path / "legacy", legacy):
        pass
    events = read_events(tmp_path / "legacy" / "tensorboard")
    assert "validation_every_updates=0" in text_value(events, "Run/settings")


def test_input_grid_is_rgb_uses_real_series_and_preserves_batch_and_rng(tmp_path):
    example = batch()
    before = {key: value.clone() for key, value in example.items()}
    python_rng, numpy_rng, torch_rng = random.getstate(), np.random.get_state(), torch.get_rng_state().clone()
    grid = input_image(example)
    with TrainingMonitor(tmp_path, configuration()) as monitor:
        monitor.batch_images(example, "training_augmented", 1)
    assert random.getstate() == python_rng
    after_numpy = np.random.get_state()
    assert after_numpy[0] == numpy_rng[0] and after_numpy[2:] == numpy_rng[2:]
    np.testing.assert_array_equal(after_numpy[1], numpy_rng[1])
    assert torch.equal(torch.get_rng_state(), torch_rng)
    for key, original in before.items():
        torch.testing.assert_close(example[key], original, equal_nan=True)
    assert grid.dtype == np.uint8 and grid.shape == (2 * (16 + 24), 4 * 24, 3)
    expected_gray = int((1 / 6) * 255)
    assert grid[24, 0].tolist() == [expected_gray] * 3
    events = read_events(tmp_path / "tensorboard")
    image_event = events.Images("Inputs/training_augmented")[0]
    decoded = Image.open(io.BytesIO(image_event.encoded_image_string))
    assert decoded.mode == "RGB"
    np.testing.assert_array_equal(np.asarray(decoded), grid)


def test_resume_purges_abandoned_updates_and_images_after_checkpoint(tmp_path):
    config = configuration()
    original = batch()
    with TrainingMonitor(tmp_path, config) as monitor:
        monitor.update(2, 0.8, {"encoder": 0.0001})
        monitor.epoch(record(step=2))
        monitor.batch_images(original, "training_augmented", 1)
        monitor.update(4, 9.0, {"encoder": 0.9})
        abandoned = record(step=4, epoch=2)
        abandoned["train_loss"] = 9.0
        monitor.epoch(abandoned)
        monitor.batch_images(original, "training_augmented", 3)
    replacement = batch()
    replacement["images"][0, 0, :, 1] = 0.25
    with TrainingMonitor(tmp_path, config, resume_step=2) as monitor:
        monitor.update(3, 0.4, {"encoder": 0.00005}, force=True)
        monitor.batch_images(replacement, "training_augmented", 3)
        resumed = record(step=4, epoch=2)
        resumed["train_loss"] = 0.3
        monitor.epoch(resumed)
    events = read_events(tmp_path / "tensorboard")
    assert [event.step for event in events.Scalars("Loss/train_running")] == [2, 3]
    assert [event.value for event in events.Scalars("Loss/train_running")] == pytest.approx([0.8, 0.4])
    assert [event.step for event in events.Scalars("Loss/train_epoch")] == [2, 4]
    assert [event.value for event in events.Scalars("Loss/train_epoch")] == pytest.approx([0.7, 0.3])
    image_events = events.Images("Inputs/training_augmented")
    assert [event.step for event in image_events] == [1, 3]
    actual = np.asarray(Image.open(io.BytesIO(image_events[1].encoded_image_string)))
    np.testing.assert_array_equal(actual, input_image(replacement))


def test_importing_history_is_idempotent_and_preserves_run_artifacts(tmp_path):
    run = tmp_path / "legacy"
    run.mkdir()
    config = configuration()
    config.pop("monitoring")
    history = [record(), record(step=4, epoch=2, missing=False)]
    (run / "config.yaml").write_text(yaml.safe_dump(config))
    (run / "history.json").write_text(json.dumps(history))
    torch.save({"weights": torch.arange(4)}, run / "last.pt")
    original = {path.name: path.read_bytes() for path in run.iterdir() if path.is_file()}
    result = import_histories(tmp_path)
    assert result == {"imported": [str(run)], "already_logged": []}
    events = read_events(run / "tensorboard")
    assert [event.step for event in events.Scalars("Loss/validation")] == [2, 4]
    assert len(events.Scalars("Learning_rate/encoder")) == 2
    assert len(events.Scalars("Learning_rate/head")) == 2
    assert not events.Tags()["images"]
    assert not any(tag.startswith("Validation_sample/") for tag in events.Tags()["scalars"])
    assert "Loss/train_recent" not in events.Tags()["scalars"]
    assert "epoch summaries only" in text_value(events, "Run/history_import")
    assert "validation-sample measurements" in text_value(events, "Run/history_import")
    assert "Validation sampling between epochs is disabled" in text_value(events, "Run/settings")
    event_bytes = {path.name: path.read_bytes() for path in (run / "tensorboard").iterdir()}
    second = import_histories(tmp_path)
    assert second == {"imported": [], "already_logged": [str(run)]}
    assert {path.name: path.read_bytes() for path in (run / "tensorboard").iterdir()} == event_bytes
    for name, content in original.items():
        assert (run / name).read_bytes() == content


def test_failed_history_import_can_be_corrected_and_retried(tmp_path):
    config = configuration()
    (tmp_path / "config.yaml").write_text(yaml.safe_dump(config))
    history = [record(), record(step=4, epoch=2)]
    history[1]["learning_rates"] = [0.0001]
    history_path = tmp_path / "history.json"
    history_path.write_text(json.dumps(history))
    original = history_path.read_bytes()
    with pytest.raises(ValueError, match="zip"):
        import_histories(tmp_path)
    assert not (tmp_path / "tensorboard").exists()
    assert history_path.read_bytes() == original
    history[1]["learning_rates"] = [0.0001, 0.0003]
    history_path.write_text(json.dumps(history))
    assert import_histories(tmp_path)["imported"] == [str(tmp_path)]
    events = read_events(tmp_path / "tensorboard")
    assert [event.step for event in events.Scalars("Loss/validation")] == [2, 4]


def test_disabled_monitor_and_inference_import_do_not_require_tensorboard(tmp_path):
    script = """
import importlib.abc
import sys

class NoTensorBoard(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'torch.utils.tensorboard' or fullname.split('.')[0] == 'tensorboard':
            raise ModuleNotFoundError('TensorBoard is intentionally unavailable')

sys.meta_path.insert(0, NoTensorBoard())
from rsna_knee_abnormality_detection.monitoring import TrainingMonitor
with TrainingMonitor(sys.argv[1], {'monitoring': {'enabled': False}}) as monitor:
    assert not monitor.should_log(10)
    assert not monitor.should_log(10, force=True)
    monitor.scalar('Loss/train', 0.5, 1)
    monitor.support({}, 0)
    monitor.sample_support({}, ['a-study'], 0)
    monitor.validation_sample({}, 10, 0.1)
    monitor.batch_images(None, 'validation', 1)
    monitor.update(10, 0.5, {}, force=True, measurements={'Loss/train_recent': 0.4})
    monitor.status(True, 1, 1)
from rsna_knee_abnormality_detection.predict import predict
assert not any(name.split('.')[0] == 'tensorboard' for name in sys.modules)
print('no TensorBoard imports')
"""
    result = subprocess.run([sys.executable, "-c", script, str(tmp_path)], capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "no TensorBoard imports" in result.stdout
    assert not (tmp_path / "tensorboard").exists()
