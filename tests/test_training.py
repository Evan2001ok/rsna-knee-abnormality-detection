"""Exercise the full study-training path, including restart and held-out data."""

import json
import subprocess
import sys
from copy import deepcopy

import numpy as np
import pandas as pd
import pytest
import torch
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from test_imaging import write_dicom

from rsna_knee_abnormality_detection.config import DEFAULTS, load_config, validate_config
from rsna_knee_abnormality_detection.constants import STUDY_ID, TARGETS
from rsna_knee_abnormality_detection.metrics import read_predictions
from rsna_knee_abnormality_detection.predict import predict
from rsna_knee_abnormality_detection.prepare import prepare_data
from rsna_knee_abnormality_detection.train import study_partitions, train


@pytest.fixture(autouse=True)
def two_threads(monkeypatch):
    import os

    previous = torch.get_num_threads()
    monkeypatch.setattr(os, "cpu_count", lambda: 2)
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)
    torch.use_deterministic_algorithms(False)


@pytest.fixture
def experiment(tmp_path):
    generator = np.random.default_rng(21)
    series_rows, labels, splits = [], [], []
    raw = tmp_path / "raw"
    raw.mkdir()
    definitions = [
        *[(f"train-{i}", 1, "train", "report", f"patient-t{i}") for i in range(6)],
        *[(f"val-{i}", 0, "train", "report", f"patient-v{i}") for i in range(4)],
        ("expert", -1, "expert_holdout", "expert", "expert-patient"),
        ("expert-patient-repeat", -1, "expert_holdout", "report", "expert-patient"),
        ("unlabeled", -2, "unlabeled", "missing", "unlabeled-patient"),
    ]
    for index, (study, fold, split, source, group) in enumerate(definitions):
        volume = generator.integers(0, 65536, size=(7, 32, 32), dtype=np.uint16)
        series = f"series-{study}"
        for slice_index, pixels in enumerate(volume):
            # File names and instance numbers intentionally disagree with
            # physical order, just as the real preparation path must handle.
            write_dicom(
                raw / "train_series" / study / series / f"slice-{6 - slice_index}.dcm",
                pixels, position=(slice_index * 2, 0, 0),
                orientation=(0, 1, 0, 0, 0, -1), instance=7 - slice_index,
            )
        series_rows.append({
            STUDY_ID: study, "SeriesInstanceUID": f"series-{study}",
            "Anatomical_Plane": "Sagittal", "Fluid_Sensitive": 1,
            "Fat_Suppression": 1,
        })
        values = {name: float((index + target) % 2) for target, name in enumerate(TARGETS)}
        if split == "train" and fold == 1:
            # Unequal observed-label counts exercise correct accumulation weights.
            for name in TARGETS[2 + index:]:
                values[name] = np.nan
            values["MCL"] = 0.25 if index % 2 else 0.75
        if source == "missing":
            values = dict.fromkeys(TARGETS, np.nan)
        labels.append({STUDY_ID: study, **values, "label_source": source})
        splits.append({STUDY_ID: study, "fold": fold, "split": split, "group_id": group})
    labels_path, splits_path = [tmp_path / name for name in ["labels.csv", "splits.csv"]]
    pd.DataFrame(series_rows).to_csv(raw / "train_series.csv", index=False)
    pd.DataFrame(labels)[[STUDY_ID]].to_csv(raw / "train.csv", index=False)
    manifest_path = prepare_data(raw, tmp_path / "cache", size=32, workers=1, strict=True)
    pd.DataFrame(labels).to_csv(labels_path, index=False)
    pd.DataFrame(splits).to_csv(splits_path, index=False)
    config = deepcopy(DEFAULTS)
    config["seed"] = 77
    config["data"].update({
        "manifest": str(manifest_path), "labels": str(labels_path), "splits": str(splits_path),
        "fold": 0, "cache_size": 32, "image_size": 32, "slices_per_series": 2, "series_per_plane": 1,
    })
    config["model"].update({"backbone": "resnet18", "pretrained": False, "encoder_chunk_size": 8})
    config["training"].update({
        "output_dir": str(tmp_path / "run"), "epochs": 2, "batch_size": 2,
        "accumulation_steps": 2, "num_workers": 0, "device": "cpu", "precision": "fp32",
        "selection_metric": "loss", "deterministic": True,
    })
    config["monitoring"].update({"log_every_updates": 1, "validation_every_updates": 1, "validation_studies": 2})
    return config


def test_training_resume_matches_uninterrupted_run_and_offline_inference(experiment, tmp_path):
    uninterrupted = deepcopy(experiment)
    uninterrupted["training"]["output_dir"] = str(tmp_path / "uninterrupted")
    train(uninterrupted)
    full = torch.load(tmp_path / "uninterrupted" / "last.pt", map_location="cpu", weights_only=True)

    train(experiment, stop_after_epoch=1)
    output = tmp_path / "run"
    status = json.loads((output / "status.json").read_text())
    assert status["epochs_finished"] == 1 and not status["completed"]
    train(experiment, resume=output / "last.pt")
    resumed = torch.load(output / "last.pt", map_location="cpu", weights_only=True)
    assert resumed["epoch"] == full["epoch"] == 2
    assert resumed["image_preparation"]["image_size"] == 32
    assert resumed["image_preparation"]["preprocess_version"]
    assert resumed["global_step"] == full["global_step"] == 4
    assert resumed["scheduler"] == full["scheduler"]
    assert resumed["optimizer"]["param_groups"] == full["optimizer"]["param_groups"]
    for name, expected in full["model"].items():
        assert torch.equal(expected, resumed["model"][name]), name
    for expected, actual in zip(full["history"], resumed["history"], strict=True):
        assert expected["train_loss"] == actual["train_loss"]
        assert expected["validation"] == actual["validation"]
    for run in [output, tmp_path / "uninterrupted"]:
        events = EventAccumulator(str(run / "tensorboard"), size_guidance={"scalars": 0, "images": 0}).Reload()
        assert [entry.step for entry in events.Scalars("Loss/train_running")] == [1, 2, 3, 4]
        assert [entry.step for entry in events.Scalars("Loss/train_epoch")] == [2, 4]
        assert [entry.step for entry in events.Scalars("Loss/validation")] == [2, 4]
        assert [entry.step for entry in events.Scalars("Validation_sample/loss")] == [1, 2, 3, 4]
        assert [entry.value for entry in events.Scalars("Validation_sample/studies")] == [2] * 4
        assert [entry.step for entry in events.Scalars("Loss/train_recent")] == [1, 2, 3, 4]
        assert all(entry.value > 0 for entry in events.Scalars("Performance/training_studies_per_second"))
        assert all(entry.value > 0 for entry in events.Scalars("Optimization/gradient_norm_before_clip"))
        assert [entry.step for entry in events.Images("Inputs/training_augmented")] == [1, 2, 3, 4]
        assert [entry.step for entry in events.Images("Inputs/validation")] == [2, 4]
        assert [entry.step for entry in events.Images("Inputs/validation_sample")] == [1, 2, 3, 4]
        sampled_ids = pd.read_csv(run / "monitoring-validation-sample.csv")[STUDY_ID].tolist()
        assert len(sampled_ids) == 2 and all(study.startswith("val-") for study in sampled_ids)
        assert sampled_ids == resumed["provenance"]["monitoring"]["validation_sample_ids"]
    status = json.loads((output / "status.json").read_text())
    assert status["completed"] and status["epochs_finished"] == 2
    partitions = pd.read_csv(output / "partitions.csv")
    assert not partitions[STUDY_ID].str.startswith("expert").any()
    assert "unlabeled" not in set(partitions[STUDY_ID])

    # Inference uses the saved configuration and never needs pretrained downloads.
    reference = read_predictions(output / "validation-last.csv")
    shuffled = reference[[STUDY_ID]].iloc[::-1]
    studies_csv = tmp_path / "prediction-studies.csv"
    shuffled.to_csv(studies_csv, index=False)
    predicted_path = predict(
        experiment["data"]["manifest"], studies_csv, [output / "last.pt"],
        tmp_path / "predictions.csv", batch_size=2, num_workers=0, device="cpu",
    )
    actual = read_predictions(predicted_path)
    assert actual[STUDY_ID].tolist() == shuffled[STUDY_ID].tolist()
    aligned = actual.set_index(STUDY_ID).loc[reference[STUDY_ID], TARGETS].to_numpy()
    np.testing.assert_allclose(aligned, reference[TARGETS].to_numpy(), rtol=0, atol=1e-6)
    assert predicted_path.with_suffix(".provenance.json").exists()

    # Exercise prediction in a fresh process where TensorBoard cannot be imported.
    script = """
import importlib.abc
import os
import sys

class NoTensorBoard(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'torch.utils.tensorboard' or fullname.split('.')[0] == 'tensorboard':
            raise ModuleNotFoundError('TensorBoard is intentionally unavailable')

sys.meta_path.insert(0, NoTensorBoard())
os.cpu_count = lambda: 2
from rsna_knee_abnormality_detection.predict import predict
predict(sys.argv[1], sys.argv[2], [sys.argv[3]], sys.argv[4], batch_size=2, num_workers=0, device='cpu')
assert not any(name.split('.')[0] == 'tensorboard' for name in sys.modules)
"""
    without_tensorboard = tmp_path / "predictions-without-tensorboard.csv"
    result = subprocess.run([
        sys.executable, "-c", script, experiment["data"]["manifest"], str(studies_csv),
        str(output / "last.pt"), str(without_tensorboard),
    ], capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
    np.testing.assert_array_equal(read_predictions(without_tensorboard)[TARGETS], actual[TARGETS])

    incompatible = {key: resumed[key] for key in ["format_version", "targets", "config", "model"]}
    incompatible["image_preparation"] = {**resumed["image_preparation"], "preprocess_version": "other-version"}
    incompatible_path = tmp_path / "incompatible-model.pt"
    torch.save(incompatible, incompatible_path)
    invalid_output = tmp_path / "incompatible-predictions.csv"
    with pytest.raises(ValueError, match="image preparation version"):
        predict(
            experiment["data"]["manifest"], studies_csv, [incompatible_path], invalid_output,
            batch_size=2, num_workers=0, device="cpu",
        )
    assert not invalid_output.exists()

    changed = deepcopy(experiment)
    changed["training"]["encoder_lr"] *= 2
    with pytest.raises(ValueError, match="settings differ"):
        train(changed, resume=output / "last.pt")
    labels_path = tmp_path / "labels.csv"
    labels_path.write_text(labels_path.read_text() + "\n")
    with pytest.raises(ValueError, match="changed since this checkpoint"):
        train(experiment, resume=output / "last.pt")


def test_training_with_logging_has_identical_weights_and_metrics(experiment, tmp_path):
    enabled = deepcopy(experiment)
    enabled["training"]["output_dir"] = str(tmp_path / "logged")
    disabled = deepcopy(experiment)
    disabled["training"]["output_dir"] = str(tmp_path / "unlogged")
    # Older configuration dictionaries omit the new loss setting. They must
    # behave exactly like an explicit target_weighting: observed configuration.
    disabled["training"].pop("target_weighting")
    disabled["monitoring"]["enabled"] = False
    train(enabled, stop_after_epoch=1)
    train(disabled, stop_after_epoch=1)
    logged = torch.load(tmp_path / "logged" / "last.pt", map_location="cpu", weights_only=True)
    unlogged = torch.load(tmp_path / "unlogged" / "last.pt", map_location="cpu", weights_only=True)
    for name, value in logged["model"].items():
        assert torch.equal(value, unlogged["model"][name]), name
    assert logged["metrics"] == unlogged["metrics"]
    assert logged["history"][0]["train_loss"] == unlogged["history"][0]["train_loss"]
    assert logged["global_step"] == unlogged["global_step"] == 2
    assert logged["scheduler"] == unlogged["scheduler"]
    assert (tmp_path / "logged" / "tensorboard").is_dir()
    assert not (tmp_path / "unlogged" / "tensorboard").exists()
    assert logged["provenance"]["training_objective"] == unlogged["provenance"]["training_objective"]
    assert logged["provenance"]["monitoring"]["checkpoint_selection"] == "full validation after each epoch only"
    assert not (tmp_path / "unlogged" / "monitoring-validation-sample.csv").exists()


def test_future_logging_defaults_and_interval_controls(tmp_path):
    path = tmp_path / "new-run.yaml"
    path.write_text("{}\n")
    config = load_config(path)
    assert config["monitoring"]["log_every_updates"] == 10
    assert config["monitoring"]["validation_every_updates"] == 0
    assert config["monitoring"]["validation_studies"] == 32
    for key in ["validation_every_updates", "validation_studies"]:
        for value in [-1, True, 1.5]:
            invalid = deepcopy(config)
            invalid["monitoring"][key] = value
            with pytest.raises(ValueError, match=key):
                validate_config(invalid)
        disabled = deepcopy(config)
        disabled["monitoring"][key] = 0
        validate_config(disabled)


def test_separate_validation_labels_never_replace_training_targets(experiment, tmp_path):
    original_train, original_val = study_partitions(experiment)
    reference = pd.read_csv(experiment["data"]["labels"])
    reference_path = tmp_path / "evaluation-labels.csv"
    reference.to_csv(reference_path, index=False)
    soft = reference.copy()
    soft.loc[soft[STUDY_ID].isin(original_train[STUDY_ID]), TARGETS] = 0.35
    soft.loc[soft[STUDY_ID].isin(original_val[STUDY_ID]), TARGETS] = 0.65
    soft.to_csv(experiment["data"]["labels"], index=False)
    experiment["data"]["validation_labels"] = str(reference_path)
    training, validation = study_partitions(experiment)
    assert training[STUDY_ID].tolist() == original_train[STUDY_ID].tolist()
    assert (training[TARGETS].to_numpy() == 0.35).all()
    np.testing.assert_equal(validation[TARGETS].to_numpy(), original_val[TARGETS].to_numpy())
    # Invalid references fail before a run can use partial or duplicate labels.
    reference.iloc[:-1].to_csv(reference_path, index=False)
    with pytest.raises(ValueError, match="Validation labels must cover"):
        study_partitions(experiment)


def test_raptor_loss_and_one_cycle_resume_use_training_only(experiment, tmp_path):
    from rsna_knee_abnormality_detection.metrics import training_positive_weights

    experiment["training"].update({"positive_weighting": "raptor", "schedule": "one_cycle"})
    reference = pd.read_csv(experiment["data"]["labels"])
    reference_path = tmp_path / "fixed-evaluation.csv"
    reference.to_csv(reference_path, index=False)
    training, _ = study_partitions(experiment)
    targets = np.tile(np.linspace(0.05, 0.95, len(TARGETS)), (len(training), 1))
    reference.loc[reference[STUDY_ID].isin(training[STUDY_ID]), TARGETS] = targets
    reference.to_csv(experiment["data"]["labels"], index=False)
    experiment["data"]["validation_labels"] = str(reference_path)
    full = deepcopy(experiment)
    full["training"]["output_dir"] = str(tmp_path / "raptor-objective-full")
    train(full)
    train(experiment, stop_after_epoch=1)
    train(experiment, resume=tmp_path / "run" / "last.pt")
    complete = torch.load(tmp_path / "raptor-objective-full" / "last.pt", weights_only=True)
    resumed = torch.load(tmp_path / "run" / "last.pt", weights_only=True)
    assert complete["scheduler"] == resumed["scheduler"]
    for name, value in complete["model"].items():
        assert torch.equal(value, resumed["model"][name]), name
    weights = list(resumed["provenance"]["training_objective"]["positive_term_weights"].values())
    np.testing.assert_allclose(weights, training_positive_weights(targets, "raptor"))
    assert "validation_labels" in resumed["provenance"]["input_sha256"]
    assert resumed["metrics"]["auc_targets"] == len(TARGETS)


def test_fixed_positive_weights_train_and_resume_without_using_other_targets(experiment, tmp_path, monkeypatch):
    import rsna_knee_abnormality_detection.train as training_module
    from rsna_knee_abnormality_detection.metrics import masked_bce_sum, training_positive_weights

    training, validation = study_partitions(experiment)
    reference = pd.read_csv(experiment["data"]["labels"])
    reference_path = tmp_path / "fixed-weight-evaluation.csv"
    reference.to_csv(reference_path, index=False)
    graded = reference.copy()
    targets = np.tile([0.08, 0.68, 0.82, 0.94] * 3, (len(training), 1))
    targets[::2, ::3] = np.nan
    graded.loc[graded[STUDY_ID].isin(training[STUDY_ID]), TARGETS] = targets
    # These values differ from both training targets and the evaluation file.
    # Neither these validation rows nor the expert group may enter the objective.
    graded.loc[graded[STUDY_ID].isin(validation[STUDY_ID]), TARGETS] = 0.01
    graded.loc[graded[STUDY_ID].str.startswith("expert"), TARGETS] = 1.0
    graded.to_csv(experiment["data"]["labels"], index=False)
    experiment["data"]["validation_labels"] = str(reference_path)
    expected = np.arange(len(TARGETS), dtype=float) / 4 + 0.75
    fixed = dict(reversed(list(zip(TARGETS, expected.tolist(), strict=True))))
    experiment["training"].update(positive_weighting="fixed", fixed_positive_weights=fixed, schedule="one_cycle")
    experiment["monitoring"].update(enabled=False, validation_every_updates=0)
    assert not np.allclose(training_positive_weights(targets, "raptor"), expected)
    resolved_calls, loss_calls = [], []

    def resolve(selected_targets, mode, fixed_weights=None):
        np.testing.assert_equal(selected_targets, targets)
        assert mode == "fixed" and fixed_weights == fixed
        result = training_positive_weights(selected_targets, mode, fixed_weights)
        np.testing.assert_array_equal(result, expected)
        resolved_calls.append(result)
        return result

    def checked_loss(logits, batch_targets, mask, target_weights=None, positive_weights=None):
        torch.testing.assert_close(positive_weights, torch.tensor(expected, dtype=torch.float32), rtol=0, atol=0)
        result = masked_bce_sum(logits, batch_targets, mask, target_weights, positive_weights)
        gradient, = torch.autograd.grad(result[0], logits, retain_graph=True)
        assert torch.equal(gradient[~mask], torch.zeros_like(gradient[~mask]))
        loss_calls.append(int((~mask).sum()))
        return result

    monkeypatch.setattr(training_module, "training_positive_weights", resolve)
    monkeypatch.setattr(training_module, "masked_bce_sum", checked_loss)
    full = deepcopy(experiment)
    full["training"]["output_dir"] = str(tmp_path / "fixed-weight-full")
    train(full)
    train(experiment, stop_after_epoch=1)
    train(experiment, resume=tmp_path / "run" / "last.pt")
    complete = torch.load(tmp_path / "fixed-weight-full" / "last.pt", map_location="cpu", weights_only=True)
    resumed = torch.load(tmp_path / "run" / "last.pt", map_location="cpu", weights_only=True)
    assert len(resolved_calls) == 3 and sum(loss_calls) > 0
    assert complete["scheduler"] == resumed["scheduler"]
    assert complete["global_step"] == resumed["global_step"] == 4
    for name, value in complete["model"].items():
        assert torch.equal(value, resumed["model"][name]), name
    for left, right in zip(complete["history"], resumed["history"], strict=True):
        assert left["train_loss"] == right["train_loss"]
        assert left["validation"] == right["validation"]
    objective = resumed["provenance"]["training_objective"]
    assert objective["positive_weighting"] == "fixed"
    assert list(objective["positive_term_weights"]) == TARGETS
    assert objective["positive_term_weights"] == fixed
    assert objective["positive_weight_formula"] is None
    assert "configuration" in objective["positive_weight_source"]
    assert "not recomputed" in objective["positive_weight_source"]
    assert objective["known_training_counts"] == dict(zip(TARGETS, np.isfinite(targets).sum(axis=0).tolist(), strict=True))
    assert resumed["provenance"]["training_ids"] == training[STUDY_ID].tolist()
    assert resumed["provenance"]["validation_ids"] == validation[STUDY_ID].tolist()
    assert not any(uid.startswith("expert") for uid in resumed["provenance"]["training_ids"])
    assert resumed["metrics"]["auc_targets"] == len(TARGETS)
    saved = json.loads((tmp_path / "run" / "provenance.json").read_text())
    assert saved["training_objective"] == objective
    changed = deepcopy(experiment)
    changed["training"]["fixed_positive_weights"]["ACL"] += 1
    monkeypatch.setattr(training_module, "training_positive_weights", training_positive_weights)
    with pytest.raises(ValueError, match="settings differ"):
        train(changed, resume=tmp_path / "run" / "last.pt")


def test_best_epoch_average_is_evaluated_without_replacing_best(experiment, tmp_path):
    experiment["training"].update({"average_top_k": 2, "selection_metric": "macro_auc"})
    train(experiment, stop_after_epoch=1)
    output = tmp_path / "run"
    assert not (output / "averaged-model.pt").exists()
    train(experiment, resume=output / "last.pt")
    averaged = torch.load(output / "averaged-model.pt", map_location="cpu", weights_only=True)
    best = torch.load(output / "best-model.pt", map_location="cpu", weights_only=True)
    history = json.loads((output / "history.json").read_text())
    assert best["metrics"]["macro_auc"] == max(row["validation"]["macro_auc"] for row in history)
    assert len(averaged["checkpoint_average"]["members"]) == 2
    assert averaged["metrics"]["auc_targets"] == len(TARGETS)
    assert "checkpoint_average" not in best
    assert len(read_predictions(output / "validation-averaged.csv")) == 4
    assert json.loads((output / "status.json").read_text())["completed"]


def test_equal_finding_training_uses_only_training_counts_and_resumes_exactly(experiment, tmp_path):
    config = deepcopy(experiment)
    config["training"]["target_weighting"] = "equal_finding"
    config["monitoring"]["enabled"] = False
    # Five studies with batch size two produce both a partial final minibatch
    # and a partial accumulation window on every epoch.
    config["debug"]["max_train_studies"] = 5
    labels = pd.read_csv(config["data"]["labels"])
    labels.loc[labels[STUDY_ID].eq("train-4"), TARGETS] = np.linspace(0, 1, len(TARGETS))
    labels.to_csv(config["data"]["labels"], index=False)
    full_config = deepcopy(config)
    full_config["training"]["output_dir"] = str(tmp_path / "equal-full")
    train(full_config)
    full = torch.load(tmp_path / "equal-full" / "last.pt", map_location="cpu", weights_only=True)
    train(config, stop_after_epoch=1)
    output = tmp_path / "run"
    train(config, resume=output / "last.pt")
    resumed = torch.load(output / "last.pt", map_location="cpu", weights_only=True)
    for name, expected in full["model"].items():
        assert torch.equal(expected, resumed["model"][name]), name
    assert full["scheduler"] == resumed["scheduler"]
    assert full["global_step"] == resumed["global_step"] == 4
    for expected, actual in zip(full["history"], resumed["history"], strict=True):
        assert expected["train_loss"] == actual["train_loss"]
        assert expected["validation"] == actual["validation"]
    objective = resumed["provenance"]["training_objective"]
    expected_counts = np.array([5, 5, 4, 3, 2, 1, 1, 1, 1, 1, 1, 1])
    assert objective["target_weighting"] == "equal_finding"
    assert objective["normalization"] == "studies"
    assert objective["known_training_counts"] == dict(zip(TARGETS, expected_counts.tolist(), strict=True))
    np.testing.assert_allclose(
        [objective["target_weights"][target] for target in TARGETS], 5 / (len(TARGETS) * expected_counts),
    )
    saved_provenance = json.loads((output / "provenance.json").read_text())
    assert saved_provenance["training_objective"] == objective
    changed = deepcopy(config)
    changed["training"]["target_weighting"] = "observed"
    with pytest.raises(ValueError, match="settings differ"):
        train(changed, resume=output / "last.pt")


def test_equal_finding_cannot_borrow_missing_target_support_from_validation(experiment, tmp_path):
    # This fixture has complete validation/expert labels, but several findings
    # have no observed training targets. Those held-out labels cannot fill them.
    experiment["training"]["target_weighting"] = "equal_finding"
    with pytest.raises(ValueError, match="no training labels for"):
        train(experiment)
    assert not (tmp_path / "run").exists()


def test_target_weighting_configuration_defaults_and_invalid_mode(tmp_path):
    path = tmp_path / "legacy.yaml"
    path.write_text("training:\n  epochs: 3\n")
    config = load_config(path)
    assert config["training"]["target_weighting"] == "observed"
    config["training"]["target_weighting"] = "equal_finding"
    validate_config(config)
    config["training"]["target_weighting"] = "per_batch"
    with pytest.raises(ValueError, match="target_weighting"):
        validate_config(config)


def test_partitions_exclude_expert_patient_and_unlabeled_studies(experiment):
    training, validation = study_partitions(experiment)
    assert len(training) == 6 and len(validation) == 4
    assert set(training[STUDY_ID]).isdisjoint(validation[STUDY_ID])
    assert set(training["group_id"]).isdisjoint(validation["group_id"])
    assert not set(training[STUDY_ID]) & {"expert", "expert-patient-repeat", "unlabeled"}


def test_rejects_mismatched_or_duplicate_study_ids(experiment):
    path = experiment["data"]["splits"]
    splits = pd.read_csv(path)
    splits.loc[0, STUDY_ID] = "different-study"
    splits.to_csv(path, index=False)
    with pytest.raises(ValueError, match="exactly the same studies"):
        study_partitions(experiment)
    splits.loc[0, STUDY_ID] = splits.loc[1, STUDY_ID]
    splits.to_csv(path, index=False)
    with pytest.raises(ValueError, match="unique"):
        study_partitions(experiment)


def test_rejects_patient_leakage_between_folds(experiment):
    path = experiment["data"]["splits"]
    splits = pd.read_csv(path)
    splits.loc[splits[STUDY_ID] == "val-0", "group_id"] = "patient-t0"
    splits.to_csv(path, index=False)
    with pytest.raises(ValueError, match="patient group crosses"):
        study_partitions(experiment)


def test_rejects_stale_split_that_reintroduces_expert_patient(experiment):
    path = experiment["data"]["splits"]
    splits = pd.read_csv(path)
    splits.loc[splits["group_id"] == "expert-patient", ["split", "fold"]] = ["train", 1]
    splits.to_csv(path, index=False)
    with pytest.raises(ValueError, match="[Ee]xpert|holdout"):
        study_partitions(experiment)


def test_unlabeled_repeat_can_share_a_training_patients_fold_without_being_fitted(experiment):
    path = experiment["data"]["splits"]
    splits = pd.read_csv(path)
    splits.loc[splits[STUDY_ID] == "unlabeled", ["group_id", "fold"]] = ["patient-t0", 1]
    splits.to_csv(path, index=False)
    training, validation = study_partitions(experiment)
    assert len(training) == 6 and len(validation) == 4
    assert "unlabeled" not in set(training[STUDY_ID]) | set(validation[STUDY_ID])


def test_training_rejects_prepared_image_size_mismatch(experiment):
    experiment["data"]["cache_size"] = 16
    with pytest.raises(ValueError, match="cache image size.*expected size"):
        train(experiment)


def test_rejects_missing_prepared_images(experiment):
    path = experiment["data"]["manifest"]
    manifest = pd.read_csv(path)
    manifest.loc[manifest[STUDY_ID] != "val-0"].to_csv(path, index=False)
    with pytest.raises(ValueError, match="Prepared images missing"):
        study_partitions(experiment)


def test_skipped_early_validation_selects_the_same_way_and_resumes_exactly(experiment, tmp_path):
    """Epochs before validation_start_epoch train identically but write no scores, so they can't be selected."""
    experiment["training"].update({"epochs": 3, "average_top_k": 2, "selection_metric": "macro_auc"})
    experiment["monitoring"].update({"validation_every_updates": 0})
    full = deepcopy(experiment)
    full["training"]["output_dir"] = str(tmp_path / "full")
    train(full)
    skipping = deepcopy(experiment)
    skipping["training"]["validation_start_epoch"] = 2
    train(skipping, stop_after_epoch=1)
    output = tmp_path / "run"
    first = torch.load(output / "last.pt", map_location="cpu", weights_only=True)
    assert first["metrics"] is None and first["history"] == [] and not (output / "history.json").exists()
    assert [row["epoch"] for row in json.loads((output / "training-history.json").read_text())] == [1]
    train(skipping, resume=output / "last.pt")
    reference = torch.load(tmp_path / "full" / "last.pt", map_location="cpu", weights_only=True)
    resumed = torch.load(output / "last.pt", map_location="cpu", weights_only=True)
    for name, expected in reference["model"].items():
        assert torch.equal(expected, resumed["model"][name]), name
    history = json.loads((output / "history.json").read_text())
    full_history = json.loads((tmp_path / "full" / "history.json").read_text())
    assert [row["epoch"] for row in history] == [2, 3]
    assert [row["validation"] for row in history] == [row["validation"] for row in full_history[1:]]
    averaged = torch.load(output / "averaged-model.pt", map_location="cpu", weights_only=True)
    assert sorted(member["epoch"] for member in averaged["checkpoint_average"]["members"]) == [2, 3]
    events = EventAccumulator(str(output / "tensorboard"), size_guidance={"scalars": 0}).Reload()
    assert len(events.Scalars("Loss/train_epoch")) == 3 and len(events.Scalars("Loss/validation")) == 2


def test_channels_last_and_compile_keep_checkpoint_names(experiment, tmp_path, monkeypatch):
    compiled = []
    monkeypatch.setattr(torch.nn.Module, "compile", lambda self, *a, **k: compiled.append(type(self).__name__))
    experiment["training"].update({"channels_last": True, "compile": True})
    train(experiment, stop_after_epoch=1)
    checkpoint = torch.load(tmp_path / "run" / "last.pt", map_location="cpu", weights_only=True)
    assert compiled == ["ResNet"] and not any("_orig_mod" in name for name in checkpoint["model"])
    assert checkpoint["model"]["encoder.conv1.weight"].is_contiguous(memory_format=torch.channels_last)


@pytest.mark.parametrize("settings, message", [
    ({"validation_start_epoch": 0}, "validation_start_epoch"),
    ({"validation_start_epoch": 3}, "validation_start_epoch"),
    ({"validation_start_epoch": 2, "average_top_k": 2, "selection_metric": "macro_auc"}, "validated epochs"),
    ({"channels_last": 1}, "channels_last"),
])
def test_speed_settings_are_checked(experiment, settings, message):
    experiment["training"].update(settings)
    with pytest.raises(ValueError, match=message):
        validate_config(experiment)
