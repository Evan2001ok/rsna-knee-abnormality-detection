import importlib
import json
import random
from copy import deepcopy

import numpy as np
import pandas as pd
import pytest
import torch
from test_training import experiment as experiment
from test_training import two_threads as two_threads

from rsna_knee_abnormality_detection.constants import STUDY_ID, TARGETS
from rsna_knee_abnormality_detection.utils import rng_state, save_checkpoint

training_module = importlib.import_module("rsna_knee_abnormality_detection.train")


def test_stop_after_last_checkpoint_recovers_best_and_resumes_identically(experiment, tmp_path, monkeypatch):
    experiment["monitoring"]["enabled"] = False
    experiment["training"].update(average_top_k=2, selection_metric="macro_auc")
    complete_config = deepcopy(experiment)
    complete_config["training"]["output_dir"] = str(tmp_path / "complete")
    training_module.train(complete_config)
    original_save = training_module.save_checkpoint

    def interrupt_best(path, saved):
        if path.name == "best-model.pt":
            raise RuntimeError("simulated interruption after last.pt")
        original_save(path, saved)

    monkeypatch.setattr(training_module, "save_checkpoint", interrupt_best)
    with pytest.raises(RuntimeError, match="simulated interruption"):
        training_module.train(experiment)
    output = tmp_path / "run"
    assert (output / "last.pt").exists()
    assert not (output / "best-model.pt").exists()
    assert not (output / "history.json").exists()
    monkeypatch.setattr(training_module, "save_checkpoint", original_save)
    # Recover the just-completed epoch without advancing to the next epoch yet.
    training_module.train(experiment, resume=output / "last.pt", stop_after_epoch=1)
    restored_history = json.loads((output / "history.json").read_text())
    assert [row["epoch"] for row in restored_history] == [1]
    assert (output / "best-artifacts.json").exists()
    training_module.train(experiment, resume=output / "last.pt")
    complete = torch.load(tmp_path / "complete" / "last.pt", weights_only=True)
    resumed = torch.load(output / "last.pt", weights_only=True)
    assert complete["scheduler"] == resumed["scheduler"]
    assert complete["metrics"] == resumed["metrics"]
    assert complete["global_step"] == resumed["global_step"]
    for name, value in complete["model"].items():
        assert torch.equal(value, resumed["model"][name]), name
    expected_best = torch.load(tmp_path / "complete" / "best-model.pt", weights_only=True)
    actual_best = torch.load(output / "best-model.pt", weights_only=True)
    assert expected_best["epoch"] == actual_best["epoch"]
    assert expected_best["metrics"] == actual_best["metrics"]
    for name, value in expected_best["model"].items():
        assert torch.equal(value, actual_best["model"][name]), name


def recovered_checkpoint(model):
    common = {
        "format_version": 1, "targets": TARGETS, "config": {"seed": 42},
        "image_preparation": {"preprocess_version": "test"},
        "provenance": {"code_sha256": "code", "input_sha256": {"labels": "labels"}},
    }
    best = {**common, "epoch": 1, "metrics": {"macro_auc": 0.9, "loss": 0.1},
            "model": {name: torch.ones_like(value) for name, value in model.state_dict().items()}}
    last = {**common, "epoch": 2, "metrics": {"macro_auc": 0.8, "loss": 0.2},
            "model": deepcopy(model.state_dict()), "best_score": 0.9}
    last["history"] = [{"epoch": ck["epoch"], "validation": ck["metrics"]} for ck in [best, last]]
    return best, last


def test_older_best_candidate_recovery_restores_model_and_rng_and_then_skips_work(tmp_path, monkeypatch):
    model = torch.nn.Linear(1, 1)
    model.train()
    best, last = recovered_checkpoint(model)
    save_checkpoint(tmp_path / "candidate-epoch-001.pt", best)
    before = rng_state()
    calls = []
    monkeypatch.setattr(training_module, "make_loader", lambda *args, **kwargs: None)

    def predict(current, *args, **kwargs):
        calls.append(True)
        assert torch.equal(current.weight, best["model"]["weight"])
        current.eval()
        random.random()
        np.random.random()
        torch.rand(2)
        frame = pd.DataFrame([["v", *([0.5] * len(TARGETS))]], columns=[STUDY_ID, *TARGETS])
        # A replay can differ slightly under mixed precision. The receipt must
        # let a subsequent resume recognize the regenerated artifact set.
        return frame, {"macro_auc": 0.89999, "loss": 0.10001}

    monkeypatch.setattr(training_module, "predict_loader", predict)
    settings = {"selection_metric": "macro_auc", "precision": "fp32"}
    for _ in range(2):
        training_module._recover_best_artifacts(tmp_path, last, model, None, ["v"], torch.device("cpu"), settings, 42)
    assert calls == [True]
    assert model.training
    for name, value in model.state_dict().items():
        assert torch.equal(value, last["model"][name])
    after = rng_state()
    assert before["python"] == after["python"]
    assert before["numpy"] == after["numpy"]
    assert torch.equal(before["torch"], after["torch"])
    assert all(torch.equal(one, two) for one, two in zip(before["cuda"], after["cuda"], strict=True))
    saved = torch.load(tmp_path / "best-model.pt", weights_only=True)
    assert saved["metrics"] == best["metrics"]
    assert json.loads((tmp_path / "metrics-best.json").read_text())["macro_auc"] == 0.89999
    # A partial or stale prediction file is also repaired even when weights agree.
    (tmp_path / "validation-best.csv").unlink()
    training_module._recover_best_artifacts(tmp_path, last, model, None, ["v"], torch.device("cpu"), settings, 42)
    assert len(calls) == 2


def test_missing_older_best_checkpoint_fails_clearly(tmp_path):
    model = torch.nn.Linear(1, 1)
    _, last = recovered_checkpoint(model)
    settings = {"selection_metric": "macro_auc", "precision": "fp32"}
    with pytest.raises(ValueError, match="Cannot recover best epoch 1"):
        training_module._recover_best_artifacts(tmp_path, last, model, None, ["v"], torch.device("cpu"), settings, 42)
