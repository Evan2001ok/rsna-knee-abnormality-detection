"""All-window fine-tune together with the OAI per-epoch draw (Oct 18, 2026):
both options in one run start from the checkpoint, keep BatchNorm frozen, and take the drawn epoch size."""

from copy import deepcopy

import numpy as np
import pandas as pd
import torch
from test_raptor_data import make_raw

from rsna_knee_abnormality_detection.config import DEFAULTS
from rsna_knee_abnormality_detection.constants import STUDY_ID, TARGETS
from rsna_knee_abnormality_detection.raptor_data import prepare_raptor_data
from rsna_knee_abnormality_detection.train import train


def test_all_window_fine_tune_with_the_oai_draw(tmp_path):
    studies = ["s0", "s1", "s2", "s3", "OAI_100_LEFT_V00", "OAI_100_LEFT_V03", "OAI_200_RIGHT_V00", "OAI_300_LEFT_V05"]
    make_raw(tmp_path / "raw", studies=studies)
    manifest = prepare_raptor_data(tmp_path / "raw", tmp_path / "cache", workers=1)
    rng = np.random.default_rng(3)
    pd.DataFrame({STUDY_ID: studies, **{name: rng.random(len(studies)).round() for name in TARGETS},
                  "label_source": "report"}).to_csv(tmp_path / "labels.csv", index=False)
    pd.DataFrame({STUDY_ID: studies, "fold": [0, 0, 1, 1, 5, 5, 5, 5], "split": "train",
                  "group_id": studies}).to_csv(tmp_path / "splits.csv", index=False)
    config = deepcopy(DEFAULTS)
    config["seed"] = 11
    config["data"].update({
        "recipe": "raptor", "manifest": str(manifest), "labels": str(tmp_path / "labels.csv"),
        "splits": str(tmp_path / "splits.csv"), "fold": 0, "cache_size": 336, "image_size": 32,
        "raptor_train_windows": 4, "raptor_eval_windows": 4, "raptor_window_sampling": "stratified",
        "raptor_series_dropout": 0.15})
    config["model"].update({"backbone": "resnet18", "pretrained": False, "pooling": "class_attention",
                            "use_metadata": False, "use_positions": False, "encoder_chunk_size": 8})
    config["training"].update({"output_dir": str(tmp_path / "base"), "epochs": 1, "batch_size": 2,
                               "accumulation_steps": 1, "num_workers": 0, "device": "cpu", "precision": "fp32",
                               "selection_metric": "loss", "deterministic": True})
    train(config)
    start = next(p for p in (tmp_path / "base" / "averaged-model.pt", tmp_path / "base" / "best-model.pt") if p.exists())
    tuned = deepcopy(config)
    tuned["data"]["raptor_train_all_windows"] = True
    tuned["training"].update({"output_dir": str(tmp_path / "tuned"), "epochs": 2, "batch_size": 1, "accumulation_steps": 1,
                              "init_checkpoint": str(start), "encoder_lr": 1e-12, "head_lr": 1e-12,
                              "freeze_batchnorm": True, "oai_epoch_draw": 2})
    train(tuned)
    last = torch.load(tmp_path / "tuned" / "last.pt", weights_only=True)
    # 2 competition training rows + 2 drawn OAI rows per epoch, one update each.
    assert [entry["global_step"] for entry in last["history"]] == [4, 8]
    before = torch.load(start, weights_only=True)["model"]
    last_model = last["model"]
    names = [n for n, v in before.items() if v.is_floating_point() and "running" not in n]
    assert names and all(torch.allclose(before[n], last_model[n], atol=1e-6) for n in names)  # started from the checkpoint
    stats = [n for n in before if "running" in n or "num_batches_tracked" in n]
    assert stats and all(torch.equal(before[n], last_model[n]) for n in stats)  # BatchNorm frozen
