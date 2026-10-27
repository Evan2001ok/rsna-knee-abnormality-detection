"""OAI follow-up visits with a per-epoch draw (Oct 17, 2026): each epoch uses every competition training row once plus
a new draw of OAI exams, taking participants in turn (one OAI participant can have several follow-up visits)."""

from collections import Counter
from copy import deepcopy

import numpy as np
import pandas as pd
import pytest
import torch
from test_raptor_data import make_raw

from rsna_knee_abnormality_detection.config import DEFAULTS, validate_config
from rsna_knee_abnormality_detection.constants import STUDY_ID, TARGETS
from rsna_knee_abnormality_detection.raptor_data import prepare_raptor_data
from rsna_knee_abnormality_detection.train import oai_epoch_indices, train


def ids():
    competition = [f"1.2.{i}" for i in range(5)]
    # 4 participants: 100 has 6 exams, 200 has 2, 300 and 400 one each.
    oai = ([f"OAI_100_{side}_V{v}" for side in ("LEFT", "RIGHT") for v in ("00", "01", "03")]
           + ["OAI_200_LEFT_V00", "OAI_200_LEFT_V01", "OAI_300_RIGHT_V00", "OAI_400_LEFT_V05"])
    return competition + oai


def test_every_competition_row_once_and_the_drawn_oai_rows():
    study_ids = ids()
    for draw in (1, 3, 4, 6, 9):
        chosen = oai_epoch_indices(study_ids, draw, seed=7)
        picked = [study_ids[i] for i in chosen]
        assert len(set(chosen.tolist())) == len(chosen) and chosen.tolist() == sorted(chosen.tolist())
        assert [s for s in picked if not s.startswith("OAI_")] == study_ids[:5]
        oai = [s for s in picked if s.startswith("OAI_")]
        assert len(oai) == draw
        people = Counter(s.split("_")[1] for s in oai)
        # Round-robin: no participant gets a second exam before everyone with exams left has one.
        if draw <= 4:
            assert max(people.values()) == 1
        if draw == 6:
            assert people["300"] == 1 and people["400"] == 1 and people["100"] == 2 and people["200"] == 2


def test_draw_repeats_for_the_same_seed_changes_with_the_epoch_and_takes_all_when_large():
    study_ids = ids()
    assert oai_epoch_indices(study_ids, 4, 5).tolist() == oai_epoch_indices(study_ids, 4, 5).tolist()
    assert len({tuple(oai_epoch_indices(study_ids, 4, s).tolist()) for s in range(20)}) > 1
    assert oai_epoch_indices(study_ids, 10, 1).tolist() == list(range(len(study_ids)))
    assert oai_epoch_indices(study_ids, 99, 1).tolist() == list(range(len(study_ids)))


def test_config_checks_the_draw():
    config = deepcopy(DEFAULTS)
    config["training"]["oai_epoch_draw"] = 2398
    validate_config(config)
    for bad in (-1, 1.5, True):
        config["training"]["oai_epoch_draw"] = bad
        with pytest.raises(ValueError, match="oai_epoch_draw"):
            validate_config(config)


def test_training_steps_follow_the_drawn_epoch_size(tmp_path):
    studies = ["s0", "s1", "s2", "s3", "OAI_100_LEFT_V00", "OAI_100_LEFT_V01", "OAI_200_RIGHT_V00", "OAI_300_LEFT_V00"]
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
        "raptor_train_windows": 4, "raptor_eval_windows": 4, "raptor_window_sampling": "stratified"})
    config["model"].update({"backbone": "resnet18", "pretrained": False, "pooling": "class_attention",
                            "use_metadata": False, "use_positions": False, "encoder_chunk_size": 8})
    config["training"].update({"epochs": 2, "batch_size": 1, "accumulation_steps": 1, "num_workers": 0,
                               "device": "cpu", "precision": "fp32", "selection_metric": "loss", "deterministic": True})
    steps = {}
    for draw in (0, 2):
        run = deepcopy(config)
        run["training"].update(output_dir=str(tmp_path / f"draw{draw}"), oai_epoch_draw=draw)
        train(run)
        history = torch.load(tmp_path / f"draw{draw}" / "last.pt", weights_only=True)["history"]
        steps[draw] = [entry["global_step"] for entry in history]
    # 2 competition + 4 OAI training rows: 6 updates per epoch with every row, 2 + 2 = 4 with a draw of 2.
    assert steps[0] == [6, 12] and steps[2] == [4, 8]
