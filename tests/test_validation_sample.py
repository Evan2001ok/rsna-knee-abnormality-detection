"""Monitoring samples and temporary validation must not change training state."""

import random
from contextlib import nullcontext
from copy import deepcopy

import numpy as np
import pandas as pd
import pytest
import torch
from torch import nn

from rsna_knee_abnormality_detection.constants import STUDY_ID, TARGETS
from rsna_knee_abnormality_detection.utils import restore_rng, rng_state
from rsna_knee_abnormality_detection.validation_sample import preserve_training_state, select_validation_sample


def assert_same_rng(first, second):
    assert first["python"] == second["python"]
    assert first["numpy"] == second["numpy"]
    assert torch.equal(first["torch"], second["torch"])
    assert len(first["cuda"]) == len(second["cuda"])
    assert all(torch.equal(a, b) for a, b in zip(first["cuda"], second["cuda"], strict=True))


@pytest.fixture(autouse=True)
def cpu_only_and_preserve_rng(monkeypatch):
    # Do not initialize a CUDA context in this separate test process; the GPU may be busy with a training run.
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    state = rng_state()
    yield
    restore_rng(state)


def validation_rows():
    values = np.zeros((100, len(TARGETS)))
    values[:, 6:] = 1
    # Six findings have a single positive, six have a single negative, and each
    # rare case occurs in a different study. A uniform sample often misses them.
    for column in range(len(TARGETS)):
        values[column, column] = 1 if column < 6 else 0
    values[90:, :2] = np.nan
    frame = pd.DataFrame(values, columns=TARGETS)
    frame.insert(0, STUDY_ID, [f"validation-{i:04d}" for i in range(len(frame))])
    frame["extra_column"] = [f"unchanged-{i}" for i in range(len(frame))]
    return frame


def test_sample_is_deterministic_independent_of_row_order_and_covers_rare_classes():
    validation = validation_rows()
    first = select_validation_sample(validation, 32, 17)
    shuffled = validation.sample(frac=1, random_state=29).reset_index(drop=True)
    second = select_validation_sample(shuffled, 32, 17)
    pd.testing.assert_frame_equal(first, second)
    assert len(first) == 32 and first[STUDY_ID].is_unique
    assert first[STUDY_ID].tolist() == sorted(first[STUDY_ID])
    assert set(first[STUDY_ID]) <= set(validation[STUDY_ID])
    assert set(validation[STUDY_ID].iloc[:12]) <= set(first[STUDY_ID])
    for target in TARGETS:
        assert (first[target] == 0).any() and (first[target] == 1).any()
    expected = validation.set_index(STUDY_ID).loc[first[STUDY_ID]].reset_index()
    pd.testing.assert_frame_equal(first, expected)
    assert set(select_validation_sample(validation, 32, 18)[STUDY_ID]) != set(first[STUDY_ID])


def test_binary_coverage_does_not_threshold_soft_targets():
    validation = validation_rows().iloc[:4].copy()
    validation.loc[:, TARGETS] = np.nan
    validation["ACL"] = [0.25, 0.75, 0, 1]
    sample = select_validation_sample(validation, 2, 1)
    assert sample[STUDY_ID].tolist() == validation[STUDY_ID].iloc[2:].tolist()
    assert sample["ACL"].tolist() == [0, 1]


def test_disjoint_binary_classes_can_all_be_covered_with_24_studies():
    validation = validation_rows()
    validation.loc[:, TARGETS] = np.nan
    for column, target in enumerate(TARGETS):
        validation.loc[2 * column, target] = 0
        validation.loc[2 * column + 1, target] = 1
    sample = select_validation_sample(validation, 24, 9)
    assert sample[STUDY_ID].tolist() == validation[STUDY_ID].iloc[:24].tolist()
    assert all(sample[target].dropna().tolist() == [0, 1] for target in TARGETS)


@pytest.mark.parametrize("limit", [1, 3, np.int64(7), 100, 101, 0])
def test_limit_and_full_validation_behavior(limit):
    validation = validation_rows().iloc[::-1]
    sample = select_validation_sample(validation, limit, 5)
    assert len(sample) == (len(validation) if limit == 0 else min(limit, len(validation)))
    assert sample[STUDY_ID].is_unique
    if limit == 0 or limit >= len(validation):
        pd.testing.assert_frame_equal(sample, validation.sort_values(STUDY_ID).reset_index(drop=True))


def test_empty_validation_and_random_fill_without_binary_labels():
    validation = validation_rows()
    empty = validation.iloc[:0]
    pd.testing.assert_frame_equal(select_validation_sample(empty, 32, 1), empty.reset_index(drop=True))
    validation.loc[:, TARGETS] = 0.33
    sample = select_validation_sample(validation, 8, 31)
    assert len(sample) == 8
    assert sample[TARGETS].eq(0.33).all().all()
    pd.testing.assert_frame_equal(sample, select_validation_sample(validation.iloc[::-1], 8, 31))


def test_selection_changes_neither_input_nor_global_random_state():
    validation = validation_rows()
    before_frame = validation.copy(deep=True)
    before_rng = rng_state()
    sample = select_validation_sample(validation, 32, 61)
    pd.testing.assert_frame_equal(validation, before_frame)
    assert_same_rng(rng_state(), before_rng)
    sample.loc[0, "ACL"] = 0.123
    sample.loc[0, "extra_column"] = "a changed returned row"
    pd.testing.assert_frame_equal(validation, before_frame)


@pytest.mark.parametrize("invalid", [-1, 1.0, True, None, "32"])
def test_invalid_limits_are_rejected(invalid):
    with pytest.raises(ValueError, match="max_studies must be a nonnegative integer"):
        select_validation_sample(validation_rows(), invalid, 1)


@pytest.mark.parametrize("invalid", [-1, 1.0, True, None])
def test_invalid_seeds_are_rejected(invalid):
    with pytest.raises(ValueError, match="seed must be a nonnegative integer"):
        select_validation_sample(validation_rows(), 32, invalid)


@pytest.mark.parametrize("ids,match", [
    (["same", "same"], "must be unique"),
    (["valid", None], "must not be blank"),
    (["valid", ""], "must not be blank"),
    (["valid", " padded"], "surrounding whitespace"),
])
def test_ambiguous_study_ids_are_rejected(ids, match):
    validation = validation_rows().iloc[:2].copy()
    validation[STUDY_ID] = ids
    with pytest.raises(ValueError, match=match):
        select_validation_sample(validation, 32, 1)


def test_missing_or_duplicate_columns_are_rejected():
    validation = validation_rows()
    with pytest.raises(ValueError, match="missing required columns"):
        select_validation_sample(validation.drop(columns=["ACL"]), 32, 1)
    duplicate = pd.concat([validation, validation[["ACL"]]], axis=1)
    with pytest.raises(ValueError, match="columns must be unique"):
        select_validation_sample(duplicate, 32, 1)


@pytest.mark.parametrize("failed", [False, True])
@pytest.mark.parametrize("mixed_modes", [False, True])
def test_validation_restores_rng_and_module_modes_without_bn_or_dropout_updates(failed, mixed_modes):
    model = nn.Sequential(nn.Linear(4, 4), nn.BatchNorm1d(4), nn.Dropout(0.5),
                          nn.Sequential(nn.Linear(4, 2), nn.Dropout(0.5)))
    model.train()
    if mixed_modes:
        model[1].eval()
        model[3][1].eval()
    control = deepcopy(model)
    inputs = torch.arange(24, dtype=torch.float32).reshape(6, 4) / 7
    modes = [module.training for module in model.modules()]
    bn_mean, bn_variance = model[1].running_mean.clone(), model[1].running_var.clone()
    bn_batches = model[1].num_batches_tracked.clone()
    parameter_pointers = [parameter.data_ptr() for parameter in model.parameters()]
    before_rng = rng_state()

    with pytest.raises(RuntimeError, match="failed validation") if failed else nullcontext():
        with preserve_training_state(model) as evaluating, torch.inference_mode():
            assert evaluating is model
            assert not any(module.training for module in model.modules())
            first = evaluating(inputs)
            second = evaluating(inputs)
            assert torch.equal(first, second)  # Dropout is disabled.
            random.random()
            np.random.random(5)
            torch.rand(5)
            if failed:
                raise RuntimeError("failed validation")

    assert [module.training for module in model.modules()] == modes
    assert_same_rng(rng_state(), before_rng)
    assert torch.equal(model[1].running_mean, bn_mean)
    assert torch.equal(model[1].running_var, bn_variance)
    assert torch.equal(model[1].num_batches_tracked, bn_batches)
    assert [parameter.data_ptr() for parameter in model.parameters()] == parameter_pointers
    with torch.no_grad():
        after_monitoring = model(inputs)
        restore_rng(before_rng)
        without_monitoring = control(inputs)
    # The next training forward, including its dropout masks and BN updates,
    # must be identical to an otherwise equal run without the validation call.
    torch.testing.assert_close(after_monitoring, without_monitoring, rtol=0, atol=0)
    for key, value in model.state_dict().items():
        assert torch.equal(value, control.state_dict()[key])


def test_state_is_restored_even_if_entering_eval_raises():
    class FailedEval(nn.Sequential):
        def train(self, mode=True):
            super().train(mode)
            if not mode:
                random.random()
                torch.rand(1)
                raise RuntimeError("eval failed")
            return self

    model = FailedEval(nn.Linear(2, 2), nn.Dropout(0.5))
    model[0].eval()
    modes, before = [module.training for module in model.modules()], rng_state()
    with pytest.raises(RuntimeError, match="eval failed"):
        with preserve_training_state(model):
            raise AssertionError("The body must not run when eval fails")
    assert [module.training for module in model.modules()] == modes
    assert_same_rng(rng_state(), before)
