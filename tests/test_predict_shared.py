"""Several models read in one pass over the data give the same blend as separate passes; T4s use fp16."""

import numpy as np
import pytest
import torch
from test_augmentation import LeftRightModel

from rsna_knee_abnormality_detection.constants import TARGETS
from rsna_knee_abnormality_detection.predict import (
    ProbabilityAverage,
    blend_weights_by_finding,
    inference_precision,
    same_image_inputs,
)
from rsna_knee_abnormality_detection.train import predict_loader


class Brightness(torch.nn.Module):
    """Scores overall brightness, which mirroring leaves unchanged."""

    def __init__(self, scale):
        super().__init__()
        self.scale = scale

    def forward(self, images, series_mask, metadata, positions):
        return (self.scale * (images.mean(dim=(1, 2, 3, 4, 5)) - 0.3)).unsqueeze(1).repeat(1, len(TARGETS))


def batch():
    images = torch.zeros(2, 1, 3, 3, 8, 8)
    images[0, ..., :4] = 1.0
    images[1, ..., :2] = 0.7
    return {"images": images, "series_mask": torch.ones(2, 1, dtype=torch.bool), "metadata": torch.zeros(2, 1, 5),
            "positions": torch.zeros(2, 1, 3), "window_planes": torch.ones(2, 1, 3, dtype=torch.long),
            "targets": torch.zeros(2, len(TARGETS)), "target_mask": torch.zeros(2, len(TARGETS), dtype=torch.bool),
            "study_id": ["a", "b"]}


def test_one_shared_pass_equals_the_weighted_blend_of_separate_passes():
    models, weights = [LeftRightModel(), Brightness(4.0), Brightness(-2.0)], [0.5, 0.3, 0.2]
    for mirror in (False, True):
        separate = sum(weight * predict_loader(model, [batch()], torch.device("cpu"), "fp32", test_time_mirror=mirror)[0][TARGETS]
                       .to_numpy() for model, weight in zip(models, weights))
        shared, _ = predict_loader(ProbabilityAverage(models, weights), [batch()], torch.device("cpu"), "fp32",
                                   test_time_mirror=mirror)
        np.testing.assert_allclose(shared[TARGETS].to_numpy(), separate, rtol=0, atol=1e-6)


def test_gpus_without_native_bf16_use_fp16_and_others_keep_their_setting(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda *args, **kwargs: True)  # PyTorch counts emulation
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device=None: (7, 5))  # Tesla T4
    assert inference_precision(torch.device("cuda"), "bf16") == "fp16"
    assert inference_precision(torch.device("cuda"), "fp32") == "fp32"
    assert inference_precision(torch.device("cpu"), "bf16") == "bf16"
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device=None: (8, 6))  # RTX 3090
    assert inference_precision(torch.device("cuda"), "bf16") == "bf16"


def test_models_share_a_pass_only_when_their_windows_are_identical():
    base = {"data": {"recipe": "raptor", "raptor_eval_windows": 112}, "seed": 42}
    assert same_image_inputs([base, {**base}])
    assert same_image_inputs([base, {**base, "data": {**base["data"], "fold": 3, "labels": "other.csv"}}])
    assert not same_image_inputs([base, {**base, "seed": 7}])
    assert not same_image_inputs([base, {**base, "data": {"recipe": "raptor", "raptor_eval_windows": 56}}])


def test_models_with_different_raptor_caches_or_views_never_share_a_pass():
    base = {"data": {"recipe": "raptor", "raptor_eval_windows": 112}, "seed": 42}
    resized = {**base, "data": {**base["data"], "raptor_view": "resize"}}
    cropped = {**base, "data": {**base["data"], "raptor_view": "center_crop"}}
    larger = {**cropped, "data": {**cropped["data"], "cache_size": 448, "raptor_variant": "raptor-dense-distinct-448-v1"}}
    assert same_image_inputs([base, resized])  # older checkpoints without the key always resized
    assert not same_image_inputs([base, cropped])
    assert not same_image_inputs([cropped, larger])


def test_per_finding_weights_blend_each_finding_separately_in_one_shared_pass():
    models = [LeftRightModel(), Brightness(4.0), Brightness(-2.0)]
    weights = np.ones((3, len(TARGETS)))
    weights[:, 0] = [0.0, 1.0, 3.0]  # first finding: only the brightness models
    weights[:, 1] = [2.0, 0.0, 0.0]  # second finding: only the first model
    separate = [predict_loader(model, [batch()], torch.device("cpu"), "fp32")[0][TARGETS].to_numpy() for model in models]
    expected = sum(w * p for w, p in zip(blend_weights_by_finding(weights, 3), separate))
    shared, _ = predict_loader(ProbabilityAverage(models, blend_weights_by_finding(weights, 3)), [batch()],
                               torch.device("cpu"), "fp32")
    np.testing.assert_allclose(shared[TARGETS].to_numpy(), expected, rtol=0, atol=1e-6)
    np.testing.assert_allclose(shared[TARGETS[1]].to_numpy(), separate[0][:, 1], rtol=0, atol=1e-6)


def test_blend_weights_are_normalised_per_finding_and_checked():
    np.testing.assert_allclose(blend_weights_by_finding(None, 4), np.full((4, len(TARGETS)), 0.25))
    np.testing.assert_allclose(blend_weights_by_finding([1, 3], 2)[:, 5], [0.25, 0.75])
    by_finding = np.ones((2, len(TARGETS)))
    by_finding[:, 2] = [1, 0]
    assert blend_weights_by_finding(by_finding, 2)[:, 2].tolist() == [1.0, 0.0]
    zero_column = by_finding.copy()
    zero_column[:, 3] = 0
    for bad in ([1.0], [1.0, -1.0], [1.0, float("nan")], np.ones((2, 3)), zero_column):
        with pytest.raises(ValueError):
            blend_weights_by_finding(bad, 2)
