"""The dense Raptor cache keeps distinct source slices and leaves unused capacity empty."""

import json
from copy import deepcopy

import cv2
import numpy as np
import pandas as pd
import pytest
import torch
from test_raptor_data import make_raw

from rsna_knee_abnormality_detection.config import DEFAULTS, validate_config
from rsna_knee_abnormality_detection.constants import RAPTOR_VARIANT_NAMES, SERIES_ID, STUDY_ID, WINDOW_FEATURES
from rsna_knee_abnormality_detection.dataset import dataset_preprocess_version, validate_dataset_manifest
from rsna_knee_abnormality_detection.raptor_data import (
    RAPTOR_DENSE_160MM_384_VERSION,
    RAPTOR_DENSE_200MM_384_VERSION,
    RAPTOR_DENSE_384_VERSION,
    RAPTOR_DENSE_448_VERSION,
    RAPTOR_DENSE_SLOTS,
    RAPTOR_DENSE_VERSION,
    RAPTOR_FINGERPRINT,
    RAPTOR_PREPROCESS_VERSION,
    RAPTOR_VARIANTS,
    RaptorDataset,
    _slice_picks,
    prepare_raptor_data,
    validate_raptor_manifest,
)


def test_variant_registry_keeps_the_scored_contract_unchanged():
    assert RAPTOR_VARIANTS[RAPTOR_PREPROCESS_VERSION]["fingerprint"] == RAPTOR_FINGERPRINT
    assert RAPTOR_VARIANTS[RAPTOR_PREPROCESS_VERSION]["depth"] == 64
    assert RAPTOR_VARIANTS[RAPTOR_DENSE_VERSION]["depth"] == 144
    assert sorted(RAPTOR_VARIANTS) == sorted(RAPTOR_VARIANT_NAMES)


@pytest.mark.parametrize("depth,count", [(30, 32), (100, 32), (33, 32), (1, 24), (320, 24)])
def test_distinct_picks_cover_the_span_without_repeats(depth, count):
    picks = _slice_picks(depth, count, distinct=True)
    lo, hi = int(depth * .06), max(int(depth * .94) - 1, int(depth * .06))
    assert len(picks) == min(count, hi - lo + 1)
    assert (np.diff(picks) > 0).all() and picks[0] == lo and picks[-1] == hi
    np.testing.assert_array_equal(_slice_picks(depth, count), np.linspace(lo, hi, count).round().astype(int)
                                  if depth > 1 else np.zeros(count, dtype=int))


def test_dense_cache_fills_distinct_slices_and_windows_stay_inside_them(tmp_path):
    make_raw(tmp_path / "raw")
    manifest = prepare_raptor_data(tmp_path / "raw", tmp_path / "cache", workers=1, version=RAPTOR_DENSE_VERSION)
    metadata = json.loads((tmp_path / "cache/studies/study.json").read_text())
    volume = np.load(tmp_path / "cache/studies/study.npy")
    assert volume.shape == (144, 336, 336) and metadata["preprocess_version"] == RAPTOR_DENSE_VERSION
    valid = np.asarray(metadata["valid_slices"])
    offset = 0
    for slot, (plane, _, count) in zip(metadata["slots"], RAPTOR_DENSE_SLOTS, strict=True):
        # Six synthetic source slices: the 6-94% span is indices 0-4, kept once each.
        assert slot["plane"] == plane and slot["selected_source_indices"] == [0, 1, 2, 3, 4]
        assert valid[offset:offset + 5].all() and not valid[offset + 5:offset + count].any()
        assert not volume[offset + 5:offset + count].any()
        offset += count
    assert len(metadata["allowed_centers"]) == 5 * 3
    assert validate_raptor_manifest(manifest, version=RAPTOR_DENSE_VERSION)["valid_windows"] == 15
    with pytest.raises(ValueError):
        validate_raptor_manifest(manifest)
    config = {"recipe": "raptor", "cache_size": 336, "raptor_variant": RAPTOR_DENSE_VERSION}
    assert dataset_preprocess_version(config) == RAPTOR_DENSE_VERSION
    assert validate_dataset_manifest(manifest, config)["slices"] == 144


def test_dense_dataset_samples_real_windows_with_descriptors(tmp_path):
    make_raw(tmp_path / "raw")
    manifest = prepare_raptor_data(tmp_path / "raw", tmp_path / "cache", workers=1, version=RAPTOR_DENSE_VERSION)
    config = {"image_size": 32, "raptor_variant": RAPTOR_DENSE_VERSION, "raptor_train_windows": 6,
              "raptor_window_sampling": "stratified"}
    studies = pd.DataFrame({STUDY_ID: ["study"]})
    training = RaptorDataset(manifest, studies, config, training=True)[0]
    evaluation = RaptorDataset(manifest, studies, config, training=False)[0]
    assert training["images"].shape == (1, 6, 3, 32, 32) and evaluation["images"].shape == (1, 42, 3, 32, 32)
    assert (training["images"].sum(dim=(-1, -2, -3)) > 0).all()
    assert training["window_features"].shape == (1, 6, len(WINDOW_FEATURES))
    with pytest.raises(ValueError):
        RaptorDataset(manifest, studies, {"image_size": 32}, training=False)


def test_unknown_cache_variant_is_rejected():
    config = deepcopy(DEFAULTS)
    config["data"].update({"recipe": "raptor", "cache_size": 336, "raptor_variant": "raptor-imaginary"})
    with pytest.raises(ValueError, match="raptor_variant"):
        validate_config(config)


def test_scored_cache_still_validates_under_the_default_variant(tmp_path):
    make_raw(tmp_path / "raw")
    manifest = prepare_raptor_data(tmp_path / "raw", tmp_path / "cache", workers=1)
    metadata = json.loads((tmp_path / "cache/studies/study.json").read_text())
    assert metadata["preprocess_version"] == RAPTOR_PREPROCESS_VERSION
    assert all(len(slot["selected_source_indices"]) == slot["count"] for slot in metadata["slots"] if slot[SERIES_ID])
    assert validate_raptor_manifest(manifest)["valid_windows"] == 54


def test_existing_cache_fingerprints_are_unchanged():
    # Prepared caches record these fingerprints; changing them would invalidate every existing cache.
    assert RAPTOR_VARIANTS[RAPTOR_PREPROCESS_VERSION]["fingerprint"] == (
        "d5e19faa558c541f0b0d9a13f896f657d6bc704ab1a7c647b8f1bc26445ed1b1")
    assert RAPTOR_VARIANTS[RAPTOR_DENSE_VERSION]["fingerprint"] == (
        "88ca747a0a4f716b34f2ab6179dd456e7752d8592d6fc17488164c4c834e7db7")
    assert RAPTOR_VARIANTS[RAPTOR_PREPROCESS_VERSION]["size"] == RAPTOR_VARIANTS[RAPTOR_DENSE_VERSION]["size"] == 336
    larger = RAPTOR_VARIANTS[RAPTOR_DENSE_448_VERSION]
    assert larger["size"] == 448 and larger["depth"] == 144 and larger["distinct"]
    assert larger["contract"] == {**RAPTOR_VARIANTS[RAPTOR_DENSE_VERSION]["contract"],
                                  "version": RAPTOR_DENSE_448_VERSION, "cache_size": 448}


def test_448_cache_keeps_the_same_crop_at_more_pixels(tmp_path):
    make_raw(tmp_path / "raw")
    manifest = prepare_raptor_data(tmp_path / "raw", tmp_path / "large", workers=1, version=RAPTOR_DENSE_448_VERSION)
    prepare_raptor_data(tmp_path / "raw", tmp_path / "small", workers=1, version=RAPTOR_DENSE_VERSION)
    large, small = np.load(tmp_path / "large/studies/study.npy"), np.load(tmp_path / "small/studies/study.npy")
    assert large.shape == (144, 448, 448) and large.dtype == np.uint8
    result = validate_raptor_manifest(manifest, expected_size=448, version=RAPTOR_DENSE_448_VERSION, verify_checksums=True)
    assert result["image_size"] == 448 and result["valid_windows"] == 15
    assert pd.read_csv(manifest)["image_size"].tolist() == [448]
    with pytest.raises(ValueError, match="cache size 448"):
        validate_raptor_manifest(manifest, expected_size=336, version=RAPTOR_DENSE_448_VERSION)
    with pytest.raises(ValueError):
        validate_raptor_manifest(manifest, version=RAPTOR_DENSE_VERSION)
    config = {"recipe": "raptor", "cache_size": 448, "raptor_variant": RAPTOR_DENSE_448_VERSION}
    assert validate_dataset_manifest(manifest, config)["slices"] == 144
    # Same 140 mm box: shrinking the 448 px slices back to 336 px nearly reproduces the 336 px cache.
    for index in np.flatnonzero(small.reshape(144, -1).any(axis=1)):
        shrunk = cv2.resize(large[index], (336, 336), interpolation=cv2.INTER_AREA).astype(float)
        assert np.abs(shrunk - small[index]).mean() < 3


def test_160mm_cache_widens_the_box_at_384_px(tmp_path):
    variant = RAPTOR_VARIANTS[RAPTOR_DENSE_160MM_384_VERSION]
    assert variant["size"] == 384 and variant["depth"] == 144 and variant["distinct"]
    assert variant["contract"] == {**RAPTOR_VARIANTS[RAPTOR_DENSE_VERSION]["contract"], "cache_size": 384,
                                   "version": RAPTOR_DENSE_160MM_384_VERSION, "crop_mm": 160.0}
    make_raw(tmp_path / "raw")  # 16 x 20 pixels of 10 mm: 140 mm keeps 14 pixels, 160 mm keeps 16
    manifest = prepare_raptor_data(tmp_path / "raw", tmp_path / "wide", workers=1, version=RAPTOR_DENSE_160MM_384_VERSION)
    prepare_raptor_data(tmp_path / "raw", tmp_path / "native", workers=1, version=RAPTOR_DENSE_384_VERSION)
    wide, native = (json.loads((tmp_path / f"{name}/studies/study.json").read_text()) for name in ("wide", "native"))
    assert {crop["size"] for slot in wide["slots"] for crop in slot["crops"]} == {16}
    assert {crop["size"] for slot in native["slots"] for crop in slot["crops"]} == {14}
    assert np.load(tmp_path / "wide/studies/study.npy").shape == (144, 384, 384)
    assert not np.array_equal(np.load(tmp_path / "wide/studies/study.npy"), np.load(tmp_path / "native/studies/study.npy"))
    result = validate_raptor_manifest(manifest, expected_size=384, version=RAPTOR_DENSE_160MM_384_VERSION, verify_checksums=True)
    assert result["image_size"] == 384
    with pytest.raises(ValueError):
        validate_raptor_manifest(manifest, version=RAPTOR_DENSE_384_VERSION)


def test_200mm_cache_never_crops_beyond_the_image(tmp_path):
    variant = RAPTOR_VARIANTS[RAPTOR_DENSE_200MM_384_VERSION]
    assert variant["size"] == 384 and variant["contract"]["crop_mm"] == 200.0
    assert variant["contract"] == {**RAPTOR_VARIANTS[RAPTOR_DENSE_160MM_384_VERSION]["contract"],
                                   "version": RAPTOR_DENSE_200MM_384_VERSION, "crop_mm": 200.0}
    make_raw(tmp_path / "raw")  # 16 x 20 pixels of 10 mm: 200 mm is wider than the image, so the whole 16 x 16 square
    prepare_raptor_data(tmp_path / "raw", tmp_path / "whole", workers=1, version=RAPTOR_DENSE_200MM_384_VERSION)
    whole = json.loads((tmp_path / "whole/studies/study.json").read_text())
    assert {crop["size"] for slot in whole["slots"] for crop in slot["crops"]} == {16}


def test_384_cache_keeps_the_same_crop_at_the_model_size(tmp_path):
    variant = RAPTOR_VARIANTS[RAPTOR_DENSE_384_VERSION]
    assert variant["size"] == 384 and variant["depth"] == 144 and variant["distinct"]
    assert variant["contract"] == {**RAPTOR_VARIANTS[RAPTOR_DENSE_VERSION]["contract"],
                                   "version": RAPTOR_DENSE_384_VERSION, "cache_size": 384}
    make_raw(tmp_path / "raw")
    manifest = prepare_raptor_data(tmp_path / "raw", tmp_path / "native", workers=1, version=RAPTOR_DENSE_384_VERSION)
    prepare_raptor_data(tmp_path / "raw", tmp_path / "small", workers=1, version=RAPTOR_DENSE_VERSION)
    native, small = np.load(tmp_path / "native/studies/study.npy"), np.load(tmp_path / "small/studies/study.npy")
    assert native.shape == (144, 384, 384) and native.dtype == np.uint8
    result = validate_raptor_manifest(manifest, expected_size=384, version=RAPTOR_DENSE_384_VERSION, verify_checksums=True)
    assert result["image_size"] == 384 and result["valid_windows"] == 15
    config = {"recipe": "raptor", "cache_size": 384, "raptor_variant": RAPTOR_DENSE_384_VERSION}
    assert validate_dataset_manifest(manifest, config)["slices"] == 144
    # Same 140 mm box: shrinking the 384 px slices back to 336 px nearly reproduces the 336 px cache.
    for index in np.flatnonzero(small.reshape(144, -1).any(axis=1)):
        shrunk = cv2.resize(native[index], (336, 336), interpolation=cv2.INTER_AREA).astype(float)
        assert np.abs(shrunk - small[index]).mean() < 3


def test_center_crop_view_returns_the_central_cached_pixels_unchanged(tmp_path):
    make_raw(tmp_path / "raw")
    manifest = prepare_raptor_data(tmp_path / "raw", tmp_path / "cache", workers=1, version=RAPTOR_DENSE_448_VERSION)
    volume = np.load(tmp_path / "cache/studies/study.npy")
    studies = pd.DataFrame({STUDY_ID: ["study"]})
    config = {"cache_size": 448, "image_size": 384, "raptor_variant": RAPTOR_DENSE_448_VERSION,
              "raptor_view": "center_crop", "raptor_eval_windows": 5}
    sample = RaptorDataset(manifest, studies, config, training=False)[0]
    record = json.loads((tmp_path / "cache/studies/study.json").read_text())
    centers = np.asarray(record["allowed_centers"])[np.linspace(0, 14, 5).round().astype(int)]
    whole = volume[centers[:, None] + np.array([-1, 0, 1])].astype(np.float32) / 255.
    assert sample["images"].shape == (1, 5, 3, 384, 384)
    np.testing.assert_array_equal(sample["images"][0].numpy(), whole[:, :, 32:416, 32:416])
    # The default view still resizes the whole cached slice.
    resized = RaptorDataset(manifest, studies, {**config, "raptor_view": "resize"}, training=False)[0]["images"]
    torch.testing.assert_close(resized[0], torch.nn.functional.interpolate(
        torch.from_numpy(whole), size=(384, 384), mode="bilinear", align_corners=False), rtol=0, atol=0)
    with pytest.raises(ValueError, match="center_crop"):
        RaptorDataset(manifest, studies, {**config, "image_size": 512}, training=False)
    with pytest.raises(ValueError, match="cache_size must be 448"):
        RaptorDataset(manifest, studies, {**config, "cache_size": 336}, training=False)


@pytest.mark.parametrize("change,match", [
    ({"image_size": 512}, "center_crop needs image_size"),
    ({"cache_size": 336}, "requires cache_size 448"),
    ({"raptor_view": "zoom"}, "raptor_view must be"),
    ({"raptor_variant": RAPTOR_DENSE_VERSION}, "requires cache_size 336"),
])
def test_config_rejects_views_and_sizes_that_do_not_fit_the_cache(change, match):
    config = deepcopy(DEFAULTS)
    config["data"].update({"recipe": "raptor", "cache_size": 448, "image_size": 384,
                           "raptor_variant": RAPTOR_DENSE_448_VERSION, "raptor_view": "center_crop", **change})
    with pytest.raises(ValueError, match=match):
        validate_config(config)
    config["data"].update({"cache_size": 448, "image_size": 384, "raptor_variant": RAPTOR_DENSE_448_VERSION,
                           "raptor_view": "center_crop"})
    validate_config(config)
