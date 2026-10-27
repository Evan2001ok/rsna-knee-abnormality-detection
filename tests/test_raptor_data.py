import hashlib
import json
import os
import shutil
import signal
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pytest
import torch
from test_imaging import write_dicom
from torch.utils.data import DataLoader

from rsna_knee_abnormality_detection import raptor_data
from rsna_knee_abnormality_detection.constants import SERIES_ID, STUDY_ID
from rsna_knee_abnormality_detection.prepare import _metadata_digest
from rsna_knee_abnormality_detection.raptor_data import (
    RAPTOR_DENSE_384_VERSION,
    RAPTOR_DENSE_KNEE_CLOSEUP_384_VERSION,
    RAPTOR_PREPROCESS_VERSION,
    RAPTOR_SIZE,
    RaptorDataset,
    prepare_raptor_data,
    validate_raptor_manifest,
)


def make_raw(root: Path, studies=("study",), split="train", planes=None):
    root.mkdir(parents=True, exist_ok=True)
    planes = planes or [("Sagittal", 1), ("Sagittal", 0), ("Coronal", 1), ("Coronal", 0), ("Axial", 0)]
    rows = []
    orientations = {"Sagittal": (0, 1, 0, 0, 0, -1), "Coronal": (1, 0, 0, 0, 0, -1),
                    "Axial": (1, 0, 0, 0, 1, 0)}
    for study in studies:
        for series_index, (plane, fluid) in enumerate(planes):
            series = f"series-{series_index}"
            rows.append({STUDY_ID: study, SERIES_ID: series, "Anatomical_Plane": plane,
                         "Fluid_Sensitive": fluid, "Fat_Suppression": fluid})
            orientation = orientations[plane]
            normal = np.cross(orientation[:3], orientation[3:])
            for index in range(6):
                pixels = np.arange(16 * 20).reshape(16, 20) + index * 37 + series_index * 21
                write_dicom(root / f"{split}_series" / study / series / f"{index}.dcm", pixels,
                            position=normal * index * 2, orientation=orientation, spacing=(10, 10))
    pd.DataFrame(rows).to_csv(root / f"{split}_series.csv", index=False)
    pd.DataFrame({STUDY_ID: list(studies)}).to_csv(root / f"{split}.csv", index=False)


def test_scored_raw_recipe_matches_crop_percentile_quantization_and_slot_preferences(tmp_path):
    raw, cache = tmp_path / "raw", tmp_path / "cache"
    # Put non-fluid sagittal first in CSV: the fluid preference must override it.
    make_raw(raw, planes=[("Sagittal", 0), ("Sagittal", 1), ("Coronal", 1), ("Coronal", 0), ("Axial", 0)])
    manifest = prepare_raptor_data(raw, cache, workers=1)
    row = pd.read_csv(manifest).iloc[0]
    volume = np.load(cache / row.cache_path)
    metadata = json.loads((cache / row.metadata_path).read_text())
    assert volume.shape == (64, 336, 336)
    assert volume.dtype == np.uint8
    assert metadata["slots"][0][SERIES_ID] == "series-1"
    assert metadata["slots"][1][SERIES_ID] == "series-0"
    # Six native slices use integer endpoints floor(6*.06)=0, floor(6*.94)-1=4.
    picks = np.linspace(0, 4, 18).round().astype(int)
    expected_raw = [np.arange(320, dtype=np.float32).reshape(16, 20) + int(index) * 37 + 21 for index in picks]
    low, high = np.percentile(np.concatenate([image.ravel() for image in expected_raw]), [2, 98])
    normalized = np.clip((expected_raw[0] - low) / (high - low + 1e-6), 0, 1)
    # 140mm / 10mm = 14 pixels, centered at (top=1,left=3).
    expected = (cv2.resize(normalized[1:15, 3:17], (336, 336), interpolation=cv2.INTER_AREA) * 255).astype(np.uint8)
    np.testing.assert_array_equal(volume[0], expected)
    assert metadata["slots"][0]["selected_source_indices"] == picks.tolist()
    assert metadata["slots"][0]["crops"][0] == {"top": 1, "left": 3, "size": 14, "source_shape": [16, 20]}
    assert validate_raptor_manifest(manifest, verify_checksums=True)["valid_windows"] == 54


def test_dataset_windows_never_cross_series_and_preserve_actual_plane_labels(tmp_path):
    raw, cache = tmp_path / "raw", tmp_path / "cache"
    make_raw(raw)
    manifest = prepare_raptor_data(raw, cache, workers=1)
    studies = pd.DataFrame({STUDY_ID: ["study"], "ACL": [.25]})
    dataset = RaptorDataset(manifest, studies, {"cache_size": 336, "image_size": 48}, training=False)
    sample = dataset[0]
    assert sample["images"].shape == (1, 42, 3, 48, 48)
    assert sample["series_mask"].tolist() == [True]
    assert sample["window_planes"].shape == (1, 42)
    assert sample["metadata"].shape == (1, 5)
    assert sample["target_mask"].sum() == 1
    assert sample["targets"][0] == .25
    metadata = json.loads((cache / "studies/study.json").read_text())
    all_centers = np.asarray(metadata["allowed_centers"])
    centers = all_centers[np.linspace(0, len(all_centers) - 1, 42).round().astype(int)]
    assert not set(centers) & {0, 17, 18, 31, 32, 43, 44, 51, 52, 63}
    expected_planes = np.where(centers < 32, 0, np.where(centers < 52, 1, 2))
    np.testing.assert_array_equal(sample["window_planes"][0], expected_planes)
    expected_positions = []
    for center in centers:
        slot = next(slot for slot in metadata["slots"] if slot["offset"] <= center < slot["offset"] + slot["count"])
        expected_positions.append(slot["selected_source_indices"][center - slot["offset"]] / (slot["source_depth"] - 1))
    np.testing.assert_allclose(sample["positions"][0], expected_positions)
    assert sample["positions"].max() <= .8  # Last sampled source slice is index4 of6, not global index63.
    assert 0 <= sample["images"].min() <= sample["images"].max() <= 1
    dataset.set_epoch(5)
    assert torch.equal(dataset[0]["images"], sample["images"])


def test_missing_slots_are_masked_and_evaluation_repeats_only_real_windows(tmp_path):
    raw, cache = tmp_path / "raw", tmp_path / "cache"
    make_raw(raw, planes=[("Sagittal", 1)])
    manifest = prepare_raptor_data(raw, cache, workers=1)
    metadata = json.loads((cache / "studies/study.json").read_text())
    assert sum(metadata["valid_slices"]) == 18
    assert metadata["allowed_centers"] == list(range(1, 17))
    sample = RaptorDataset(manifest, pd.DataFrame({STUDY_ID: ["study"]}), {"image_size": 32})[0]
    assert sample["images"].shape == (1, 42, 3, 32, 32)
    assert sample["window_planes"].unique().tolist() == [0]
    assert (sample["images"].sum(dim=(-1, -2, -3)) > 0).all()


def test_epoch_sampling_and_brightness_are_reproducible_in_persistent_workers(tmp_path):
    raw, cache = tmp_path / "raw", tmp_path / "cache"
    make_raw(raw)
    manifest = prepare_raptor_data(raw, cache, workers=1)
    studies = pd.DataFrame({STUDY_ID: ["study"]})
    config = {"image_size": 32, "raptor_train_windows": 12}
    dataset = RaptorDataset(manifest, studies, config, training=True, seed=72)
    first = dataset[0]
    assert torch.equal(first["images"], dataset[0]["images"])
    loader = DataLoader(dataset, batch_size=1, num_workers=1, persistent_workers=True)
    assert torch.equal(next(iter(loader))["images"][0], first["images"])
    dataset.set_epoch(1)
    changed = next(iter(loader))["images"][0]
    assert not torch.equal(changed, first["images"])
    assert torch.equal(changed, dataset[0]["images"])
    dataset.set_epoch(0)
    assert torch.equal(dataset[0]["images"], first["images"])


def test_resume_checks_source_cache_and_manifest_fingerprints(tmp_path):
    raw, cache = tmp_path / "raw", tmp_path / "cache"
    make_raw(raw)
    manifest = prepare_raptor_data(raw, cache, workers=1)
    path = cache / "studies/study.npy"
    before = path.stat().st_mtime_ns
    prepare_raptor_data(raw, cache, workers=1)
    assert pd.read_csv(manifest).iloc[0].status == "reused"
    assert path.stat().st_mtime_ns == before
    source = raw / "train_series/study/series-0/0.dcm"
    stat = source.stat()
    os.utime(source, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1000))
    prepare_raptor_data(raw, cache, workers=1)
    row = pd.read_csv(manifest).iloc[0]
    assert row.status == "prepared" and "cache_rebuilt" in row.warnings
    with path.open("r+b") as handle:
        handle.seek(-1, os.SEEK_END)
        original = handle.read(1)
        handle.seek(-1, os.SEEK_END)
        handle.write(bytes([original[0] ^ 255]))
    with pytest.raises(ValueError, match="checksum mismatch"):
        validate_raptor_manifest(manifest, verify_checksums=True)
    prepare_raptor_data(raw, cache, workers=1)
    assert validate_raptor_manifest(manifest, verify_checksums=True)["studies"] == 1


def test_invalid_series_fails_study_and_keeps_previous_manifest(tmp_path):
    raw, cache = tmp_path / "raw", tmp_path / "cache"
    make_raw(raw)
    manifest = prepare_raptor_data(raw, cache, workers=1)
    previous = manifest.read_bytes()
    (raw / "train_series/study/series-0/0.dcm").write_bytes(b"broken")
    with pytest.raises(RuntimeError, match="failed for 1 studies"):
        prepare_raptor_data(raw, cache, workers=1)
    assert manifest.read_bytes() == previous
    report = json.loads((cache / "qc.json").read_text())
    assert report["summary"]["failed_studies"] == 1
    assert report["studies"][0]["error"]


def test_no_nonzero_windows_fails_instead_of_returning_a_zero_image_study(tmp_path):
    raw, cache = tmp_path / "raw", tmp_path / "cache"
    make_raw(raw, planes=[("Axial", 0)])
    for index in range(6):
        write_dicom(raw / "train_series/study/series-0" / f"{index}.dcm", np.zeros((16, 20)),
                    position=(0, 0, index), spacing=(10, 10))
    with pytest.raises(RuntimeError, match="failed for 1 studies"):
        prepare_raptor_data(raw, cache, workers=1)
    assert not (cache / "manifest.csv").exists()
    assert "no valid three-slice windows" in json.loads((cache / "qc.json").read_text())["studies"][0]["error"]


def test_validator_rejects_wrong_size_version_and_tampered_windows(tmp_path):
    raw, cache = tmp_path / "raw", tmp_path / "cache"
    make_raw(raw)
    manifest = prepare_raptor_data(raw, cache, workers=1)
    with pytest.raises(ValueError, match="expects cache size"):
        validate_raptor_manifest(manifest, expected_size=320)
    metadata_path = cache / "studies/study.json"
    metadata = json.loads(metadata_path.read_text())
    metadata["allowed_centers"][0] = 17  # Channel three would be from the next series.
    metadata["metadata_payload_sha256"] = _metadata_digest(metadata)
    metadata_path.write_text(json.dumps(metadata))
    table = pd.read_csv(manifest)
    table.loc[0, "metadata_sha256"] = hashlib.sha256(metadata_path.read_bytes()).hexdigest()
    table.to_csv(manifest, index=False)
    with pytest.raises(ValueError, match="stored windows cross a series boundary"):
        validate_raptor_manifest(manifest)
    metadata["preprocess_version"] = "old-version"
    metadata["metadata_payload_sha256"] = _metadata_digest(metadata)
    metadata_path.write_text(json.dumps(metadata))
    table.loc[0, "metadata_sha256"] = hashlib.sha256(metadata_path.read_bytes()).hexdigest()
    table.to_csv(manifest, index=False)
    with pytest.raises(ValueError, match="version mismatch"):
        validate_raptor_manifest(manifest)


def test_subset_validation_does_not_read_other_studies_and_test_prep_supports_processes(tmp_path):
    raw, cache = tmp_path / "raw", tmp_path / "cache"
    make_raw(raw, studies=("a", "b"), split="test")
    manifest = prepare_raptor_data(raw, cache, split="test", workers=2)
    (cache / "studies/b.npy").unlink()
    result = validate_raptor_manifest(manifest, study_ids=["a"])
    assert result["studies"] == 1
    assert result["preprocess_version"] == RAPTOR_PREPROCESS_VERSION
    assert result["image_size"] == RAPTOR_SIZE
    assert result["image_preparation"]["windows"].endswith("selected series")
    with pytest.raises(ValueError, match="Invalid Raptor cache for b"):
        validate_raptor_manifest(manifest)


# Bad-series insurance (drop_bad_series=True, used by the Kaggle bundle): a failing series is dropped instead of
# failing its whole study, and nothing changes for studies where no series fails.
def _break_series(series_dir: Path, how: str) -> None:
    if how == "unreadable_file":
        (series_dir / "0.dcm").write_bytes(b"broken")
    elif how == "missing_folder":
        shutil.rmtree(series_dir)
    else:  # an inconsistent series: two slices at the same position fail the series checks
        write_dicom(series_dir / "1.dcm", np.ones((16, 20)), position=(0, 0, 0), orientation=(1, 0, 0, 0, 1, 0),
                    spacing=(10, 10))


def _study_files(cache: Path, study: str) -> tuple[np.ndarray, dict, bytes, bytes]:
    npy, meta = cache / f"studies/{study}.npy", cache / f"studies/{study}.json"
    return np.load(npy), json.loads(meta.read_text()), npy.read_bytes(), meta.read_bytes()


class _EndlessLoop(BaseException):
    """Not an Exception, so neither the insurance nor the study's own error handling can swallow it."""


@pytest.fixture
def no_endless_loop():
    """Fail instead of hanging if a slot ever keeps switching between series that already failed."""
    def stop(*_):
        raise _EndlessLoop("preparation did not finish within 60 seconds")
    previous = signal.signal(signal.SIGALRM, stop)
    signal.alarm(60)
    yield
    signal.alarm(0)
    signal.signal(signal.SIGALRM, previous)


def _slot_span(slot: dict) -> np.ndarray:
    return np.arange(slot["offset"], slot["offset"] + slot["count"])


def _assert_only_slot_changed(volume, metadata, clean_volume, clean_metadata, index):
    """Every cached slice and slot record except slot ``index`` is the clean build's."""
    rest = np.ones(len(volume), dtype=bool)
    rest[_slot_span(clean_metadata["slots"][index])] = False
    np.testing.assert_array_equal(volume[rest], clean_volume[rest])
    assert ([slot for number, slot in enumerate(metadata["slots"]) if number != index]
            == [slot for number, slot in enumerate(clean_metadata["slots"]) if number != index])


@pytest.mark.parametrize("how", ["unreadable_file", "missing_folder", "inconsistent_series"])
def test_a_bad_series_without_a_replacement_leaves_its_slot_empty_and_the_rest_unchanged(tmp_path, how):
    raw = tmp_path / "raw"
    make_raw(raw, studies=("a", "b"))
    clean = prepare_raptor_data(raw, tmp_path / "clean", workers=1)
    _break_series(raw / "train_series/b/series-4", how)  # the only Axial series of study b (slot 4)
    with pytest.raises(RuntimeError, match="failed for 1 studies"):  # without insurance the study still fails
        prepare_raptor_data(raw, tmp_path / "strict", workers=1)
    insured = prepare_raptor_data(raw, tmp_path / "insured", workers=1, drop_bad_series=True)
    # Study a has no bad series: its cache, metadata and manifest row are byte for byte the clean ones.
    assert _study_files(tmp_path / "insured", "a")[2:] == _study_files(tmp_path / "clean", "a")[2:]
    # Manifest lines: the header, then studies a and b.
    assert insured.read_text().splitlines()[:2] == clean.read_text().splitlines()[:2]
    insured_rows = pd.read_csv(insured).set_index(STUDY_ID)
    # Study b: slots 0-3 (the first 52 cached slices) are unchanged and slot 4 is empty, like a study without it.
    volume, metadata, _, _ = _study_files(tmp_path / "insured", "b")
    clean_volume, clean_metadata, _, _ = _study_files(tmp_path / "clean", "b")
    np.testing.assert_array_equal(volume[:52], clean_volume[:52])
    assert not volume[52:].any() and not any(metadata["valid_slices"][52:])
    assert metadata["slots"][:4] == clean_metadata["slots"][:4]
    assert metadata["slots"][4] == {"slot": 4, "plane": "Axial", "requested_fluid": -1, "offset": 52, "count": 12,
                                    SERIES_ID: None, "Fluid_Sensitive": None, "Fat_Suppression": None}
    assert [(item["slot"], item[SERIES_ID], item["replacement"]) for item in metadata["dropped_series"]] == [
        (4, "series-4", None)]
    assert "missing_slot:4:Axial" in metadata["warnings"]
    assert any(warning.startswith("dropped_series:slot4:series-4:") for warning in metadata["warnings"])
    assert "dropped_series" in insured_rows.loc["b", "warnings"]
    summary = json.loads((tmp_path / "insured/qc.json").read_text())["summary"]
    assert summary["failed_studies"] == 0
    assert [(item[STUDY_ID], item["slot"], item[SERIES_ID]) for item in summary["dropped_series"]] == [
        ("b", 4, "series-4")]
    assert validate_raptor_manifest(insured, verify_checksums=True)["studies"] == 2
    sample = RaptorDataset(insured, pd.DataFrame({STUDY_ID: ["b"]}), {"image_size": 32})[0]
    assert set(sample["window_planes"].unique().tolist()) == {0, 1}  # sagittal and coronal windows only
    # A study with a dropped series is prepared again on the next run rather than reused.
    prepare_raptor_data(raw, tmp_path / "insured", workers=1, drop_bad_series=True)
    assert pd.read_csv(insured).set_index(STUDY_ID)["status"].to_dict() == {"a": "reused", "b": "prepared"}


def test_a_bad_series_is_replaced_by_the_next_series_of_its_plane(tmp_path):
    raw = tmp_path / "raw"
    # A spare fluid-sensitive sagittal series (series-5) that the first choice (series-0) leaves unused.
    make_raw(raw, planes=[("Sagittal", 1), ("Sagittal", 0), ("Coronal", 1), ("Coronal", 0), ("Axial", 0),
                          ("Sagittal", 1)])
    # make_raw's series differ only by a constant, which normalisation removes; give the spare its own pattern.
    for index in range(6):
        write_dicom(raw / f"train_series/study/series-5/{index}.dcm", (np.arange(320).reshape(16, 20) % 7) * 50 + index,
                    position=(-2 * index, 0, 0), orientation=(0, 1, 0, 0, 0, -1), spacing=(10, 10))
    prepare_raptor_data(raw, tmp_path / "clean", workers=1)
    _break_series(raw / "train_series/study/series-0", "unreadable_file")
    prepare_raptor_data(raw, tmp_path / "insured", workers=1, drop_bad_series=True)
    volume, metadata, insured_npy, _ = _study_files(tmp_path / "insured", "study")
    clean_volume, clean_metadata, _, _ = _study_files(tmp_path / "clean", "study")
    assert metadata["slots"][0][SERIES_ID] == "series-5"
    assert [(item["slot"], item[SERIES_ID], item["replacement"]) for item in metadata["dropped_series"]] == [
        (0, "series-0", "series-5")]
    # The other slots keep their series and images.
    np.testing.assert_array_equal(volume[18:], clean_volume[18:])
    assert metadata["slots"][1:] == clean_metadata["slots"][1:]
    assert not np.array_equal(volume[:18], clean_volume[:18])
    # The replacement gives exactly the cache of a study that never had series-0.
    table = pd.read_csv(raw / "train_series.csv")
    table[table[SERIES_ID] != "series-0"].to_csv(raw / "train_series.csv", index=False)
    prepare_raptor_data(raw, tmp_path / "without", workers=1)
    _, without_metadata, without_npy, _ = _study_files(tmp_path / "without", "study")
    assert insured_npy == without_npy
    for key in ("slots", "valid_slices", "allowed_centers", "window_planes", "cache_sha256"):
        assert metadata[key] == without_metadata[key]


@pytest.mark.parametrize("slot", [0, 2])
def test_a_failing_fluid_series_without_a_spare_leaves_its_slot_empty_and_no_other_slot_moves(tmp_path, slot,
                                                                                             no_endless_loop):
    # The usual layout: the plane's other series already fills the next slot (1 or 3), so there is no replacement.
    # That series must stay in its own slot: taking it would put one series in two slots.
    raw = tmp_path / "raw"
    make_raw(raw)
    prepare_raptor_data(raw, tmp_path / "clean", workers=1)
    _break_series(raw / f"train_series/study/series-{slot}", "unreadable_file")
    manifest = prepare_raptor_data(raw, tmp_path / "insured", workers=1, drop_bad_series=True)
    volume, metadata, _, _ = _study_files(tmp_path / "insured", "study")
    clean_volume, clean_metadata, _, _ = _study_files(tmp_path / "clean", "study")
    assert [item[SERIES_ID] for item in metadata["slots"]] == [None if index == slot else f"series-{index}"
                                                               for index in range(5)]
    assert not volume[_slot_span(clean_metadata["slots"][slot])].any()
    _assert_only_slot_changed(volume, metadata, clean_volume, clean_metadata, slot)
    assert [(item["slot"], item[SERIES_ID], item["replacement"]) for item in metadata["dropped_series"]] == [
        (slot, f"series-{slot}", None)]
    validate_raptor_manifest(manifest, verify_checksums=True)


@pytest.mark.parametrize("spare", ["readable", "unreadable_file", "missing_folder"])
def test_a_spare_replaces_a_failing_series_only_if_it_reads_and_never_takes_another_slots_series(tmp_path, spare,
                                                                                                no_endless_loop):
    # series-5 is a second non-fluid sagittal series. When the fluid-sensitive series-0 (slot 0) fails, no fluid
    # series is left, so the choice falls back to the non-fluid series that no other slot uses: series-5, not
    # series-1 (slot 1). If series-5 fails as well, slot 0 is left empty.
    raw = tmp_path / "raw"
    make_raw(raw, planes=[("Sagittal", 1), ("Sagittal", 0), ("Coronal", 1), ("Coronal", 0), ("Axial", 0),
                          ("Sagittal", 0)])
    prepare_raptor_data(raw, tmp_path / "clean", workers=1)
    _break_series(raw / "train_series/study/series-0", "unreadable_file")
    if spare != "readable":
        _break_series(raw / "train_series/study/series-5", spare)
    manifest = prepare_raptor_data(raw, tmp_path / "insured", workers=1, drop_bad_series=True)
    volume, metadata, _, _ = _study_files(tmp_path / "insured", "study")
    clean_volume, clean_metadata, _, _ = _study_files(tmp_path / "clean", "study")
    first = "series-5" if spare == "readable" else None
    assert [item[SERIES_ID] for item in metadata["slots"]] == [first, "series-1", "series-2", "series-3", "series-4"]
    _assert_only_slot_changed(volume, metadata, clean_volume, clean_metadata, 0)
    assert volume[:18].any() == (spare == "readable")
    expected = [(0, "series-0", "series-5")] + ([] if spare == "readable" else [(0, "series-5", None)])
    assert [(item["slot"], item[SERIES_ID], item["replacement"]) for item in metadata["dropped_series"]] == expected
    validate_raptor_manifest(manifest, verify_checksums=True)


@pytest.mark.parametrize("error", [OSError, MemoryError])
def test_a_one_off_read_error_is_read_again_instead_of_dropping_the_series(tmp_path, monkeypatch, error):
    raw = tmp_path / "raw"
    make_raw(raw)
    clean = prepare_raptor_data(raw, tmp_path / "clean", workers=1)
    read, failed = raptor_data._read_pixels, []

    def flaky(path):
        if path.parent.name == "series-4" and not failed:
            failed.append(path)
            raise error("a one-off error")
        return read(path)

    monkeypatch.setattr(raptor_data, "_read_pixels", flaky)  # workers=1 prepares in this process
    insured = prepare_raptor_data(raw, tmp_path / "insured", workers=1, drop_bad_series=True)
    assert len(failed) == 1
    # The second read worked, so the study is exactly the clean build: nothing dropped or recorded.
    assert insured.read_bytes() == clean.read_bytes()
    assert _study_files(tmp_path / "insured", "study")[2:] == _study_files(tmp_path / "clean", "study")[2:]
    # Without insurance the same one-off error still fails the study, as before.
    failed.clear()
    with pytest.raises(RuntimeError, match="failed for 1 studies"):
        prepare_raptor_data(raw, tmp_path / "strict", workers=1)
    assert len(failed) == 1


@pytest.mark.parametrize("faults", [
    ["unreadable_file"] * 5,
    # A reading error in slot 0 and listing errors in slots 1-4: without insurance the listing errors come first.
    ["unreadable_file"] + ["missing_folder"] * 4,
    # The only listing error is in the last slot, and it still comes first.
    ["inconsistent_series"] * 4 + ["missing_folder"],
])
def test_a_study_whose_every_series_fails_still_fails_with_the_same_error(tmp_path, faults, no_endless_loop):
    raw = tmp_path / "raw"
    make_raw(raw)
    for index, how in enumerate(faults):
        _break_series(raw / f"train_series/study/series-{index}", how)
    errors = []
    for name, insured in (("strict", False), ("insured", True)):
        with pytest.raises(RuntimeError, match="failed for 1 studies"):
            prepare_raptor_data(raw, tmp_path / name, workers=1, drop_bad_series=insured)
        assert not (tmp_path / name / "manifest.csv").exists()
        report = json.loads((tmp_path / name / "qc.json").read_text())
        assert "dropped_series" not in report["summary"]
        errors.append(report["studies"][0]["error"])
    assert errors[0] == errors[1] and errors[0]


# Record layouts written by the code before the insurance (checked against that code on Oct 17, 2026).
QC_KEYS = ["preprocess_version", "input_contract", "split", "data_root", "selected_studies", "summary", "studies"]
QC_SUMMARY_KEYS = ["studies", "failed_studies", "prepared_studies", "reused_studies", "studies_with_warnings", "seconds"]
MANIFEST_COLUMNS = [STUDY_ID, "cache_path", "metadata_path", "num_slices", "image_size", "num_valid_windows",
                    "num_slots_used", "preprocess_version", "preprocess_fingerprint", "source_fingerprint",
                    "cache_sha256", "metadata_sha256", "status", "warnings"]
METADATA_KEYS = {STUDY_ID, "shape", "dtype", "preprocess_version", "preprocess_fingerprint", "input_contract",
                 "source_fingerprint", "cache_sha256", "slots", "valid_slices", "allowed_centers", "window_planes",
                 "warnings", "metadata_payload_sha256"}
_PLAN_KEYS = {"slot", "plane", "requested_fluid", "offset", "count", SERIES_ID, "Fluid_Sensitive", "Fat_Suppression"}
_READ_KEYS = {"source_depth", "ordering", "selected_source_indices", "selected_source_files", "selected_positions",
              "selected_pixel_spacings_mm", "selected_orientations_lps", "selected_image_positions_lps",
              "normalization", "crops", "warnings"}
MAIN_SLOT_KEYS = {False: _PLAN_KEYS, True: _PLAN_KEYS | _READ_KEYS | {"source_fingerprint"}}  # by "series present"
CLOSEUP_KEYS = {False: {"slot", "plane", "requested_fluid", "offset", "count", "closeup", SERIES_ID},
                True: _PLAN_KEYS | _READ_KEYS | {"closeup", "source_slot", "center_lps", "center_source"}}


@pytest.mark.parametrize("version", [RAPTOR_PREPROCESS_VERSION, RAPTOR_DENSE_384_VERSION,
                                     RAPTOR_DENSE_KNEE_CLOSEUP_384_VERSION])
def test_clean_studies_are_byte_identical_with_and_without_insurance(tmp_path, version):
    raw = tmp_path / "raw"
    make_raw(raw, studies=("a", "b"))
    make_raw(tmp_path / "one", studies=("c",), planes=[("Coronal", 0), ("Sagittal", 1)])  # a study missing series
    for name in ("train.csv", "train_series.csv"):
        pd.concat([pd.read_csv(root / name) for root in (raw, tmp_path / "one")]).to_csv(raw / name, index=False)
    shutil.copytree(tmp_path / "one/train_series/c", raw / "train_series/c")
    anchors = None
    if version == RAPTOR_DENSE_KNEE_CLOSEUP_384_VERSION:
        anchors = tmp_path / "anchors"
        anchors.mkdir()
        (anchors / "a.json").write_text(json.dumps({"joint_lps": [-4., 95., -75.], "patella_lps": [-4., 55., 10.],
                                                    "method": "test"}))
    outputs = {}
    for name, insured in (("strict", False), ("insured", True)):
        manifest = prepare_raptor_data(raw, tmp_path / name, workers=1, version=version, anchors_dir=anchors,
                                       drop_bad_series=insured)
        report = json.loads((tmp_path / name / "qc.json").read_text())
        # The same records as the code before the insurance: no new key anywhere, with or without it.
        assert list(report) == QC_KEYS and list(report["summary"]) == QC_SUMMARY_KEYS
        assert all(list(row) == [*MANIFEST_COLUMNS, "error"] for row in report["studies"])
        assert list(pd.read_csv(manifest).columns) == MANIFEST_COLUMNS
        for path in (tmp_path / name / "studies").glob("*.json"):
            metadata = json.loads(path.read_text())
            assert set(metadata) == METADATA_KEYS | ({"anchor"} if anchors else set())
            assert "dropped" not in path.read_text()
            for slot in metadata["slots"]:
                closeup, filled = "closeup" in slot, slot[SERIES_ID] is not None
                assert set(slot) == (CLOSEUP_KEYS if closeup else MAIN_SLOT_KEYS)[filled], (path.name, slot["slot"])
        outputs[name] = {path.name: path.read_bytes() for path in sorted((tmp_path / name / "studies").iterdir())}
        outputs[name]["manifest"] = manifest.read_bytes()
    assert len(outputs["strict"]) == 7 and outputs["strict"] == outputs["insured"]


def test_an_unreadable_closeup_is_left_empty_and_the_main_slots_are_unchanged(tmp_path):
    raw, anchors = tmp_path / "raw", tmp_path / "anchors"
    make_raw(raw)
    anchors.mkdir()
    # The kneecap at z = 10 mm is nearest axial slice 5, so the axial close-up reads slices 1, 3 and 5; the main
    # axial slot reads only slices 0-4 (the 6-94% span), so a bad slice 5 harms only the close-up.
    (anchors / "study.json").write_text(json.dumps({"joint_lps": [-4., 95., -75.], "patella_lps": [-4., 55., 10.],
                                                    "method": "test"}))
    version = RAPTOR_DENSE_KNEE_CLOSEUP_384_VERSION
    prepare_raptor_data(raw, tmp_path / "clean", workers=1, version=version, anchors_dir=anchors)
    write_dicom(raw / "train_series/study/series-4/5.dcm", np.arange(320).reshape(16, 20), position=(0, 0, 10),
                orientation=(1, 0, 0, 0, 1, 0), spacing=(10, 10), photometric="PALETTE COLOR")
    with pytest.raises(RuntimeError, match="failed for 1 studies"):
        prepare_raptor_data(raw, tmp_path / "strict", workers=1, version=version, anchors_dir=anchors)
    manifest = prepare_raptor_data(raw, tmp_path / "insured", workers=1, version=version, anchors_dir=anchors,
                                   drop_bad_series=True)
    volume, metadata, _, _ = _study_files(tmp_path / "insured", "study")
    clean_volume, clean_metadata, _, _ = _study_files(tmp_path / "clean", "study")
    axial = metadata["slots"][6]  # the axial close-up
    assert axial["closeup"] == "axial_kneecap" and axial[SERIES_ID] is None
    assert clean_metadata["slots"][6][SERIES_ID] == "series-4"
    span = slice(axial["offset"], axial["offset"] + axial["count"])
    assert clean_volume[span].any() and not volume[span].any()
    rest = np.ones(len(volume), dtype=bool)
    rest[span] = False
    np.testing.assert_array_equal(volume[rest], clean_volume[rest])
    # Slot 4 reads the same slices; only its file fingerprint changed with the rewritten slice 5.
    unfingerprinted = [{**slot, "source_fingerprint": None} for slot in metadata["slots"]]
    clean_unfingerprinted = [{**slot, "source_fingerprint": None} for slot in clean_metadata["slots"]]
    assert unfingerprinted[:6] + unfingerprinted[7:] == clean_unfingerprinted[:6] + clean_unfingerprinted[7:]
    assert [(item["slot"], item[SERIES_ID], item["replacement"]) for item in metadata["dropped_series"]] == [
        (6, "series-4", None)]
    assert "missing_closeup:6:axial_kneecap" in metadata["warnings"]
    validate_raptor_manifest(manifest, version=version, verify_checksums=True)
