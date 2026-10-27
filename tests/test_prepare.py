import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from test_imaging import write_dicom

from rsna_knee_abnormality_detection import prepare as preparation
from rsna_knee_abnormality_detection.constants import SERIES_ID, STUDY_ID
from rsna_knee_abnormality_detection.prepare import prepare_data, prepare_one_series, validate_manifest


def make_dataset(root: Path, rows, split="train"):
    root.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{STUDY_ID: study, SERIES_ID: series, "Fluid_Sensitive": 1,
                   "Fat_Suppression": 0, "Anatomical_Plane": "Axial"}
                  for study, series in rows]).to_csv(root / f"{split}_series.csv", index=False)
    pd.DataFrame({STUDY_ID: list(dict.fromkeys(study for study, _ in rows))}).to_csv(root / f"{split}.csv", index=False)
    for study, series in rows:
        directory = root / f"{split}_series" / study / series
        for index in range(2):
            write_dicom(directory / f"{index}.dcm", np.arange(12).reshape(3, 4) + index * 20,
                        position=(0, 0, index * 2))


def test_manifest_cache_reuse_and_subset_do_not_delete_cache(tmp_path):
    root, cache = tmp_path / "data", tmp_path / "cache"
    make_dataset(root, [("study-a", "series-a"), ("study-b", "series-b")])
    manifest = prepare_data(root, cache, size=16, workers=1)
    rows = pd.read_csv(manifest)
    assert len(rows) == 2
    assert set(rows["status"]) == {"prepared"}
    assert (cache / rows.iloc[0]["cache_path"]).exists()
    assert (cache / rows.iloc[0]["metadata_path"]).exists()
    assert len(rows.iloc[0]["metadata_sha256"]) == 64
    volume = np.load(cache / rows.iloc[0]["cache_path"])
    assert volume.shape == (2, 16, 16)
    assert volume.dtype == np.uint16
    before = (cache / rows.iloc[0]["cache_path"]).stat().st_mtime_ns
    prepare_data(root, cache, size=16, workers=1)
    assert set(pd.read_csv(manifest)["status"]) == {"reused"}
    assert (cache / rows.iloc[0]["cache_path"]).stat().st_mtime_ns == before
    prepare_data(root, cache, size=16, workers=1, study_ids=["study-a"])
    assert pd.read_csv(manifest)[STUDY_ID].tolist() == ["study-a"]
    assert (cache / rows.iloc[1]["cache_path"]).exists()


def test_source_size_and_corruption_invalidate_cache(tmp_path):
    series, path = tmp_path / "source", tmp_path / "cached.npy"
    source = series / "one.dcm"
    write_dicom(source)
    args = dict(study_id="study", series_id="series", anatomical_plane="Axial", size=16)
    first = prepare_one_series(series, path, **args)
    assert first["status"] == "prepared"
    assert prepare_one_series(series, path, **args)["status"] == "reused"
    stat = source.stat()
    os.utime(source, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1000))
    changed = prepare_one_series(series, path, **args)
    assert changed["status"] == "prepared"
    assert "cache_rebuilt:source_fingerprint_changed" in changed["warnings"]
    args["size"] = 24
    resized = prepare_one_series(series, path, **args)
    assert resized["status"] == "prepared"
    assert np.load(path).shape == (1, 24, 24)
    assert "cache_rebuilt:preprocess_fingerprint_changed" in resized["warnings"]
    with path.open("r+b") as handle:
        handle.seek(-1, os.SEEK_END)
        original = handle.read(1)
        handle.seek(-1, os.SEEK_END)
        handle.write(bytes([original[0] ^ 255]))
    rebuilt = prepare_one_series(series, path, **args)
    assert rebuilt["status"] == "prepared"
    assert "cache_rebuilt:checksum_mismatch" in rebuilt["warnings"]


def test_failed_series_recorded_and_strict_mode_rejects_partial_study(tmp_path):
    root, cache = tmp_path / "data", tmp_path / "cache"
    make_dataset(root, [("study", "good"), ("study", "bad")])
    (root / "train_series/study/bad/0.dcm").write_bytes(b"broken")
    manifest = prepare_data(root, cache, size=16, workers=1)
    assert pd.read_csv(manifest)[SERIES_ID].tolist() == ["good"]
    report = json.loads((cache / "qc.json").read_text())
    assert report["summary"]["failed_series"] == 1
    assert any(row["status"] == "failed" and "Could not decode" in row["error"] for row in report["series"])
    previous_manifest = manifest.read_bytes()
    with pytest.raises(RuntimeError, match="1 unusable series"):
        prepare_data(root, cache, size=16, workers=1, strict=True)
    assert manifest.read_bytes() == previous_manifest


def test_study_with_no_usable_series_always_fails_after_qc(tmp_path):
    root, cache = tmp_path / "data", tmp_path / "cache"
    make_dataset(root, [("study", "bad")])
    (root / "train_series/study/bad/0.dcm").write_bytes(b"broken")
    with pytest.raises(RuntimeError, match="1 studies have no usable series"):
        prepare_data(root, cache, size=16, workers=1)
    assert not (cache / "manifest.csv").exists()
    report = json.loads((cache / "qc.json").read_text())
    assert report["studies_without_usable_series"] == ["study"]


def test_study_without_metadata_rows_is_not_silently_lost(tmp_path):
    root, cache = tmp_path / "data", tmp_path / "cache"
    make_dataset(root, [("good", "one")])
    pd.DataFrame({STUDY_ID: ["good", "missing"]}).to_csv(root / "train.csv", index=False)
    with pytest.raises(RuntimeError, match="1 studies have no usable series"):
        prepare_data(root, cache, size=16, workers=1)
    assert json.loads((cache / "qc.json").read_text())["studies_without_usable_series"] == ["missing"]


def test_test_split_and_process_workers(tmp_path):
    root, cache = tmp_path / "data", tmp_path / "cache"
    make_dataset(root, [("study-a", "series-a"), ("study-b", "series-b")], split="test")
    manifest = prepare_data(root, cache, split="test", size=16, workers=2, limit_studies=1)
    assert pd.read_csv(manifest)[STUDY_ID].tolist() == ["study-a"]
    assert json.loads((cache / "qc.json").read_text())["split"] == "test"


def test_broken_metadata_and_truncated_array_are_rebuilt(tmp_path):
    series, path = tmp_path / "source", tmp_path / "cached.npy"
    write_dicom(series / "one.dcm")
    args = dict(study_id="study", series_id="series", anatomical_plane="Axial", size=16)
    prepare_one_series(series, path, **args)
    path.with_suffix(".json").write_text("not JSON")
    assert prepare_one_series(series, path, **args)["status"] == "prepared"
    path.write_bytes(path.read_bytes()[:130])
    rebuilt = prepare_one_series(series, path, **args)
    assert rebuilt["status"] == "prepared"
    assert np.load(path).shape == (1, 16, 16)
    metadata_path = path.with_suffix(".json")
    metadata = json.loads(metadata_path.read_text())
    metadata["geometry"]["slice_spacing_mm"] = 999
    metadata_path.write_text(json.dumps(metadata))
    rebuilt = prepare_one_series(series, path, **args)
    assert "cache_rebuilt:metadata_checksum_mismatch" in rebuilt["warnings"]


def test_validate_manifest_checks_metadata_without_reading_array_checksums(tmp_path, monkeypatch):
    root, cache = tmp_path / "data", tmp_path / "cache"
    make_dataset(root, [("study-a", "series-a"), ("study-b", "series-b")])
    manifest = prepare_data(root, cache, size=16, workers=1)

    def no_full_array_read(_):
        raise AssertionError("Default validation must not hash full image arrays")

    monkeypatch.setattr(preparation, "_sha256_file", no_full_array_read)
    result = validate_manifest(manifest, expected_size=16)
    assert result["studies"] == 2
    assert result["series"] == 2
    assert result["slices"] == 4
    assert result["image_size"] == 16
    assert not result["array_checksums_verified"]


def test_validate_manifest_rejects_size_and_version_mismatch(tmp_path, monkeypatch):
    root, cache = tmp_path / "data", tmp_path / "cache"
    make_dataset(root, [("study", "series")])
    manifest = prepare_data(root, cache, size=16, workers=1)
    with pytest.raises(ValueError, match="size 16 differs from expected size 24"):
        validate_manifest(manifest, expected_size=24)
    monkeypatch.setattr(preparation, "PREPROCESS_VERSION", "incompatible-new-preprocessing")
    with pytest.raises(ValueError, match="preprocessing version"):
        validate_manifest(manifest)


def test_validate_manifest_subset_does_not_inspect_unrelated_cache(tmp_path):
    root, cache = tmp_path / "data", tmp_path / "cache"
    make_dataset(root, [("study-a", "series-a"), ("study-b", "series-b")])
    manifest = prepare_data(root, cache, size=16, workers=1)
    (cache / "series/study-b/series-b.npy").unlink()
    result = validate_manifest(manifest, study_ids=["study-a"], verify_checksums=True)
    assert result["studies"] == 1
    assert result["array_checksums_verified"]
    with pytest.raises(ValueError, match="study-b/series-b"):
        validate_manifest(manifest)
    with pytest.raises(ValueError, match="no manifest entries"):
        validate_manifest(manifest, study_ids=["absent"])


def test_validate_manifest_rejects_cache_changed_after_manifest_was_written(tmp_path):
    root, cache = tmp_path / "data", tmp_path / "cache"
    make_dataset(root, [("study", "series")])
    manifest = prepare_data(root, cache, size=16, workers=1)
    path = root / "train_series/study/series/0.dcm"
    stat = path.stat()
    os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1000))
    prepare_one_series(path.parent, cache / "series/study/series.npy",
                       study_id="study", series_id="series", anatomical_plane="Axial", size=16)
    with pytest.raises(ValueError, match="metadata checksum differs from manifest"):
        validate_manifest(manifest)


def test_validate_manifest_detects_pixel_damage_only_when_checksums_requested(tmp_path):
    root, cache = tmp_path / "data", tmp_path / "cache"
    make_dataset(root, [("study", "series")])
    manifest = prepare_data(root, cache, size=16, workers=1)
    path = cache / "series/study/series.npy"
    with path.open("r+b") as handle:
        handle.seek(-1, os.SEEK_END)
        original = handle.read(1)
        handle.seek(-1, os.SEEK_END)
        handle.write(bytes([original[0] ^ 255]))
    # The inexpensive check deliberately does not claim to read every pixel.
    assert validate_manifest(manifest)["series"] == 1
    with pytest.raises(ValueError, match="cached array checksum mismatch"):
        validate_manifest(manifest, verify_checksums=True)


@pytest.mark.parametrize("damage", ["truncated", "trailing_bytes", "wrong_dtype"])
def test_validate_manifest_detects_bad_array_header_or_length_without_full_hash(tmp_path, damage):
    root, cache = tmp_path / "data", tmp_path / "cache"
    make_dataset(root, [("study", "series")])
    manifest = prepare_data(root, cache, size=16, workers=1)
    path = cache / "series/study/series.npy"
    if damage == "truncated":
        path.write_bytes(path.read_bytes()[:-1])
    elif damage == "trailing_bytes":
        with path.open("ab") as handle:
            handle.write(b"unexpected")
    else:
        np.save(path, np.zeros((2, 16, 16), dtype=np.uint8))
    with pytest.raises(ValueError, match="Invalid cache"):
        validate_manifest(manifest)


def test_validate_manifest_rejects_tampered_metadata_and_inconsistent_geometry(tmp_path):
    root, cache = tmp_path / "data", tmp_path / "cache"
    make_dataset(root, [("study", "series")])
    manifest = prepare_data(root, cache, size=16, workers=1)
    metadata_path = cache / "series/study/series.json"
    metadata = json.loads(metadata_path.read_text())
    metadata["geometry"]["slice_positions_mm"] = [2, 0]
    metadata_path.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="metadata checksum differs from manifest"):
        validate_manifest(manifest)
    # Correct file hashes must still not legitimize inconsistent geometry.
    metadata["metadata_payload_sha256"] = preparation._metadata_digest(metadata)
    metadata_path.write_text(json.dumps(metadata))
    table = pd.read_csv(manifest)
    table.loc[0, "metadata_sha256"] = hashlib.sha256(metadata_path.read_bytes()).hexdigest()
    table.to_csv(manifest, index=False)
    with pytest.raises(ValueError, match="positions must be finite and strictly increasing"):
        validate_manifest(manifest)
