"""Build a resumable image cache and a complete, reviewable quality report."""

from __future__ import annotations

import concurrent.futures
import contextlib
import fcntl
import hashlib
import json
import multiprocessing
import os
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .constants import PLANES, SERIES_ID, STUDY_ID
from .imaging import PREPROCESS_VERSION, load_dicom_series, prepare_volume, source_fingerprint

_REQUIRED = [STUDY_ID, SERIES_ID, "Anatomical_Plane", "Fluid_Sensitive", "Fat_Suppression"]


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w") as handle:
            json.dump(value, handle, indent=2, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w") as handle:
            frame.to_csv(handle, index=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _metadata_digest(metadata: dict[str, Any]) -> str:
    payload = {key: value for key, value in metadata.items() if key != "metadata_payload_sha256"}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, allow_nan=False).encode()).hexdigest()


def _safe_component(value: str, name: str) -> str:
    if not value or value in {".", ".."} or Path(value).name != value or "/" in value or "\\" in value:
        raise ValueError(f"Invalid {name}: {value!r}")
    return value


def _validate_geometry(metadata: dict[str, Any], row: dict[str, Any], depth: int, size: int) -> None:
    """Check geometry needed to interpret the cached pixels, without reading them."""
    geometry, processing = metadata["geometry"], metadata["processing"]
    if geometry["num_slices"] != depth or geometry["anatomical_plane"] != row["Anatomical_Plane"]:
        raise ValueError("geometry disagrees with the manifest depth or anatomical plane")
    if geometry["ordering"] not in {"physical_position", "instance_number", "file_name"}:
        raise ValueError("unknown slice ordering")
    if geometry["ordering"] != row["slice_ordering"]:
        raise ValueError("geometry and manifest disagree on slice ordering")
    if not isinstance(geometry["warnings"], list) or not all(isinstance(value, str) for value in geometry["warnings"]):
        raise ValueError("geometry warnings must be a list of strings")
    if geometry["ordering"] != "physical_position" and not any(
        warning.startswith("slice_order_fallback:") for warning in geometry["warnings"]
    ):
        raise ValueError("fallback slice ordering has no recorded warning")
    vector_fields = {
        "pixel_spacings_mm": 2, "original_pixel_spacings_mm": 2,
        "image_positions_lps": 3, "oriented_image_positions_lps": 3,
        "original_orientations_lps": 6, "oriented_orientations_lps": 6,
        "native_shapes": 2, "oriented_shapes": 2,
    }
    for field, width in vector_fields.items():
        values = geometry[field]
        if not isinstance(values, list) or len(values) != depth:
            raise ValueError(f"{field} must contain one entry per slice")
        present = [value for value in values if value is not None]
        if present:
            array = np.asarray(present, dtype=np.float64)
            if array.shape != (len(present), width) or not np.isfinite(array).all():
                raise ValueError(f"invalid values in {field}")
            if ("spacings" in field or "shapes" in field) and (array <= 0).any():
                raise ValueError(f"non-positive values in {field}")
        if "shapes" in field and len(present) != depth:
            raise ValueError(f"missing slice shape in {field}")
    for field in ("source_frames", "slice_positions_mm", "slice_thicknesses_mm"):
        if not isinstance(geometry[field], list) or len(geometry[field]) != depth:
            raise ValueError(f"{field} must contain one entry per slice")
    if not isinstance(geometry["orientation_standardized"], bool):
        raise ValueError("orientation_standardized must be a boolean")
    if str(row["orientation_standardized"]).lower() not in {"true", "false"}:
        raise ValueError("invalid orientation flag in manifest")
    if geometry["orientation_standardized"] != (str(row["orientation_standardized"]).lower() == "true"):
        raise ValueError("geometry and manifest disagree on orientation")
    if geometry["orientation_standardized"] and any(value is None for value in geometry["oriented_orientations_lps"]):
        raise ValueError("standardized orientation is missing direction vectors")
    spacing = geometry["slice_spacing_mm"]
    row_spacing = row["slice_spacing_mm"]
    if spacing is None:
        if not pd.isna(row_spacing):
            raise ValueError("geometry and manifest disagree on slice spacing")
    elif not np.isfinite(float(spacing)) or float(spacing) <= 0 or not np.isclose(float(spacing), float(row_spacing)):
        raise ValueError("invalid or inconsistent slice spacing")
    if geometry["ordering"] == "physical_position":
        positions = np.asarray(geometry["slice_positions_mm"], dtype=np.float64)
        if not np.isfinite(positions).all() or (depth > 1 and (np.diff(positions) <= 0).any()):
            raise ValueError("physical slice positions must be finite and strictly increasing")
        if depth > 1 and (spacing is None or not np.isclose(float(spacing), np.median(np.diff(positions)))):
            raise ValueError("slice spacing disagrees with physical slice positions")
    if processing["size"] != size or processing["dtype"] != "uint16":
        raise ValueError("processing settings disagree with the cached array")
    normalization = processing["normalization"]
    if normalization["method"] != "volume_percentile" or normalization["percentiles"] != [0.5, 99.5]:
        raise ValueError("unsupported intensity normalization")
    low, high = float(normalization["low"]), float(normalization["high"])
    if not np.isfinite([low, high]).all() or high <= low:
        raise ValueError("invalid intensity normalization range")
    transforms = processing["resize_transforms"]
    if not isinstance(transforms, list) or len(transforms) != depth:
        raise ValueError("resize transforms must contain one entry per slice")
    for transform in transforms:
        height, width = transform["resized_shape"]
        top, left = transform["padding_top_left"]
        if not (0 < height <= size and 0 < width <= size and 0 <= top <= size - height and 0 <= left <= size - width):
            raise ValueError("resize or padding extends outside the cached image")


def validate_manifest(manifest: str | Path, expected_size: int | None = None,
                      verify_checksums: bool = False,
                      study_ids: list[str] | None = None) -> dict[str, Any]:
    """Validate selected cached series before training or prediction.

    Always verifies the small metadata files, their fingerprints, current
    preprocessing version, geometry, NumPy headers, and exact file lengths.
    Pixel bytes are read only with verify_checksums=True, which detects damage
    inside an otherwise valid array. Raw DICOM files need not be present; rerun
    prepare_data to check whether those source files changed since preparation.
    This function does not alter any files or rebuild a stale cache.
    """
    manifest = Path(manifest).resolve()
    if expected_size is not None and (isinstance(expected_size, bool) or int(expected_size) != expected_size or expected_size < 8):
        raise ValueError("expected_size must be an integer of at least 8")
    required = _REQUIRED + [
        "cache_path", "metadata_path", "num_slices", "image_size", "slice_spacing_mm",
        "slice_ordering", "orientation_standardized", "source_fingerprint",
        "preprocess_fingerprint", "cache_sha256", "metadata_sha256",
    ]
    text_fields = [STUDY_ID, SERIES_ID, "cache_path", "metadata_path", "source_fingerprint",
                   "preprocess_fingerprint", "cache_sha256", "metadata_sha256"]
    table = pd.read_csv(manifest, dtype={field: str for field in text_fields})
    missing = set(required) - set(table.columns)
    if missing:
        raise ValueError(f"Manifest is missing cache validation columns: {sorted(missing)}")
    if study_ids is not None:
        requested = set(map(str, study_ids))
        missing_studies = requested - set(table[STUDY_ID])
        if missing_studies:
            raise ValueError(f"Requested studies have no manifest entries: {sorted(missing_studies)[:5]}")
        table = table.loc[table[STUDY_ID].isin(requested)]
    if table.empty:
        raise ValueError("No cached studies selected for validation")
    non_null = [column for column in required if column != "slice_spacing_mm"]
    if table[non_null].isna().any().any():
        raise ValueError("Selected manifest rows contain missing identifiers or cache metadata")
    if table.duplicated([STUDY_ID, SERIES_ID]).any():
        raise ValueError("Selected manifest rows contain duplicate study/series identifiers")
    if table["cache_path"].duplicated().any() or table["metadata_path"].duplicated().any():
        raise ValueError("Different manifest rows refer to the same cache or metadata file")
    sizes: set[int] = set()
    total_slices, total_bytes = 0, 0
    for row in table.to_dict(orient="records"):
        context = f"{row[STUDY_ID]}/{row[SERIES_ID]}"
        try:
            for flag in ("Fluid_Sensitive", "Fat_Suppression"):
                if row[flag] not in (0, 1):
                    raise ValueError(f"{flag} must be zero or one")
            if row["Anatomical_Plane"] not in PLANES:
                raise ValueError("unknown anatomical plane")
            depth, size = int(row["num_slices"]), int(row["image_size"])
            if depth < 1 or depth != row["num_slices"] or size < 8 or size != row["image_size"]:
                raise ValueError("invalid slice count or image size")
            if expected_size is not None and size != expected_size:
                raise ValueError(f"cache image size {size} differs from expected size {expected_size}")
            paths = []
            for field in ("cache_path", "metadata_path"):
                path = Path(row[field])
                if not path.is_absolute():
                    path = manifest.parent / path
                paths.append(path)
            cache_path, metadata_path = paths
            if cache_path.suffix != ".npy" or metadata_path != cache_path.with_suffix(".json"):
                raise ValueError("cache_path and metadata_path must identify the same .npy/.json pair")
            metadata_bytes = metadata_path.read_bytes()
            if hashlib.sha256(metadata_bytes).hexdigest() != row["metadata_sha256"]:
                raise ValueError("metadata checksum differs from manifest; the cache or manifest is stale")
            metadata = json.loads(metadata_bytes)
            if not isinstance(metadata, dict) or metadata.get("metadata_payload_sha256") != _metadata_digest(metadata):
                raise ValueError("metadata payload checksum mismatch")
            for field in (STUDY_ID, SERIES_ID, "source_fingerprint", "preprocess_fingerprint", "cache_sha256"):
                if metadata.get(field) != row[field]:
                    raise ValueError(f"metadata and manifest disagree on {field}")
            for field in ("source_fingerprint", "preprocess_fingerprint", "cache_sha256", "metadata_sha256"):
                value = row[field]
                if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
                    raise ValueError(f"invalid SHA-256 value in {field}")
            settings = metadata["preprocessing"]
            if settings["version"] != PREPROCESS_VERSION:
                raise ValueError(f"preprocessing version {settings['version']!r} differs from current {PREPROCESS_VERSION!r}")
            expected_settings = {"version": PREPROCESS_VERSION, "size": size,
                                 "anatomical_plane": row["Anatomical_Plane"]}
            expected_fingerprint = hashlib.sha256(json.dumps(expected_settings, sort_keys=True).encode()).hexdigest()
            if settings != expected_settings or metadata["preprocess_fingerprint"] != expected_fingerprint:
                raise ValueError("preprocessing settings or fingerprint disagree with the manifest")
            expected_shape = [depth, size, size]
            if metadata["shape"] != expected_shape:
                raise ValueError("metadata shape disagrees with manifest")
            _validate_geometry(metadata, row, depth, size)
            volume = np.load(cache_path, mmap_mode="r", allow_pickle=False)
            try:
                if not isinstance(volume, np.memmap) or volume.dtype != np.uint16 or list(volume.shape) != expected_shape:
                    raise ValueError("cached array header has the wrong dtype or shape")
                file_bytes = cache_path.stat().st_size
                if file_bytes != volume.offset + volume.nbytes:
                    raise ValueError("cached array is truncated or has unexpected trailing bytes")
            finally:
                if isinstance(volume, np.memmap):
                    volume._mmap.close()
            if verify_checksums and _sha256_file(cache_path) != row["cache_sha256"]:
                raise ValueError("cached array checksum mismatch")
            sizes.add(size)
            total_slices += depth
            total_bytes += file_bytes
        except (OSError, ValueError, TypeError, KeyError, AttributeError, OverflowError) as exc:
            raise ValueError(f"Invalid cache for {context}: {exc}") from exc
    if len(sizes) != 1:
        raise ValueError(f"Selected manifest rows contain different cache image sizes: {sorted(sizes)}")
    return {"studies": int(table[STUDY_ID].nunique()), "series": len(table), "slices": total_slices,
            "image_size": sizes.pop(), "preprocess_version": PREPROCESS_VERSION,
            "cache_bytes": total_bytes, "array_checksums_verified": bool(verify_checksums)}


def _valid_cache(cache_path: Path, metadata_path: Path, expected: dict[str, Any]) -> tuple[dict[str, Any] | None, str | None]:
    if not cache_path.exists() and not metadata_path.exists():
        return None, None
    try:
        with metadata_path.open() as handle:
            metadata = json.load(handle)
        if not isinstance(metadata, dict) or metadata.get("metadata_payload_sha256") != _metadata_digest(metadata):
            return None, "cache_rebuilt:metadata_checksum_mismatch"
        for key, value in expected.items():
            if metadata.get(key) != value:
                return None, f"cache_rebuilt:{key}_changed"
        volume = np.load(cache_path, mmap_mode="r", allow_pickle=False)
        if volume.dtype != np.uint16 or list(volume.shape) != metadata.get("shape") or volume.ndim != 3 or volume.shape[0] < 1:
            return None, "cache_rebuilt:shape_or_dtype_invalid"
        # Also catches truncated arrays, extra trailing bytes, and pixel damage.
        del volume
        if _sha256_file(cache_path) != metadata.get("cache_sha256"):
            return None, "cache_rebuilt:checksum_mismatch"
        return metadata, None
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
        return None, f"cache_rebuilt:invalid_cache:{type(exc).__name__}"


def prepare_one_series(series_dir: Path, cache_path: Path, *, study_id: str,
                       series_id: str, anatomical_plane: str, size: int = 320) -> dict[str, Any]:
    """Prepare one series atomically, or verify and reuse an identical cache.

    Metadata is written last, so an interrupted pair of files will fail validation
    and be rebuilt. Both the preprocessing settings and source file stats must
    match; cached array bytes are checked against a stored SHA-256 checksum.
    """
    series_dir, cache_path = Path(series_dir), Path(cache_path)
    if cache_path.suffix != ".npy":
        raise ValueError("cache_path must end in .npy")
    source_hash, files = source_fingerprint(series_dir)
    settings = {"version": PREPROCESS_VERSION, "size": int(size), "anatomical_plane": anatomical_plane}
    preprocess_hash = hashlib.sha256(json.dumps(settings, sort_keys=True).encode()).hexdigest()
    expected = {"preprocess_fingerprint": preprocess_hash, "source_fingerprint": source_hash,
                STUDY_ID: study_id, SERIES_ID: series_id}
    metadata_path = cache_path.with_suffix(".json")
    metadata, rebuild_warning = _valid_cache(cache_path, metadata_path, expected)
    reused = metadata is not None
    if metadata is None:
        images, geometry = load_dicom_series(series_dir, anatomical_plane)
        volume, processing = prepare_volume(images, size, geometry["pixel_spacings_mm"])
        # Avoid committing a cache while its source files are being modified.
        end_hash, _ = source_fingerprint(series_dir)
        if end_hash != source_hash:
            raise RuntimeError("Source files changed during preparation; retry after the download finishes")
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary = tempfile.mkstemp(prefix=f".{cache_path.name}.", suffix=".tmp", dir=cache_path.parent)
        try:
            with os.fdopen(descriptor, "wb") as handle:
                np.save(handle, volume, allow_pickle=False)
                handle.flush()
                os.fsync(handle.fileno())
            cache_hash = _sha256_file(Path(temporary))
            os.replace(temporary, cache_path)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
        metadata = {**expected, "preprocessing": settings, "shape": list(volume.shape),
                    "cache_sha256": cache_hash, "source_file_count": len(files),
                    "geometry": geometry, "processing": processing}
        metadata["metadata_payload_sha256"] = _metadata_digest(metadata)
        _atomic_json(metadata_path, metadata)
    warnings = list(metadata["geometry"]["warnings"])
    if rebuild_warning:
        warnings.append(rebuild_warning)
    return {"num_slices": metadata["shape"][0], "image_size": metadata["shape"][1],
            "slice_spacing_mm": metadata["geometry"]["slice_spacing_mm"],
            "slice_ordering": metadata["geometry"]["ordering"],
            "orientation_standardized": metadata["geometry"]["orientation_standardized"],
            "source_fingerprint": source_hash, "preprocess_fingerprint": preprocess_hash,
            "cache_sha256": metadata["cache_sha256"],
            "metadata_sha256": _sha256_file(metadata_path),
            "status": "reused" if reused else "prepared", "warnings": warnings}


def _prepare_job(job: dict[str, Any]) -> dict[str, Any]:
    row = job["row"]
    try:
        result = prepare_one_series(Path(job["series_dir"]), Path(job["cache_path"]),
                                    study_id=row[STUDY_ID], series_id=row[SERIES_ID],
                                    anatomical_plane=row["Anatomical_Plane"], size=job["size"])
        return {**row, **result, "cache_path": job["relative_cache_path"],
                "metadata_path": str(Path(job["relative_cache_path"]).with_suffix(".json")), "error": None}
    except Exception as exc:
        return {**row, "status": "failed", "warnings": [],
                "error": f"{type(exc).__name__}: {exc}"}


@contextlib.contextmanager
def _preparation_lock(output_dir: Path):
    with (output_dir / ".prepare.lock").open("a") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"Another preparation process is using {output_dir}") from exc
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def prepare_data(data_root: Path, output_dir: Path, split: str = "train", size: int = 320,
                 workers: int = 8, study_ids: list[str] | None = None,
                 limit_studies: int | None = None, strict: bool = False) -> Path:
    """Write manifest.csv and QC reports for the selected train or test studies.

    A failed series is recorded in qc.json and qc.csv. If any selected study has
    no usable series, or strict=True and any series failed, raise after writing
    the reports and leave any previous manifest untouched. A subset run replaces
    the manifest with that subset but never removes existing image caches.
    """
    if split not in {"train", "test"}:
        raise ValueError("split must be 'train' or 'test'")
    if size < 8 or workers < 1:
        raise ValueError("size must be at least 8 and workers must be positive")
    if limit_studies is not None and limit_studies < 1:
        raise ValueError("limit_studies must be positive")
    data_root, output_dir = Path(data_root).resolve(), Path(output_dir).resolve()
    table = pd.read_csv(data_root / f"{split}_series.csv", dtype={STUDY_ID: str, SERIES_ID: str})
    missing = set(_REQUIRED) - set(table.columns)
    if missing:
        raise ValueError(f"Series table is missing columns: {sorted(missing)}")
    if table[_REQUIRED].isna().any().any():
        raise ValueError("Series table contains missing identifiers or metadata")
    if table.duplicated([STUDY_ID, SERIES_ID]).any():
        raise ValueError("Series table contains duplicate study/series identifiers")
    for name in ("Fluid_Sensitive", "Fat_Suppression"):
        if not table[name].isin([0, 1]).all():
            raise ValueError(f"{name} must contain only 0 or 1")
        table[name] = table[name].astype(int)
    if not table["Anatomical_Plane"].isin(PLANES).all():
        bad = sorted(set(table["Anatomical_Plane"]) - set(PLANES))
        raise ValueError(f"Unknown anatomical plane values: {bad}")
    studies_path = data_root / f"{split}.csv"
    if studies_path.exists():
        studies = pd.read_csv(studies_path, usecols=[STUDY_ID], dtype={STUDY_ID: str})
        if studies[STUDY_ID].isna().any() or studies[STUDY_ID].duplicated().any():
            raise ValueError(f"{split}.csv has missing or duplicate study identifiers")
        available = sorted(studies[STUDY_ID].tolist())
        unknown_in_table = set(table[STUDY_ID]) - set(available)
        if unknown_in_table:
            raise ValueError(f"Series table contains studies absent from {split}.csv: {sorted(unknown_in_table)[:5]}")
    else:
        available = sorted(table[STUDY_ID].unique().tolist())
    if study_ids is not None:
        requested = set(map(str, study_ids))
        unknown = requested - set(available)
        if unknown:
            raise ValueError(f"Requested studies are absent from {split}.csv: {sorted(unknown)[:5]}")
        selected = [study for study in available if study in requested]
    else:
        selected = available
    if limit_studies is not None:
        selected = selected[:limit_studies]
    if not selected:
        raise ValueError("No studies selected")
    table = table.loc[table[STUDY_ID].isin(selected), _REQUIRED].sort_values([STUDY_ID, SERIES_ID])
    for study in selected:
        _safe_component(study, STUDY_ID)
    for series in table[SERIES_ID]:
        _safe_component(series, SERIES_ID)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.csv"
    with _preparation_lock(output_dir):
        jobs = []
        inventory_warnings = []
        for study in selected:
            study_dir = data_root / f"{split}_series" / study
            listed = set(table.loc[table[STUDY_ID] == study, SERIES_ID])
            if study_dir.is_dir():
                for path in sorted(study_dir.iterdir()):
                    if path.is_dir() and path.name not in listed:
                        inventory_warnings.append({STUDY_ID: study, SERIES_ID: path.name,
                                                   "warning": "series_directory_not_in_metadata_table"})
        for row in table.to_dict(orient="records"):
            relative = Path("series") / row[STUDY_ID] / f"{row[SERIES_ID]}.npy"
            jobs.append({"row": row, "series_dir": str(data_root / f"{split}_series" / row[STUDY_ID] / row[SERIES_ID]),
                         "cache_path": str(output_dir / relative), "relative_cache_path": str(relative), "size": size})
        started = time.monotonic()

        def collect(iterator):
            results = []
            failures = 0
            for index, result in enumerate(iterator, 1):
                results.append(result)
                failures += result["status"] == "failed"
                if index % 500 == 0 or index == len(jobs):
                    progress = {"finished_series": index, "total_series": len(jobs),
                                "failed_series": failures, "seconds": time.monotonic() - started}
                    _atomic_json(output_dir / "progress.json", progress)
                    if len(jobs) >= 500:
                        print(json.dumps(progress), flush=True)
            return results

        if workers == 1:
            results = collect(map(_prepare_job, jobs))
        else:
            # Processes avoid decoder/thread oversubscription; results are stable
            # in table order even when jobs finish at different times.
            with concurrent.futures.ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn")) as pool:
                results = collect(pool.map(_prepare_job, jobs, chunksize=1))
        usable = [row for row in results if row["status"] != "failed"]
        failed = [row for row in results if row["status"] == "failed"]
        missing_studies = sorted(set(selected) - {row[STUDY_ID] for row in usable})
        report = {"preprocess_version": PREPROCESS_VERSION, "data_root": str(data_root), "split": split,
                  "size": size, "selected_studies": selected,
                  "summary": {"selected_studies": len(selected), "expected_series": len(results),
                              "usable_series": len(usable), "failed_series": len(failed),
                              "prepared_series": sum(row["status"] == "prepared" for row in results),
                              "reused_series": sum(row["status"] == "reused" for row in results),
                              "series_with_warnings": sum(bool(row["warnings"]) for row in results),
                              "studies_without_usable_series": len(missing_studies)},
                  "studies_without_usable_series": missing_studies,
                  "inventory_warnings": inventory_warnings, "series": results}
        _atomic_json(output_dir / "qc.json", report)
        qc_rows = [{**row, "warnings": "|".join(row["warnings"])} for row in results]
        _atomic_csv(output_dir / "qc.csv", pd.DataFrame(qc_rows, columns=list(dict.fromkeys(_REQUIRED + [key for row in qc_rows for key in row]))))
        if missing_studies or (strict and failed):
            raise RuntimeError(f"Preparation failed: {len(failed)} unusable series; {len(missing_studies)} studies have no usable series. "
                               f"See {output_dir / 'qc.json'}. Existing manifest was not replaced.")
        manifest_rows = [{key: ("|".join(value) if key == "warnings" else value)
                          for key, value in row.items() if key != "error"} for row in usable]
        _atomic_csv(manifest_path, pd.DataFrame(manifest_rows))
    return manifest_path
