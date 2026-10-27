"""Read MRI slices without losing their geometry or cropping the knee.

Arrays use rows then columns. The canonical directions are expressed in DICOM's
LPS coordinates (positive x = left, y = posterior, z = superior): sagittal
columns point posterior and rows inferior; coronal columns point left and rows
inferior; axial columns point left and rows posterior. We only transpose/flip
pixels. Oblique scans keep their obliquity and their actual orientation is saved.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pydicom
from PIL import Image

PREPROCESS_VERSION = "knee-dicom-v1"
_CANONICAL = {
    "Sagittal": (np.array([0., 1., 0.]), np.array([0., 0., -1.]), np.array([1., 0., 0.])),
    "Coronal": (np.array([1., 0., 0.]), np.array([0., 0., -1.]), np.array([0., 1., 0.])),
    "Axial": (np.array([1., 0., 0.]), np.array([0., 1., 0.]), np.array([0., 0., 1.])),
}


class DicomSeriesError(ValueError):
    """A series cannot safely be turned into a volume."""


def _vector(value: Any, length: int) -> np.ndarray | None:
    try:
        vector = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError):
        return None
    if vector.shape != (length,) or not np.isfinite(vector).all():
        return None
    return vector


def _functional_value(ds: Any, frame: int, sequence: str, name: str) -> Any:
    """Read enhanced-MR per-frame values before shared or classic attributes."""
    per_frame = getattr(ds, "PerFrameFunctionalGroupsSequence", ())
    shared = getattr(ds, "SharedFunctionalGroupsSequence", ())
    groups = ([per_frame[frame]] if frame < len(per_frame) else []) + list(shared[:1])
    for group in groups:
        items = getattr(group, sequence, ())
        if items and hasattr(items[0], name):
            return getattr(items[0], name)
    return getattr(ds, name, None)


def source_fingerprint(series_dir: Path) -> tuple[str, list[Path]]:
    """Fingerprint file names, sizes and nanosecond modification times.

    Source pixels are read once during preparation; this stat fingerprint avoids
    reading the entire raw dataset again for every resumable run.
    """
    series_dir = Path(series_dir)
    if not series_dir.is_dir():
        raise DicomSeriesError(f"Series directory does not exist: {series_dir}")
    files = sorted(p for p in series_dir.iterdir() if p.is_file() and p.suffix.lower() in {".dcm", ".ima"})
    if not files:
        raise DicomSeriesError(f"No DICOM files in {series_dir}")
    records = []
    for path in files:
        stat = path.stat()
        records.append([path.name, stat.st_size, stat.st_mtime_ns])
    payload = json.dumps(records, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest(), files


def _orientation_operation(iop: np.ndarray, plane: str) -> tuple[bool, bool, bool, float]:
    target_col, target_row, _ = _CANONICAL[plane]
    old_col, old_row = iop[:3], iop[3:]
    candidates = []
    for transpose in (False, True):
        col, row = (old_row, old_col) if transpose else (old_col, old_row)
        col_score, row_score = float(col @ target_col), float(row @ target_row)
        candidates.append((min(abs(col_score), abs(row_score)), transpose, row_score < 0, col_score < 0))
    score, transpose, flip_rows, flip_cols = max(candidates)
    return transpose, flip_rows, flip_cols, score


def _reorient_frame(frame: dict[str, Any], operation: tuple[bool, bool, bool, float]) -> None:
    transpose, flip_rows, flip_cols, _ = operation
    image = frame["image"]
    iop = frame["orientation"]
    col, row = iop[:3].copy(), iop[3:].copy()
    position = None if frame["position"] is None else frame["position"].copy()
    spacing = frame["spacing"]
    row_spacing, col_spacing = (1., 1.) if spacing is None else spacing
    if transpose:
        image = image.T
        col, row = row, col
        row_spacing, col_spacing = col_spacing, row_spacing
    if flip_rows:
        if position is not None and spacing is not None:
            position += row * row_spacing * (image.shape[0] - 1)
        row = -row
        image = image[::-1]
    if flip_cols:
        if position is not None and spacing is not None:
            position += col * col_spacing * (image.shape[1] - 1)
        col = -col
        image = image[:, ::-1]
    # Missing spacing means the new upper-left position cannot be reconstructed.
    if spacing is None and (flip_rows or flip_cols):
        position = None
    frame.update(image=image, oriented_orientation=np.concatenate([col, row]),
                 oriented_position=position,
                 oriented_spacing=None if spacing is None else [row_spacing, col_spacing])


def load_dicom_series(series_dir: Path, anatomical_plane: str) -> tuple[list[np.ndarray], dict[str, Any]]:
    """Decode every frame, physically order slices and orient the in-plane axes.

    Native frame shapes are retained in the returned list. Unsupported or
    ambiguous volumes raise DicomSeriesError; every geometry fallback is recorded
    in metadata["warnings"]. No unreadable slice is silently dropped.
    """
    if anatomical_plane not in _CANONICAL:
        raise DicomSeriesError(f"Unsupported anatomical plane: {anatomical_plane!r}")
    _, files = source_fingerprint(series_dir)
    frames: list[dict[str, Any]] = []
    warnings: list[str] = []
    transfer_syntaxes: set[str] = set()
    for path in files:
        try:
            ds = pydicom.dcmread(path)
            pixels = ds.pixel_array
        except Exception as exc:
            raise DicomSeriesError(f"Could not decode {path.name}: {type(exc).__name__}: {exc}") from exc
        photometric = str(getattr(ds, "PhotometricInterpretation", ""))
        if int(getattr(ds, "SamplesPerPixel", 1)) != 1 or photometric not in {"MONOCHROME1", "MONOCHROME2"}:
            raise DicomSeriesError(f"Unsupported pixel type in {path.name}: {photometric}")
        if pixels.ndim == 2:
            pixels = pixels[None, ...]
        if pixels.ndim != 3:
            raise DicomSeriesError(f"Unexpected pixel dimensions in {path.name}: {pixels.shape}")
        if pixels.shape[0] > 1 and not hasattr(ds, "PerFrameFunctionalGroupsSequence"):
            raise DicomSeriesError(f"Multi-frame file {path.name} has no per-frame geometry")
        transfer_syntaxes.add(str(getattr(ds.file_meta, "TransferSyntaxUID", "unknown")))
        for index, raw in enumerate(pixels):
            image = np.asarray(raw, dtype=np.float32)
            padding = getattr(ds, "PixelPaddingValue", None)
            padding_mask = None
            if padding is not None:
                padding_end = getattr(ds, "PixelPaddingRangeLimit", padding)
                low, high = sorted((float(padding), float(padding_end)))
                padding_mask = (image >= low) & (image <= high)
            if hasattr(ds, "ModalityLUTSequence"):
                image = pydicom.pixels.apply_modality_lut(raw, ds).astype(np.float32)
            else:
                slope = _functional_value(ds, index, "PixelValueTransformationSequence", "RescaleSlope")
                intercept = _functional_value(ds, index, "PixelValueTransformationSequence", "RescaleIntercept")
                slope = 1. if slope is None else float(slope)
                intercept = 0. if intercept is None else float(intercept)
                if not math.isfinite(slope) or slope == 0 or not math.isfinite(intercept):
                    raise DicomSeriesError(f"Invalid rescale values in {path.name}")
                image = image * slope + intercept
            if padding_mask is not None:
                image[padding_mask] = np.nan
            orientation = _vector(_functional_value(ds, index, "PlaneOrientationSequence", "ImageOrientationPatient"), 6)
            if orientation is not None:
                col, row = orientation[:3], orientation[3:]
                if abs(np.linalg.norm(col) - 1) > .02 or abs(np.linalg.norm(row) - 1) > .02 or abs(col @ row) > .02:
                    raise DicomSeriesError(f"Invalid direction cosines in {path.name}")
                orientation = np.r_[col / np.linalg.norm(col), row / np.linalg.norm(row)]
            position = _vector(_functional_value(ds, index, "PlanePositionSequence", "ImagePositionPatient"), 3)
            spacing = _vector(_functional_value(ds, index, "PixelMeasuresSequence", "PixelSpacing"), 2)
            if spacing is not None and np.any(spacing <= 0):
                spacing = None
            instance = getattr(ds, "InstanceNumber", None)
            try:
                instance = float(instance) if instance is not None else None
            except (TypeError, ValueError):
                instance = None
            if instance is not None and not math.isfinite(instance):
                instance = None
            thickness = _functional_value(ds, index, "PixelMeasuresSequence", "SliceThickness")
            frames.append({"image": image, "original_shape": list(image.shape),
                           "orientation": orientation, "original_orientation": orientation,
                           "position": position,
                           "spacing": spacing, "instance": instance, "frame_index": index,
                           "source": path.name, "sop_uid": str(getattr(ds, "SOPInstanceUID", "")),
                           "photometric": photometric,
                           "slice_thickness": float(thickness) if thickness is not None else None})
    if not frames:
        raise DicomSeriesError(f"No image frames in {series_dir}")
    valid_images = [f["image"][np.isfinite(f["image"])] for f in frames]
    nonempty = [a for a in valid_images if a.size]
    if not nonempty:
        raise DicomSeriesError("Every pixel is invalid or marked as padding")
    global_min, global_max = min(float(a.min()) for a in nonempty), max(float(a.max()) for a in nonempty)
    if global_min == global_max:
        raise DicomSeriesError("The entire series has constant intensity")
    inverse_count = 0
    for frame in frames:
        if frame["photometric"] == "MONOCHROME1":
            frame["image"] = global_min + global_max - frame["image"]
            inverse_count += 1
    if 0 < inverse_count < len(frames):
        warnings.append("mixed_photometric_interpretation")
    blank_count = sum(a.size == 0 or float(a.min()) == float(a.max()) for a in valid_images)
    if blank_count:
        warnings.append(f"constant_or_padding_only_slices:{blank_count}")
    orientations = [frame["orientation"] for frame in frames if frame["orientation"] is not None]
    reference = orientations[0] if orientations else None
    if reference is not None and any(not np.allclose(value, reference, atol=.015, rtol=0) for value in orientations):
        raise DicomSeriesError("Mixed slice orientations: cannot safely treat this series as one volume")
    if reference is None:
        warnings.append("missing_orientation:no_in_plane_standardization")
    elif len(orientations) != len(frames):
        warnings.append("missing_orientation:inferred_from_other_slices")
        for frame in frames:
            if frame["orientation"] is None:
                frame["orientation"] = reference.copy()
    all_positions = all(frame["position"] is not None for frame in frames)
    _, _, canonical_depth = _CANONICAL[anatomical_plane]
    normal = None
    if reference is not None:
        normal = np.cross(reference[:3], reference[3:])
        normal /= np.linalg.norm(normal)
        if normal @ canonical_depth < 0:
            normal = -normal
        if abs(float(normal @ canonical_depth)) < math.cos(math.radians(35)):
            warnings.append("plane_label_and_geometry_disagree_or_highly_oblique")
    if normal is not None and all_positions:
        ordering = "physical_position"
        for frame in frames:
            frame["slice_position"] = float(frame["position"] @ normal)
        frames.sort(key=lambda frame: (frame["slice_position"], frame["source"], frame["frame_index"]))
    elif all(frame["instance"] is not None for frame in frames):
        ordering = "instance_number"
        warnings.append("slice_order_fallback:InstanceNumber")
        frames.sort(key=lambda frame: (frame["instance"], frame["source"], frame["frame_index"]))
    else:
        ordering = "file_name"
        warnings.append("slice_order_fallback:file_name")
        frames.sort(key=lambda frame: (frame["source"], frame["frame_index"]))
    deduplicated: list[dict[str, Any]] = []
    seen_sops: dict[tuple[str, int], dict[str, Any]] = {}
    dropped = []
    for frame in frames:
        duplicate = None
        if ordering == "physical_position" and deduplicated and abs(frame["slice_position"] - deduplicated[-1]["slice_position"]) < 1e-3:
            duplicate = deduplicated[-1]
        sop_key = (frame["sop_uid"], frame["frame_index"])
        if frame["sop_uid"] and sop_key in seen_sops:
            duplicate = seen_sops[sop_key]
        if duplicate is not None:
            if frame["image"].shape != duplicate["image"].shape or not np.array_equal(frame["image"], duplicate["image"], equal_nan=True):
                raise DicomSeriesError("Duplicate slice position or SOP UID has different pixels; possible repeated echoes/acquisitions")
            dropped.append({"source": frame["source"], "frame_index": frame["frame_index"]})
            continue
        deduplicated.append(frame)
        if frame["sop_uid"]:
            seen_sops[sop_key] = frame
    frames = deduplicated
    if dropped:
        warnings.append(f"identical_duplicate_slices_removed:{len(dropped)}")
    operation = (False, False, False, 0.)
    standardized = False
    if reference is not None:
        operation = _orientation_operation(reference, anatomical_plane)
        if operation[3] >= math.cos(math.radians(35)):
            standardized = True
            if operation[3] < math.cos(math.radians(5)):
                warnings.append("oblique_geometry_preserved_without_rotating_pixels")
            for frame in frames:
                _reorient_frame(frame, operation)
        else:
            warnings.append("orientation_not_standardized:high_obliquity_or_plane_mismatch")
    if not standardized:
        for frame in frames:
            frame.update(oriented_orientation=frame["orientation"], oriented_position=frame["position"], oriented_spacing=frame["spacing"])
    shapes = [list(frame["image"].shape) for frame in frames]
    if len({tuple(shape) for shape in shapes}) > 1:
        warnings.append("mixed_frame_sizes:each_frame_resized_with_padding")
    spacings = [frame["oriented_spacing"] for frame in frames]
    if any(spacing is None for spacing in spacings):
        warnings.append("missing_pixel_spacing:assumed_square_pixels_for_resize")
    positions = [frame.get("slice_position") for frame in frames]
    slice_spacing = None
    if ordering == "physical_position" and len(frames) > 1:
        differences = np.diff(positions)
        slice_spacing = float(np.median(differences))
        if not np.allclose(differences, slice_spacing, rtol=.05, atol=.05):
            warnings.append("nonuniform_slice_spacing")
    metadata = {
        "anatomical_plane": anatomical_plane,
        "num_source_files": len(files), "num_slices": len(frames),
        "ordering": ordering, "slice_spacing_mm": slice_spacing,
        "slice_positions_mm": positions,
        "slice_normal_lps": None if normal is None else normal.tolist(),
        "pixel_spacings_mm": [None if v is None else list(map(float, v)) for v in spacings],
        "original_pixel_spacings_mm": [None if f["spacing"] is None else f["spacing"].tolist() for f in frames],
        "image_positions_lps": [None if f["position"] is None else f["position"].tolist() for f in frames],
        "oriented_image_positions_lps": [None if f["oriented_position"] is None else f["oriented_position"].tolist() for f in frames],
        "original_orientations_lps": [None if f["original_orientation"] is None else f["original_orientation"].tolist() for f in frames],
        "oriented_orientations_lps": [None if f["oriented_orientation"] is None else f["oriented_orientation"].tolist() for f in frames],
        "slice_thicknesses_mm": [f["slice_thickness"] for f in frames],
        "native_shapes": [f["original_shape"] for f in frames],
        "oriented_shapes": shapes,
        "source_frames": [{"file": f["source"], "frame": f["frame_index"], "instance_number": f["instance"]} for f in frames],
        "dropped_duplicate_frames": dropped,
        "monochrome1_frames_inverted": inverse_count,
        "orientation_standardized": standardized,
        "orientation_operation": {"transpose": operation[0] if standardized else False,
                                  "flip_rows": operation[1] if standardized else False,
                                  "flip_columns": operation[2] if standardized else False},
        "transfer_syntaxes": sorted(transfer_syntaxes), "warnings": warnings,
    }
    return [f["image"] for f in frames], metadata


def prepare_volume(images: Sequence[np.ndarray], size: int = 320,
                   pixel_spacings: Sequence[Sequence[float] | None] | None = None) -> tuple[np.ndarray, dict[str, Any]]:
    """Clip using whole-volume percentiles, then resize and pad every full slice.

    No center crop is used. Physical in-plane aspect ratio is preserved when
    PixelSpacing is available. uint16 keeps more intensity detail than uint8;
    training can divide by 65535 to get values between zero and one.
    """
    if size < 8:
        raise ValueError("size must be at least 8 pixels")
    if not len(images):
        raise DicomSeriesError("Cannot prepare an empty series")
    if pixel_spacings is None:
        pixel_spacings = [None] * len(images)
    if len(pixel_spacings) != len(images):
        raise ValueError("There must be one pixel spacing entry per image")
    # A deterministic sample bounds memory even for unusually long series.
    total = sum(np.asarray(image).size for image in images)
    stride = max(1, math.ceil(total / 1_000_000))
    samples = [np.asarray(image).ravel()[::stride] for image in images]
    sample = np.concatenate(samples)
    sample = sample[np.isfinite(sample)]
    if sample.size == 0:
        raise DicomSeriesError("No finite pixels available for normalization")
    low, high = map(float, np.percentile(sample, [0.5, 99.5]))
    if high <= low:
        low, high = float(sample.min()), float(sample.max())
    if high <= low:
        raise DicomSeriesError("No intensity range available for normalization")
    volume = np.zeros((len(images), size, size), dtype=np.uint16)
    transforms = []
    for index, (image, spacing) in enumerate(zip(images, pixel_spacings)):
        image = np.asarray(image, dtype=np.float32)
        if image.ndim != 2 or min(image.shape) == 0:
            raise DicomSeriesError(f"Invalid slice shape: {image.shape}")
        normalized = np.clip((image - low) / (high - low), 0, 1)
        normalized = np.nan_to_num(normalized, nan=0., posinf=1., neginf=0.)
        row_spacing, col_spacing = (1., 1.) if spacing is None else map(float, spacing)
        if min(row_spacing, col_spacing) <= 0 or not np.isfinite([row_spacing, col_spacing]).all():
            raise DicomSeriesError(f"Invalid pixel spacing: {spacing}")
        physical_h, physical_w = image.shape[0] * row_spacing, image.shape[1] * col_spacing
        scale = size / max(physical_h, physical_w)
        height = max(1, min(size, int(round(physical_h * scale))))
        width = max(1, min(size, int(round(physical_w * scale))))
        resized = np.asarray(Image.fromarray(normalized).resize((width, height), Image.Resampling.BILINEAR))
        top, left = (size - height) // 2, (size - width) // 2
        volume[index, top:top + height, left:left + width] = np.rint(resized * 65535).astype(np.uint16)
        transforms.append({"resized_shape": [height, width], "padding_top_left": [top, left],
                           "resize_factor_rows": height / image.shape[0],
                           "resize_factor_columns": width / image.shape[1],
                           "output_pixel_spacing_mm": None if spacing is None else [physical_h / height, physical_w / width]})
    return volume, {"normalization": {"method": "volume_percentile", "percentiles": [0.5, 99.5],
                                      "low": low, "high": high, "sample_stride": stride},
                    "size": size, "dtype": "uint16", "resize_transforms": transforms}
