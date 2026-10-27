"""Raptor's scored image recipe, with audited caching and series-safe windows.

Adapted from Dread Development (Johnathan Wagner), Apache License 2.0:
https://www.kaggle.com/code/dreaddevelopment/knee-mri-twelve-findings-from-a-single-model?scriptVersionId=344173275
https://www.kaggle.com/code/dreaddevelopment/knee-mri-training-the-twelve-finding-model?scriptVersionId=347608187

Changes: portable paths, explicit failures, source/cache checksums, reproducible
sampling, and no three-channel window crossing MRI series or missing slices.
Pixels follow the scored recipe; our safer window selection is a deliberate
change, so this does not claim an exact reproduction of its leaderboard score.
"""

from __future__ import annotations

import concurrent.futures
import hashlib
import json
import os
import tempfile
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
import pydicom
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from .anchors import centred_box, project_to_pixel, read_anchor
from .constants import PLANES, RAPTOR_VARIANT_SIZES, SERIES_ID, SIDE_FEATURE, STUDY_ID, TARGETS, WINDOW_FEATURES
from .imaging import source_fingerprint
from .prepare import _atomic_csv, _atomic_json, _metadata_digest, _preparation_lock, _safe_component, _sha256_file

RAPTOR_PREPROCESS_VERSION = "raptor-widedense-series-safe-v1"
RAPTOR_SIZE = 336
RAPTOR_SLOTS = (("Sagittal", 1, 18), ("Sagittal", 0, 14), ("Coronal", 1, 12),
                ("Coronal", 0, 8), ("Axial", -1, 12))
RAPTOR_DEPTH = sum(slot[2] for slot in RAPTOR_SLOTS)
RAPTOR_SOURCE = (
    "https://www.kaggle.com/code/dreaddevelopment/knee-mri-twelve-findings-from-a-single-model"
    "?scriptVersionId=344173275"
)
RAPTOR_CONTRACT = {
    "version": RAPTOR_PREPROCESS_VERSION,
    "cache_size": RAPTOR_SIZE,
    "cache_dtype": "uint8",
    "slots": [list(slot) for slot in RAPTOR_SLOTS],
    "span": [0.06, 0.94],
    "crop_mm": 140.0,
    "crop_spacing": "PixelSpacing[0], square pixel crop as scored source",
    "normalization_percentiles": [2.0, 98.0],
    "normalization_precision": "NumPy percentile float64 scalars, matching the scored source",
    "percentiles_over": "selected slices of each series before cropping; repeated selections retained",
    "slice_order": "signed ImageOrientationPatient cross-product, without canonical depth reversal",
    "resize": "OpenCV INTER_AREA",
    "quantization": "clip01 then multiply255 then uint8 truncation",
    "windows": "three valid consecutive cache positions within one selected series",
    "source": RAPTOR_SOURCE,
}
RAPTOR_FINGERPRINT = hashlib.sha256(json.dumps(RAPTOR_CONTRACT, sort_keys=True).encode()).hexdigest()
# A denser variant keeps every source slice of the 6-94% span when it fits, so thin
# structures are not skipped between cached slices. Crop, intensity and size match.
RAPTOR_DENSE_VERSION = "raptor-dense-distinct-v1"
RAPTOR_DENSE_SLOTS = (("Sagittal", 1, 32), ("Sagittal", 0, 24), ("Coronal", 1, 32),
                      ("Coronal", 0, 24), ("Axial", -1, 32))
RAPTOR_DENSE_CONTRACT = {
    **RAPTOR_CONTRACT, "version": RAPTOR_DENSE_VERSION, "slots": [list(slot) for slot in RAPTOR_DENSE_SLOTS],
    "percentiles_over": "the distinct selected slices of each series before cropping",
    "slice_selection": "every source slice of the span when it fits the slot, otherwise evenly spaced distinct "
                       "slices rounded half up; unused slot positions stay empty and outside every window",
}
# The same dense layout and 140 mm crop stored at 448 px, so a central crop of the
# cache shows the middle of the knee at full detail instead of an enlarged 336 px image.
RAPTOR_DENSE_448_VERSION = "raptor-dense-distinct-448-v1"
RAPTOR_DENSE_448_CONTRACT = {**RAPTOR_DENSE_CONTRACT, "version": RAPTOR_DENSE_448_VERSION, "cache_size": 448}
# The same dense layout and 140 mm crop stored at 384 px, so a 384 px model sees stored detail instead of 336 px
# slices enlarged to 384 (the public Raptor model's "native 384" input).
RAPTOR_DENSE_384_VERSION = "raptor-dense-distinct-384-v1"
RAPTOR_DENSE_384_CONTRACT = {**RAPTOR_DENSE_CONTRACT, "version": RAPTOR_DENSE_384_VERSION, "cache_size": 384}
# A wider 160 mm box (most knee scans cover 160 mm; the 140 mm crop cuts into the front of the knee on the sagittal
# view in about 31% of studies) stored at 384 px, which keeps about the detail of the 140 mm box at 336 px.
RAPTOR_DENSE_160MM_384_VERSION = "raptor-dense-distinct-160mm-384-v1"
RAPTOR_DENSE_160MM_384_CONTRACT = {**RAPTOR_DENSE_CONTRACT, "version": RAPTOR_DENSE_160MM_384_VERSION,
                                   "cache_size": 384, "crop_mm": 160.0}
# 200 mm at 384 px: wider than 97% of scans, so in practice each scan's whole image (the crop never exceeds the image).
RAPTOR_DENSE_200MM_384_VERSION = "raptor-dense-distinct-200mm-384-v1"
RAPTOR_DENSE_200MM_384_CONTRACT = {**RAPTOR_DENSE_CONTRACT, "version": RAPTOR_DENSE_200MM_384_VERSION,
                                   "cache_size": 384, "crop_mm": 200.0}
# The native 384 px layout with every 140 mm box centred on the knee joint instead of the image centre (knee-centred
# test, Oct 13, 2026): the joint comes from an anchor record in patient coordinates and is placed in each slice by its
# own DICOM geometry; a box that would leave the image is shifted inside, and a study or slice without an anchor keeps
# the image-centre box.
RAPTOR_DENSE_KNEE_384_VERSION = "raptor-dense-distinct-knee-384-v1"
RAPTOR_DENSE_KNEE_384_CONTRACT = {**RAPTOR_DENSE_CONTRACT, "version": RAPTOR_DENSE_KNEE_384_VERSION, "cache_size": 384,
                                  "crop_center": "knee joint anchor (patient coordinates) placed by each slice's DICOM "
                                                 "geometry, shifted inside the image; image centre without an anchor"}
# Close-up views modelled on the "v4" views in the code of R7, heliosli's public ConvNeXt-small model (close-up
# test, Oct 13, 2026): 100 mm boxes at 384 px, each from a series that already fills a main slot, CLOSEUP_SLICES
# slices about CLOSEUP_SPACING_MM apart around the centre point. Centres in patient coordinates (LPS: +x left,
# +y posterior, +z superior): the kneecap found by the knee finder, else the joint moved 40 mm forward (and 25 mm up
# on the sagittal series, R7's fallback); the coronal pair 30 mm either side of the joint. Without a joint anchor the
# close-up slots stay empty.
CLOSEUP_SLICES = 16
CLOSEUP_MM = 100.
CLOSEUP_SPACING_MM = 3.
KNEECAP_FALLBACK_MM = {"Sagittal": (0., -40., 25.), "Axial": (0., -40., 0.)}
RAPTOR_CLOSEUPS = (("sagittal_kneecap", "Sagittal", (0, 1), "kneecap"), ("axial_kneecap", "Axial", (4,), "kneecap"),
                   ("coronal_right", "Coronal", (2, 3), (-30., 0., 0.)), ("coronal_left", "Coronal", (2, 3), (30., 0., 0.)))
RAPTOR_DENSE_KNEE_CLOSEUP_384_VERSION = "raptor-dense-distinct-knee-closeup-384-v1"
RAPTOR_DENSE_KNEE_CLOSEUP_384_CONTRACT = {
    **RAPTOR_DENSE_KNEE_384_CONTRACT, "version": RAPTOR_DENSE_KNEE_CLOSEUP_384_VERSION,
    "closeups": [[name, plane, list(sources), centre if isinstance(centre, str) else list(centre)]
                 for name, plane, sources, centre in RAPTOR_CLOSEUPS],
    "closeup_mm": CLOSEUP_MM, "closeup_slices": CLOSEUP_SLICES, "closeup_spacing_mm": CLOSEUP_SPACING_MM,
    "kneecap_fallback_mm": {plane: list(offset) for plane, offset in KNEECAP_FALLBACK_MM.items()},
}
RAPTOR_VARIANTS = {
    version: {"slots": slots, "depth": sum(slot[2] for slot in slots) + len(contract.get("closeups", [])) * CLOSEUP_SLICES,
              "distinct": distinct, "contract": contract, "closeups": RAPTOR_CLOSEUPS if "closeups" in contract else (),
              "size": contract["cache_size"], "knee_centred": "crop_center" in contract,
              "fingerprint": hashlib.sha256(json.dumps(contract, sort_keys=True).encode()).hexdigest()}
    for version, slots, distinct, contract in (
        (RAPTOR_PREPROCESS_VERSION, RAPTOR_SLOTS, False, RAPTOR_CONTRACT),
        (RAPTOR_DENSE_VERSION, RAPTOR_DENSE_SLOTS, True, RAPTOR_DENSE_CONTRACT),
        (RAPTOR_DENSE_448_VERSION, RAPTOR_DENSE_SLOTS, True, RAPTOR_DENSE_448_CONTRACT),
        (RAPTOR_DENSE_384_VERSION, RAPTOR_DENSE_SLOTS, True, RAPTOR_DENSE_384_CONTRACT),
        (RAPTOR_DENSE_160MM_384_VERSION, RAPTOR_DENSE_SLOTS, True, RAPTOR_DENSE_160MM_384_CONTRACT),
        (RAPTOR_DENSE_200MM_384_VERSION, RAPTOR_DENSE_SLOTS, True, RAPTOR_DENSE_200MM_384_CONTRACT),
        (RAPTOR_DENSE_KNEE_384_VERSION, RAPTOR_DENSE_SLOTS, True, RAPTOR_DENSE_KNEE_384_CONTRACT),
        (RAPTOR_DENSE_KNEE_CLOSEUP_384_VERSION, RAPTOR_DENSE_SLOTS, True, RAPTOR_DENSE_KNEE_CLOSEUP_384_CONTRACT),
    )
}
# How a cached slice becomes the model's image: resized to image_size, or its central image_size pixels.
RAPTOR_VIEWS = ("resize", "center_crop")
if {version: variant["size"] for version, variant in RAPTOR_VARIANTS.items()} != RAPTOR_VARIANT_SIZES:
    raise RuntimeError("constants.RAPTOR_VARIANT_SIZES must list every Raptor preparation variant and its cache size")


def raptor_variant(version: str) -> dict[str, Any]:
    if version not in RAPTOR_VARIANTS:
        raise ValueError(f"Unknown Raptor preparation version {version!r}; choose one of {sorted(RAPTOR_VARIANTS)}")
    return RAPTOR_VARIANTS[version]


def _pick_series(rows: list[dict[str, Any]], plane: str, fluid: int, taken: set[str]) -> dict[str, Any] | None:
    """The source's choice for one slot: the first series of the plane (CSV order) not in ``taken``, preferring the
    requested fluid setting; None when the plane has no such series."""
    candidates = [row for row in rows if row["Anatomical_Plane"] == plane and row[SERIES_ID] not in taken]
    preferred = [row for row in candidates if row["Fluid_Sensitive"] == fluid] if fluid in (0, 1) else []
    return (preferred or candidates)[0] if candidates else None


def _select_slots(rows: list[dict[str, Any]], layout=RAPTOR_SLOTS) -> list[dict[str, Any]]:
    """Keep CSV order and the same first-unused series preferences as the source."""
    used: set[str] = set()
    slots, offset = [], 0
    for index, (plane, fluid, count) in enumerate(layout):
        selected = _pick_series(rows, plane, fluid, used)
        slot = {"slot": index, "plane": plane, "requested_fluid": fluid, "offset": offset, "count": count,
                SERIES_ID: None, "Fluid_Sensitive": None, "Fat_Suppression": None}
        if selected is not None:
            used.add(selected[SERIES_ID])
            slot.update({key: selected[key] for key in (SERIES_ID, "Fluid_Sensitive", "Fat_Suppression")})
        slots.append(slot)
        offset += count
    return slots


def _source_plan(study_dir: Path, rows: list[dict[str, Any]], layout=RAPTOR_SLOTS,
                 unlisted: dict[int, Exception] | None = None) -> tuple[list[dict[str, Any]], str]:
    """The planned series of each slot and a fingerprint of their files.

    By default a series whose files cannot be listed fails the study. With ``unlisted`` (a dict, given only when
    bad-series insurance is on) that error is stored there under the slot number and the slot's file fingerprint is
    None, so the caller can drop just that series. Studies whose files all list are planned exactly as before.
    """
    slots = _select_slots(rows, layout)
    for slot in slots:
        if slot[SERIES_ID] is not None:
            try:
                fingerprint, _ = source_fingerprint(study_dir / slot[SERIES_ID])
            except Exception as error:
                if unlisted is None:
                    raise
                unlisted[slot["slot"]], fingerprint = error, None
            slot["source_fingerprint"] = fingerprint
    fingerprint = hashlib.sha256(json.dumps(slots, sort_keys=True).encode()).hexdigest()
    return slots, fingerprint


def _ordered_headers(series_dir: Path) -> tuple[list[dict[str, Any]], list[str]]:
    _, files = source_fingerprint(series_dir)
    records, warnings = [], []
    for path in files:
        ds = pydicom.dcmread(path, stop_before_pixels=True)
        iop, ipp = ds.get("ImageOrientationPatient"), ds.get("ImagePositionPatient")
        if iop is not None and ipp is not None:
            orientation, position = np.asarray(iop, float), np.asarray(ipp, float)
            if orientation.shape != (6,) or position.shape != (3,) or not np.isfinite(np.r_[orientation, position]).all():
                raise ValueError(f"Invalid geometry in {path.name}")
            normal = np.cross(orientation[:3], orientation[3:])
            if np.linalg.norm(normal) < 0.9:
                raise ValueError(f"Invalid direction cosines in {path.name}")
            order = float(position @ normal)
            ordering = "physical_position"
        else:
            instance = ds.get("InstanceNumber")
            if instance is None:
                raise ValueError(f"No physical position or InstanceNumber in {path.name}")
            order, ordering = float(instance), "instance_number"
            warnings.append("slice_order_fallback:InstanceNumber")
            orientation, position = None, None
        spacing = ds.get("PixelSpacing")
        if spacing is None:
            pixel_spacing = [0.5, 0.5]
            warnings.append("missing_pixel_spacing:source_recipe_default_0.5mm")
        else:
            pixel_spacing = list(map(float, spacing))
            if len(pixel_spacing) != 2 or not np.isfinite(pixel_spacing).all() or min(pixel_spacing) <= 0:
                raise ValueError(f"Invalid PixelSpacing in {path.name}")
            if not np.isclose(*pixel_spacing, rtol=0.01, atol=1e-6):
                warnings.append("anisotropic_pixel_spacing:source_recipe_uses_row_spacing")
        if not np.isfinite(order):
            raise ValueError(f"Invalid slice position in {path.name}")
        records.append({"path": path, "order": order, "ordering": ordering, "pixel_spacing_mm": pixel_spacing,
                        "orientation": None if orientation is None else orientation.tolist(),
                        "position": None if position is None else position.tolist()})
    if len({record["ordering"] for record in records}) != 1:
        raise ValueError("Mixed physical and instance-number ordering within a series")
    orientations = [record["orientation"] for record in records if record["orientation"] is not None]
    if orientations and any(not np.allclose(value, orientations[0], atol=.015, rtol=0) for value in orientations):
        raise ValueError("Mixed image orientations within a series")
    records.sort(key=lambda record: (record["order"], record["path"].name))
    if len(records) > 1 and (np.diff([record["order"] for record in records]) <= 0).any():
        raise ValueError("Duplicate slice positions/instance numbers need inspection before preparing this series")
    return records, sorted(set(warnings))


def _read_pixels(path: Path) -> np.ndarray:
    ds = pydicom.dcmread(path)
    image = pydicom.pixels.apply_modality_lut(ds.pixel_array, ds).astype(np.float32)
    if image.ndim != 2 or min(image.shape) < 1 or not np.isfinite(image).all():
        raise ValueError(f"Expected a finite, single-frame grayscale image in {path.name}")
    if ds.get("PhotometricInterpretation") == "MONOCHROME1":
        image = image.max() - image
    elif ds.get("PhotometricInterpretation") != "MONOCHROME2":
        raise ValueError(f"Unsupported PhotometricInterpretation in {path.name}")
    return image


def _slice_picks(depth: int, count: int, distinct: bool = False) -> np.ndarray:
    """Source slice indices over the 6-94% span of a series.

    The scored recipe spreads exactly ``count`` picks and may repeat slices. The
    distinct variant keeps every slice of the span when it fits, otherwise evenly
    spaced distinct slices, so it can return fewer than ``count`` picks.
    """
    lo, hi = int(depth * .06), max(int(depth * .94) - 1, int(depth * .06))
    if not distinct:
        return np.linspace(lo, hi, count).round().astype(int) if depth > 1 else np.zeros(count, dtype=int)
    span = np.arange(lo, hi + 1)
    return span if len(span) <= count else np.unique(np.floor(np.linspace(lo, hi, count) + .5).astype(int))


def _closeup_picks(records: list[dict[str, Any]], center_lps, count: int, spacing_mm: float) -> np.ndarray:
    """Up to ``count`` source slices about ``spacing_mm`` apart, centred on the slice nearest ``center_lps``."""
    if any(record["orientation"] is None or record["position"] is None for record in records):
        raise ValueError("Close-ups need slice geometry")
    orientation = np.asarray(records[0]["orientation"], float)
    normal = np.cross(orientation[:3], orientation[3:])
    distances = [abs(float((np.asarray(center_lps, float) - np.asarray(record["position"], float)) @ normal))
                 for record in records]
    nearest = int(np.argmin(distances))
    gap = float(np.median(np.diff([record["order"] for record in records]))) if len(records) > 1 else spacing_mm
    step = max(1, int(round(spacing_mm / max(gap, 1e-3))))
    picks = nearest + step * np.arange(-(count // 2), count - count // 2)
    return picks[(picks >= 0) & (picks < len(records))]


def _series_images(series_dir: Path, count: int, distinct: bool = False,
                   size: int = RAPTOR_SIZE, crop_mm: float = 140.,
                   center_lps: list[float] | None = None, picks_for=None) -> tuple[np.ndarray, dict[str, Any]]:
    records, warnings = _ordered_headers(series_dir)
    depth = len(records)
    picks = _slice_picks(depth, count, distinct) if picks_for is None else picks_for(records)
    chosen = [records[min(int(index), depth - 1)] for index in picks]
    decoded: dict[Path, np.ndarray] = {}
    for record in chosen:
        if record["path"] not in decoded:
            decoded[record["path"]] = _read_pixels(record["path"])
    images = [decoded[record["path"]] for record in chosen]
    # Retain NumPy's float64 scalar type. Converting these to Python floats would
    # keep normalized pixels in float32 under NumPy 2 and change uint8 rounding.
    low, high = np.percentile(np.concatenate([image.ravel() for image in images]), [2., 98.])
    volume = np.zeros((count, size, size), dtype=np.uint8)
    crops = []
    for index, (image, record) in enumerate(zip(images, chosen, strict=True)):
        normalized = np.clip((image - low) / (high - low + 1e-6), 0, 1)
        height, width = image.shape
        crop_size = min(int(round(crop_mm / max(record["pixel_spacing_mm"][0], 1e-3))), height, width)
        if crop_size < 1:
            raise ValueError("The physical center crop has zero size")
        top, left = (height - crop_size) // 2, (width - crop_size) // 2
        crop = {}
        if center_lps is not None and record["orientation"] is not None and record["position"] is not None:
            row, column, distance = project_to_pixel(center_lps, record["orientation"], record["position"],
                                                     record["pixel_spacing_mm"])
            top, left = centred_box(row, column, crop_size, height, width)
            crop = {"center": "anchor", "anchor_row_col": [round(row, 1), round(column, 1)],
                    "anchor_plane_distance_mm": round(distance, 1)}
        elif center_lps is not None:
            crop = {"center": "image_no_geometry"}
        cropped = normalized[top:top + crop_size, left:left + crop_size]
        volume[index] = (cv2.resize(cropped, (size, size), interpolation=cv2.INTER_AREA) * 255).astype(np.uint8)
        crops.append({"top": top, "left": left, "size": crop_size, "source_shape": list(image.shape), **crop})
    filled = len(chosen)
    valid = volume[:filled].reshape(filled, -1).sum(axis=1) > 0
    if not valid.all():
        warnings.append(f"zero_image_slices_excluded_from_windows:{int((~valid).sum())}")
    return volume, {
        "source_depth": depth, "ordering": records[0]["ordering"],
        "selected_source_indices": picks.tolist(),
        "selected_source_files": [record["path"].name for record in chosen],
        "selected_positions": [record["order"] for record in chosen],
        "selected_pixel_spacings_mm": [record["pixel_spacing_mm"] for record in chosen],
        "selected_orientations_lps": [record["orientation"] for record in chosen],
        "selected_image_positions_lps": [record["position"] for record in chosen],
        "normalization": {"low": float(low), "high": float(high), "percentiles": [2., 98.]},
        "crops": crops, "warnings": warnings,
    }


def _closeup_slot(number: int, offset: int, name: str, plane: str, sources, centre, slots: list[dict[str, Any]],
                  anchor: dict[str, Any] | None) -> dict[str, Any]:
    """A close-up slot's plan: the first present main slot among ``sources`` and the centre point, or no series."""
    slot = {"slot": number, "plane": plane, "requested_fluid": -1, "offset": offset, "count": CLOSEUP_SLICES,
            "closeup": name, SERIES_ID: None}
    source = next((slots[index] for index in sources if slots[index][SERIES_ID] is not None), None)
    joint = None if anchor is None else anchor.get("joint_lps")
    if source is None or joint is None:
        return slot
    if centre == "kneecap":
        kneecap = anchor.get("patella_lps")
        point = kneecap if kneecap is not None else (np.asarray(joint) + KNEECAP_FALLBACK_MM[plane]).tolist()
        how = "kneecap" if kneecap is not None else "kneecap_fallback"
    else:
        point, how = (np.asarray(joint) + np.asarray(centre)).tolist(), "joint_offset"
    slot.update({SERIES_ID: source[SERIES_ID], "source_slot": source["slot"], "Fluid_Sensitive": source["Fluid_Sensitive"],
                 "Fat_Suppression": source["Fat_Suppression"], "center_lps": [float(v) for v in point], "center_source": how})
    return slot


def _drop_series(slot: dict[str, Any], error: Exception, rows: list[dict[str, Any]], slots: list[dict[str, Any]],
                 dropped: list[dict[str, Any]]) -> None:
    """Bad-series insurance: record why a main slot's series failed, then move the slot to the next usable series.

    The replacement follows the source's own choice rule (``_pick_series``) among the plane's series that no other
    slot uses and that have not failed already. Without one the slot is left empty. Other slots always keep their
    series, so their images stay the same. This is not always the layout of a study that never had the series: when
    a fluid-sensitive series fails and the plane's other series already fills the next slot, that series stays where
    it is and the first slot is left empty, whereas slot selection would have moved it into the first slot.
    """
    dropped.append({"slot": slot["slot"], "plane": slot["plane"], SERIES_ID: slot[SERIES_ID],
                    "error": f"{type(error).__name__}: {error}"[:300]})
    taken = {other[SERIES_ID] for other in slots if other[SERIES_ID] is not None} | {item[SERIES_ID] for item in dropped}
    replacement = _pick_series(rows, slot["plane"], slot["requested_fluid"], taken)
    slot.pop("source_fingerprint", None)  # an empty slot has none; a replacement gets its own before it is read
    for key in (SERIES_ID, "Fluid_Sensitive", "Fat_Suppression"):
        slot[key] = None if replacement is None else replacement[key]
    dropped[-1]["replacement"] = slot[SERIES_ID]


def _allowed_windows(slots: list[dict[str, Any]], valid: list[bool]) -> tuple[list[int], list[int]]:
    centers, planes = [], []
    for slot in slots:
        if slot[SERIES_ID] is None:
            continue
        start, end = slot["offset"], slot["offset"] + slot["count"]
        for center in range(start + 1, end - 1):
            if all(valid[center - 1:center + 2]):
                centers.append(center)
                planes.append(PLANES.index(slot["plane"]))
    return centers, planes


def _read_metadata(path: Path, expected_sha: str | None = None,
                   version: str = RAPTOR_PREPROCESS_VERSION) -> dict[str, Any]:
    variant = raptor_variant(version)
    payload = path.read_bytes()
    if expected_sha is not None and hashlib.sha256(payload).hexdigest() != expected_sha:
        raise ValueError("Metadata checksum mismatch; manifest or cache is stale")
    metadata = json.loads(payload)
    if metadata.get("metadata_payload_sha256") != _metadata_digest(metadata):
        raise ValueError("Metadata payload checksum mismatch")
    if metadata.get("preprocess_version") != version:
        raise ValueError("Raptor preprocessing version mismatch")
    if metadata.get("preprocess_fingerprint") != variant["fingerprint"] or metadata.get("input_contract") != variant["contract"]:
        raise ValueError("Raptor preprocessing contract mismatch")
    if metadata.get("shape") != [variant["depth"], variant["size"], variant["size"]] or metadata.get("dtype") != "uint8":
        raise ValueError("Raptor cached shape/dtype metadata mismatch")
    valid, slots = metadata["valid_slices"], metadata["slots"]
    if (len(valid) != variant["depth"] or not all(isinstance(value, bool) for value in valid)
            or len(slots) != len(variant["slots"]) + len(variant["closeups"])):
        raise ValueError("Invalid slice mask or slot list")
    offset = 0
    used = set()
    main, closeups = slots[:len(variant["slots"])], slots[len(variant["slots"]):]
    for index, (slot, (plane, fluid, count)) in enumerate(zip(main, variant["slots"], strict=True)):
        if (slot["slot"], slot["plane"], slot["requested_fluid"], slot["offset"], slot["count"]) != (index, plane, fluid, offset, count):
            raise ValueError("Slot layout differs from the Raptor contract")
        series = slot[SERIES_ID]
        if series is None:
            if any(valid[offset:offset + count]):
                raise ValueError("A missing series has valid cached slices")
        else:
            if series in used:
                raise ValueError("The same MRI series was used for two slots")
            used.add(series)
            # Dense variants may leave the end of a slot empty; those positions are never valid.
            filled = len(slot["selected_source_indices"])
            if not 1 <= filled <= count or (filled != count and not variant["distinct"]):
                raise ValueError("Invalid number of cached slices in a slot")
            for field in ("selected_source_files", "selected_source_indices", "selected_positions", "crops"):
                if len(slot[field]) != filled:
                    raise ValueError(f"Invalid per-slice provenance: {field}")
            if any(valid[offset + filled:offset + count]):
                raise ValueError("Empty slot positions are marked as valid slices")
        offset += count
    for index, (slot, (name, plane, sources, _)) in enumerate(zip(closeups, variant["closeups"], strict=True)):
        number = len(variant["slots"]) + index
        if ((slot["slot"], slot["plane"], slot["offset"], slot["count"], slot.get("closeup"))
                != (number, plane, offset, CLOSEUP_SLICES, name)):
            raise ValueError("Close-up slot layout differs from the contract")
        series = slot[SERIES_ID]
        filled = 0 if series is None else len(slot["selected_source_indices"])
        if series is not None:
            if series not in {main[i][SERIES_ID] for i in sources} or slot.get("source_slot") not in sources:
                raise ValueError("A close-up must reuse the series of one of its source slots")
            if not 1 <= filled <= CLOSEUP_SLICES or any(len(slot[field]) != filled for field in ("selected_source_files", "crops")):
                raise ValueError("Invalid close-up slices")
        if any(valid[offset + filled:offset + CLOSEUP_SLICES]):
            raise ValueError("Empty close-up positions are marked as valid slices")
        offset += CLOSEUP_SLICES
    centers, planes = _allowed_windows(slots, valid)
    if not centers or centers != metadata["allowed_centers"] or planes != metadata["window_planes"]:
        raise ValueError("No valid windows, or stored windows cross a series boundary/missing slice")
    return metadata


def _check_array(path: Path, depth: int = RAPTOR_DEPTH, size: int = RAPTOR_SIZE) -> int:
    volume = np.load(path, mmap_mode="r", allow_pickle=False)
    try:
        if not isinstance(volume, np.memmap) or volume.shape != (depth, size, size) or volume.dtype != np.uint8:
            raise ValueError(f"Raptor cache must be uint8 [{depth},{size},{size}]")
        size = path.stat().st_size
        if size != volume.offset + volume.nbytes:
            raise ValueError("Raptor cache is truncated or has trailing bytes")
        return size
    finally:
        if isinstance(volume, np.memmap):
            volume._mmap.close()


def _prepare_study(job: dict[str, Any]) -> dict[str, Any]:
    cv2.setNumThreads(1)
    study, output = job[STUDY_ID], Path(job["output_dir"])
    relative = Path("studies") / f"{study}.npy"
    cache_path, metadata_path = output / relative, (output / relative).with_suffix(".json")
    study_dir = Path(job["study_dir"])
    version = job.get("version", RAPTOR_PREPROCESS_VERSION)
    variant = raptor_variant(version)
    anchor = job.get("anchor") if variant["knee_centred"] else None
    # Bad-series insurance (used at inference): a series that cannot be read or fails its checks is dropped instead
    # of failing the whole study, and the study fails only when no series is usable. It is off by default, so
    # training preparation still stops on any bad series. Studies where nothing fails come out byte for byte the same.
    insured = bool(job.get("drop_bad_series", False))
    unlisted: dict[int, Exception] | None = {} if insured else None
    dropped: list[dict[str, Any]] = []
    first_error: Exception | None = None
    try:
        slots, source_hash = _source_plan(study_dir, job["rows"], variant["slots"], unlisted)
        metadata, rebuilt = None, None
        if cache_path.exists() or metadata_path.exists():
            try:
                candidate = _read_metadata(metadata_path, version=version)
                if candidate[STUDY_ID] != study or candidate["source_fingerprint"] != source_hash:
                    raise ValueError("source fingerprint or study ID changed")
                if candidate.get("anchor") != anchor:
                    raise ValueError("knee anchor changed")
                if candidate.get("dropped_series"):
                    # Its replacement series are outside the planned fingerprint, so it is never reused.
                    raise ValueError("a study with dropped series is always prepared again")
                _check_array(cache_path, variant["depth"], variant["size"])
                if _sha256_file(cache_path) != candidate["cache_sha256"]:
                    raise ValueError("array checksum changed")
                metadata = candidate
            except (OSError, ValueError, KeyError, TypeError, AttributeError) as exc:
                rebuilt = f"cache_rebuilt:{type(exc).__name__}:{exc}"
        reused = metadata is not None
        if metadata is None:
            volume = np.zeros((variant["depth"], variant["size"], variant["size"]), dtype=np.uint8)
            warnings = []
            if variant["knee_centred"] and (anchor is None or anchor["joint_lps"] is None):
                warnings.append(f"no_knee_anchor:image_centre:{None if anchor is None else anchor['method']}")
            for slot in slots:
                # Without insurance this loop reads each planned series once and any error fails the study, as before.
                # With it, a series that fails to read is read once more (the error may be a one-off, such as a brief
                # file-system or memory error), then dropped, and the slot tries its replacement, if any (_drop_series).
                retried = None
                while slot[SERIES_ID] is not None:
                    try:
                        if "source_fingerprint" not in slot:  # a replacement series: fingerprint its files first
                            slot["source_fingerprint"], _ = source_fingerprint(study_dir / slot[SERIES_ID])
                        elif slot["source_fingerprint"] is None:  # the planned series' files could not be listed
                            raise unlisted[slot["slot"]]
                        images, geometry = _series_images(study_dir / slot[SERIES_ID], slot["count"], variant["distinct"],
                                                           variant["size"], variant["contract"]["crop_mm"],
                                                           None if anchor is None else anchor["joint_lps"])
                        break
                    except Exception as error:
                        if not insured:
                            raise
                        first_error = first_error or error
                        if slot.get("source_fingerprint") is not None and retried != slot[SERIES_ID]:
                            retried = slot[SERIES_ID]  # its files list, so it is worth one more read before dropping
                            continue
                        _drop_series(slot, error, job["rows"], slots, dropped)
                        warnings.append(f"dropped_series:slot{slot['slot']}:{dropped[-1][SERIES_ID]}:"
                                        f"{type(error).__name__}:{slot[SERIES_ID] or 'left_empty'}")
                if slot[SERIES_ID] is None:
                    warnings.append(f"missing_slot:{slot['slot']}:{slot['plane']}")
                    continue
                volume[slot["offset"]:slot["offset"] + slot["count"]] = images
                slot.update(geometry)
                warnings.extend(f"slot{slot['slot']}:{value}" for value in geometry["warnings"])
            if dropped and all(slot[SERIES_ID] is None for slot in slots):
                # No series is usable: fail the study with the error a run without insurance raises. That run lists
                # every planned series' files before reading any, so a listing error comes before any reading error.
                raise unlisted[min(unlisted)] if unlisted else first_error
            offset = sum(slot[2] for slot in variant["slots"])
            for index, (name, plane, sources, centre) in enumerate(variant["closeups"]):
                slot = _closeup_slot(len(variant["slots"]) + index, offset + index * CLOSEUP_SLICES, name, plane,
                                     sources, centre, slots, anchor)
                images = None
                if slot[SERIES_ID] is not None:
                    point = slot["center_lps"]
                    for attempt in range(2 if insured else 1):  # with insurance a failed read is tried once more
                        try:
                            images, geometry = _series_images(
                                study_dir / slot[SERIES_ID], CLOSEUP_SLICES, True, variant["size"], CLOSEUP_MM, point,
                                picks_for=lambda records, point=point: _closeup_picks(records, point, CLOSEUP_SLICES,
                                                                                      CLOSEUP_SPACING_MM))
                            break
                        except Exception as error:
                            if not insured:
                                raise
                            if attempt == 0:
                                continue
                            # Insurance: a close-up that cannot be read stays empty (its main slot keeps its images).
                            dropped.append({"slot": slot["slot"], "plane": plane, SERIES_ID: slot[SERIES_ID],
                                            "error": f"{type(error).__name__}: {error}"[:300], "replacement": None})
                            warnings.append(f"dropped_series:slot{slot['slot']}:{slot[SERIES_ID]}:"
                                            f"{type(error).__name__}:left_empty")
                            slot = _closeup_slot(slot["slot"], slot["offset"], name, plane, (), centre, slots, None)
                if images is not None:
                    volume[slot["offset"]:slot["offset"] + len(images)] = images
                    slot.update(geometry)
                    warnings.extend(f"slot{slot['slot']}:{value}" for value in geometry["warnings"])
                else:
                    warnings.append(f"missing_closeup:{slot['slot']}:{name}")
                slots.append(slot)
            valid = (volume.reshape(variant["depth"], -1).sum(axis=1) > 0).tolist()
            centers, planes = _allowed_windows(slots, valid)
            if not centers:
                raise ValueError("Study has no valid three-slice windows within a series")
            _, end_hash = _source_plan(study_dir, job["rows"], variant["slots"], None if unlisted is None else {})
            replaced = {item["replacement"] for item in dropped if item["replacement"] is not None}
            if end_hash != source_hash or any(  # replacement series lie outside the plan, so check their files too
                    slot.get("source_fingerprint") != source_fingerprint(study_dir / slot[SERIES_ID])[0]
                    for slot in slots[:len(variant["slots"])] if slot[SERIES_ID] in replaced):
                raise ValueError("Raw source files changed while this study was being prepared")
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            descriptor, temporary = tempfile.mkstemp(prefix=f".{study}.", suffix=".tmp", dir=cache_path.parent)
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
            metadata = {STUDY_ID: study, "shape": list(volume.shape), "dtype": "uint8",
                        "preprocess_version": version, "preprocess_fingerprint": variant["fingerprint"],
                        "input_contract": variant["contract"], "source_fingerprint": source_hash,
                        "cache_sha256": cache_hash, "slots": slots, "valid_slices": valid,
                        "allowed_centers": centers, "window_planes": planes, "warnings": warnings}
            if variant["knee_centred"]:
                metadata["anchor"] = anchor
            if dropped:  # only studies that lost a series carry this key; clean studies' metadata is unchanged
                metadata["dropped_series"] = dropped
            metadata["metadata_payload_sha256"] = _metadata_digest(metadata)
            _atomic_json(metadata_path, metadata)
        warnings = list(metadata["warnings"])
        if rebuilt:
            warnings.append(rebuilt)
        result = {STUDY_ID: study, "cache_path": str(relative), "metadata_path": str(relative.with_suffix(".json")),
                  "num_slices": variant["depth"], "image_size": variant["size"],
                  "num_valid_windows": len(metadata["allowed_centers"]),
                  "num_slots_used": sum(slot[SERIES_ID] is not None for slot in metadata["slots"]),
                  "preprocess_version": version, "preprocess_fingerprint": variant["fingerprint"],
                  "source_fingerprint": source_hash, "cache_sha256": metadata["cache_sha256"],
                  "metadata_sha256": _sha256_file(metadata_path), "status": "reused" if reused else "prepared",
                  "warnings": warnings, "error": None}
        if dropped:  # taken out again by prepare_raptor_data for the QC summary, so it never becomes a manifest column
            result["dropped_series"] = dropped
        return result
    except Exception as exc:
        return {STUDY_ID: study, "status": "failed", "warnings": [], "error": f"{type(exc).__name__}: {exc}"}


def prepare_raptor_data(data_root: str | Path, output_dir: str | Path, split: str = "train", workers: int = 4,
                        study_ids: list[str] | None = None, limit_studies: int | None = None,
                        version: str = RAPTOR_PREPROCESS_VERSION, anchors_dir: str | Path | None = None,
                        drop_bad_series: bool = False) -> Path:
    """Prepare a Raptor image cache, by default the scored 64-slice recipe, one atomic cache per study.

    Knee-centred variants need ``anchors_dir`` (one ``<StudyInstanceUID>.json`` anchor record per study); a study
    without a usable joint anchor keeps the image-centre boxes and is listed with a warning.

    ``drop_bad_series`` (bad-series insurance, used by the Kaggle bundle): a series that cannot be read or fails its
    checks no longer fails its whole study. After one more read attempt, its slot takes the next usable series of the
    same plane that no other slot uses, or is left empty; the other slots keep their series (see ``_drop_series``).
    Only a study with no usable series fails, with the error a run without insurance would give. Each drop is listed
    in that study's warnings and metadata (``dropped_series``) and in the QC summary's ``dropped_series``; both exist
    only when something was dropped. Studies where nothing fails are prepared byte for byte as without it.
    """
    contract = raptor_variant(version)["contract"]
    knee_centred = raptor_variant(version)["knee_centred"]
    if knee_centred != (anchors_dir is not None):
        raise ValueError("Knee-centred variants need anchors_dir, and only they accept it")
    if split not in {"train", "test"} or workers < 1:
        raise ValueError("split must be train/test and workers must be positive")
    if limit_studies is not None and limit_studies < 1:
        raise ValueError("limit_studies must be positive")
    data_root, output = Path(data_root).resolve(), Path(output_dir).resolve()
    study_table = pd.read_csv(data_root / f"{split}.csv", usecols=[STUDY_ID], dtype={STUDY_ID: str})
    if study_table[STUDY_ID].isna().any() or study_table[STUDY_ID].duplicated().any():
        raise ValueError("Study IDs must be nonmissing and unique")
    selected = sorted(study_table[STUDY_ID].tolist())
    if study_ids is not None:
        requested = set(map(str, study_ids))
        if requested - set(selected):
            raise ValueError(f"Unknown requested study IDs: {sorted(requested - set(selected))[:3]}")
        selected = [study for study in selected if study in requested]
    if limit_studies is not None:
        selected = selected[:limit_studies]
    if not selected:
        raise ValueError("No studies selected")
    for study in selected:
        _safe_component(study, STUDY_ID)
    table = pd.read_csv(data_root / f"{split}_series.csv", dtype={STUDY_ID: str, SERIES_ID: str})
    if STUDY_ID in table:
        # Check only the selected studies' rows: one malformed row for another study must not block these.
        table = table[table[STUDY_ID].isin(set(selected))]
    required = [STUDY_ID, SERIES_ID, "Anatomical_Plane", "Fluid_Sensitive", "Fat_Suppression"]
    if not set(required).issubset(table.columns) or table[required].isna().any().any():
        raise ValueError("Missing series IDs or metadata")
    if table.duplicated([STUDY_ID, SERIES_ID]).any() or not table["Anatomical_Plane"].isin(PLANES).all():
        raise ValueError("Duplicate series rows or unknown anatomical planes")
    for flag in ("Fluid_Sensitive", "Fat_Suppression"):
        if not table[flag].isin([0, 1]).all():
            raise ValueError(f"{flag} must contain zero or one")
        table[flag] = table[flag].astype(int)
    for series in table[SERIES_ID]:
        _safe_component(series, SERIES_ID)
    records = {study: frame[required].to_dict(orient="records") for study, frame in table.groupby(STUDY_ID, sort=False)}
    output.mkdir(parents=True, exist_ok=True)
    manifest = output / "manifest.csv"
    with _preparation_lock(output):
        jobs = [{STUDY_ID: study, "study_dir": str(data_root / f"{split}_series" / study),
                 "output_dir": str(output), "rows": records.get(study, []), "version": version,
                 "drop_bad_series": bool(drop_bad_series)} for study in selected]
        for job in jobs:
            if knee_centred:
                job["anchor"] = read_anchor(anchors_dir, job[STUDY_ID])
        started = time.monotonic()
        results = []
        print(f"Preparing {len(jobs)} {split} studies with {workers} workers; recipe {version}", flush=True)
        pool = concurrent.futures.ProcessPoolExecutor(max_workers=workers) if workers > 1 else None
        try:
            iterator = pool.map(_prepare_study, jobs, chunksize=1) if pool is not None else map(_prepare_study, jobs)
            for result in iterator:
                results.append(result)
                if len(results) in {1, 10} or len(results) % 50 == 0 or len(results) == len(jobs):
                    elapsed = time.monotonic() - started
                    rate = len(results) / max(elapsed, .001)
                    print(f"{len(results)}/{len(jobs)} studies; {elapsed:.1f}s elapsed; {rate:.2f} studies/s; "
                          f"ETA {(len(jobs) - len(results)) / rate:.0f}s; latest={result['status']}", flush=True)
        finally:
            if pool is not None:
                pool.shutdown(wait=True)
        failures = [row for row in results if row["status"] == "failed"]
        # Series dropped by bad-series insurance leave the study rows here, so the manifest's columns never change.
        dropped = [{STUDY_ID: row[STUDY_ID], **item} for row in results for item in row.pop("dropped_series", [])]
        report = {"preprocess_version": version, "input_contract": contract,
                  "split": split, "data_root": str(data_root), "selected_studies": selected,
                  "summary": {"studies": len(results), "failed_studies": len(failures),
                              "prepared_studies": sum(row["status"] == "prepared" for row in results),
                              "reused_studies": sum(row["status"] == "reused" for row in results),
                              "studies_with_warnings": sum(bool(row["warnings"]) for row in results),
                              "seconds": time.monotonic() - started}, "studies": results}
        if dropped:
            report["summary"]["dropped_series"] = dropped
        _atomic_json(output / "qc.json", report)
        flat = [{**row, "warnings": "|".join(row["warnings"])} for row in results]
        _atomic_csv(output / "qc.csv", pd.DataFrame(flat))
        if failures:
            raise RuntimeError(f"Raptor preparation failed for {len(failures)} studies; see {output / 'qc.json'}. "
                               "Existing manifest was not replaced.")
        _atomic_csv(manifest, pd.DataFrame([{key: value for key, value in row.items() if key != "error"} for row in flat]))
    return manifest


def validate_raptor_manifest(manifest: str | Path, expected_size: int | None = None,
                             study_ids: list[str] | None = None, verify_checksums: bool = False,
                             version: str = RAPTOR_PREPROCESS_VERSION) -> dict[str, Any]:
    """Verify metadata/header consistency; optionally read and hash every cache."""
    variant = raptor_variant(version)
    if expected_size is not None and expected_size != variant["size"]:
        raise ValueError(f"Raptor expects cache size {variant['size']}, got {expected_size}")
    manifest = Path(manifest).resolve()
    table = pd.read_csv(manifest, dtype={STUDY_ID: str})
    required = {STUDY_ID, "cache_path", "metadata_path", "num_slices", "image_size", "num_valid_windows",
                "preprocess_version", "preprocess_fingerprint", "source_fingerprint", "cache_sha256", "metadata_sha256"}
    if not required.issubset(table.columns):
        raise ValueError(f"Raptor manifest missing columns: {sorted(required - set(table.columns))}")
    if study_ids is not None:
        requested = set(map(str, study_ids))
        if requested - set(table[STUDY_ID]):
            raise ValueError("Requested studies are missing from the Raptor manifest")
        table = table.loc[table[STUDY_ID].isin(requested)]
    if table.empty or table[list(required)].isna().any().any() or table[STUDY_ID].duplicated().any():
        raise ValueError("Raptor manifest must contain nonmissing, unique studies")
    if table["cache_path"].duplicated().any() or table["metadata_path"].duplicated().any():
        raise ValueError("Raptor manifest reuses a cache path for different studies")
    cache_bytes, windows = 0, 0
    for row in table.to_dict(orient="records"):
        try:
            cache_path, metadata_path = (manifest.parent / row["cache_path"], manifest.parent / row["metadata_path"])
            if metadata_path != cache_path.with_suffix(".json"):
                raise ValueError("Cache and metadata do not form a pair")
            metadata = _read_metadata(metadata_path, row["metadata_sha256"], version)
            for key in (STUDY_ID, "preprocess_version", "preprocess_fingerprint", "source_fingerprint", "cache_sha256"):
                if metadata[key] != row[key]:
                    raise ValueError(f"Manifest and metadata disagree on {key}")
            if row["image_size"] != variant["size"] or row["num_slices"] != variant["depth"]:
                raise ValueError("Manifest image shape differs from the Raptor recipe")
            if row["num_valid_windows"] != len(metadata["allowed_centers"]):
                raise ValueError("Manifest and metadata disagree on valid window count")
            cache_bytes += _check_array(cache_path, variant["depth"], variant["size"])
            if verify_checksums and _sha256_file(cache_path) != row["cache_sha256"]:
                raise ValueError("Cached image checksum mismatch")
            windows += len(metadata["allowed_centers"])
        except (OSError, ValueError, KeyError, TypeError, AttributeError) as exc:
            raise ValueError(f"Invalid Raptor cache for {row[STUDY_ID]}: {exc}") from exc
    return {"studies": len(table), "series": int(table.get("num_slots_used", pd.Series(dtype=int)).sum()),
            "slices": len(table) * variant["depth"], "image_size": variant["size"],
            "preprocess_version": version, "cache_bytes": cache_bytes,
            "valid_windows": windows, "array_checksums_verified": bool(verify_checksums),
            "image_preparation": variant["contract"]}


# Unit vectors in patient coordinates (+x patient left, +y posterior, +z superior).
_ANTERIOR, _SUPERIOR = np.array([0., -1., 0.]), np.array([0., 0., 1.])


def _slice_geometry(slot: dict[str, Any], local_index: int):
    orientations, positions = slot.get("selected_orientations_lps"), slot.get("selected_image_positions_lps")
    if not orientations or not positions or orientations[local_index] is None or positions[local_index] is None:
        return None
    return np.asarray(orientations[local_index], float), np.asarray(positions[local_index], float)


def knee_side(slots: list[dict[str, Any]]) -> float:
    """+1 for a left knee, -1 for a right knee, 0 when unknown.

    The scanned knee lies on its own side of the patient (+x is the patient's
    left), so the center of a front or top view gives the side. This matched the
    DICOM laterality in 2,138 of 2,203 training studies that record it.
    """
    for index in (2, 3, 4):
        slot = slots[index]
        if slot[SERIES_ID] is None:
            continue
        middle = len(slot["selected_source_indices"]) // 2
        geometry = _slice_geometry(slot, middle)
        if geometry is None:
            continue
        orientation, position = geometry
        rows, columns = slot["crops"][middle]["source_shape"]
        row_spacing, column_spacing = slot["selected_pixel_spacings_mm"][middle]
        center = position + orientation[:3] * columns / 2 * column_spacing + orientation[3:] * rows / 2 * row_spacing
        return 1.0 if center[0] > 0 else -1.0
    return 0.0


def window_features(slots: list[dict[str, Any]], centers: list[int]) -> np.ndarray:
    """Describe each window: plane, sequence flags, knee side, and anatomical offset.

    The offset is the middle slice's distance from its stack's center in units of
    40 mm: toward the outer (lateral) side for sagittal windows, the front for
    coronal windows, and the head for axial windows. Missing geometry gives 0.
    """
    side = knee_side(slots)
    directions = {"Sagittal": np.array([side, 0., 0.]), "Coronal": _ANTERIOR, "Axial": _SUPERIOR}
    by_center, stack_centers = {}, {}
    for slot in slots:
        if slot[SERIES_ID] is None:
            continue
        filled = len(slot["selected_source_indices"])
        points = [_slice_geometry(slot, index) for index in range(filled)]
        known = [point[1] for point in points if point is not None]
        stack_centers[slot["slot"]] = np.mean(known, axis=0) if known else None
        for local_index in range(filled):
            by_center[slot["offset"] + local_index] = (slot, points[local_index])
    result = np.zeros((len(centers), len(WINDOW_FEATURES)), dtype=np.float32)
    for row, center in enumerate(centers):
        slot, point = by_center[center]
        result[row, PLANES.index(slot["plane"])] = 1.0
        result[row, 3:5] = [float(slot["Fluid_Sensitive"]), float(slot["Fat_Suppression"])]
        result[row, SIDE_FEATURE] = side
        middle = stack_centers[slot["slot"]]
        if point is not None and middle is not None:
            result[row, -1] = np.clip(float((point[1] - middle) @ directions[slot["plane"]]) / 40.0, -2.0, 2.0)
    return result


def load_window_guide(path, records: dict[str, dict[str, Any]], top: int) -> dict[str, np.ndarray]:
    """Guided training windows (Oct 17, 2026): per study, a [findings, top] array of window indices into the study's
    allowed centers (-1 = none), read from a CSV with columns StudyInstanceUID, finding, rank (1 = most attended) and
    center (the window's middle slice in the cache). Studies outside ``records`` are skipped; a center the cache does
    not allow means the guide was made from a different cache."""
    if not path or isinstance(top, bool) or top < 1:
        raise ValueError("A window guide needs a file and a positive top count")
    table = pd.read_csv(path, dtype={STUDY_ID: str})
    missing = {STUDY_ID, "finding", "rank", "center"} - set(table.columns)
    if missing:
        raise ValueError(f"Window guide lacks columns {sorted(missing)}")
    if not set(table["finding"]) <= set(TARGETS) or table.duplicated([STUDY_ID, "finding", "rank"]).any():
        raise ValueError("Window guide has unknown findings or repeated ranks")
    table = table[(table["rank"] >= 1) & (table["rank"] <= top) & table[STUDY_ID].isin(records)]
    guide = {}
    for study, rows in table.groupby(STUDY_ID, sort=False):
        position = {int(center): index for index, center in enumerate(records[study]["centers"])}
        found = np.full((len(TARGETS), top), -1, dtype=np.int64)
        for finding, rank, center in zip(rows["finding"], rows["rank"], rows["center"], strict=True):
            if int(center) not in position:
                raise ValueError(f"Window guide center {center} is not an allowed window of {study}")
            found[list(TARGETS).index(finding), int(rank) - 1] = position[int(center)]
        guide[study] = found
    return guide


def _series_draw(group: np.ndarray, number: int, rng: np.random.Generator, spread: bool = False) -> np.ndarray:
    """A series' share of training windows (group = its window positions in depth order). Spread (Oct 17, 2026): one
    random window from each of `number` equal stretches of the series; otherwise random positions, as before."""
    if not spread or number >= len(group):
        return rng.choice(group, size=number, replace=len(group) < number)
    return np.array([rng.choice(part) for part in np.array_split(group, number)], dtype=group.dtype)


class RaptorDataset(Dataset):
    """One bag per study; image values remain [0,1] for the model to normalize."""

    def __init__(self, manifest: str | Path, studies: pd.DataFrame, config: dict,
                 training: bool = False, seed: int = 42) -> None:
        self.manifest_path = Path(manifest).resolve()
        self.training, self.seed = bool(training), int(seed)
        self.image_size = int(config.get("image_size", 384))
        self.windows = int(config.get("raptor_train_windows", 12) if training else config.get("raptor_eval_windows", 42))
        self.augment = bool(config.get("augmentation", {}).get("enabled", True))
        # Older checkpoints lack these keys; their defaults keep the original random draws.
        self.sampling = config.get("raptor_window_sampling", "random")
        self.version = config.get("raptor_variant", RAPTOR_PREPROCESS_VERSION)
        variant = raptor_variant(self.version)
        self.depth, self.cache_size = variant["depth"], variant["size"]
        self.view = config.get("raptor_view", "resize")
        self.series_dropout = float(config.get("raptor_series_dropout", 0.0))
        # Windows spread within each series (Oct 17, 2026; stratified draws only).
        self.window_spread = bool(config.get("raptor_window_spread", False))
        # Fine-tuning on all windows (Oct 17, 2026): training uses every window of the kept series instead of a draw.
        self.all_windows = bool(config.get("raptor_train_all_windows", False)) and self.training
        # Optional subset of cache slots to read windows from (the knee-centred test reads only the main slots of the
        # close-up cache).
        self.slot_subset = None if config.get("raptor_slots") is None else np.asarray(config["raptor_slots"], dtype=np.int64)
        if self.image_size < 1 or self.windows < 1 or self.seed < 0:
            raise ValueError("Positive image/window sizes and a nonnegative seed are required")
        if self.sampling not in ("random", "stratified") or not 0 <= self.series_dropout < 1:
            raise ValueError("raptor_window_sampling must be random/stratified and raptor_series_dropout in [0, 1)")
        if config.get("cache_size", self.cache_size) != self.cache_size:
            raise ValueError(f"Raptor cache_size must be {self.cache_size} for {self.version}")
        if self.view not in RAPTOR_VIEWS or (self.view == "center_crop" and self.image_size > self.cache_size):
            raise ValueError(f"raptor_view must be one of {RAPTOR_VIEWS}; center_crop needs image_size <= cache_size")
        if STUDY_ID not in studies or studies[STUDY_ID].isna().any() or studies[STUDY_ID].duplicated().any():
            raise ValueError("Studies must have unique, nonmissing study IDs")
        self.studies = studies.reset_index(drop=True).copy()
        self.studies[STUDY_ID] = self.studies[STUDY_ID].astype(str)
        labels = self.studies.reindex(columns=TARGETS).apply(pd.to_numeric, errors="raise")
        values = labels.to_numpy(dtype=np.float32, na_value=np.nan)
        known = ~np.isnan(values)
        if not np.isfinite(values[known]).all() or ((values[known] < 0) | (values[known] > 1)).any():
            raise ValueError("Known targets must be probabilities between zero and one")
        self.targets, self.target_masks = np.where(known, values, 0.).astype(np.float32), known
        self._epoch = torch.zeros((), dtype=torch.int64).share_memory_()
        table = pd.read_csv(self.manifest_path, dtype={STUDY_ID: str})
        if table[STUDY_ID].duplicated().any():
            raise ValueError("Raptor manifest repeats study IDs")
        rows = table.set_index(STUDY_ID).to_dict(orient="index")
        self._records = {}
        for study in self.studies[STUDY_ID]:
            if study not in rows:
                raise ValueError(f"No Raptor cache for {study}")
            row = rows[study]
            metadata = _read_metadata(self.manifest_path.parent / row["metadata_path"], row["metadata_sha256"], self.version)
            if metadata[STUDY_ID] != study:
                raise ValueError("Study ID and cached provenance disagree")
            features = np.zeros(5, dtype=np.float32)
            present = [slot for slot in metadata["slots"] if slot[SERIES_ID] is not None]
            for slot in present:
                features[PLANES.index(slot["plane"])] = 1.
            features[3:] = [np.mean([slot[flag] for slot in present]) for flag in ("Fluid_Sensitive", "Fat_Suppression")]
            # Previews show depth within the actual source series, not an index
            # into the artificial concatenation of five unrelated MRI series.
            source_positions, center_slots = {}, {}
            for slot in present:
                for local_index, source_index in enumerate(slot["selected_source_indices"]):
                    source_positions[slot["offset"] + local_index] = source_index / max(slot["source_depth"] - 1, 1)
                    center_slots[slot["offset"] + local_index] = slot["slot"]
            self._records[study] = {"path": self.manifest_path.parent / row["cache_path"],
                                    "centers": np.asarray(metadata["allowed_centers"], dtype=np.int64),
                                    "positions": np.asarray([source_positions[center] for center in metadata["allowed_centers"]], dtype=np.float32),
                                    "planes": np.asarray(metadata["window_planes"], dtype=np.int64),
                                    "slots": np.asarray([center_slots[center] for center in metadata["allowed_centers"]], dtype=np.int64),
                                    "window_features": window_features(metadata["slots"], metadata["allowed_centers"]),
                                    "metadata": features}
            if self.slot_subset is not None:
                record = self._records[study]
                keep = np.isin(record["slots"], self.slot_subset)
                if not keep.any():
                    raise ValueError(f"No windows in data.raptor_slots for {study}")
                for key in ("centers", "positions", "planes", "slots", "window_features"):
                    record[key] = record[key][keep]
        # Guided windows (Oct 17, 2026): in training, data.raptor_guided_windows of each study's windows are one of the
        # top windows of an earlier model's attention (data.raptor_guide_file) for that many randomly chosen findings;
        # the rest are drawn as before. Studies not in the guide keep the ordinary draw; prediction never reads it.
        self.guided_windows = int(config.get("raptor_guided_windows", 0)) if self.training else 0
        self._guide: dict[str, np.ndarray] = {}
        if self.guided_windows:
            if not 0 < self.guided_windows < self.windows or self.guided_windows > len(TARGETS):
                raise ValueError("raptor_guided_windows must be below the training window count and at most the finding count")
            self._guide = load_window_guide(config.get("raptor_guide_file"), self._records,
                                            int(config.get("raptor_guide_top", 3)))

    def __len__(self) -> int:
        return len(self.studies)

    def set_epoch(self, epoch: int) -> None:
        if int(epoch) != epoch or epoch < 0:
            raise ValueError("epoch must be a nonnegative integer")
        self._epoch.fill_(int(epoch))

    def _sample_windows(self, record: dict[str, Any], rng: np.random.Generator,
                        guide: np.ndarray | None = None) -> np.ndarray:
        """Optionally hide whole series, then draw training windows overall or per series.

        Hidden series mimic studies that lack a sequence; each plane in the study
        always keeps at least one series. Stratified draws give every kept series
        a share proportional to its windows, so no view is missed by chance.
        """
        slots, planes = record["slots"], record["planes"]
        present = np.unique(slots)
        kept = present[rng.random(len(present)) >= self.series_dropout]
        for plane in np.unique(planes):
            plane_slots = np.unique(slots[planes == plane])
            if not np.isin(plane_slots, kept).any():
                kept = np.append(kept, rng.choice(plane_slots))
        candidates = np.flatnonzero(np.isin(slots, kept))
        if getattr(self, "all_windows", False):
            return candidates
        windows, guided = self.windows, None
        if guide is not None:
            # One of the top windows (in a kept series) of each of guided_windows randomly chosen findings; the rest
            # of the draw then comes from the other windows.
            picks = []
            for finding in rng.choice(len(guide), size=self.guided_windows, replace=False):
                top = guide[finding][guide[finding] >= 0]
                top = top[np.isin(top, candidates)]
                if len(top):
                    picks.append(rng.choice(top))
            guided = np.unique(np.asarray(picks, dtype=np.int64))
            rest = np.setdiff1d(candidates, guided)
            candidates = rest if len(rest) else candidates
            windows -= len(guided)
        if self.sampling == "random":
            drawn = rng.choice(candidates, size=windows, replace=len(candidates) < windows)
        else:
            groups = [candidates[slots[candidates] == slot] for slot in np.sort(kept)]
            sizes = np.array([len(group) for group in groups], dtype=float)
            quotas = sizes / sizes.sum() * windows
            counts = np.floor(quotas).astype(int)
            # Largest remainders fill the windows left after rounding down.
            for position in np.argsort(-(quotas - counts), kind="stable")[: windows - counts.sum()]:
                counts[position] += 1
            drawn = np.concatenate([_series_draw(group, number, rng, getattr(self, "window_spread", False))
                                    for group, number in zip(groups, counts, strict=True) if number > 0])
        return drawn if guided is None else np.sort(np.concatenate([guided, drawn]))

    def __getitem__(self, index: int) -> dict[str, Any]:
        study = self.studies.iloc[index][STUDY_ID]
        record = self._records[study]
        identity = int.from_bytes(hashlib.blake2b(study.encode(), digest_size=8).digest(), "little")
        rng = np.random.default_rng(np.random.SeedSequence([self.seed, int(self._epoch.item()) if self.training else 0,
                                                           identity & 0xffffffff, identity >> 32]))
        count = len(record["centers"])
        if self.training and (self.sampling != "random" or self.series_dropout > 0 or self.all_windows or self._guide):
            choices = self._sample_windows(record, rng, self._guide.get(study))
        elif self.training:
            choices = rng.choice(count, size=self.windows, replace=count < self.windows)
        else:
            choices = np.linspace(0, count - 1, self.windows).round().astype(int)
        centers = record["centers"][choices]
        volume = np.load(record["path"], mmap_mode="r", allow_pickle=False)
        try:
            if volume.shape != (self.depth, self.cache_size, self.cache_size) or volume.dtype != np.uint8:
                raise ValueError(f"Wrong Raptor cache shape or dtype: {record['path']}")
            indices = centers[:, None] + np.array([-1, 0, 1])
            if self.view == "center_crop":
                # Keep the central pixels at their cached detail; only these rows and columns are read.
                top = (self.cache_size - self.image_size) // 2
                window = slice(top, top + self.image_size)
                images = torch.from_numpy(np.asarray(volume[indices, window, window], dtype=np.float32) / 255.)
            else:
                images = torch.from_numpy(np.asarray(volume[indices], dtype=np.float32) / 255.)
        finally:
            if isinstance(volume, np.memmap):
                volume._mmap.close()
        if self.view == "resize" and self.image_size != self.cache_size:
            images = F.interpolate(images, (self.image_size, self.image_size), mode="bilinear", align_corners=False)
        if self.training and self.augment:
            images = (images * float(rng.uniform(.9, 1.1))).clamp_(0., 1.)
        return {"images": images.unsqueeze(0), "series_mask": torch.ones(1, dtype=torch.bool),
                "metadata": torch.from_numpy(record["metadata"].copy()).unsqueeze(0),
                "positions": torch.from_numpy(record["positions"][choices].copy()).unsqueeze(0),
                "window_planes": torch.from_numpy(record["planes"][choices].copy()).unsqueeze(0),
                # Which series each window comes from (for the optional per-series training loss).
                "window_slots": torch.from_numpy(record["slots"][choices].copy()).unsqueeze(0),
                "window_features": torch.from_numpy(record["window_features"][choices].copy()).unsqueeze(0),
                "targets": torch.from_numpy(self.targets[index].copy()),
                "target_mask": torch.from_numpy(self.target_masks[index].copy()), "study_id": study}
