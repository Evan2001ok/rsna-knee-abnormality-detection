"""Knee anchor points in patient coordinates, placed in stored DICOM slices to centre crops on the knee.

An anchor record (one JSON per study, ``<StudyInstanceUID>.json``) holds ``joint_lps``: the knee joint centre in patient
coordinates (LPS, mm), for example from the knee finder in the public notebook of R7 (heliosli's public ConvNeXt-small
model); we saved its points with a separate script that is not included here. Because the point is physical, it is
placed in each slice with that slice's own DICOM geometry, as stored, without any canonical flips.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np


def project_to_pixel(point_lps, orientation, position, pixel_spacing) -> tuple[float, float, float]:
    """Row, column and signed distance from the slice plane (mm) of a patient-coordinate point in one DICOM slice.

    ``orientation`` is ImageOrientationPatient (row direction cosine, then column direction cosine), ``position`` is
    ImagePositionPatient (centre of the first pixel) and ``pixel_spacing`` is PixelSpacing (row spacing, column
    spacing), so column = offset . row_cosine / column_spacing and row = offset . column_cosine / row_spacing.
    """
    o = np.asarray(orientation, float)
    offset = np.asarray(point_lps, float) - np.asarray(position, float)
    if o.shape != (6,) or offset.shape != (3,):
        raise ValueError("Expected six direction cosines and three-dimensional points")
    column = float(offset @ o[:3]) / float(pixel_spacing[1])
    row = float(offset @ o[3:]) / float(pixel_spacing[0])
    distance = float(offset @ np.cross(o[:3], o[3:]))
    return row, column, distance


def centred_box(row: float, column: float, size: int, height: int, width: int) -> tuple[int, int]:
    """Top-left corner of a ``size`` square centred on (row, column), shifted to stay inside the image.

    For the image centre ((height - 1) / 2, (width - 1) / 2) this gives the plain centre crop's corner,
    ((height - size) // 2, (width - size) // 2).
    """
    if size > height or size > width or size < 1:
        raise ValueError("The crop must fit inside the image")
    top = int(math.floor(row - size / 2 + .5))
    left = int(math.floor(column - size / 2 + .5))
    return min(max(top, 0), height - size), min(max(left, 0), width - size)


def read_anchor(anchors_dir: str | Path, study: str) -> dict[str, Any]:
    """The study's anchor record reduced to what the cache needs: the joint and, when found, the kneecap (LPS, mm)."""
    path = Path(anchors_dir) / f"{study}.json"
    if not path.is_file():
        return {"joint_lps": None, "patella_lps": None, "method": "missing_record"}
    record = json.loads(path.read_text())
    points = {}
    for key in ("joint_lps", "patella_lps"):
        point = record.get(key) if record.get("joint_lps") is not None else None
        if point is not None:
            point = [float(value) for value in point]
            if len(point) != 3 or not all(math.isfinite(value) for value in point):
                raise ValueError(f"Invalid {key} for {study}")
        points[key] = point
    return {**points, "method": record.get("method") or ("error" if record.get("error") else None)}
