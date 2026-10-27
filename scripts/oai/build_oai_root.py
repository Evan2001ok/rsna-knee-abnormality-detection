#!/usr/bin/env python3
"""Unpack downloaded OAI series into the competition's folder layout, so scripts/prepare_raptor.py can read them.

Writes <root>/train.csv, <root>/train_series.csv and <root>/train_series/<study>/<series>/<file>.dcm, where a study is
one knee at one visit (OAI_<id>_<SIDE>_<visit>) and the series are named after their OAI description. Only knees with
all three wanted series downloaded are included. Series flags follow the challenge's meaning of "fluid sensitive"
(fluid bright and fat suppressed): the sagittal intermediate-weighted fat-suppressed TSE and the axial reformat of the
water-excitation DESS are fluid sensitive; the coronal intermediate-weighted TSE (no fat suppression) is not.
Each axial reformat also stores one sagittal reference image; images whose orientation differs from the majority of
their series are dropped, and so are files that are not DICOM (some follow-up archives also hold NDA's image03.json).
"""
from __future__ import annotations

import argparse
import tarfile
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd
import pydicom
from pydicom.errors import InvalidDicomError

SERIES = {  # OAI kind -> (plane, fluid sensitive, fat suppressed)
    "SAG_IW_TSE": ("Sagittal", 1, 1),
    "COR_IW_TSE": ("Coronal", 0, 0),
    "AX_MPR": ("Axial", 1, 1),
}


def unpack(job: tuple[str, str]) -> str:
    archive, target = Path(job[0]), Path(job[1])
    done = target / ".complete"
    if done.exists():
        return "reused"
    target.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive) as tar:
        for member in tar.getmembers():
            if not member.isfile():
                continue
            name = Path(member.name).name
            if not name or name.startswith(".") or "/" in name or "\\" in name:
                continue  # only plain file names; never follow paths from the archive
            source = tar.extractfile(member)
            with open(target / f"{name}.dcm", "wb") as out:
                out.write(source.read())
    orientation = {}
    for path in target.glob("*.dcm"):
        try:
            header = pydicom.dcmread(path, stop_before_pixels=True)
        except InvalidDicomError:
            path.unlink()
            continue
        orientation[path] = tuple(round(float(x), 3) for x in header.ImageOrientationPatient)
    majority = Counter(orientation.values()).most_common(1)[0][0]
    for path, value in orientation.items():
        if value != majority:
            path.unlink()
    done.touch()
    return "unpacked"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--raw", type=Path, required=True, help="Folder the downloader wrote to")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--limit", type=int, help="Only the first N complete knees (for a test)")
    parser.add_argument("--chunk", type=int, default=0, help="Which chunk of the complete knees to unpack (0-based)")
    parser.add_argument("--chunks", type=int, default=1, help="Split the complete knees into this many chunks")
    args = parser.parse_args()
    sel = pd.read_csv(args.selection, dtype={"src_subject_id": str})
    sel = sel[sel["kind"].isin(SERIES)]
    sel["archive"] = [str(args.raw / a) for a in sel["download_alias"]]
    sel = sel[[Path(a).exists() for a in sel["archive"]]]
    sel["study"] = "OAI_" + sel["src_subject_id"] + "_" + sel["side"] + "_" + sel["visit"]
    counts = sel.groupby("study")["kind"].nunique()
    complete = sorted(counts[counts == len(SERIES)].index)
    if args.limit:
        complete = complete[:args.limit]
    size = -(-len(complete) // args.chunks)  # chunks keep disk use low: unpack, prepare, delete, next
    complete = complete[args.chunk * size:(args.chunk + 1) * size]
    sel = sel[sel["study"].isin(set(complete))].drop_duplicates(["study", "kind"])
    jobs = [(a, str(args.root / "train_series" / s / k)) for a, s, k in zip(sel["archive"], sel["study"], sel["kind"])]
    with ProcessPoolExecutor(args.workers) as pool:
        results = list(pool.map(unpack, jobs, chunksize=8))
    series = pd.DataFrame({"StudyInstanceUID": sel["study"], "SeriesInstanceUID": sel["kind"],
                           "Fluid_Sensitive": [SERIES[k][1] for k in sel["kind"]],
                           "Fat_Suppression": [SERIES[k][2] for k in sel["kind"]],
                           "Anatomical_Plane": [SERIES[k][0] for k in sel["kind"]]})
    args.root.mkdir(parents=True, exist_ok=True)
    series.sort_values(["StudyInstanceUID", "SeriesInstanceUID"]).to_csv(args.root / "train_series.csv", index=False)
    pd.DataFrame({"StudyInstanceUID": complete}).to_csv(args.root / "train.csv", index=False)
    print(f"{len(complete)} knees, {len(jobs)} series ({results.count('unpacked')} unpacked, "
          f"{results.count('reused')} reused) -> {args.root}")


if __name__ == "__main__":
    main()
