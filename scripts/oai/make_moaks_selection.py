#!/usr/bin/env python3
"""Download list for one OAI image package: the three series we use for every knee graded with MOAKS (MRI
Osteoarthritis Knee Score) at that visit.

Reads the package's image03.txt (image index) and files.csv (package file IDs and sizes, from the NDA API) under
data/oai/packages/<package>/, keeps SAG_IW_TSE, COR_IW_TSE and AX_MPR for knees in the visit's MOAKS table, and writes
selection_<visit>_moaks.csv in the downloader's format (package_file_id, download_alias, file_size, ...).
Usage: make_moaks_selection.py --package <package-id> --visit 01
(<package-id> is the ID of the OAI image package you created on the NIMH Data Archive.)
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

KINDS = ["SAG_IW_TSE", "COR_IW_TSE", "AX_MPR"]
MOAKS = Path("data/oai/assessments/Semi-Quant Scoring_ASCII")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", required=True)
    parser.add_argument("--visit", required=True, help="two-digit MOAKS visit code, e.g. 00, 01, 08")
    args = parser.parse_args()
    folder = Path("data/oai/packages") / args.package
    index = pd.read_csv(folder / "image03.txt", sep="\t", skiprows=[1], low_memory=False,
                        usecols=["src_subject_id", "image_file", "image_description", "visit"])
    files = pd.read_csv(folder / "files.csv")
    index["alias"] = "image03/" + index["image_file"].str.split("/", n=4).str[4]
    index = index.merge(files, left_on="alias", right_on="download_alias")
    index["kind"] = index["image_description"].str.replace(r"_(RIGHT|LEFT)$", "", regex=True)
    index["side"] = index["image_description"].str.extract(r"_(RIGHT|LEFT)$")[0]
    if set(index["visit"]) != {f"V{args.visit}"}:
        raise ValueError(f"Package {args.package} holds visits {sorted(set(index['visit']))}, not V{args.visit}")
    moaks = pd.read_csv(MOAKS / f"kMRI_SQ_MOAKS_BICL{args.visit}.txt", sep="|", low_memory=False, encoding="latin-1")
    moaks.columns = [c.upper() if c.lower() in {"id", "side"} else c for c in moaks.columns]
    side = pd.to_numeric(moaks["SIDE"].astype(str).str.extract(r"^\s*(\d)", expand=False)).map({1: "RIGHT", 2: "LEFT"})
    knees = set(zip(moaks["ID"].astype(str), side))
    graded = pd.Series([(str(a), b) in knees for a, b in zip(index["src_subject_id"], index["side"], strict=True)],
                       index=index.index)
    chosen = index[index["kind"].isin(KINDS) & graded]
    chosen = chosen[["package_file_id", "download_alias", "file_size", "src_subject_id", "side", "kind",
                     "image_description", "visit"]]
    if chosen["package_file_id"].duplicated().any():
        raise ValueError("A package file was selected twice")
    output = folder / f"selection_v{args.visit}_moaks.csv"
    chosen.to_csv(output, index=False)
    complete = chosen.groupby(["src_subject_id", "side"])["kind"].nunique().eq(len(KINDS)).sum()
    print(f"V{args.visit}: {len(knees)} MOAKS knees, {len(chosen)} series ({chosen['file_size'].sum() / 1e9:.1f} GB), "
          f"{complete} knees with all three series -> {output}")


if __name__ == "__main__":
    main()
