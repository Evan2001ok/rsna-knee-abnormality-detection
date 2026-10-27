#!/usr/bin/env python3
"""Copy an OAI combined training folder with only the OAI rows' labels replaced (Oct 16, 2026; used to test the
hybrid OAI labels of prepare_oai_hybrid_labels.py). An earlier script (not included) did the opposite and swapped the
competition rows. Competition rows, manifest, splits and validation labels stay byte-identical; the OAI study sets must
match exactly.

Run from the repo root (oai_v00_336_v7own: OAI baseline knees, 336 px cache, v7 labels = our round-5 labels, and our
own OAI labels; the copy gets the hybrid OAI labels):
  .venv/bin/python scripts/oai/swap_oai_labels.py data/combined/oai_v00_336_v7own \
      data/labels/oai_hybrid_v1v3/labels.csv data/combined/oai_v00_336_v7oaihybrid
"""

from __future__ import annotations

import hashlib
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ID = "StudyInstanceUID"


def main() -> None:
    src, new_labels, dst = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
    if dst.exists():
        raise FileExistsError(dst)
    combined = pd.read_csv(src / "labels.csv", dtype={ID: str})
    new = pd.read_csv(new_labels, dtype={ID: str}).set_index(ID)
    targets = [c for c in combined.columns if c not in (ID, "label_source")]
    if len(targets) != 12 or not set(targets) <= set(new.columns):
        raise ValueError(f"Unexpected label columns: {targets}")
    oai = combined["label_source"] == "oai"
    ids = combined.loc[oai, ID]
    if not set(ids) <= set(new.index):
        raise ValueError("Every OAI training row needs a new label")
    swapped = combined.copy()
    swapped.loc[oai, targets] = new.loc[ids, targets].to_numpy()
    if not swapped.loc[~oai].equals(combined.loc[~oai]):
        raise ValueError("Competition rows must stay unchanged")
    values = swapped.loc[oai, targets].to_numpy(float)
    if not np.all(np.isnan(values) | ((values >= 0) & (values <= 1))):
        raise ValueError("Labels must be within [0, 1]")
    shutil.copytree(src, dst)
    swapped.to_csv(dst / "labels.csv", index=False)
    digest = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()  # noqa: E731
    for name in ("manifest.csv", "splits.csv", "validation_labels.csv"):
        if digest(src / name) != digest(dst / name):
            raise ValueError(f"{name} changed")
    changed = int((swapped.loc[oai, targets].fillna(-1).to_numpy() != combined.loc[oai, targets].fillna(-1).to_numpy()).sum())
    provenance = {"source": str(src), "new_oai_labels": str(new_labels), "new_oai_labels_sha256": digest(new_labels),
                  "oai_rows": int(oai.sum()), "competition_rows": int((~oai).sum()), "changed_cells": changed,
                  "labels_sha256": digest(dst / "labels.csv"),
                  "rule": f"OAI rows from {new_labels}; competition rows and every other file copied from {src}"}
    (dst / "provenance.json").write_text(json.dumps(provenance, indent=1) + "\n")
    print(json.dumps(provenance))


if __name__ == "__main__":
    main()
