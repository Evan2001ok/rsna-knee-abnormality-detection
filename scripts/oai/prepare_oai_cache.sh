#!/usr/bin/env bash
# Unpack the downloaded OAI series chunk by chunk, build the Raptor image cache for each chunk, delete the unpacked
# files, then merge the chunk manifests (absolute paths) into one. Any visit's selection works (we used baseline V00).
# Run from the repo root:
#   scripts/oai/prepare_oai_cache.sh <variant> <cache-dir> <chunks> <selection-csv>
# <chunks>: how many parts to unpack one after another, to keep disk use low ("" means 8).
# <selection-csv>: the list from make_moaks_selection.py, e.g. data/oai/packages/<package-id>/selection_v00_moaks.csv.
# Optional settings: RAW (folder the downloader wrote to, default data/oai/raw), OAI_WORK (where each chunk is unpacked
# for a while, default data/oai), PYTHON (default .venv/bin/python), WORKERS (preparation workers, default 10).
set -euo pipefail
VARIANT=$1; CACHE=$2; CHUNKS=${3:-8}
SELECTION=${4:?"give the selection CSV, e.g. data/oai/packages/<package-id>/selection_v00_moaks.csv"}
RAW=${RAW:-data/oai/raw}; OAI_WORK=${OAI_WORK:-data/oai}; PY=${PYTHON:-.venv/bin/python}
WORKERS=${WORKERS:-10}
ROOT=$OAI_WORK/root-$(basename "$CACHE")  # one unpack folder per cache, so visits can run side by side
for k in $(seq 0 $((CHUNKS - 1))); do
  if [ -f "$CACHE/chunk$k/manifest.csv" ]; then echo "chunk $k done"; continue; fi
  rm -rf "$ROOT"
  "$PY" scripts/oai/build_oai_root.py --selection "$SELECTION" --raw "$RAW" \
    --root "$ROOT" --chunk "$k" --chunks "$CHUNKS" --workers 12
  (PYTHONPATH=src "$PY" scripts/prepare_raptor.py --data-root "$ROOT" \
    --output-dir "$CACHE/chunk$k" --variant "$VARIANT" --workers "$WORKERS" | tail -2)
  rm -rf "$ROOT"
done
"$PY" - "$CACHE" "$CHUNKS" <<'PY'
import sys
from pathlib import Path
import pandas as pd
cache, chunks = Path(sys.argv[1]).resolve(), int(sys.argv[2])
parts = []
for k in range(chunks):
    m = pd.read_csv(cache / f"chunk{k}" / "manifest.csv", dtype={"StudyInstanceUID": str})
    for column in ("cache_path", "metadata_path"):
        m[column] = [str(cache / f"chunk{k}" / v) for v in m[column]]
    parts.append(m)
merged = pd.concat(parts, ignore_index=True)
assert not merged["StudyInstanceUID"].duplicated().any()
merged.to_csv(cache / "manifest.csv", index=False)
print(f"merged {len(merged)} OAI knees into {cache / 'manifest.csv'}")
PY
