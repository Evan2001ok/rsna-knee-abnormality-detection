#!/usr/bin/env python3
# Kept exactly as it ran on our rented cloud machine on October 17, 2026. Only these comment lines are new.
# It made splits_all.csv for the final training folder, data/combined/oai_v00_336_consensus58_4f/: the 100 held-out
# studies go to fold 6, and every other row keeps its fold and trains. It read those 100 studies from an earlier
# folder, data/combined/oai_v00_336_v7own/splits_all.csv (not included, like every training folder). Its fold 6 held
# the same 100 studies as fold 6 of data/splits/all_studies/splits.csv. Step 2.6 of entry_points.md rebuilds that file
# and does this step from it.
# The second half wrote the four base configs from coatnet_384_consensus58_fold0.yaml in the cloud copy of
# configs/backbones/ (CFGS below). That file is not in this repository (scripts/make_consensus_combined.py writes it
# from fold configs that are not included either), so this half stops with an error here. The four configs it wrote
# are configs/final/coatnet_384_consensus58_all_s42.yaml to s45.yaml, as each run recorded them in its config.yaml.
# The record named below (experiments/consensus-fulldata-20261017.json) is in our private repository.
"""Cloud (Oct 17, 2026; experiments/consensus-fulldata-20261017.json): splits_all.csv for the consensus58 folder (the same
100 placeholder studies as oai_v00_336_v7own/splits_all.csv moved to fold 6) and the full-data configs for seeds 42-45."""
import json
from pathlib import Path

import pandas as pd
import yaml

ID = "StudyInstanceUID"
FOLDER = Path("data/combined/oai_v00_336_consensus58_4f")
CFGS = Path("/workspace/rsna-crop200/configs/backbones")
src = pd.read_csv(FOLDER / "splits.csv")
old = pd.read_csv("data/combined/oai_v00_336_v7own/splits_all.csv")
placeholder = set(old.loc[old["fold"] == 6, ID])
assert len(placeholder) == 100
rows = src[ID].isin(placeholder)
assert rows.sum() == 100 and src.loc[rows, "fold"].between(0, 3).all()
out = src.copy()
out.loc[rows, "fold"] = 6
target = FOLDER / "splits_all.csv"
assert not target.exists()
out.to_csv(target, index=False)
counts = out["fold"].value_counts().sort_index().to_dict()
assert counts[6] == 100 and sum(v for k, v in counts.items() if k != 6) == 6705
text = (CFGS / "coatnet_384_consensus58_fold0.yaml").read_text()
written = {}
for seed in (42, 43, 44, 45):
    run = f"coatnet_384_consensus58_all_s{seed}"
    new = text
    for a, b in (("seed: 42\n", f"seed: {seed}\n"),
                 (f"  splits: {FOLDER}/splits.csv\n", f"  splits: {FOLDER}/splits_all.csv\n"),
                 ("  fold: 0\n", "  fold: 6\n"),
                 ("  output_dir: runs/coatnet_384_consensus58_fold0\n", f"  output_dir: runs/{run}\n"),
                 ("  validation_start_epoch: 17\n", "  validation_start_epoch: 22\n")):
        assert new.count(a) == 1, a
        new = new.replace(a, b)
    note = (f"# Oct 17, 2026 (experiments/consensus-fulldata-20261017.json): coatnet_384_consensus58_fold0.yaml as a full-data "
            f"model, seed {seed}: every study trains except 100 placeholder validation studies (fold 6 of splits_all.csv), averaged "
            f"over epochs 22-24; the next line is the source config's own note.\n")
    cfg = yaml.safe_load(new)
    assert cfg["seed"] == seed and cfg["data"]["fold"] == 6 and cfg["training"]["output_dir"] == f"runs/{run}"
    changed = [(x, y) for x, y in zip(text.splitlines(), new.splitlines(), strict=True) if x != y]
    assert len(changed) == (4 if seed == 42 else 5), changed
    path = CFGS / f"{run}.yaml"
    assert not path.exists()
    path.write_text(note + new)
    written[str(path)] = len(changed)
print(json.dumps({"splits_all": str(target), "fold_counts": counts, "configs": written}, indent=1))
