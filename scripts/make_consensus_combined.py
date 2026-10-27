#!/usr/bin/env python3
"""Build the consensus training folder and the four fold configs from the hybrid OAI-label arm's folder and configs
(Oct 16, 2026). The hybrid arm was our five-fold CoAtNet run on the v7 labels (our round-5 labels) plus the OAI baseline
knees with the hybrid OAI labels (scripts/oai/prepare_oai_hybrid_labels.py); its folder is
data/combined/oai_v00_336_v7oaihybrid (336 px image cache).

Folder: competition rows take the consensus labels (data/labels/consensus_v1) and the 4-fold split
(data/splits/four_fold); the 53 unlabeled studies join OAI in fold 5 (always training, never validated); OAI rows,
manifest and validation labels are the hybrid's, unchanged. Configs: each hybrid fold-k config (k = 0-3) with only the
four data paths, the output folder and the fixed positive weights (the hybrid's five-fold mean) changed. The hybrid's
fold configs are not included, so without them the configs step is skipped with a message.

Run from the repo root: .venv/bin/python scripts/make_consensus_combined.py LABELS SPLITS [--with-experts]   (or --configs-only)
--with-experts (Oct 16): the 58 expert studies join fold 5 with their gold labels (label_source gold), into
data/combined/oai_v00_336_consensus58_4f and configs coatnet_384_consensus58_fold0-3. The final configs (configs/final)
read this folder, with splits_all.csv from scripts/make_all_studies_split.py.
--theirs (Oct 17): LABELS is data/labels/theirs_v1/labels.csv (the teammates' readings alone), into
data/combined/oai_v00_336_theirs_4f and configs coatnet_384_theirs_fold0-3; the 58 expert studies stay out of training.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ID = "StudyInstanceUID"
SRC, DST = Path("data/combined/oai_v00_336_v7oaihybrid"), Path("data/combined/oai_v00_336_consensus4f")
CFGS = Path("configs/backbones")
OLD_RUN, NEW_RUN = "coatnet_384_v7oai_oaihybrid_fold", "coatnet_384_consensus4f_fold"
FOLDS = 4
EXPERTS = False
RELABEL = "the competition rows relabelled (equal thirds of v7, theirs and the all-model blend)"
RECORD = "experiments/consensus-labels-20261016.json"


def build_folder(labels_path: Path, splits_path: Path, with_experts: bool = False) -> dict:
    read = lambda p: pd.read_csv(p, dtype={ID: str, "group_id": str})  # noqa: E731
    comb, splits = read(SRC / "labels.csv"), read(SRC / "splits.csv")
    new, four = read(labels_path).set_index(ID), read(splits_path).set_index(ID)
    targets = [c for c in new.columns if c != "label_source"]
    assert len(targets) == 12, targets
    comp = comb["label_source"].ne("oai")
    comp_ids = comb.loc[comp, ID]
    assert set(comp_ids) == set(new.index) == set(four.index), "competition studies must match"
    out = comb.copy()
    out.loc[comp, targets] = new.loc[comp_ids, targets].to_numpy(float)
    out.loc[comp, "label_source"] = new.loc[comp_ids, "label_source"].to_numpy()
    assert out.loc[~comp].equals(comb.loc[~comp]), "OAI label rows must stay unchanged"
    sp = splits.set_index(ID)
    s_out = sp.copy()
    ids = list(comp_ids)
    s_out.loc[ids, "fold"] = four.loc[ids, "fold"].to_numpy()
    s_out.loc[ids, "split"] = four.loc[ids, "split"].to_numpy()
    unlabeled = four.index[four["fold"] == -2]
    s_out.loc[unlabeled, "fold"] = 5
    s_out.loc[unlabeled, "split"] = "train"
    experts = four.index[four["fold"] == -1]
    if with_experts:  # Oct 16: the 58 expert studies join training, with their gold labels as they are
        assert (out.set_index(ID).loc[experts, "label_source"] == "expert").all()
        out.loc[out[ID].isin(experts), "label_source"] = "gold"
        s_out.loc[experts, "fold"] = 5
        s_out.loc[experts, "split"] = "train"
    oai = comb.loc[~comp, ID]
    assert s_out.loc[oai].equals(sp.loc[oai]), "OAI split rows must stay unchanged"
    assert (s_out.loc[s_out["split"] == "expert_holdout", "fold"] == -1).all() and not s_out["split"].eq("unlabeled").any()
    counts = s_out["fold"].value_counts().sort_index().to_dict()
    always = len(oai) + len(unlabeled) + (len(experts) if with_experts else 0)
    assert all(1000 < counts[k] < 1150 for k in range(FOLDS)) and counts[5] == always, counts
    values = out[targets].to_numpy(float)
    assert np.all((np.isnan(values)) | ((values >= 0) & (values <= 1)))
    DST.mkdir(parents=True, exist_ok=False)
    out.to_csv(DST / "labels.csv", index=False)
    s_out.reset_index()[[ID, "fold", "split", "group_id"]].to_csv(DST / "splits.csv", index=False)
    for name in ("manifest.csv", "validation_labels.csv"):
        shutil.copy2(SRC / name, DST / name)
    sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()  # noqa: E731
    rec = {"rows": len(out), "competition_rows": int(comp.sum()), "oai_rows": int((~comp).sum()), "fold_counts": counts,
           "label_sources": out["label_source"].value_counts().to_dict(),
           "inputs": {str(labels_path): sha(labels_path), str(splits_path): sha(splits_path), str(SRC / "labels.csv"): sha(SRC / "labels.csv")},
           "labels_sha256": sha(DST / "labels.csv"), "splits_sha256": sha(DST / "splits.csv"),
           "rule": f"competition rows: labels from {labels_path} and the 4-fold split; the 53 unlabeled"
                   f"{' and the 58 expert studies (gold labels, label_source gold)' if with_experts else ''} in fold 5; OAI rows, "
                   f"manifest and validation labels from {SRC}"}
    (DST / "provenance.json").write_text(json.dumps(rec, indent=2) + "\n")
    return rec


def write_configs() -> dict:
    sources = [CFGS / f"{OLD_RUN}{k}.yaml" for k in range(5)]
    if missing := [str(p) for p in sources if not p.exists()]:  # the hybrid's fold configs are not in this repo
        print(f"Skipping the fold configs: template configs not found: {', '.join(missing)}", file=sys.stderr)
        return {"configs_skipped": missing}
    weights = [yaml.safe_load(p.read_text())["training"]["fixed_positive_weights"] for p in sources]
    mean = {t: float(np.mean([w[t] for w in weights])) for t in weights[0]}
    written = {}
    for k in range(FOLDS):
        text = sources[k].read_text()
        old_block = "".join(f"    {t}: {weights[k][t]!r}\n" for t in weights[k])
        new_block = "".join(f"    {t}: {mean[t]!r}\n" for t in mean)
        assert text.count(old_block) == 1, f"fold {k}: positive weight block not found verbatim"
        new = text.replace(old_block, new_block).replace(str(SRC), str(DST)).replace(f"{OLD_RUN}{k}", f"{NEW_RUN}{k}")
        note = (f"# Oct 16-17, 2026 ({RECORD}): {sources[k].name} with "
                f"{RELABEL}, the 4-fold split, the 53 "
                f"unlabeled studies{' and the 58 expert studies (gold labels)' if EXPERTS else ''} in training ({DST}) and the hybrid's five-fold mean positive weights; the next line is "
                f"the source config's own note.\n")
        cfg = yaml.safe_load(new)
        assert cfg["data"]["fold"] == k and cfg["training"]["output_dir"] == f"runs/{NEW_RUN}{k}"
        assert all(str(DST) in cfg["data"][key] for key in ("manifest", "labels", "validation_labels", "splits"))
        changed = [(a, b) for a, b in zip(text.splitlines(), new.splitlines(), strict=True) if a != b]
        allowed = len(mean) + 6  # 12 weights + 4 data paths + output folder + the source note naming the old folder
        assert len(changed) <= allowed, changed
        out = CFGS / f"{NEW_RUN}{k}.yaml"
        assert not out.exists(), out
        out.write_text(note + new)
        written[str(out)] = len(changed)
    return {"configs": written, "fixed_positive_weights_mean_of_hybrid_folds": mean}


if __name__ == "__main__":
    if "--with-experts" in sys.argv:
        sys.argv.remove("--with-experts")
        DST, NEW_RUN, EXPERTS = Path("data/combined/oai_v00_336_consensus58_4f"), "coatnet_384_consensus58_fold", True
    elif "--theirs" in sys.argv:
        sys.argv.remove("--theirs")
        DST, NEW_RUN, EXPERTS = Path("data/combined/oai_v00_336_theirs_4f"), "coatnet_384_theirs_fold", False
        RELABEL = "the competition rows relabelled with the teammates' 0.7 VLM + 0.3 LLM readings alone (data/labels/theirs_v1)"
        RECORD = "experiments/overnight-gpu-20261017.json"
    else:
        EXPERTS = False
    if sys.argv[1] == "--configs-only":  # the folder already exists
        record = write_configs()
    else:
        record = {"folder": build_folder(Path(sys.argv[1]), Path(sys.argv[2]), EXPERTS), **write_configs()}
    print(json.dumps(record, indent=1))
