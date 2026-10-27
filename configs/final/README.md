# Final model configs

These are the settings for our four CoAtNet-2 models in the final team blend (Kaggle submission 57456685; the same models were also in 57375497 and 57436336; all scored 0.967 public / 0.956 private).

Each seed (42 to 45) trains in two steps:

1. **Base training:** `coatnet_384_consensus58_all_s42.yaml` to `s45.yaml`. 24 epochs on all studies except 100 held out (fold 6). About 2.9 h per seed on one RTX 5090.
2. **All-window fine-tune:** `coatnet_384_consensus58_all_ftall_s42.yaml` to `s45.yaml`. 4 more epochs on every window, starting from the base run's `averaged-model.pt`. About 6.5 h per seed.

From the repo root, run the base file first, then the fine-tune (once per seed), then export the four models:

```
uv run knee train --config configs/final/coatnet_384_consensus58_all_s42.yaml
uv run knee train --config configs/final/coatnet_384_consensus58_all_ftall_s42.yaml
uv run knee export --checkpoints runs/coatnet_384_consensus58_all_ftall_s4{2,3,4,5}/averaged-model.pt --output-dir bundle --wheelhouse wheels --eval-windows 56
```

The main README explains the `wheels` folder (offline install files for Kaggle).

The seed files differ only in `seed` and `output_dir` (and, in the fine-tune files, `init_checkpoint`, which points to the same seed's base run). In the file names, `384` is the image size, `consensus58` the final labels with the 58 expert studies, `all` training on all studies but 100, `ftall` the fine-tune on all windows and `s42` the seed.

All eight files read one training folder, `data/combined/oai_v00_336_consensus58_4f/` (OAI first visit, 336 px image cache, final labels with the 58 expert studies, 4 folds). It is not included, because it has one row per competition study: the final labels, the image cache list (`manifest.csv`), the checking labels and the splits for every study, plus the 2,398 OAI knees. Rebuild it with the scripts: `scripts/make_consensus_combined.py <final labels> <4-fold splits> --with-experts` makes the folder, and `scripts/make_all_studies_split.py` makes its `splits_all.csv` (fold 6 is the 100 held-out studies). The main README lists the steps before that.

The final models were trained with an earlier copy of this code. This version has the same fine-tune options these files use (`raptor_train_all_windows`, `init_checkpoint`, `freeze_batchnorm`).
