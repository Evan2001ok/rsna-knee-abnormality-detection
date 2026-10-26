# Frozen RSNA soft-label snapshots

These are the seven historical CSV snapshots from our local handover. Every non-Gold row is copied byte-for-byte. Each contains 4,407 unique `StudyInstanceUID` values and the same 12 target columns, in the order recorded in [manifest.json](manifest.json). They are experimental training targets, not independently calibrated clinical probabilities.

| File | Role |
|---|---|
| [rsna_llm_v1_original.csv](rsna_llm_v1_original.csv) | First report-based label snapshot. |
| [rsna_llm_v2_original.csv](rsna_llm_v2_original.csv) | Second report-based label snapshot; historical source filename `labels_gold_override.csv`. |
| [rsna_vlm_original.csv](rsna_vlm_original.csv) | Original image/report VLM snapshot (VLM-A). |
| [rsna_vlm_opus_original.csv](rsna_vlm_opus_original.csv) | Alternate image/report VLM snapshot (VLM-B). |
| [conv_labels_used.csv](conv_labels_used.csv) | Old recipe: 0.7 VLM-A + 0.3 L. |
| [conv_new_labels_used.csv](conv_new_labels_used.csv) | New Conv/C2: 0.7 VLM-B + 0.3 L. |
| [dino_new_labels_used.csv](dino_new_labels_used.csv) | New DINO/X1: 0.5 VLM-B + 0.5 L. |

Here `L = 0.7 V1 + 0.3 V2`. Each file contains 4,349 report/VLM-derived rows and 58 Gold placeholder rows.

**Official Gold labels are not included.** The 58 official expert labels are competition data, so their rows hold placeholders only: every target is `0.5`, and in the three `*_labels_used.csv` files `label_source` is `official_gold_placeholder`. The value carries no annotation. Training is unaffected, because every training entry overrides these rows with the official values from your own competition `train.csv` (`load_vlm_labels`: `blended.loc[gold.index] = gold`). You therefore need authorized competition access to train. `tools/validate_rsna_labels.py` fails if any Gold row is not an exact placeholder. The placeholders were inserted with `tools/add_gold_placeholders.py`. These historical source snapshots are already processed tables, not guaranteed untouched raw LLM/VLM responses. Preserve the original evaluation/fold policy; do not use training targets as an independent evaluation ground truth. Gold cases were not a never-trained external test set in this pipeline.

The `label_source` string `vlm07_llm03` is retained exactly, including in `dino_new_labels_used.csv`, whose actual blend is 0.5/0.5. Use the documented formula and numerical checks, not that legacy string, to identify a recipe. Do not blend an already mixed training snapshot a second time.

## Use

From the repository root:

```bash
python tools/validate_rsna_labels.py
python tools/init_workspace.py --layout local
python tools/validate_rsna_labels.py --paths config/paths.json
```

Both `--layout local` and `--layout kaggle` now map the four source inputs to the included CSVs; Kaggle keeps its original mounts for other assets. This assumes the cloned/extracted repository is available to the notebook. The initializer refuses to overwrite existing configs: edit the four RSNA source mappings in an old `config/paths.json` using `config/paths.example.json`, then audit that actual config with `--paths config/paths.json`.

Use `python tools/run_pipeline.py --script code/train/conv.py --folds 0 --plan` (or the chosen recipe), then remove `--plan` after supplying and checking all other assets. Directly running `code/train/*.py` bypasses path mapping and the double-mixing guard. The three mixed snapshots are reference outputs only. The wrapper rejects them as source inputs, including renamed exports retaining `label_source`, and rejects using the same file for multiple sources.

`python tools/validate_rsna_labels.py --training-loaders` additionally executes the original label-loader functions from all ten training entries and compares their outputs with these three snapshots. It needs numpy/pandas but no GPU. Its metadata fixture inherits saved Gold values; this is a consistency test, not independent verification of official metadata or clinical accuracy. The official `train.csv` is still required for training.
The SHA256 of every CSV is in `manifest.json` and the repository root manifest. Download a file using its GitHub **Raw / Download raw file** control, or clone this branch. Reports, free-text evidence, MRI, OAI participant data, OOF and trained weights are not included. Obtain the official competition data and follow its applicable terms separately. No new project-wide license or upstream-data rights are granted by this snapshot.
