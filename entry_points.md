# Entry points

These are the commands, in order, that rebuild the multi-backbone half of the team's selected submission. Run them from the repository root. `SETTINGS.json` lists where each kind of file lives.

The training scripts keep the Kaggle paths they ran with. `tools/run_pipeline.py --script <file>` maps every `/kaggle/input/...` and `/kaggle/working/...` path in a script to a local path through `config/paths.json`, then runs it. Add `--plan` to print the mapping without running anything.

- Steps marked **(not public)** need files the repository cannot share: the OAI images and OAI labels (your own NIMH Data Archive access) and the official Gold labels (your own competition `train.csv`).
- Only these steps use the internet: the OAI download, the LLM/VLM scoring (only if you rebuild the labels instead of using the frozen files) and downloading the public timm initialisation weights. Training and prediction run offline.

## 0. Set up

```bash
python -m pip install -r requirements/pipeline.txt   # after installing a PyTorch/torchvision pair for your CUDA
python tools/preflight.py --dependencies             # checks the packages
python tools/init_workspace.py --layout local        # writes config/paths.json from config/paths.example.json
python tools/preflight.py                            # offline package and hash checks
```

Edit `config/paths.json` if your data folders differ from the defaults. `ENVIRONMENT.md` has the details.

## 1. Prepare the RSNA image cache

1. Put the competition download, unchanged, in `data/rsna/` (`train.csv`, `train_series.csv`, `train_series/`, ...).
2. Build the 384 px cache in four parts (24 slices per slot, 130 mm physical crop, slices in physical order):

   ```bash
   python tools/run_pipeline.py --script code/rsna/cache384.py --part 0
   python tools/run_pipeline.py --script code/rsna/cache384.py --part 1
   python tools/run_pipeline.py --script code/rsna/cache384.py --part 2
   python tools/run_pipeline.py --script code/rsna/cache384.py --part 3
   ```

   Each part writes one shard, `work/knee_v5_384px_24sl_central80_crop130_physicalsort_part<N>/`. On Kaggle the same shards came from the team's `dataprepare-of-rsnav5part1–4` notebooks, which is where the training scripts mount them.

## 2. RSNA labels

The final models train on three frozen label files that are already in the repository:

| File | Role |
|---|---|
| `labels/rsna/rsna_vlm_original.csv` | VLM |
| `labels/rsna/rsna_llm_v1_original.csv` | V1 |
| `labels/rsna/rsna_llm_v2_original.csv` | V2 |

Each training script blends them as `0.7 × VLM + 0.21 × V1 + 0.09 × V2`. It then replaces the 58 Gold placeholder rows with the official labels from `data/rsna/train.csv` **(not public: your own competition download)** and gives those studies loss weight 3.0. To check the label files and paths:

```bash
python tools/validate_rsna_labels.py --paths config/paths.json
```

Rebuilding the labels from the reports and images instead is optional; it is described in `REPRODUCE.md`, section 3.

## 3. OAI images and labels (not public)

The OAI images and soft labels need your own OAI access through the NIMH Data Archive. `REPRODUCE.md`, section 4, gives the download, quality-check, cache and label commands. The training scripts read four outputs of those steps:

- `data/oai/kaggle_upload_384/`
- `data/oai/kaggle_upload_followup6250_384/`
- `labels/oai/OAI_soft_labels_2399_v3.csv`
- `labels/oai/OAI_soft_followup6250_v3.csv`

At train time each script draws a fresh, participant-balanced OAI sample every epoch at 58% RSNA / 42% OAI.

## 4. Initialisation weights

Put the public timm weights where the mapped paths point, for example `data/notebooks/minminandy/dinov2s/vit_small_patch14_dinov2.lvd142m/pytorch_model.bin`. Print every expected path with `--plan` (step 5). `README.md`, section "Initial backbone weights for training", gives the source of each; the DINOv3 weights are under the DINOv3 License.

## 5. Train the four final model families

Each command runs one exact training program from `code/train/final_blend/`. Each uses one GPU; the times are those of the original runs on one NVIDIA RTX Pro 6000.

```bash
python tools/run_pipeline.py --script code/train/final_blend/dinov2s_fullseed.py --plan   # print the path mapping first
python tools/run_pipeline.py --script code/train/final_blend/dinov2s_fullseed.py            # DINOv2-S, full data, seeds 42-45, about 2.4-4.3 h per seed
python tools/run_pipeline.py --script code/train/final_blend/effv2s_fullseed.py             # EfficientNetV2-S, full data, seeds 42-45, about 2.1-3.6 h per seed
python tools/run_pipeline.py --script code/train/final_blend/dinov3_convnext_4fold.py       # DINOv3 ConvNeXt-Tiny, folds 0-3, about 1.7 h per fold
python tools/run_pipeline.py --script code/train/final_blend/convnext_graycopy24_folds01.py # ConvNeXt-Tiny graycopy24, folds 0-1, about 2.7-2.9 h per fold
python tools/run_pipeline.py --script code/train/final_blend/convnext_graycopy24_folds23.py # ConvNeXt-Tiny graycopy24, folds 2-3, about 2.7 h per fold
```

- Each script writes its run folder under `work/` (see `MODEL_CHECKPOINT_DIR` in `SETTINGS.json`). The model of each seed or fold is `fold0_final.pt`.
- The selected submission uses:
  - DINOv2-S: seeds 42–44. The original run stopped during seed 45 at Kaggle's 12-hour limit, and seed 45 was not used.
  - EfficientNetV2-S: seeds 42–45.
  - DINOv3 ConvNeXt-Tiny and ConvNeXt-Tiny graycopy24: folds 0–3.
- `code/train/final_blend/README.md` has the per-seed and per-fold times, the settings shared by all five programs, and the original logs.

## 6. Predict: the selected submission

Prediction is a Kaggle notebook. On 2 × T4 with internet off and the original environment, run either version of [lihu6666/rsna-knee-10th-place-inference-final-selected-team](https://www.kaggle.com/code/lihu6666/rsna-knee-10th-place-inference-final-selected-team?scriptVersionId=360860299):

- **Version 5** made submission 57490913 (0.967 public / 0.957 private). It reads the team's private weight datasets.
- **[Version 17](https://www.kaggle.com/code/lihu6666/rsna-knee-10th-place-inference-final-selected-team?scriptVersionId=364782508)** runs the same code from public inputs only, including the public [multi-backbone weights dataset](https://www.kaggle.com/datasets/lihu6666/rsna-knee-model-10th-multi-seed-team).

Either version runs the four multi-backbone families, the CoAtNet-2 group and R7, blends them at 70 / 21 / 9 and writes `/kaggle/working/submission.csv`. Step 5 of the CoAtNet branch's [`entry_points.md`](https://github.com/Evan2001ok/rsna-knee-abnormality-detection/blob/coatnet-pipeline/entry_points.md) lists the log lines that confirm every group ran. The notebook reads the multi-backbone weights from fixed mount paths with the run-folder names of step 5. Retrained models therefore have to follow the same folder layout as the public weights dataset. We have not tested a run with retrained weights.
