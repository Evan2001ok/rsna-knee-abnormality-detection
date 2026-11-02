# Final blend: exact training code for the multi-backbone models

These are the exact training programs behind the four multi-backbone model families in the team's selected submission. That submission is 57490913: 70% multi-backbone, 21% CoAtNet-2 and 9% R7, scoring 0.967 public and 0.957 private. The inference notebook is [Version 5 of the final team notebook](https://www.kaggle.com/code/lihu6666/rsna-knee-10th-place-inference-final-selected-team?scriptVersionId=360860299).

Each `.py` file is the code of one team-private Kaggle notebook version, copied byte for byte with nothing edited. `kaggle_notebooks/` keeps those five notebooks as Kaggle saved them, with their outputs (the training logs) and no images. [`sha256.txt`](sha256.txt) lists the hash of every file, and each `.py` hash equals the SHA-256 of the concatenated code cells of its notebook. The CoAtNet-2 half of the blend is on the [`coatnet-pipeline`](https://github.com/Evan2001ok/rsna-knee-abnormality-detection/tree/coatnet-pipeline) branch. R7 is heliosli's public model.

## Training record

| Family in the final blend | Script | Kaggle version | Runs used in the blend | Epochs | Training time on the GPU |
|---|---|---|---|---:|---|
| DINOv2-S (`dino`) | [`dinov2s_fullseed.py`](dinov2s_fullseed.py) | 360446358 | Full data, seeds 42, 43 and 44 | 16 | Seed 42: 143.9 min; 43: 215.6 min; 44: 256.4 min |
| EfficientNetV2-S (`eff`) | [`effv2s_fullseed.py`](effv2s_fullseed.py) | 360444710 | Full data, seeds 42, 43, 44 and 45 | 12 | 126.8 + 139.8 + 201.7 + 214.5 = 682.9 min |
| DINOv3 ConvNeXt-Tiny (`dino3`) | [`dinov3_convnext_4fold.py`](dinov3_convnext_4fold.py) | 358232793 | Folds 0–3 | 16 | 103.6 + 102.4 + 102.7 + 102.5 = 411.3 min |
| ConvNeXt-Tiny graycopy24 (`newconv`), folds 0–1 | [`convnext_graycopy24_folds01.py`](convnext_graycopy24_folds01.py) | 358835560 | Folds 0 and 1 | 12 | 175.7 + 163.5 = 339.2 min |
| ConvNeXt-Tiny graycopy24 (`newconv`), folds 2–3 | [`convnext_graycopy24_folds23.py`](convnext_graycopy24_folds23.py) | 358835914 | Folds 2 and 3 | 12 | 163.7 + 163.0 = 326.7 min |

All five ran as Kaggle notebooks in the same environment:

- GPU: NVIDIA RTX Pro 6000 (Kaggle accelerator `nvidiaRtxPro6000`).
- Software: Python 3.12.13 (notebook metadata) and `timm 1.0.26` (training logs).
- Run dates (UTC), from the notebook metadata: DINOv3 on 14 October 2026, ConvNeXt on 15–16 October, DINOv2 and EfficientNetV2 on 20–21 October.
- Kaggle image: **TBD**. The image digest appears only on the owner's run page, so each notebook owner should fill in their runs.

### Hardware and software

| | Training the four final families | Kaggle prediction (Version 5) |
|---|---|---|
| Machine | Kaggle notebooks, one per run | Kaggle notebook, internet off |
| Operating system, CPU, RAM, disk | Not recorded | See the CoAtNet branch README, "Hardware and software" |
| GPU | 1 × NVIDIA RTX Pro 6000 per run | 2 × NVIDIA Tesla T4 |
| Python | 3.12.13 | 3.13.15 |
| PyTorch, torchvision, NumPy | Not recorded; they come with the Kaggle image | 2.11.0, 0.26.0; NumPy 2.1.3 (part of the image) |
| timm | 1.0.26 | 1.0.26 |
| Numbers | bf16 autocast for most of the math; batch 4 × 4 accumulation steps = 16 | fp16 |
| Kaggle image | TBD: visible to each notebook owner | `gcr.io/kaggle-private-byod/python@sha256:2757e0c7d1e0a9cb43da657b97e223c321a98f5014bdf64f44f2f6b083ad2b2f` |
| DataLoader | 12 worker processes | |

The prediction column comes from the CoAtNet branch README, which records Version 5's environment.

Training is not bit-for-bit repeatable, even with this exact code:
- the scripts set `torch.backends.cudnn.benchmark = True`;
- they use bf16 autocast;
- the OAI sample and the augmentations are random draws;
- the results also depend on the GPU type and the library versions.

A rerun gives slightly different weights, and on other GPUs the differences can be larger.

The full-data runs use all 4,407 RSNA studies with no validation split and keep the last epoch. The fold runs use the 4-fold split (3,305 or 3,306 training studies). For each fold they save the EMA weights of the epoch with the best derived AUC on that fold's held-out studies, and the log names that epoch: for example, ConvNeXt fold 2 kept epoch 9 and fold 3 kept epoch 8.

## Settings shared by all five

| Setting | Value |
|---|---|
| Training labels | `0.7 × VLM + 0.21 × V1 + 0.09 × V2` (the original image-and-report VLM, `labels_vlm_imgrep.csv`), with the 58 official Gold rows overriding their studies; Gold studies get loss weight 3.0. Section 5.4 of the [branch README](../../../README.md#54-why-blend-label-sources) explains the recipe. |
| OAI | Pool of 8,640 usable baseline and follow-up examinations. Every epoch keeps all RSNA training studies and draws a fresh, participant-balanced OAI sample at 58% RSNA / 42% OAI (e.g. 3,305 + 2,393, or 4,407 + 3,191 for full data). |
| Input | Five slots, 24 slices each, 130 mm physical crop. Model input is 384 px, or 392 px for DINOv2. |
| Optimisation | AdamW, learning rate 2e-4 (DINO backbones 2e-5, heads 2e-4), 4 accumulation steps, EMA 0.995, auxiliary per-slot BCE 0.2. |

## Things to know before rerunning

1. **DINOv2 seed 45 is not in the blend.** `run_full_data_seeds()` loops over seeds 42–45, but the Kaggle run hit the 12-hour session limit (43,200 s) during seed 45. The blend therefore uses seeds 42–44.
2. **DINOv2 input:** the model docstring and a log line say "8 consecutive 3-slice groups". The code actually runs `x.unfold(1, 3, 2)`, which gives 11 overlapping 3-slice windows per slot. The code is what ran.
3. **ConvNeXt `newconv` = graycopy24.** Each of its 24 slices is one grayscale slice repeated into three channels. The two ConvNeXt scripts are identical except for their last line, `folds=(0, 1)` against `folds=(2, 3)`. They match `code/train/conv_gray384.py` except for one header comment and that last line, and `dinov3_convnext_4fold.py` matches `code/train/dino3.py` except for one blank line.
4. **Input paths are Kaggle mounts.** The scripts read `/kaggle/input/...` paths unchanged.
   - `datasets/minminandy/rsna-file/train.csv` is the team's private copy of the official competition `train.csv`. Use the competition's own `train.csv` instead.
   - The other inputs are:
     - the 384 px RSNA cache (`notebooks/*/dataprepare-of-rsnav5part1–4`);
     - the three label files (`llm-scored`, `llm-scorev2`, `vlm-score`);
     - the OAI images and soft labels;
     - the timm initialisation weights.
   - Section [External downloads and access](../../../README.md#external-downloads-and-access) of the branch README covers access to each of them. To run locally, change those paths; nothing else needs changing.
5. **Where the weights went.** The runs wrote the following folders, which the final notebook mounts:
   - `rsna_dinov2s_392_oai_mix_unfreeze9/seed{42,43,44}/`
   - `rsna_effv2s_384_oai8649_mix58_42_seven/seed{42..45}/`
   - `rsna_dinov3_convnext_tiny_384_oai8649_mix58_42_seven_16ep_splitlr/fold{0..3}/`
   - `rsna_convnext_tiny_384_oai8649_mix58_42_seven_graycopy24/fold{0..3}/`

   The public [multi-backbone weights dataset](https://www.kaggle.com/datasets/lihu6666/rsna-knee-model-10th-multi-seed-team) holds files with the same names and sizes.

## How the 70 / 21 / 9 weights were chosen

The weights come from two nested 70 / 30 splits. Each step is recorded in the team's Kaggle submission history:

1. **CoAtNet-2 0.7 : R7 0.3 (chosen by Max).** Max's own CoAtNet + R7 submissions blended the two branches at 0.7 / 0.3. The first of them notes "weight chosen on 4,296 out-of-fold studies" (submission "Knee OAI5 R7 LB": public 0.961, private 0.949). The later CoAtNet + R7 submissions kept the same ratio.
2. **Multi-backbone 0.7 : (CoAtNet-2 + R7) 0.3 (set by LP).** LP's submission "Knee Team Dhcons Newconv LB — 0.7 rank(our sets) + 0.3 rank(Max)" introduced the outer split. "Knee Team Dhcons All Newconv LB -7:3" (public 0.966, private 0.955) then ran it.
3. **Together:** 0.7, 0.3 × 0.7 = 0.21 and 0.3 × 0.3 = 0.09. Before the deadline, the selected submission ("Knee Team -all-LB — Version 5", 0.967 / 0.957) added Max's stage-4 final step to this blend:
   - convert each branch's predictions to percentile ranks;
   - combine the ranks with a weighted noisy-OR;
   - apply the sibling corrections: Lateral OA − 0.10 × Medial OA, PF OA − 0.10 × Medial OA, Synovitis − 0.20 × Effusion.

**Why this candidate was selected:** TBD by LP / Max. Public scores could not separate it from the 50 / 30 / 20 and 30 / 45 / 25 blends, which also scored 0.967 public.
