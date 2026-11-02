# Entry points

The commands that rebuild our part of the solution, in order. Run them from this folder. Folder names are the
defaults listed in `SETTINGS.json`; each command's `--help` lists its options, and each script's opening lines explain
its inputs.

- `uv run` runs a command inside the environment that `uv sync --locked` made. If you used pip instead
  (`python3.12 -m venv .venv`, then `. .venv/bin/activate` and `pip install -r requirements-train.txt`), leave out
  `uv run`. Keep the folder name `.venv`: `scripts/oai/prepare_oai_cache.sh` runs `.venv/bin/python` unless the
  `PYTHON` variable names another Python.
- Steps marked **(not public)** need files we cannot share: competition-derived labels and predictions, the OAI
  images, or other people's predictions. The README's "What is not included, and why" explains each one.
- Only these steps use the internet: the GPT step (paid; it sends study images, never reports, and needs your own
  API key), the OAI download (needs your own access), the public-label downloads, the first training run (timm's
  public starting weights) and the wheel downloads for Kaggle.

## 0. Set up

```bash
uv sync --locked                     # reads pyproject.toml and uv.lock; writes .venv/
uv run knee --help                   # lists the knee commands
CUDA_VISIBLE_DEVICES="" uv run pytest -q   # the checks, on the CPU only (about 6 minutes; 500 passed, 3 skipped)
```

## 1. Prepare the data

1. Put the competition download, unchanged, in `data/rsna-knee-abnormality-detection/`.
2. Image cache (336 px), the one the final models read:

   ```bash
   uv run python scripts/prepare_raptor.py --variant raptor-dense-distinct-v1
   ```

   Reads `data/rsna-knee-abnormality-detection/` (`train.csv`, `train_series.csv`, `train_series/`). Writes
   `data/processed/raptor-dense/train/`: `manifest.csv`, quality records (`qc.csv`, `qc.json`) and, in `studies/`,
   one image file and one metadata file per study (68 GB on our disk for the 4,407 training studies). Add
   `--split test` for the test studies (written to `data/processed/raptor-dense/test/`); only `knee predict` needs
   that, because the Kaggle bundle prepares test images itself.
3. OAI knees **(not public: needs your own access at the NIMH Data Archive)**:

   ```bash
   uv run python scripts/oai/make_moaks_selection.py --package <NDA package id> --visit 00
   read -r -p 'NDA user: ' NDA_USERNAME; read -rs -p 'NDA password: ' NDA_PASSWORD; echo
   export NDA_USERNAME NDA_PASSWORD   # typed at a prompt, so they do not land in the shell's history file
   uv run python scripts/oai/download_nda_selection.py --package <NDA package id> \
       --selection data/oai/packages/<NDA package id>/selection_v00_moaks.csv --out data/oai/raw
   bash scripts/oai/prepare_oai_cache.sh raptor-dense-distinct-v1 <OAI cache folder> 8 \
       data/oai/packages/<NDA package id>/selection_v00_moaks.csv
   uv run python scripts/oai/prepare_oai_labels.py
   uv run python scripts/oai/prepare_oai_hybrid_labels.py --output-dir data/labels/oai_hybrid_v1v3
   ```

   - `make_moaks_selection.py` reads the MOAKS grade tables in `data/oai/assessments/` and the NDA package's file
     lists (`image03.txt`, `files.csv`) in `data/oai/packages/<id>/`; it writes `selection_v00_moaks.csv` there
     (three series per graded knee, first visit).
   - `download_nda_selection.py` writes the series to `data/oai/raw/`. It reads the user name and password from the
     environment only and never writes them to disk.
   - `prepare_oai_cache.sh` unpacks the series in 8 parts into the competition's folder layout
     (`build_oai_root.py`), makes the same image cache as step 2 and writes `<OAI cache folder>/manifest.csv`.
   - `prepare_oai_labels.py` turns the MOAKS grades into our 12 findings: `data/labels/oai_moaks_v1/`.
   - `prepare_oai_hybrid_labels.py` averages those 50/50 with the teammates' OAI readings for seven findings. It
     reads `data/teammates/labels/oai/OAI_soft_labels_2399_v3.csv` **(not public as a file)** and our round-4
     models' predictions on the OAI knees, `artifacts/oai-check/pred957_oai.csv` **(not public)**.

## 2. Labels and splits

1. Public yes/no report labels (Pilkwang, CC0), then the competition labels and the folds:

   ```bash
   uv run python scripts/fetch_public_labels.py
   uv run knee labels --train-csv data/rsna-knee-abnormality-detection/train.csv \
       --weak-csv data/labels/public/pilkwang/report_labels_binary.csv
   uv run knee split                                              # 5 folds, seed 42
   uv run knee split --folds 4 --output-dir data/splits/four_fold   # 4 folds, seed 42 (final rounds)
   ```

   `fetch_public_labels.py` writes `data/labels/public/pilkwang/`. `knee labels` writes `data/labels/baseline/`
   (`labels.csv`, `expert_labels.csv` with the 58 expert studies, and more). `knee split` reads
   `data/labels/baseline/labels.csv` and writes `splits.csv` to `data/splits/baseline/` or `data/splits/four_fold/`.
   On November 2, 2026 we reran these three `knee` commands with this code: `labels.csv`, `expert_labels.csv`,
   `report_labels.csv` and both `splits.csv` files came out byte for byte the same as the files we used.
2. Checking labels (used only to score, never to train):

   ```bash
   uv run python scripts/prepare_severity_validation_labels.py                       # -> data/labels/baseline_severity/
   uv run python scripts/prepare_gpt_validation_labels.py --gpt-labels <GPT csv>      # -> data/labels/severity_gpt_agreement/
   ```

3. GPT image readings **(paid; needs an `OPENAI_API_KEY=...` line in `.env`)**: `scripts/gpt_image_labels.py` with
   the steps `prepare`, `estimate`, `run` (needs `--yes`) and `parse`, each with `--studies <study list CSV>`. We used
   `--provider openai --effort medium --service-tier flex` (the flex tier is half price, slower and sometimes busy;
   the script retries). Only images are sent, never reports. Output: `artifacts/gpt-image-labels/v1/`.
4. Labels of each round **(not public: from round 1 on, each round mixes in out-of-fold predictions of models trained
   in the round before, so an exact rebuild means training every round again)**:

   | Round | Script | Writes (default or as we ran it) |
   | --- | --- | --- |
   | 1 | `prepare_raptor_labels.py --download-source` | `data/labels/raptor/` |
   | 1 | `prepare_averaged_labels.py --findings "Medial Meniscus" "Lateral Meniscus" Synovitis Contusion Fracture --output-dir data/labels/avg_v2` | `data/labels/avg_v2/` |
   | 1 | Train the five ResNet34 teachers (`configs/followup/`, `configs/teachers/`), then `prepare_student_labels.py --raptor-labels data/labels/avg_v2/labels.csv --teacher-runs <the five teacher runs> --output-dir data/labels/student_v2` | `data/labels/student_v2/` |
   | 2 | `prepare_gpt_mixed_labels.py --gpt-labels <GPT csv> --output-dir data/labels/student_v2_gpt_full` | `data/labels/student_v2_gpt_full/` |
   | 3 | `prepare_round3_labels.py` (first needs `artifacts/teacher-recipe-decision.json` from `choose_teacher_recipe.py`) | `data/labels/student_v3_round3/` |
   | 4 | `prepare_source_choice_labels.py` | `data/labels/source_choice_v2/` |
   | 5 | `prepare_source_choice_labels.py --sources report gpt --output-dir data/labels/source_choice_v6_report_gpt` (`v6`), then train `configs/backbones/` on it, then `prepare_source_choice_labels.py --coatnet-set runs/coatnet_384native_srcchoice6_fold --output-dir data/labels/source_choice_v7_own_coatnet` | `data/labels/source_choice_v7_own_coatnet/` (`v7`) |
   | 6 | `prepare_packed_labels.py --mode r7mix --r7-share 0.3 --output-dir data/labels/v7_r7mix30` | `data/labels/v7_r7mix30/` |
   | Final | `prepare_consensus_labels.py` (reads the teammates' public labels at `data/teammates/labels/rsna/conv_labels_used.csv`) | `data/labels/consensus_v1/labels.csv` |

   All scripts are in `scripts/` and run as `uv run python scripts/<name>`. Between rounds, models are trained
   with `knee train` on the new labels; the scripts' opening lines name the runs they read.
5. The training folder of the final models **(not public)**. It starts from an earlier training folder that already
   holds the OAI knees with their final labels:

   ```bash
   uv run python scripts/oai/build_combined_inputs.py --labels <labels.csv> --validation-labels <checking labels.csv> \
       --splits <splits.csv> --manifest data/processed/raptor-dense/train/manifest.csv \
       --oai-labels data/labels/oai_moaks_v1/labels.csv --oai-manifest <OAI cache folder>/manifest.csv \
       --output-dir data/combined/oai_v00_336_v7own
   uv run python scripts/oai/swap_oai_labels.py data/combined/oai_v00_336_v7own \
       data/labels/oai_hybrid_v1v3/labels.csv data/combined/oai_v00_336_v7oaihybrid
   uv run python scripts/make_consensus_combined.py data/labels/consensus_v1/labels.csv \
       data/splits/four_fold/splits.csv --with-experts
   ```

   - `build_combined_inputs.py` adds the OAI knees (fold 5: they always train and never check). We built
     `oai_v00_336_v7own` with our round-5 labels; `make_consensus_combined.py` later replaces the labels and folds
     of every competition row, so these inputs only matter for the OAI rows, the manifest and the checking labels.
   - `swap_oai_labels.py` copies that folder with only the OAI labels replaced by the hybrid OAI labels.
   - `make_consensus_combined.py` reads `data/combined/oai_v00_336_v7oaihybrid/` and writes
     `data/combined/oai_v00_336_consensus58_4f/`: `labels.csv` (final labels; the 58 expert studies keep the host's
     labels), `splits.csv` (4 folds; the 58 expert and 53 unlabelled studies in fold 5), plus copies of `manifest.csv`
     and `validation_labels.csv`. It skips writing fold configs when their templates are missing, as here.
6. The 100 held-out studies (`splits_all.csv`, fold 6). In the final runs, they are exactly the 100 studies in fold 6
   of our `data/splits/all_studies/splits.csv` (checked against the runs' records). They were first picked on
   October 11, 2026 by `scripts/make_all_studies_split.py` (at random with seed 0, making sure every finding has
   positive and negative studies), and every later full-data folder reused the same 100 studies. To rebuild that file:

   ```bash
   uv run python scripts/make_all_studies_split.py --splits data/splits/baseline/splits.csv \
       --validation-labels data/labels/baseline_severity/labels.csv --output data/splits/all_studies/splits.csv \
       --minimum 2
   ```

   On November 2, 2026 this command, with this code, rebuilt our file byte for byte. With the default `--minimum 3`,
   it picks the same 100 studies and then adds 24 more, because those 100 have only 2 Synovitis negatives in today's
   checking labels.

   Our final folder got its `splits_all.csv` from `scripts/make_fulldata.py`, kept exactly as it ran on the cloud
   machine. That script read the 100 studies from an earlier training folder,
   `data/combined/oai_v00_336_v7own/splits_all.csv` (its fold 6 holds the same 100 studies), and it also wrote the
   four base configs that are now in `configs/final/`. The lines below do the same first step from
   `data/splits/all_studies/splits.csv` instead, with the same checks on the counts:

   ```python
   import pandas as pd
   ID = "StudyInstanceUID"
   folder = "data/combined/oai_v00_336_consensus58_4f/"
   held = pd.read_csv("data/splits/all_studies/splits.csv")
   held = set(held.loc[held["fold"] == 6, ID])   # the 100 held-out studies
   splits = pd.read_csv(folder + "splits.csv")
   rows = splits[ID].isin(held)
   assert len(held) == 100 and rows.sum() == 100 and splits.loc[rows, "fold"].between(0, 3).all()
   splits.loc[rows, "fold"] = 6
   counts = splits["fold"].value_counts()
   assert counts[6] == 100 and counts.drop(6).sum() == 6705   # 6,705 rows train
   splits.to_csv(folder + "splits_all.csv", index=False)
   ```

## 3. Train the four final models

Once per seed (42, 43, 44 and 45), base training first, then the all-window fine-tune:

```bash
uv run knee train --config configs/final/coatnet_384_consensus58_all_s42.yaml
uv run knee train --config configs/final/coatnet_384_consensus58_all_ftall_s42.yaml
```

- Base training reads `data/combined/oai_v00_336_consensus58_4f/` (`manifest.csv`, `labels.csv`,
  `validation_labels.csv`, `splits_all.csv`) and the image caches that `manifest.csv` points to. On its first run,
  timm downloads the public ImageNet starting weights of `coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k` (internet needed
  once). It writes `runs/coatnet_384_consensus58_all_s42/`; the model is `averaged-model.pt` (the average of
  epochs 22, 23 and 24).
- The fine-tune reads the same folder plus `runs/coatnet_384_consensus58_all_s42/averaged-model.pt` (its
  `init_checkpoint`). It writes `runs/coatnet_384_consensus58_all_ftall_s42/`; the model is `averaged-model.pt`
  (the average of its best 3 of 4 epochs on the 100 held-out studies).
- Each run also writes its logs and records to its folder (see `LOGS_DIR` in `SETTINGS.json`). A run refuses to start
  in a folder that is not empty; `--resume runs/<run>/last.pt` continues a stopped run.
- Each run uses one GPU. Times and memory: the "Run time" table in the README.

## 4. Export the Kaggle bundle

```bash
uv run python scripts/download_inference_wheels.py --python-version 3.13 --extra opencv-python-headless --output-dir wheels
pip download timm==1.0.26 --no-deps -d wheels
uv run knee export --checkpoints runs/coatnet_384_consensus58_all_ftall_s4{2,3,4,5}/averaged-model.pt \
    --output-dir bundle --wheelhouse wheels --eval-windows 56
```

- The first two lines download the offline install files for Kaggle's Python 3.13 into `wheels/`
  (`requirements-inference.txt` lists what our stage installs from them).
- `knee export` reads the four fine-tuned models and `wheels/`, and writes `bundle/`: `model-0.pt` to `model-3.pt`
  (seeds 42 to 45, set to predict with 56 windows), `bundle.json`, `src/` (the code), `wheelhouse/`,
  `third_party/` (the Apache-2.0 notice), `run_inference.py` and `inference.ipynb`. It refuses a `bundle/` folder that is not empty.
- Upload `bundle/` as a Kaggle dataset. Ours is public:
  [max6296/knee-coatnet-consensus-all-ftall-w56-bundle](https://www.kaggle.com/datasets/max6296/knee-coatnet-consensus-all-ftall-w56-bundle),
  version 1, the exact files the selected submission used. It was made with older export code (see "Exact training
  code" in the README).

## 5. Predict

**The selected team submission (57490913).** Run Version 5 of the Kaggle notebook
[lihu6666/rsna-knee-10th-place-inference-final-selected-team](https://www.kaggle.com/code/lihu6666/rsna-knee-10th-place-inference-final-selected-team?scriptVersionId=360860299)
on 2 x T4 with internet off. It reads the competition data, our bundle (above), the teammates' weights and heliosli's
two R7 datasets, runs the three groups one after another and writes `/kaggle/working/submission.csv`. Version 5 reads
the teammates' weights from their private datasets. Their public dataset
[lihu6666/rsna-knee-model-10th-multi-seed-team](https://www.kaggle.com/datasets/lihu6666/rsna-knee-model-10th-multi-seed-team)
holds files with the same names and sizes. [Version 17](https://www.kaggle.com/code/lihu6666/rsna-knee-10th-place-inference-final-selected-team?scriptVersionId=364782508) of the same notebook runs Version 5's code from
public inputs only: its first cell waits until every input is attached and links `/kaggle/temp/team_weights` to
wherever Kaggle mounted that dataset (`/kaggle/input/rsna-knee-model-10th-multi-seed-team` in the owner's notebook,
`/kaggle/input/datasets/lihu6666/rsna-knee-model-10th-multi-seed-team` in anyone else's copy), and the team code reads
its weights through that link. Version 17 and a copy under another account both passed the checks below, and their
submission files were identical. To rerun it, open Version 17 and choose Copy & Edit; keep 2 x T4, internet off and
the original environment.
A missing input does not stop the notebook: it leaves that group out and still writes `submission.csv`. So check the
log for `INPUTS_READY {... 'missing': []}` (Version 17 only), `THEIRS_STAGE_DONE {"ok": true, ...}`, `OURS_STAGE_DONE {"ok": true, ...}`, `R7_STAGE_DONE {"ok": true, ...}`
and, in the last line (`NOTEBOOK_COMPLETE`), `"weights_used": {"theirs": 0.7, "ours": 0.21, "r7": 0.09}`.

**Our four models alone.** Either attach the bundle to a Kaggle notebook and run the bundle's own `inference.ipynb`,
or run it on your own GPU machine:

```bash
uv run python bundle/run_inference.py --data-root data/rsna-knee-abnormality-detection --work-dir <output folder>
```

This reads `test.csv` and `test_series/` in `--data-root`, checks the models' checksums, prepares the test images in
temporary folders inside `--work-dir` and deletes them after use, and writes `<output folder>/submission.csv`,
`inference-summary.json` and `prediction-parts/`.

`knee predict --manifest <prepared test manifest.csv> --studies <test.csv> --checkpoints <models>` also predicts,
from a cache made with `prepare_raptor.py --split test`. With the plain `averaged-model.pt` files it uses each model's
own setting, 112 windows per study, not the 56 of the submitted bundle.

**Rebuilding the team notebook.** `scripts/build_s6_blend_notebook.py` and `scripts/build_team_combined_notebook.py`,
with the commands and inputs in the README section "Kaggle prediction and the team blend". They need the teammates'
notebook and weights and heliosli's public R7 notebook.
