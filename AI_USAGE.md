# AI usage guide: module map and execution workflow

> **Public code and RSNA label edition.** This branch includes pipeline source, prompts, configurations and seven frozen RSNA label CSVs in [labels/rsna](labels/rsna/README.md). OAI labels, OOF predictions, split/cohort tables, MRI caches and trained weights remain external. No final team-score reproduction or GPU rerun is claimed. See [PUBLIC_RELEASE.md](PUBLIC_RELEASE.md).

This document is for an AI assistant operating this repository. Read `README.md`, `ENVIRONMENT.md`, `REPRODUCE.md` and this guide before choosing an entry point. All commands below are run from the package root. Respect the human user's requested scope; the presence of a command here is not authorization to start expensive training, upload data or change remote services.

## 1. What is included and what is external

Use `README.md#external-downloads-and-access` and `config/external_resources.json` when locating assets. Distinguish public upstream initialization from private task-trained checkpoints. The catalog records the inspection date and unknown visibility explicitly; do not promise public access or change an artifact's sharing settings merely because its path is referenced.

Included: RSNA224/384 cache builders; ten training entry points; original report/VLM/OAI prompts; task generation and strict score merging; OAI baseline/follow-up processing; historical six/eight-suite inference; external-asset interfaces (RSNA label snapshots are included; OAI labels, splits and OOF are external).

Not embedded: raw MRI, original NDA score tables, model weights, authentication, a universal remote scoring SDK, or the final team0.967 inference pipeline. Historical ensemble specs remain unchanged when new training sources are added. Do not silently substitute the new grayscale model for the original2.5D ensemble member.

Source status is recorded in `config/training_source_provenance.json`. DINOv3 training was recovered from executed Kaggle Version3, fixed scriptVersionId358232793. It is actual training source, not a reconstruction from inference.

## 2. Start with the checks that do not execute models

```bash
python tools/preflight.py --report preflight.json
```

By default this checks source hashes, Python syntax and prompts. Optional --assets additionally checks separately supplied private label/mask tables, 4407-study OOF alignment and fixed A/B/AB blends. It does not convert MRI or prove GPU correctness.

Install the appropriate environment following `ENVIRONMENT.md`. PyTorch/CUDA must be a compatible pair; do not replace Kaggle's torch installation blindly. Processing/training requirements are in `requirements/pipeline.txt`, NDA tools in `requirements/nda.txt`. The pinned timm version is1.0.26; DINOv3 requires its recorded model name and external safetensors weights.

```bash
python tools/preflight.py --dependencies --report dependency_check.json
```

The default dependency group is pipeline-only; use `--dependency-group nda` for downloads or `all` for both. ABI/CUDA compatibility still needs real environment validation. Run `python tools/check_reproduction.py` for source/import/prompt/CLI and synthetic merge tests. Initialize once with `python tools/init_workspace.py --layout local` (or `kaggle`) before using default path mappings.

## 3. Paths, fold selection and outputs

Use `tools/init_workspace.py` once to create `config/paths.json`, then edit mappings to the actual data, cache and weights. `${PACKAGE}` resolves to this package root. The longest prefix wins. Exact224 and384 OAI mappings are separate. The broad `/kaggle/input` mapping is only a fallback; inspect it before execution.

```bash
python tools/run_pipeline.py --script code/train/conv224.py --folds 0 --plan
python tools/run_pipeline.py --script code/train/conv_gray384.py --folds 0 --plan
python tools/run_pipeline.py --script code/train/dino3.py --folds 0 --plan
python tools/preflight.py --external --report external_check.json
```

`--plan` parses source and prints path replacements without importing or executing the target. `--external` inventories assets across the package, including other models; inspect the assets actually needed for the chosen stage.

The adapter changes only plain/dynamic leading string path prefixes, cache PART_ID and explicit fold selection. It does not rewrite arbitrary command-line paths or arbitrary embedded Python source strings; inference globals are configured by the original orchestrator. `--folds 0` selects fold0 even when recovered source calls `run_all_folds(Cfg(), folds=(0,1,2,3))`. With no `--folds`, preserve the source default. Supported folds are0–3. No general hyperparameter override or GPU `--check-only` mode is supplied: do not invent CLI flags.

Use distinct output directories when changing a recipe or running a comparison. Inspect resume/checkpoint behavior before reusing an output directory; do not overwrite existing results to make paths fit.

## 4. RSNA DICOM to cache

| Entry point | Native cache | Physical crop | Slices / slots | Shards |
|---|---|---|---|---|
| `code/rsna/cache224.py` |224×224 uint8 |160mm |24 per slot,5 slots, central80% |2: parts0,1 |
| `code/rsna/cache384.py` |384×384 uint8 |130mm |24 per slot,5 slots, central80% |4: parts0,1,2,3 |

```bash
python tools/run_pipeline.py --script code/rsna/cache224.py --part 0
python tools/run_pipeline.py --script code/rsna/cache224.py --part 1
python tools/run_pipeline.py --script code/rsna/cache384.py --part 0
# Repeat cache384 with --part 1,2,3 as separate invocations.
```

Outputs are `train_cache.npy`, `train_mask.npy`, `study_ids.npy` and `manifest.json`. Keep each shard's files together. Verify IDs, disjoint shard coverage, slot order, dtype, dimensions and mask alignment. Physical sorting/laterality and sampling must match the training source. Do not resize an already resized224 cache and call it the original384 recipe.

Slot order: SAG_FLUID_FS, COR_FLUID_FS, AX_FLUID_FS, COR_T1, SAG_T1. Missing image slots are not negative disease labels.

## 5. Choose the correct training recipe

| Script under `code/train/` | RSNA cache / actual backbone input | Representation | Main source / role |
|---|---|---|---|
| `conv224.py` |224 /224 |8 non-overlapping three-slice tokens |Notebook4 cell19; seven OAI targets; original224 recipe |
| `conv.py` |384 /384 |11 overlapping tokens, window3/stride2 |Notebook4 cell7; original ConvNeXt2.5D recipe |
| `conv_gray384.py` |384 /384 |24 single slices, repeated to3 channels |Notebook4 cell8; first complete program retained after duplicate-AST check |
| `eff.py` |384 /384 |11 overlapping2.5D tokens |Original EfficientNetV2-S recipe |
| `tiny.py` |384 /384 |11 overlapping2.5D tokens |Original TinyViT-21M recipe |
| `dino.py` |384 /392 |11 overlapping2.5D tokens |Original DINOv2-S recipe,16 epochs |
| `rad.py` |384 /224 |24 grayscale-copy tokens |RadImageNet ResNet50; Cfg.img_size describes cache, not actual model input |
| `conv_new.py` |384 /384 |Original2.5D representation |Alternate image-report label-source recipe; not the grayscale model |
| `dino_new.py` |384 /392 |Original2.5D representation |Alternate label source and mixture; not DINOv3 |
| `dino3.py` |384 /384 |11 overlapping2.5D tokens |Recovered fourfold DINOv3-ConvNeXt-Tiny;16 epochs; backbone LR2e-5, heads2e-4 |

Training uses the corresponding resolution of OAI image caches. The supplied224 training entry requires224 OAI, while the above384-cache recipes use384 OAI. Model input differences for DINOv2/Rad do not change the native cache source.

```bash
python tools/run_pipeline.py --script code/train/conv224.py --folds 0
python tools/run_pipeline.py --script code/train/conv_gray384.py --folds 0
python tools/run_pipeline.py --script code/train/dino3.py --folds 0
# Run one selected recipe at a time; after validation, select all folds:
python tools/run_pipeline.py --script code/train/dino3.py --folds 0,1,2,3
```

These commands start training; use them only when that stage is requested. First verify external paths and inspect a real sample/batch in the intended GPU environment. No new GPU test is implied by successful static preflight.

Preserve source-specific optimizer, labels, normalization, sampling, EMA and schedule. Historical comments can be stale; executable configuration, checkpoint metadata and manifests have priority. The gray-copy source has one executable program; the original notebook had two AST-identical copies, which must not both be run. Its old comment mentions a two-fold pair, but executable defaults run all four: use explicit --folds rather than trusting that comment.

Outputs include per-fold checkpoints, OOF/Gold predictions and label-use audits. `fold0_final.pt` can be the checkpoint name inside every fold directory; the filename alone does not identify fold0. In the audited recipes, `final` is the best derived-AUC EMA epoch, not automatically the last epoch. Inspect each source's checkpoint-selection logic.

The original split groups report hashes, not verified patient/site IDs. Gold participates in fold training/validation. Do not describe it as a permanent untouched holdout. Historical384 ranking-mask handling is documented in the external audit; source preservation does not certify that every loss masks unknown OAI entries correctly.

## 6. RSNA report/VLM labeling

Use the included frozen CSVs in `labels/rsna/` for historical reproduction unless rescoring is requested. Generating new labels changes the experimental target and needs a separate frozen snapshot.

Report workflow:

```bash
python tools/prepare_report_tasks.py --train-csv data/rsna/train.csv --output work/report_tasks
```

Apply `code/labeling/rsna/report_v1_prompt.txt` and `report_v2_prompt.txt` separately to the generated report tasks. Use separate output directories. Keep UID and the documented JSONL schema. Do not expose Gold answers as annotation evidence.

Image/report workflow: run `make_montages.py` and `make_tasks.py` under `code/labeling/rsna/` using the exact arguments in `REPRODUCE.md`, then apply `vlm_image_report_prompt.txt`. Tasks list the report, image paths and orientation/series metadata. The scorer must actually review the images; task export alone is not VLM inference.

```bash
python tools/merge_rsna_scores.py --train-csv data/rsna/train.csv --results work/vlm_results --output labels/rsna/rsna_vlm_original.csv
```

Run the strict merger independently for each source. It validates IDs, coverage and probabilities, overrides official Gold by UID and saves evidence. Training performs its own documented source blend: do not blend an already blended CSV again. Remote model access is configured externally; identical prompts do not guarantee identical future model outputs.

Target order: ACL, MCL, Medial Meniscus, Lateral Meniscus, Medial OA, Lateral OA, PF OA, Effusion, Synovitis, Baker's, Contusion, Fracture. Align by UID and class names, never assumed CSV row order.

## 7. OAI download, processing, v3 mapping and packaging

Follow Section4 of `REPRODUCE.md` for the full commands. Main module groups:

- Selection/download: `oai_manifest.py`, `code/oai/download_followup_MRI.py`, and externally supplied frozen manifests under `code/oai/score_audit/`. Supply original scores, authorized NDA access and raw archives externally; do not inspect or copy credential files.
- QC: `check_completed_downloads.py`, `check_followup_downloads.py`. Download/archive corruption and geometry review are different statuses; checking present files does not prove the full queue is downloaded.
- Baseline image conversion: `preprocess_downloads.py` for224/crop160, `preprocess_downloads_384.py` for384/crop130. Both retain24 central80% slices and original bundled pixel transforms. Baseline defaults to20 items; the documented full-queue command explicitly raises `--limit`.
- Follow-up conversion: `watch_followup_dual_cache.py` generates both resolutions and records progress. Check existing processes/ledgers before launching another watcher. It can wait for unfinished downloads; absence of new output alone is not completion.
- Packaging: `pack_kaggle_images.py`, `pack_kaggle_images_384.py`, `pack_followup_caches.py --resolution 224` or `384`. Check image arrays, series masks, IDs/index and manifests together. Do not delete originals or intermediate caches unless cleanup is explicitly in scope.

For v3 labels, prepare baseline tasks with `prepare_soft_full2399_v3.py` and follow-up tasks with `prepare_followup_soft_tasks.py`. Apply `oai_vlm_soft_mapping_prompt_v3_seven_targets.txt`. Preserve same-exam MRI, professional readings, dictionary, side and visit. `generate_labels_v1.py` is a legacy rules/provenance producer used upstream by baseline task preparation; its hard labels are not v3 scoring evidence.

Seven supported targets: Medial Meniscus, Lateral Meniscus, Medial OA, Lateral OA, PF OA, Effusion, Synovitis. A probability expresses uncertainty; a validity mask expresses whether an estimate can be made. Unknown is not negative. Do not infer the other five classes from missing entries.

```bash
python tools/merge_oai_scores.py --index data/oai/kaggle_upload_384/knee_index.json --tasks code/oai/soft_full2399_v3/tasks --results work/oai_baseline_results --output labels/oai/OAI_soft_labels_2399_v3.csv
python tools/merge_oai_scores.py --index data/oai/kaggle_upload_followup6250_384/index.csv --tasks code/oai/soft_followup6250_v3/tasks --results work/oai_followup_results --output labels/oai/OAI_soft_followup6250_v3.csv
```

The strict merger checks full prompt schema, identity, visit, coverage and score/mask consistency. Preserve task evidence. Historical2399 review JSON has schema differences; reuse the externally supplied frozen final CSV when reproducing those experiments rather than inventing missing fields. Original raw score tables are not embedded.

## 8. Inference and analysis

`code/inference/ensemble6.py` implements the historical six-suite A blend. `ensemble8.py` also computes the new-two-suite B and0.7A+0.3B. They embed model-specific preprocessing/architecture support and expect actual checkpoint inputs from `config/eight_model_manifest.json`.

```bash
python tools/run_pipeline.py --script code/inference/ensemble6.py --plan
# Execute after verifying checkpoints and the native test-data layout:
python tools/run_pipeline.py --script code/inference/ensemble6.py
```

Preserve each member's input dimensions, channels, crop, normalization, precision and TTA. Adding a training recipe does not automatically register a new inference member. The newly supplied gray-copy ConvNeXt is not silently added to these historical blends. No automatic Kaggle submission is performed.

Existing `results/oof/` and `results/metrics.csv` describe stored historical predictions, not predictions rerun from every newly added script. Offline preflight can validate their fixed blends; it cannot establish new model complementarity or replicate the final team0.967 pipeline. Do not equate public3-study timing with hidden-set runtime.

## 9. Reading notebook, maintenance and provenance

`our_pipeline.ipynb` is a source-reading/restoration notebook. Its writefile cells restore source text, not a start-to-finish experiment. Read the full repository docs/configs as well; executing the reading notebook does not obtain data, weights or scoring access.

For the new entries, `config/training_source_provenance.json` records exact source hashes, notebook cells, the gray duplicate check and fixed DINOv3 training URL. `verification.json` records source-restoration coverage. The private historical `AUDIT_REPORT.json` is not distributed here. Use `REPRODUCIBILITY_AUDIT.md`, current preflight and the source checks for this public release.

When editing this package, run `python tools/validate_training_sources.py` first, then `python tools/refresh_bundle.py` to update the corresponding manifest hashes and restoration notebook; finally run `python tools/preflight.py`. To produce a ZIP, pass `--zip` with a path outside the package to refresh_bundle.py. The old parent-workspace builder is historical and can regenerate the October15 baseline without these supplements; do not run it as though it is the current release builder. Keep computational edits separate from portability changes and record new checks honestly. Never mark static validation as a successful full GPU run.

## 10. Completion report checklist

Report the selected script and source version, input/cache identity, label snapshot, fold/seed, native representation, output paths and checks actually run. State missing external assets or failures precisely. Do not claim that training, VLM review, GPU inference, publication or a particular leaderboard score happened merely because code and prompts exist.

## Public-release execution-order correction

Follow REPRODUCE.md as the authoritative stage order. Added cohort/index generation sources are `verify_cohort_scope.py`, `download_followup_indices.py`, `investigate_followup.py`, `match_followup_indices.py`, `deduplicate_followup.py`; `tools/prepare_oai_download_manifest.py` exports the full distinct three-sequence download queue. These are metadata/candidate stages, not final labels. They do not replace VLM evidence review. Baseline raw reading provenance precedes follow-up index matching. Follow-up series conversion precedes evidence-task generation, and those tasks must precede follow-up shard packing. The old order that packed before tasks was invalid.

Use `preflight.py --external --recipe NAME` for one selected training/inference recipe, instead of demanding every historical backbone/checkpoint. Only the seven explicitly reviewed RSNA label snapshots are included. Other generated labels/OOF/cohort manifests remain external and ignored; packaging excludes them even after local restore. Model training source and operational prompts were not modified by this audit.

## Frozen RSNA label release: October 25, 2026

Seven historical snapshots are now included in [labels/rsna](labels/rsna/README.md), with file hashes, exact recipes and Gold-override notes. Run `python tools/validate_rsna_labels.py`. No model computation, prompt, image preprocessing or trained-weight visibility changed. Original private Kaggle label links remain historical references; the included RSNA CSVs are the public alternative. OAI labels and trained checkpoints are not part of this release.

## RSNA label input guard

Both new initialization layouts point the four RSNA source-label inputs to the bundled public CSVs; other Kaggle mounts remain unchanged. Existing configs are preserved. Run `python tools/validate_rsna_labels.py --paths config/paths.json` before training. The three `*_labels_used.csv` files are reference outputs, not source inputs. Use `tools/run_pipeline.py`: it rejects mixed-label exports and duplicate source-file mappings before model import. Raw training files are unchanged and direct execution bypasses this protection. See the recipe/source mapping table in README.
