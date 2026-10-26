# Reproduction workflow

Run commands from the cloned repository root on branch `codex/our-pipeline`. This release includes source and seven frozen RSNA label snapshots: inputs marked **external** are not downloaded or supplied by the repository. The original workstation, attachments and notebook are not required. Exact historical labels/weights and the same environment are required for exact historical experiments; rescoring prompts can change the target snapshot.

## 0. Offline checks, first-time paths and dependencies

```bash
python tools/preflight.py
python tools/check_reproduction.py
python tools/validate_training_sources.py
python tools/init_workspace.py --layout local
python tools/validate_rsna_labels.py --paths config/paths.json
```

Initialize once. It creates input/output directories and `config/paths.json`; it refuses to overwrite an existing config. For original Kaggle mounts, initialize with `--layout kaggle` instead. That layout maps the four included RSNA source labels and keeps other source mount paths unchanged. On a local machine, edit the generated prefix mappings to point to your actual inputs. Longest-prefix matching applies to plain paths and the leading literal of dynamic OAI path strings. `/kaggle/temp` is mapped to `work/temp` locally. CLI arguments passed after `--` must already be real paths; they are not automatically remapped.

Follow [ENVIRONMENT.md](ENVIRONMENT.md): install a matching PyTorch/torchvision CUDA pair, then `requirements/pipeline.txt`. Downloads additionally need `requirements/nda.txt`.

```bash
python tools/preflight.py --dependencies
# Only for NDA download stages:
python tools/preflight.py --dependencies --dependency-group nda
```

Dependency presence is not a GPU/ABI test. No training is started by the checks above.

## 1. Supply external assets and select a recipe

Download links and checked public/private/application status are listed in [README: external downloads](README.md#external-downloads-and-access), with a machine-readable map in [external_resources.json](config/external_resources.json). Obtain access before following the placement table; a path example does not supply the asset.

| Input | Local template path / contract |
|---|---|
| Official RSNA tables and MRI | `data/rsna/train.csv`, `train_series.csv`, `train_series/<study>/<series>/*.dcm`; retain string study/series IDs and the official metadata columns. |
| RSNA label sources | `labels/rsna/rsna_llm_v1_original.csv`, `rsna_llm_v2_original.csv`, `rsna_vlm_original.csv`; new-label recipes also need `rsna_vlm_opus_original.csv`. Each has `StudyInstanceUID` and the12 target columns. |
| OAI soft labels | `labels/oai/OAI_soft_labels_2399_v3.csv`, `OAI_soft_followup6250_v3.csv`; keep `knee_id`, side/visit metadata, seven probabilities and their `__mask` columns. |
| OAI image caches | Native224 or384 packages under `data/oai/kaggle_upload_*`; keep `manifest.json`, all `partN_cache.npy`, `partN_series_mask.npy`, `partN_knee_ids.npy`, and baseline `knee_index.json` / follow-up `index.csv`. |
| Initial backbone weights | Map each recipe's exact source weight file/directory from `config/resource_inventory.json`. These are external, not created by initialization. |
| Historical OOF/splits | Only required for optional historical analysis via `preflight.py --assets`; not needed to start ordinary training. |

All ten preserved training entries use OAI mixed supervision. This release does not add an RSNA-only flag or implement a new OAI-pretraining stage. `conv224.py` uses native224 RSNA/OAI, while `conv.py`, `conv_gray384.py`, `eff.py`, `tiny.py`, `dino.py`, `dino_new.py`, `rad.py`, `conv_new.py`, `dino3.py` use384 caches. Actual backbone input is392 for DINOv2,224 for Rad, and384 for the other384 recipes. Gray Conv/Rad use24 single-slice tokens copied to three channels; the other384 recipes use11 overlapping three-slice tokens. See `AI_USAGE.md` for the full table.

Keep originals separate from generated outputs. Supply frozen labels for historical training, or generate them through Sections3/4 before training. Do not silently replace a missing label source with another recipe.

## 2. Build or mount RSNA image caches

For native224 run `cache224.py` with parts0/1. For native384 run `cache384.py` with parts0/1/2/3:

```bash
python tools/run_pipeline.py --script code/rsna/cache384.py --part 0 --plan
python tools/run_pipeline.py --script code/rsna/cache384.py --part 0
# Repeat for parts1,2,3. For224 use cache224.py and only parts0,1.
```

The output folder names in `config/paths.example.json` match the actual builders: `work/knee_v5_384px_24sl_central80_crop130_physicalsort_partN` or `work/knee_v5_224px_24sl_central80_crop160_physicalsort_partN`. Each contains `train_cache.npy`, `train_mask.npy`, `study_ids.npy`, `manifest.json`. Keep sidecars together; join shards by study ID.

**Storage:**4407 studies need about78.0GB of384 uint8 pixel arrays (four shards, about19.5GB each) or26.5GB at224 (two shards), before overhead. On a20GB-per-notebook output budget, build/publish each shard in a separate job; do not accumulate all shards in one output directory/session. Local sequential builds require enough space for the whole selected resolution.

## 3. RSNA labeling, if frozen labels are unavailable

The three operational prompts are in `code/labeling/rsna/`. A separately configured model/agent reviews the report/images and returns JSONL. Exporting tasks does not score them, and no universal API client or credentials are embedded.

### Report V1 and V2

```bash
python tools/prepare_report_tasks.py --train-csv data/rsna/train.csv --output work/report_tasks
```

Tasks contain non-Gold `uid` and `report` only. Apply `report_v1_prompt.txt` and `report_v2_prompt.txt` independently, replacing `{report}`. Save one object per study per JSONL line, with `uid` plus all12 class keys; each value follows the prompt's `{grade, stated, evidence}` schema. Keep V1/V2 output directories separate. The outer UID wrapper is required even though the prompt's class-only JSON example omits it.

### Image/report VLM

```bash
python code/labeling/rsna/make_montages.py --data-root data/rsna --output work/montages
python code/labeling/rsna/make_tasks.py --montages work/montages --output work/vlm_tasks
```

Apply `vlm_image_report_prompt.txt` with `{TASK_FILE}` / `{OUT_FILE}` and inspect the listed images in a fresh context. Return one JSON object per study line with `uid`,12 scores and requested evidence. Montage generation excludes Gold; studies with no available images are omitted by the image task exporter and must be resolved before a complete merge. Do not manufacture scores to fill them.

### Merge each source separately, then configure its training path

```bash
python tools/merge_rsna_scores.py --train-csv data/rsna/train.csv --results work/vlm_results --output labels/rsna/rsna_vlm_original.csv
python tools/merge_rsna_scores.py --train-csv data/rsna/train.csv --results work/report_v1_results --output labels/rsna/rsna_llm_v1_original.csv
python tools/merge_rsna_scores.py --train-csv data/rsna/train.csv --results work/report_v2_results --output labels/rsna/rsna_llm_v2_original.csv
```

The merger rejects missing/unknown/duplicate IDs and invalid scores; official Gold overrides matching UIDs. An evidence JSONL is saved alongside the CSV. For an alternate VLM source, use its own review outputs and the configured alternate VLM CSV path. Blending is done inside the selected training source; do not preblend an already mixed target.

## 4. OAI: existing packages or optional raw-data rebuild

For ordinary reproduction, supply the existing native-resolution caches and frozen v3 labels from authorized sources, then skip to Section5. OAI participant manifests, original scores, images and package inventories are external. No individual data are embedded in the source below. Work that uses OAI data must carry the OAI acknowledgement and dataset identifiers; see [README: Acknowledgements and Data Access](README.md#acknowledgements-and-data-access).

### 4.1 Rebuild baseline metadata and the candidate queue

Supply authorized NDA accounts/packages. The original score table goes at `code/oai/scores/oai_kmrisemiquant01.txt`; the field dictionary is bundled. `code/oai/baseline/image03.txt` and the package inventory under `code/oai/tool_state/NDA/nda-tools/downloadcmd/packages/<ID>/package_file_metadata_<ID>.txt` are created by the NDA tool. The recorded baseline package ID1249823 is a historical account-specific identifier; use an accessible package containing the same snapshot and set `OAI_BASELINE_PACKAGE_ID` to its ID if different.

```bash
python tools/run_pipeline.py --script code/oai_manifest.py -- YOUR_NDA_USERNAME --metadata --package=YOUR_ACCESSIBLE_BASELINE_PACKAGE_ID
python tools/run_pipeline.py --script code/oai/verify_cohort_scope.py
python tools/run_pipeline.py --script code/oai_manifest.py -- YOUR_NDA_USERNAME --images --package=YOUR_ACCESSIBLE_BASELINE_PACKAGE_ID --destination=/absolute/path/OAI/baseline_2399 --workers=8
```

These are command templates: replace username, package ID and destination first. Metadata download does not obtain the raw score table automatically; obtain that authorized table separately or use the downloader's `--scores=PACKAGE_ID` mode. `verify_cohort_scope.py` selects baseline knees by recorded major MOAKS availability, independent of image positivity, and writes private `all_mri_candidates.csv`, `major_moaks_baseline_knees.csv` and the five-sequence S3 list under `code/oai/score_audit/`. Its existing cohort logic is preserved. Snapshot differences can change counts; packing checks the recorded2399/12053 contract rather than silently choosing a new cohort.

### 4.2 Baseline QC → native caches → packaging → raw reading provenance

```bash
python tools/run_pipeline.py --script code/oai/check_completed_downloads.py
python tools/run_pipeline.py --script code/oai/preprocess_downloads.py -- --limit=20000
python tools/run_pipeline.py --script code/oai/preprocess_downloads_384.py -- --limit=20000
python tools/run_pipeline.py --script code/oai/pack_kaggle_images.py
python tools/run_pipeline.py --script code/oai/pack_kaggle_images_384.py
python tools/run_pipeline.py --script code/oai/generate_labels_v1.py -- --images /absolute/path/OAI/kaggle_upload_384 --output code/oai/labels_v1
```

Absolute CLI paths must point to the same OAI root configured in `paths.json`. Baseline conversion defaults to20 studies; the explicit limit covers the queue. The legacy label generator is used here only to expose original reading fields in `labels_v1/provenance.jsonl` and cohort membership. Its hard thresholds are not v3 annotation evidence or the training target.

### 4.3 Follow-up indices → candidates → exact-record deduplication → download queue

Copy `config/nda_packages.example.json` to a private location and fill your accessible positive integer V01/V03/V05/V06 package IDs. Set `NDA_PACKAGE_MAP` to its absolute path and `NDA_USERNAME` to your NDA username. Both follow-up downloaders use that configuration; the matching stage uses the IDs recorded in index status.

```bash
python tools/run_pipeline.py --script code/oai/download_followup_indices.py
python tools/run_pipeline.py --script code/oai/match_followup_indices.py
python tools/run_pipeline.py --script code/oai/deduplicate_followup.py
python tools/prepare_oai_download_manifest.py --candidates code/oai/score_audit/followup_MRI_candidates.csv --output code/oai/score_audit/followup_deduplicated --expected-exams 6250 --expected-archives 18798
python tools/run_pipeline.py --script code/oai/download_followup_MRI.py
```

`match_followup_indices.py` invokes `investigate_followup.py`, which needs the baseline `labels_v1/labels.csv` generated above. The four-target rule audit is a historical candidate-selection filter, not final supervision. Deduplication and the download exporter remove only identical file records and retain distinct archive/date candidates. Exporting the full18798-archive queue is not clinical or geometry adjudication; the original QC, task identity/date checks and packer's nonidentical-candidate masking still apply. Download exports refuse to overwrite an existing queue; preserve your frozen snapshot. The recorded full queue is6250 exams /18798 archives /153388869421 bytes, not6250 independent people.

### 4.4 Follow-up QC → per-series caches → evidence tasks → shards

```bash
python tools/run_pipeline.py --script code/oai/check_followup_downloads.py
python tools/run_pipeline.py --script code/oai/watch_followup_dual_cache.py
python tools/run_pipeline.py --script code/oai/prepare_followup_soft_tasks.py
python tools/run_pipeline.py --script code/oai/pack_followup_caches.py -- --resolution 224
python tools/run_pipeline.py --script code/oai/pack_followup_caches.py -- --resolution 384
```

**Tasks must precede follow-up packing.** The packer reads task identity/timepoint confirmation and masks unresolved/conflicting image slots. The watcher may wait for unfinished downloads; absent output is not proof of completion. Inspect quality/preprocessing summaries, geometry review, masks, IDs, `UPLOAD_READY.json` and manifests. Keep original archives and do not enable `--release-generated-series` on the first run. Dual caches and raw archives need separate disk-capacity planning; no extra full-volume copy is necessary.

### 4.5 Baseline/follow-up VLM mapping and strict merging

```bash
python tools/run_pipeline.py --script code/oai/prepare_soft_full2399_v3.py -- 0 2399
```

Baseline tasks need packed384 images, baseline reading provenance and the384 series ledger. Follow-up tasks were produced before packing above. Apply `code/oai/oai_vlm_soft_mapping_prompt_v3_seven_targets.txt`, replacing its task/output paths, and review the same-exam MRI, raw readings and dictionary. First check about100 cases, then complete the queue. Return the exact v3 schema: seven `p/mask` targets, evidence and provenance. Keep unknown entries masked; do not substitute another visit or old hard labels. The English prompt reference is explanatory; the preserved original is operational.

```bash
python tools/merge_oai_scores.py --index /absolute/path/OAI/kaggle_upload_384/knee_index.json --tasks code/oai/soft_full2399_v3/tasks --results work/oai_baseline_results --output labels/oai/OAI_soft_labels_2399_v3.csv
python tools/merge_oai_scores.py --index /absolute/path/OAI/kaggle_upload_followup6250_384/index.csv --tasks code/oai/soft_followup6250_v3/tasks --results work/oai_followup_results --output labels/oai/OAI_soft_followup6250_v3.csv
```

The same labels may join the matching224 cache by complete identity, not row position. The merger requires `prompt_version=oai-soft-mapping-v3-seven-targets`, `definition_version`, `definition_alignment` and the full per-target schema. Old2399 review JSON has historical schema differences; reuse an authorized frozen final CSV for exact historical training rather than inventing fields.

## 5. Check one selected recipe, then train

```bash
python tools/preflight.py --external --recipe conv
python tools/run_pipeline.py --script code/train/conv.py --folds 0 --plan
python tools/run_pipeline.py --script code/train/conv.py --folds 0
```

External preflight checks existence, not tensor correctness. It is deliberately run before training so missing original label sources cannot silently invoke a source fallback. Inspect real samples/masks and a real forward/backward batch in the intended GPU environment. The unchanged original training entry does not expose a universal `--check-only` option. After one fold works, train the other folds with `--folds 1,2,3`; use `--folds 0,1,2,3` for a new full run, not as a promised checkpoint resume. For224 use recipe `conv224` and its native224 caches. The adapter overrides explicit fold calls while preserving model computations.

Outputs remain in each source's configured work directory. A `fold0_final.pt` filename may occur inside every `foldN/`; the directory and checkpoint metadata identify the fold. Gold is held out in its assigned fold but can enter training/checkpoint selection in other folds. The report-hash grouping is not verified patient/site grouping. Adding a recovered training entry does not silently register it in the historical inference ensemble.

## 6. Inference with matching trained checkpoints

Supply official test DICOM, `test_series.csv` and `sample_submission.csv`; training caches/label CSVs are not required. Six suites need24 checkpoints; eight suites need32. Map `config/eight_model_manifest.json` run roots to the corresponding model outputs/dataset mounts. Newly trained output folder names may differ from the historical run-root names: add an explicit prefix mapping, rather than assuming the broad `/kaggle/input` fallback will find them.

```bash
python tools/preflight.py --external --recipe ensemble6
python tools/run_pipeline.py --script code/inference/ensemble6.py --plan
python tools/run_pipeline.py --script code/inference/ensemble6.py
```

For eight suites use `ensemble8`. Each preserves its actual input dimensions, representation, normalization, precision and TTA; do not concurrently run both ensembles in one GPU session. The six-suite output is `work/submission.csv` under local mappings; eight-suite inference also saves its fixed A/B/blend predictions. No submission is automated.

These are the historical384 array-cache ensembles, **not the later352 S6 batched-cache pipeline or final team0.967 ensemble**. They retain a whole shared test cache. At1300 studies the raw384 array alone is about23.0GB; check the actual temporary filesystem and free space before full inference. This release does not claim that a20GB filesystem can hold that cache or that public-three-study timing predicts hidden-set runtime.

## 7. Verification boundaries and maintenance

`tools/check_reproduction.py` audits direct and embedded Python imports, prompt/link presence, ten training dry-runs, six RSNA cache-part dry-runs, inference entry plans, actual dynamic OAI path mappings and synthetic RSNA/OAI merge behavior. It uses no real patient data and does not import models. `tools/validate_training_sources.py` checks recovered source hashes and fold-adapter fidelity. `preflight.py --assets` additionally checks externally restored historical OOF/labels; it fails clearly if absent.

GPU training/inference, MRI conversion at full scale, NDA downloads and remote VLM scoring have not been rerun by these checks. Complete numerical reproduction requires external asset snapshots and an actual compatible GPU run. No original CUDA lockfile has been invented.

After source/doc edits, run the reproduction/source checks, then `python tools/refresh_bundle.py` to regenerate the reading notebook and source hashes, then `python tools/preflight.py`. Publication packaging uses a source allowlist and excludes local restored labels, OOF, cohort/score files, credentials and weight artifacts.

## Frozen RSNA label release: October 25, 2026

Seven historical snapshots are now included in [labels/rsna](labels/rsna/README.md), with file hashes, exact recipes and Gold-override notes. Run `python tools/validate_rsna_labels.py`. No model computation, prompt, image preprocessing or trained-weight visibility changed. Original private Kaggle label links remain historical references; the included RSNA CSVs are the public alternative. OAI labels and trained checkpoints are not part of this release.

## Use RSNA sources once, then verify the training targets

Follow the recipe table in [README](README.md#frozen-labels-and-optional-historical-analysis). Supply the four source CSVs already included in `labels/rsna/`; the three `*_labels_used.csv` snapshots are reference outputs, never source inputs. The documented `tools/run_pipeline.py` wrapper checks for mixed-target inputs and duplicate source paths before model imports. An old local config is not migrated automatically; copy its four RSNA mappings from `config/paths.example.json` while preserving unrelated mounts, and run `python tools/validate_rsna_labels.py --paths config/paths.json`. Optional `--training-loaders` executes the original ten label loaders without training, using a Gold metadata fixture to compare all values with the saved reference snapshots.
