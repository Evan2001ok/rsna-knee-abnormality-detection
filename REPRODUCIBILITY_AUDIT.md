# Public-source reproducibility audit — October 25, 2026

## Corrections made

- Added omitted baseline cohort, follow-up index download, score-availability audit, matching and conservative deduplication sources, plus a metadata-only download-queue exporter. No participant rows or archive lists were published.
- Made package IDs caller-configurable. Follow-up index download uses NDA_PACKAGE_MAP; matching uses IDs recorded in its status file. Baseline package selection is configurable.
- Corrected stage order: baseline reading provenance precedes follow-up matching; follow-up per-series conversion precedes evidence tasks; those tasks precede packing because the packer reads task identity/timepoint checks.
- Added workspace initialization, local/Kaggle layouts, dynamic OAI path mapping and a local temporary-cache mapping.
- At the initial source audit, labels/OOF/splits/cohort manifests were external. The subsequent authorized RSNA label release adds seven frozen RSNA CSVs only; OAI labels, OOF, splits, cohort manifests and the private historical audit remain external.
- Separated pipeline/NDA dependency checks and added recipe-specific external checks. Failed requested checks now report passed=false, and missing private historical assets fail clearly.
- Restricted publication packaging to source/config/document allowlists, excluding restored labels, OOF, scores, cohort tables, generated annotation writers, local configuration and credentials.

## Fidelity and checks

The ten original training sources, RSNA 224/384 builders, OAI pixel transforms, four operational prompts and historical six/eight-suite inference remain unchanged. Edits concern packaging/path utilities, metadata/cohort tools and instructions. Training provenance is in config/training_source_provenance.json; added metadata source provenance is in config/reproduction_source_provenance.json.

```bash
python tools/check_reproduction.py
python tools/validate_training_sources.py
python tools/preflight.py
```

The reproduction check scans direct imports and 14 embedded inference sources, checks prompts and documentation links, plans ten training entries, six cache parts and two inference entries, checks actual dynamic OAI paths, and runs synthetic label/queue/configuration tests. It verifies Gold exclusion/override, invalid-score rejection, OAI visit/mask handling, exact-record queue deduplication, snapshot preservation and publication exclusions. Source checks verify recovered hashes and fold selection without importing models.

These are offline checks using synthetic fixtures, not real patient data or a new GPU training run. Dependency names cover direct imports; torch/torchvision are installed separately as a compatible pair. A complete original CUDA/codec lockfile was unavailable and has not been invented.

## External inputs and remaining verification

Authorized MRI, original reading/metadata tables, OAI labels, initial weights and trained checkpoints are external. Historical OOF/splits are optional external analysis assets. Remote LLM/VLM access is separately configured; prompts/task exporters/mergers do not imply a scoring service or identical future probabilities.

The final combined team inference and later 352 S6 batching code are outside this historical standalone 384-cache release. The final team inference is published separately as a public Kaggle notebook; see [README](README.md#final-team-inference-and-public-trained-weights). Whole-array 384 inference needs about 23GB of raw pixels for 1300 studies, so no 20GB filesystem guarantee is made. Actual GPU/DataLoader execution, DICOM decoding, storage capacity, numerical equivalence, NDA transfers and full-scale processing/training/inference have not been verified by these offline checks. See REPRODUCE.md for stage order and asset contracts.

## Frozen RSNA label release: October 25, 2026

Seven historical snapshots are now included in [labels/rsna](labels/rsna/README.md), with file hashes, exact recipes and Gold-override notes. Run `python tools/validate_rsna_labels.py`. No model computation, prompt, image preprocessing or trained-weight visibility changed. Original private Kaggle label links remain historical references; the included RSNA CSVs are the public alternative. OAI labels and trained checkpoints are not part of this release.

## RSNA source-path and single-blend audit

All ten training entries were checked after the actual AST path transformation. Their original `read_scored_labels` / `load_vlm_labels` functions were executed on the bundled source CSVs, using a metadata fixture that marks the 58 Gold placeholder rows as Gold (synthetic 0/1 values, since official Gold labels are not published) and leaves non-Gold targets missing. Every output matched its intended frozen training snapshot within 1.12e-16 on all 4,349 non-Gold studies and 12 targets. This verifies path/source/ratio/override consistency, not independent Gold provenance, fold correctness, clinical correctness or a GPU training run.

The path wrapper now rejects already mixed label exports (including renamed files retaining `label_source`) and duplicate source-file paths before model imports. Both new initializer layouts map four RSNA source labels to bundled CSVs. Existing configs remain untouched and can be checked with `python tools/validate_rsna_labels.py --paths config/paths.json`. The ten original training sources, prompts and pixel processing remain byte-identical.
