# Public source release

## Scope

This is our side of the RSNA Knee pipeline, on branch `codex/our-pipeline`. The original working folder and private handover are unchanged. The repository contains RSNA 224/384 preprocessing, ten original training entry points, RSNA LLM/VLM prompts and merge scripts, OAI preprocessing and v3 mapping prompts/tools, historical six/eight-suite inference, dependency declarations and an AI usage guide.

The final combined team inference and teammates' private pipelines are not included. The recorded leaderboard numbers describe historical experiments under the documented conditions; publishing source is not an independent reproduction.

## External assets

See [README: downloads and access](README.md#external-downloads-and-access) for checked public/private/application status, exact label filenames, upstream initialization weights and task-trained checkpoint links. [external_resources.json](config/external_resources.json) maps all55 concrete original input paths to31 historical resource pages. Private links are identifiers, not public download grants.

Obtain competition MRI and official tables through authorized competition access. Obtain OAI MRI and original reading tables through authorized access and follow applicable terms. The seven frozen RSNA label snapshots are included in `labels/rsna/`; supply OAI labels and trained checkpoints separately. This repository deliberately excludes participant/cohort manifests, OAI labels, splits, OOF probabilities, MRI images, credentials and weights. `code/oai/score_dictionary.csv` describes score fields and is not a participant table.

Training scripts keep original asset filenames and Kaggle mounts for provenance. Configure the actual locations in a local `config/paths.json`, using `config/paths.example.json`. Paths mentioning `labels/oai/`, `results/` or `code/oai/score_audit/` require external inputs; no example rows are substituted for real data.

For exact historical reproduction, the original frozen labels, fold assignments and checkpoint assets are needed. Re-running labeling prompts can produce different outputs. Prompt task exporters and strict result mergers are included; remote scoring service access is external.

## Checks

```bash
python tools/preflight.py
python tools/check_reproduction.py
python tools/validate_training_sources.py
python tools/run_pipeline.py --script code/train/conv224.py --folds 0 --plan
```

The above checks are offline and do not run training. After restoring the private handover assets, `python tools/preflight.py --assets` additionally checks label masks, split/OOF alignment and fixed blends. It fails if those assets are missing. Follow ENVIRONMENT.md for actual dependencies and GPU compatibility, then REPRODUCE.md for the stage sequence and AI_USAGE.md for the module map.

## Source fidelity

`conv224.py` is the exact original notebook cell. `conv_gray384.py` retains one complete program from two AST-identical copies. `dino3.py` is recovered executed source from the fixed Kaggle version recorded in `config/training_source_provenance.json`. No model architecture, training loss or image processing was changed for publication. Documentation and package tools were adapted for external data. The reproducibility audit also added original cohort/index-selection source and a metadata-only download-queue exporter; account-specific package selection is configurable. Follow-up task generation is documented before packing, and the path adapter handles dynamic OAI paths. See REPRODUCIBILITY_AUDIT.md for the exact checks and boundaries.

No new project-wide license is assigned in this release. Existing third-party attribution and upstream terms remain applicable.

## Frozen RSNA label release: October 25, 2026

Seven historical snapshots are now included in [labels/rsna](labels/rsna/README.md), with file hashes, exact recipes and Gold-override notes. Run `python tools/validate_rsna_labels.py`. No model computation, prompt, image preprocessing or trained-weight visibility changed. Original private Kaggle label links remain historical references; the included RSNA CSVs are the public alternative. OAI labels and trained checkpoints are not part of this release.

## RSNA label input guard

Both new initialization layouts point the four RSNA source-label inputs to the bundled public CSVs; other Kaggle mounts remain unchanged. Existing configs are preserved. Run `python tools/validate_rsna_labels.py --paths config/paths.json` before training. The three `*_labels_used.csv` files are reference outputs, not source inputs. Use `tools/run_pipeline.py`: it rejects mixed-label exports and duplicate source-file mappings before model import. Raw training files are unchanged and direct execution bypasses this protection. See the recipe/source mapping table in README.
