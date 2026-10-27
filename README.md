# RSNA Knee: Shared Improvement Roadmap and Essential Source Handover

> **Public code and RSNA label edition.** This branch includes pipeline source, prompts, configurations and seven frozen RSNA label CSVs in [labels/rsna](labels/rsna/README.md). OAI labels, OOF predictions, split/cohort tables, MRI caches and trained weights remain external. No final team-score reproduction or GPU rerun is claimed. See [PUBLIC_RELEASE.md](PUBLIC_RELEASE.md).

The high scores came from improving both what the models see and what they are trained to predict. The work progressed through MRI cropping, slice order and sequence organization; class-specific spatial and temporal evidence selection; report/VLM label blending; and OAI supervision. Different backbones and label recipes then provided complementary predictions.

**EfficientNetV2-S, ConvNeXt-Tiny, DINOv2, TinyViT and the other backbones belong to the same evolving approach.** Early EfficientNet experiments established cropping, resolution, augmentation and EMA improvements. Later experiments, including those on ConvNeXt, refined the shared encoder, attention, sampling, labels and external-data usage. The resulting recipe was transferred across backbones. The notebook groups experiments by model to identify where a measurement was made, not to describe independent development routes.

This public edition contains essential source and model configuration; seven frozen RSNA labels are included; OAI labels, OOF predictions and splits remain external. It does not contain MRI data or model weights. Historical experiments remain in the separate full archive; the explanations of the individual improvements are retained here.

## Start here: public source reproduction

```bash
git clone --branch codex/our-pipeline --single-branch https://github.com/Evan2001ok/rsna-knee-abnormality-detection.git
cd rsna-knee-abnormality-detection
python tools/preflight.py
python tools/check_reproduction.py
python tools/init_workspace.py --layout local
python tools/validate_rsna_labels.py --paths config/paths.json
```

Use `--layout kaggle` instead of `local` to keep original Kaggle mounts for other assets; both layouts map the four RSNA source-label inputs to this repository's CSVs. Run initialization once; it preserves an existing configuration. These commands do not download data or start training. Next edit `config/paths.json`, install the environment, and follow [REPRODUCE.md](REPRODUCE.md) in order. [REPRODUCIBILITY_AUDIT.md](REPRODUCIBILITY_AUDIT.md) records the checked scope.

**Execution order:** configure paths/environment → obtain authorized inputs → build/mount matching RSNA and OAI caches → supply frozen labels or generate/review/merge them → check the selected recipe's assets → train one fold → train the remaining folds → map the selected checkpoints → infer. All ten preserved training recipes use OAI; there is no supplied `--disable-oai` flag. OAI rebuilding is optional when its packaged caches and labels are already available.

For example, after preparing the384 inputs:

```bash
python tools/preflight.py --dependencies
python tools/preflight.py --external --recipe conv
python tools/run_pipeline.py --script code/train/conv.py --folds 0 --plan
# Remove --plan only after the required inputs and GPU environment are ready.
```

The public source is not a self-contained data/weight release. Frozen RSNA label CSVs are included; OAI labels, original reading tables, MRI, splits/OOF and checkpoints must be provided separately. Historical OOF numbers below are recorded analysis, not files distributed with this branch.

## External downloads and access

**Checked October25,2026.** The GitHub source is public; many historical Kaggle input artifacts are still **Private**. Links below identify the exact resources, not a promise that every reader can download them. Seven RSNA label snapshots are now published in this branch; the original private Kaggle resources and all trained-weight visibility remain unchanged.

**Access key:** **Public upstream** = file listing inspected without sign-in; **Account/terms** = provider login and required terms; **Apply** = official OAI access process; **Private** = an explicit Kaggle Private header, requiring the owner's authorized sharing; **Unverified** = accessible in our signed-in session, but public download was not independently established. Full binary downloads were not run for this check.

### Data and image caches

| Resource and link | Access | What to obtain / alternative |
|---|---|---|
| [Official RSNA competition data](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/data) · [rules](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/rules) | **Account/terms** | Official MRI, `train.csv`, `train_series.csv`; inference needs available test files and `sample_submission.csv`. Sign in and satisfy the competition's access conditions. Hidden test MRI is supplied by competition reruns, not a public download. |
| [Original OAI data portal](https://nda.nih.gov/oai/) · [request permissions](https://nda.nih.gov/user/dashboard/data_permissions) | **Apply** | Use your own NDA account and complete current OAI permission/data-use requirements. Obtain MRI archives, `image03` indices, `oai_kmrisemiquant01.txt` and package inventories. Personal package IDs in old experiments are not reusable universal download IDs. |
| RSNA224 cache outputs: [part0](https://www.kaggle.com/code/zhuyifanss/fork-of-fork-of-dataprepare-of-rsnav2-part1), [part1](https://www.kaggle.com/code/zhuyifanss/fork-of-fork-of-dataprepare-of-rsnav2-part2) | **Private** | Notebook outputs, not a single Dataset. Need both native224/crop160 shards. Alternatively build them from authorized RSNA MRI with `code/rsna/cache224.py`. |
| RSNA384 cache outputs: [part0](https://www.kaggle.com/code/zhuyifanss/dataprepare-of-rsnav5part1), [part1](https://www.kaggle.com/code/zhuyifanss/dataprepare-of-rsnav5part2), [part2](https://www.kaggle.com/code/minminandy/dataprepare-of-rsnav5part3), [part3](https://www.kaggle.com/code/minminandy/dataprepare-of-rsnav5part4) | **Private** | Need all four native384/crop130 shards, or rebuild with `code/rsna/cache384.py`. Keep each shard's cache, mask, study IDs and manifest together. |
| [OAI baseline2399,224](https://www.kaggle.com/datasets/minminandy/oai-baseline-2399-224px-images) | **Private** | Native224 baseline image package for `conv224.py`; no disease labels in this image package. |
| [OAI follow-up6250,224](https://www.kaggle.com/datasets/minminandy/oai-followup-6250-224px-images) | **Private** | Native224 follow-up package; join by complete knee/visit identity. |
| [OAI baseline2399,384](https://www.kaggle.com/datasets/minminandy/oai-baseline-2399-384px-images) | **Private** | Native384 baseline package. Retain every shard listed by its manifest, IDs/masks and `knee_index.json`. |
| [OAI follow-up6250,384](https://www.kaggle.com/datasets/minminandy/oai-followup-6250-384px-images) | **Private** | Native384 follow-up package, manifest and `index.csv`. Other384 training recipes need this resolution even when model input is later resized. |

OAI image-package sharing is subject to the owner's decision and applicable source access conditions; a Kaggle share does not replace NDA/OAI authorization. If the private packages cannot be shared, follow the authorized raw-data rebuild in [REPRODUCE.md](REPRODUCE.md), rather than treating the links as public.

### Frozen labels and optional historical analysis

| Resource and link | Access | Exact file / use |
|---|---|---|
| [RSNA report labels V1](labels/rsna/rsna_llm_v1_original.csv) | **Public in this repository** | Frozen `rsna_llm_v1_original.csv`; original private Kaggle filename `train_scored.csv`. |
| [RSNA report labels V2](labels/rsna/rsna_llm_v2_original.csv) | **Public in this repository** | Frozen `rsna_llm_v2_original.csv`; historical source filename `labels_gold_override.csv`. |
| [RSNA original image/report VLM labels](labels/rsna/rsna_vlm_original.csv) | **Public in this repository** | Frozen `rsna_vlm_original.csv`; original filename `labels_vlm_imgrep.csv`. |
| [RSNA alternate image-report VLM labels](labels/rsna/rsna_vlm_opus_original.csv) | **Public in this repository** | Frozen `rsna_vlm_opus_original.csv`; used by `conv_new` / `dino_new`. |
| [Three actual mixed training snapshots](labels/rsna/README.md) | **Public in this repository** | `conv_labels_used.csv`, `conv_new_labels_used.csv`, `dino_new_labels_used.csv`; 4,349 mixed rows + 58 Gold placeholder rows each (official Gold labels not included). |
| [OAI baseline v3 labels](https://www.kaggle.com/datasets/minminandy/oai-soft-labels-2399-v3) | **Private** | `OAI_soft_labels_2399_v3.csv`; seven probabilities, validity masks and identity metadata. |
| [OAI follow-up v3 labels](https://www.kaggle.com/datasets/minminandy/oai-soft-followup6250-v3) | **Private** | `OAI_soft_followup6250_v3.csv`; retain visit as well as knee identity. |
| [Historical RSNA official-table mirror](https://www.kaggle.com/datasets/minminandy/rsna-file) | **Private; optional mirror** | Its `train.csv` is an original source path. Use the authorized official competition table instead via the provided path mapping; this mirror is not an extra label source. |
| Historical OOF and split tables | **Not published here** | No public download URL has been verified. Ask the authors for an authorized frozen snapshot if reproducing the recorded OOF analysis. Ordinary training/inference does not require these stored analysis outputs. |

For private artifacts, contact the owner shown by the Kaggle link with your Kaggle username and required access approvals; availability is not guaranteed. We have not sent a sharing request or changed visibility. If frozen labels are unavailable, the original prompts, task exporters and strict mergers can generate a new reviewed snapshot using separately configured model access. **Newly generated labels are not guaranteed to reproduce historical probabilities or leaderboard scores.**

The seven CSVs each contain 4,407 unique studies: 4,349 derived rows and 58 Gold placeholder rows (all targets `0.5`). The official Gold labels are competition data and are not published; training replaces the placeholders from your own `train.csv`. See [label documentation](labels/rsna/README.md) and [SHA256/recipe manifest](labels/rsna/manifest.json). Both new local and Kaggle configurations use the included source CSVs. Existing `config/paths.json` is never overwritten: compare its four RSNA mappings with `config/paths.example.json`, then run `python tools/validate_rsna_labels.py --paths config/paths.json`.

**Use source labels, mix exactly once.** The unchanged training entries read VLM, V1 and V2 and compute the mixture internally. The three `*_labels_used.csv` files are reference outputs only; do not map any of the three source inputs to them. The documented `tools/run_pipeline.py` entry refuses mixed-target CSVs (including renamed exports retaining `label_source`) and duplicate source-file mappings before importing the model. Do not bypass this wrapper for the documented reproduction path.

Let `L = 0.7 V1 + 0.3 V2`; `VLM-A` denotes the original image-report source and `VLM-B` the alternate source:

| Training entry | VLM input | Internal target before Gold override | Reference output only |
|---|---|---|---|
| `conv`, `conv224`, `conv_gray384`, `eff`, `dino`, `dino3`, `rad`, `tiny` | `rsna_vlm_original.csv` | `0.7 VLM-A + 0.3 L` = `0.7 VLM-A + 0.21 V1 + 0.09 V2` | `conv_labels_used.csv` |
| `conv_new` | `rsna_vlm_opus_original.csv` | `0.7 VLM-B + 0.3 L` = `0.7 VLM-B + 0.21 V1 + 0.09 V2` | `conv_new_labels_used.csv` |
| `dino_new` | `rsna_vlm_opus_original.csv` | `0.5 VLM-B + 0.5 L` = `0.5 VLM-B + 0.35 V1 + 0.15 V2` | `dino_new_labels_used.csv` |

Every recipe then overrides the same 58 complete official Gold rows from the separately obtained competition `train.csv`. `label_source=vlm07_llm03` is a legacy export string, not the actual new-DINO ratio. The code's label ratio is separate from the RSNA/OAI sample ratio and any ensemble weights. A label-path check alone does not supply MRI, OAI inputs or initialization weights.

### Initial backbone weights for training

These initialize a backbone; they are **not** the task-trained weights needed by `ensemble6.py` / `ensemble8.py`. The public upstream links are alternatives for obtaining the named architecture's initialization; their hashes have not been matched to the historical Kaggle copies. Verify state-dict keys/shapes and retain the recipe's original normalization. Configure an explicit file mapping when filenames differ; do not pass a safetensors file to a torch-load-only entry.

| Backbone | Public/upstream weight source | Original Kaggle mirror and checked access | Required file / recipe |
|---|---|---|---|
| ConvNeXt-Tiny | [timm files](https://huggingface.co/timm/convnext_tiny.fb_in22k_ft_in1k_384/tree/main) — **Public upstream** | [convnext-tiny](https://www.kaggle.com/code/zhuyifanss/convnext-tiny) — **Unverified** | `pytorch_model.bin`; `conv`, `conv224`, `conv_gray384`, `conv_new`. |
| EfficientNetV2-S | [timm files](https://huggingface.co/timm/tf_efficientnetv2_s.in21k_ft_in1k/tree/main) — **Public upstream** | [efficientv2-s](https://www.kaggle.com/code/minminandy/efficientv2-s) — **Unverified** | Source mirror uses `tf_efficientnetv2_s_in21k_ft_in1k.pth`; upstream lists `pytorch_model.bin`. Map the actual compatible torch state-dict file explicitly. |
| DINOv2-S | [timm files](https://huggingface.co/timm/vit_small_patch14_dinov2.lvd142m/tree/main) — **Public upstream** | [dinov2s](https://www.kaggle.com/code/minminandy/dinov2s) — **Private** | `pytorch_model.bin`; `dino` / `dino_new`, actual model input392. |
| TinyViT-21M-384 | [timm files](https://huggingface.co/timm/tiny_vit_21m_384.dist_in22k_ft_in1k/tree/main) — **Public upstream** | [tiny-vit-21m-384](https://www.kaggle.com/code/minminandy/tiny-vit-21m-384) — **Private** | `pytorch_model.bin`; `tiny`. |
| DINOv3-ConvNeXt-Tiny | [timm files](https://huggingface.co/timm/convnext_tiny.dinov3_lvd1689m/tree/main) — **Public upstream; [DINOv3 License](#dinov3-pretrained-weights)** | [dinvo3convnext-timm](https://www.kaggle.com/code/lihu6666/dinvo3convnext-timm) — **Private** | `model.safetensors`; `dino3`. Use the linked timm-format source and review its license; other distributors may impose their own access flow. |
| RadImageNet ResNet50 | [Official repository](https://github.com/BMEII-AI/RadImageNet) → [PyTorch archive](https://drive.google.com/file/d/1RHt2GnuOYlc_gcoTETtBDSW73mFyRAtR/view) — **Public instructions; provider login may be needed** | [Kaggle PyTorch model v1](https://www.kaggle.com/models/dew389/radimagenet50/pytorch/default/1) — **Unverified** | Historical mirror file `ResNet50.pt`; anonymous Drive inspection showed sign-in. Verify the extracted PyTorch ResNet50 state dict; do not substitute TensorFlow/H5 weights. Raw RadImageNet images are not needed for this recipe. |

### Task-trained checkpoints for the historical ensembles

Every linked dataset below currently displays **Private**. Ask its owner for authorized access, or train a new suite with the supplied source and authorized inputs. Public ImageNet/DINO/RadImageNet initialization cannot substitute for these learned competition heads/checkpoints.

| Suite | Dataset link | Historical run-root folder | Used in |
|---|---|---|---|
| Original Conv | [distillstage1conv384](https://www.kaggle.com/datasets/minminandy/distillstage1conv384) | `rsna_convnext_tiny_384_oai8649_mix58_42_seven` | A / six and eight |
| EfficientNetV2-S | [distillstage1effs384](https://www.kaggle.com/datasets/minminandy/distillstage1effs384) | `rsna_effv2s_384_oai8649_mix58_42_seven` | A |
| DINOv2-S | [distillstage1dinov2s384](https://www.kaggle.com/datasets/zhuyifanss/distillstage1dinov2s384) | `rsna_dinov2s_392_oai_mix_unfreeze9` | A |
| DINOv3-ConvNeXt | [dinov3-convnext-tiny-lihu-of-fork-of-rsna-cd4e8a](https://www.kaggle.com/datasets/lihu6666/dinov3-convnext-tiny-lihu-of-fork-of-rsna-cd4e8a) | `rsna_dinov3_convnext_tiny_384_oai8649_mix58_42_seven_16ep_splitlr` | A |
| RadImageNet ResNet50 | [distillstage1radimageresnet224](https://www.kaggle.com/datasets/zhuyifanss/distillstage1radimageresnet224) | `rsna_radimagenet50_cache384_input224_oai_graycopy24` | A |
| TinyViT | [distillstage1tinyvit384](https://www.kaggle.com/datasets/minminandy/distillstage1tinyvit384) | `rsna_tinyvit21m384_oai8649_mix58_42_seven` | A |
| New-label Conv | [distillstage1conv-difflabel384](https://www.kaggle.com/datasets/zhuyifanss/distillstage1conv-difflabel384) | `rsna_convnext_tiny_384_oai8649_mix58_42_seven` | B / eight only |
| New-label DINOv2 | [distillstage1dino-difflabel394](https://www.kaggle.com/datasets/zhuyifanss/distillstage1dino-difflabel394) | `rsna_dinov2s_392_oai_mix_unfreeze9` | B / eight only |

Inside each run root retain `fold0/fold0_final.pt`, `fold1/fold0_final.pt`, `fold2/fold0_final.pt`, `fold3/fold0_final.pt`: six suites need24 files; eight need32. The `394` in the new DINO dataset slug is a historical name, **not** a new model input size; its audited input is392. Exact mount paths and historical dataset-version identifiers are in [eight_model_manifest.json](config/eight_model_manifest.json). A dataset's latest page can change; verify its contents against that manifest.

Dedicated task-trained checkpoints for the recovered `conv224.py` and `conv_gray384.py` entries have no separately verified public download in this catalog. Request the corresponding frozen artifacts or train them; they are not silently substituted for the historical Conv ensemble member.

**Minimum acquisition paths:** cache rebuilding needs official access plus the cache source; retraining additionally needs the selected recipe's OAI packages, label snapshots (or new scoring) and initialization weights; running a trained ensemble needs only its task-trained checkpoints and authorized test inputs. The private snapshots remain a real prerequisite for exact historical reproduction. Later352 S6, R7 and the final combined team models are not dependencies of this branch.

The machine-readable [external resource catalog](config/external_resources.json) maps every concrete original input path to its resource link, checked access status and consumer. Follow [REPRODUCE.md](REPRODUCE.md) for placement and run order. Public/private checks describe this inspection date; they do not grant access or promise future availability.

## 1. Results and measurement conventions

### 1.1 Recorded model-suite results

A “single-model suite” means one architecture and training recipe, potentially containing four fold checkpoints. It does not necessarily mean one checkpoint. These LB results come from the user's notebook and experiment notes.

| Model/stage | Input and inference conditions | Recorded LB |
| --- | --- | ---: |
| Early ConvNeXt-Tiny baseline | 224, 2.5D, shared encoder | 0.938 |
| Improved ConvNeXt-Tiny | 224, 2.5D, no TTA | 0.957 |
| High-scoring ConvNeXt-Tiny | 384, 2.5D, TTA | 0.964 |
| High-scoring EfficientNetV2-S | 384, 2.5D, shared encoder, TTA | 0.963 |
| High-scoring DINOv2-S | Listed under 384 experiments; actual input 392, no TTA | 0.963 |
| High-scoring TinyViT-21M | 384, 2.5D, TTA | 0.961 |
| RadImageNet ResNet50 | 384 cache resized to 224, TTA, as recorded in the notebook | 0.957 |

There is a RadImageNet version distinction: the experiment note calls this result “2.5D,” whereas the checkpoint version audited for the eight-suite ensemble actually uses **24 single-slice grayscale tokens, each replicated into three channels**. The implementation is preserved rather than changed to match an old heading. No scores have been invented for blank notebook entries, grayscale variants or DINOv3.

### 1.2 Interpreting individual gains

Increments such as `+0.001` and `+0.003` are measurements from their respective experiments. They explain the development process, but they are not a single controlled ablation table with identical conditions. They must not be summed to predict the final LB.

**224 was the main inexpensive experimentation setting; 384 was the transfer setting for the mature recipe.** According to the user's experiments, attention, sampling, label and external-data improvements validated at 224 continued to produce positive gains of a similar magnitude at 384. This motivated screening changes at 224 before transferring the recipe. It does not establish an independently measured, numerically identical 384 gain for every change.

The 0.957 and 0.964 endpoints also differ in TTA conditions. Their `+0.007` difference cannot be attributed entirely to resolution. This handover audits source, notes and existing OOF; it does not rerun every experiment or independently recheck every historical submission.

## 2. One improvement roadmap: input quality to a shared high-scoring recipe

The roadmap is: **cropping/resolution → augmentation/EMA → shared multi-sequence encoder and class attention → physical slice order and 2.5D sampling → report/VLM label blending → OAI soft supervision and candidate pooling → higher-resolution transfer → backbone/label-recipe ensembling.** Some changes were developed in overlapping iterations. The table groups them by purpose; its experiment column identifies the measurement source, not an additional gain from changing backbones.

### 2.1 Recorded changes and their purpose

| Component | Change | Recorded result/gain | Experiment context and purpose |
| --- | --- | --- | --- |
| Input quality | Center crop 160 → 130 mm; resolution 224 → 320 | **0.925 → 0.937**, approximately `+0.012` | EfficientNet stage. More effective pixels for knee structures. These were joint changes, so their effects cannot be separated. |
| Training robustness | Add augmentation | **0.937 → 0.942**, approximately `+0.005` | EfficientNet stage. Reduce dependence on fixed pose and appearance; retained in later recipes. |
| Inference stability | TTA | Approximately `+0.003` | EfficientNet measurement. Later retained with a smaller view budget. |
| Parameter stability | Weight EMA | Approximately `+0.002` | EfficientNet measurement. Later validation/checkpoint saving uses EMA. |
| Image appearance | Add a set of intensity augmentations | Approximately `+0.003` | EfficientNet measurement for the set as a whole: contrast, noise, bias field and effective resolution. |
| Architecture exploration | Model five slots independently | Approximately `+0.001`; limited benefit | EfficientNet exploration. Smaller models benefited more in the notes. The main recipe subsequently used a shared encoder with sequence-specific aggregation. |
| Shared-architecture baseline | 224 / 2.5D / shared encoder | **0.938** | A ConvNeXt measurement point within the common development route. |
| Report supervision | Blend a second LLM label source | `+0.001` | ConvNeXt measurement. Reduce dependence on one report-labeling source's scoring tendencies. |
| Evidence localization | Label-wise attention plus spatial attention pooling | `+0.004` | ConvNeXt measurement. Select spatial and temporal evidence separately for each class; transferred to other backbones. |
| Slice order | Improve preprocessing order | `+0.001` | ConvNeXt measurement. Avoid feeding inconsistent slice sequences to the aggregation model. |
| Depth context | Overlap / jitter | `+0.003` | ConvNeXt measurement. Overlapping three-slice windows retain local depth context; jitter reduces fixed grouping-boundary dependence. |
| Sampling coverage | 18 → 24 slices | `+0.001` | ConvNeXt measurement. Increase depth coverage, with matching cache and model changes. |
| Ranking objective | Add ranking loss | `+0.001` | ConvNeXt measurement. Add a within-batch, per-class ranking constraint alongside BCE. |
| Image/report supervision | Blend VLM and LLM labels | `+0.003` | ConvNeXt measurement. Bring image evidence and competition label definitions into supervision; reused across backbones. |
| External supervision | Mix OAI after aligning labels | `+0.003` | ConvNeXt measurement. Add knee-image supervision for corresponding targets; later refined into v3 soft scores. |
| External candidate pool | Expand OAI and control sampling | `+0.001` | ConvNeXt measurement. Increase the available cases without uncontrolled growth in external-data exposure per epoch. |
| Mature recipe at 224 | Improved 224 / 2.5D, no TTA | **0.957** | ConvNeXt endpoint for the mature 224 recipe. |
| Higher-resolution transfer | Transfer effective recipe to 384, retaining backbone/TTA settings | ConvNeXt **0.964**; EfficientNet and DINOv2 approximately **0.963** | Results from the same approach on different backbones; see Section 1 for actual input and inference conditions. |
| Side experiment | Ensemble-prediction distillation | Approximately `+0.001` in one experiment; `+0.002` in another mixture | Some positive observations, followed by experiments with no gain. Not a prerequisite for every high-scoring model. |

The intensity-augmentation `+0.003` is for the group, not for every transformation separately. The shared encoder, backbone changes and final endpoints have not been assigned unsupported independent gains.

The OAI-expansion note mentions a **6:2** ratio, while the audited saved training code uses approximately **58% RSNA / 42% OAI**. These are different experiment records, not interchangeable descriptions of one configuration. Use the relevant code and log for each run.

### 2.2 Why the changes work together

The changes address different parts of the pipeline and can therefore coexist in one model suite:

- **Physical ordering, cropping and 24-slice coverage** determine what image evidence reaches the model.
- **Overlapping 2.5D windows** let a pretrained 2D encoder see adjacent slices and local depth information.
- **Class-specific spatial and temporal attention** lets meniscus, ligament, cartilage and fluid targets select different evidence rather than share one averaged representation.
- **Multi-sequence fusion** learns which slots matter for each target while excluding missing sequences.
- **VLM/LLM blending and OAI soft labels** improve the supervision target and its alignment with the competition's definition of a positive finding.
- **Augmentation, EMA and TTA** improve training robustness, parameter stability and inference stability, respectively.

Together these changes form the mature recipe. EfficientNet, ConvNeXt, DINOv2 and TinyViT implement this recipe while retaining their own pretraining, input dimensions, normalization and necessary adaptations.

## 3. Transferring the same approach across resolutions and backbones

### 3.1 Fast validation at 224, transfer to 384

224 and 384 are not independently developed pipelines. Most changes were screened at the less expensive 224 setting, and effective preprocessing, attention, sampling, supervision and training settings were transferred to 384. The user's experiments observed positive gains of a comparable magnitude after transfer. Higher-resolution runs also used the appropriate tighter physical crop and pretrained backbone; input construction must still match the checkpoint.

The correct narrative is “improve the shared recipe, then implement it at higher resolution and across backbones.” Scores such as 0.938, 0.957, 0.963 and 0.964 are measurements at different stages and conditions, not separate methods distinguished by model name. Subtracting endpoints across conditions does not isolate a single change.

### 3.2 Share the recipe, retain model-specific conditions

What is shared is sequence organization, class-specific spatial/temporal aggregation, label blending, OAI usage and the overall training design. Not every parameter is identical. DINOv2 actually takes 392 inputs; the audited RadImageNet version takes 224 inputs and 24 grayscale tokens. Each backbone retains its normalization, pretraining and fine-tuning settings.

Ensembling extends this same approach: different backbones and label mixtures retain different inductive biases and borderline-case judgments on top of a common strong recipe. Their actual complementarity is assessed using the aligned OOF results in Section 7.

## 4. Actual image organization and model implementation

### 4.1 DICOM to cache

RSNA uses five slots: `SAG_FLUID_FS`, `COR_FLUID_FS`, `AX_FLUID_FS`, `COR_T1`, `SAG_T1`. Processing includes sequence selection, physical-position sorting, central approximately 80% sampling, a physical-size center crop, percentile intensity normalization, padding/resizing, uint8 caching and laterality canonicalization.

The 224 cache uses a 160 mm crop; the 384 cache uses a 130 mm crop. Both retain 24 slices per slot. Keep `study_ids.npy`, sequence masks and cache manifests together with the image arrays. Shards are aligned by ID, not by assumed row positions.

The current overlapping 2.5D path applies a three-slice window with stride two: starts `0,2,...,20`, producing **11 tokens** from 24 cached slices. The three channels are adjacent MRI slices, not RGB or copies of one slice. Earlier 224 runs used eight nonoverlapping triplets; the audited RadImageNet run uses 24 grayscale-copy tokens. Actual `unfold`/encoder construction is more reliable than stale comments or the `n_groups` field.

### 4.2 A shared encoder still preserves sequence differences

The five slots share one visual encoder to avoid redundant backbone modeling, followed by separate slot aggregation paths. Features are not simply globally averaged. Each target first applies spatial attention pooling, followed by a BiGRU and class-wise temporal attention across depth. Each slot predicts 12 targets, followed by class-wise sequence fusion.

In the ConvNeXt configuration, BiGRU hidden size is 256 and attention hidden size is 128. Within a slot, classes reuse that slot's BiGRU parameters, but their inputs already contain class-specific spatial features. The gate mainly learns per-class sequence weights and applies the per-study missing-slot mask; it should not be described as an unsupported, more elaborate dynamic router.

Alongside main BCE, an **0.20-weight expert auxiliary BCE** provides supervision to available slots. Missing slots are not encoded, do not contribute auxiliary BCE and are excluded from valid fusion. The historical ranking term is **0.1-weight within-batch pairwise logistic ranking**, grouping with `yb >= 0.5`. It has not been redesigned to handle every form of uncertain soft target.

### 4.3 Augmentation and EMA settings

These settings come from the audited ConvNeXt configuration. Other versions retain their own source settings.

| Component | Setting | Purpose/constraint |
| --- | --- | --- |
| Rotation | ±25°, probability 0.5 | Pose variation during training. |
| Perspective | Scale 0.05–0.10, probability 0.5 | Mild geometric variation. |
| Geometric consistency | One transform shared across the study's slots/slices | Avoid independently perturbing every slice. This does not imply spatial registration between independently acquired sequences. |
| Depth jitter | -1/0/+1 with probabilities 0.25/0.50/0.25 | Reduce dependence on fixed slice groups. |
| Sequence-order perturbation | Probability 0.5 | Depth-order augmentation in the original code, distinct from in-plane HFlip/VFlip. |
| Simulated low resolution | Scale 0.5–1.0, probability 0.15 | Cover differences in effective acquisition resolution. |
| Blur | Sigma 0.3–1.0, probability 0.1 | Mild imaging blur. |
| Bias field | Strength 0.25, grid 4, probability 0.2 | Slowly varying MRI intensity nonuniformity. |
| Noise | Sigma 2–3 on the 0–255 scale, probability 0.3 | Mild signal-to-noise variation. |
| Gamma | 0.7–1.4, probability 0.3 | Nonlinear contrast variation. |
| Brightness/contrast | Each ±0.25, probability 0.5 | Intensity appearance variation. |
| Intensity consistency | Sample per slot; share settings across its 24 slices | Preserve within-sequence continuity. |
| EMA | Decay 0.995, warmup 10 optimizer steps | Validation/checkpoint saving uses EMA. Warmup is not measured in micro-batches. |

Ordinary HFlip/VFlip that would change anatomical laterality is not used. bf16, channels-last and backbone chunking primarily improve feasibility and speed; they are not assigned unmeasured LB gains. The audited ConvNeXt configuration uses batch size 4, accumulation 4, AdamW LR 2e-4, weight decay 1e-2 and 12 epochs. DINO and other backbones retain their own freezing and LR choices.

## 5. Labels: the three methods in the user's notebook

The labeling description follows the three sections of `RSNA_Knee (2).ipynb`: **LLM prompt V1, LLM prompt V2, and the MRI montage plus report VLM workflow**. Their original prompts are in `code/labeling/rsna/`. This is not a substitution of another team's image-only labeling method.

### 5.1 LLM V1: structured soft labels from reports

V1 receives only the clinical report. It handles multilingual phrasing, negation scope, anatomical compartments, injury grades, cartilage degeneration and contusion/fracture distinctions. Each target returns `grade`, `stated` and a verbatim report `evidence` span.

The report is the evidence source. Unmentioned, explicitly normal and explicitly abnormal findings must be distinguished. Even conservative indirect inferences allowed by the prompt, such as for synovitis, still come from the report: the model has not seen the MRI.

**An LLM can faithfully align with a report and still conflict with image-derived labels.** Clinical reporting priorities and thresholds may differ from the competition's image-reading definitions. A report may omit a visible finding or emphasize a mild change below the competition threshold. Faithful report extraction cannot remove that supervision mismatch; not every discrepancy means the LLM misread the text.

### 5.2 LLM V2: from severity scores toward competition-positive probabilities

V2 retains the report-evidence framework but more explicitly asks for the probability that image experts would mark the target positive, rather than a severity rating alone.

It asks for two decimal places, continuous judgments rather than mechanically copied anchors, and greater care with marrow edema/reconversion descriptions that could be incorrectly mapped to Contusion. Blending the two report label sources was recorded as approximately `+0.001`.

This improves report-to-target interpretation without adding image evidence. The report/image information gap remains. These are experimental soft targets, not independently calibrated probabilities.

### 5.3 VLM: competition definitions, reports and images in one judgment

In the user's workflow, the RSNA annotation criteria supplied for the experiment are included in the VLM task. The model reviews the MRI montages and original report against those criteria rather than extracting keywords alone. Exact thresholds follow the saved notebook prompt; this handover does not introduce new definitions or independently reverify their official source.

The workflow preserves these constraints:

1. Select multiple sequences for each study and generate sampled montages in physical slice order, retaining sequence descriptions, orientation and slice/tile numbers.
2. Read the report and all supplied montages. Use the report to locate findings and also inspect findings it does not mention.
3. **When images and report disagree, image evidence takes priority**, because the target is based on image interpretation. The model is not required to ignore images to remain faithful to the report.
4. Judge whether each finding meets its competition-positive definition, and record concise image evidence and agreement/disagreement with the report.
5. Use independent task contexts, restrict inputs to the task and listed images, avoid old labels/validation results, save progress per case, then merge and validate.

The notebook emphasizes high-grade ligament injuries, surface-reaching meniscal tears or qualifying abnormal morphology, high-grade cartilage loss, and distinctions between small amounts of fluid/cysts and findings reaching the target threshold. **Detecting any abnormality is not the same as meeting the target's positive definition.**

Thus “aligning images and reports” means interpreting both under the same competition definitions, not forcing agreement. The VLM may retain report-supported findings while revising labels affected by omissions or disagreement with the images.

The notebook records a 58-Gold labeling comparison of approximately **0.903 for report-based labels → 0.921 for image-plus-report labels**. This is labeling macro AUC, not the final image model's LB, and not a newly rerun controlled comparison of every prompt. A 10-case repeat correlation of approximately 0.986 was also recorded as a small stability observation.

### 5.4 Why blend label sources?

Report labels retain information from the clinical narrative; VLM labels add image evidence and competition-definition alignment. Sampled montages can also omit slices or be unclear. Keeping both sources is useful without assuming either is always correct.

Here `VLM-A` is the original image-report label source and `VLM-B` is the alternate image-report label source. These neutral names do not change the historical source files or provenance records.

The old recipe is:

`L = 0.7 × V1 + 0.3 × V2`

`Y_old = 0.7 × VLM-A + 0.3 × L = 0.7 × VLM-A + 0.21 × V1 + 0.09 × V2`

The two new recipes are:

- **New Conv / C2:** `0.7 × VLM-B + 0.3 × L`.
- **New DINO / X1:** `0.5 × VLM-B + 0.5 × L`.

Official Gold targets override matching UIDs. Mixtures always align the same study and target columns. Artifact filenames may still contain `vlm07_llm03` after recipe changes; establish the actual recipe from code and CSV contents, not the filename.

VLM/LLM blending was recorded as approximately `+0.003`. Different label mixtures also provide ensemble diversity: models differ not only in architecture, but in supervision preferences for uncertain and borderline cases.

## 6. OAI: from hard-label mismatch to a soft-supervision pool

### 6.1 Why more data alone was insufficient

OAI scores describe location, type and grade. “Degeneration” or a cartilage grade cannot automatically be treated as competition positivity. When a hard mapping misaligns definitions, increasing OAI exposure can amplify the bias. Label improvements helped in the user's experiments, but unrestricted follow-up expansion, pretraining and distillation did not consistently provide further gains.

v3 combines same-visit MRI, original readings and the dictionary to estimate the probability of meeting the target definition, with a separate validity mask. It does not simply divide a grade by its maximum or apply a fixed hard cutoff. Intermediate probabilities express uncertainty; abstention is for insufficient evidence. Missing values are not automatic negatives.

### 6.2 Scope and training usage

The candidate pool contains **2,399 baseline examinations plus 6,250 follow-up examinations**, not 8,649 independent participants. Follow-ups from the same person are not newly independent subjects.

The seven OAI targets are Medial Meniscus, Lateral Meniscus, Medial OA, Lateral OA, PF OA, Effusion and Synovitis. Loading OAI does not provide ground-truth labels for the remaining five targets. Earlier four-class supervision used Lateral Meniscus, PF OA, Lateral OA and Synovitis. Four classes performed better at the 2,399 stage, whereas seven later performed better with the expanded pool; the conclusion depends on cohort and label version.

The three OAI sequences map to the first three RSNA slots: `SAG_IW_TSE → SAG_FLUID_FS`, `COR_IW_TSE → COR_FLUID_FS`, `AX_MPR → AX_FLUID_FS`; the remaining slots are masked. Reusing anatomical planes does not make the acquisition contrasts identical. AX_MPR is a DESS reconstruction, not an independent acquisition confirming a finding.

The common saved setup samples **3,305 RSNA + 2,393 OAI per epoch**, approximately 58/42, from the larger OAI pool. Some logs show 2,398 + 6,242 usable OAI examinations after cache/mask filtering; those counts do not replace the candidate counts. Align participant, side and visit between labels and caches.

The evidence supports better labels, controlled external exposure and a larger candidate pool—not a guarantee that more OAI is better. Downloading, dual-resolution caching, QC, sharding, task preparation, the v3 prompt and result validation are included in the handover.

## 7. Ensemble complementarity and what has been demonstrated

### 7.1 The original six suites

The six are ConvNeXt-Tiny, EfficientNetV2-S, DINOv2-S, DINOv3 ConvNeXt-Tiny, RadImageNet ResNet50 and TinyViT21M. The eight historical OOF sets used in the recorded analysis were aligned to the same 4,407 UIDs, four folds and Gold flags; they are external to this public release.

Against the same old-Conv reference labels, the best original individual suite has OOF approximately **0.95737**; the equal-weight original-six ensemble **A** reaches approximately **0.96078**, a gain of **0.00341**. This supports an ensemble benefit under one common reference. A scores approximately 0.94030 on the 58 Gold cases; this is small and may have influenced other-fold training/checkpoint selection, so it is not an independent weight-tuning blind test.

Average per-class prediction correlations are approximately 0.962–0.976 among the original main models and 0.944–0.949 between RadImageNet and the others. RadImageNet's individual old-reference score is lower, approximately 0.95131, but its different predictions can still complement the ensemble. Neither individual-score ranking nor lower correlation alone establishes the best subset; evaluate the actual blend against a common reference.

### 7.2 New-label diversity does not establish 70/30 as optimal

The grouped formulation is:

`A = mean(original six suite probabilities)`

`B = mean(new Conv probabilities, new DINO probabilities)`

`P = 0.7 × A + 0.3 × B`

Each original suite therefore receives `0.7/6`; each new suite receives `0.3/2 = 0.15`. This is not eight-way equal weighting. The recipes still predict the same 12 targets and can be blended after ID/class alignment; differing supervision is one source of diversity.

| Fixed blend | Old Conv reference | New Conv reference | New DINO reference | 58 Gold held-out OOF |
| --- | ---: | ---: | ---: | ---: |
| A: original six, equal weights | 0.96078 | 0.95834 | 0.94884 | 0.94030 |
| B: new two, equal weights | 0.95736 | 0.95730 | 0.94816 | 0.93219 |
| 0.7A + 0.3B | 0.96069 | 0.95885 | 0.94965 | 0.93878 |

**Only comparisons within a column use the same evaluation standard.** Adding B improves both new-reference scores but does not exceed A on the old reference or Gold. This is not evidence of a universal improvement. A and 70/30 remain separate experiments to test transfer to the real test distribution; OOF does not establish 70/30 as optimal.

### 7.3 Folds, TTA and between-suite blending

Each suite retains its input and normalization, aggregates fold/TTA logits according to its implementation and then applies sigmoid. Between suites, probabilities are averaged. Current lightweight TTA adds one +12° view to old Conv and new Conv: those two use 0°/+12°; the others use one view.

An earlier original-six version with Conv/Eff TTA completed three public test studies on Kaggle in approximately 5.32 minutes. This is neither the current eight-suite full-run runtime nor a reliable extrapolation to the private test set. The included OOF uses each study's held-out fold, not the same distribution as four-fold-plus-TTA test inference.

## 8. Essential files and reproduction

| Path | Purpose |
| --- | --- |
| `code/rsna/` | 224/384 cache generation. |
| `code/labeling/rsna/` | Original three-method prompts; montage, task and merge scripts. |
| `code/oai/` | Download checking, dual-resolution caches, sharding and v3 tasks/validation, including necessary parsing/metadata dependencies. |
| `code/train/` | Ten training entry points: the original five backbones, two label variants, and recovered224 ConvNeXt,384 graycopy24 ConvNeXt and DINOv3 training. |
| `code/inference/` | Standalone six/eight-suite inference with embedded model implementations. |
| `tools/` | Path adaptation, strict label merging and offline preflight checks. |
| `config/` | Example path mappings, external input inventory and model/source configuration. `labels/rsna/` includes seven reviewed frozen CSVs; OAI labels and `results/` remain external and ignored. |
| `our_pipeline.ipynb` | Essential-source reading/restoration notebook; source cells use `%%writefile`, not automatic training/scoring. |
| `manifest.json`, `REPRODUCIBILITY_AUDIT.md` | Source hashes, publication scope and reproducibility checks; the private historical `AUDIT_REPORT.json` is not included. |

Use the cache → labels → training → inference sequence. Historical failed branches and per-case agent/writer scripts remain in the full archive. `code/oai/generate_labels_v1.py` is retained for upstream reading provenance/metadata, not because current training should revert to its historical hard labels.

Our split has **4,407 studies and four folds**. The 58 Gold studies are held out in their assigned fold but may enter training/checkpoint selection in other folds; they are not a never-trained external test set. The DINOv3 training source has now been recovered from executed Kaggle Version3 (scriptVersionId358232793); its original source, inference and configuration are included; historical OOF is external. New training runs were not executed during recovery.

MRI, original scores, caches and weights must be supplied separately. Configure local/Kaggle paths and install dependencies. The five original OAI pixel-conversion functions are bundled with identical ASTs; their original workstation attachment dependencies now use package-local paths. These packaging changes do not alter model recipes.

## 9. Running and checking the package

See **[REPRODUCE.md](REPRODUCE.md)** for commands, input/output contracts, scoring JSONL formats, path configuration and external assets. See **[ENVIRONMENT.md](ENVIRONMENT.md)** for dependencies and known versions. Start with `python tools/preflight.py --report preflight.json`; then check dependencies and external inputs as appropriate.

The audit added baseline QC, cohort/queue generation source, strict label merging/Gold override and follow-up CSV export, plus a runtime path adapter that preserves network computation. Cohort manifests and S3 object lists are generated privately from authorized inputs, not distributed here. Per-file edits are recorded in the manifest. The notebook is for reading/restoring source; run the full workflow from the extracted ZIP root and keep configuration, dictionaries and cohort metadata together.

Documentation and notebook narrative are in English. Original operational prompts and code comments are preserved to avoid silently changing the experiment. An English reading translation of the OAI prompt is provided in `docs/OAI_PROMPT_ENGLISH_REFERENCE.md`; the original remains the reproduction reference.

## Acknowledgements and Data Access

### Osteoarthritis Initiative (OAI)

We used knee MRI and expert readings from the Osteoarthritis Initiative (OAI) for external supervision. This branch contains the multi-backbone path, which uses NDA Collection 2343; NDA Study S3430; DOI: [10.15154/vgnc-0x45](https://doi.org/10.15154/vgnc-0x45). The team's CoAtNet path is on the [`coatnet-pipeline`](https://github.com/Evan2001ok/rsna-knee-abnormality-detection/tree/coatnet-pipeline) branch of this repository and has its own NDA Study DOI: [10.15154/wfxg-g926](https://doi.org/10.15154/wfxg-g926).

> Data and/or research tools used in the preparation of this manuscript were obtained and analyzed from the controlled access datasets distributed from the Osteoarthritis Initiative (OAI), a data repository housed within the NIMH Data Archive (NDA). OAI is a collaborative informatics system created by the National Institute of Mental Health and the National Institute of Arthritis, Musculoskeletal and Skin Diseases (NIAMS) to provide a worldwide resource to quicken the pace of biomarker identification, scientific investigation and OA drug development. Dataset identifier(s): NDA Collection 2343; NDA Study S3430; DOI: 10.15154/vgnc-0x45 (multi-backbone branch); DOI: 10.15154/wfxg-g926 (CoAtNet branch).

The OAI acknowledgement follows the official NDA wording. It is the same text as Section 6.2 of our [10th-place solution write-up](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/writeups/10th-place-solution).

OAI images and participant-level data are not redistributed in this repository. Users must obtain access independently through the [official OAI portal](https://nda.nih.gov/oai/) and comply with the terms applicable to their access. See [External downloads and access](#external-downloads-and-access) and [REPRODUCE.md, Section 4](REPRODUCE.md#4-oai-existing-packages-or-optional-raw-data-rebuild).

### DINOv3 pretrained weights

The DINOv3 ConvNeXt-Tiny group (`dino3`) is initialized from [`timm/convnext_tiny.dinov3_lvd1689m`](https://huggingface.co/timm/convnext_tiny.dinov3_lvd1689m), which Meta distributes under the [DINOv3 License](https://ai.meta.com/resources/models-and-libraries/dinov3-license), not Apache-2.0 like the other timm initializations listed in [Initial backbone weights for training](#initial-backbone-weights-for-training). This repository does not include those weights; download them from the upstream source and accept its terms.

Any checkpoint fine-tuned from them is a derivative work of the DINO Materials. If such checkpoints are released, they will be distributed only under the DINOv3 License, with a copy of the license, and not under any other license that applies to this repository. As the license requires, we acknowledge the use of DINOv3 in our write-up and here.

## Source supplement: October25,2026

Read [AI_USAGE.md](AI_USAGE.md) for the module-by-module workflow. The supplement adds `code/train/conv224.py`, `conv_gray384.py` and `dino3.py`. The224 entry is exact source from Notebook4 cell19. The grayscale entry retains one complete program from cell8 after confirming the two copies have identical executable ASTs. DINOv3 is exact recovered training source from [Kaggle Version3](https://www.kaggle.com/code/lihu6666/dinov3-conv-lihu-of-fork-of-rsna-train-cd4e8a?scriptVersionId=358232793), not an inference-derived approximation.

Sources and hashes are in `config/training_source_provenance.json`. The historical six/eight-suite inference recipes are unchanged; their recorded OOF remains external. The final team0.967 pipeline remains outside this package. These additions have source/CLI/static checks, not a newly completed GPU training run.
