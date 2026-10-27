# RSNA Knee Abnormality Detection

Code for Team MEJAL's 10th-place solution to the Kaggle competition [RSNA Knee Abnormality Detection](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection). The team write-up covering both halves is the [10th-place solution](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/writeups/10th-place-solution).

The source code is on two branches, one per development path:

| Branch | Path |
|---|---|
| **[codex/our-pipeline](https://github.com/Evan2001ok/rsna-knee-abnormality-detection/tree/codex/our-pipeline)** | Multi-source soft supervision and the multi-backbone, multi-sequence models |
| **[coatnet-pipeline](https://github.com/Evan2001ok/rsna-knee-abnormality-detection/tree/coatnet-pipeline)** | Iterative label distillation and the CoAtNet-2 window-level MIL models |

## codex/our-pipeline

The branch contains 224/384 MRI preprocessing, shared-encoder model training, RSNA LLM/VLM labeling prompts and tools, OAI v3 soft-label mapping and preprocessing, and historical six/eight-suite ensemble inference.

- [Method and experiment notes](https://github.com/Evan2001ok/rsna-knee-abnormality-detection/blob/codex/our-pipeline/README.md)
- [Reproduction workflow](https://github.com/Evan2001ok/rsna-knee-abnormality-detection/blob/codex/our-pipeline/REPRODUCE.md)
- [AI module usage guide](https://github.com/Evan2001ok/rsna-knee-abnormality-detection/blob/codex/our-pipeline/AI_USAGE.md)
- [Public release scope and external assets](https://github.com/Evan2001ok/rsna-knee-abnormality-detection/blob/codex/our-pipeline/PUBLIC_RELEASE.md)

This is a source-only release. MRI, participant/cohort manifests, patient/study label tables, OOF predictions, trained weights and credentials are not distributed here. The final combined team inference is outside this branch. Historical scores in the documentation describe the recorded experiments, not a new independent reproduction.

## coatnet-pipeline

The branch contains the CoAtNet path: iterative label distillation with report labels, image-only readings and out-of-fold predictions, CoAtNet-2 window-level attention-MIL training, OAI baseline supervision, all-window fine-tuning, and the scripts that build the team blend notebook.

- [Branch overview and instructions](https://github.com/Evan2001ok/rsna-knee-abnormality-detection/blob/coatnet-pipeline/README.md)
- [Detailed write-up of the CoAtNet path](https://github.com/Evan2001ok/rsna-knee-abnormality-detection/blob/coatnet-pipeline/docs/WRITEUP.md)
