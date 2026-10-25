# RSNA Knee Abnormality Detection

Our source code is maintained on the **[codex/our-pipeline](https://github.com/Evan2001ok/rsna-knee-abnormality-detection/tree/codex/our-pipeline)** branch.

The branch contains 224/384 MRI preprocessing, shared-encoder model training, RSNA LLM/VLM labeling prompts and tools, OAI v3 soft-label mapping and preprocessing, and historical six/eight-suite ensemble inference.

- [Method and experiment notes](https://github.com/Evan2001ok/rsna-knee-abnormality-detection/blob/codex/our-pipeline/README.md)
- [Reproduction workflow](https://github.com/Evan2001ok/rsna-knee-abnormality-detection/blob/codex/our-pipeline/REPRODUCE.md)
- [AI module usage guide](https://github.com/Evan2001ok/rsna-knee-abnormality-detection/blob/codex/our-pipeline/AI_USAGE.md)
- [Public release scope and external assets](https://github.com/Evan2001ok/rsna-knee-abnormality-detection/blob/codex/our-pipeline/PUBLIC_RELEASE.md)

This is a source-only release. MRI, participant/cohort manifests, patient/study label tables, OOF predictions, trained weights and credentials are not distributed here. The final combined team inference is outside this release. Historical scores in the documentation describe the recorded experiments, not a new independent reproduction.
