# RSNA Knee Abnormality Detection

Code for Team MEJAL's 10th-place solution to the Kaggle competition [RSNA Knee Abnormality Detection](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection). The team write-up covering both halves is the [10th-place solution](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/writeups/10th-place-solution).

The source code is on two branches, one per development path:

| Branch | Path |
|---|---|
| **[multibackbone-pipeline](https://github.com/Evan2001ok/rsna-knee-abnormality-detection/tree/multibackbone-pipeline)** | Multi-source soft supervision and the multi-backbone, multi-sequence models |
| **[coatnet-pipeline](https://github.com/Evan2001ok/rsna-knee-abnormality-detection/tree/coatnet-pipeline)** | Iterative label distillation and the CoAtNet-2 window-level MIL models |

## multibackbone-pipeline

The branch contains 224/384 MRI preprocessing, shared-encoder model training, RSNA LLM/VLM labeling prompts and tools, OAI v3 soft-label mapping and preprocessing, and historical six/eight-suite ensemble inference.

- [Method and experiment notes](https://github.com/Evan2001ok/rsna-knee-abnormality-detection/blob/multibackbone-pipeline/README.md)
- [Reproduction workflow](https://github.com/Evan2001ok/rsna-knee-abnormality-detection/blob/multibackbone-pipeline/REPRODUCE.md)
- [AI module usage guide](https://github.com/Evan2001ok/rsna-knee-abnormality-detection/blob/multibackbone-pipeline/AI_USAGE.md)
- [Public release scope and external assets](https://github.com/Evan2001ok/rsna-knee-abnormality-detection/blob/multibackbone-pipeline/PUBLIC_RELEASE.md)
- [Weights of the multi-backbone models](https://www.kaggle.com/datasets/lihu6666/rsna-knee-model-10th-multi-seed-team) (public Kaggle dataset with DINOv2-Small, EfficientNetV2-S, ConvNeXt-Tiny and DINOv3-ConvNeXt-Tiny checkpoints)

The branch holds source code and seven frozen RSNA study-level label files; their 58 expert-study rows hold placeholder values, not expert labels. MRI, participant/cohort manifests, OAI labels, splits, OOF predictions, credentials and trained weights are not distributed here (see the weights dataset above). The final combined team inference is outside this branch (see [Final team inference](#final-team-inference)). Historical scores in the documentation describe the recorded experiments, not a new independent reproduction.

## coatnet-pipeline

The branch contains the CoAtNet path: iterative label distillation with report labels, image-only readings and out-of-fold predictions, CoAtNet-2 window-level attention-MIL training, OAI baseline supervision, all-window fine-tuning, and the scripts that build the 50/30/20 team-blend notebook (multi-backbone models 50%, CoAtNet-2 models 30%, R7 20%; one of the team's two final picks, submission 57456685: 0.967 public / 0.956 private).

- [Branch overview and instructions](https://github.com/Evan2001ok/rsna-knee-abnormality-detection/blob/coatnet-pipeline/README.md)
- [Detailed write-up of the CoAtNet path](https://github.com/Evan2001ok/rsna-knee-abnormality-detection/blob/coatnet-pipeline/docs/WRITEUP.md)
- [Weights of the four CoAtNet-2 models](https://www.kaggle.com/datasets/max6296/knee-coatnet-consensus-all-ftall-w56-bundle) (public Kaggle dataset, CC BY-NC 4.0)

## Final team inference

Version 5 of LP's public Kaggle notebook produced the team's selected submission (57490913: 0.967 public / 0.957 private; 70% multi-backbone, 21% CoAtNet-2, 9% R7): [RSNA Knee 10th place inference-Final selected team, Version 5](https://www.kaggle.com/code/lihu6666/rsna-knee-10th-place-inference-final-selected-team?scriptVersionId=360860299). Later versions of that notebook are follow-up experiments (the latest adds a TinyViT branch), so use the Version 5 link. LP's GitHub repository [lihu-1/rsna-knee-abnormality-detection](https://github.com/lihu-1/rsna-knee-abnormality-detection) holds a copy of Version 5 with notes on its inputs. Version 5 reads the multi-backbone checkpoints from the teammates' private Kaggle datasets; checkpoints with the same file names are in the public weights dataset linked above.

## Licence

The code on both branches is released under CC BY-NC 4.0 (see [LICENSE](LICENSE)), as the competition rules require for winning solutions. Each weights dataset shows its licence on its Kaggle page. Third-party code, data and pretrained weights keep their own terms. For example, checkpoints fine-tuned from DINOv3 follow the [DINOv3 License](https://ai.meta.com/resources/models-and-libraries/dinov3-license) (see [DINOv3 pretrained weights](https://github.com/Evan2001ok/rsna-knee-abnormality-detection/blob/multibackbone-pipeline/README.md#dinov3-pretrained-weights)), and R7 is heliosli's public model, only linked here, not shared: its [notebook](https://www.kaggle.com/code/heliosli/0-957-latest-dataset-score-in-one-click) and [weights](https://www.kaggle.com/datasets/heliosli/rsna-knee-compact-r7cnxfsss).

## Thanks

Thank you to everyone on Team MEJAL (Evan zhu, LP, minmin andy, Jak ma and Max) for the long hours and hard work that went into this result: supervision and prompt design, model training and tuning, the CoAtNet path, inference speed-ups, the final ensemble, and the clean-up that made this public release possible. Each part of this solution was built together.
