"""Reproducible study/patient folds with expert studies reserved before training."""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold

from .constants import STUDY_ID, TARGETS
from .labels import file_record, read_study_csv


def make_splits(
    labels_csv: Path,
    output_dir: Path,
    n_folds: int = 5,
    seed: int = 42,
    groups_csv: Path | None = None,
) -> Path:
    """Assign folds without placing any expert patient's studies in training.

    The optional groups CSV must map every study exactly once to a trusted
    ``group_id`` (for example, a patient identifier). Without it the study is the
    grouping unit; repeated patients then cannot be ruled out.

    ``fold == -1`` marks the expert holdout, including other studies in the same
    patient group. ``fold == -2`` marks groups with no labeled studies. Every
    member of a report-labeled patient group receives the same nonnegative fold,
    including unknown studies, which retain ``split == 'unlabeled'`` and stay
    out of the supervised loss. Selecting fold k for validation means training
    only on its other folds. Soft labels are thresholded ONLY for balancing folds.

    If image teachers are added later, their entire fitting process must exclude
    the evaluation groups. A global out-of-fold target file alone does not make
    a later student split independent of its validation patients.
    """
    if not isinstance(n_folds, int) or isinstance(n_folds, bool) or n_folds < 2:
        raise ValueError("n_folds must be an integer of at least 2")
    labels_csv, output_dir = Path(labels_csv), Path(output_dir)
    labels = read_study_csv(labels_csv).sort_values(STUDY_ID).reset_index(drop=True)
    if "label_source" not in labels:
        raise ValueError("labels.csv must include label_source; run the labels command first")
    if labels["label_source"].isna().any() or not labels["label_source"].isin(["expert", "report", "missing"]).all():
        raise ValueError("label_source must be expert, report, or missing")
    has_targets = labels[TARGETS].notna().any(axis=1)
    if ((labels["label_source"] == "missing") != ~has_targets).any():
        raise ValueError("label_source disagrees with missing target values")

    notes = []
    if groups_csv is None:
        labels["group_id"] = labels[STUDY_ID]
        notes.append("No patient mapping supplied: folds are separated by study; repeated patients cannot be ruled out.")
    else:
        groups_csv = Path(groups_csv)
        groups = read_study_csv(groups_csv, require_targets=False)
        if "group_id" not in groups:
            raise ValueError("The groups CSV must have StudyInstanceUID and group_id columns")
        group_values = groups["group_id"]
        if group_values.isna().any() or group_values.str.strip().eq("").any():
            raise ValueError("group_id cannot be blank")
        if group_values.ne(group_values.str.strip()).any():
            raise ValueError("group_id contains leading or trailing whitespace")
        if set(groups[STUDY_ID]) != set(labels[STUDY_ID]):
            raise ValueError("The groups CSV must contain exactly the same study IDs as labels.csv")
        labels["group_id"] = labels[STUDY_ID].map(groups.set_index(STUDY_ID)["group_id"])

    holdout_groups = set(labels.loc[labels["label_source"] == "expert", "group_id"])
    holdout = labels["group_id"].isin(holdout_groups)
    eligible = ~holdout & has_targets
    training = labels.loc[eligible].copy()
    training_groups = sorted(training["group_id"].unique())
    if len(training_groups) < n_folds:
        raise ValueError(
            f"Only {len(training_groups)} labeled groups remain after reserving the expert holdout; "
            f"cannot make {n_folds} folds. Import report labels or use fewer folds."
        )

    # Balance both positive and negative target coverage at group level. A group
    # may contain several studies or both findings over time; it is still kept
    # entirely in one fold. Missing targets are neither positive nor negative.
    positive = (training[TARGETS] >= 0.5) & training[TARGETS].notna()
    negative = (training[TARGETS] < 0.5) & training[TARGETS].notna()
    features = pd.concat([positive.add_suffix("_positive"), negative.add_suffix("_negative")], axis=1)
    features["group_id"] = training["group_id"]
    group_features = features.groupby("group_id").max().reindex(training_groups)
    fold_by_group = {}
    stratifier = MultilabelStratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    for fold, (_, valid_indices) in enumerate(stratifier.split(np.zeros(len(training_groups)), group_features.astype(int).to_numpy())):
        for index in valid_indices:
            fold_by_group[training_groups[index]] = fold

    splits = labels[[STUDY_ID, "group_id"]].copy()
    splits["fold"] = -2
    splits["split"] = "unlabeled"
    splits.loc[holdout, ["fold", "split"]] = [-1, "expert_holdout"]
    # The fold belongs to the entire patient, even when some of that patient's
    # studies have no known targets. This also keeps later self-training from
    # accidentally assigning an unknown groupmate to a different fold.
    assigned_group = labels["group_id"].isin(fold_by_group)
    splits.loc[assigned_group, "fold"] = labels.loc[assigned_group, "group_id"].map(fold_by_group).astype(int)
    splits.loc[eligible, "split"] = "train"
    splits["fold"] = splits["fold"].astype(int)
    splits = splits[[STUDY_ID, "fold", "split", "group_id"]]

    counts = []
    for fold in [-1, *range(n_folds)]:
        values = labels.loc[splits["fold"] == fold, TARGETS]
        missing_classes = []
        for target in TARGETS:
            series = values[target]
            positives, negatives = int((series == 1).sum()), int((series == 0).sum())
            soft = int((series.notna() & ~series.isin([0, 1])).sum())
            counts.append({
                "fold": fold,
                "target": target,
                "studies": len(values),
                "known": int(series.notna().sum()),
                "missing": int(series.isna().sum()),
                "binary_positive": positives,
                "binary_negative": negatives,
                "soft": soft,
                "positive_at_0_5": int((series >= 0.5).sum()),
                "negative_at_0_5": int((series < 0.5).sum()),
            })
            if positives == 0 or negatives == 0:
                missing_classes.append(target)
        if missing_classes:
            notes.append(f"Fold {fold} lacks both binary classes for: {', '.join(missing_classes)}. ROC AUC may be unavailable.")
    for note in notes:
        warnings.warn(note, UserWarning, stacklevel=2)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "splits.csv"
    splits.to_csv(output_path, index=False)
    pd.DataFrame(counts).to_csv(output_dir / "fold_target_counts.csv", index=False)
    summary = {
        "seed": seed,
        "n_folds": n_folds,
        "labels_csv": file_record(labels_csv),
        "groups_csv": file_record(groups_csv) if groups_csv is not None else None,
        "grouping": "provided group_id" if groups_csv is not None else "StudyInstanceUID",
        "expert_studies": int((labels["label_source"] == "expert").sum()),
        "reserved_expert_group_studies": int(holdout.sum()),
        "train_studies": int(eligible.sum()),
        "train_groups": len(training_groups),
        "unlabeled_studies": int((splits["split"] == "unlabeled").sum()),
        "unlabeled_studies_with_assigned_fold": int(((splits["split"] == "unlabeled") & (splits["fold"] >= 0)).sum()),
        "fold_study_counts": {str(key): int(value) for key, value in splits["fold"].value_counts().sort_index().items()},
        "warnings": notes,
        "policy": {
            "expert_groups_reserved_before_folding": True,
            "unknown_groupmates": "same fold as labeled groupmates; split remains unlabeled and excluded from supervised loss",
            "soft_labels": "thresholded at 0.5 only to balance groups, never changed in labels.csv",
            "teacher_student_validation": "all teacher and student fitting must exclude the final evaluation groups",
        },
        "splits_csv": file_record(output_path),
    }
    (output_dir / "split_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return output_path
