#!/usr/bin/env python3
"""Make severity-aware validation labels from the Pilkwang report readings.

Pilkwang readings are Kaggle user pilkwang's public report labels (report_labels_v2.csv, saved by
fetch_public_labels.py). Each finding has a verdict (YES, NO, UNK) and a score; for YES the score
0.68, 0.82 or 0.94 marks severity 1, 2 or 3.
The host's definitions (forum topic 733343) count only moderate or large, or high-grade, findings
for ACL, MCL, the three osteoarthritis compartments, effusion and Baker's cyst. The baseline
validation labels (data/labels/baseline/labels.csv, from `knee labels`) mark every mention as
present. Here, for those seven findings, a report that asserts the finding only as mild (Pilkwang
severity 1: "minimal, trace, small, mild, slight, grade 1") becomes absent. Everything else,
including expert and missing rows, is unchanged. The output is for checking and choosing
checkpoints, not for training.

Run: uv run python scripts/prepare_severity_validation_labels.py
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from rsna_knee_abnormality_detection.constants import STUDY_ID, TARGETS
from rsna_knee_abnormality_detection.labels import file_record, read_study_csv

THRESHOLD_FINDINGS = ["ACL", "MCL", "Medial OA", "Lateral OA", "PF OA", "Effusion", "Baker's"]
SEVERITY_SCORES = {0.68: 1, 0.82: 2, 0.94: 3}
VERDICT_VALUES = {"YES": 1.0, "NO": 0.0}


def prepare_severity_labels(baseline_csv: Path, pilkwang_csv: Path, output_dir: Path) -> Path:
    baseline_csv, pilkwang_csv, output_dir = map(Path, (baseline_csv, pilkwang_csv, output_dir))
    if output_dir.resolve() in {baseline_csv.parent.resolve(), pilkwang_csv.parent.resolve()}:
        raise ValueError("The output directory must not hold an input")
    baseline = read_study_csv(baseline_csv)
    pilkwang = read_study_csv(pilkwang_csv).set_index(STUDY_ID)
    report = baseline["label_source"].eq("report").to_numpy()
    report_ids = baseline.loc[report, STUDY_ID]
    if not set(report_ids) <= set(pilkwang.index):
        raise ValueError("Every report study needs a Pilkwang reading")
    labels = baseline.copy()
    changes = {}
    for name in TARGETS:
        verdict = pilkwang.loc[report_ids, f"{name}__verdict"].to_numpy()
        expected = np.array([VERDICT_VALUES.get(value, np.nan) for value in verdict])
        current = baseline.loc[report, name].to_numpy(dtype=float)
        if not np.array_equal(np.isnan(expected), np.isnan(current)) or not np.allclose(
                expected[~np.isnan(expected)], current[~np.isnan(current)]):
            raise ValueError(f"{name}: baseline labels do not match the Pilkwang verdicts")
        scores = pilkwang.loc[report_ids, name].round(2).to_numpy()
        yes = verdict == "YES"
        if not np.isin(scores[yes], list(SEVERITY_SCORES)).all():
            raise ValueError(f"{name}: unexpected Pilkwang score for a present finding")
        if name in THRESHOLD_FINDINGS:
            mild = yes & (scores == 0.68)
            updated = np.where(mild, 0.0, current)
            labels.loc[report, name] = updated
            changes[name] = {"mild_mentions_set_absent": int(mild.sum()), "present_before": int(yes.sum()),
                             "present_after": int((updated == 1).sum())}
    if not labels.loc[~report].equals(baseline.loc[~report]):
        raise AssertionError("Expert and missing rows must stay unchanged")
    untouched = [name for name in TARGETS if name not in THRESHOLD_FINDINGS]
    if not labels[untouched].equals(baseline[untouched]):
        raise AssertionError("Findings without a severity threshold must stay unchanged")
    inputs = {str(path): file_record(path) for path in (baseline_csv, pilkwang_csv)}
    output_dir.mkdir(parents=True, exist_ok=True)
    labels_path = output_dir / "labels.csv"
    labels.to_csv(labels_path, index=False)
    (output_dir / "label_provenance.json").write_text(json.dumps({
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "Validation labels that follow the host's severity thresholds; not for training.",
        "rule": ("For ACL, MCL, Medial OA, Lateral OA, PF OA, Effusion and Baker's, a report finding the Pilkwang reader "
                 "graded severity 1 (minimal, trace, small, mild, slight, grade 1) is absent; severity 2 (stated without "
                 "qualification, moderate) and 3 stay present. Other findings and expert/missing rows are unchanged."),
        "definitions_source": "https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/733343",
        "inputs": inputs,
        "changes": changes,
        "check_on_expert_studies": ("Agreement with expert labels on the 58 studies for the seven findings: "
                                    "78.7% with every mention present, 83.6% with this rule (diagnostic only)."),
        "limitations": [
            "Severity comes from one reader's wording-based grade; reports often do not grade findings.",
            "Unqualified mentions count as present, although some may be mild on the images.",
            "Contusion and effusion remain the least reliable report-derived targets.",
        ],
        "labels_csv": file_record(labels_path),
        "reproduction": "uv run python scripts/prepare_severity_validation_labels.py",
    }, indent=2) + "\n")
    return labels_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-labels", type=Path, default=Path("data/labels/baseline/labels.csv"))
    parser.add_argument("--pilkwang", type=Path, default=Path("data/labels/public/pilkwang/report_labels_v2.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/labels/baseline_severity"))
    args = parser.parse_args()
    path = prepare_severity_labels(args.baseline_labels, args.pilkwang, args.output_dir)
    print(json.dumps(json.loads((path.parent / "label_provenance.json").read_text())["changes"]))


if __name__ == "__main__":
    main()
