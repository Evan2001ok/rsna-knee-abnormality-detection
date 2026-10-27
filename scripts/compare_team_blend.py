#!/usr/bin/env python3
"""Offline blend test for the merged team (Oct 15, 2026; rules fixed before results): the teammates' ensemble, our
v7 + OAI folds (five CoAtNet folds trained on v7, our round-5 labels, plus the OAI knees) and R7 (heliosli's public
model), all as out-of-fold predictions on the 4,296 cross-validation studies, blended by per-finding percentile ranks as
on Kaggle, and judged on three label sets (severity-aware report labels, report + GPT agreement labels, the teammates'
labels cut at 0.5) with paired bootstrap intervals (scripts/compare_validation.py), plus the 58 expert studies as
information.

Their files are downloaded data: read as CSV only, nothing from them is executed. The teammates' and R7's out-of-fold
predictions are not public.
Run: .venv/bin/python scripts/compare_team_blend.py [--out artifacts/team-blend]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from compare_validation import compare

from rsna_knee_abnormality_detection.constants import STUDY_ID, TARGETS
from rsna_knee_abnormality_detection.metrics import score_predictions

THEIRS = Path("artifacts/team-models-20261015")
THEIR_SETS = {  # key in their notebook -> downloaded dataset folder
    "conv": "minminandy__distillstage1conv384", "eff": "minminandy__distillstage1effs384",
    "dino": "zhuyifanss__distillstage1dinov2s384", "rad": "zhuyifanss__distillstage1radimageresnet224",
    "tiny": "minminandy__distillstage1tinyvit384", "dino3": "lihu6666__dinov3-convnext-tiny-lihu-of-fork-of-rsna-cd4e8a",
    "conv_new": "zhuyifanss__distillstage1conv-difflabel384", "dino_new": "zhuyifanss__distillstage1dino-difflabel394",
}
GROUP_A = ("conv", "eff", "dino", "dino3", "rad", "tiny")  # as their notebook (dino3 shared by a teammate on Oct 15)
GROUP_B = ("conv_new", "dino_new")
OURS = Path("artifacts/ours/oof/v7_oai_336.csv")
OURS_EXPERT = Path("artifacts/ours/expert58/v7_oai_336_fold_mean.csv")
R7 = [Path(f"artifacts/s6r7-results/r7_oof_f{k}/predictions.csv") for k in range(4)]
R7_EXPERT = Path("artifacts/s6r7-results/r7_gold/predictions.csv")
SPLITS = Path("data/splits/baseline/splits.csv")
EXPERT_LABELS = Path("data/labels/baseline/expert_labels.csv")
JUDGES = {"severity_aware": Path("data/labels/baseline_severity/labels.csv"),
          "agreement": Path("data/labels/severity_gpt_agreement/labels.csv")}


def read(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype={STUDY_ID: str})
    if frame[STUDY_ID].duplicated().any():
        raise ValueError(f"duplicate studies in {path}")
    values = frame[TARGETS].to_numpy(float)
    if not np.isfinite(values).all() or (values < 0).any() or (values > 1).any():
        raise ValueError(f"predictions must be probabilities: {path}")
    return frame.set_index(STUDY_ID)[TARGETS]


def their_frames() -> dict[str, pd.DataFrame]:
    frames = {}
    for key, folder in THEIR_SETS.items():
        found = sorted((THEIRS / folder).glob("*/oof_predictions.csv"))
        if len(found) != 1:
            raise ValueError(f"{key}: expected one oof_predictions.csv, found {found}")
        frames[key] = read(found[0])
    return frames


def theirs_ensemble(frames: dict[str, pd.DataFrame], ids, a_keys=GROUP_A, b_keys=GROUP_B) -> pd.DataFrame:
    """Their notebook's rule: mean probability inside each group, then 0.7 A + 0.3 B (or one group alone)."""
    parts = []
    if a_keys:
        parts.append((0.7 if b_keys else 1.0, sum(frames[k].loc[ids] for k in a_keys) / len(a_keys)))
    if b_keys:
        parts.append((0.3 if a_keys else 1.0, sum(frames[k].loc[ids] for k in b_keys) / len(b_keys)))
    return sum(w * p for w, p in parts)


def ranks(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.rank(method="average", pct=True)


def blend(members: list[tuple[float, pd.DataFrame]]) -> pd.DataFrame:
    total = sum(w for w, _ in members)
    return sum(w * ranks(f) for w, f in members) / total


def write(frame: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.clip(0, 1).reset_index().rename(columns={"index": STUDY_ID}).to_csv(path, index=False)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("artifacts/team-blend"))
    parser.add_argument("--samples", type=int, default=2000)
    args = parser.parse_args()
    splits = pd.read_csv(SPLITS, dtype={STUDY_ID: str}).set_index(STUDY_ID)["fold"]
    frames = their_frames()
    ours = read(OURS)
    r7 = pd.concat([read(p) for p in R7])
    if r7.index.duplicated().any():
        raise ValueError("R7 out-of-fold studies overlap")
    cv = splits.index[splits >= 0]
    ids = cv.intersection(ours.index).intersection(r7.index)
    for f in frames.values():
        ids = ids.intersection(f.index)
    ids = ids.sort_values()
    if len(ids) != 4296:
        raise ValueError(f"expected the 4,296 cross-validation studies, got {len(ids)}")

    def candidates(t, o, r, f):
        c = {"theirs": blend([(1, t)]), "ours": blend([(1, o)]), "r7": blend([(1, r)]),
             "ours70_r7_30": blend([(0.7, o), (0.3, r)])}
        for w in (0.1, 0.2, 0.3, 0.4, 0.5):
            c[f"theirs_ours{int(w * 100)}"] = blend([(1 - w, t), (w, o)])
        for w in (0.1, 0.2, 0.3):
            c[f"theirs_r7_{int(w * 100)}"] = blend([(1 - w, t), (w, r)])
        for a, b in ((0.3, 0.1), (0.3, 0.2), (0.2, 0.1), (0.2, 0.2), (0.4, 0.1)):
            c[f"theirs{int(round((1 - a - b) * 100))}_ours{int(a * 100)}_r7_{int(b * 100)}"] = blend([(1 - a - b, t), (a, o), (b, r)])
        for k in (*GROUP_A, *GROUP_B):  # leave one of their sets out (information for Kaggle's 9-hour run-time limit)
            t_k = theirs_ensemble(f, t.index, tuple(x for x in GROUP_A if x != k), tuple(x for x in GROUP_B if x != k))
            c[f"theirs_without_{k}"] = blend([(1, t_k)])
            c[f"theirs_without_{k}_ours30"] = blend([(0.7, t_k), (0.3, o)])
        return c

    theirs = theirs_ensemble(frames, ids)
    cands = candidates(theirs, ours.loc[ids], r7.loc[ids], frames)
    paths = {name: write(frame, args.out / "cv" / f"{name}.csv") for name, frame in cands.items()}

    # Their training labels, cut at 0.5, as the judge that leans their way.
    found = sorted((THEIRS / THEIR_SETS["conv"]).glob("*/labels_used_vlm07_llm03.csv"))
    their_labels = pd.read_csv(found[0], dtype={STUDY_ID: str}).set_index(STUDY_ID)[TARGETS]
    if not their_labels.index.is_unique or not set(ids) <= set(their_labels.index):
        raise ValueError("their labels must cover every study once")
    judges = dict(JUDGES)
    judges["theirs_labels"] = write((their_labels.loc[ids] >= 0.5).astype(float), args.out / "judge_theirs_labels.csv")

    summary = {"studies": len(ids), "judges": {}, "experts": {}}
    for judge, labels in judges.items():
        summary["judges"][judge] = {}
        for ref in ("theirs", "theirs_ours30"):
            others = [str(p) for n, p in paths.items() if n != ref]
            report = compare(labels, paths[ref], others, args.out / f"compare_{judge}_vs_{ref}.json", args.samples)
            summary["judges"][judge][ref] = {
                "reference_auc": round(report["reference_auc"], 4),
                "candidates": {Path(c["predictions"]).stem: {
                    "auc": round(c["auc"], 4), "difference": round(c["auc_difference"], 4),
                    "interval": [round(x, 4) for x in c["paired_95_percent_interval"]]} for c in report["comparisons"]}}

    # The 58 expert studies (information only). Ours and R7's predictions there are held out; the teammates' models
    # trained on these studies, so their scores here are flattered.
    expert = pd.read_csv(EXPERT_LABELS, dtype={STUDY_ID: str}).set_index(STUDY_ID)[TARGETS]
    eids = expert.index.sort_values()
    their_gold = {k: f.loc[eids] for k, f in frames.items()}
    e_cands = candidates(theirs_ensemble(their_gold, eids), read(OURS_EXPERT).loc[eids], read(R7_EXPERT).loc[eids],
                         their_gold)
    for name, frame in e_cands.items():
        summary["experts"][name] = round(score_predictions(expert.loc[eids].to_numpy(), frame.loc[eids].clip(0, 1).to_numpy())["macro_auc"], 4)
    (args.out / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    print(json.dumps({"studies": len(ids), "written": str(args.out / "summary.json")}))


if __name__ == "__main__":
    main()
