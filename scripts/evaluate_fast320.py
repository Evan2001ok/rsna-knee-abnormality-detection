#!/usr/bin/env python3
"""Judge the arms of the fast-component plan (Oct 15, 2026: make our part of the team notebook faster without losing
what it adds to the team blend) by rules fixed before results.

  windows  pooled out-of-fold predictions of the five v7 + OAI 384 px folds with 112 / 56 / 28 prediction windows
           (artifacts/fast320/windows-test/<run>/validation-plain-w<N>.csv)
  plain    the five 320 px v7 + OAI folds (P) against the five 384 px folds, pooled, plus the team blend as information
  arms     R (pairwise ranking loss) / S (spatial attention inside each window) / K (packed labels: the v7 rule with a
           0.7 ours + 0.3 R7 model source) fold 0 against P fold 0 on fold 0's studies, alone and inside the team blend
           (theirs 0.5 + arm 0.3 + R7 0.2), and for K the no-R7 blend (theirs 0.5 + K 0.5)
  arms384  the 384 px arms D (0.7 v7 + 0.3 R7 labels), A (per-series loss) and F (per-series slice-order flip) against
           the 384 px v7 + OAI fold 0, with the submitted "theirs" (eff + dino + dino3) in the team blend; D's no-R7
           blend; experts for information
  fivefold384  one arm's five folds (--arm; H = v7 with the hybrid OAI labels) against the five 384 px v7 + OAI folds

v7 = our round-5 labels; OAI = the extra Osteoarthritis Initiative knees; R7 = heliosli's public model.
Judges: severity-aware report labels, report + GPT agreement labels, the teammates' labels cut at 0.5; paired bootstrap
(scripts/compare_validation.py).
Run: .venv/bin/python scripts/evaluate_fast320.py {windows|plain|arms|arms384|fivefold384}
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from compare_team_blend import R7, THEIR_SETS, THEIRS, read, their_frames, theirs_ensemble
from compare_validation import compare
from fit_team_member_weights import auc_columns

from rsna_knee_abnormality_detection.constants import STUDY_ID, TARGETS

OUT = Path("artifacts/fast320/evaluation")
JUDGES = {"severity_aware": Path("data/labels/baseline_severity/labels.csv"),
          "agreement": Path("data/labels/severity_gpt_agreement/labels.csv")}
BASE384 = "runs/coatnet_384_srcchoice7own_oai_fold"
PLAIN320 = "runs/coatnet_320_srcchoice7own_oai_fold"
ARMS = {"R": "runs/coatnet_320_v7oai_rank_fold0", "S": "runs/coatnet_320_v7oai_spatial_fold0",
        "K": "runs/coatnet_320_v7oai_packed_fold0"}
# 384 px arms trained overnight on Oct 15: judged against the 384 px v7 + OAI fold 0, with the team blend's "theirs"
# being the submitted part (equal mean of eff, dino, dino3).
ARMS384 = {"D": "runs/coatnet_384_v7oai_r7mix30_fold0", "A": "runs/coatnet_384_v7oai_seriesaux_fold0",
           "F": "runs/coatnet_384_v7oai_seriesflip_fold0"}
SUBMITTED_THEIRS = ("eff", "dino", "dino3")
EXPERT_LABELS = Path("data/labels/baseline/expert_labels.csv")


def their_labels_judge() -> Path:
    path = OUT / "judge_theirs_labels.csv"
    if not path.exists():
        found = sorted((THEIRS / THEIR_SETS["conv"]).glob("*/labels_used_vlm07_llm03.csv"))
        labels = pd.read_csv(found[0], dtype={STUDY_ID: str}).set_index(STUDY_ID)[TARGETS]
        write((labels >= 0.5).astype(float), path)
    return path


def write(frame: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = frame.clip(0, 1)
    frame.index.name = STUDY_ID
    frame.reset_index().to_csv(path, index=False)
    return path


def pooled(prefix: str, name: str = "validation-averaged-model-plain.csv") -> pd.DataFrame:
    frame = pd.concat([read(Path(f"{prefix}{k}") / name) for k in range(5)])
    if not frame.index.is_unique:
        raise ValueError(f"{prefix}: folds overlap")
    return frame.sort_index()


def judge_all(reference: Path, candidates: list[Path], tag: str, judges: dict[str, Path]) -> dict:
    result = {}
    for judge, labels in judges.items():
        report = compare(labels, reference, [str(c) for c in candidates], OUT / f"{tag}_{judge}.json")
        result[judge] = {"reference_auc": round(report["reference_auc"], 4), **{
            Path(c["predictions"]).stem: {"difference": round(c["auc_difference"], 4),
                                          "interval": [round(x, 4) for x in c["paired_95_percent_interval"]]}
            for c in report["comparisons"]}}
    return result


def team(arm: pd.DataFrame, ids, with_r7: bool = True, arm_share: float = 0.3, theirs_keys=None) -> pd.DataFrame:
    frames = their_frames()
    theirs = (theirs_ensemble(frames, ids) if theirs_keys is None
              else theirs_ensemble(frames, ids, tuple(theirs_keys), ())).rank(pct=True)
    r7 = pd.concat([read(p) for p in R7]).loc[ids].rank(pct=True)
    mine = arm.loc[ids].rank(pct=True)
    if with_r7:
        return 0.5 * theirs + arm_share * mine + (1 - 0.5 - arm_share) * r7
    return 0.5 * theirs + 0.5 * mine


def windows() -> dict:
    base = Path("artifacts/fast320/windows-test")
    runs = [f"coatnet_384_srcchoice7own_oai_fold{k}" for k in range(5)]
    original = pd.concat([read(Path("runs") / r / "validation-averaged-model-plain.csv") for r in runs]).sort_index()
    paths = {}
    for w in (112, 56, 28):
        frame = pd.concat([read(base / r / f"validation-plain-w{w}.csv") for r in runs]).sort_index()
        paths[w] = write(frame, OUT / "windows" / f"w{w}.csv")
        if w == 112:
            check = float((frame - original.loc[frame.index]).abs().max().max())
    result = judge_all(paths[112], [paths[56], paths[28]], "windows", JUDGES)
    result["w112_matches_saved_predictions_max_difference"] = round(check, 6)
    passing = [w for w, name in ((56, "w56"), (28, "w28"))
               if result["severity_aware"][name]["difference"] >= -0.001 and result["agreement"][name]["difference"] >= -0.0005]
    result["accepted_windows"] = min(passing) if passing else 112
    return result


def plain() -> dict:
    p384, p320 = pooled(BASE384), pooled(PLAIN320)
    a, b = write(p384, OUT / "plain" / "v7oai_384.csv"), write(p320.loc[p384.index], OUT / "plain" / "v7oai_320.csv")
    result = judge_all(a, [b], "plain", JUDGES)
    result["accepted"] = (result["severity_aware"]["v7oai_320"]["difference"] >= -0.003
                          and result["agreement"]["v7oai_320"]["difference"] >= -0.001)
    ids = p384.index
    t384, t320 = write(team(p384, ids), OUT / "plain" / "team_384.csv"), write(team(p320, ids), OUT / "plain" / "team_320.csv")
    result["team_blend_information"] = judge_all(t384, [t320], "plain_team", {**JUDGES, "theirs_labels": their_labels_judge()})
    return result


def arms() -> dict:
    p0 = read(Path(f"{PLAIN320}0") / "validation-averaged-model-plain.csv").sort_index()
    ids = p0.index
    ref = write(p0, OUT / "arms" / "P0.csv")
    ref_team = write(team(p0, ids), OUT / "arms" / "team_P0.csv")
    judges3 = {**JUDGES, "theirs_labels": their_labels_judge()}
    result = {"studies": len(ids)}
    for arm, run in ARMS.items():
        path = Path(run) / "validation-averaged-model-plain.csv"
        if not path.exists():
            result[arm] = "missing"
            continue
        frame = read(path).loc[ids]
        alone = judge_all(ref, [write(frame, OUT / "arms" / f"{arm}0.csv")], f"arm{arm}_alone", JUDGES)
        blended = judge_all(ref_team, [write(team(frame, ids), OUT / "arms" / f"team_{arm}0.csv")], f"arm{arm}_team", judges3)
        a, t = alone, blended
        rule_alone = (a["severity_aware"][f"{arm}0"]["difference"] >= 0.0015 and a["severity_aware"][f"{arm}0"]["interval"][0] > -0.001
                      and a["agreement"][f"{arm}0"]["difference"] >= -0.0005)
        rule_team = (t["severity_aware"][f"team_{arm}0"]["difference"] >= 0.001
                     and t["theirs_labels"][f"team_{arm}0"]["difference"] >= -0.0005)
        result[arm] = {"alone": alone, "team": blended, "adopt": bool(rule_alone or rule_team),
                       "by": [n for n, ok in (("alone", rule_alone), ("team", rule_team)) if ok]}
        if arm == "K":
            no_r7 = judge_all(ref_team, [write(team(frame, ids, with_r7=False), OUT / "arms" / "team_K0_noR7.csv")],
                              "armK_noR7", judges3)
            result["K"]["no_r7"] = no_r7
            result["K"]["replaces_r7"] = bool(no_r7["severity_aware"]["team_K0_noR7"]["difference"] >= -0.001
                                              and no_r7["theirs_labels"]["team_K0_noR7"]["difference"] >= -0.001)
    return result


def expert_auc(run: str) -> float | None:
    path = Path(run) / "expert58-plain.csv"
    if not path.exists():
        return None
    labels = pd.read_csv(EXPERT_LABELS, dtype={STUDY_ID: str}).set_index(STUDY_ID)[TARGETS]
    values = read(path).loc[labels.index]
    return round(auc_columns(labels.to_numpy(float), values.to_numpy(float)), 4)


def arms384() -> dict:
    ref = read(Path(f"{BASE384}0") / "validation-averaged-model-plain.csv").sort_index()
    ids = ref.index
    ref_path = write(ref, OUT / "arms384" / "ref0.csv")
    team_ref = write(team(ref, ids, theirs_keys=SUBMITTED_THEIRS), OUT / "arms384" / "team_ref0.csv")
    judges3 = {**JUDGES, "theirs_labels": their_labels_judge()}
    result = {"studies": len(ids), "theirs": list(SUBMITTED_THEIRS), "experts_reference": expert_auc(f"{BASE384}0")}
    for arm, run in ARMS384.items():
        path = Path(run) / "validation-averaged-model-plain.csv"
        if not path.exists():
            result[arm] = "missing"
            continue
        frame = read(path).loc[ids]
        alone = judge_all(ref_path, [write(frame, OUT / "arms384" / f"{arm}0.csv")], f"a384_{arm}_alone", JUDGES)
        blended = judge_all(team_ref, [write(team(frame, ids, theirs_keys=SUBMITTED_THEIRS), OUT / "arms384" / f"team_{arm}0.csv")],
                            f"a384_{arm}_team", judges3)
        a, t = alone, blended
        rule_alone = (a["severity_aware"][f"{arm}0"]["difference"] >= 0.0015 and a["severity_aware"][f"{arm}0"]["interval"][0] > -0.001
                      and a["agreement"][f"{arm}0"]["difference"] >= -0.0005)
        rule_team = (t["severity_aware"][f"team_{arm}0"]["difference"] >= 0.001
                     and t["theirs_labels"][f"team_{arm}0"]["difference"] >= -0.0005)
        detail = json.loads((OUT / f"a384_{arm}_alone_severity_aware.json").read_text())["comparisons"][0]
        result[arm] = {"alone": alone, "team": blended, "adopt": bool(rule_alone or rule_team),
                       "by": [n for n, ok in (("alone", rule_alone), ("team", rule_team)) if ok],
                       "per_finding_severity_aware": {k: round(v, 4) for k, v in detail["per_finding_difference"].items()},
                       "experts": expert_auc(run)}
        if arm == "D":
            no_r7 = judge_all(team_ref, [write(team(frame, ids, with_r7=False, theirs_keys=SUBMITTED_THEIRS),
                                               OUT / "arms384" / "team_D0_noR7.csv")], "a384_D_noR7", judges3)
            result["D"]["no_r7"] = no_r7
            result["D"]["replaces_r7"] = bool(no_r7["severity_aware"]["team_D0_noR7"]["difference"] >= -0.001
                                              and no_r7["theirs_labels"]["team_D0_noR7"]["difference"] >= -0.001)
    return result


FIVE_FOLD = {"D": "runs/coatnet_384_v7oai_r7mix30_fold", "A": "runs/coatnet_384_v7oai_seriesaux_fold",
             "H": "runs/coatnet_384_v7oai_oaihybrid_fold", "F": "runs/coatnet_384_v7oai_seriesflip_fold"}


def expert_mean_auc(prefix: str) -> float | None:
    paths = [Path(f"{prefix}{k}") / "expert58-plain.csv" for k in range(5)]
    if not all(p.exists() for p in paths):
        return None
    labels = pd.read_csv(EXPERT_LABELS, dtype={STUDY_ID: str}).set_index(STUDY_ID)[TARGETS]
    mean = sum(read(p).loc[labels.index] for p in paths) / 5
    return round(auc_columns(labels.to_numpy(float), mean.to_numpy(float)), 4)


def fivefold384(arm: str) -> dict:
    """Five-fold gate (rule fixed before results): the arm's five folds pooled out of fold against the 384 px v7 + OAI
    five folds, alone and inside the team blend (submitted theirs)."""
    ref = pooled(BASE384)
    ids = ref.index
    cand = pooled(FIVE_FOLD[arm]).loc[ids]
    d = OUT / f"fivefold384_{arm}"
    alone = judge_all(write(ref, d / "ref.csv"), [write(cand, d / "arm.csv")], f"ff384_{arm}_alone", JUDGES)
    judges3 = {**JUDGES, "theirs_labels": their_labels_judge()}
    team_ref = write(team(ref, ids, theirs_keys=SUBMITTED_THEIRS), d / "team_ref.csv")
    blended = judge_all(team_ref, [write(team(cand, ids, theirs_keys=SUBMITTED_THEIRS), d / "team_arm.csv")],
                        f"ff384_{arm}_team", judges3)
    a = {j: alone[j]["arm"] for j in JUDGES}
    t = {j: blended[j]["team_arm"] for j in judges3}
    rule_team = (t["severity_aware"]["difference"] >= 0.001 and t["agreement"]["difference"] >= -0.0005
                 and t["theirs_labels"]["difference"] >= -0.0005)
    rule_alone = (a["severity_aware"]["difference"] >= 0.0015 and a["severity_aware"]["interval"][0] > 0
                  and a["agreement"]["difference"] >= -0.0005)
    detail = json.loads((OUT / f"ff384_{arm}_alone_severity_aware.json").read_text())["comparisons"][0]
    result = {"arm": arm, "studies": len(ids), "alone": a, "team": t, "pass": bool(rule_team or rule_alone),
              "by": [n for n, ok in (("team", rule_team), ("alone", rule_alone)) if ok],
              "per_finding_severity_aware": {k: round(v, 4) for k, v in detail["per_finding_difference"].items()},
              "experts_five_fold_mean": {"arm": expert_mean_auc(FIVE_FOLD[arm]), "reference": expert_mean_auc(BASE384)}}
    if arm == "D":
        no_r7 = judge_all(team_ref, [write(team(cand, ids, with_r7=False, theirs_keys=SUBMITTED_THEIRS), d / "team_noR7.csv")],
                          f"ff384_{arm}_noR7", judges3)
        result["no_r7"] = {j: no_r7[j]["team_noR7"] for j in judges3}
        result["replaces_r7"] = bool(result["no_r7"]["severity_aware"]["difference"] >= -0.001
                                     and result["no_r7"]["theirs_labels"]["difference"] >= -0.001)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("part", choices=["windows", "plain", "arms", "arms384", "fivefold384"])
    parser.add_argument("--arm", choices=sorted(FIVE_FOLD), help="fivefold384 only")
    args = parser.parse_args()
    if args.part == "fivefold384":
        if args.arm is None:
            parser.error("fivefold384 needs --arm")
        result = fivefold384(args.arm)
        name = f"fivefold384_{args.arm}"
    else:
        result = {"windows": windows, "plain": plain, "arms": arms, "arms384": arms384}[args.part]()
        name = args.part
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{name}.json").write_text(json.dumps(result, indent=1) + "\n")
    print(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
