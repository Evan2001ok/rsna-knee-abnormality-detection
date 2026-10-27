#!/usr/bin/env python3
"""Replay stage 4 of the team notebook (the blend) on out-of-fold predictions (Oct 17, 2026; rules fixed beforehand).

The notebook's own blend cells, exactly as scripts/build_team_combined_notebook.py embeds them (BLEND = the plain
rank blend used until then, BLEND_NOISYOR = the noisy-OR + sibling correction + insurance; insurance = a study one part
failed on takes the other parts' blend), are run on files laid out the way Kaggle would see them, on each of the 10
held-out halves of the 4,296 cross-validation studies (refit_blend_compare SEEDS x both directions; nothing is fitted,
the settings are fixed). Parts: theirs = mean of dino, eff, dino3, newconv probabilities (their rule; newconv is a
teammate's newer ConvNeXt-tiny set); ours = consensus seed 42; R7. Scored with the three judges (severity-aware,
agreement, their labels); the expert ("gold") rows are left out when loading.

Also: damage scenarios for the insurance (one part fails on some studies, flagged the way the notebook would see it), R7
not scoring 10% / 30% with and without the 3% rule (information), and the cell on the real files of a 3-study Kaggle
check run of the notebook.

Run: CUDA_VISIBLE_DEVICES="" PYTHONPATH=scripts nice .venv/bin/python scripts/replay_team_blend_postproc.py
"""

from __future__ import annotations

import json
import tempfile
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from build_team_combined_notebook import BLEND, BLEND_NOISYOR
from refit_blend_compare import SEEDS, macro_auc
from scipy.stats import norm, rankdata
from search_team_blend import load

from rsna_knee_abnormality_detection.constants import TARGETS

OUT = Path("artifacts/postproc/replay")
THEIRS = ("dino", "eff", "dino3", "newconv")
W = {"theirs": 0.5, "ours": 0.3, "r7": 0.2}
SIBLINGS = {"Lateral OA": ("Medial OA", -0.10), "PF OA": ("Medial OA", -0.10), "Synovitis": ("Effusion", -0.20)}
VERIFIER = 0.00048  # gain an earlier separate check (not included) found for seed 42, siblings fixed at -.1/-.1/-.2
CHECK_RUN = Path("artifacts/kaggle-team-dhcons-all-newconv/output-check")
# The cell as first built, with its 3% rule (R7 dropped for every study if it missed over 3%), removed on Oct 17;
# kept for the information rows only
WITH_R7_RULE = BLEND_NOISYOR.replace("""    failed['r7'] |= {u for u in test_ids if u not in scored}
""", """    failed['r7'] |= {u for u in test_ids if u not in scored}
    if len(failed['r7']) > 0.03 * n:
        notes['r7'] = 'left out: failed on %d of %d studies' % (len(failed['r7']), n)
        del parts['r7']
""")
assert WITH_R7_RULE != BLEND_NOISYOR

DATA: dict = {}


def write(path: Path, ids: list[str], values: np.ndarray) -> None:
    frame = pd.DataFrame(values, columns=TARGETS)
    frame.insert(0, "StudyInstanceUID", ids)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)  # full precision, as the out-of-fold arrays


def run_cell(code: str, ids: list[str], parts: dict, r7_scored: list[str] | None = None, ours_filled: list[str] | None = None,
             theirs_empty: list[str] | None = None) -> tuple[np.ndarray, dict]:
    """Lay out the files as on Kaggle, run the cell, return its submission (rows in ids order) and receipt."""
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        write(root / "data/sample_submission.csv", ids, np.full((len(ids), 12), 0.5))
        for name, values in parts.items():
            write(root / f"working/{name}_sub.csv", ids, values)
        scored = ids if r7_scored is None else r7_scored
        write(root / "pred/worker0/predictions.csv", scored, np.full((len(scored), 12), 0.3))  # which studies R7 scored
        if ours_filled is not None:
            (root / "working/inference-summary.json").write_text(json.dumps({"filled_study_ids": ours_filled}))
        if theirs_empty is not None:
            (root / "working/theirs").mkdir(parents=True, exist_ok=True)
            (root / "working/theirs/no_series_studies.json").write_text(json.dumps(theirs_empty))
        local = (code.replace("/kaggle/working", str(root / "working"))
                 .replace("/kaggle/input/competitions/rsna-knee-abnormality-detection", str(root / "data")))
        env = {"THEIRS_OK": True, "OURS_OK": True, "R7_OK": True, "pred": root / "pred", "NB_T0": time.time(),
               "THEIRS_DONE": list(THEIRS), "THEIRS_SECONDS": 0.0, "OURS_SECONDS": 0.0}
        exec(compile(local, "blend", "exec"), env)
        out = pd.read_csv(root / "working/submission.csv", dtype={"StudyInstanceUID": str}).set_index("StudyInstanceUID")
        return out.loc[ids, TARGETS].to_numpy(float), json.loads((root / "working/blend_receipt.json").read_text())


def reference(parts: dict) -> np.ndarray:
    """The noisy-OR + sibling formulas of that earlier check (its script is not included) in numpy, no failures."""
    n = len(parts["r7"])
    r = {k: rankdata(v, axis=0) / n for k, v in parts.items()}
    blended = -sum(W[k] * np.log(1 - r[k] + 0.5 / n) for k in ("theirs", "ours", "r7"))
    z = norm.ppf(rankdata(blended, axis=0) / n - 0.5 / n)
    out = z.copy()
    for label, (partner, weight) in SIBLINGS.items():
        out[:, TARGETS.index(label)] = z[:, TARGETS.index(label)] + weight * z[:, TARGETS.index(partner)]
    return rankdata(out, axis=0) / n


def judged(scores: np.ndarray, idx: np.ndarray) -> list[float]:
    return [macro_auc(scores, (y[idx], k[idx])) for y, k in DATA["judges"]]


def task(arg: tuple) -> dict:
    seed, fit_half, idx = arg
    ids = [DATA["ids"][i] for i in idx]
    parts = {k: v[idx] for k, v in DATA["parts"].items()}
    n = len(ids)
    old, _ = run_cell(BLEND, ids, parts)
    new, receipt = run_cell(BLEND_NOISYOR, ids, parts)
    assert receipt["failed_studies"] == {"theirs": 0, "ours": 0, "r7": 0} and receipt["left_out"] == {}
    healthy = judged(new, idx)
    row = {"seed": seed, "fit_half": fit_half, "old": judged(old, idx), "new": healthy,
           "max_diff_vs_reference": float(np.abs(new - reference(parts)).max())}
    rng = np.random.default_rng(1000 * seed + fit_half)
    damage = {}
    for part, share in (("ours", 0.05), ("theirs", 0.05), ("r7", 0.025)):
        rows = np.sort(rng.choice(n, size=int(round(share * n)), replace=False))
        bad = {k: v.copy() for k, v in parts.items()}
        flagged = [ids[i] for i in rows]
        kwargs = {}
        if part == "ours":  # our bundle fills a failed study with the column average and lists it
            bad["ours"][rows] = np.delete(parts["ours"], rows, axis=0).mean(0)
            kwargs["ours_filled"] = flagged
        elif part == "theirs":  # a study their sets read no series for: meaningless values
            donors = rng.choice(np.setdiff1d(np.arange(n), rows), size=len(rows), replace=False)
            bad["theirs"][rows] = parts["theirs"][donors]
            kwargs["theirs_empty"] = flagged
        else:  # R7 fills a study it did not score with the column mean; its workers' files lack it
            bad["r7"][rows] = np.delete(parts["r7"], rows, axis=0).mean(0)
            kwargs["r7_scored"] = [u for i, u in enumerate(ids) if i not in set(rows)]
        new_bad, receipt = run_cell(BLEND_NOISYOR, ids, bad, **kwargs)
        assert receipt["failed_studies"][part] == len(rows), receipt
        old_kwargs = {"r7_scored": kwargs["r7_scored"]} if part == "r7" else {}
        old_bad, _ = run_cell(BLEND, ids, bad, **old_kwargs)
        damage[part] = {"studies": len(rows), "new": judged(new_bad, idx), "old": judged(old_bad, idx)}
    row["damage"] = damage
    info = {}
    for share in (0.10, 0.30):
        rows = np.sort(rng.choice(n, size=int(round(share * n)), replace=False))
        bad = {k: v.copy() for k, v in parts.items()}
        bad["r7"][rows] = np.delete(parts["r7"], rows, axis=0).mean(0)
        scored = [u for i, u in enumerate(ids) if i not in set(rows)]
        with_rule, receipt = run_cell(WITH_R7_RULE, ids, bad, r7_scored=scored)
        assert "r7" not in receipt["weights_used"]
        without_rule, receipt = run_cell(BLEND_NOISYOR, ids, bad, r7_scored=scored)
        assert "r7" in receipt["weights_used"]
        info[f"r7_not_scored_{share:.0%}"] = {"with_3pct_rule": judged(with_rule, idx), "without_rule": judged(without_rule, idx)}
    row["r7_info"] = info
    return row


def summary(diffs: list[list[float]]) -> dict:
    d = np.array(diffs)  # halves x judges
    m = d.mean(1)
    return {"mean_of_judges": round(float(m.mean()), 6), "positive_halves": int((m > 0).sum()),
            "per_judge": dict(zip(("severity", "agreement", "their_labels"), (round(float(x), 6) for x in d.mean(0)))),
            "per_judge_positive": [int((d[:, j] > 0).sum()) for j in range(3)]}


def real_files() -> dict:
    sub = pd.read_csv(CHECK_RUN / "submission.csv", dtype={"StudyInstanceUID": str})
    ids = list(sub["StudyInstanceUID"])
    parts = {k: pd.read_csv(CHECK_RUN / f"{k}_sub.csv", dtype={"StudyInstanceUID": str}).set_index("StudyInstanceUID")
             .loc[ids, TARGETS].to_numpy(float) for k in ("theirs", "ours", "r7")}
    filled = json.loads((CHECK_RUN / "inference-summary.json").read_text())["filled_study_ids"]
    old, _ = run_cell(BLEND, ids, parts, ours_filled=filled)
    new, receipt = run_cell(BLEND_NOISYOR, ids, parts, ours_filled=filled)
    return {"rows": len(new), "in_range": bool(((new > 0) & (new <= 1)).all()), "failed_studies": receipt["failed_studies"],
            "left_out": receipt["left_out"], "old_cell_reproduces_the_check_run": bool(np.abs(
                old - sub.set_index("StudyInstanceUID").loc[ids, TARGETS].to_numpy(float)).max() <= 1e-6),
            "new_matches_reference": bool(np.abs(new - reference(parts)).max() <= 1e-6)}


def main() -> None:
    ids, arrays, judges = load()
    DATA.update(ids=list(ids), judges=judges, parts={"theirs": sum(arrays["theirs"][k] for k in THEIRS) / len(THEIRS),
                                                     "ours": arrays["ours"]["s42"], "r7": arrays["r7"]})
    tasks = []
    for seed in SEEDS:
        order = np.random.default_rng(seed).permutation(len(ids))
        halves = [np.sort(order[: len(ids) // 2]), np.sort(order[len(ids) // 2:])]
        for fit_half in (0, 1):
            tasks.append((seed, fit_half, halves[1 - fit_half]))
    with Pool(10) as pool:
        rows = pool.map(task, tasks)
    full_new, _ = run_cell(BLEND_NOISYOR, DATA["ids"], DATA["parts"])
    result = {
        "rules": "experiments/postproc-noisyor-insurance-20261017.json",
        "max_diff_vs_reference": max(r["max_diff_vs_reference"] for r in rows + [{"max_diff_vs_reference": float(
            np.abs(full_new - reference(DATA["parts"])).max())}]),
        "replay_gain_new_vs_old": summary([np.subtract(r["new"], r["old"]).tolist() for r in rows]),
        "verifier": VERIFIER,
        "insurance_loss_vs_healthy_new_cell": {p: summary([np.subtract(r["damage"][p]["new"], r["new"]).tolist() for r in rows])
                                               for p in ("ours", "theirs", "r7")},
        "today_cell_same_damage_vs_its_healthy": {p: summary([np.subtract(r["damage"][p]["old"], r["old"]).tolist()
                                                              for r in rows]) for p in ("ours", "theirs", "r7")},
        "damaged_studies_per_half": {p: rows[0]["damage"][p]["studies"] for p in ("ours", "theirs", "r7")},
        "r7_info_vs_healthy": {k: {v: summary([np.subtract(r["r7_info"][k][v], r["new"]).tolist() for r in rows])
                                   for v in ("with_3pct_rule", "without_rule")} for k in rows[0]["r7_info"]},
        "real_check_run_files": real_files(),
        "halves": rows,
    }
    g = result["replay_gain_new_vs_old"]
    loss = result["insurance_loss_vs_healthy_new_cell"]
    result["pass"] = {
        "matches_verifier_formulas": result["max_diff_vs_reference"] <= 1e-6,
        "replay_gain": abs(g["mean_of_judges"] - VERIFIER) <= 0.00002 and g["positive_halves"] >= 9,
        "insurance": all(loss[p]["mean_of_judges"] >= -0.0002 for p in loss),
        # R7's worker files and their no-series file are made up here; the Kaggle check run tests the real ones
        "real_files": (result["real_check_run_files"]["rows"] == 3 and result["real_check_run_files"]["in_range"]
                       and not any(result["real_check_run_files"]["failed_studies"].values())
                       and not result["real_check_run_files"]["left_out"]
                       and result["real_check_run_files"]["old_cell_reproduces_the_check_run"]
                       and result["real_check_run_files"]["new_matches_reference"]),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "result.json").write_text(json.dumps(result, indent=1) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "halves"}, indent=1))


if __name__ == "__main__":
    main()
