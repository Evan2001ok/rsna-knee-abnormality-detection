#!/usr/bin/env python3
"""Choose the 5-fold teacher recipe from fold-0 scores with a rule fixed in advance, then write its fold configs.

Teachers are the first-round fold models whose out-of-fold predictions refine the student labels
(prepare_student_labels.py). Rule (fixed on 2026-10-05, before the candidate finished): use the candidate recipe
unless its fold-0 macro AUC with mirrored test-time predictions (each study's prediction averaged with that of its
left/right mirror image) is more than MARGIN below the best alternative; otherwise use that alternative. MARGIN is
about twice the run-to-run noise. Each fold config is the chosen config with only data.fold and training.output_dir changed.
The decision is written to --decision; write_student_config.py and prepare_round3_labels.py read it by default
from artifacts/teacher-recipe-decision.json.
"""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from rsna_knee_abnormality_detection.config import load_config
from rsna_knee_abnormality_detection.utils import file_hash, write_json

MARGIN = 0.002


def mirrored_auc(run: Path) -> float:
    return json.loads((run / "mirror-tta-averaged-model.json").read_text())["mirror_tta"]["macro_auc"]


def choose(candidate: dict, alternatives: list[dict], margin: float = MARGIN) -> dict:
    best = max(alternatives, key=lambda item: item["auc"])
    return candidate if candidate["auc"] >= best["auc"] - margin else best


def replace_once(text: str, old: str, new: str) -> str:
    if text.count(old) != 1:
        raise ValueError(f"Expected exactly one {old.strip()!r} line in the config")
    return text.replace(old, new)


def fold_config(text: str, source: Path, fold: int, output_dir: str) -> str:
    current = load_config(source)
    text = replace_once(text, f"  fold: {current['data']['fold']}\n", f"  fold: {fold}\n")
    text = replace_once(text, f"  output_dir: {current['training']['output_dir']}\n", f"  output_dir: {output_dir}\n")
    note = f"# Teacher fold {fold} for the teacher-student round, written by scripts/choose_teacher_recipe.py from {source}.\n"
    return note + text


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-config", type=Path, required=True)
    parser.add_argument("--candidate-run", type=Path, required=True)
    parser.add_argument("--alternative-configs", type=Path, nargs="+", required=True)
    parser.add_argument("--alternative-runs", type=Path, nargs="+", required=True)
    parser.add_argument("--folds", type=int, nargs="+", default=[1, 2, 3, 4])
    parser.add_argument("--config-dir", type=Path, default=Path("configs/teachers"))
    parser.add_argument("--decision", type=Path, required=True)
    args = parser.parse_args()
    if len(args.alternative_configs) != len(args.alternative_runs):
        raise ValueError("Give one run per alternative config")
    entries = [{"name": config.stem, "config": str(config), "run": str(run), "auc": mirrored_auc(run)}
               for config, run in [(args.candidate_config, args.candidate_run),
                                   *zip(args.alternative_configs, args.alternative_runs)]]
    candidate, alternatives = entries[0], entries[1:]
    chosen = choose(candidate, alternatives)
    source = Path(chosen["config"])
    if load_config(source)["data"]["fold"] != 0:
        raise ValueError("The chosen recipe's config must be its fold-0 run")
    args.config_dir.mkdir(parents=True, exist_ok=True)
    text = source.read_text()
    fold_configs, fold_runs = [], []
    for fold in args.folds:
        output_dir = f"runs/teacher_{chosen['name']}_fold{fold}"
        path = args.config_dir / f"{chosen['name']}_fold{fold}.yaml"
        content = fold_config(text, source, fold, output_dir)
        if path.exists() and path.read_text() != content:
            raise FileExistsError(f"{path} exists with different content")
        path.write_text(content)
        settings = load_config(path)
        if settings["data"]["fold"] != fold or settings["training"]["output_dir"] != output_dir:
            raise AssertionError(f"{path} did not get the intended fold and output directory")
        fold_configs.append(str(path))
        fold_runs.append(output_dir)
    write_json(args.decision, {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "rule": (f"Use the candidate unless its mirrored fold-0 macro AUC is more than {MARGIN} below the best "
                 "alternative; otherwise use that alternative."),
        "candidate": candidate, "alternatives": alternatives, "chosen": chosen["name"],
        "chosen_config_sha256": file_hash(source),
        "teacher_runs": [chosen["run"], *fold_runs], "fold_configs": fold_configs,
    })
    print("\n".join(fold_configs))


if __name__ == "__main__":
    main()
