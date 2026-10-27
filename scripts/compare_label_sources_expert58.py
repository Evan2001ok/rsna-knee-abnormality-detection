#!/usr/bin/env python3
"""Score label sources against the 58 expert-labeled studies, finding by finding.

Sources: image-only labels from hosted AI models (by default our GPT and Claude image readings, which
are not public; --image-labels name=path, repeatable), our five fold models (none trained on these
studies), the public report readings available for these studies (Pilkwang scores, stevenleehans v2,
GPT-5.6) and their mean, plus rank blends. Uses studies every source covers.
Paired bootstrap intervals compare macro AUCs. A diagnostic: 58 studies only resolve large differences.
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from rsna_knee_abnormality_detection.constants import STUDY_ID, TARGETS

PUBLIC = Path("artifacts/public-training-review/labels")
GPT_DIR = Path("artifacts/gpt-image-labels/v1")
DEFAULT_IMAGE_LABELS = [f"gpt_image={GPT_DIR / 'labels-replies-medium.csv'}",
                        f"claude_image={GPT_DIR / 'labels-replies-claude-high.csv'}"]


def read(path) -> pd.DataFrame:
    return pd.read_csv(path, dtype={STUDY_ID: str}).set_index(STUDY_ID)


def macro_auc(truth: np.ndarray, scores: np.ndarray, weights: np.ndarray | None = None) -> tuple[float, list[float]]:
    per = []
    for column in range(truth.shape[1]):
        y, s = truth[:, column], scores[:, column]
        present = y if weights is None else y[weights > 0]
        per.append(roc_auc_score(y, s, sample_weight=weights) if len(set(present)) == 2 else np.nan)
    return float(np.nanmean(per)), per


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-labels", action="append", help="name=path of an image-only label CSV (repeatable)")
    parser.add_argument("--teacher-predictions", type=Path, default=GPT_DIR / "teacher5-expert58-predictions.csv")
    parser.add_argument("--output", type=Path, default=Path("artifacts/label-sources-expert58.json"))
    parser.add_argument("--samples", type=int, default=2000)
    args = parser.parse_args()
    image_sources = dict(item.split("=", 1) for item in (args.image_labels or DEFAULT_IMAGE_LABELS))
    expert = read("data/labels/baseline/expert_labels.csv")[TARGETS]
    sources = {name: read(path)[TARGETS] for name, path in image_sources.items()}
    sources.update({
        "our_5_folds": read(args.teacher_predictions)[TARGETS],
        "report_pilkwang": read("data/labels/public/pilkwang/report_labels_v2.csv")[TARGETS],
        "report_stevenleehans": read(PUBLIC / "stevenleehans_rsna-knee-llm-report-labels/llm_labels_v2.csv")[TARGETS],
        "report_gpt56": read(PUBLIC / "lixin73_rsna-knee-llm-report-labels-sol56/labels_llm_gpt56sol.csv")[TARGETS],
    })
    common = expert.index
    for frame in sources.values():
        common = common.intersection(frame.dropna(how="any").index)
    truth = expert.loc[common].to_numpy()
    sources = {name: frame.loc[common] for name, frame in sources.items()}
    sources["report_mean"] = (sources["report_pilkwang"] + sources["report_stevenleehans"] + sources["report_gpt56"]) / 3
    ranked = {name: frame.rank(pct=True) for name, frame in sources.items()}
    blend = lambda names: sum(ranked[name] for name in names) / len(names)  # noqa: E731
    image_names = list(image_sources)
    for name in image_names:
        sources[f"blend_{name}_folds"] = blend([name, "our_5_folds"])
    if len(image_names) > 1:
        sources["blend_images"] = blend(image_names)
        sources["blend_images_folds"] = blend([*image_names, "our_5_folds"])
        sources["blend_images_folds_report"] = blend([*image_names, "our_5_folds", "report_mean"])
    result = {"studies": len(common), "positives": dict(zip(TARGETS, truth.sum(axis=0).astype(int).tolist())), "sources": {}}
    for name, frame in sources.items():
        macro, per = macro_auc(truth, frame.to_numpy())
        result["sources"][name] = {"macro_auc": round(macro, 4), "per_finding": dict(zip(TARGETS, [round(v, 4) for v in per]))}
    pairs = [(name, "report_mean") for name in image_names] + [(name, "our_5_folds") for name in image_names]
    pairs += list(itertools.combinations(image_names, 2))
    pairs += [(f"blend_{name}_folds", "our_5_folds") for name in image_names]
    if len(image_names) > 1:
        pairs += [("blend_images_folds", "our_5_folds"), ("blend_images_folds_report", "our_5_folds")]
    counts = np.random.default_rng(42).multinomial(len(common), np.full(len(common), 1 / len(common)), size=args.samples)
    result["comparisons"] = {}
    for candidate, reference in pairs:
        differences = [macro_auc(truth, sources[candidate].to_numpy(), w.astype(float))[0]
                       - macro_auc(truth, sources[reference].to_numpy(), w.astype(float))[0] for w in counts]
        low, high = np.nanpercentile(differences, [2.5, 97.5])
        result["comparisons"][f"{candidate} minus {reference}"] = {
            "difference": round(result["sources"][candidate]["macro_auc"] - result["sources"][reference]["macro_auc"], 4),
            "paired_95_percent_interval": [round(float(low), 4), round(float(high), 4)]}
    args.output.write_text(json.dumps(result, indent=1) + "\n")
    print(json.dumps({"studies": result["studies"], "macro": {k: v["macro_auc"] for k, v in result["sources"].items()},
                      "comparisons": result["comparisons"]}, indent=1))


if __name__ == "__main__":
    main()
