#!/usr/bin/env python3
"""Score a finished run's model on its validation fold with and without mirrored test-time predictions.

Mirroring averages each study's prediction with its left/right mirror image
(coronal/axial windows flipped, sagittal slice order reversed). Only the saved
model and the run's own validation studies are used; nothing is trained or tuned.
"""

import argparse
import json
from pathlib import Path

import torch

from rsna_knee_abnormality_detection.constants import STUDY_ID
from rsna_knee_abnormality_detection.dataset import make_dataset
from rsna_knee_abnormality_detection.predict import load_model
from rsna_knee_abnormality_detection.train import make_loader, predict_loader, study_partitions
from rsna_knee_abnormality_detection.utils import file_hash, write_json


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True, help="Run directory containing the model")
    parser.add_argument("--model", default="averaged-model.pt", help="Model file inside the run directory")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    checkpoint_path = args.run / args.model
    device = torch.device("cuda")
    model, checkpoint = load_model(checkpoint_path, device)
    config = checkpoint["config"]
    _, validation = study_partitions(config)
    recorded = json.loads((args.run / "provenance.json").read_text())["validation_ids"]
    if validation[STUDY_ID].tolist() != recorded:
        raise ValueError("Validation studies differ from the ones recorded for this run")
    data = make_dataset(config["data"]["manifest"], validation, config["data"], training=False, seed=config["seed"])
    settings = {"batch_size": args.batch_size, "num_workers": args.workers, "device": "cuda"}
    result = {"model": {"path": str(checkpoint_path), "sha256": file_hash(checkpoint_path)},
              "validation_studies": len(validation), "validation_labels": config["data"].get("validation_labels")}
    for name, mirror in (("plain", False), ("mirror_tta", True)):
        loader = make_loader(data, settings, training=False, seed=config["seed"])
        predictions, metrics = predict_loader(model, loader, device, config["training"]["precision"],
                                              test_time_mirror=mirror)
        output = args.run / f"validation-{Path(args.model).stem}-{name}.csv"
        predictions.to_csv(output, index=False)
        result[name] = {"macro_auc": metrics["macro_auc"], "predictions": str(output),
                        "per_finding_auc": {key: value["auc"] for key, value in metrics["per_target"].items()}}
    result["mirror_gain"] = result["mirror_tta"]["macro_auc"] - result["plain"]["macro_auc"]
    write_json(args.run / f"mirror-tta-{Path(args.model).stem}.json", result)
    # Single-point charts beside the run's own training curves, at its final update.
    from torch.utils.tensorboard import SummaryWriter

    step = json.loads((args.run / "history.json").read_text())[-1]["global_step"]
    with SummaryWriter(log_dir=str(args.run / "tensorboard")) as writer:
        prefix = f"Mirror_TTA_{Path(args.model).stem.replace('-', '_')}"
        for name in ("plain", "mirror_tta"):
            writer.add_scalar(f"{prefix}/macro_12_findings_{name}", result[name]["macro_auc"], step)
            for finding, value in result[name]["per_finding_auc"].items():
                writer.add_scalar(f"{prefix}_by_finding/{finding.replace(' ', '_')}_{name}", value, step)
        writer.add_scalar(f"{prefix}/gain", result["mirror_gain"], step)
    print(json.dumps({key: result[key] for key in ("validation_studies", "mirror_gain")} |
                     {name: result[name]["macro_auc"] for name in ("plain", "mirror_tta")}))


if __name__ == "__main__":
    main()
