"""Command line entry points; heavy modules load only when needed."""

import argparse
import json
from pathlib import Path


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Train and evaluate study-level knee MRI models.")
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("labels", help="Import report targets and preserve expert labels")
    p.add_argument("--train-csv", type=Path, required=True)
    p.add_argument("--weak-csv", type=Path)
    p.add_argument("--source")
    p.add_argument("--output-dir", type=Path, default=Path("data/labels/baseline"))
    p = sub.add_parser("split", help="Create fixed study/patient folds and an expert holdout")
    p.add_argument("--labels", type=Path, default=Path("data/labels/baseline/labels.csv"))
    p.add_argument("--output-dir", type=Path, default=Path("data/splits/baseline"))
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--groups-csv", type=Path)
    p = sub.add_parser("prepare", help="Decode, order, normalize, and cache DICOM volumes")
    p.add_argument("--data-root", type=Path, default=Path("data/rsna-knee-abnormality-detection"))
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--split", choices=["train", "test"], default="train")
    p.add_argument("--size", type=int, default=320)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--study-ids", type=Path, help="CSV containing StudyInstanceUID, or text with one ID per line")
    p.add_argument("--limit-studies", type=int)
    p.add_argument("--strict", action="store_true", help="Fail on any unusable series")
    p = sub.add_parser("verify-cache", help="Check cached image geometry, sizes, and optional pixel checksums")
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--size", type=int, default=320)
    p.add_argument("--full", action="store_true", help="Read every cached pixel to verify its checksum")
    p = sub.add_parser("select-studies", help="Write the exact training/validation IDs selected by a config")
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p = sub.add_parser("preview", help="View the images the model will actually receive")
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--output", type=Path, default=Path("artifacts/inputs.jpg"))
    p.add_argument("--count", type=int, default=4)
    p.add_argument("--training", action="store_true")
    p = sub.add_parser("train", help="Fit one fold and save predictions, metrics, and resumable checkpoints")
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--resume", type=Path)
    p.add_argument("--fold", type=int, help="Override data.fold; also supply --output-dir for a new fold")
    p.add_argument("--output-dir", type=Path)
    p.add_argument("--stop-after-epoch", type=int, help="Stop cleanly at this epoch while keeping the original schedule")
    p = sub.add_parser("predict", help="Predict with one or more checkpoints, without downloading weights")
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--studies", type=Path, required=True)
    p.add_argument("--checkpoints", type=Path, nargs="+", required=True)
    p.add_argument("--output", type=Path, default=Path("submission.csv"))
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--device", default="cuda")
    p.add_argument("--weights", type=float, nargs="+")
    p = sub.add_parser("evaluate", help="Score predictions against unchanged reference labels")
    p.add_argument("--labels", type=Path, required=True)
    p.add_argument("--predictions", type=Path, required=True)
    p.add_argument("--output", type=Path)
    p = sub.add_parser("ensemble", help="Blend predictions after strict ID/column alignment")
    p.add_argument("--inputs", type=Path, nargs="+", required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--weights", type=float, nargs="+")
    p.add_argument("--method", choices=["mean", "rank"], default="mean")
    p.add_argument("--labels", type=Path)
    p = sub.add_parser("export", help="Build a source/model bundle and offline Kaggle notebook")
    p.add_argument("--checkpoints", type=Path, nargs="+", required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--wheelhouse", type=Path, help="Optional directory of offline dependency wheels")
    p.add_argument("--test-time-mirror", action="store_true", help="Average each study with its mirror image")
    p.add_argument("--eval-windows", type=int, help="Prediction windows per study (default: each model's own setting)")
    p.add_argument("--test-time-rotations", help="In-plane rotated views to average, in degrees, e.g. -12,0,12 (default: none)")
    p.add_argument("--weights-json", type=Path,
                   help="JSON list, one entry per checkpoint: a blend weight, or a mapping from every finding to one")
    p = sub.add_parser("import-history", help="Import saved training history into TensorBoard once")
    p.add_argument("--runs-dir", type=Path, default=Path("runs"))
    args = parser.parse_args(argv)
    if args.command == "labels":
        from .labels import prepare_labels
        result = prepare_labels(args.train_csv, args.output_dir, args.weak_csv, args.source)
    elif args.command == "split":
        from .splits import make_splits
        result = make_splits(args.labels, args.output_dir, args.folds, args.seed, args.groups_csv)
    elif args.command == "prepare":
        from .prepare import prepare_data
        study_ids = None
        if args.study_ids:
            if args.study_ids.suffix.lower() == ".csv":
                import pandas as pd

                from .constants import STUDY_ID
                study_ids = pd.read_csv(args.study_ids, dtype=str)[STUDY_ID].tolist()
            else:
                study_ids = [line.strip() for line in args.study_ids.read_text().splitlines() if line.strip()]
        result = prepare_data(args.data_root, args.output_dir, args.split, args.size, args.workers,
                              study_ids, args.limit_studies, args.strict)
    elif args.command == "verify-cache":
        from .prepare import validate_manifest
        result = validate_manifest(args.manifest, expected_size=args.size, verify_checksums=args.full)
        print(json.dumps(result, indent=2, allow_nan=False))
        return
    elif args.command == "select-studies":
        import pandas as pd

        from .config import load_config
        from .constants import STUDY_ID
        from .train import study_partitions
        frames = study_partitions(load_config(args.config), require_manifest=False)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        pd.concat(frames)[[STUDY_ID]].to_csv(args.output, index=False)
        result = args.output
    elif args.command in {"train", "preview"}:
        from .config import load_config
        config = load_config(args.config)
        if args.command == "train":
            from .train import train
            if args.fold is not None:
                if args.fold != config["data"]["fold"] and args.output_dir is None:
                    parser.error("A different fold requires its own --output-dir")
                config["data"]["fold"] = args.fold
            if args.output_dir is not None:
                config["training"]["output_dir"] = str(args.output_dir)
            result = train(config, args.resume, args.stop_after_epoch)
        else:
            from .preview import preview_inputs
            result = preview_inputs(config, args.output, args.count, args.training)
    elif args.command == "predict":
        from .predict import predict
        result = predict(args.manifest, args.studies, args.checkpoints, args.output, args.batch_size,
                         args.workers, args.device, args.weights)
    elif args.command == "evaluate":
        from .metrics import evaluate_files
        from .utils import write_json
        result = evaluate_files(args.labels, args.predictions)
        if args.output:
            write_json(args.output, result)
        print(json.dumps(result, indent=2, allow_nan=False))
        return
    elif args.command == "ensemble":
        from .ensemble import ensemble_predictions
        result = ensemble_predictions(args.inputs, args.output, args.weights, args.method, args.labels)
    elif args.command == "export":
        from .bundle import export_bundle
        weights = json.loads(args.weights_json.read_text()) if args.weights_json else None
        rotations = None if args.test_time_rotations is None else tuple(float(a) for a in args.test_time_rotations.split(","))
        result = export_bundle(args.checkpoints, args.output_dir, args.wheelhouse, args.test_time_mirror, weights,
                               args.eval_windows, rotations)
    elif args.command == "import-history":
        from .monitoring import import_histories
        result = json.dumps(import_histories(args.runs_dir), indent=2)
    print(result)
