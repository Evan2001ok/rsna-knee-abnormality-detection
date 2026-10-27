#!/usr/bin/env python3
"""Run heliosli's public compact ConvNeXt-small bundles (S6, also called R6, and R7) outside Kaggle on training studies.

October 13, 2026: used to choose S6's blend weight and to try R7. Both bundles train four fold models on hash folds of
the training pool without the 58 expert ("gold") studies: fold k holds the studies with
int(sha1(StudyInstanceUID)[:8], 16) % 4 == k, and model f<k> never trained on them (compact_train.py in heliosli's
bundle: tr = pool_fold != k). So running only model f<k> on the studies of hash fold k gives clean out-of-fold
predictions for every training study. The 58 expert studies were never trained on (the bundles' manifests:
gold_in_training 0) and are scored with all four models, as on Kaggle.

The notebook's own setup cell runs unchanged except for four paths (inputs, working folder, temporary folder and the data
folder, which lays the chosen studies out as a test set with links into train_series). It checks every code and weight
file by SHA-256 and installs heliosli's pinned pydicom and timm wheels into this Python. The run step is the notebook's
probe_dual.py call with more loader workers. We ran it in a separate Python environment kept only for this, since the
setup cell installs its own wheels into it.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

PATCHES = (
    ("INPUT = Path('/kaggle/input')", "INPUT = Path(os.environ['PC_INPUT'])"),
    ("WORK = Path('/kaggle/working')", "WORK = Path(os.environ['PC_WORK'])"),
    ("TMP = TMP / 'probe'", "TMP = Path(os.environ['PC_TMP'])"),
    ("DATA = find_mount(COMPETITION, 'test.csv')", "DATA = Path(os.environ['PC_DATA'])"),
)


def lay_out(competition: Path, ids: list[str], data: Path) -> int:
    """Write test.csv, test_series.csv, sample_submission.csv and test_series/<study> links for the chosen studies."""
    (data / "test_series").mkdir(parents=True, exist_ok=True)
    wanted = set(ids)
    with open(competition / "train_series.csv", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = [r for r in reader if r["StudyInstanceUID"] in wanted]
        fields = reader.fieldnames
    if {r["StudyInstanceUID"] for r in rows} != wanted:
        raise SystemExit("some studies have no series rows")
    with open(data / "test_series.csv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    with open(competition / "sample_submission.csv", newline="") as handle:
        columns = next(csv.reader(handle))
    (data / "test.csv").write_text("StudyInstanceUID\n" + "".join(u + "\n" for u in ids))
    with open(data / "sample_submission.csv", "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(columns)
        writer.writerows([u] + ["0.5"] * (len(columns) - 1) for u in ids)
    for u in ids:
        link = data / "test_series" / u
        if not link.exists():
            os.symlink(competition / "train_series" / u, link)
    return len(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--setup-cell", type=Path, required=True, help="The notebook's setup cell as a .py file")
    parser.add_argument("--studies", type=Path, required=True, help="CSV with a StudyInstanceUID column")
    parser.add_argument("--fold", type=int, default=None,
                        help="Keep only model f<fold> and the studies of that hash fold; omit to use all four models")
    parser.add_argument("--competition", type=Path, required=True)
    parser.add_argument("--inputs", type=Path, required=True, help="Folder holding the downloaded datasets by slug")
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--loader-workers", type=int, default=12)
    parser.add_argument("--max-fallback-fraction", default=None)
    args = parser.parse_args()
    started = time.time()
    with open(args.studies, newline="") as handle:
        ids = [row["StudyInstanceUID"] for row in csv.DictReader(handle)]
    if args.fold is not None:
        import hashlib

        ids = [u for u in ids if int(hashlib.sha1(u.encode()).hexdigest()[:8], 16) % 4 == args.fold]
    work = args.work.resolve()
    if work.exists():
        shutil.rmtree(work)
    data, tmp, out = work / "data", work / "tmp", work / "out"
    for folder in (tmp, out):
        folder.mkdir(parents=True)
    series = lay_out(args.competition.resolve(), ids, data)
    print(f"LAID_OUT {len(ids)} studies, {series} series, fold {args.fold}", flush=True)
    setup = args.setup_cell.read_text()
    for old, new in PATCHES:
        if setup.count(old) != 1:
            raise SystemExit(f"setup cell: expected exactly one {old!r}")
        setup = setup.replace(old, new)
    os.environ.update(PC_INPUT=str(args.inputs.resolve()), PC_WORK=str(out), PC_TMP=str(tmp), PC_DATA=str(data))
    namespace = {"NB_T0": started, "__name__": "setup_cell"}
    exec(compile(setup, str(args.setup_cell), "exec"), namespace)
    bundle = Path(namespace["BUNDLE"])
    if args.fold is not None:
        for path in bundle.glob(namespace["ARM"] + "_s*_f*.pt"):
            if not path.name.endswith(f"_f{args.fold}.pt"):
                path.unlink()
    models = sorted(p.name for p in bundle.glob(namespace["ARM"] + "_s*_f*.pt"))
    print("MODELS", models, flush=True)
    pred = tmp / "prediction"
    command = [sys.executable, "-u", str(Path(namespace["SRC"]) / "probe_dual.py"), "--data", str(data), "--split", "test",
               "--code", str(namespace["CODE"]), "--bundle", str(bundle), "--arm", namespace["ARM"], "--output", str(pred),
               "--amp", "auto", "--loader-workers", str(args.loader_workers), "--deadline", str(started + 23 * 3600),
               "--worker", namespace["WORKER"], "--worker-args", json.dumps(namespace["WORKER_ARGS"])]
    if args.max_fallback_fraction is not None:
        command += ["--max-fallback-fraction", str(args.max_fallback_fraction)]
    env = dict(os.environ, PYTHONPATH=str(namespace["CODE"]) + os.pathsep + os.environ.get("PYTHONPATH", ""),
               PYTHONDONTWRITEBYTECODE="1")
    rc = subprocess.run(command, env=env).returncode
    for log in sorted(pred.glob("worker*/log.txt")):
        shutil.copyfile(log, out / (log.parent.name + "_log.txt"))
    if rc != 0:
        raise SystemExit(f"controller failed with exit code {rc}")
    shutil.copyfile(pred / "submission.csv", out / "predictions.csv")
    receipt = json.loads((pred / "submission_receipt.json").read_text())
    receipt.update(models=models, fold=args.fold, studies=len(ids), bundle_id=namespace["BUNDLE_ID"],
                   seconds_total=round(time.time() - started, 1))
    (out / "receipt.json").write_text(json.dumps(receipt, indent=1))
    print("DONE", json.dumps({k: receipt.get(k) for k in ("status", "rows", "scored", "fallback_studies", "seconds",
                                                          "per_study_mean", "models")}), flush=True)


if __name__ == "__main__":
    main()
