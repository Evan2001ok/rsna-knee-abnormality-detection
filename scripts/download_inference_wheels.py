#!/usr/bin/env python3
"""Collect small offline inference dependencies for Linux x86_64 Kaggle kernels.

The verified local baseline uses CPython 3.12. For a Python 3.11 or 3.13 Kaggle image,
build a separate wheelhouse with --python-version 3.11 or 3.13. Bundles that use the Raptor
image preparation also need the training OpenCV version: add --extra opencv-python-headless. PyTorch, torchvision,
NumPy, pandas, SciPy, scikit-learn and Pillow are expected in the Kaggle image.
The additional DICOM decoders require NumPy 2.x; check that before installation.

Run from the repository:
    uv run python scripts/download_inference_wheels.py --python-version 3.12

Downloads use locked add-on versions and binary wheels only. The final check
resolves dependencies using only the downloaded files, without installing them.
See https://pip.pypa.io/en/stable/cli/pip_download/ for cross-platform options.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import shutil
import subprocess
import sys
import tempfile
import tomllib
from datetime import datetime, timezone
from pathlib import Path

ADDON_NAMES = ("pydicom", "pylibjpeg", "pylibjpeg-libjpeg", "pylibjpeg-openjpeg", "pyjpegls", "pyyaml")
PLATFORM = "manylinux2014_x86_64"
# A compatible offline fallback, not a request to replace the kernel's NumPy.
# Avoid bundling the newer local NumPy 2.5 with older Kaggle SciPy environments.
NUMPY_FALLBACK = "2.2.6"


def pip_command() -> list[str]:
    """Use installed pip, or an isolated uv tool without changing this project."""
    if importlib.util.find_spec("pip") is not None:
        return [sys.executable, "-m", "pip"]
    if executable := shutil.which("uv"):
        return [executable, "tool", "run", "--from", "pip", "pip"]
    if executable := shutil.which("uvx"):
        return [executable, "--from", "pip", "pip"]
    raise RuntimeError("Install pip or uv to download the offline inference wheels")


def locked_requirements(lock_path: Path, names: tuple[str, ...] = ADDON_NAMES) -> list[str]:
    with Path(lock_path).open("rb") as stream:
        packages = tomllib.load(stream)["package"]
    requirements = []
    for name in names:
        versions = {package["version"] for package in packages if package["name"] == name}
        if len(versions) != 1:
            raise ValueError(f"Expected one locked version for {name}, found {sorted(versions)}")
        requirements.append(f"{name}=={versions.pop()}")
    return requirements


def download_wheels(output_dir: Path, python_version: str, lock_path: Path, extra_names: tuple[str, ...] = ()) -> Path:
    if python_version not in {"3.11", "3.12", "3.13"}:
        raise ValueError("Choose Python 3.11, 3.12 or 3.13; build one wheelhouse per Python version")
    output_dir, lock_path = Path(output_dir), Path(lock_path)
    requirements = locked_requirements(lock_path, ADDON_NAMES + tuple(extra_names))
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Use an empty output directory to avoid mixing Python versions: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    requirements_path = output_dir / "requirements-inference.txt"
    requirements_path.write_text("\n".join(requirements) + "\n")
    constraints_path = output_dir / "constraints-download.txt"
    constraints_path.write_text(f"# Download fallback only; do not force-reinstall NumPy on Kaggle.\nnumpy=={NUMPY_FALLBACK}\n")
    runner = pip_command()
    base = [*runner, "--isolated", "--disable-pip-version-check"]
    target_options = [
        "--only-binary=:all:", "--platform", PLATFORM,
        "--python-version", python_version, "--implementation", "cp",
        "--abi", "cp" + python_version.replace(".", ""),
    ]
    subprocess.run([
        *base, "download", *target_options, "--progress-bar", "off",
        "--dest", str(output_dir), "--requirement", str(requirements_path),
        "--constraint", str(constraints_path),
    ], check=True)
    # --ignore-installed forces a complete dependency check: the downloader's
    # own environment cannot conceal a missing wheel in the exported directory.
    report_path = output_dir / "offline-resolution.json"
    with tempfile.TemporaryDirectory(prefix="rsna-offline-wheel-check-") as target:
        subprocess.run([
            *base, "install", *target_options, "--dry-run", "--ignore-installed",
            "--no-index", "--find-links", str(output_dir), "--target", target,
            "--report", str(report_path), "--requirement", str(requirements_path),
        ], check=True)
    wheels = sorted(output_dir.glob("*.whl"))
    if not wheels:
        raise RuntimeError("No dependency wheels were downloaded")
    if any("torch" in wheel.name.lower() or "nvidia" in wheel.name.lower() for wheel in wheels):
        raise RuntimeError("Unexpected PyTorch/CUDA wheel; review the add-on dependencies")
    resolved = json.loads(report_path.read_text())
    resolved_packages = {
        item["metadata"]["name"]: item["metadata"]["version"]
        for item in resolved["install"]
    }
    pip_version = subprocess.check_output([*runner, "--version"], text=True).strip()
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "python_version": python_version,
        "implementation": "cp",
        "platform": PLATFORM,
        "architecture": "x86_64",
        "requirements": requirements,
        "numpy_fallback": NUMPY_FALLBACK,
        "numpy_policy": "Keep installed NumPy >=2,<3; do not use --upgrade or --force-reinstall. Fallback wheel is for dependency completeness.",
        "expected_host_packages": ["torch", "torchvision", "numpy", "pandas", "scipy", "scikit-learn", "Pillow"],
        "local_baseline_python": "3.12",
        "lock_sha256": hashlib.sha256(lock_path.read_bytes()).hexdigest(),
        "pip": pip_version,
        "offline_dependency_resolution": "passed with --no-index --ignore-installed --dry-run",
        "resolved_packages": resolved_packages,
        "wheels": [
            {"file": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "bytes": path.stat().st_size}
            for path in wheels
        ],
    }
    manifest_path = output_dir / "wheelhouse.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python-version", choices=["3.11", "3.12", "3.13"], default="3.12")
    parser.add_argument("--extra", action="append", choices=["opencv-python-headless"], default=[],
                        help="Also download this locked package (Raptor image preparation needs OpenCV)")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--lock", type=Path, default=Path(__file__).resolve().parents[1] / "uv.lock")
    args = parser.parse_args()
    output = args.output_dir or Path(f"artifacts/inference-wheels-py{args.python_version.replace('.', '')}")
    manifest = download_wheels(output, args.python_version, args.lock, tuple(args.extra))
    print(f"Offline dependency wheels verified: {manifest}")
    print("Use this wheelhouse with a matching Python version and an existing NumPy 2.x environment.")


if __name__ == "__main__":
    main()
