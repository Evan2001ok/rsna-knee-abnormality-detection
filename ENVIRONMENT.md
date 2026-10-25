# Environment and dependencies

Python 3.11 is recommended. Use separate environments for downloads and GPU training. This is a handover recommendation, not a claim that every original experiment used one fully locked environment.

## Processing, training and inference

First install a compatible PyTorch/torchvision pair for your CUDA environment. CPU execution is sufficient for some QC and merging steps; full training requires a GPU. Then run in an isolated environment:

```bash
python -m pip install -r requirements/pipeline.txt
python tools/preflight.py --dependencies --report dependency_check.json
python -m pip freeze > environment_installed.txt
```

Saved Kaggle inference logs confirm `timm==1.0.26`, so that version is pinned. The environment must also support the DINOv3 model name used by the code. A complete, verifiable original torch/CUDA lockfile was unavailable; no versions have been invented. `torch/torchvision` are excluded from the generic pip list to avoid replacing Kaggle's existing CUDA pair. The remaining requirements cover necessary imports; save your own environment freeze after successful execution.

`numpy/pandas/pydicom/Pillow/opencv-python/scikit-learn/tqdm/matplotlib` support caching, task preparation, training and visualization; `torch/torchvision/timm` support the vision models. timm may install safetensors/huggingface-hub as dependencies. Bundled inference expects actual checkpoints and does not require downloading weights online. `requirements_reference.txt` is historical reference only; follow this document for installation.

## NDA downloads

```bash
python -m pip install -r requirements/nda.txt
python tools/preflight.py --dependencies --dependency-group nda
```

The recorded NDA Tools version is 0.7.0. Its import name is `NDATools`; its distribution name is `nda-tools`. Supply your own account, accessible packages and authorized data. Passwords, tokens and personal authentication directories are not included. Offline checks cannot guarantee NDA connectivity or login access.

## LLM/VLM scoring

The original notebook used an agent environment that could read tasks and images. This handover includes neither a universal LLM SDK nor an implicit automated scoring service. To repeat scoring, configure model access separately, apply the original prompts to the generated tasks, and save JSONL outputs for the merger. Operational prompts are unchanged. Remote model versions, sampling and the supplied image montages can affect scores obtained on a rerun.

## Platform and decoding checks

The original GPU entries were written for Kaggle/Linux, including multiprocessing/signal handling. A successful Windows offline check does not certify Windows GPU/DataLoader behavior. DICOM pixel decoding depends on TransferSyntaxUID: test a real sample from each input source; compressed syntaxes may require compatible pydicom decoding plugins in addition to the direct imports listed here. No untested original codec/CUDA version is asserted.

`tools/check_reproduction.py` scans direct imports and embedded inference source strings against declared dependency names. It does not resolve native ABI compatibility or download anything. NDA QC/conversion also uses the relevant NumPy/pydicom/PyTorch processing environment; the isolated NDA dependency group checks only the download tool.
