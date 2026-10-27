"""Retain scored epochs and average compatible inference checkpoints on CPU."""

from __future__ import annotations

import math
import pickle
import re
from copy import deepcopy
from pathlib import Path

import torch

from .utils import file_hash, save_checkpoint

_INFERENCE_FIELDS = ("format_version", "targets", "config", "epoch", "metrics", "provenance", "image_preparation")
_CANDIDATE_KEY = "_checkpoint_average_candidate"
_CANDIDATE_PATTERN = re.compile(r"candidate-epoch-([0-9]{3,})\.pt\Z")


def _epoch(value) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError("Checkpoint epochs must be positive integers")
    return value


def _score(value) -> float:
    if isinstance(value, bool) or value is None:
        raise ValueError("Checkpoint averaging requires finite validation macro_auc")
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError("Checkpoint averaging requires finite validation macro_auc") from error
    if not math.isfinite(result) or not 0 <= result <= 1:
        raise ValueError("Checkpoint averaging requires finite validation macro_auc in [0,1]")
    return result


def _contract(checkpoint: dict) -> dict:
    if not isinstance(checkpoint, dict) or any(key not in checkpoint for key in (*_INFERENCE_FIELDS, "model")):
        raise ValueError("Incomplete inference checkpoint")
    if checkpoint["format_version"] != 1:
        raise ValueError("Unsupported checkpoint format_version")
    provenance = checkpoint["provenance"]
    if not isinstance(provenance, dict) or not provenance.get("code_sha256") or not provenance.get("input_sha256"):
        raise ValueError("Checkpoint provenance needs code_sha256 and input_sha256")
    if not isinstance(checkpoint["model"], dict) or not checkpoint["model"]:
        raise ValueError("Checkpoint model state must be nonempty")
    _epoch(checkpoint["epoch"])
    _score(checkpoint["metrics"].get("macro_auc"))
    return {key: checkpoint[key] for key in ("format_version", "targets", "config", "image_preparation")} | {
        "code_sha256": provenance["code_sha256"], "input_sha256": provenance["input_sha256"],
    }


def _check_contract(actual: dict, expected: dict) -> None:
    for name in expected:
        if actual[name] != expected[name]:
            raise ValueError(f"Checkpoint {name} mismatch; averaging requires the same training run")


def _candidate_name(epoch: int) -> str:
    return f"candidate-epoch-{epoch:03d}.pt"


def _owned_candidate(checkpoint: dict, epoch: int) -> bool:
    return isinstance(checkpoint, dict) and checkpoint.get(_CANDIDATE_KEY) == {
        "version": 1, "epoch": epoch, "ranking_metric": "macro_auc",
    }


def retain_top_checkpoint(output: str | Path, inference_checkpoint: dict, history: list[dict], top_k: int) -> list[str]:
    """Save the top K epochs by macro AUC; return filenames in score order.

    Call after saving ``last.pt`` with the new history. On resume, call with that
    recovered checkpoint before training again. Its current epoch can be saved
    again; missing older selected candidates cannot be reconstructed and fail.
    Tied scores retain the earlier epoch. Only files written by this function
    with a matching run contract are eligible for deletion.
    """
    if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 1:
        raise ValueError("top_k must be a positive integer")
    contract = _contract(inference_checkpoint)
    current_epoch = inference_checkpoint["epoch"]
    scores = {}
    for record in history:
        epoch = _epoch(record["epoch"])
        if epoch in scores:
            raise ValueError("Checkpoint history contains duplicate epochs")
        if epoch > current_epoch:
            raise ValueError("Checkpoint history extends past the recovered epoch")
        scores[epoch] = _score(record["validation"].get("macro_auc"))
    if current_epoch not in scores or scores[current_epoch] != _score(inference_checkpoint["metrics"].get("macro_auc")):
        raise ValueError("Current checkpoint macro_auc does not match its history")
    selected = sorted(scores, key=lambda epoch: (-scores[epoch], epoch))[:top_k]
    names = [_candidate_name(epoch) for epoch in selected]
    output = Path(output)

    # Verify required history before either creating or deleting a candidate.
    for epoch, name in zip(selected, names, strict=True):
        path = output / name
        if not path.exists():
            if epoch != current_epoch:
                raise FileNotFoundError(f"Required top checkpoint is missing: {path}")
            continue
        existing = torch.load(path, map_location="cpu", weights_only=True)
        if not _owned_candidate(existing, epoch):
            raise ValueError(f"Refusing to overwrite a file not created by checkpoint retention: {path}")
        _check_contract(_contract(existing), contract)
        if existing["epoch"] != epoch or _score(existing["metrics"].get("macro_auc")) != scores[epoch]:
            raise ValueError(f"Candidate does not match checkpoint history: {path}")

    output.mkdir(parents=True, exist_ok=True)
    if current_epoch in selected:
        saved = {key: inference_checkpoint[key] for key in _INFERENCE_FIELDS}
        saved["model"] = inference_checkpoint["model"]
        saved[_CANDIDATE_KEY] = {"version": 1, "epoch": current_epoch, "ranking_metric": "macro_auc"}
        save_checkpoint(output / _candidate_name(current_epoch), saved)

    for path in output.glob("candidate-epoch-*.pt"):
        match = _CANDIDATE_PATTERN.fullmatch(path.name)
        if not match or path.name in names:
            continue
        try:
            checkpoint = torch.load(path, map_location="cpu", weights_only=True)
            removable = _owned_candidate(checkpoint, int(match[1])) and _contract(checkpoint) == contract
        except (OSError, EOFError, RuntimeError, ValueError, KeyError, TypeError, pickle.UnpicklingError):
            # Unrecognized files are not ours to delete. Required candidates
            # were checked above, where any corruption is a hard failure.
            continue
        if removable:
            path.unlink()
    return names


def average_checkpoints(paths: list[str | Path], output_path: str | Path) -> dict:
    """Write and return an inference checkpoint with equal-weight model averages.

    Floating parameters and buffers accumulate in float64 on CPU and are cast
    back to their saved dtypes. Integer and boolean buffers come from the best
    macro-AUC member (earlier epoch breaks ties). Source metrics are recorded in
    the audit; the averaged model's ``metrics`` is empty until it is evaluated.
    """
    sources = [Path(path) for path in paths]
    if not sources:
        raise ValueError("Provide at least one checkpoint to average")
    resolved = [path.resolve() for path in sources]
    if len(set(resolved)) != len(resolved):
        raise ValueError("Duplicate checkpoint paths would weight a member twice")
    output_path = Path(output_path)
    if output_path.resolve() in resolved or output_path.name in {"best-model.pt", "last.pt"}:
        raise ValueError("Averaging must preserve source checkpoints, best-model.pt, and last.pt")

    reference_contract = None
    schema = {}
    accumulators = {}
    integer_buffers = {}
    best_rank = None
    result = None
    members = []
    seen_epochs = set()
    with torch.no_grad():
        for path in sources:
            checkpoint = torch.load(path, map_location="cpu", weights_only=True)
            contract = _contract(checkpoint)
            if reference_contract is None:
                reference_contract = contract
            else:
                _check_contract(contract, reference_contract)
            epoch, score = checkpoint["epoch"], _score(checkpoint["metrics"].get("macro_auc"))
            if epoch in seen_epochs:
                raise ValueError("Duplicate checkpoint epochs would weight a member twice")
            seen_epochs.add(epoch)
            state = checkpoint["model"]
            if schema and set(state) != set(schema):
                raise ValueError("Checkpoint model state keys mismatch")
            rank = (-score, epoch)
            is_best = best_rank is None or rank < best_rank
            for name, tensor in state.items():
                if not isinstance(tensor, torch.Tensor) or tensor.is_complex():
                    raise ValueError(f"Unsupported model state value: {name}")
                shape_dtype = (tensor.shape, tensor.dtype)
                if name in schema and schema[name] != shape_dtype:
                    raise ValueError(f"Checkpoint model shape/dtype mismatch: {name}")
                schema[name] = shape_dtype
                if tensor.is_floating_point():
                    if not torch.isfinite(tensor).all():
                        raise ValueError(f"Checkpoint contains nonfinite model values: {name}")
                    if name not in accumulators:
                        accumulators[name] = tensor.to(torch.float64).clone()
                    else:
                        accumulators[name].add_(tensor.to(torch.float64))
                elif is_best:
                    integer_buffers[name] = tensor.clone()
            if is_best:
                best_rank = rank
                result = deepcopy({key: checkpoint[key] for key in _INFERENCE_FIELDS})
            members.append({"file": str(path.resolve()), "sha256": file_hash(path), "epoch": epoch, "macro_auc": score})

        averaged = {}
        for name, (_, dtype) in schema.items():
            if name in accumulators:
                tensor = accumulators[name].div_(len(sources)).to(dtype=dtype)
                if not torch.isfinite(tensor).all():
                    raise ValueError(f"Averaging produced nonfinite model values: {name}")
                averaged[name] = tensor
            else:
                averaged[name] = integer_buffers[name]
    result["model"] = averaged
    result["metrics"] = {}
    result["checkpoint_average"] = {
        "method": "equal_weight_parameters_and_floating_buffers",
        "accumulator_dtype": "float64", "ranking_metric": "macro_auc",
        "integer_buffers_from_epoch": result["epoch"],
        "members": sorted(members, key=lambda member: (-member["macro_auc"], member["epoch"])),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_checkpoint(output_path, result)
    return result
