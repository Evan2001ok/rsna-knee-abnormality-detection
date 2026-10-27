from copy import deepcopy

import pytest
import torch

from rsna_knee_abnormality_detection.checkpoint_average import average_checkpoints, retain_top_checkpoint
from rsna_knee_abnormality_detection.constants import TARGETS
from rsna_knee_abnormality_detection.utils import file_hash, save_checkpoint


def checkpoint(epoch, score, value=1.0):
    return {
        "format_version": 1,
        "epoch": epoch,
        "targets": TARGETS,
        "config": {"model": {"backbone": "test"}, "data": {"fold": 0}},
        "metrics": {"macro_auc": score, "loss": 0.5},
        "image_preparation": {"preprocess_version": "test-v1", "image_size": 384},
        "provenance": {"code_sha256": "code-a", "input_sha256": {"labels": "labels-a", "manifest": "manifest-a"}},
        "model": {
            "weight": torch.tensor([value], dtype=torch.float32),
            "running_mean": torch.tensor([epoch / 2], dtype=torch.float16),
            "running_var": torch.tensor([epoch], dtype=torch.float64),
            "num_batches_tracked": torch.tensor(epoch * 100, dtype=torch.int64),
            "flag": torch.tensor(epoch % 2 == 0),
        },
        # A caller may pass the last training checkpoint when recovering a run.
        "optimizer": {"state": "must not appear in inference files"},
        "rng": {"test": "must not appear in inference files"},
    }


def record(saved):
    return {"epoch": saved["epoch"], "validation": saved["metrics"]}


def test_retention_keeps_top_epochs_and_preserves_other_files(tmp_path):
    best = tmp_path / "best-model.pt"
    best.write_bytes(b"original best model")
    best_hash = file_hash(best)
    unrelated = tmp_path / "candidate-epoch-099.pt"
    save_checkpoint(unrelated, checkpoint(99, 0.99))
    unrelated_hash = file_hash(unrelated)
    (tmp_path / "candidate-epoch-notes.pt").write_bytes(b"user notes")
    history = []
    for epoch, score in enumerate([0.7, 0.7, 0.9, 0.6], start=1):
        saved = checkpoint(epoch, score)
        history.append(record(saved))
        selected = retain_top_checkpoint(tmp_path, saved, history, 2)
    # The earlier epoch wins a score tie; neither the last nor all epochs are kept.
    assert selected == ["candidate-epoch-003.pt", "candidate-epoch-001.pt"]
    assert not (tmp_path / "candidate-epoch-002.pt").exists()
    assert not (tmp_path / "candidate-epoch-004.pt").exists()
    assert file_hash(best) == best_hash
    assert file_hash(unrelated) == unrelated_hash
    assert (tmp_path / "candidate-epoch-notes.pt").read_bytes() == b"user notes"
    candidate = torch.load(tmp_path / selected[0], weights_only=True)
    assert "optimizer" not in candidate and "rng" not in candidate


def test_resume_can_recreate_current_candidate_but_missing_older_candidate_fails(tmp_path):
    first, second = checkpoint(1, 0.9), checkpoint(2, 0.8)
    retain_top_checkpoint(tmp_path, first, [record(first)], 2)
    history = [record(first), record(second)]
    # last.pt can be saved before a crash interrupts writing its candidate.
    selected = retain_top_checkpoint(tmp_path, second, history, 2)
    assert selected == ["candidate-epoch-001.pt", "candidate-epoch-002.pt"]
    (tmp_path / selected[1]).unlink()
    retain_top_checkpoint(tmp_path, second, history, 2)
    before = file_hash(tmp_path / selected[1])
    (tmp_path / selected[0]).unlink()
    with pytest.raises(FileNotFoundError, match="Required top checkpoint"):
        retain_top_checkpoint(tmp_path, second, history, 2)
    assert file_hash(tmp_path / selected[1]) == before


def test_resume_removes_abandoned_future_candidate_from_same_run(tmp_path):
    saved = checkpoint(1, 0.7)
    selected = retain_top_checkpoint(tmp_path, saved, [record(saved)], 2)
    future = torch.load(tmp_path / selected[0], weights_only=True)
    future["epoch"] = 9
    future["_checkpoint_average_candidate"]["epoch"] = 9
    save_checkpoint(tmp_path / "candidate-epoch-009.pt", future)
    assert retain_top_checkpoint(tmp_path, saved, [record(saved)], 2) == selected
    assert not (tmp_path / "candidate-epoch-009.pt").exists()


def test_retention_rejects_foreign_collision_and_changed_history(tmp_path):
    saved = checkpoint(1, 0.7)
    path = tmp_path / "candidate-epoch-001.pt"
    save_checkpoint(path, saved)
    original_hash = file_hash(path)
    with pytest.raises(ValueError, match="not created by checkpoint retention"):
        retain_top_checkpoint(tmp_path, saved, [record(saved)], 1)
    assert file_hash(path) == original_hash
    path.unlink()
    retain_top_checkpoint(tmp_path, saved, [record(saved)], 1)
    changed = checkpoint(2, 0.8)
    with pytest.raises(ValueError, match="does not match checkpoint history"):
        retain_top_checkpoint(tmp_path, changed, [record(checkpoint(1, 0.6)), record(changed)], 2)


def test_average_uses_float64_and_best_member_integer_buffers_without_changing_inputs(tmp_path):
    paths = []
    for epoch, score, value in [(1, 0.8, 1e8), (2, 0.9, 1.0), (3, 0.7, -1e8)]:
        path = tmp_path / f"epoch-{epoch}.pt"
        save_checkpoint(path, checkpoint(epoch, score, value))
        paths.append(path)
    hashes = [file_hash(path) for path in paths]
    best = tmp_path / "best-model.pt"
    best.write_bytes(b"original best")
    result = average_checkpoints(paths, tmp_path / "averaged-model.pt")
    # Float32 accumulation would lose the +1 before the cancellation.
    torch.testing.assert_close(result["model"]["weight"], torch.tensor([1 / 3], dtype=torch.float32), rtol=0, atol=0)
    assert result["model"]["running_mean"].dtype == torch.float16
    assert result["model"]["running_var"].dtype == torch.float64
    assert result["model"]["running_mean"].item() == 1
    assert result["model"]["running_var"].item() == 2
    assert result["model"]["num_batches_tracked"].item() == 200
    assert result["model"]["flag"].item() is True
    assert result["epoch"] == 2
    assert result["metrics"] == {}  # Member metrics are not the averaged model's result.
    assert "optimizer" not in result and "rng" not in result
    audit = result["checkpoint_average"]
    assert audit["accumulator_dtype"] == "float64"
    assert audit["integer_buffers_from_epoch"] == 2
    assert [member["epoch"] for member in audit["members"]] == [2, 1, 3]
    assert {member["sha256"] for member in audit["members"]} == set(hashes)
    assert [file_hash(path) for path in paths] == hashes
    assert best.read_bytes() == b"original best"
    restored = torch.load(tmp_path / "averaged-model.pt", weights_only=True)
    assert restored["checkpoint_average"] == audit
    for name, value in result["model"].items():
        assert torch.equal(value, restored["model"][name])
        assert value.device.type == "cpu"


@pytest.mark.parametrize("change,field", [
    (lambda ck: ck["config"]["data"].update(fold=1), "config"),
    (lambda ck: ck.update(targets=list(reversed(TARGETS))), "targets"),
    (lambda ck: ck["image_preparation"].update(preprocess_version="different"), "image_preparation"),
    (lambda ck: ck["provenance"].update(code_sha256="different"), "code_sha256"),
    (lambda ck: ck["provenance"]["input_sha256"].update(labels="different"), "input_sha256"),
])
def test_average_rejects_mismatched_training_contract(tmp_path, change, field):
    one, two = checkpoint(1, 0.7), deepcopy(checkpoint(2, 0.8))
    change(two)
    paths = [tmp_path / "one.pt", tmp_path / "two.pt"]
    for path, saved in zip(paths, [one, two], strict=True):
        save_checkpoint(path, saved)
    output = tmp_path / "average.pt"
    with pytest.raises(ValueError, match=field):
        average_checkpoints(paths, output)
    assert not output.exists()


@pytest.mark.parametrize("kind", ["keys", "shape", "dtype", "nonfinite"])
def test_average_rejects_incompatible_or_nonfinite_model_states(tmp_path, kind):
    first, second = checkpoint(1, 0.7), checkpoint(2, 0.8)
    if kind == "keys":
        del second["model"]["weight"]
    elif kind == "shape":
        second["model"]["weight"] = torch.ones(2)
    elif kind == "dtype":
        second["model"]["weight"] = torch.ones(1, dtype=torch.float64)
    else:
        second["model"]["weight"].fill_(float("nan"))
    paths = [tmp_path / "one.pt", tmp_path / "two.pt"]
    for path, saved in zip(paths, [first, second], strict=True):
        save_checkpoint(path, saved)
    with pytest.raises(ValueError, match="mismatch|nonfinite"):
        average_checkpoints(paths, tmp_path / "average.pt")


def test_average_rejects_duplicate_members_and_source_overwrite(tmp_path):
    one = tmp_path / "one.pt"
    save_checkpoint(one, checkpoint(1, 0.7))
    with pytest.raises(ValueError, match="Duplicate checkpoint paths"):
        average_checkpoints([one, one], tmp_path / "average.pt")
    for destination in [one, tmp_path / "best-model.pt", tmp_path / "last.pt"]:
        with pytest.raises(ValueError, match="preserve source checkpoints"):
            average_checkpoints([one], destination)


@pytest.mark.parametrize("score", [None, float("nan"), float("inf")])
def test_retention_rejects_undefined_macro_auc(tmp_path, score):
    saved = checkpoint(1, score)
    with pytest.raises(ValueError, match="finite validation macro_auc"):
        retain_top_checkpoint(tmp_path, saved, [record(saved)], 2)
    assert not list(tmp_path.iterdir())
