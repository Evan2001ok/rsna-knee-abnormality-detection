import hashlib
import importlib.util
import io
import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from rsna_knee_abnormality_detection.constants import STUDY_ID, TARGETS
from rsna_knee_abnormality_detection.labels import read_study_csv

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "prepare_raptor_labels.py"
SPEC = importlib.util.spec_from_file_location("prepare_raptor_labels", SCRIPT)
raptor = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(raptor)


@pytest.fixture
def inputs(tmp_path, monkeypatch):
    """Use the real split sizes with small synthetic targets and exact string IDs."""
    ids = [f"{i:05d}" for i in range(4407)]
    values = np.full((4407, len(TARGETS)), np.nan)
    values[-58:] = np.arange(58 * len(TARGETS)).reshape(58, -1) % 2
    train = pd.DataFrame(values, columns=TARGETS)
    train.insert(0, STUDY_ID, ids)
    baseline = train.copy()
    baseline.loc[53:4348, TARGETS] = 0
    baseline.loc[53:905, "ACL"] = np.nan
    baseline["label_source"] = ["missing"] * 53 + ["report"] * 4296 + ["expert"] * 58
    source_values = 0.05 + (np.arange(4349 * len(TARGETS)).reshape(4349, -1) % 19) * 0.05
    source = pd.DataFrame(source_values, columns=TARGETS)
    source.insert(0, STUDY_ID, ids[:4349])
    source = source.iloc[::-1]  # Source row order must never affect the study join.
    splits = pd.DataFrame({
        STUDY_ID: ids,
        "fold": [-2] * 53 + [0] * 853 + [1 + i % 4 for i in range(3443)] + [-1] * 58,
        "split": ["unlabeled"] * 53 + ["train"] * 4296 + ["expert_holdout"] * 58,
        "group_id": ids,
    })
    paths = {}
    for name, frame in [("train", train), ("baseline", baseline), ("source", source), ("splits", splits)]:
        paths[name] = tmp_path / name / f"{name}.csv"
        paths[name].parent.mkdir()
        frame.to_csv(paths[name], index=False)
    monkeypatch.setattr(raptor, "EXPECTED_SOURCE_SHA256", hashlib.sha256(paths["source"].read_bytes()).hexdigest())
    paths["output"] = tmp_path / "derived"
    return paths


def prepare(paths):
    return raptor.prepare_raptor_labels(
        paths["source"], paths["train"], paths["baseline"], paths["splits"], paths["output"]
    )


def rewrite(paths, name, change):
    frame = pd.read_csv(paths[name], dtype={STUDY_ID: str, "group_id": str})
    change(frame).to_csv(paths[name], index=False)


def test_soft_targets_preserve_experts_membership_and_original_evaluation_labels(inputs):
    original_bytes = {name: path.read_bytes() for name, path in inputs.items() if name != "output"}
    output = prepare(inputs)
    labels = read_study_csv(output).set_index(STUDY_ID)
    source = read_study_csv(inputs["source"]).set_index(STUDY_ID)
    train = read_study_csv(inputs["train"]).set_index(STUDY_ID)
    assert list(labels.index) == list(train.index)
    np.testing.assert_array_equal(labels.loc[source.index, TARGETS], source[TARGETS])
    np.testing.assert_array_equal(labels.loc[train.index[-58:], TARGETS], train[TARGETS].iloc[-58:])
    assert labels["label_source"].value_counts().to_dict() == {"report": 4349, "expert": 58}
    assert all(path.read_bytes() == original_bytes[name] for name, path in inputs.items() if name != "output")

    splits = read_study_csv(inputs["splits"], require_targets=False).set_index(STUDY_ID)
    excluded = splits.index[splits["split"].eq("unlabeled")]
    assert len(excluded) == 53
    assert labels.loc[excluded, TARGETS].notna().all().all()
    assert not set(excluded) & set(splits.index[splits["split"].eq("train")])
    baseline = read_study_csv(inputs["baseline"]).set_index(STUDY_ID)
    val_ids = splits.index[splits["fold"].eq(0)]
    assert baseline.loc[val_ids, "ACL"].isna().all()
    assert labels.loc[val_ids, "ACL"].notna().all()

    audit = json.loads((output.parent / "label_provenance.json").read_text())
    assert audit["source"]["dataset_version"] == 3
    assert audit["source"]["pinned_csv_sha256"] == hashlib.sha256(original_bytes["source"]).hexdigest()
    assert audit["source_copy"]["sha256"] == audit["source"]["pinned_csv_sha256"]
    assert audit["labels_csv"]["sha256"] == hashlib.sha256(output.read_bytes()).hexdigest()
    assert {name: record["studies"] for name, record in audit["partitions"].items()} == raptor.EXPECTED_PARTITIONS
    assert audit["configuration"]["data.validation_labels"] == str(inputs["baseline"])
    first_labels = output.read_bytes()
    prepare(inputs)
    assert output.read_bytes() == first_labels


@pytest.mark.parametrize("change,match", [
    (lambda frame: frame.iloc[1:], "exactly training studies minus"),
    (lambda frame: frame.assign(StudyInstanceUID=["04349", *frame[STUDY_ID].iloc[1:]]), "exactly training studies minus"),
    (lambda frame: pd.concat([frame, frame.iloc[:1]], ignore_index=True), "duplicate study IDs"),
    (lambda frame: frame.assign(ACL=np.nan), "must be finite"),
    (lambda frame: frame.assign(ACL=1.1), "between 0 and 1"),
    (lambda frame: frame.assign(ACL=np.inf), "between 0 and 1"),
    (lambda frame: frame.assign(ACL=0.33), "pinned Raptor dataset"),
])
def test_invalid_or_unpinned_public_sources_fail_before_writing(inputs, change, match):
    rewrite(inputs, "source", change)
    with pytest.raises(ValueError, match=match):
        prepare(inputs)
    assert not inputs["output"].exists()


@pytest.mark.parametrize("name,change,match", [
    ("baseline", lambda frame: frame.iloc[1:], "exactly the competition"),
    ("baseline", lambda frame: frame.assign(ACL=0.25), "expert target values differ"),
    ("baseline", lambda frame: frame.assign(label_source="report"), "expert study IDs"),
    ("splits", lambda frame: frame.assign(fold=frame["fold"].mask(frame["fold"].eq(-1), 1)), "fold -1"),
    ("splits", lambda frame: frame.assign(fold=frame["fold"].mask(frame["fold"].eq(0), 1)), "partition sizes changed"),
    ("splits", lambda frame: frame.assign(group_id="same-patient"), "patient group crosses"),
])
def test_baseline_identity_and_holdout_changes_are_rejected(inputs, name, change, match):
    rewrite(inputs, name, change)
    with pytest.raises(ValueError, match=match):
        prepare(inputs)
    assert not inputs["output"].exists()


def test_output_cannot_overwrite_baseline_inputs(inputs):
    before = inputs["baseline"].read_bytes()
    inputs["output"] = inputs["baseline"].parent
    with pytest.raises(ValueError, match="must not overwrite"):
        prepare(inputs)
    assert inputs["baseline"].read_bytes() == before


def test_download_verifies_pinned_csv_before_writing(tmp_path, monkeypatch):
    raw = b"the pinned public CSV snapshot\r\n"
    archive = io.BytesIO()
    with zipfile.ZipFile(archive, "w") as stream:
        stream.writestr(raptor.SOURCE_FILENAME, raw)
    requests = []

    def get(request, timeout):
        requests.append(request.full_url)
        return io.BytesIO(archive.getvalue())

    monkeypatch.setattr(raptor.urllib.request, "urlopen", get)
    output = tmp_path / "source" / raptor.SOURCE_FILENAME
    with pytest.raises(ValueError, match="does not match"):
        raptor.download_source(output)
    assert not output.exists()
    monkeypatch.setattr(raptor, "EXPECTED_SOURCE_SHA256", hashlib.sha256(raw).hexdigest())
    assert raptor.download_source(output).read_bytes() == raw
    assert all("datasetVersionNumber=3" in url for url in requests)
    count = len(requests)
    raptor.download_source(output)
    assert len(requests) == count
