import numpy as np
import pandas as pd
import pytest
import torch
from torch.utils.data import DataLoader

from rsna_knee_abnormality_detection.constants import STUDY_ID, TARGETS
from rsna_knee_abnormality_detection.dataset import KneeDataset


@pytest.fixture
def prepared(tmp_path):
    rows = []
    # The slice-dependent intensities make coverage and neighbour selection
    # visible without relying on how random numbers happen to be generated.
    for study, series, plane, depth, fluid, fat in [
        ("study-a", "sag-a", "Sagittal", 21, True, True),
        ("study-a", "sag-b", "Sagittal", 17, False, False),
        ("study-a", "ax-a", "Axial", 19, True, None),
        ("study-b", "cor-b", "Coronal", 15, None, False),
    ]:
        volume = np.broadcast_to(
            np.linspace(0, 65535, depth).astype(np.uint16)[:, None, None], (depth, 32, 32)
        ).copy()
        np.save(tmp_path / f"{series}.npy", volume)
        rows.append({
            STUDY_ID: study, "SeriesInstanceUID": series, "Anatomical_Plane": plane,
            "Fluid_Sensitive": fluid, "Fat_Suppression": fat,
            "cache_path": f"{series}.npy", "num_slices": depth,
        })
    manifest = tmp_path / "manifest.csv"
    pd.DataFrame(rows).to_csv(manifest, index=False)
    studies = pd.DataFrame([
        {STUDY_ID: "study-a", "ACL": 1.0, "MCL": 0.0, "Effusion": 0.7},
        {STUDY_ID: "study-b", "ACL": 0.0, "MCL": 1.0},
    ])
    config = {"image_size": 32, "slices_per_series": 4, "augmentation": {"enabled": False}}
    return manifest, studies, config


def test_complete_study_layout_labels_and_missing_views(prepared):
    dataset = KneeDataset(*prepared)
    item = dataset[0]
    assert item["images"].shape == (3, 4, 3, 32, 32)
    assert item["series_mask"].tolist() == [True, False, True]
    assert item["metadata"][0].tolist() == [1, 0, 0, 1, 1]
    assert item["metadata"][2].tolist() == [0, 0, 1, 1, -1]
    assert torch.count_nonzero(item["images"][1]) == 0
    assert item["target_mask"].sum() == 3
    assert item["target_mask"][TARGETS.index("Medial Meniscus")].item() is False
    assert item["targets"][TARGETS.index("Medial Meniscus")] == 0
    assert item["targets"][TARGETS.index("Effusion")] == pytest.approx(0.7)
    assert item["study_id"] == "study-a"
    assert torch.allclose(item["positions"][0], torch.tensor([0, 0.35, 0.65, 1]))
    assert torch.allclose(item["images"][0, :, 1, 0, 0], item["positions"][0], atol=2e-5)
    # At the first slice, the preceding neighbour is clamped to the same slice.
    assert torch.equal(item["images"][0, 0, 0], item["images"][0, 0, 1])
    assert item["images"][0, 0, 2, 0, 0] > item["images"][0, 0, 1, 0, 0]
    assert torch.isfinite(item["images"]).all()


def test_validation_is_fixed_and_inference_can_have_no_targets(prepared):
    manifest, studies, config = prepared
    dataset = KneeDataset(manifest, studies[[STUDY_ID]], {**config, "augmentation": {"enabled": True}})
    before = dataset[0]
    dataset.set_epoch(100)
    after = dataset[0]
    assert torch.equal(before["images"], after["images"])
    assert torch.equal(before["positions"], after["positions"])
    assert not before["target_mask"].any()


def test_training_randomness_is_reproducible_per_study_and_epoch(prepared):
    manifest, studies, config = prepared
    config = {**config, "augmentation": {"enabled": True}}
    first = KneeDataset(manifest, studies, config, training=True, seed=19)
    second = KneeDataset(manifest, studies, config, training=True, seed=19)
    a = first[0]
    assert torch.equal(a["images"], first[0]["images"])
    assert torch.equal(a["images"], second[0]["images"])
    first.set_epoch(1)
    second.set_epoch(1)
    b = first[0]
    assert not torch.equal(a["images"], b["images"])
    assert torch.equal(b["images"], second[0]["images"])
    assert not torch.equal(a["positions"], b["positions"])
    assert b["images"].min() >= 0 and b["images"].max() <= 1
    assert torch.count_nonzero(b["images"][1]) == 0


def test_geometric_transform_matches_across_neighbor_channels(prepared):
    manifest, studies, config = prepared
    # Each slice contains the same spatial pattern. Independent transforms
    # would make their edges disagree and invent an inconsistent 3D structure.
    pattern = np.zeros((21, 32, 32), dtype=np.uint16)
    pattern[:, 5:20, 9:23] = 65535
    np.save(manifest.parent / "sag-a.npy", pattern)
    augmentation = {"enabled": True, "contrast": 0, "brightness": 0, "noise_std": 0}
    dataset = KneeDataset(manifest, studies, {**config, "augmentation": augmentation}, training=True)
    groups = dataset[0]["images"][0]
    assert torch.equal(groups[0, 0], groups[-1, -1])
    assert torch.equal(groups[0, 0], groups[0, 1])


def test_multiple_slots_and_batch_collation(prepared):
    manifest, studies, config = prepared
    dataset = KneeDataset(manifest, studies, {**config, "series_per_plane": 2})
    batch = next(iter(DataLoader(dataset, batch_size=2)))
    assert batch["images"].shape == (2, 6, 4, 3, 32, 32)
    assert batch["series_mask"][0].tolist() == [True, True, False, False, True, False]
    assert batch["series_mask"][1].tolist() == [False, False, True, False, False, False]
    assert batch["study_id"] == ["study-a", "study-b"]


def test_epoch_reaches_persistent_workers(prepared):
    dataset = KneeDataset(*prepared, training=True)
    loader = DataLoader(dataset, batch_size=2, num_workers=1, persistent_workers=True, multiprocessing_context="spawn")
    first = next(iter(loader))["positions"]
    dataset.set_epoch(2)
    second = next(iter(loader))["positions"]
    assert not torch.equal(first, second)
    del loader


def test_rejects_study_without_any_usable_images(prepared):
    manifest, _, config = prepared
    with pytest.raises(ValueError, match="No usable MRI series"):
        KneeDataset(manifest, pd.DataFrame([{STUDY_ID: "missing"}]), config)


def test_rejects_invalid_targets_and_cache_depth(prepared):
    manifest, studies, config = prepared
    bad = studies.copy()
    bad.loc[0, "ACL"] = 2
    with pytest.raises(ValueError, match="between zero and one"):
        KneeDataset(manifest, bad, config)
    dataset = KneeDataset(manifest, studies, config)
    np.save(manifest.parent / "sag-a.npy", np.zeros((3, 32, 32), dtype=np.uint16))
    with pytest.raises(ValueError, match="depth differs"):
        dataset[0]
