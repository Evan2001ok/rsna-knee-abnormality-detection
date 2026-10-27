import json

import numpy as np
import pandas as pd
import pytest
from test_raptor_data import make_raw

from rsna_knee_abnormality_detection.anchors import centred_box, project_to_pixel, read_anchor
from rsna_knee_abnormality_detection.constants import SERIES_ID, STUDY_ID
from rsna_knee_abnormality_detection.raptor_data import (
    CLOSEUP_SLICES,
    RAPTOR_DENSE_384_VERSION,
    RAPTOR_DENSE_KNEE_384_VERSION,
    RAPTOR_DENSE_KNEE_CLOSEUP_384_VERSION,
    RAPTOR_VARIANTS,
    RaptorDataset,
    prepare_raptor_data,
    validate_raptor_manifest,
)


@pytest.mark.parametrize("orientation", [(0, 1, 0, 0, 0, -1), (1, 0, 0, 0, 0, -1), (1, 0, 0, 0, 1, 0),
                                         (-1, 0, 0, 0, 1, 0), (np.cos(.3), np.sin(.3), 0, 0, 0, -1)])
def test_projection_inverts_the_dicom_pixel_to_patient_mapping(orientation):
    o = np.asarray(orientation, float)
    position, spacing = np.array([-80., 40., 25.]), (0.4, 0.5)   # (row spacing, column spacing), deliberately unequal
    row, column, off_plane = 37.0, 121.0, 6.0
    point = position + o[:3] * column * spacing[1] + o[3:] * row * spacing[0] + np.cross(o[:3], o[3:]) * off_plane
    got = project_to_pixel(point, o, position, spacing)
    np.testing.assert_allclose(got, (row, column, off_plane), atol=1e-9)


def test_centred_box_matches_the_plain_centre_crop_and_stays_inside():
    for height, width, size in ((16, 20, 14), (17, 20, 14), (512, 448, 350)):
        assert centred_box((height - 1) / 2, (width - 1) / 2, size, height, width) == ((height - size) // 2, (width - size) // 2)
    assert centred_box(-50, 1e4, 100, 300, 400) == (0, 300)
    assert centred_box(150, 200, 100, 300, 400) == (100, 150)
    with pytest.raises(ValueError):
        centred_box(0, 0, 500, 300, 400)


def test_read_anchor_handles_missing_and_failed_records(tmp_path):
    assert read_anchor(tmp_path, "s")["joint_lps"] is None
    (tmp_path / "a.json").write_text(json.dumps({"joint_lps": [1, 2, 3], "method": "skm_tea_joint"}))
    (tmp_path / "b.json").write_text(json.dumps({"error": "boom"}))
    assert read_anchor(tmp_path, "a") == {"joint_lps": [1.0, 2.0, 3.0], "patella_lps": None, "method": "skm_tea_joint"}
    assert read_anchor(tmp_path, "b") == {"joint_lps": None, "patella_lps": None, "method": "error"}


def test_knee_cache_equals_the_plain_cache_when_the_anchor_is_the_image_centre(tmp_path):
    # make_raw writes 16x20 slices at 10 mm spacing: a 140 mm box is 14 px, so the plain crop is rows 1-14, columns 3-16.
    make_raw(tmp_path / "raw", studies=("centre", "shifted", "none"))
    plain = prepare_raptor_data(tmp_path / "raw", tmp_path / "plain", workers=1, version=RAPTOR_DENSE_384_VERSION)
    anchors = tmp_path / "anchors"
    anchors.mkdir()
    # The Axial series of make_raw has orientation (1,0,0,0,1,0) at z = 2*index: pixel (row 7.5, column 9.5) is
    # x = 95 mm, y = 75 mm. Putting the joint there keeps the axial box at the image centre; every other plane
    # sees the same point at another place, so only the axial slot is compared.
    (anchors / "centre.json").write_text(json.dumps({"joint_lps": [95., 75., 4.], "method": "test"}))
    (anchors / "shifted.json").write_text(json.dumps({"joint_lps": [0., 0., 4.], "method": "test"}))
    knee = prepare_raptor_data(tmp_path / "raw", tmp_path / "knee", workers=1, version=RAPTOR_DENSE_KNEE_384_VERSION,
                               anchors_dir=anchors)
    validate_raptor_manifest(knee, version=RAPTOR_DENSE_KNEE_384_VERSION, verify_checksums=True)
    plain_meta = json.loads((tmp_path / "plain/studies/centre.json").read_text())
    knee_meta = json.loads((tmp_path / "knee/studies/centre.json").read_text())
    axial = [slot for slot in knee_meta["slots"] if slot["plane"] == "Axial"][0]
    plain_axial = [slot for slot in plain_meta["slots"] if slot["plane"] == "Axial"][0]
    assert [(c["top"], c["left"]) for c in axial["crops"]] == [(c["top"], c["left"]) for c in plain_axial["crops"]]
    assert all(c["center"] == "anchor" for c in axial["crops"])
    start, count = axial["offset"], len(axial["selected_source_indices"])
    np.testing.assert_array_equal(np.load(tmp_path / "knee/studies/centre.npy")[start:start + count],
                                  np.load(tmp_path / "plain/studies/centre.npy")[start:start + count])
    # A joint at the corner pulls every axial box to the top-left edge; a study without a record keeps the centre.
    shifted = json.loads((tmp_path / "knee/studies/shifted.json").read_text())
    assert {(c["top"], c["left"]) for c in [s for s in shifted["slots"] if s["plane"] == "Axial"][0]["crops"]} == {(0, 0)}
    none = json.loads((tmp_path / "knee/studies/none.json").read_text())
    assert any(w.startswith("no_knee_anchor") for w in none["warnings"])
    assert none["anchor"] == {"joint_lps": None, "patella_lps": None, "method": "missing_record"}
    assert plain.exists()


def test_knee_variant_requires_anchors_and_others_refuse_them(tmp_path):
    make_raw(tmp_path / "raw")
    with pytest.raises(ValueError, match="anchors_dir"):
        prepare_raptor_data(tmp_path / "raw", tmp_path / "k", workers=1, version=RAPTOR_DENSE_KNEE_384_VERSION)
    with pytest.raises(ValueError, match="anchors_dir"):
        prepare_raptor_data(tmp_path / "raw", tmp_path / "p", workers=1, version=RAPTOR_DENSE_384_VERSION, anchors_dir=tmp_path)


def test_closeup_cache_adds_four_100mm_slots_from_the_main_series(tmp_path):
    # make_raw: 16x20 slices at 10 mm, so a 100 mm close-up box is 10 px. Sagittal pixel (r, c) sits at y = 10c, z = -10r;
    # coronal and axial at x = 10c. The joint below is at the sagittal image centre.
    make_raw(tmp_path / "raw", studies=("kneecap", "fallback", "none"))
    anchors = tmp_path / "anchors"
    anchors.mkdir()
    (anchors / "kneecap.json").write_text(json.dumps({"joint_lps": [-4., 95., -75.], "patella_lps": [-4., 55., -50.],
                                                      "method": "test"}))
    (anchors / "fallback.json").write_text(json.dumps({"joint_lps": [-4., 95., -75.], "method": "test"}))
    variant = RAPTOR_VARIANTS[RAPTOR_DENSE_KNEE_CLOSEUP_384_VERSION]
    assert variant["depth"] == 144 + 4 * CLOSEUP_SLICES and len(variant["closeups"]) == 4
    manifest = prepare_raptor_data(tmp_path / "raw", tmp_path / "cache", workers=1,
                                   version=RAPTOR_DENSE_KNEE_CLOSEUP_384_VERSION, anchors_dir=anchors)
    validate_raptor_manifest(manifest, version=RAPTOR_DENSE_KNEE_CLOSEUP_384_VERSION, verify_checksums=True)
    meta = json.loads((tmp_path / "cache/studies/kneecap.json").read_text())
    main, close = meta["slots"][:5], meta["slots"][5:]
    assert [s["closeup"] for s in close] == ["sagittal_kneecap", "axial_kneecap", "coronal_right", "coronal_left"]
    for slot in close:
        assert slot[SERIES_ID] == main[slot["source_slot"]][SERIES_ID]
        assert all(c["size"] == 10 and c["center"] == "anchor" for c in slot["crops"])
    sagittal = close[0]
    assert sagittal["center_source"] == "kneecap"
    # Kneecap at y = 55, z = -50: sagittal column 5.5, row 5, so the 10 px box starts at row 0 (shifted inside), column 1.
    assert {(c["top"], c["left"]) for c in sagittal["crops"]} == {(0, 1)}
    right, left = close[2], close[3]
    assert right["center_lps"][0] == -34. and left["center_lps"][0] == 26.
    assert json.loads((tmp_path / "cache/studies/fallback.json").read_text())["slots"][5]["center_source"] == "kneecap_fallback"
    none = json.loads((tmp_path / "cache/studies/none.json").read_text())
    assert all(s[SERIES_ID] is None for s in none["slots"][5:])
    assert sum(w.startswith("missing_closeup") for w in none["warnings"]) == 4
    volume = np.load(tmp_path / "cache/studies/kneecap.npy")
    assert volume.shape == (208, 384, 384) and volume[144:].any() and not np.load(tmp_path / "cache/studies/none.npy")[144:].any()


def test_raptor_slots_restricts_windows_to_the_chosen_slots(tmp_path):
    make_raw(tmp_path / "raw", studies=("kneecap",))
    anchors = tmp_path / "anchors"
    anchors.mkdir()
    (anchors / "kneecap.json").write_text(json.dumps({"joint_lps": [-4., 95., -75.], "method": "test"}))
    manifest = prepare_raptor_data(tmp_path / "raw", tmp_path / "cache", workers=1,
                                   version=RAPTOR_DENSE_KNEE_CLOSEUP_384_VERSION, anchors_dir=anchors)
    studies = pd.DataFrame({STUDY_ID: ["kneecap"]})
    config = {"raptor_variant": RAPTOR_DENSE_KNEE_CLOSEUP_384_VERSION, "image_size": 384, "cache_size": 384,
              "raptor_eval_windows": 50, "raptor_train_windows": 12, "raptor_window_sampling": "stratified"}
    every = RaptorDataset(manifest, studies, config)
    main_only = RaptorDataset(manifest, studies, {**config, "raptor_slots": [0, 1, 2, 3, 4]}, training=True)
    assert set(every._records["kneecap"]["slots"]) - {0, 1, 2, 3, 4}
    assert set(main_only._records["kneecap"]["slots"]) <= {0, 1, 2, 3, 4}
    assert main_only[0]["images"].shape[1] == 12
    with pytest.raises(ValueError, match="No windows"):
        RaptorDataset(manifest, studies, {**config, "raptor_slots": [99]})
