from pathlib import Path

import numpy as np
import pytest
from pydicom.dataset import Dataset, FileDataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, MRImageStorage, RLELossless, generate_uid

from rsna_knee_abnormality_detection.imaging import DicomSeriesError, load_dicom_series, prepare_volume


def write_dicom(path: Path, pixels=None, *, position=(0, 0, 0), orientation=(1, 0, 0, 0, 1, 0),
                spacing=(1, 1), instance=1, photometric="MONOCHROME2", slope=1, intercept=0,
                sop_uid=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    if pixels is None:
        pixels = np.arange(12).reshape(3, 4)
    pixels = np.asarray(pixels, dtype=np.uint16)
    sop_uid = sop_uid or generate_uid()
    meta = FileMetaDataset()
    meta.TransferSyntaxUID = ExplicitVRLittleEndian
    meta.MediaStorageSOPClassUID = MRImageStorage
    meta.MediaStorageSOPInstanceUID = sop_uid
    ds = FileDataset(str(path), {}, file_meta=meta, preamble=b"\0" * 128)
    ds.SOPClassUID = MRImageStorage
    ds.SOPInstanceUID = sop_uid
    ds.Rows, ds.Columns = pixels.shape
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = photometric
    ds.BitsAllocated, ds.BitsStored, ds.HighBit, ds.PixelRepresentation = 16, 16, 15, 0
    ds.PixelData = pixels.tobytes()
    ds.RescaleSlope, ds.RescaleIntercept = slope, intercept
    if position is not None:
        ds.ImagePositionPatient = list(position)
    if orientation is not None:
        ds.ImageOrientationPatient = list(orientation)
    if spacing is not None:
        ds.PixelSpacing = list(spacing)
    if instance is not None:
        ds.InstanceNumber = instance
    ds.save_as(path, enforce_file_format=True)
    return ds


def test_physical_order_overrides_filenames_and_instance_numbers(tmp_path):
    for name, z, instance, base in [("a", 4, 1, 40), ("b", 0, 3, 0), ("c", 2, 2, 20)]:
        write_dicom(tmp_path / f"{name}.dcm", np.arange(6).reshape(2, 3) + base,
                    position=(0, 0, z), instance=instance)
    images, metadata = load_dicom_series(tmp_path, "Axial")
    assert [image[0, 0] for image in images] == [0, 20, 40]
    assert metadata["ordering"] == "physical_position"
    assert metadata["slice_positions_mm"] == [0, 2, 4]
    assert metadata["slice_spacing_mm"] == 2


def test_in_plane_orientation_changes_pixels_and_origin_consistently(tmp_path):
    pixels = np.arange(6).reshape(2, 3)
    write_dicom(tmp_path / "one.dcm", pixels, orientation=(-1, 0, 0, 0, -1, 0),
                position=(10, 20, 0), spacing=(2, .5))
    images, metadata = load_dicom_series(tmp_path, "Axial")
    np.testing.assert_array_equal(images[0], pixels[::-1, ::-1])
    np.testing.assert_allclose(metadata["oriented_image_positions_lps"][0], [9, 18, 0])
    np.testing.assert_allclose(metadata["oriented_orientations_lps"][0], [1, 0, 0, 0, 1, 0])


def test_transpose_swaps_spacing_for_sagittal(tmp_path):
    pixels = np.arange(6).reshape(2, 3)
    write_dicom(tmp_path / "one.dcm", pixels, orientation=(0, 0, -1, 0, 1, 0), spacing=(2, .5))
    images, metadata = load_dicom_series(tmp_path, "Sagittal")
    np.testing.assert_array_equal(images[0], pixels.T)
    assert metadata["pixel_spacings_mm"] == [[.5, 2.]]


def test_rescale_and_monochrome1_inversion(tmp_path):
    pixels = np.arange(6).reshape(2, 3)
    write_dicom(tmp_path / "one.dcm", pixels, photometric="MONOCHROME1", slope=2, intercept=10)
    images, metadata = load_dicom_series(tmp_path, "Axial")
    np.testing.assert_array_equal(images[0], 20 - pixels * 2)
    assert metadata["monochrome1_frames_inverted"] == 1


def test_padding_preserves_full_image_and_physical_aspect():
    image = np.arange(32, dtype=np.float32).reshape(4, 8) + 1
    volume, metadata = prepare_volume([image], size=16)
    assert volume.dtype == np.uint16
    assert volume.shape == (1, 16, 16)
    assert metadata["resize_transforms"][0]["resized_shape"] == [8, 16]
    assert np.count_nonzero(volume[0, :4]) == 0
    assert np.count_nonzero(volume[0, 12:]) == 0
    assert volume[0, 11, 15] == 65535
    square, physical = prepare_volume([image], size=16, pixel_spacings=[(2, 1)])
    assert physical["resize_transforms"][0]["resized_shape"] == [16, 16]
    assert square.shape == (1, 16, 16)


def test_mixed_sizes_are_retained_and_padded(tmp_path):
    write_dicom(tmp_path / "a.dcm", np.arange(8).reshape(2, 4), position=(0, 0, 0))
    write_dicom(tmp_path / "b.dcm", np.arange(16).reshape(4, 4), position=(0, 0, 2))
    images, metadata = load_dicom_series(tmp_path, "Axial")
    assert [a.shape for a in images] == [(2, 4), (4, 4)]
    assert any("mixed_frame_sizes" in warning for warning in metadata["warnings"])
    volume, processing = prepare_volume(images, 8, metadata["pixel_spacings_mm"])
    assert volume.shape == (2, 8, 8)
    assert processing["resize_transforms"][0]["padding_top_left"] == [2, 0]


def test_identical_duplicates_removed_but_different_echoes_rejected(tmp_path):
    pixels = np.arange(6).reshape(2, 3)
    write_dicom(tmp_path / "a.dcm", pixels)
    write_dicom(tmp_path / "b.dcm", pixels)
    images, metadata = load_dicom_series(tmp_path, "Axial")
    assert len(images) == 1
    assert metadata["warnings"] == ["identical_duplicate_slices_removed:1"]
    write_dicom(tmp_path / "b.dcm", pixels + 1)
    with pytest.raises(DicomSeriesError, match="Duplicate slice position"):
        load_dicom_series(tmp_path, "Axial")


def test_missing_geometry_fallback_is_explicit(tmp_path):
    write_dicom(tmp_path / "a.dcm", np.arange(6).reshape(2, 3) + 10,
                position=None, orientation=None, spacing=None, instance=2)
    write_dicom(tmp_path / "b.dcm", position=None, orientation=None, spacing=None, instance=1)
    images, metadata = load_dicom_series(tmp_path, "Axial")
    assert images[0][0, 0] == 0
    assert metadata["ordering"] == "instance_number"
    assert not metadata["orientation_standardized"]
    assert "slice_order_fallback:InstanceNumber" in metadata["warnings"]
    assert "missing_pixel_spacing:assumed_square_pixels_for_resize" in metadata["warnings"]


def test_mixed_orientation_and_bad_pixels_do_not_silently_drop_slices(tmp_path):
    write_dicom(tmp_path / "a.dcm")
    write_dicom(tmp_path / "b.dcm", orientation=(0, 1, 0, 0, 0, -1))
    with pytest.raises(DicomSeriesError, match="Mixed slice orientations"):
        load_dicom_series(tmp_path, "Axial")
    (tmp_path / "b.dcm").write_bytes(b"broken DICOM")
    with pytest.raises(DicomSeriesError, match="Could not decode b.dcm"):
        load_dicom_series(tmp_path, "Axial")


def test_enhanced_multiframe_uses_per_frame_positions(tmp_path):
    path = tmp_path / "multi.dcm"
    ds = write_dicom(path)
    pixels = np.stack([np.arange(12).reshape(3, 4) + 20, np.arange(12).reshape(3, 4)])
    ds.NumberOfFrames = 2
    ds.PixelData = pixels.astype(np.uint16).tobytes()
    groups = []
    for z in [3, 0]:
        group, position = Dataset(), Dataset()
        position.ImagePositionPatient = [0, 0, z]
        group.PlanePositionSequence = [position]
        groups.append(group)
    ds.PerFrameFunctionalGroupsSequence = groups
    ds.save_as(path, enforce_file_format=True)
    images, metadata = load_dicom_series(tmp_path, "Axial")
    assert [a[0, 0] for a in images] == [0, 20]
    assert metadata["slice_positions_mm"] == [0, 3]


def test_compressed_pixels_are_decoded_without_changing_values(tmp_path):
    path = tmp_path / "compressed.dcm"
    pixels = np.arange(60).reshape(6, 10)
    ds = write_dicom(path, pixels)
    ds.compress(RLELossless)
    ds.save_as(path, enforce_file_format=True)
    images, metadata = load_dicom_series(tmp_path, "Axial")
    np.testing.assert_array_equal(images[0], pixels)
    assert metadata["transfer_syntaxes"] == [str(RLELossless)]


def test_pixel_padding_is_excluded_from_normalization(tmp_path):
    path = tmp_path / "padding.dcm"
    pixels = np.zeros((10, 10), dtype=np.uint16)
    pixels[4:6, :] = np.arange(100, 120).reshape(2, 10)
    ds = write_dicom(path, pixels)
    ds.PixelPaddingValue = 0
    ds.save_as(path, enforce_file_format=True)
    images, _ = load_dicom_series(tmp_path, "Axial")
    assert np.isnan(images[0][0, 0])
    volume, processing = prepare_volume(images, size=10)
    assert processing["normalization"]["low"] > 99
    assert np.all(volume[0, :4] == 0)
