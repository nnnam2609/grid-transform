from __future__ import annotations

import zipfile
from pathlib import Path

import numpy as np
from roifile import ImagejRoi, ROI_TYPE

from grid_transform.apps.add_asd2_gtgrd_reference import (
    BASENAME,
    CERVICAL_LABELS,
    DYNAMIC_LABELS,
    canonicalize_reviewed_rois,
    normalize_registered_triplet,
    validate_annotation_round_trip,
    write_annotation_zip,
)


def test_registered_triplet_uses_shared_range_and_channel_order() -> None:
    frames = [np.full((136, 136), value, dtype=np.uint16) for value in (0, 100, 200)]
    triplet, intensity = normalize_registered_triplet(frames)
    assert triplet.shape == (480, 480, 3)
    assert triplet.dtype == np.uint8
    assert np.all(triplet[..., 0] == 0)
    assert np.all(triplet[..., 1] == 128)
    assert np.all(triplet[..., 2] == 255)
    assert intensity["source_min"] == 0.0
    assert intensity["source_max"] == 200.0


def test_scaled_asd2_zip_is_deterministic_and_preserves_roi_topology(
    tmp_path: Path,
) -> None:
    dynamic = np.column_stack([np.linspace(20.0, 40.0, 50), np.linspace(50.0, 80.0, 50)])
    contours = {label: dynamic + float(index) for index, label in enumerate(DYNAMIC_LABELS)}
    contours.update(
        {
            label: np.array([[70.0 + index, 50.0], [75.0 + index, 55.0], [72.0 + index, 60.0]])
            for index, label in enumerate(CERVICAL_LABELS)
        }
    )
    canonical = canonicalize_reviewed_rois(contours)
    first = tmp_path / "first.zip"
    second = tmp_path / "second.zip"
    write_annotation_zip(first, canonical)
    write_annotation_zip(second, canonical)

    assert first.read_bytes() == second.read_bytes()
    assert validate_annotation_round_trip(first, canonical) < 1e-3
    with zipfile.ZipFile(first) as archive:
        upper = ImagejRoi.frombytes(archive.read(f"{BASENAME}_incisior-hard-palate.roi"))
        c1 = ImagejRoi.frombytes(archive.read(f"{BASENAME}_c1.roi"))
    assert upper.roitype == ROI_TYPE.POLYLINE
    assert c1.roitype == ROI_TYPE.FREEHAND
    upper_native = contours["upper-incisor"]
    assert np.allclose(upper.coordinates(), upper_native * (480.0 / 136.0), atol=2e-5)
