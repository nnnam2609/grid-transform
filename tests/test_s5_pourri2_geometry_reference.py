from __future__ import annotations

import zipfile
from pathlib import Path

import numpy as np
from roifile import ImagejRoi, ROI_TYPE

from grid_transform.apps.build_s5_pourri2_geometry_reference import (
    canonicalize_contours,
    load_reviewed_lower_incisor,
    write_annotation_zip,
)


def test_load_reviewed_lower_incisor_uses_exact_50_point_prototype(
    tmp_path: Path,
) -> None:
    path = (
        tmp_path
        / "cases"
        / "P8_S5_F0792"
        / "contours_npy"
        / "lower-incisor.npy"
    )
    path.parent.mkdir(parents=True)
    expected = np.column_stack(
        [np.linspace(30.0, 70.0, 50), np.linspace(40.0, 90.0, 50)]
    )
    np.save(path, expected)

    actual, source_path = load_reviewed_lower_incisor(tmp_path, "P8", 792)

    assert source_path == path
    assert np.array_equal(actual, expected)


def test_candidate_roi_topology_and_reviewed_lower_round_trip(
    tmp_path: Path,
) -> None:
    lower = np.column_stack(
        [np.linspace(20.0, 40.0, 50), np.linspace(50.0, 80.0, 50)]
    )
    native = {
        "lower-incisor": lower,
        "tongue": lower + np.array([10.0, -5.0]),
    }
    cervical = {
        f"c{index}": np.array(
            [[200.0 + index, 300.0], [205.0 + index, 304.0], [201.0 + index, 309.0]]
        )
        for index in range(1, 7)
    }
    canonical, _ = canonicalize_contours(native, cervical)
    path = tmp_path / "case.zip"

    write_annotation_zip(path, "case", canonical)

    with zipfile.ZipFile(path) as archive:
        lower_roi = ImagejRoi.frombytes(
            archive.read("case_mandible-incisior.roi")
        )
        c1_roi = ImagejRoi.frombytes(archive.read("case_c1.roi"))
    expected_lower = lower * (480.0 / 136.0)
    assert lower_roi.roitype == ROI_TYPE.POLYLINE
    assert c1_roi.roitype == ROI_TYPE.FREEHAND
    assert np.allclose(lower_roi.coordinates(), expected_lower, atol=2e-5)
