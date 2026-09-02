from __future__ import annotations

import numpy as np
import pytest

from grid_transform.apps.promote_final_imagej_rois_to_gtgrd import (
    SCALE,
    identify_display_label,
    native_matrix_to_triplet,
    validate_proper_rigid,
)


def test_native_matrix_to_triplet_scales_only_translation() -> None:
    matrix = np.asarray([[0.0, -1.0, 2.0], [1.0, 0.0, -3.0]])
    converted = native_matrix_to_triplet(matrix)

    np.testing.assert_allclose(converted[:, :2], matrix[:, :2])
    np.testing.assert_allclose(converted[:, 2], matrix[:, 2] * SCALE)


def test_validate_proper_rigid_rejects_scale() -> None:
    with pytest.raises(ValueError, match="not proper rigid"):
        validate_proper_rigid(np.asarray([[1.01, 0.0, 0.0], [0.0, 1.01, 0.0]]))


def test_identify_display_label_accepts_prefixed_imagej_member() -> None:
    assert identify_display_label("010_P10_soft-palate-midline.roi") == "soft-palate-midline"
    assert identify_display_label("001_ASD2_C6.roi") == "c6"
