from __future__ import annotations

import numpy as np

from grid_transform.transform_helpers import apply_transform


def build_resize_affine(reference_shape: tuple[int, int], target_shape: tuple[int, int]) -> dict[str, np.ndarray]:
    ref_h, ref_w = reference_shape
    tgt_h, tgt_w = target_shape
    sx = tgt_w / ref_w
    sy = tgt_h / ref_h
    return {
        "A": np.array([[sx, 0.0], [0.0, sy]], dtype=float),
        "t": np.array([0.0, 0.0], dtype=float),
    }


def resize_inverse_mapping(reference_shape: tuple[int, int], target_shape: tuple[int, int]):
    ref_h, ref_w = reference_shape
    tgt_h, tgt_w = target_shape
    sx = tgt_w / ref_w
    sy = tgt_h / ref_h

    def mapping(target_pts):
        pts = np.asarray(target_pts, dtype=float)
        source = pts.copy()
        source[:, 0] /= sx
        source[:, 1] /= sy
        return source

    return mapping


def transform_reference_contours(reference_contours: dict[str, np.ndarray], affine: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {
        name: np.asarray(apply_transform(affine, np.asarray(points, dtype=float)), dtype=float)
        for name, points in reference_contours.items()
    }
