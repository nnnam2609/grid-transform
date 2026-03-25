from __future__ import annotations

import numpy as np

from grid_transform.apps.method4_transform import (
    apply_tps,
    apply_transform,
    build_step1_anchors,
    build_step2_controls,
    estimate_affine,
    extract_true_landmarks,
    fit_tps,
    resample_polyline,
)


DEFAULT_ARTICULATORS = (
    "incisior-hard-palate",
    "soft-palate-midline",
    "soft-palate",
    "tongue",
    "mandible-incisior",
    "pharynx",
)

DEFAULT_SMOOTHING_PASSES = 3
CONTOUR_SMOOTHING_PASSES = {
    "tongue": 5,
    "soft-palate": 4,
    "soft-palate-midline": 4,
    "pharynx": 4,
}


def resolve_common_articulators(
    target_contours: dict,
    source_contours: dict,
    requested: str | None = None,
    *,
    defaults=DEFAULT_ARTICULATORS,
) -> list[str]:
    """Resolve a stable shared articulator list across two contour dictionaries."""
    common = sorted(set(target_contours) & set(source_contours))
    if requested:
        requested_names = [name.strip() for name in requested.split(",") if name.strip()]
    else:
        requested_names = list(defaults)

    names = [name for name in requested_names if name in common]
    if not names:
        raise ValueError(f"No valid common articulators found. Common keys: {common}")
    return names


def build_two_step_transform(source_grid, target_grid):
    """Build the axis-first affine + TPS mapping used across the experiments."""
    lm_src = extract_true_landmarks(source_grid)
    lm_tgt = extract_true_landmarks(target_grid)

    step1_src, step1_tgt, step1_labels = build_step1_anchors(lm_src, lm_tgt)
    step1_affine = estimate_affine(step1_src, step1_tgt)
    step1_lm = {
        name: (None if value is None else apply_transform(step1_affine, value))
        for name, value in lm_src.items()
    }

    step2_src, step2_tgt, step2_labels = build_step2_controls(step1_lm, lm_tgt)
    step2_tps = fit_tps(step2_src, step2_tgt, smoothing=0.0)

    def apply_two_step(pts):
        affine_pts = apply_transform(step1_affine, pts)
        return apply_tps(step2_tps, affine_pts)

    return {
        "step1_affine": step1_affine,
        "step1_labels": step1_labels,
        "step2_labels": step2_labels,
        "apply_two_step": apply_two_step,
    }


def transform_contours(contours: dict, mapping_fn, names) -> dict:
    """Apply a point mapping function to selected contours."""
    return {
        name: np.asarray(mapping_fn(np.asarray(contours[name], dtype=float)), dtype=float)
        for name in names
    }


def smooth_open_contour(points: np.ndarray, passes: int = DEFAULT_SMOOTHING_PASSES) -> np.ndarray:
    """Smooth one open contour while keeping its endpoints fixed."""
    points = np.asarray(points, dtype=float)
    if len(points) < 5 or passes <= 0:
        return points.copy()

    smoothed = resample_polyline(points, n=len(points))
    original = smoothed.copy()
    kernel = np.array([1.0, 4.0, 6.0, 4.0, 1.0], dtype=float)
    kernel /= np.sum(kernel)

    for _ in range(passes):
        for dim in range(2):
            padded = np.pad(smoothed[:, dim], (2, 2), mode="edge")
            filtered = np.convolve(padded, kernel, mode="valid")
            smoothed[1:-1, dim] = filtered[1:-1]
        smoothed[0] = original[0]
        smoothed[-1] = original[-1]

    return smoothed


def smooth_transformed_contours(contours: dict) -> dict:
    """Apply contour-specific smoothing after geometric mapping."""
    return {
        name: smooth_open_contour(points, passes=CONTOUR_SMOOTHING_PASSES.get(name, DEFAULT_SMOOTHING_PASSES))
        for name, points in contours.items()
    }
