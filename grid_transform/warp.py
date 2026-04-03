from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import map_coordinates

from grid_transform.config import TONGUE_COLOR
from grid_transform.image_utils import as_grayscale_float
from grid_transform.figures import format_frame


TARGET_COLOR = "#16a085"
SOURCE_COLOR = "#3a86ff"
WARPED_COLOR = "#ef476f"


def _as_grayscale_float(source_image) -> np.ndarray:
    return as_grayscale_float(source_image)


def precompute_inverse_warp(target_shape, inverse_mapping_fn, source_shape):
    target_h, target_w = target_shape[:2]
    yy, xx = np.meshgrid(np.arange(target_h), np.arange(target_w), indexing="ij")
    target_pts = np.column_stack([xx.ravel(), yy.ravel()])
    source_pts = np.asarray(inverse_mapping_fn(target_pts), dtype=float)

    source_x = source_pts[:, 0].reshape(target_h, target_w)
    source_y = source_pts[:, 1].reshape(target_h, target_w)
    source_h, source_w = source_shape[:2]
    mask = (
        (source_x >= 0)
        & (source_x <= source_w - 1)
        & (source_y >= 0)
        & (source_y <= source_h - 1)
    )
    return source_x, source_y, mask


def warp_array_with_precomputed_inverse_warp(source_image, source_x, source_y, valid_mask):
    source_arr = _as_grayscale_float(source_image)
    warped = map_coordinates(
        source_arr,
        [source_y, source_x],
        order=1,
        mode="constant",
        cval=0.0,
    )
    warped *= valid_mask

    return np.clip(warped, 0, 255).astype(np.uint8), valid_mask.astype(np.uint8) * 255


def warp_image_to_target_space(source_image, target_shape, inverse_mapping_fn):
    """Warp the full source image into target image space using inverse mapping."""
    source_arr = _as_grayscale_float(source_image)
    source_x, source_y, valid_mask = precompute_inverse_warp(
        target_shape,
        inverse_mapping_fn,
        source_arr.shape,
    )
    return warp_array_with_precomputed_inverse_warp(source_arr, source_x, source_y, valid_mask)


def plot_articulator_set(ax, contours, articulators, base_color, moved=False):
    """Draw articulator overlays, with tongue highlighted consistently."""
    for name in articulators:
        pts = np.asarray(contours[name], dtype=float)
        if name == "tongue":
            ax.plot(
                pts[:, 0],
                pts[:, 1],
                "--" if moved else "-",
                color=TONGUE_COLOR,
                lw=2.2,
                alpha=0.95,
            )
        else:
            ax.plot(pts[:, 0], pts[:, 1], color=base_color, lw=1.8, alpha=0.85)


def save_comparison_figure(
    target_image,
    source_image,
    warped_source_image,
    target_contours,
    source_contours,
    warped_source_contours,
    articulators,
    output_path: Path,
):
    """Save a compact visual comparison of source, target, and warped source."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 14))
    axes = axes.ravel()

    ax = axes[0]
    format_frame(ax, target_image, "Target speaker")
    plot_articulator_set(ax, target_contours, articulators, TARGET_COLOR, moved=False)

    ax = axes[1]
    format_frame(ax, source_image, "Source speaker")
    plot_articulator_set(ax, source_contours, articulators, SOURCE_COLOR, moved=False)

    ax = axes[2]
    format_frame(ax, warped_source_image, "Warped source speaker\nin target space")
    plot_articulator_set(ax, warped_source_contours, articulators, WARPED_COLOR, moved=True)

    ax = axes[3]
    format_frame(ax, target_image, "Warped source vs target articulators")
    for name in articulators:
        target_pts = np.asarray(target_contours[name], dtype=float)
        moved_pts = np.asarray(warped_source_contours[name], dtype=float)
        if name == "tongue":
            ax.plot(target_pts[:, 0], target_pts[:, 1], color=TONGUE_COLOR, lw=2.4, alpha=0.95)
            ax.plot(moved_pts[:, 0], moved_pts[:, 1], "--", color=TONGUE_COLOR, lw=2.0, alpha=0.90)
        else:
            ax.plot(target_pts[:, 0], target_pts[:, 1], color=TARGET_COLOR, lw=2.0, alpha=0.95)
            ax.plot(moved_pts[:, 0], moved_pts[:, 1], color=WARPED_COLOR, lw=2.0, alpha=0.90)
    ax.legend(
        [
            plt.Line2D([0], [0], color=TARGET_COLOR, lw=2.5),
            plt.Line2D([0], [0], color=WARPED_COLOR, lw=2.5),
            plt.Line2D([0], [0], color=TONGUE_COLOR, lw=2.5),
            plt.Line2D([0], [0], color=TONGUE_COLOR, lw=2.5, linestyle="--"),
        ],
        ["Target articulators", "Warped source articulators", "Target tongue", "Warped source tongue"],
        loc="upper right",
        framealpha=0.92,
        fontsize=10,
    )

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
