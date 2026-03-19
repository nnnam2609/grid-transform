from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy.ndimage import map_coordinates

from create_speaker_grid import (
    DEFAULT_VTNL_DIR,
    PROJECT_DIR,
    TONGUE_COLOR,
    VT_SEG_DIR,
    load_frame_npy,
    load_frame_vtnl,
)

if str(VT_SEG_DIR) not in sys.path:
    sys.path.insert(0, str(VT_SEG_DIR))

from vt_grid import build_grid
from method4_transform import (
    apply_tps,
    apply_transform,
    build_step1_anchors,
    build_step2_controls,
    estimate_affine,
    extract_true_landmarks,
    fit_tps,
)


DEFAULT_ARTICULATORS = (
    "incisior-hard-palate",
    "soft-palate-midline",
    "soft-palate",
    "tongue",
    "mandible-incisior",
    "pharynx",
)

TARGET_COLOR = "#16a085"
SOURCE_COLOR = "#3a86ff"
WARPED_COLOR = "#ef476f"


def build_two_step_transform(source_grid, target_grid):
    """Build the forward two-step transform from source space to target space."""
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
        "step1_labels": step1_labels,
        "step2_labels": step2_labels,
        "apply_two_step": apply_two_step,
    }


def warp_image_to_target_space(source_image, target_shape, inverse_mapping_fn):
    """Warp the full source image into target image space using inverse mapping."""
    source_arr = np.asarray(source_image, dtype=float)
    if source_arr.ndim == 3:
        source_arr = source_arr[..., 0]

    target_h, target_w = target_shape[:2]
    yy, xx = np.meshgrid(np.arange(target_h), np.arange(target_w), indexing="ij")
    target_pts = np.column_stack([xx.ravel(), yy.ravel()])
    source_pts = np.asarray(inverse_mapping_fn(target_pts), dtype=float)

    source_x = source_pts[:, 0].reshape(target_h, target_w)
    source_y = source_pts[:, 1].reshape(target_h, target_w)

    warped = map_coordinates(
        source_arr,
        [source_y, source_x],
        order=1,
        mode="constant",
        cval=0.0,
    )
    mask = (
        (source_x >= 0)
        & (source_x <= source_arr.shape[1] - 1)
        & (source_y >= 0)
        & (source_y <= source_arr.shape[0] - 1)
    )
    warped *= mask

    return np.clip(warped, 0, 255).astype(np.uint8), mask.astype(np.uint8) * 255


def transform_contours(contours, mapping_fn, names):
    """Apply the point mapping to selected articulator contours."""
    return {
        name: np.asarray(mapping_fn(np.asarray(contours[name], dtype=float)), dtype=float)
        for name in names
    }


def resolve_articulators(target_contours, source_contours, requested):
    """Pick a stable shared articulator set for overlay visualization."""
    common = sorted(set(target_contours) & set(source_contours))
    if requested:
        names = [name.strip() for name in requested.split(",") if name.strip() in common]
    else:
        names = [name for name in DEFAULT_ARTICULATORS if name in common]
    if not names:
        raise ValueError(f"No common articulators found. Common keys: {common}")
    return names


def format_frame(ax, image, title):
    """Display one frame in image coordinates."""
    img = np.asarray(image)
    h_img, w_img = img.shape[:2]
    ax.imshow(image, cmap="gray")
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.axis("off")
    ax.set_xlim(0, w_img)
    ax.set_ylim(h_img, 0)


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
    output_path,
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


def build_parser():
    parser = argparse.ArgumentParser(
        description="Warp the whole source speaker image into target speaker space with affine + TPS.",
    )
    parser.add_argument("--target-speaker", default="1640_s10_0654", help="VTNL target speaker/image name.")
    parser.add_argument("--source-frame", type=int, default=143020, help="nnUNet source frame number.")
    parser.add_argument("--case", default="2008-003^01-1791/test", help="nnUNet case relative path.")
    parser.add_argument("--vtnl-dir", type=Path, default=DEFAULT_VTNL_DIR, help="Folder containing VTNL images and ROI zip files.")
    parser.add_argument(
        "--articulators",
        help="Comma-separated articulators for overlay visualization.",
    )
    parser.add_argument(
        "--warped-image-output",
        type=Path,
        default=PROJECT_DIR / "outputs" / "source_speaker_warped_to_target.png",
        help="Where to save the warped source image.",
    )
    parser.add_argument(
        "--mask-output",
        type=Path,
        default=PROJECT_DIR / "outputs" / "source_speaker_warped_mask.png",
        help="Where to save the valid warp mask.",
    )
    parser.add_argument(
        "--figure-output",
        type=Path,
        default=PROJECT_DIR / "outputs" / "source_speaker_warped_to_target_comparison.png",
        help="Where to save the comparison figure.",
    )
    return parser


def main():
    args = build_parser().parse_args()

    target_image, target_contours = load_frame_vtnl(args.target_speaker, args.vtnl_dir)
    source_image, source_contours = load_frame_npy(
        args.source_frame,
        VT_SEG_DIR / "data_80" / args.case,
        VT_SEG_DIR / "results" / "nnunet_080" / "inference_contours" / args.case,
    )

    articulators = resolve_articulators(target_contours, source_contours, args.articulators)

    target_grid = build_grid(target_image, target_contours, n_vert=9, n_points=250, frame_number=0)
    source_grid = build_grid(source_image, source_contours, n_vert=9, n_points=250, frame_number=args.source_frame)

    forward_transform = build_two_step_transform(source_grid, target_grid)
    inverse_transform = build_two_step_transform(target_grid, source_grid)

    warped_source_arr, warped_mask_arr = warp_image_to_target_space(
        source_image,
        np.asarray(target_image).shape,
        inverse_transform["apply_two_step"],
    )
    warped_source_contours = transform_contours(
        source_contours,
        forward_transform["apply_two_step"],
        articulators,
    )

    args.warped_image_output.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(warped_source_arr).save(args.warped_image_output)
    Image.fromarray(warped_mask_arr).save(args.mask_output)

    save_comparison_figure(
        target_image,
        source_image,
        warped_source_arr,
        target_contours,
        source_contours,
        warped_source_contours,
        articulators,
        args.figure_output,
    )

    print("Forward Step 1 anchors:", ", ".join(forward_transform["step1_labels"]))
    print("Forward Step 2 controls:", ", ".join(forward_transform["step2_labels"]))
    print("Articulators shown:", ", ".join(articulators))
    print(f"Saved warped image: {args.warped_image_output}")
    print(f"Saved warp mask: {args.mask_output}")
    print(f"Saved figure: {args.figure_output}")


if __name__ == "__main__":
    main()
