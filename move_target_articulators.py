from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from create_speaker_grid import (
    PROJECT_DIR,
    VT_SEG_DIR,
    load_frame_npy,
    load_frame_vtnl,
)

if str(VT_SEG_DIR) not in sys.path:
    sys.path.insert(0, str(VT_SEG_DIR))

from vt_grid import build_grid
from method4_transform import (
    SOURCE_COLOR,
    TARGET_COLOR,
    apply_tps,
    apply_transform,
    build_step1_anchors,
    build_step2_controls,
    estimate_affine,
    extract_true_landmarks,
    fit_tps,
    polyline_rms,
    resample_polyline,
)


DEFAULT_ARTICULATORS = (
    "incisior-hard-palate",
    "soft-palate-midline",
    "soft-palate",
    "mandible-incisior",
    "pharynx",
)

MOVED_SOURCE_COLOR = "#ef476f"
RAW_TARGET_COLOR = "#16a085"
SOURCE_ARTIC_COLOR = "#3a86ff"


def parse_articulators(text: str | None, target_contours: dict, source_contours: dict):
    """Resolve the articulator names to compare."""
    common = sorted(set(target_contours) & set(source_contours))
    if text:
        requested = [item.strip() for item in text.split(",") if item.strip()]
        names = [name for name in requested if name in common]
    else:
        names = [name for name in DEFAULT_ARTICULATORS if name in common]

    if not names:
        raise ValueError(f"No valid common articulators found. Common keys: {common}")
    return names


def build_two_step_transform(source_grid, target_grid):
    """Build the same axis-first affine + TPS transform used in Method 4."""
    lm_src = extract_true_landmarks(source_grid)
    lm_tgt = extract_true_landmarks(target_grid)

    step1_src, step1_tgt, step1_labels = build_step1_anchors(lm_src, lm_tgt)
    step1_affine = estimate_affine(step1_src, step1_tgt)
    step1_lm = {name: (None if value is None else apply_transform(step1_affine, value)) for name, value in lm_src.items()}

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


def transform_contours(contours: dict, mapping_fn, names):
    """Apply a point mapping function to selected contours."""
    moved = {}
    for name in names:
        moved[name] = np.asarray(mapping_fn(np.asarray(contours[name], dtype=float)), dtype=float)
    return moved


def compute_articulator_errors(moved_contours: dict, target_contours: dict):
    """Compute per-articulator RMS after resampling onto a common number of points."""
    errors = {}
    for name, moved in moved_contours.items():
        target = np.asarray(target_contours[name], dtype=float)
        n_samples = max(len(moved), len(target), 120)
        moved_rs = resample_polyline(moved, n=n_samples)
        target_rs = resample_polyline(target, n=n_samples)
        errors[name] = polyline_rms(moved_rs, target_rs)
    return errors


def plot_contours(ax, contours: dict, names, color, *, lw=2.0, alpha=0.9, label_prefix=None):
    """Draw a selected contour set on an axes."""
    handles = []
    for name in names:
        contour = np.asarray(contours[name], dtype=float)
        handle, = ax.plot(contour[:, 0], contour[:, 1], color=color, lw=lw, alpha=alpha)
        handles.append((handle, f"{label_prefix}{name}" if label_prefix else name))
    return handles


def format_frame(ax, image, title):
    """Show an image with image-space coordinates preserved."""
    img_arr = np.asarray(image)
    h_img, w_img = img_arr.shape[:2]
    ax.imshow(image, cmap="gray")
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.axis("off")
    ax.set_xlim(0, w_img)
    ax.set_ylim(h_img, 0)


def save_comparison_figure(
    target_image,
    source_image,
    target_contours,
    source_contours,
    moved_source_contours,
    articulators,
    errors,
    output_path: Path,
):
    """Render a three-panel comparison figure."""
    fig, axes = plt.subplots(1, 3, figsize=(22, 7))

    ax = axes[0]
    format_frame(ax, target_image, "Target speaker\nraw articulators")
    plot_contours(ax, target_contours, articulators, RAW_TARGET_COLOR, lw=2.2, alpha=0.95)

    ax = axes[1]
    format_frame(ax, source_image, "Source speaker\nraw articulators")
    plot_contours(ax, source_contours, articulators, SOURCE_ARTIC_COLOR, lw=2.2, alpha=0.95)

    ax = axes[2]
    format_frame(ax, target_image, "Moved source vs target\nin target space")
    target_handles = plot_contours(ax, target_contours, articulators, RAW_TARGET_COLOR, lw=2.2, alpha=0.90, label_prefix="Target: ")
    moved_handles = plot_contours(ax, moved_source_contours, articulators, MOVED_SOURCE_COLOR, lw=2.2, alpha=0.90, label_prefix="Moved source: ")

    legend_handles = []
    legend_labels = []
    for handle, label in target_handles[:1] + moved_handles[:1]:
        legend_handles.append(handle)
        legend_labels.append(label.split(":")[0])
    ax.legend(legend_handles, legend_labels, loc="upper right", framealpha=0.92, fontsize=10)

    report_lines = ["Articulator RMS (px)"]
    for name in articulators:
        report_lines.append(f"{name}: {errors[name]:.2f}")
    ax.text(
        0.02,
        0.02,
        "\n".join(report_lines),
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=9.5,
        family="monospace",
        bbox=dict(boxstyle="round,pad=0.35", fc="white", alpha=0.95),
    )

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def build_parser():
    parser = argparse.ArgumentParser(
        description="Move source articulators into target space with the same affine + TPS pipeline.",
    )
    parser.add_argument("--target-speaker", default="1640_s10_0654", help="VTNL target speaker/image name.")
    parser.add_argument("--source-frame", type=int, default=143020, help="nnUNet source frame number.")
    parser.add_argument("--case", default="2008-003^01-1791/test", help="nnUNet case relative path.")
    parser.add_argument(
        "--articulators",
        help="Comma-separated articulator list. Default uses the common moving articulators.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_DIR / "outputs" / "source_moved_to_target_articulators.png",
        help="Output comparison image path.",
    )
    return parser


def main():
    args = build_parser().parse_args()

    target_image, target_contours = load_frame_vtnl(args.target_speaker, PROJECT_DIR / "VTNL")
    source_image, source_contours = load_frame_npy(
        args.source_frame,
        VT_SEG_DIR / "data_80" / args.case,
        VT_SEG_DIR / "results" / "nnunet_080" / "inference_contours" / args.case,
    )

    articulators = parse_articulators(args.articulators, target_contours, source_contours)

    # Fit source -> target so the moved source can be compared directly on top of target.
    target_grid = build_grid(target_image, target_contours, n_vert=9, n_points=250, frame_number=0)
    source_grid = build_grid(source_image, source_contours, n_vert=9, n_points=250, frame_number=args.source_frame)
    transform = build_two_step_transform(source_grid, target_grid)

    moved_source_contours = transform_contours(source_contours, transform["apply_two_step"], articulators)
    errors = compute_articulator_errors(moved_source_contours, target_contours)

    save_comparison_figure(
        target_image,
        source_image,
        target_contours,
        source_contours,
        moved_source_contours,
        articulators,
        errors,
        args.output,
    )

    print("Step 1 anchors:", ", ".join(transform["step1_labels"]))
    print("Step 2 controls:", ", ".join(transform["step2_labels"]))
    print("Articulators:", ", ".join(articulators))
    for name in articulators:
        print(f"{name}: RMS = {errors[name]:.2f} px")
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
