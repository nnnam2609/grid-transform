from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from grid_transform.config import (
    DEFAULT_VTLN_DIR,
    DEFAULT_OUTPUT_DIR,
    TONGUE_COLOR,
    VT_SEG_CONTOURS_ROOT,
    VT_SEG_DATA_ROOT,
)
from grid_transform.io import load_frame_npy, load_frame_vtln
from grid_transform.transform_helpers import (
    LANDMARK_REPORT_ORDER,
    TOP_AXIS_ANCHOR_ORDER,
    apply_tps,
    apply_transform,
    build_step1_anchors,
    build_step2_controls,
    compute_grid_line_errors,
    compute_metrics,
    compute_named_point_errors,
    estimate_affine,
    extract_true_landmarks,
    fit_tps,
    map_landmarks,
    point_error,
    polyline_rms,
    resample_polyline,
    rms_point_error,
    smooth_segment_preserve_ends,
    smooth_top_axis_polyline,
    smooth_transformed_grid,
)
from grid_transform.vt import build_grid, print_grid_summary


TARGET_COLOR = "#16a085"
SOURCE_COLOR = "#d97706"
def format_error_value(value):
    """Pretty-print one numeric error value."""
    return "--" if value is None else f"{value:5.2f}"


def build_error_report_lines(step0_errors, step1_errors, step2_errors, names):
    """Build compact report lines for Step 0 / Step 1 / Step 2 errors."""
    lines = ["name   Raw  Affine  Aff+TPS"]
    for name in names:
        lines.append(
            f"{name:<4} {format_error_value(step0_errors.get(name))} "
            f"{format_error_value(step1_errors.get(name))} "
            f"{format_error_value(step2_errors.get(name))}"
        )
    return lines


def draw_grid_single_color(ax, horiz, vert, color, *, alpha=0.95, lw_major=3.0, lw_minor=1.0):
    """Draw one grid using a single color and thickness to indicate outer lines."""
    for i, h in enumerate(horiz):
        lw = lw_major if i in (0, len(horiz) - 1) else lw_minor
        ax.plot(h[:, 0], h[:, 1], "-", color=color, lw=lw, alpha=alpha)
    for j, v in enumerate(vert):
        lw = lw_major if j in (0, len(vert) - 1) else lw_minor
        ax.plot(v[:, 0], v[:, 1], "-", color=color, lw=lw, alpha=alpha)


def draw_named_points(ax, pts, labels, color, marker="o", dx=6, dy=-6):
    for pt, label in zip(pts, labels):
        ax.plot(pt[0], pt[1], marker=marker, ms=7, color=color, mec="white", mew=1.2, zorder=8)
        ax.text(
            pt[0] + dx,
            pt[1] + dy,
            label,
            fontsize=9,
            color=color,
            fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.85),
            zorder=9,
        )


def draw_grid_line_labels(ax, horiz, vert, color, *, alpha=0.95):
    """Draw H1..Hn and V1..Vn labels directly on a grid."""
    for i, line in enumerate(horiz):
        label_pt = np.asarray(line[8], dtype=float)
        ax.text(
            label_pt[0] - 10,
            label_pt[1],
            f"H{i + 1}",
            fontsize=8.5,
            color=color,
            fontweight="bold",
            ha="right",
            va="center",
            bbox=dict(boxstyle="round,pad=0.15", fc="white", alpha=alpha),
            zorder=10,
        )

    for j, line in enumerate(vert):
        label_idx = min(16, len(line) - 1)
        label_pt = np.asarray(line[label_idx], dtype=float)
        ax.text(
            label_pt[0],
            label_pt[1] - 10,
            f"V{j + 1}",
            fontsize=8.5,
            color=color,
            fontweight="bold",
            ha="center",
            va="bottom",
            bbox=dict(boxstyle="round,pad=0.15", fc="white", alpha=alpha),
            zorder=10,
        )


def format_target_frame(ax, image, title):
    image_arr = np.asarray(image)
    h_img, w_img = image_arr.shape[:2]
    ax.imshow(image, cmap="gray")
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.axis("off")
    ax.set_xlim(0, w_img)
    ax.set_ylim(h_img, 0)


def draw_tongue_overlay(ax, target_contours, source_contours, mapping_fn):
    """Overlay target and mapped source tongue on the final Method 4 view."""
    if "tongue" not in target_contours or "tongue" not in source_contours:
        return

    target_tongue = np.asarray(target_contours["tongue"], dtype=float)
    moved_source_tongue = np.asarray(mapping_fn(np.asarray(source_contours["tongue"], dtype=float)), dtype=float)
    ax.plot(
        target_tongue[:, 0],
        target_tongue[:, 1],
        color=TONGUE_COLOR,
        lw=2.4,
        alpha=0.95,
        zorder=18,
        label="Target tongue",
    )
    ax.plot(
        moved_source_tongue[:, 0],
        moved_source_tongue[:, 1],
        "--",
        color=TONGUE_COLOR,
        lw=2.0,
        alpha=0.95,
        zorder=18,
        label="Mapped source tongue",
    )


def run_method4(target_grid, source_grid, output_dir: Path):
    """Run the axis-first affine + TPS pipeline and save explanatory figures."""
    lm_tgt = extract_true_landmarks(target_grid)
    lm_src = extract_true_landmarks(source_grid)
    step0_point_errors = compute_named_point_errors(lm_src, lm_tgt, LANDMARK_REPORT_ORDER)

    step1_anchor_src, step1_anchor_tgt, step1_anchor_labels = build_step1_anchors(lm_src, lm_tgt)
    step1_affine = estimate_affine(step1_anchor_src, step1_anchor_tgt)

    step1_lm = map_landmarks(lambda pts: apply_transform(step1_affine, pts), lm_src)
    step1_horiz_raw = [apply_transform(step1_affine, h) for h in source_grid.horiz_lines]
    step1_horiz, step1_vert = smooth_transformed_grid(
        step1_horiz_raw,
        step1_lm,
        source_grid.n_vert,
        top_axis_passes=4,
    )
    step1_point_errors = compute_named_point_errors(step1_lm, lm_tgt, LANDMARK_REPORT_ORDER)

    step1_anchor_hat = apply_transform(step1_affine, step1_anchor_src)
    step1_anchor_rms = rms_point_error(step1_anchor_hat, step1_anchor_tgt)

    step2_ctrl_src, step2_ctrl_tgt, step2_ctrl_labels = build_step2_controls(step1_lm, lm_tgt)
    step2_tps = fit_tps(step2_ctrl_src, step2_ctrl_tgt, smoothing=0.0)

    def apply_two_step(pts):
        affine_pts = apply_transform(step1_affine, pts)
        return apply_tps(step2_tps, affine_pts)

    mapped_final = map_landmarks(apply_two_step, lm_src)
    final_metrics = compute_metrics(mapped_final, lm_tgt)
    final_horiz_raw = [apply_two_step(h) for h in source_grid.horiz_lines]
    final_horiz, final_vert = smooth_transformed_grid(
        final_horiz_raw,
        mapped_final,
        source_grid.n_vert,
        top_axis_passes=4,
    )
    step2_point_errors = compute_named_point_errors(mapped_final, lm_tgt, LANDMARK_REPORT_ORDER)

    step0_horiz_errors, step0_vert_errors = compute_grid_line_errors(
        source_grid.horiz_lines,
        source_grid.vert_lines,
        target_grid,
    )
    step1_horiz_errors, step1_vert_errors = compute_grid_line_errors(
        step1_horiz,
        step1_vert,
        target_grid,
    )
    step2_horiz_errors, step2_vert_errors = compute_grid_line_errors(
        final_horiz,
        final_vert,
        target_grid,
    )

    output_dir.mkdir(parents=True, exist_ok=True)

    # Two grids only figure.
    fig, ax = plt.subplots(figsize=(12, 12))
    format_target_frame(
        ax,
        target_grid.image,
        "Axis-first affine + TPS: two grids only\none color per speaker, thickness = outer vs inner grid",
    )
    draw_grid_single_color(ax, target_grid.horiz_lines, target_grid.vert_lines, TARGET_COLOR, alpha=0.95, lw_major=3.2, lw_minor=1.2)
    draw_grid_single_color(ax, final_horiz, final_vert, SOURCE_COLOR, alpha=0.88, lw_major=2.8, lw_minor=0.9)
    ax.legend(
        [
            plt.Line2D([0], [0], color=TARGET_COLOR, lw=3),
            plt.Line2D([0], [0], color=SOURCE_COLOR, lw=3),
        ],
        ["Target grid", "Source grid after Method 4"],
        fontsize=10,
        loc="upper right",
        framealpha=0.92,
    )
    grid_only_path = output_dir / "method4_axis_affine_two_grids_only.png"
    fig.savefig(grid_only_path, dpi=220, bbox_inches="tight")
    plt.close(fig)

    # Step-by-step figure.
    fig, axes = plt.subplots(2, 3, figsize=(24, 14))
    axes = axes.ravel()

    ax = axes[0]
    format_target_frame(ax, target_grid.image, "Step 0: before alignment")
    draw_grid_single_color(ax, target_grid.horiz_lines, target_grid.vert_lines, TARGET_COLOR, alpha=0.95, lw_major=3.0, lw_minor=1.1)
    draw_grid_single_color(ax, source_grid.horiz_lines, source_grid.vert_lines, SOURCE_COLOR, alpha=0.70, lw_major=2.6, lw_minor=0.8)

    ax = axes[1]
    format_target_frame(
        ax,
        target_grid.image,
        f"Step 1a: affine fit on {len(step1_anchor_labels)} axis anchors",
    )
    draw_named_points(ax, step1_anchor_tgt, step1_anchor_labels, TARGET_COLOR, marker="o")
    draw_named_points(ax, step1_anchor_hat, step1_anchor_labels, SOURCE_COLOR, marker="s", dx=6, dy=10)
    for src_pt, tgt_pt in zip(step1_anchor_hat, step1_anchor_tgt):
        ax.annotate("", xy=tgt_pt, xytext=src_pt, arrowprops=dict(arrowstyle="->", color="black", lw=1.1, alpha=0.75))
    ax.text(
        0.02,
        0.03,
        f"Anchor RMS after affine = {step1_anchor_rms:.2f} px",
        transform=ax.transAxes,
        fontsize=11,
        fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.92),
    )

    ax = axes[2]
    format_target_frame(ax, target_grid.image, "Step 1b: affine-mapped source grid from axis anchors")
    draw_grid_single_color(ax, target_grid.horiz_lines, target_grid.vert_lines, TARGET_COLOR, alpha=0.95, lw_major=3.0, lw_minor=1.1)
    draw_grid_single_color(ax, step1_horiz, step1_vert, SOURCE_COLOR, alpha=0.85, lw_major=2.6, lw_minor=0.8)

    ax = axes[3]
    format_target_frame(ax, target_grid.image, f"Step 2a: TPS controls after Step 1 ({len(step2_ctrl_labels)} pts)")
    draw_named_points(ax, step2_ctrl_tgt, step2_ctrl_labels, TARGET_COLOR, marker="o")
    draw_named_points(ax, step2_ctrl_src, step2_ctrl_labels, SOURCE_COLOR, marker="s", dx=6, dy=10)
    for src_pt, tgt_pt in zip(step2_ctrl_src, step2_ctrl_tgt):
        ax.annotate("", xy=tgt_pt, xytext=src_pt, arrowprops=dict(arrowstyle="->", color="black", lw=1.1, alpha=0.75))
    ax.text(
        0.02,
        0.03,
        "TPS keeps the anchors aligned and bends the remaining source grid.",
        transform=ax.transAxes,
        fontsize=10,
        bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.92),
    )

    ax = axes[4]
    format_target_frame(ax, target_grid.image, "Step 2b: final axis-first affine + TPS result")
    draw_grid_single_color(ax, target_grid.horiz_lines, target_grid.vert_lines, TARGET_COLOR, alpha=0.95, lw_major=3.0, lw_minor=1.1)
    draw_grid_single_color(ax, final_horiz, final_vert, SOURCE_COLOR, alpha=0.88, lw_major=2.8, lw_minor=0.9)
    draw_grid_line_labels(ax, target_grid.horiz_lines, target_grid.vert_lines, TARGET_COLOR, alpha=0.82)
    draw_tongue_overlay(ax, target_grid.contours, source_grid.contours, apply_two_step)
    ax.legend(loc="upper right", fontsize=9, framealpha=0.92)

    ax = axes[5]
    ax.axis("off")
    overview_lines = [
        "Axis-first affine + TPS summary",
        "Raw = source, Affine = Step 1, Aff+TPS = Step 2",
        "",
        f"Step 1 anchors: {', '.join(step1_anchor_labels)}",
        f"Step 1 anchor RMS: {step1_anchor_rms:.2f} px",
        f"Final spine RMS:   {final_metrics['spine_rms']:.2f} px",
        f"Final horiz RMS:   {final_metrics['horiz_axis_rms']:.2f} px" if final_metrics["horiz_axis_rms"] is not None else "Final horiz RMS:   --",
        f"Final vert RMS:    {final_metrics['vert_axis_rms']:.2f} px" if final_metrics["vert_axis_rms"] is not None else "Final vert RMS:    --",
        f"Final L1 error:    {final_metrics['L1_err']:.2f} px",
        f"Final L6 error:    {final_metrics['L6_err']:.2f} px",
        f"Final M1 error:    {final_metrics['M1_err']:.2f} px" if final_metrics["M1_err"] is not None else "Final M1 error:    --",
        f"Final P1 error:    {final_metrics['P1_err']:.2f} px" if final_metrics["P1_err"] is not None else "Final P1 error:    --",
        f"Final L 2-pt RMS:  {final_metrics['L_2pt_rms']:.2f} px",
        "",
        "Point errors by step",
    ]
    point_report_lines = build_error_report_lines(
        step0_point_errors,
        step1_point_errors,
        step2_point_errors,
        LANDMARK_REPORT_ORDER,
    )
    horiz_report_lines = build_error_report_lines(
        step0_horiz_errors,
        step1_horiz_errors,
        step2_horiz_errors,
        [f"H{i + 1}" for i in range(len(step0_horiz_errors))],
    )
    vert_report_lines = build_error_report_lines(
        step0_vert_errors,
        step1_vert_errors,
        step2_vert_errors,
        [f"V{i + 1}" for i in range(len(step0_vert_errors))],
    )

    ax.text(
        0.02,
        0.98,
        "\n".join(overview_lines + point_report_lines),
        va="top",
        ha="left",
        family="monospace",
        fontsize=7.8,
        bbox=dict(boxstyle="round,pad=0.4", fc="white", alpha=0.96),
    )
    ax.text(
        0.62,
        0.98,
        "\n".join(["Grid line RMS by step", ""] + horiz_report_lines + [""] + vert_report_lines),
        va="top",
        ha="left",
        family="monospace",
        fontsize=7.8,
        bbox=dict(boxstyle="round,pad=0.4", fc="white", alpha=0.96),
    )

    plt.tight_layout()
    step_by_step_path = output_dir / "method4_axis_affine_step_by_step.png"
    fig.savefig(step_by_step_path, dpi=220, bbox_inches="tight")
    plt.close(fig)

    print("Step 1 anchors:", ", ".join(step1_anchor_labels))
    print(f"Step 1 anchor RMS: {step1_anchor_rms:.2f} px")
    print(f"Final spine RMS: {final_metrics['spine_rms']:.2f} px")
    print(f"Final horiz axis RMS: {final_metrics['horiz_axis_rms']:.2f} px" if final_metrics["horiz_axis_rms"] is not None else "Final horiz axis RMS: --")
    print(f"Final vert axis RMS: {final_metrics['vert_axis_rms']:.2f} px" if final_metrics["vert_axis_rms"] is not None else "Final vert axis RMS: --")
    print("Final horizontal line RMS:", ", ".join(f"{name}={value:.2f}" for name, value in step2_horiz_errors.items()))
    print("Final vertical line RMS:", ", ".join(f"{name}={value:.2f}" for name, value in step2_vert_errors.items()))
    print(f"Saved: {grid_only_path}")
    print(f"Saved: {step_by_step_path}")


def build_parser():
    parser = argparse.ArgumentParser(description="Run Method 4 as a standalone script.")
    parser.add_argument("--target-speaker", default="1640_s10_0829", help="VTLN target speaker/image name.")
    parser.add_argument("--source-frame", type=int, default=143020, help="nnUNet source frame number.")
    parser.add_argument("--case", default="2008-003^01-1791/test", help="nnUNet case relative path.")
    parser.add_argument("--vtln-dir", type=Path, default=DEFAULT_VTLN_DIR, help="Folder containing VTLN images and ROI zip files.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Where to save the figures.")
    return parser


def main(argv: list[str] | None = None):
    args = build_parser().parse_args(argv)

    target_image, target_contours = load_frame_vtln(args.target_speaker, args.vtln_dir)
    source_image, source_contours = load_frame_npy(
        args.source_frame,
        VT_SEG_DATA_ROOT / args.case,
        VT_SEG_CONTOURS_ROOT / args.case,
    )

    target_grid = build_grid(target_image, target_contours, n_vert=9, n_points=250, frame_number=0)
    source_grid = build_grid(source_image, source_contours, n_vert=9, n_points=250, frame_number=args.source_frame)

    print("Target grid")
    print_grid_summary(target_grid)
    print("Source grid")
    print_grid_summary(source_grid)

    run_method4(target_grid, source_grid, args.output_dir)


if __name__ == "__main__":
    main()
