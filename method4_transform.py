from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

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

from vt_grid import build_grid, print_grid_summary


TARGET_COLOR = "#16a085"
SOURCE_COLOR = "#d97706"
TOP_AXIS_ANCHOR_ORDER = ("I1", "I2", "I3", "I4", "I5", "I6", "I7", "C1")
LANDMARK_REPORT_ORDER = (
    "I1", "I2", "I3", "I4", "I5", "I6", "I7",
    "P1",
    "C1", "C2", "C3", "C4", "C5", "C6",
    "M1", "L6",
)


def extract_true_landmarks(grid):
    """Collect the landmarks used by the standalone Method 4 pipeline."""
    lm = {
        "C": np.array([grid.cervical_centers[f"c{i}"] for i in range(1, 7)], dtype=float),
        "L1": np.asarray(grid.left_pts[0], dtype=float),
        "L6": np.asarray(grid.left_pts[-1], dtype=float),
        "M1": np.asarray(grid.M1_point, dtype=float) if grid.M1_point is not None else None,
        "P1": np.asarray(grid.P1_point, dtype=float) if grid.P1_point is not None else None,
    }
    for idx in range(1, 7):
        lm[f"C{idx}"] = np.asarray(grid.cervical_centers[f"c{idx}"], dtype=float)

    for name in ("I1", "I2", "I3", "I4", "I5", "I6", "I7"):
        if grid.I_points is not None and name in grid.I_points:
            lm[name] = np.asarray(grid.I_points[name], dtype=float)
        else:
            lm[name] = None
    return lm


def point_error(src, dst):
    if src is None or dst is None:
        return None
    return float(np.linalg.norm(np.asarray(src, dtype=float) - np.asarray(dst, dtype=float)))


def rms_point_error(src, dst):
    src = np.asarray(src, dtype=float)
    dst = np.asarray(dst, dtype=float)
    return float(np.sqrt(np.mean(np.sum((src - dst) ** 2, axis=1))))


def polyline_rms(line_a, line_b):
    """Pointwise RMS distance between two already-corresponded polylines."""
    line_a = np.asarray(line_a, dtype=float)
    line_b = np.asarray(line_b, dtype=float)
    return float(np.sqrt(np.mean(np.sum((line_a - line_b) ** 2, axis=1))))


def resample_polyline(pts, n=250):
    """Resample a polyline to equally spaced samples."""
    pts = np.asarray(pts, dtype=float)
    if len(pts) == 1:
        return np.repeat(pts, n, axis=0)
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    keep = np.r_[True, seg > 1e-8]
    pts = pts[keep]
    if len(pts) < 2:
        return np.repeat(pts, n, axis=0)
    arc = np.cumsum(np.r_[0.0, np.linalg.norm(np.diff(pts, axis=0), axis=1)])
    arc /= max(float(arc[-1]), 1e-8)
    t = np.linspace(0.0, 1.0, n)
    return np.column_stack([
        np.interp(t, arc, pts[:, 0]),
        np.interp(t, arc, pts[:, 1]),
    ])


def estimate_affine(src, dst):
    """Fit a full 2D affine transform by least squares."""
    src, dst = np.asarray(src, float), np.asarray(dst, float)
    n = len(src)
    mat = np.zeros((2 * n, 6))
    vec = np.zeros(2 * n)
    for i in range(n):
        x, y = src[i]
        u, v = dst[i]
        mat[2 * i] = [x, y, 1, 0, 0, 0]
        mat[2 * i + 1] = [0, 0, 0, x, y, 1]
        vec[2 * i] = u
        vec[2 * i + 1] = v
    params, *_ = np.linalg.lstsq(mat, vec, rcond=None)
    affine = np.array([[params[0], params[1]], [params[3], params[4]]], dtype=float)
    shift = np.array([params[2], params[5]], dtype=float)
    return {"A": affine, "t": shift, "type": "affine"}


def apply_transform(transform, pts):
    """Apply an affine transform to one point or an array of points."""
    pts = np.asarray(pts, float)
    single = pts.ndim == 1
    if single:
        pts = pts.reshape(1, 2)
    out = (transform["A"] @ pts.T).T + transform["t"]
    return out[0] if single else out


def fit_tps(src, dst, smoothing=0.0):
    """Fit a thin-plate spline using scipy RBFInterpolator."""
    from scipy.interpolate import RBFInterpolator

    src, dst = np.asarray(src, float), np.asarray(dst, float)
    rbf_x = RBFInterpolator(src, dst[:, 0], kernel="thin_plate_spline", smoothing=smoothing)
    rbf_y = RBFInterpolator(src, dst[:, 1], kernel="thin_plate_spline", smoothing=smoothing)
    return {"rbf_x": rbf_x, "rbf_y": rbf_y}


def apply_tps(tps, pts):
    """Apply a fitted thin-plate spline to one point or an array of points."""
    pts = np.asarray(pts, float)
    single = pts.ndim == 1
    if single:
        pts = pts.reshape(1, 2)
    out = np.column_stack([tps["rbf_x"](pts), tps["rbf_y"](pts)])
    return out[0] if single else out


def build_step1_anchors(lm_src, lm_tgt):
    """
    Build Step 1 anchors.

    Step 1 uses only the horizontal and vertical axes:
    - horizontal axis: I1..I7, P1, C1
    - vertical axis:   C1..C5

    C1 belongs to both axes conceptually, but it is included only once in the
    affine fit.
    """
    src_blocks = []
    tgt_blocks = []
    labels = []

    for name in ("I1", "I2", "I3", "I4", "I5", "I6", "I7", "P1", "C1", "C2", "C3", "C4", "C5"):
        if lm_src.get(name) is not None and lm_tgt.get(name) is not None:
            src_blocks.append(lm_src[name].reshape(1, 2))
            tgt_blocks.append(lm_tgt[name].reshape(1, 2))
            labels.append(name)

    return np.vstack(src_blocks), np.vstack(tgt_blocks), labels


def build_step2_controls(step1_lm, lm_tgt):
    """
    Build TPS controls after Step 1.

    Step 2 keeps the axis points from Step 1 and adds the remaining landmarks
    so TPS can refine the rest of the grid.
    """
    src_blocks = []
    tgt_blocks = []
    labels = []

    for name in ("I1", "I2", "I3", "I4", "I5", "I6", "I7", "P1", "C1", "C2", "C3", "C4", "C5", "C6", "M1", "L6"):
        if step1_lm.get(name) is not None and lm_tgt.get(name) is not None:
            src_blocks.append(step1_lm[name].reshape(1, 2))
            tgt_blocks.append(lm_tgt[name].reshape(1, 2))
            labels.append(name)

    return np.vstack(src_blocks), np.vstack(tgt_blocks), labels


def map_landmarks(mapping_fn, lm):
    """Apply a point-mapping function to the landmark dictionary."""
    mapped = {}
    for name, value in lm.items():
        if value is None:
            mapped[name] = None
        else:
            mapped[name] = mapping_fn(value)
    return mapped


def compute_metrics(mapped, target):
    metrics = {
        "spine_rms": rms_point_error(mapped["C"], target["C"]),
        "L1_err": point_error(mapped["L1"], target["L1"]),
        "L6_err": point_error(mapped["L6"], target["L6"]),
        "M1_err": point_error(mapped.get("M1"), target.get("M1")),
        "P1_err": point_error(mapped.get("P1"), target.get("P1")),
    }
    metrics["L_2pt_rms"] = rms_point_error(
        np.vstack([mapped["L1"], mapped["L6"]]),
        np.vstack([target["L1"], target["L6"]]),
    )

    horiz_axis_names = [
        name for name in ("I1", "I2", "I3", "I4", "I5", "I6", "I7", "P1", "C1")
        if mapped.get(name) is not None and target.get(name) is not None
    ]
    if horiz_axis_names:
        metrics["horiz_axis_rms"] = rms_point_error(
            np.vstack([mapped[name] for name in horiz_axis_names]),
            np.vstack([target[name] for name in horiz_axis_names]),
        )
    else:
        metrics["horiz_axis_rms"] = None

    vert_axis_names = [
        name for name in ("C1", "C2", "C3", "C4", "C5")
        if mapped.get(name) is not None and target.get(name) is not None
    ]
    if vert_axis_names:
        metrics["vert_axis_rms"] = rms_point_error(
            np.vstack([mapped[name] for name in vert_axis_names]),
            np.vstack([target[name] for name in vert_axis_names]),
        )
    else:
        metrics["vert_axis_rms"] = None

    return metrics


def compute_named_point_errors(mapped, target, names):
    """Return point errors for a fixed ordered list of landmark names."""
    return {name: point_error(mapped.get(name), target.get(name)) for name in names}


def compute_grid_line_errors(horiz_lines, vert_lines, target_grid):
    """Return per-line RMS errors against the target grid."""
    horiz_errors = {
        f"H{i + 1}": polyline_rms(horiz_lines[i], target_grid.horiz_lines[i])
        for i in range(min(len(horiz_lines), len(target_grid.horiz_lines)))
    }
    vert_errors = {
        f"V{j + 1}": polyline_rms(vert_lines[j], target_grid.vert_lines[j])
        for j in range(min(len(vert_lines), len(target_grid.vert_lines)))
    }
    return horiz_errors, vert_errors


def smooth_segment_preserve_ends(segment, passes=3):
    """Smooth one segment while keeping its endpoints fixed."""
    segment = np.asarray(segment, dtype=float)
    if len(segment) < 5:
        return segment.copy()

    kernel = np.array([1.0, 4.0, 6.0, 4.0, 1.0], dtype=float)
    kernel /= np.sum(kernel)

    smoothed = segment.copy()
    for _ in range(passes):
        for dim in range(2):
            padded = np.pad(smoothed[:, dim], (2, 2), mode="edge")
            filtered = np.convolve(padded, kernel, mode="valid")
            smoothed[1:-1, dim] = filtered[1:-1]
        smoothed[0] = segment[0]
        smoothed[-1] = segment[-1]
    return smoothed


def smooth_top_axis_polyline(line, anchor_points, passes=3):
    """Smooth the top horizontal axis between fixed anatomical anchors."""
    line = np.asarray(line, dtype=float)
    if len(line) < 5 or len(anchor_points) < 2:
        return line.copy()

    smoothed = line.copy()
    anchor_indices = []
    search_start = 0
    for point in anchor_points:
        point = np.asarray(point, dtype=float)
        idx_rel = int(np.argmin(np.linalg.norm(smoothed[search_start:] - point[None, :], axis=1)))
        idx = search_start + idx_rel
        anchor_indices.append(idx)
        search_start = idx

    for start_idx, end_idx in zip(anchor_indices[:-1], anchor_indices[1:]):
        if end_idx - start_idx >= 4:
            smoothed[start_idx:end_idx + 1] = smooth_segment_preserve_ends(
                smoothed[start_idx:end_idx + 1],
                passes=passes,
            )

    for idx, point in zip(anchor_indices, anchor_points):
        smoothed[idx] = np.asarray(point, dtype=float)

    return smoothed


def smooth_transformed_grid(horiz_lines, landmark_map, n_vert, *, top_axis_passes=3):
    """Smooth zig-zag artifacts on H1 and rebuild consistent vertical lines."""
    smoothed_horiz = [np.asarray(line, dtype=float).copy() for line in horiz_lines]
    top_axis_points = [
        np.asarray(landmark_map[name], dtype=float)
        for name in TOP_AXIS_ANCHOR_ORDER
        if landmark_map.get(name) is not None
    ]
    if smoothed_horiz and len(top_axis_points) >= 2:
        smoothed_horiz[0] = smooth_top_axis_polyline(
            smoothed_horiz[0],
            top_axis_points,
            passes=top_axis_passes,
        )

    np_samples = len(smoothed_horiz[0])
    vert_idx = np.linspace(0, np_samples - 1, n_vert).astype(int)
    smoothed_vert = []
    for vi in vert_idx:
        cross = np.array([smoothed_horiz[row_idx][vi] for row_idx in range(len(smoothed_horiz))], dtype=float)
        smoothed_vert.append(resample_polyline(cross, np_samples))

    return smoothed_horiz, smoothed_vert


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
    parser.add_argument("--target-speaker", default="1640_s10_0654", help="VTNL target speaker/image name.")
    parser.add_argument("--source-frame", type=int, default=143020, help="nnUNet source frame number.")
    parser.add_argument("--case", default="2008-003^01-1791/test", help="nnUNet case relative path.")
    parser.add_argument("--vtnl-dir", type=Path, default=DEFAULT_VTNL_DIR, help="Folder containing VTNL images and ROI zip files.")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_DIR / "outputs", help="Where to save the figures.")
    return parser


def main():
    args = build_parser().parse_args()

    target_image, target_contours = load_frame_vtnl(args.target_speaker, args.vtnl_dir)
    source_image, source_contours = load_frame_npy(
        args.source_frame,
        VT_SEG_DIR / "data_80" / args.case,
        VT_SEG_DIR / "results" / "nnunet_080" / "inference_contours" / args.case,
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
