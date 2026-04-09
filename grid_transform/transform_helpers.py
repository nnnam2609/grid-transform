from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np


TOP_AXIS_ANCHOR_ORDER = ("I1", "I2", "I3", "I4", "I5", "I6", "I7", "C1")
LANDMARK_REPORT_ORDER = (
    "I1",
    "I2",
    "I3",
    "I4",
    "I5",
    "I6",
    "I7",
    "P1",
    "C1",
    "C2",
    "C3",
    "C4",
    "C5",
    "C6",
    "M1",
    "L6",
)


def extract_true_landmarks(grid):
    """Collect the landmarks used by the affine + TPS pipeline."""
    landmarks = {
        "C": np.array([grid.cervical_centers[f"c{i}"] for i in range(1, 7)], dtype=float),
        "L1": np.asarray(grid.left_pts[0], dtype=float),
        "L6": np.asarray(grid.left_pts[-1], dtype=float),
        "M1": np.asarray(grid.M1_point, dtype=float) if grid.M1_point is not None else None,
        "P1": np.asarray(grid.P1_point, dtype=float) if grid.P1_point is not None else None,
    }
    for idx in range(1, 7):
        landmarks[f"C{idx}"] = np.asarray(grid.cervical_centers[f"c{idx}"], dtype=float)

    for name in ("I1", "I2", "I3", "I4", "I5", "I6", "I7"):
        if grid.I_points is not None and name in grid.I_points:
            landmarks[name] = np.asarray(grid.I_points[name], dtype=float)
        else:
            landmarks[name] = None
    return landmarks


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


def resample_polyline(pts, n: int = 250):
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
    return np.column_stack(
        [
            np.interp(t, arc, pts[:, 0]),
            np.interp(t, arc, pts[:, 1]),
        ]
    )


def estimate_affine(src, dst):
    """Fit a full 2D affine transform by least squares."""
    src, dst = np.asarray(src, float), np.asarray(dst, float)
    n_points = len(src)
    mat = np.zeros((2 * n_points, 6))
    vec = np.zeros(2 * n_points)
    for index in range(n_points):
        x_coord, y_coord = src[index]
        u_coord, v_coord = dst[index]
        mat[2 * index] = [x_coord, y_coord, 1, 0, 0, 0]
        mat[2 * index + 1] = [0, 0, 0, x_coord, y_coord, 1]
        vec[2 * index] = u_coord
        vec[2 * index + 1] = v_coord
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


def fit_tps(src, dst, smoothing: float = 0.0):
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
    - vertical axis:   C1..C6

    C1 belongs to both axes conceptually, but it is included only once in the
    affine fit.
    """
    src_blocks = []
    tgt_blocks = []
    labels = []

    for name in ("I1", "I2", "I3", "I4", "I5", "I6", "I7", "P1", "C1", "C2", "C3", "C4", "C5", "C6"):
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


def map_landmarks(mapping_fn, landmarks):
    """Apply a point-mapping function to the landmark dictionary."""
    mapped = {}
    for name, value in landmarks.items():
        mapped[name] = None if value is None else mapping_fn(value)
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
        name
        for name in ("I1", "I2", "I3", "I4", "I5", "I6", "I7", "P1", "C1")
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
        name
        for name in ("C1", "C2", "C3", "C4", "C5")
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


def smooth_segment_preserve_ends(segment, passes: int = 3):
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


def smooth_top_axis_polyline(line, anchor_points, passes: int = 3):
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
            smoothed[start_idx : end_idx + 1] = smooth_segment_preserve_ends(
                smoothed[start_idx : end_idx + 1],
                passes=passes,
            )

    for idx, point in zip(anchor_indices, anchor_points):
        smoothed[idx] = np.asarray(point, dtype=float)

    return smoothed


def smooth_transformed_grid(horiz_lines, landmark_map, n_vert, *, top_axis_passes: int = 3):
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

    n_samples = len(smoothed_horiz[0])
    vert_idx = np.linspace(0, n_samples - 1, n_vert).astype(int)
    smoothed_vert = []
    for vi in vert_idx:
        cross = np.array([smoothed_horiz[row_idx][vi] for row_idx in range(len(smoothed_horiz))], dtype=float)
        smoothed_vert.append(resample_polyline(cross, n_samples))

    return smoothed_horiz, smoothed_vert


def format_target_frame(ax, image, title: str):
    image_arr = np.asarray(image)
    height, width = image_arr.shape[:2]
    ax.imshow(image, cmap="gray")
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.axis("off")
    ax.set_xlim(0, width)
    ax.set_ylim(height, 0)


def choose_tab20_label_colors(labels, *, tongue_color):
    cmap = plt.get_cmap("tab20")
    colors = {}
    for idx, label in enumerate(labels):
        colors[label] = tongue_color if label == "tongue" else cmap(idx % 20)
    return colors
