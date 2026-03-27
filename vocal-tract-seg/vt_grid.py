"""
Vocal Tract Curvilinear Grid
=============================

Build and visualize an anatomically-anchored curvilinear grid on
mid-sagittal MRI frames of the vocal tract.

Usage
-----
    from vt_grid import build_grid, visualize_grid

    grid = build_grid(image, contours)
    visualize_grid(grid)

    # Or one-liner:
    from vt_grid import build_and_show_grid
    grid = build_and_show_grid(image, contours, frame_number=143020)
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Optional, Tuple, List
from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------
@dataclass
class GridData:
    """All computed grid information, returned by ``build_grid``."""

    # Horizontal & vertical polylines  (each is an (NP, 2) array)
    horiz_lines: List[np.ndarray] = field(default_factory=list)
    vert_lines:  List[np.ndarray] = field(default_factory=list)

    # Anchor points  (left/right endpoints of every horizontal line)
    left_pts:  np.ndarray = None   # (n_h, 2)
    right_pts: np.ndarray = None   # (n_h, 2)

    # Named anatomical landmarks
    I1: np.ndarray = None          # incisior-hard-palate lowest-left
    M1: np.ndarray = None          # mandible-incisior highest
    L6: np.ndarray = None          # mandible-incisior lowest

    I_points: Dict[str, np.ndarray] = None   # I1-I7
    cervical_centers: Dict[str, tuple] = None  # c1-c6 centroids
    P1_point: np.ndarray = None    # pharynx intersection (may be None)
    M1_point: np.ndarray = None

    # Reference curves  (resampled)
    vt_curve: np.ndarray = None    # vocal-tract curve I1->C1
    spine_curve: np.ndarray = None # spine C1->C6

    # Original inputs (kept for visualisation)
    image: object = None
    contours: Dict[str, np.ndarray] = None
    frame_number: int = 0

    # Label per horizontal line
    h_labels: List[str] = field(default_factory=list)

    # Grid sizes
    n_horiz: int = 0
    n_vert:  int = 0

    # H1 (topmost horizontal line) construction metadata
    palate_end_idx: int = 0        # index of I5 on the resampled H1
    h1_dev_ratio: float = 0.0     # max perpendicular deviation / chord length
    h1_max_clamp: float = 0.0     # fraction of points that were clamped (0 = pure midline)
    hard_errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    contour_point_counts: Dict[str, int] = field(default_factory=dict)


@dataclass
class GridContourValidation:
    """Preflight contour validation shared by build and diagnostics."""

    hard_errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    point_counts: Dict[str, int] = field(default_factory=dict)


class GridValidationError(ValueError):
    """Raised when required contours are missing for grid construction."""
    pass


REQUIRED_GRID_CONTOURS = (
    "incisior-hard-palate",
    "mandible-incisior",
    "c1",
    "c2",
    "c3",
    "c4",
    "c5",
    "c6",
)
OPTIONAL_GRID_CONTOURS = (
    "pharynx",
    "tongue",
)
SPARSE_CORE_THRESHOLDS = {
    "incisior-hard-palate": 10,
    "mandible-incisior": 10,
    "soft-palate-midline": 8,
    "c1": 8,
    "c2": 8,
    "c3": 8,
    "c4": 8,
    "c5": 8,
    "c6": 8,
}


DEFAULT_VIS_STYLE = {
    'image_cmap': 'gray',
    'contour_color': 'gray',
    'contour_alpha': 0.2,
    'mandible_color': 'orange',
    'mandible_alpha': 0.5,
    'tongue_color': '#ff006e',
    'tongue_alpha': 0.9,
    'horiz_color': 'c',
    'vert_color': 'y',
    'vt_color': 'g',
    'spine_color': 'b',
    'guide_color': 'magenta',
    'i_color': 'red',
    'i_label_color': 'red',
    'i_label_fc': 'white',
    'c_color': 'blue',
    'c_label_color': 'blue',
    'c_label_fc': 'white',
    'p1_color': 'red',
    'p1_label_color': 'red',
    'p1_label_fc': 'white',
    'm1_color': 'red',
    'm1_label_color': 'red',
    'm1_label_fc': 'white',
    'l6_color': 'lime',
    'l6_label_color': 'lime',
    'l6_label_fc': 'black',
    'interp_color': 'cyan',
    'interp_label_color': 'cyan',
    'interp_label_fc': 'black',
}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------
def contour_point_counts(contours: Dict[str, np.ndarray]) -> Dict[str, int]:
    """Return point counts for each available contour."""
    counts: Dict[str, int] = {}
    for name, points in contours.items():
        if points is None:
            counts[name] = 0
            continue
        counts[name] = int(len(np.asarray(points, dtype=float)))
    return counts


def validate_grid_contours(contours: Dict[str, np.ndarray]) -> GridContourValidation:
    """Validate contour availability before attempting to build a grid."""
    point_counts = contour_point_counts(contours)
    hard_errors: List[str] = []
    warnings: List[str] = []

    missing_required = [name for name in REQUIRED_GRID_CONTOURS if point_counts.get(name, 0) <= 0]
    if missing_required:
        hard_errors.append(
            "missing_required_contour: missing required contours "
            + ", ".join(missing_required)
        )

    for name, threshold in SPARSE_CORE_THRESHOLDS.items():
        count = point_counts.get(name, 0)
        if 0 < count < threshold:
            warnings.append(
                f"sparse_core_contour: {name} has only {count} points (< {threshold})"
            )

    return GridContourValidation(
        hard_errors=hard_errors,
        warnings=warnings,
        point_counts=point_counts,
    )


def _resample(pts: np.ndarray, n: int = 400) -> np.ndarray:
    """Resample a polyline to *n* equally-spaced points by arc-length."""
    pts = np.asarray(pts, dtype=float)
    d = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    keep = np.r_[True, d > 1e-8]
    pts = pts[keep]
    if len(pts) < 2:
        return np.tile(pts[0], (n, 1))
    arc = np.cumsum(np.r_[0, np.linalg.norm(np.diff(pts, axis=0), axis=1)])
    arc /= arc[-1]
    t = np.linspace(0, 1, n)
    return np.column_stack([np.interp(t, arc, pts[:, 0]),
                            np.interp(t, arc, pts[:, 1])])


def _path_length(path: np.ndarray) -> float:
    return float(np.sum(np.linalg.norm(np.diff(path, axis=0), axis=1)))


def _point_to_line_distance(pt: np.ndarray, line_start: np.ndarray, line_end: np.ndarray) -> float:
    """Distance from a 2D point to the infinite line through two points."""
    vec = np.asarray(line_end, dtype=float) - np.asarray(line_start, dtype=float)
    rel = np.asarray(pt, dtype=float) - np.asarray(line_start, dtype=float)
    norm = np.linalg.norm(vec)
    if norm < 1e-8:
        return float(np.linalg.norm(rel))
    cross_z = vec[0] * rel[1] - vec[1] * rel[0]
    return float(abs(cross_z) / norm)


def _find_pharynx_intersection(start_pt, end_pt, contours: dict) -> Optional[np.ndarray]:
    """Return the pharynx intersection on the segment start_pt -> end_pt."""
    if 'pharynx' not in contours or contours['pharynx'] is None:
        return None

    try:
        from shapely.geometry import LineString

        start_pt = np.asarray(start_pt, dtype=float)
        end_pt = np.asarray(end_pt, dtype=float)
        seg = LineString([tuple(start_pt), tuple(end_pt)])
        inter = seg.intersection(LineString(contours['pharynx']))
        if inter.is_empty:
            return None
        if inter.geom_type == 'Point':
            return np.array([inter.x, inter.y], dtype=float)
        if inter.geom_type == 'MultiPoint':
            pts = [np.array([p.x, p.y], dtype=float) for p in inter.geoms]
            return pts[int(np.argmin([np.linalg.norm(p - start_pt) for p in pts]))]
        if inter.geom_type == 'GeometryCollection':
            pts = []
            for geom in inter.geoms:
                if geom.geom_type == 'Point':
                    pts.append(np.array([geom.x, geom.y], dtype=float))
            if pts:
                return pts[int(np.argmin([np.linalg.norm(p - start_pt) for p in pts]))]
    except Exception:
        return None

    return None


def _find_cervical_centers(contours: dict) -> dict:
    """Centroids of c1-c6."""
    centers = {}
    for cv in ('c1', 'c2', 'c3', 'c4', 'c5', 'c6'):
        if cv in contours and len(contours[cv]) > 0:
            centers[cv] = (float(np.mean(contours[cv][:, 0])),
                           float(np.mean(contours[cv][:, 1])))
    return centers


def _find_I_points(contours: dict) -> Tuple[Optional[dict], Optional[np.ndarray]]:
    """Return (I_points dict, smoothed_path) or (None, None)."""
    key = 'incisior-hard-palate'
    if key not in contours:
        return None, None

    ihp = contours[key]
    n = len(ihp)
    pct = max(1, int(0.05 * n))

    # I1 = lowest-left, I5 = lowest-right
    left_candidates = ihp[np.argsort(ihp[:, 0])][:pct]
    I1 = left_candidates[np.argmax(left_candidates[:, 1])]
    right_candidates = ihp[np.argsort(ihp[:, 0])[::-1]][:pct]
    I5 = right_candidates[np.argmax(right_candidates[:, 1])]

    # Shortest contour path between I1 and I5
    li = np.where((ihp == I1).all(axis=1))[0][0]
    ri = np.where((ihp == I5).all(axis=1))[0][0]
    if ri > li:
        p_fwd = ihp[li:ri + 1]
        p_bwd = np.concatenate([ihp[ri:], ihp[:li + 1][::-1]])
    else:
        p_fwd = np.concatenate([ihp[li:], ihp[:ri + 1]])
        p_bwd = ihp[ri:li + 1][::-1]
    path = p_fwd if _path_length(p_fwd) <= _path_length(p_bwd) else p_bwd

    # Ensure path runs I1 → I5 (not reversed)
    d_start_I1 = np.linalg.norm(path[0] - I1)
    d_start_I5 = np.linalg.norm(path[0] - I5)
    if d_start_I5 < d_start_I1:
        path = path[::-1]

    # Smooth via spline (fall back to raw if it fails)
    # Adapt smoothing to number of points (sparse ROI data needs less)
    try:
        from scipy.interpolate import splprep, splev
        dists = np.cumsum(np.r_[0, np.linalg.norm(np.diff(path, axis=0), axis=1)])
        dists /= dists[-1]
        k_order = min(3, len(path) - 1)
        s_val = min(50, max(1, len(path) * 0.5))   # adaptive smoothing
        tck, _ = splprep([path[:, 0], path[:, 1]], u=dists, s=s_val, k=k_order)
        smooth = np.column_stack(splev(np.linspace(0, 1, 400), tck))
    except Exception:
        smooth = _resample(path, 400)

    # Equally-spaced I1-I5
    cdist = np.cumsum(np.r_[0, np.linalg.norm(np.diff(smooth, axis=0), axis=1)])
    total = cdist[-1]
    I_points = {'I1': I1, 'I5': I5}
    for k, label in enumerate(('I2', 'I3', 'I4')):
        idx = np.argmin(np.abs(cdist - (k + 1) / 4 * total))
        I_points[label] = smooth[idx]

    return I_points, smooth


def _find_soft_palate_points(
    contours: dict,
    c1: Optional[np.ndarray] = None,
    *,
    smooth_strength: float = 12.0,
    n_resample: int = 200,
    middle_weight: float = 0.35,
    gradient_weight: float = 0.45,
    overlap_weight: float = 0.0,
    middle_target: float = 0.55,
    overlap_sigma: float = 0.10,
    search_start: float = 0.35,
    search_end: float = 0.80,
) -> Tuple[Optional[dict], Optional[np.ndarray]]:
    """
    Return soft-palate landmarks and contour path.

    I6 is the first point of ``soft-palate-midline``.
    I7 is refined from a weighted score that favours:
    - points near the middle of the midline,
    - points with high tangent-angle gradient,
    - optionally, points near the current overlap-based I7 from the I6->C1 line.
    """
    key = 'soft-palate-midline'
    if key not in contours or len(contours[key]) == 0:
        return None, None

    path = np.asarray(contours[key], dtype=float)
    if len(path) == 1:
        return {'I6': path[0], 'I7': path[0]}, path

    # Remove repeated points so the smoothing/gradient is not dominated by
    # zero-length steps.
    keep_idx = [0]
    for idx in range(1, len(path)):
        if np.linalg.norm(path[idx] - path[keep_idx[-1]]) > 1e-6:
            keep_idx.append(idx)
    work_path = path[keep_idx]

    if len(work_path) <= 2:
        return {'I6': path[0], 'I7': work_path[-1]}, path

    # Smooth the midline before computing the final overlap point.
    try:
        from scipy.interpolate import splprep, splev

        arc = np.cumsum(np.r_[0, np.linalg.norm(np.diff(work_path, axis=0), axis=1)])
        arc /= max(arc[-1], 1e-8)
        tck, _ = splprep(
            [work_path[:, 0], work_path[:, 1]],
            u=arc,
            s=smooth_strength,
            k=min(3, len(work_path) - 1),
        )
        u_new = np.linspace(0, 1, n_resample)
        x_smooth, y_smooth = splev(u_new, tck)
        dx, dy = splev(u_new, tck, der=1)
        smooth_path = np.column_stack([x_smooth, y_smooth])
    except Exception:
        smooth_path = _resample(work_path, n_resample)
        u_new = np.linspace(0, 1, len(smooth_path))
        grads = np.gradient(smooth_path, axis=0)
        dx, dy = grads[:, 0], grads[:, 1]

    I6 = np.asarray(path[0], dtype=float)
    smooth_path[0] = I6
    I7 = np.asarray(smooth_path[-1], dtype=float)
    overlap_I7 = None

    if c1 is not None:
        try:
            from shapely.geometry import LineString

            line_I6_C1 = LineString([tuple(I6), tuple(np.asarray(c1, dtype=float))])
            midline = LineString(smooth_path)
            inter = midline.intersection(line_I6_C1)

            candidates = []
            if not inter.is_empty:
                if inter.geom_type == 'Point':
                    candidates = [np.array([inter.x, inter.y], dtype=float)]
                elif inter.geom_type == 'MultiPoint':
                    candidates = [np.array([p.x, p.y], dtype=float) for p in inter.geoms]
                elif inter.geom_type == 'GeometryCollection':
                    for geom in inter.geoms:
                        if geom.geom_type == 'Point':
                            candidates.append(np.array([geom.x, geom.y], dtype=float))

            if candidates:
                line_vec = np.asarray(c1, dtype=float) - I6
                line_len2 = float(np.dot(line_vec, line_vec))
                if line_len2 > 1e-8:
                    # Keep the last intersection along the I6 -> C1 direction.
                    scores = [float(np.dot(pt - I6, line_vec) / line_len2) for pt in candidates]
                    overlap_I7 = candidates[int(np.argmax(scores))]
                    I7 = overlap_I7
        except Exception:
            pass

    # Refine I7 with a weighted score that prefers a point near the middle
    # of the midline, with strong local bending, while staying near the
    # overlap-based I7 found above.
    angles_deg = np.degrees(np.unwrap(np.arctan2(dy, dx)))
    grad_score = np.abs(np.gradient(angles_deg))
    grad_score /= max(float(np.max(grad_score)), 1e-8)

    lo = int(np.clip(search_start, 0.0, 1.0) * (len(smooth_path) - 1))
    hi = int(np.clip(search_end, 0.0, 1.0) * (len(smooth_path) - 1))
    hi = max(lo + 1, hi)

    if overlap_weight > 0.0 and overlap_I7 is not None:
        overlap_idx = int(np.argmin(np.linalg.norm(
            smooth_path - overlap_I7[None, :], axis=1)))
        overlap_u = float(u_new[overlap_idx])
    else:
        overlap_u = middle_target

    best_idx = lo
    best_score = -np.inf
    for idx in range(lo, hi + 1):
        middle_score = max(0.0, 1.0 - abs(float(u_new[idx]) - middle_target) / max(middle_target, 1e-8))
        near_overlap = 0.0
        if overlap_weight > 0.0:
            near_overlap = float(np.exp(-0.5 * ((float(u_new[idx]) - overlap_u) / max(overlap_sigma, 1e-8)) ** 2))
        score = (
            middle_weight * middle_score
            + gradient_weight * float(grad_score[idx])
            + overlap_weight * near_overlap
        )
        if score > best_score:
            best_score = score
            best_idx = idx

    I7 = np.asarray(smooth_path[best_idx], dtype=float)

    return {'I6': I6, 'I7': I7}, smooth_path


def _find_M1(contours: dict) -> Optional[np.ndarray]:
    """M1 = highest (min-y) point of mandible-incisior."""
    if 'mandible-incisior' not in contours:
        return None
    mc = contours['mandible-incisior']
    return mc[np.argmin(mc[:, 1])] if len(mc) > 0 else None


def _find_L6(contours: dict) -> Optional[np.ndarray]:
    """L6 = lowest (max-y) point of mandible-incisior."""
    if 'mandible-incisior' not in contours:
        return None
    mc = contours['mandible-incisior']
    return mc[np.argmax(mc[:, 1])].astype(float) if len(mc) > 0 else None


def _extend_to_c1(smooth_path, c1, contours):
    """Bézier extension from end of smooth_path to c1; find P1 on pharynx."""
    path = np.asarray(smooth_path, dtype=float)
    c1 = np.asarray(c1, dtype=float)
    last = path[-1]

    # Tangent at end
    if len(path) >= 3:
        tg = (path[-1] - path[-2]) + (path[-2] - path[-3])
    elif len(path) >= 2:
        tg = path[-1] - path[-2]
    else:
        tg = c1 - last
    norm = np.linalg.norm(tg)
    tg = tg / norm if norm > 1e-8 else (c1 - last) / max(np.linalg.norm(c1 - last), 1e-8)

    # Find P1 (pharynx intersection)
    P1 = _find_pharynx_intersection(last, c1, contours)

    # Incoming direction at C1
    if P1 is not None:
        u = c1 - P1
        un = np.linalg.norm(u)
        u = u / un if un > 1e-8 else tg
    else:
        u = (c1 - last)
        un = np.linalg.norm(u)
        u = u / un if un > 1e-8 else tg

    L = np.linalg.norm(c1 - last)
    if L < 1e-6:
        return path.copy(), P1

    d = np.clip(0.35 * L, 0.10 * L, 0.60 * L)
    B0, B3 = last, c1
    B1 = B0 + d * tg
    B2 = B3 - 0.85 * d * u
    t_arr = np.linspace(0, 1, 120)[:, None]
    omt = 1 - t_arr
    ext = omt**3 * B0 + 3 * omt**2 * t_arr * B1 + 3 * omt * t_arr**2 * B2 + t_arr**3 * B3
    full = np.vstack([path, ext[1:]])
    return full, P1


def _clamp_path_to_chord(
    path: np.ndarray,
    start_pt: np.ndarray,
    end_pt: np.ndarray,
    max_dev_ratio: float,
) -> Tuple[np.ndarray, float, float]:
    """Clamp a path's perpendicular deviation from the start->end chord."""
    pts = np.asarray(path, dtype=float)
    start = np.asarray(start_pt, dtype=float)
    end = np.asarray(end_pt, dtype=float)
    if len(pts) == 0:
        return np.vstack([start, end]), 0.0, 0.0

    chord = end - start
    chord_len = float(np.linalg.norm(chord))
    if chord_len < 1e-8:
        return np.tile(start, (len(pts), 1)), 0.0, 0.0

    unit = chord / chord_len
    normal = np.array([-unit[1], unit[0]], dtype=float)
    rel = pts - start[None, :]

    along = rel @ unit
    along = np.clip(along, 0.0, chord_len)
    along = np.maximum.accumulate(along)

    perp = rel @ normal
    max_dev = max(float(max_dev_ratio), 0.0) * chord_len
    if max_dev <= 0.0:
        clamped_perp = np.zeros_like(perp)
        clamped_mask = np.abs(perp) > 1e-8
    else:
        clamped_perp = np.clip(perp, -max_dev, max_dev)
        clamped_mask = np.abs(perp) > max_dev

    clamped = start[None, :] + along[:, None] * unit[None, :] + clamped_perp[:, None] * normal[None, :]
    clamped[0] = start
    clamped[-1] = end

    max_dev_ratio_observed = float(np.max(np.abs(perp)) / chord_len) if len(perp) else 0.0
    clamp_fraction = float(np.count_nonzero(clamped_mask) / max(len(clamped_mask), 1))
    return clamped, max_dev_ratio_observed, clamp_fraction


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def build_grid(
    image,
    contours: Dict[str, np.ndarray],
    *,
    n_vert: int = 9,
    n_points: int = 250,
    frame_number: int = 0,
    max_dev_ratio: float = 0.25,
) -> GridData:
    """
    Build an anatomically-anchored curvilinear grid.

    Parameters
    ----------
    image : PIL.Image or np.ndarray
        The MRI frame (used only for visualization later).
    contours : dict[str, ndarray]
        Articulator contours.  Keys expected:
        ``incisior-hard-palate``, ``mandible-incisior``,
        ``c1``-``c6``, and optionally ``pharynx``.
    n_vert : int
        Number of vertical grid lines (default 9).
    n_points : int
        Number of sample points per grid line (default 250).
    frame_number : int
        Frame identifier (for labeling only).

    Returns
    -------
    GridData
        Dataclass with all grid curves, anchor points, and metadata.
    """
    NP = n_points
    g = GridData()
    g.image = image
    g.contours = contours
    g.frame_number = frame_number
    g.n_vert = n_vert
    validation = validate_grid_contours(contours)
    g.contour_point_counts = dict(validation.point_counts)
    g.hard_errors = list(validation.hard_errors)
    g.warnings = list(validation.warnings)
    if validation.hard_errors:
        raise GridValidationError("; ".join(validation.hard_errors))

    # --- Cervical centres ---
    cc = _find_cervical_centers(contours)
    g.cervical_centers = cc
    sk = [k for k in ('c1', 'c2', 'c3', 'c4', 'c5', 'c6') if k in cc]
    spine_pts = np.array([cc[k] for k in sk], dtype=float)  # (n_h, 2)
    n_h = len(sk)
    g.n_horiz = n_h

    # --- I points & VT curve ---
    I_points, smooth_path = _find_I_points(contours)
    g.I_points = I_points
    g.I1 = np.asarray(I_points['I1'], dtype=float) if I_points else None

    c1 = np.array(cc['c1'], dtype=float) if 'c1' in cc else None
    soft_palate_points, soft_palate_path = _find_soft_palate_points(contours, c1=c1)
    if I_points and soft_palate_points:
        I_points = dict(I_points)
        I_points.update(soft_palate_points)
        g.I_points = I_points

    vt_seed = smooth_path
    if vt_seed is not None and soft_palate_path is not None:
        if np.linalg.norm(vt_seed[-1] - soft_palate_path[0]) < 1e-6:
            vt_seed = np.vstack([vt_seed, soft_palate_path[1:]])
        else:
            vt_seed = np.vstack([vt_seed, soft_palate_path])

    # Build the H1 / VT path as I1 -> ... -> I6 -> I7 -> C1 when possible.
    vt_full = None
    if I_points and c1 is not None:
        P1_i7 = None
        vt_parts = []

        if smooth_path is not None:
            vt_parts.append(np.asarray(smooth_path, dtype=float))

        if 'I6' in I_points and 'I7' in I_points:
            i6 = np.asarray(I_points['I6'], dtype=float)
            i7 = np.asarray(I_points['I7'], dtype=float)
            soft_part = np.vstack([i6, i7])
            if len(vt_parts) > 0 and np.linalg.norm(vt_parts[-1][-1] - soft_part[0]) < 1e-6:
                vt_parts.append(soft_part[1:])
            else:
                vt_parts.append(soft_part)

            P1_i7 = _find_pharynx_intersection(i7, c1, contours)

        vt_parts.append(np.asarray(c1, dtype=float).reshape(1, 2))
        vt_full = np.vstack([part for part in vt_parts if len(part) > 0])
        g.vt_curve = _resample(vt_full, NP)
        g.P1_point = P1_i7
    else:
        g.vt_curve = _resample(vt_seed, NP) if vt_seed is not None else None

    if g.P1_point is None:
        g.warnings.append("missing_P1: no pharynx intersection found for the H1 tail")

    # --- M1 & L6 ---
    M1 = _find_M1(contours)
    L6 = _find_L6(contours)
    g.M1 = np.asarray(M1, dtype=float) if M1 is not None else None
    g.M1_point = g.M1
    g.L6 = np.asarray(L6, dtype=float) if L6 is not None else None

    # --- Left / Right endpoints ---
    right_pts = spine_pts.copy()
    spine_arc = np.cumsum(np.r_[0, np.linalg.norm(np.diff(spine_pts, axis=0), axis=1)])
    arc_c2, arc_c6 = spine_arc[1], spine_arc[-1]

    left_pts = np.zeros((n_h, 2), dtype=float)
    left_pts[0] = g.I1                          # I1  -> C1
    left_pts[1] = g.M1                          # M1  -> C2
    left_pts[-1] = g.L6                         # L6  -> C6
    for i in range(2, n_h - 1):                  # L3-L5 interpolated
        frac = (spine_arc[i] - arc_c2) / (arc_c6 - arc_c2)
        left_pts[i] = g.M1 + frac * (g.L6 - g.M1)

    g.left_pts = left_pts
    g.right_pts = right_pts

    # --- VT-curve shape offset (gives curvature to horiz lines) ---
    u = np.linspace(0, 1, NP)
    vt = g.vt_curve
    vt_base = vt[0][None, :] + u[:, None] * (vt[-1] - vt[0])[None, :]
    vt_offset = vt - vt_base

    # --- Horizontal lines ---
    g.horiz_lines = []
    for i in range(n_h):
        L, R = left_pts[i], right_pts[i]
        base = L[None, :] + u[:, None] * (R - L)[None, :]
        vt_len = max(np.linalg.norm(vt[-1] - vt[0]), 1e-8)
        scale = np.linalg.norm(R - L) / vt_len
        if i == 0 and g.I_points is not None:
            # ---- H1: composite curve  L1 -> I5 -> C1 ----
            #
            # Part 1 (L1 -> I5): follows the palate segmentation
            #   contour directly from the VT curve.
            #
            # Part 2 (I5 -> C1): blended overlap path between
            #   the soft-palate midline and the direct chord.
            #   Each midline point is decomposed into along-chord
            #   and perpendicular components.  The perpendicular
            #   component is clamped to ±max_dev so the path
            #   follows the midline where it stays close to the
            #   chord, but is pulled back when it strays too far.

            I5 = np.asarray(g.I_points['I5'], dtype=float)
            h1_source = np.asarray(vt_full if vt_full is not None else vt, dtype=float)
            i5_idx = int(np.argmin(np.linalg.norm(h1_source - I5[None, :], axis=1)))
            palate_part = h1_source[:i5_idx + 1].copy()
            tail_part = h1_source[i5_idx:].copy()
            if len(palate_part) == 0:
                palate_part = I5.reshape(1, 2)
            if len(tail_part) == 0:
                tail_part = np.vstack([I5, R])
            tail_blended, h1_dev_ratio, h1_max_clamp = _clamp_path_to_chord(
                tail_part,
                I5,
                R,
                max_dev_ratio,
            )
            if np.linalg.norm(palate_part[-1] - tail_blended[0]) < 1e-6:
                h1_source = np.vstack([palate_part, tail_blended[1:]])
            else:
                h1_source = np.vstack([palate_part, tail_blended])
            h1 = _resample(h1_source, NP)
            h1[0] = L    # exact L1 (= I1)
            h1[-1] = R   # exact C1

            # Store H1 metadata.
            g.palate_end_idx = int(np.argmin(
                np.linalg.norm(h1 - I5[None, :], axis=1)))
            g.h1_dev_ratio = h1_dev_ratio
            g.h1_max_clamp = h1_max_clamp

            g.horiz_lines.append(h1)
        else:
            blend = 1.0 - i / max(n_h - 1, 1)
            g.horiz_lines.append(base + blend * scale * vt_offset)

    # --- Vertical lines ---
    vert_idx = np.linspace(0, NP - 1, n_vert).astype(int)
    g.vert_lines = []
    for vi in vert_idx:
        cross = np.array([g.horiz_lines[i][vi] for i in range(n_h)])
        g.vert_lines.append(_resample(cross, NP))

    # --- Spine reference curve ---
    g.spine_curve = _resample(spine_pts, 400)

    # --- Labels ---
    g.h_labels = ['L1->I6->I7->C1', 'M1->C2']
    for i in range(2, n_h - 1):
        g.h_labels.append(f'L{i + 1}->{sk[i].upper()}')
    g.h_labels.append(f'L6->{sk[-1].upper()}')

    return g


# ---------------------------------------------------------------------------
def visualize_grid(
    grid: GridData,
    *,
    figsize: Tuple[int, int] = (16, 16),
    show_contours: bool = True,
    show_landmarks: bool = True,
    show_labels: bool = True,
    style: Optional[dict] = None,
    ax: Optional[plt.Axes] = None,
) -> plt.Figure:
    """
    Visualise a ``GridData`` object.

    Parameters
    ----------
    grid : GridData
        As returned by ``build_grid``.
    figsize : tuple
        Figure size (only used when *ax* is None).
    show_contours : bool
        Draw faint articulator contours in the background.
    show_landmarks : bool
        Draw I1-I7, C1-C6, M1, L6, P1 markers.
    show_labels : bool
        Draw text labels next to landmarks.
    ax : matplotlib Axes, optional
        Draw on an existing axes instead of creating a new figure.

    Returns
    -------
    matplotlib.figure.Figure
    """
    created_axes = ax is None
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure
    style_cfg = dict(DEFAULT_VIS_STYLE)
    if style:
        style_cfg.update(style)

    # Image
    ax.imshow(grid.image, cmap=style_cfg['image_cmap'])
    title = f"Frame {grid.frame_number}: Anatomical Grid"
    title += f"\n{style_cfg['horiz_color']} = horizontal | {style_cfg['vert_color']} = vertical"
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.axis('off')

    # Clamp axes to image bounds so grid lines outside don't shrink the image
    try:
        img_arr = np.asarray(grid.image)
        h, w = img_arr.shape[:2]
        ax.set_xlim(0, w)
        ax.set_ylim(h, 0)
    except Exception:
        pass

    # Faint contours
    if show_contours and grid.contours:
        for name, co in grid.contours.items():
            ax.plot(co[:, 0], co[:, 1], color=style_cfg['contour_color'], lw=0.5,
                    alpha=style_cfg['contour_alpha'])

    # Highlight mandible-incisior
    if 'mandible-incisior' in grid.contours:
        mc = grid.contours['mandible-incisior']
        ax.plot(mc[:, 0], mc[:, 1], color=style_cfg['mandible_color'], lw=2,
                alpha=style_cfg['mandible_alpha'],
                label='Mandible-incisior')

    # Highlight tongue explicitly when available.
    if 'tongue' in grid.contours:
        tongue = grid.contours['tongue']
        ax.plot(
            tongue[:, 0],
            tongue[:, 1],
            color=style_cfg['tongue_color'],
            lw=2.2,
            alpha=style_cfg['tongue_alpha'],
            label='Tongue',
            zorder=13,
        )

    # Horizontal grid lines (cyan)
    n_h = grid.n_horiz
    for i, line in enumerate(grid.horiz_lines):
        lw = 2.5 if i == 0 or i == n_h - 1 else 1.2
        ax.plot(line[:, 0], line[:, 1], '-', color=style_cfg['horiz_color'], lw=lw, alpha=0.7)

    # Vertical grid lines (yellow)
    for j, line in enumerate(grid.vert_lines):
        lw = 2.5 if j == 0 or j == grid.n_vert - 1 else 1.0
        ax.plot(line[:, 0], line[:, 1], '-', color=style_cfg['vert_color'], lw=lw, alpha=0.65)

    # VT reference curve (green)
    if grid.vt_curve is not None:
        ax.plot(grid.vt_curve[:, 0], grid.vt_curve[:, 1],
                '-', color=style_cfg['vt_color'], lw=3.5, label='VT (I1->C1)', zorder=15)

    # Spine reference curve (blue)
    if grid.spine_curve is not None:
        ax.plot(grid.spine_curve[:, 0], grid.spine_curve[:, 1],
                '-', color=style_cfg['spine_color'], lw=3.5, label='Spine (C1->C6)', zorder=15)

    if show_landmarks:
        _draw_landmarks(ax, grid, show_labels, style_cfg)

    ax.legend(loc='upper right', fontsize=12, framealpha=0.92)
    if created_axes:
        plt.tight_layout()
    return fig


def _draw_landmarks(ax, g: GridData, show_labels: bool, style_cfg: dict):
    """Draw all anatomical landmarks on *ax*."""
    # I points (red circles)
    if g.I_points:
        for lb in ('I1', 'I2', 'I3', 'I4', 'I5', 'I6', 'I7'):
            if lb not in g.I_points:
                continue
            p = g.I_points[lb]
            ax.plot(p[0], p[1], 'o', color=style_cfg['i_color'], ms=9, mec='white', mew=2, zorder=16)
            if show_labels:
                ha = 'right' if lb == 'I1' else 'left'
                dx = -12 if lb == 'I1' else 8
                ax.text(p[0] + dx, p[1] - 4, lb, fontsize=10, color=style_cfg['i_label_color'],
                        fontweight='bold', ha=ha,
                        bbox=dict(boxstyle='round,pad=0.25', fc=style_cfg['i_label_fc'], alpha=0.85))

    # Show the I6 -> I7 -> C1 guide, with P1 marked on the I7 -> C1 segment.
    if g.I_points and 'I6' in g.I_points and 'I7' in g.I_points:
        guide_pts = [
            np.asarray(g.I_points['I6'], dtype=float),
            np.asarray(g.I_points['I7'], dtype=float),
        ]
        if g.P1_point is not None:
            guide_pts.append(np.asarray(g.P1_point, dtype=float))
        if g.cervical_centers and 'c1' in g.cervical_centers:
            guide_pts.append(np.asarray(g.cervical_centers['c1'], dtype=float))
        guide_pts = np.asarray(guide_pts, dtype=float)
        ax.plot(guide_pts[:, 0], guide_pts[:, 1], '--', color=style_cfg['guide_color'],
                lw=2.0, alpha=0.8, zorder=14, label='I6-I7-C1 guide with P1')

    # C points (blue circles)
    if g.cervical_centers:
        for cv in ('c1', 'c2', 'c3', 'c4', 'c5', 'c6'):
            if cv in g.cervical_centers:
                p = g.cervical_centers[cv]
                ax.plot(p[0], p[1], 'o', color=style_cfg['c_color'], ms=9, mec='white', mew=2, zorder=16)
                if show_labels:
                    ax.text(p[0] - 18, p[1], cv.upper(), fontsize=10,
                            color=style_cfg['c_label_color'], fontweight='bold', ha='right',
                            bbox=dict(boxstyle='round,pad=0.25', fc=style_cfg['c_label_fc'], alpha=0.85))

    # P1 (red circle)
    if g.P1_point is not None:
        p = g.P1_point
        ax.plot(p[0], p[1], 'o', color=style_cfg['p1_color'], ms=9, mec='white', mew=2, zorder=16)
        if show_labels:
            ax.text(p[0] + 8, p[1] - 4, 'P1', fontsize=10, color=style_cfg['p1_label_color'],
                    fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.25', fc=style_cfg['p1_label_fc'], alpha=0.85))

    # M1 (red circle)
    if g.M1 is not None:
        ax.plot(g.M1[0], g.M1[1], 'o', color=style_cfg['m1_color'], ms=9, mec='white', mew=2, zorder=16)
        if show_labels:
            ax.text(g.M1[0] + 8, g.M1[1], 'M1', fontsize=10, color=style_cfg['m1_label_color'],
                    fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.25', fc=style_cfg['m1_label_fc'], alpha=0.85))

    # L6 (lime diamond)
    if g.L6 is not None:
        ax.plot(g.L6[0], g.L6[1], 'D', color=style_cfg['l6_color'], ms=10, mec='white',
                mew=2, zorder=16)
        if show_labels:
            ax.text(g.L6[0] + 8, g.L6[1], 'L6', fontsize=10, color=style_cfg['l6_label_color'],
                    fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.25', fc=style_cfg['l6_label_fc'], alpha=0.8))

    # Interpolated L3-L5 (cyan squares)
    n_h = g.n_horiz
    if g.left_pts is not None:
        for i in range(2, n_h - 1):
            p = g.left_pts[i]
            ax.plot(p[0], p[1], 's', color=style_cfg['interp_color'], ms=7, mec='white',
                    mew=1.5, zorder=16)
            if show_labels:
                ax.text(p[0] - 12, p[1], f'L{i + 1}', fontsize=9,
                        color=style_cfg['interp_label_color'], fontweight='bold', ha='right',
                        bbox=dict(boxstyle='round,pad=0.2', fc=style_cfg['interp_label_fc'], alpha=0.7))


# ---------------------------------------------------------------------------
def build_and_show_grid(
    image,
    contours: Dict[str, np.ndarray],
    *,
    n_vert: int = 9,
    n_points: int = 250,
    frame_number: int = 0,
    figsize: Tuple[int, int] = (16, 16),
    show: bool = True,
) -> GridData:
    """
    Convenience wrapper: build the grid **and** visualise it.

    Parameters
    ----------
    image, contours, n_vert, n_points, frame_number
        Forwarded to ``build_grid``.
    figsize : tuple
        Figure size.
    show : bool
        Call ``plt.show()`` automatically.

    Returns
    -------
    GridData
    """
    grid = build_grid(image, contours,
                      n_vert=n_vert, n_points=n_points,
                      frame_number=frame_number)
    fig = visualize_grid(grid, figsize=figsize)
    if show:
        plt.show()

    # Print summary
    print(f"Grid: {grid.n_horiz} horizontal x {grid.n_vert} vertical")
    for i in range(grid.n_horiz):
        lbl = grid.h_labels[i] if i < len(grid.h_labels) else f"H{i+1}"
        print(f"  H{i + 1}: {lbl}  "
              f"left={grid.left_pts[i].round(1)}  "
              f"right={grid.right_pts[i].round(1)}")

    return grid


# ---------------------------------------------------------------------------
def print_grid_summary(grid: GridData) -> None:
    """Print a text summary of the grid to stdout."""
    print(f"\n{'=' * 60}")
    print(f"GRID SUMMARY – Frame {grid.frame_number}")
    print(f"{'=' * 60}")
    print(f"Horizontal lines: {grid.n_horiz}")
    print(f"Vertical lines:   {grid.n_vert}")

    print("\nHorizontal line endpoints:")
    for i in range(grid.n_horiz):
        lbl = grid.h_labels[i] if i < len(grid.h_labels) else f"H{i+1}"
        print(f"  H{i + 1} ({lbl}):  "
              f"left=({grid.left_pts[i][0]:.1f}, {grid.left_pts[i][1]:.1f})  "
              f"right=({grid.right_pts[i][0]:.1f}, {grid.right_pts[i][1]:.1f})")

    print("\nLandmarks:")
    if grid.I_points:
        for lb in ('I1', 'I2', 'I3', 'I4', 'I5', 'I6', 'I7'):
            if lb not in grid.I_points:
                continue
            p = grid.I_points[lb]
            print(f"  {lb}: ({p[0]:.1f}, {p[1]:.1f})")
    if grid.M1 is not None:
        print(f"  M1: ({grid.M1[0]:.1f}, {grid.M1[1]:.1f})")
    if grid.L6 is not None:
        print(f"  L6: ({grid.L6[0]:.1f}, {grid.L6[1]:.1f})")
    if grid.P1_point is not None:
        print(f"  P1: ({grid.P1_point[0]:.1f}, {grid.P1_point[1]:.1f})")
    if grid.cervical_centers:
        for cv in ('c1', 'c2', 'c3', 'c4', 'c5', 'c6'):
            if cv in grid.cervical_centers:
                p = grid.cervical_centers[cv]
                print(f"  {cv.upper()}: ({p[0]:.1f}, {p[1]:.1f})")

    if grid.vt_curve is not None:
        print(f"\nVT curve length: {_path_length(grid.vt_curve):.1f} px")
    print(f"{'=' * 60}\n")
