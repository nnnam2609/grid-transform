from __future__ import annotations

"""Render nine-speaker S5 pourri #2 contour/landmark/transform sheets.

P2 is excluded because its selected /u/ frame has no vocal-fold contour.  The
dynamic S5 contours are never used to refit geometry: each fixed speaker-to-
reference mapping is calibrated from the curated VTLN reference grids, then
applied unchanged to the selected S5 contours.  No post-transform smoothing is
used.
"""

import argparse
import csv
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import cv2
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from grid_transform.analysis_shared import load_curated_speakers
from grid_transform.config import DEFAULT_VTLN_DIR, PROJECT_DIR
from grid_transform.fixed_target_stage_overlays import (
    choose_speaker_colors,
    save_stage_contour_overlay_figure,
)
from grid_transform.transfer import build_two_step_transform
from grid_transform.transform_helpers import (
    apply_tps,
    apply_transform,
    build_step2_controls,
    extract_true_landmarks,
    fit_tps,
    map_landmarks,
)
from grid_transform.vtln_bundle import scale_contours_to_triplet_space
from grid_transform.warp import warp_image_to_target_space


SESSION = "S5"
REFERENCE_SPEAKER = "P7"
EXCLUDED_SPEAKERS = ("P2",)
SPEAKERS = ("P1", "P3", "P4", "P5", "P6", "P7", "P8", "P9", "P10")
SOURCE_SHAPE = (136, 136)
TARGET_SHAPE = (480, 480)
NATIVE_MM_PER_PIXEL = 1.62
TARGET_MM_PER_PIXEL = NATIVE_MM_PER_PIXEL * SOURCE_SHAPE[1] / TARGET_SHAPE[1]
MRI_CROP = (90, 92, 270, 270)  # x0, y0, width, height in the 450x600 review AVI
MRI_WARP_BOUNDARY_TRIM_PX = 6

CONTOUR_ORDER = (
    "arytenoid-cartilage",
    "epiglottis",
    "lower-incisor",
    "lower-lip",
    "pharynx",
    "soft-palate-midline",
    "thyroid-cartilage",
    "tongue",
    "upper-incisor",
    "upper-lip",
    "vocal-folds",
)
AFFINE_CONTROL_ORDER = (
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
)
TPS_CONTROL_ORDER = AFFINE_CONTROL_ORDER + ("M1", "L6")
LOWER_SHAPE_CONTROL_ORDER = ("M-", "M+")
CERVICAL_ORDER = ("C1", "C2", "C3", "C4", "C5", "C6")
STAGES = ("ready", "affine", "affine_tps")
METRIC_STAGES = ("affine", "affine_tps")

CONTOUR_COLORS = {
    "arytenoid-cartilage": "#ff595e",
    "epiglottis": "#ff924c",
    "lower-incisor": "#ffca3a",
    "lower-lip": "#c5e327",
    "pharynx": "#52b788",
    "soft-palate-midline": "#2ec4b6",
    "thyroid-cartilage": "#00b4d8",
    "tongue": "#4361ee",
    "upper-incisor": "#9b5de5",
    "upper-lip": "#f15bb5",
    "vocal-folds": "#f8f9fa",
}
LANDMARK_COLORS = {
    "I1_I5": "#39d353",
    "I6_I7": "#58a6ff",
    "C1_C6": "#00e5ff",
    "P1_M1_L6": "#ff9f1c",
}

DEFAULT_SELECTION_CSV = (
    PROJECT_DIR
    / "outputs"
    / "s5_pourri_repetitions_p1_p10_20260814"
    / "pourri_three_repetitions_p1_p10.csv"
)
DEFAULT_ALIGNMENT_SUMMARY = DEFAULT_SELECTION_CSV.parent / "render_summary.json"
DEFAULT_INFERENCE_ROOT = PROJECT_DIR.parent / "Preprocess" / "inference"
DEFAULT_OLD_ROOT = PROJECT_DIR.parent / "Data" / "Need-to-verify-alignment"
DEFAULT_VT_TOOLS_ROOT = PROJECT_DIR.parent / "vt_tools"
DEFAULT_OUTPUT_DIR = PROJECT_DIR / "outputs" / "s5_pourri2_p7_affine_tps_20260814"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Render P1/P3-P10 S5 pourri #2 11-contour sheets before, after affine, "
            "and after affine+TPS in a selected reference-speaker space."
        )
    )
    parser.add_argument("--selection-csv", type=Path, default=DEFAULT_SELECTION_CSV)
    parser.add_argument("--alignment-summary", type=Path, default=DEFAULT_ALIGNMENT_SUMMARY)
    parser.add_argument("--inference-root", type=Path, default=DEFAULT_INFERENCE_ROOT)
    parser.add_argument(
        "--review-root",
        type=Path,
        default=None,
        help=(
            "Optional review_s5_pourri2 root. When supplied, load the frozen "
            "per-case contours_npy snapshots instead of mutable inference contours."
        ),
    )
    parser.add_argument("--old-root", type=Path, default=DEFAULT_OLD_ROOT)
    parser.add_argument("--vtln-dir", type=Path, default=DEFAULT_VTLN_DIR)
    parser.add_argument("--vt-tools-root", type=Path, default=DEFAULT_VT_TOOLS_ROOT)
    parser.add_argument(
        "--reference-speaker",
        choices=SPEAKERS,
        default=REFERENCE_SPEAKER,
        help="Reference speaker for the fixed curated transform and dynamic P2CP target.",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"No rows to write: {path}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def load_pourri2_selection(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        selected = [
            row
            for row in csv.DictReader(handle)
            if row["candidate"] == "pourri/r2" and row["speaker"] not in EXCLUDED_SPEAKERS
        ]
    actual = tuple(row["speaker"] for row in selected)
    if actual != SPEAKERS:
        raise ValueError(f"Expected pourri/r2 rows for {SPEAKERS}, got {actual}")
    for row in selected:
        expected_source = "old_textgrid_fallback" if row["speaker"] == "P10" else "incoming_*_c"
        if row["alignment_source"] != expected_source:
            raise ValueError(
                f"Unexpected alignment source for {row['speaker']}: {row['alignment_source']}"
            )
    return selected


def load_frame_contours(inference_root: Path, speaker: str, frame: int) -> dict[str, np.ndarray]:
    contour_dir = inference_root / speaker / SESSION / "contours"
    paths = sorted(contour_dir.glob(f"{frame:04d}_*.npy"))
    prefix = f"{frame:04d}_"
    contours = {
        path.stem[len(prefix) :]: np.asarray(np.load(path, allow_pickle=False), dtype=float)
        for path in paths
    }
    if set(contours) != set(CONTOUR_ORDER):
        missing = sorted(set(CONTOUR_ORDER) - set(contours))
        extra = sorted(set(contours) - set(CONTOUR_ORDER))
        raise ValueError(f"{speaker} F{frame:04d}: missing={missing}, extra={extra}")
    for label, points in contours.items():
        if points.shape != (50, 2) or not np.isfinite(points).all():
            raise ValueError(f"Invalid {speaker} F{frame:04d} {label}: {points.shape}")
    return scale_contours_to_triplet_space(contours, SOURCE_SHAPE, TARGET_SHAPE)


def load_review_contours(review_root: Path, speaker: str, frame: int) -> dict[str, np.ndarray]:
    case_dir = review_root / "cases" / f"{speaker}_{SESSION}_F{frame:04d}"
    metadata_path = case_dir / "metadata.json"
    contour_dir = case_dir / "contours_npy"
    if not metadata_path.is_file():
        raise FileNotFoundError(metadata_path)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if (
        metadata.get("speaker") != speaker
        or metadata.get("session") != SESSION
        or int(metadata.get("frame", -1)) != frame
    ):
        raise ValueError(f"Review metadata does not match requested case: {metadata_path}")
    contours = {
        path.stem: np.asarray(np.load(path, allow_pickle=False), dtype=float)
        for path in sorted(contour_dir.glob("*.npy"))
    }
    if set(contours) != set(CONTOUR_ORDER):
        missing = sorted(set(CONTOUR_ORDER) - set(contours))
        extra = sorted(set(contours) - set(CONTOUR_ORDER))
        raise ValueError(f"{case_dir.name}: missing={missing}, extra={extra}")
    for label, points in contours.items():
        if points.shape != (50, 2) or not np.isfinite(points).all():
            raise ValueError(f"Invalid {case_dir.name} {label}: {points.shape}")
    return scale_contours_to_triplet_space(contours, SOURCE_SHAPE, TARGET_SHAPE)


def crop_mri_panel(frame: np.ndarray) -> np.ndarray:
    if frame.shape[:2] != (600, 450):
        raise ValueError(f"Expected 450x600 review frame, got {frame.shape}")
    x0, y0, width, height = MRI_CROP
    crop = frame[y0 : y0 + height, x0 : x0 + width]
    if crop.shape[:2] != (height, width):
        raise ValueError(f"Invalid MRI crop: {crop.shape}")
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    return cv2.resize(gray, TARGET_SHAPE[::-1], interpolation=cv2.INTER_CUBIC)


def load_mri_frame(old_root: Path, speaker: str, frame: int) -> np.ndarray:
    path = old_root / speaker / SESSION / f"VIDEO_{speaker}_{SESSION}.avi"
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise FileNotFoundError(path)
    capture.set(cv2.CAP_PROP_POS_FRAMES, frame - 1)
    ok, image = capture.read()
    capture.release()
    if not ok:
        raise RuntimeError(f"Could not decode {speaker} F{frame:04d}: {path}")
    return crop_mri_panel(image)


def import_p2cp(vt_tools_root: Path) -> tuple[Callable[..., float], Callable[..., float]]:
    metrics_path = vt_tools_root / "vt_tools" / "metrics.py"
    if not metrics_path.is_file():
        raise FileNotFoundError(metrics_path)
    resolved = str(vt_tools_root.resolve())
    if resolved not in sys.path:
        sys.path.insert(0, resolved)
    from vt_tools.metrics import p2cp_mean, p2cp_rms

    return p2cp_mean, p2cp_rms


def identity_landmarks(landmarks: dict[str, np.ndarray | None]) -> dict[str, np.ndarray | None]:
    return {
        name: None if point is None else np.asarray(point, dtype=float).copy()
        for name, point in landmarks.items()
    }


def identity_contours(contours: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {label: np.asarray(points, dtype=float).copy() for label, points in contours.items()}


def identity_mapping(points: np.ndarray) -> np.ndarray:
    return np.asarray(points, dtype=float).copy()


def invert_affine(affine: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    matrix = np.asarray(affine["A"], dtype=float)
    shift = np.asarray(affine["t"], dtype=float)
    inverse_matrix = np.linalg.inv(matrix)
    return {"A": inverse_matrix, "t": -(inverse_matrix @ shift), "type": "affine"}


def refine_inverse_mapping(
    target_points: np.ndarray,
    *,
    forward_mapping: Callable[[np.ndarray], np.ndarray],
    initial_inverse_mapping: Callable[[np.ndarray], np.ndarray],
    iterations: int = 8,
    epsilon: float = 0.25,
) -> np.ndarray:
    """Numerically invert the exact forward map, seeded by a reverse TPS fit."""
    target = np.asarray(target_points, dtype=float)
    single = target.ndim == 1
    if single:
        target = target.reshape(1, 2)
    estimate = np.asarray(initial_inverse_mapping(target), dtype=float).copy()
    dx = np.array([epsilon, 0.0])
    dy = np.array([0.0, epsilon])

    for _ in range(iterations):
        residual = np.asarray(forward_mapping(estimate), dtype=float) - target
        if float(np.max(np.linalg.norm(residual, axis=1))) < 1e-5:
            break
        derivative_x = (
            np.asarray(forward_mapping(estimate + dx), dtype=float)
            - np.asarray(forward_mapping(estimate - dx), dtype=float)
        ) / (2.0 * epsilon)
        derivative_y = (
            np.asarray(forward_mapping(estimate + dy), dtype=float)
            - np.asarray(forward_mapping(estimate - dy), dtype=float)
        ) / (2.0 * epsilon)
        determinant = (
            derivative_x[:, 0] * derivative_y[:, 1]
            - derivative_y[:, 0] * derivative_x[:, 1]
        )
        valid = np.isfinite(determinant) & (np.abs(determinant) > 1e-8)
        if not np.any(valid):
            break
        step = np.zeros_like(estimate)
        step[valid, 0] = (
            residual[valid, 0] * derivative_y[valid, 1]
            - residual[valid, 1] * derivative_y[valid, 0]
        ) / determinant[valid]
        step[valid, 1] = (
            derivative_x[valid, 0] * residual[valid, 1]
            - derivative_x[valid, 1] * residual[valid, 0]
        ) / determinant[valid]
        step_norm = np.linalg.norm(step, axis=1)
        large = step_norm > 20.0
        step[large] *= (20.0 / step_norm[large])[:, None]
        estimate[valid] -= step[valid]

    return estimate[0] if single else estimate


def warp_mri_for_stage(
    source_mri: np.ndarray,
    inverse_mapping: Callable[[np.ndarray], np.ndarray],
    *,
    boundary_trim_px: int = MRI_WARP_BOUNDARY_TRIM_PX,
) -> tuple[np.ndarray, float]:
    warped, valid_mask = warp_image_to_target_space(
        source_mri,
        TARGET_SHAPE,
        inverse_mapping,
    )
    valid = np.asarray(valid_mask > 0, dtype=np.uint8)
    if boundary_trim_px > 0:
        kernel_size = 2 * boundary_trim_px + 1
        kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
        valid = cv2.erode(
            valid,
            kernel,
            iterations=1,
            borderType=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        warped = np.where(valid > 0, warped, 0).astype(np.uint8)
    return warped, float(np.mean(valid > 0))


def jacobian_stats(mapping: Callable[[np.ndarray], np.ndarray]) -> dict[str, float | int]:
    coordinates = np.linspace(0.0, TARGET_SHAPE[1] - 1.0, 25)
    xx, yy = np.meshgrid(coordinates, coordinates)
    points = np.column_stack([xx.ravel(), yy.ravel()])
    epsilon = 1e-3
    dx = np.array([epsilon, 0.0])
    dy = np.array([0.0, epsilon])
    derivative_x = (mapping(points + dx) - mapping(points - dx)) / (2.0 * epsilon)
    derivative_y = (mapping(points + dy) - mapping(points - dy)) / (2.0 * epsilon)
    determinants = derivative_x[:, 0] * derivative_y[:, 1] - derivative_y[:, 0] * derivative_x[:, 1]
    return {
        "jacobian_min": float(np.min(determinants)),
        "jacobian_median": float(np.median(determinants)),
        "jacobian_max": float(np.max(determinants)),
        "jacobian_nonpositive_fraction": float(np.mean(determinants <= 0.0)),
        "jacobian_sample_count": int(len(determinants)),
    }


def lower_incisor_lateral_controls(
    contour: np.ndarray,
    grid: Any,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the two signed-width extrema around the M1->L6 axis."""
    points = np.asarray(contour, dtype=float)
    m1 = np.asarray(grid.M1, dtype=float)
    l6 = np.asarray(grid.L6, dtype=float)
    axis = l6 - m1
    length = float(np.linalg.norm(axis))
    if points.ndim != 2 or points.shape[1] != 2 or length <= 1e-8:
        raise ValueError("Cannot derive lower-incisor lateral controls")
    normal = np.array([-axis[1], axis[0]], dtype=float) / length
    signed_distance = (points - m1) @ normal
    return (
        points[int(np.argmin(signed_distance))].copy(),
        points[int(np.argmax(signed_distance))].copy(),
    )


def build_stage_geometry(
    contours_by_speaker: dict[str, dict[str, np.ndarray]],
    loaded_speakers: dict[str, Any],
    *,
    reference_speaker: str = REFERENCE_SPEAKER,
    speakers: tuple[str, ...] = SPEAKERS,
    lower_shape_controls: bool = False,
) -> tuple[
    dict[str, dict[str, dict[str, np.ndarray]]],
    dict[str, dict[str, dict[str, np.ndarray | None]]],
    dict[str, dict[str, object]],
    dict[str, dict[str, Callable[[np.ndarray], np.ndarray]]],
]:
    stage_contours = {stage: {} for stage in STAGES}
    stage_landmarks = {stage: {} for stage in STAGES}
    inverse_mappings = {stage: {} for stage in METRIC_STAGES}
    transform_records: dict[str, dict[str, object]] = {}
    if reference_speaker not in speakers:
        raise ValueError(
            f"Reference speaker must be one of {speakers}, got {reference_speaker}"
        )
    target_grid = loaded_speakers[reference_speaker].grid
    target_landmarks = extract_true_landmarks(target_grid)

    for speaker in speakers:
        raw_contours = contours_by_speaker[speaker]
        source_landmarks = extract_true_landmarks(loaded_speakers[speaker].grid)
        stage_contours["ready"][speaker] = identity_contours(raw_contours)
        stage_landmarks["ready"][speaker] = identity_landmarks(source_landmarks)

        if speaker == reference_speaker:
            stage_contours["affine"][speaker] = identity_contours(raw_contours)
            stage_contours["affine_tps"][speaker] = identity_contours(raw_contours)
            stage_landmarks["affine"][speaker] = identity_landmarks(source_landmarks)
            stage_landmarks["affine_tps"][speaker] = identity_landmarks(source_landmarks)
            inverse_mappings["affine"][speaker] = identity_mapping
            inverse_mappings["affine_tps"][speaker] = identity_mapping
            transform_records[speaker] = {
                "policy": "identity",
                "affine_matrix": np.eye(2).tolist(),
                "affine_shift": np.zeros(2).tolist(),
                "affine_controls": list(AFFINE_CONTROL_ORDER),
                "tps_controls": list(
                    TPS_CONTROL_ORDER
                    + (LOWER_SHAPE_CONTROL_ORDER if lower_shape_controls else ())
                ),
                "tps_smoothing": 0.0,
                "tps_control_residual_max_px": 0.0,
                "tps_m1_residual_px": 0.0,
                "tps_l6_residual_px": 0.0,
                "jacobian_min": 1.0,
                "jacobian_median": 1.0,
                "jacobian_max": 1.0,
                "jacobian_nonpositive_fraction": 0.0,
                "jacobian_sample_count": 625,
                "mri_inverse_policy": "identity",
                "mri_inverse_roundtrip_rms_px": 0.0,
            }
            continue

        fit = build_two_step_transform(loaded_speakers[speaker].grid, target_grid)
        if tuple(fit["step1_labels"]) != AFFINE_CONTROL_ORDER:
            raise ValueError(
                f"{speaker}: affine controls changed: {tuple(fit['step1_labels'])}"
            )
        if tuple(fit["step2_labels"]) != TPS_CONTROL_ORDER:
            raise ValueError(f"{speaker}: TPS controls changed: {tuple(fit['step2_labels'])}")
        affine = fit["step1_affine"]
        affine_mapping = lambda points, affine=affine: apply_transform(affine, points)
        tps_mapping = fit["apply_two_step"]
        step2_labels = tuple(fit["step2_labels"])
        lower_source_affine: tuple[np.ndarray, np.ndarray] | None = None
        lower_target: tuple[np.ndarray, np.ndarray] | None = None
        if lower_shape_controls:
            source_landmarks_affine = map_landmarks(
                affine_mapping,
                source_landmarks,
            )
            step2_source, step2_target, base_labels = build_step2_controls(
                source_landmarks_affine,
                target_landmarks,
            )
            lower_source = lower_incisor_lateral_controls(
                loaded_speakers[speaker].contours["mandible-incisior"],
                loaded_speakers[speaker].grid,
            )
            lower_target = lower_incisor_lateral_controls(
                loaded_speakers[reference_speaker].contours["mandible-incisior"],
                target_grid,
            )
            lower_source_affine = tuple(
                np.asarray(affine_mapping(point), dtype=float) for point in lower_source
            )
            step2_source = np.vstack([step2_source, *lower_source_affine])
            step2_target = np.vstack([step2_target, *lower_target])
            step2_tps = fit_tps(step2_source, step2_target, smoothing=0.0)
            tps_mapping = (
                lambda points,
                affine_mapping=affine_mapping,
                step2_tps=step2_tps: apply_tps(
                    step2_tps,
                    affine_mapping(points),
                )
            )
            step2_labels = tuple(base_labels) + LOWER_SHAPE_CONTROL_ORDER
        stage_contours["affine"][speaker] = {
            label: affine_mapping(points) for label, points in raw_contours.items()
        }
        stage_contours["affine_tps"][speaker] = {
            label: tps_mapping(points) for label, points in raw_contours.items()
        }
        affine_landmarks = map_landmarks(affine_mapping, source_landmarks)
        tps_landmarks = map_landmarks(tps_mapping, source_landmarks)
        stage_landmarks["affine"][speaker] = affine_landmarks
        stage_landmarks["affine_tps"][speaker] = tps_landmarks
        inverse_affine = invert_affine(affine)
        inverse_mappings["affine"][speaker] = (
            lambda points, inverse_affine=inverse_affine: apply_transform(
                inverse_affine, points
            )
        )
        inverse_tps_source, inverse_tps_target, inverse_tps_labels = build_step2_controls(
            affine_landmarks, target_landmarks
        )
        if lower_shape_controls:
            if lower_source_affine is None or lower_target is None:
                raise RuntimeError("Missing lower-incisor shape controls")
            inverse_tps_source = np.vstack(
                [inverse_tps_source, *lower_source_affine]
            )
            inverse_tps_target = np.vstack([inverse_tps_target, *lower_target])
            inverse_tps_labels.extend(LOWER_SHAPE_CONTROL_ORDER)
        if tuple(inverse_tps_labels) != step2_labels:
            raise ValueError(
                f"{speaker}: inverse TPS controls changed: {tuple(inverse_tps_labels)}"
            )
        inverse_tps = fit_tps(
            inverse_tps_target,
            inverse_tps_source,
            smoothing=0.0,
        )
        initial_inverse_full_mapping = (
            lambda points, inverse_tps=inverse_tps, inverse_affine=inverse_affine: apply_transform(
                inverse_affine,
                apply_tps(inverse_tps, points),
            )
        )
        inverse_full_mapping = (
            lambda points,
            tps_mapping=tps_mapping,
            initial_inverse_full_mapping=initial_inverse_full_mapping: refine_inverse_mapping(
                points,
                forward_mapping=tps_mapping,
                initial_inverse_mapping=initial_inverse_full_mapping,
            )
        )
        inverse_mappings["affine_tps"][speaker] = inverse_full_mapping
        roundtrip_probe = np.vstack([raw_contours[label] for label in CONTOUR_ORDER])
        roundtrip_residual = inverse_full_mapping(tps_mapping(roundtrip_probe)) - roundtrip_probe
        tps_displacements = np.vstack(
            [
                stage_contours["affine_tps"][speaker][label]
                - stage_contours["affine"][speaker][label]
                for label in CONTOUR_ORDER
            ]
        )
        tps_control_residuals = {
            label: float(
                np.linalg.norm(
                    np.asarray(tps_landmarks[label], dtype=float)
                    - np.asarray(target_landmarks[label], dtype=float)
                )
            )
            for label in TPS_CONTROL_ORDER
        }
        transform_records[speaker] = {
            "policy": f"fixed_curated_vtln_reference_to_{reference_speaker.lower()}",
            "source_reference": loaded_speakers[speaker].spec.basename,
            "target_reference": loaded_speakers[reference_speaker].spec.basename,
            "affine_matrix": np.asarray(affine["A"], dtype=float).tolist(),
            "affine_shift": np.asarray(affine["t"], dtype=float).tolist(),
            "affine_controls": list(fit["step1_labels"]),
            "tps_controls": list(step2_labels),
            "tps_smoothing": 0.0,
            "tps_control_residual_max_px": max(tps_control_residuals.values()),
            "tps_m1_residual_px": tps_control_residuals["M1"],
            "tps_l6_residual_px": tps_control_residuals["L6"],
            "tps_mean_increment_mm": float(
                np.mean(np.linalg.norm(tps_displacements, axis=1)) * TARGET_MM_PER_PIXEL
            ),
            "tps_max_increment_mm": float(
                np.max(np.linalg.norm(tps_displacements, axis=1)) * TARGET_MM_PER_PIXEL
            ),
            "mri_inverse_policy": (
                "numerical inverse of the exact forward affine+TPS map, initialized by a "
                "reverse zero-smoothing TPS on the same 16 controls"
            ),
            "mri_inverse_roundtrip_rms_px": float(
                np.sqrt(np.mean(np.sum(roundtrip_residual**2, axis=1)))
            ),
            **jacobian_stats(tps_mapping),
        }
    return stage_contours, stage_landmarks, transform_records, inverse_mappings


def compute_p2cp_rows(
    stage_contours: dict[str, dict[str, dict[str, np.ndarray]]],
    p2cp_mean: Callable[..., float],
    p2cp_rms: Callable[..., float],
    *,
    reference_speaker: str = REFERENCE_SPEAKER,
) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    target = stage_contours["ready"][reference_speaker]
    contour_rows: list[dict[str, object]] = []
    speaker_rows: list[dict[str, object]] = []
    overall_rows: list[dict[str, object]] = []

    for stage in METRIC_STAGES:
        stage_speaker_rows: list[dict[str, object]] = []
        for speaker in SPEAKERS:
            per_speaker: list[dict[str, object]] = []
            for label in CONTOUR_ORDER:
                source_points = stage_contours[stage][speaker][label]
                target_points = target[label]
                row = {
                    "stage": stage,
                    "speaker": speaker,
                    "reference_speaker": reference_speaker,
                    "contour": label,
                    "point_count": len(source_points),
                    "p2cp_mean_mm": float(p2cp_mean(source_points, target_points) * TARGET_MM_PER_PIXEL),
                    "p2cp_rms_mm": float(p2cp_rms(source_points, target_points) * TARGET_MM_PER_PIXEL),
                }
                contour_rows.append(row)
                per_speaker.append(row)
            speaker_row = {
                "stage": stage,
                "speaker": speaker,
                "reference_speaker": reference_speaker,
                "n_contours": len(per_speaker),
                "p2cp_mean_contour_macro_mm": float(
                    np.mean([row["p2cp_mean_mm"] for row in per_speaker])
                ),
                "p2cp_rms_contour_macro_mm": float(
                    np.mean([row["p2cp_rms_mm"] for row in per_speaker])
                ),
            }
            stage_speaker_rows.append(speaker_row)
            speaker_rows.append(speaker_row)

        evaluated = [row for row in stage_speaker_rows if row["speaker"] != reference_speaker]
        overall_rows.append(
            {
                "stage": stage,
                "reference_speaker": reference_speaker,
                "n_source_speakers": len(evaluated),
                "n_contours": len(CONTOUR_ORDER),
                "p2cp_mean_equal_speaker_macro_mm": float(
                    np.mean([row["p2cp_mean_contour_macro_mm"] for row in evaluated])
                ),
                "p2cp_rms_equal_speaker_macro_mm": float(
                    np.mean([row["p2cp_rms_contour_macro_mm"] for row in evaluated])
                ),
            }
        )

    affine_by_speaker = {
        str(row["speaker"]): row for row in speaker_rows if row["stage"] == "affine"
    }
    for row in speaker_rows:
        affine = affine_by_speaker[str(row["speaker"])]
        row["delta_rms_vs_affine_mm"] = float(
            row["p2cp_rms_contour_macro_mm"] - affine["p2cp_rms_contour_macro_mm"]
        )
    affine_overall = next(row for row in overall_rows if row["stage"] == "affine")
    for row in overall_rows:
        row["delta_rms_vs_affine_mm"] = float(
            row["p2cp_rms_equal_speaker_macro_mm"]
            - affine_overall["p2cp_rms_equal_speaker_macro_mm"]
        )
    return contour_rows, speaker_rows, overall_rows


def landmark_color(name: str) -> str:
    if name in {"I1", "I2", "I3", "I4", "I5"}:
        return LANDMARK_COLORS["I1_I5"]
    if name in {"I6", "I7"}:
        return LANDMARK_COLORS["I6_I7"]
    if name in set(CERVICAL_ORDER):
        return LANDMARK_COLORS["C1_C6"]
    return LANDMARK_COLORS["P1_M1_L6"]


def square_view_box(
    point_sets: list[np.ndarray],
    *,
    padding: float = 16.0,
    minimum_side: float = 260.0,
) -> tuple[float, float, float, float]:
    points = np.vstack([np.asarray(item, dtype=float).reshape(-1, 2) for item in point_sets])
    lower = np.min(points, axis=0)
    upper = np.max(points, axis=0)
    center = 0.5 * (lower + upper)
    side = max(float(np.max(upper - lower) + 2.0 * padding), minimum_side)
    side = min(side, float(min(TARGET_SHAPE)))
    x0 = float(np.clip(center[0] - 0.5 * side, 0.0, TARGET_SHAPE[1] - side))
    y0 = float(np.clip(center[1] - 0.5 * side, 0.0, TARGET_SHAPE[0] - side))
    return x0, x0 + side, y0, y0 + side


def setup_axis(
    axis: plt.Axes,
    background: np.ndarray | None,
    *,
    view_box: tuple[float, float, float, float] | None = None,
) -> None:
    x0, x1, y0, y1 = view_box or (0.0, TARGET_SHAPE[1], 0.0, TARGET_SHAPE[0])
    axis.set_xlim(x0, x1)
    axis.set_ylim(y1, y0)
    axis.set_aspect("equal")
    axis.set_xticks([])
    axis.set_yticks([])
    axis.set_facecolor("#10151f")
    if background is not None:
        axis.imshow(background, cmap="gray", extent=(0, 480, 480, 0), vmin=0, vmax=255)
    for spine in axis.spines.values():
        spine.set_color("#7d8590")
        spine.set_linewidth(0.8)


def plot_contours(
    axis: plt.Axes,
    contours: dict[str, np.ndarray],
    *,
    linewidth: float = 1.8,
    alpha: float = 1.0,
) -> None:
    for label in CONTOUR_ORDER:
        points = contours[label]
        axis.plot(
            points[:, 0],
            points[:, 1],
            color=CONTOUR_COLORS[label],
            linewidth=linewidth,
            alpha=alpha,
            solid_capstyle="round",
        )


def draw_cervical(axis: plt.Axes, landmarks: dict[str, np.ndarray | None]) -> None:
    points = np.vstack([landmarks[name] for name in CERVICAL_ORDER])
    axis.plot(points[:, 0], points[:, 1], color=LANDMARK_COLORS["C1_C6"], linewidth=1.0, alpha=0.8)
    axis.scatter(
        points[:, 0],
        points[:, 1],
        s=18,
        marker="o",
        facecolor=LANDMARK_COLORS["C1_C6"],
        edgecolor="#071018",
        linewidth=0.5,
        zorder=6,
    )
    for name, point in zip(CERVICAL_ORDER, points):
        axis.text(point[0] + 5, point[1], name, color=LANDMARK_COLORS["C1_C6"], fontsize=6)


def figure_axes(title: str) -> tuple[plt.Figure, np.ndarray]:
    figure, axes = plt.subplots(3, 3, figsize=(14.5, 14.8))
    figure.patch.set_facecolor("white")
    figure.suptitle(title, fontsize=16, fontweight="bold", y=0.985)
    return figure, axes


def contour_legend(
    *,
    include_target: bool,
    reference_speaker: str = REFERENCE_SPEAKER,
) -> list[Line2D]:
    handles = [
        Line2D([0], [0], color=CONTOUR_COLORS[label], lw=2.0, label=label)
        for label in CONTOUR_ORDER
    ]
    handles.append(
        Line2D(
            [0],
            [0],
            marker="o",
            color=LANDMARK_COLORS["C1_C6"],
            lw=1.0,
            label="C1-C6 auxiliary",
        )
    )
    if include_target:
        handles.append(
            Line2D(
                [0],
                [0],
                color="#d0d7de",
                lw=1.0,
                label=f"{reference_speaker} pourri #2 target",
            )
        )
    return handles


def save_figure(figure: plt.Figure, path: Path, handles: list[Line2D]) -> None:
    figure.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.012),
        ncol=4,
        frameon=False,
        fontsize=7.5,
    )
    figure.tight_layout(rect=(0.015, 0.075, 0.985, 0.955))
    figure.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def render_ready_sheet(
    path: Path,
    *,
    with_mri: bool,
    rows_by_speaker: dict[str, dict[str, str]],
    stage_contours: dict[str, dict[str, dict[str, np.ndarray]]],
    stage_landmarks: dict[str, dict[str, dict[str, np.ndarray | None]]],
    mri_by_speaker: dict[str, np.ndarray],
) -> None:
    suffix = "MRI overlay" if with_mri else "contour-only"
    figure, axes = figure_axes(
        f"S5 pourri #2 /u/ - 11 contours + fixed C1-C6 ({suffix})\n"
        "P2 excluded | native S5 midpoint frames | no geometric transform"
    )
    for axis, speaker in zip(axes.flat, SPEAKERS):
        setup_axis(axis, mri_by_speaker[speaker] if with_mri else None)
        plot_contours(axis, stage_contours["ready"][speaker])
        draw_cervical(axis, stage_landmarks["ready"][speaker])
        row = rows_by_speaker[speaker]
        axis.set_title(
            f"{speaker} | F{int(row['selected_frame']):04d} | {float(row['duration_ms']):.0f} ms",
            fontsize=10,
            fontweight="bold",
        )
    save_figure(figure, path, contour_legend(include_target=False))


def render_landmark_sheet(
    path: Path,
    *,
    with_mri: bool,
    stage_landmarks: dict[str, dict[str, dict[str, np.ndarray | None]]],
    affine_mri_by_speaker: dict[str, np.ndarray],
    reference_speaker: str = REFERENCE_SPEAKER,
) -> None:
    suffix = "affine-warped source MRI" if with_mri else "landmark-only"
    figure, axes = figure_axes(
        f"P# -> {reference_speaker} transform landmarks after affine ({suffix})\n"
        f"x: affine-mapped source | o: {reference_speaker} target | TPS adds M1 and L6"
    )
    target = stage_landmarks["ready"][reference_speaker]
    for axis, speaker in zip(axes.flat, SPEAKERS):
        source = stage_landmarks["affine"][speaker]
        visible_controls = [
            point
            for name in TPS_CONTROL_ORDER
            for point in (source[name], target[name])
            if point is not None
        ]
        view_box = square_view_box(visible_controls, padding=22.0)
        setup_axis(
            axis,
            affine_mri_by_speaker[speaker] if with_mri else None,
            view_box=view_box,
        )
        for name in TPS_CONTROL_ORDER:
            source_point = source[name]
            target_point = target[name]
            if source_point is None or target_point is None:
                continue
            color = landmark_color(name)
            axis.annotate(
                "",
                xy=target_point,
                xytext=source_point,
                arrowprops={"arrowstyle": "-", "color": color, "alpha": 0.42, "lw": 0.75},
            )
            axis.scatter(
                target_point[0],
                target_point[1],
                s=28,
                marker="o",
                facecolor="none",
                edgecolor=color,
                linewidth=1.1,
                zorder=5,
            )
            axis.scatter(
                source_point[0],
                source_point[1],
                s=24,
                marker="x",
                color=color,
                linewidth=1.0,
                zorder=6,
            )
            axis.text(source_point[0] + 4, source_point[1] - 3, name, color=color, fontsize=6)
        axis.set_title(
            f"{speaker} -> {reference_speaker} | affine residuals | TPS 16",
            fontsize=10,
            fontweight="bold",
        )
    handles = [
        Line2D([0], [0], marker="x", color="#7d8590", lw=0, label="source landmark"),
        Line2D(
            [0],
            [0],
            marker="o",
            markerfacecolor="none",
            color="#7d8590",
            lw=0,
            label=f"{reference_speaker} target",
        ),
        Line2D([0], [0], color=LANDMARK_COLORS["I1_I5"], lw=2, label="I1-I5"),
        Line2D([0], [0], color=LANDMARK_COLORS["I6_I7"], lw=2, label="I6-I7"),
        Line2D([0], [0], color=LANDMARK_COLORS["C1_C6"], lw=2, label="C1-C6"),
        Line2D([0], [0], color=LANDMARK_COLORS["P1_M1_L6"], lw=2, label="P1/M1/L6"),
    ]
    save_figure(figure, path, handles)


def render_tps_m1_l6_match_sheet(
    path: Path,
    *,
    stage_contours: dict[str, dict[str, dict[str, np.ndarray]]],
    stage_landmarks: dict[str, dict[str, dict[str, np.ndarray | None]]],
    reference_speaker: str = REFERENCE_SPEAKER,
) -> None:
    """Show that the S5-derived M1/L6 controls interpolate exactly after TPS."""
    figure, axes = figure_axes(
        f"S5 pourri #2 lower-incisor M1/L6 after TPS -> {reference_speaker}\n"
        "colored: mapped source | gray: dynamic target | markers: mapped x over target o"
    )
    target_contour = stage_contours["ready"][reference_speaker]["lower-incisor"]
    target_landmarks = stage_landmarks["ready"][reference_speaker]
    for axis, speaker in zip(axes.flat, SPEAKERS):
        mapped_contour = stage_contours["affine_tps"][speaker]["lower-incisor"]
        mapped_landmarks = stage_landmarks["affine_tps"][speaker]
        visible = [mapped_contour, target_contour]
        visible.extend(
            point.reshape(1, 2)
            for name in ("M1", "L6")
            for point in (mapped_landmarks[name], target_landmarks[name])
            if point is not None
        )
        setup_axis(axis, None, view_box=square_view_box(visible, padding=18.0))
        axis.plot(
            target_contour[:, 0],
            target_contour[:, 1],
            color="#d0d7de",
            linewidth=3.0,
            alpha=0.95,
            zorder=2,
        )
        axis.plot(
            mapped_contour[:, 0],
            mapped_contour[:, 1],
            color=CONTOUR_COLORS["lower-incisor"],
            linewidth=2.0,
            alpha=0.95,
            zorder=3,
        )
        residuals = []
        for name, marker, color in (
            ("M1", "o", "#ff4d6d"),
            ("L6", "D", "#39d353"),
        ):
            mapped = np.asarray(mapped_landmarks[name], dtype=float)
            target = np.asarray(target_landmarks[name], dtype=float)
            residuals.append(float(np.linalg.norm(mapped - target)))
            axis.scatter(
                target[0],
                target[1],
                s=90,
                marker=marker,
                facecolor="none",
                edgecolor="white",
                linewidth=2.2,
                zorder=5,
            )
            axis.scatter(
                mapped[0],
                mapped[1],
                s=62,
                marker="x",
                color=color,
                linewidth=2.0,
                zorder=6,
            )
            axis.text(mapped[0] + 2, mapped[1] - 2, name, color=color, fontsize=8)
        axis.set_title(
            f"{speaker} -> {reference_speaker} | M1 {residuals[0]:.2e}px | "
            f"L6 {residuals[1]:.2e}px",
            fontsize=9,
            fontweight="bold",
        )
    handles = [
        Line2D(
            [0],
            [0],
            color=CONTOUR_COLORS["lower-incisor"],
            lw=2,
            label="mapped lower-incisor",
        ),
        Line2D([0], [0], color="#d0d7de", lw=3, label=f"{reference_speaker} dynamic target"),
        Line2D([0], [0], marker="x", color="#ff4d6d", lw=0, label="mapped M1"),
        Line2D([0], [0], marker="x", color="#39d353", lw=0, label="mapped L6"),
        Line2D(
            [0],
            [0],
            marker="o",
            markerfacecolor="none",
            markeredgecolor="black",
            lw=0,
            label="target control (under x)",
        ),
    ]
    save_figure(figure, path, handles)


def render_mapped_sheet(
    path: Path,
    *,
    stage: str,
    with_mri: bool,
    stage_contours: dict[str, dict[str, dict[str, np.ndarray]]],
    stage_landmarks: dict[str, dict[str, dict[str, np.ndarray | None]]],
    speaker_metrics: dict[tuple[str, str], dict[str, object]],
    stage_mri_by_speaker: dict[str, np.ndarray],
    reference_speaker: str = REFERENCE_SPEAKER,
) -> None:
    stage_title = "After affine" if stage == "affine" else "After affine + TPS"
    suffix = "same-stage warped source MRI" if with_mri else "contour-only"
    figure, axes = figure_axes(
        f"S5 pourri #2 /u/ mapped to {reference_speaker} - {stage_title} ({suffix})\n"
        f"colored: mapped source | gray: {reference_speaker} target | square anatomy zoom"
    )
    target = stage_contours["ready"][reference_speaker]
    for axis, speaker in zip(axes.flat, SPEAKERS):
        mapped = stage_contours[stage][speaker]
        cervical = [
            stage_landmarks[stage][speaker][name]
            for name in CERVICAL_ORDER
            if stage_landmarks[stage][speaker][name] is not None
        ]
        view_box = square_view_box(
            [*mapped.values(), *target.values(), *cervical],
        )
        setup_axis(
            axis,
            stage_mri_by_speaker[speaker] if with_mri else None,
            view_box=view_box,
        )
        for label in CONTOUR_ORDER:
            points = target[label]
            axis.plot(points[:, 0], points[:, 1], color="#d0d7de", linewidth=1.0, alpha=0.72)
        plot_contours(axis, mapped, linewidth=1.75)
        draw_cervical(axis, stage_landmarks[stage][speaker])
        metric = speaker_metrics[(speaker, stage)]
        title = (
            f"{speaker} | mean {metric['p2cp_mean_contour_macro_mm']:.2f} mm | "
            f"RMS {metric['p2cp_rms_contour_macro_mm']:.2f} mm"
        )
        if stage == "affine_tps":
            title += f" | delta {metric['delta_rms_vs_affine_mm']:+.2f}"
        axis.set_title(title, fontsize=9.5, fontweight="bold")
    save_figure(
        figure,
        path,
        contour_legend(include_target=True, reference_speaker=reference_speaker),
    )


def render_all_stage_contour_overlay(
    path: Path,
    *,
    stage: str,
    stage_contours: dict[str, dict[str, dict[str, np.ndarray]]],
    loaded_speakers: dict[str, Any],
    reference_speaker: str = REFERENCE_SPEAKER,
    speakers: tuple[str, ...] = SPEAKERS,
) -> None:
    """Render one legacy-V2-style population overlay for the dynamic contours."""
    legacy_stage_name = {
        "ready": "init",
        "affine": "affine",
        "affine_tps": "tps",
    }[stage]
    stage_payload = {
        "reference_id": reference_speaker,
        "median_speaker": reference_speaker,
        "mapped_contours": stage_contours[stage],
    }
    save_stage_contour_overlay_figure(
        loaded_speakers,
        list(speakers),
        list(speakers),
        list(CONTOUR_ORDER),
        stage_payload,
        choose_speaker_colors(loaded_speakers),
        stage_name=legacy_stage_name,
        cohort_name="all",
        target_speaker_id=reference_speaker,
        output_path=path,
        open_visual_labels=CONTOUR_ORDER,
    )


def image_validation(path: Path) -> dict[str, object]:
    image = cv2.imread(str(path))
    return {
        "path": str(path.resolve()),
        "readable": image is not None,
        "width": 0 if image is None else int(image.shape[1]),
        "height": 0 if image is None else int(image.shape[0]),
        "bytes": path.stat().st_size if path.is_file() else 0,
        "sha256": sha256_file(path) if path.is_file() else None,
    }


def run(args: argparse.Namespace) -> dict[str, object]:
    reference_speaker = str(args.reference_speaker)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    selection_rows = load_pourri2_selection(args.selection_csv)
    rows_by_speaker = {row["speaker"]: row for row in selection_rows}
    if args.review_root is None:
        contours_by_speaker = {
            speaker: load_frame_contours(
                args.inference_root, speaker, int(rows_by_speaker[speaker]["selected_frame"])
            )
            for speaker in SPEAKERS
        }
        contour_source = "mutable inference snapshot"
    else:
        contours_by_speaker = {
            speaker: load_review_contours(
                args.review_root, speaker, int(rows_by_speaker[speaker]["selected_frame"])
            )
            for speaker in SPEAKERS
        }
        contour_source = "frozen review_s5_pourri2 per-case snapshots"
    mri_by_speaker = {
        speaker: load_mri_frame(
            args.old_root, speaker, int(rows_by_speaker[speaker]["selected_frame"])
        )
        for speaker in SPEAKERS
    }
    loaded_speakers = load_curated_speakers(args.vtln_dir, list(SPEAKERS))
    stage_contours, stage_landmarks, transform_records, inverse_mappings = (
        build_stage_geometry(
            contours_by_speaker,
            loaded_speakers,
            reference_speaker=reference_speaker,
        )
    )
    stage_mri_by_stage: dict[str, dict[str, np.ndarray]] = {
        stage: {} for stage in METRIC_STAGES
    }
    for stage in METRIC_STAGES:
        for speaker in SPEAKERS:
            warped_mri, valid_fraction = warp_mri_for_stage(
                mri_by_speaker[speaker],
                inverse_mappings[stage][speaker],
            )
            stage_mri_by_stage[stage][speaker] = warped_mri
            transform_records[speaker][f"{stage}_mri_valid_fraction"] = valid_fraction
    canonical_mean, canonical_rms = import_p2cp(args.vt_tools_root)
    contour_rows, speaker_rows, overall_rows = compute_p2cp_rows(
        stage_contours,
        canonical_mean,
        canonical_rms,
        reference_speaker=reference_speaker,
    )
    speaker_metrics = {
        (str(row["speaker"]), str(row["stage"])): row for row in speaker_rows
    }
    outputs = {
        "ready_clean": output_dir / "01_contours_ready_clean.png",
        "ready_mri": output_dir / "01_contours_ready_mri.png",
        "landmarks_clean": output_dir / "02_affine_tps_landmarks_clean.png",
        "landmarks_mri": output_dir / "02_affine_tps_landmarks_mri.png",
        "affine_clean": output_dir / "03_after_affine_clean.png",
        "affine_mri": output_dir / "03_after_affine_mri.png",
        "affine_tps_clean": output_dir / "04_after_affine_tps_clean.png",
        "affine_tps_mri": output_dir / "04_after_affine_tps_mri.png",
        "all_init_contours_overlay": output_dir / "05_all_init_contours_overlay.png",
        "all_affine_contours_overlay": output_dir / "06_all_affine_contours_overlay.png",
        "all_tps_contours_overlay": output_dir / "07_all_tps_contours_overlay.png",
    }
    render_ready_sheet(
        outputs["ready_clean"],
        with_mri=False,
        rows_by_speaker=rows_by_speaker,
        stage_contours=stage_contours,
        stage_landmarks=stage_landmarks,
        mri_by_speaker=mri_by_speaker,
    )
    render_ready_sheet(
        outputs["ready_mri"],
        with_mri=True,
        rows_by_speaker=rows_by_speaker,
        stage_contours=stage_contours,
        stage_landmarks=stage_landmarks,
        mri_by_speaker=mri_by_speaker,
    )
    render_landmark_sheet(
        outputs["landmarks_clean"],
        with_mri=False,
        stage_landmarks=stage_landmarks,
        affine_mri_by_speaker=stage_mri_by_stage["affine"],
        reference_speaker=reference_speaker,
    )
    render_landmark_sheet(
        outputs["landmarks_mri"],
        with_mri=True,
        stage_landmarks=stage_landmarks,
        affine_mri_by_speaker=stage_mri_by_stage["affine"],
        reference_speaker=reference_speaker,
    )
    for stage, prefix in (("affine", "affine"), ("affine_tps", "affine_tps")):
        render_mapped_sheet(
            outputs[f"{prefix}_clean"],
            stage=stage,
            with_mri=False,
            stage_contours=stage_contours,
            stage_landmarks=stage_landmarks,
            speaker_metrics=speaker_metrics,
            stage_mri_by_speaker=stage_mri_by_stage[stage],
            reference_speaker=reference_speaker,
        )
        render_mapped_sheet(
            outputs[f"{prefix}_mri"],
            stage=stage,
            with_mri=True,
            stage_contours=stage_contours,
            stage_landmarks=stage_landmarks,
            speaker_metrics=speaker_metrics,
            stage_mri_by_speaker=stage_mri_by_stage[stage],
            reference_speaker=reference_speaker,
        )
    for stage, output_key in (
        ("ready", "all_init_contours_overlay"),
        ("affine", "all_affine_contours_overlay"),
        ("affine_tps", "all_tps_contours_overlay"),
    ):
        render_all_stage_contour_overlay(
            outputs[output_key],
            stage=stage,
            stage_contours=stage_contours,
            loaded_speakers=loaded_speakers,
            reference_speaker=reference_speaker,
        )

    selected_frame_rows = [
        {
            "speaker": speaker,
            "selected_frame": int(rows_by_speaker[speaker]["selected_frame"]),
            "duration_ms": float(rows_by_speaker[speaker]["duration_ms"]),
            "alignment_source": rows_by_speaker[speaker]["alignment_source"],
            "contour_count": len(contours_by_speaker[speaker]),
            "vtln_reference": loaded_speakers[speaker].spec.basename,
        }
        for speaker in SPEAKERS
    ]
    csv_paths = {
        "selected_frames": output_dir / "selected_frames.csv",
        "p2cp_per_contour": output_dir / "p2cp_per_contour.csv",
        "p2cp_per_speaker": output_dir / "p2cp_per_speaker.csv",
        "p2cp_overall": output_dir / "p2cp_overall.csv",
    }
    write_csv(csv_paths["selected_frames"], selected_frame_rows)
    write_csv(csv_paths["p2cp_per_contour"], contour_rows)
    write_csv(csv_paths["p2cp_per_speaker"], speaker_rows)
    write_csv(csv_paths["p2cp_overall"], overall_rows)

    nonpositive = [
        speaker
        for speaker, record in transform_records.items()
        if float(record["jacobian_nonpositive_fraction"]) > 0.0
    ]
    validations = {name: image_validation(path) for name, path in outputs.items()}
    if not all(item["readable"] and item["width"] > 0 and item["height"] > 0 for item in validations.values()):
        raise RuntimeError("At least one rendered sheet failed image validation")
    summary = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS" if not nonpositive else "PASS_WITH_TPS_FOLDING_WARNING",
        "method": {
            "phone": "S5 pourri repetition 2, sole tier-2 /u/, midpoint frame",
            "included_speakers": list(SPEAKERS),
            "excluded_speakers": list(EXCLUDED_SPEAKERS),
            "exclusion_reason": "P2 selected /u/ frame lacks vocal-folds; no imputation used",
            "reference_speaker": reference_speaker,
            "dynamic_metric_target": (
                f"{reference_speaker}/S5/F{int(rows_by_speaker[reference_speaker]['selected_frame']):04d}"
            ),
            "transform_target_reference": loaded_speakers[reference_speaker].spec.basename,
            "affine_controls": list(AFFINE_CONTROL_ORDER),
            "tps_controls": list(TPS_CONTROL_ORDER),
            "tps_smoothing": 0.0,
            "post_transform_contour_smoothing": "none",
            "mri_overlay_policy": (
                "ready uses each native selected-frame MRI; affine and affine+TPS use "
                f"that speaker's selected-frame MRI warped into {reference_speaker} space with the "
                "corresponding inverse resampling map"
            ),
            "mri_interpolation": "linear inverse resampling; zero outside source support",
            "mri_warp_boundary_trim_px": MRI_WARP_BOUNDARY_TRIM_PX,
            "mri_warp_boundary_trim_policy": (
                "visualization-only erosion of the valid warped-image support to remove "
                "bright source-frame border lines"
            ),
            "mapped_sheet_view": (
                f"visualization-only per-panel square crop around all mapped and {reference_speaker} target "
                "contours; geometry and P2CP remain in the unchanged 480x480 coordinates"
            ),
            "all_stage_overlay_policy": (
                "legacy V2 white-canvas population style; reference thick/opaque, other "
                "speakers thin/translucent; all dynamic contours preserve their stored "
                "open-polyline geometry with no artificial last-to-first segment"
            ),
            "all_stage_overlay_contour_policy": "preserve_all_stored_open_polylines",
            "c1_c6_policy": "fixed per-speaker auxiliary anchors from curated VTLN references",
            "source_to_target_resize": "136x136 to 480x480 only",
            "target_mm_per_pixel": TARGET_MM_PER_PIXEL,
        },
        "metric": {
            "implementation": str((args.vt_tools_root / "vt_tools" / "metrics.py").resolve()),
            "functions": ["vt_tools.metrics.p2cp_mean", "vt_tools.metrics.p2cp_rms"],
            "target": f"{reference_speaker} S5 pourri #2 selected 11 contours",
            "speaker_aggregation": "equal mean across 11 per-contour values",
            "overall_aggregation": (
                "equal mean across eight non-reference speakers; "
                f"{reference_speaker} identity excluded"
            ),
            "overall": overall_rows,
        },
        "inputs": {
            "selection_csv": str(args.selection_csv.resolve()),
            "selection_csv_sha256": sha256_file(args.selection_csv),
            "alignment_summary": str(args.alignment_summary.resolve()),
            "alignment_summary_sha256": (
                sha256_file(args.alignment_summary) if args.alignment_summary.is_file() else None
            ),
            "inference_root": str(args.inference_root.resolve()),
            "review_root": (
                None if args.review_root is None else str(args.review_root.resolve())
            ),
            "contour_source": contour_source,
            "old_root": str(args.old_root.resolve()),
            "vtln_dir": str(args.vtln_dir.resolve()),
        },
        "speaker_inputs": selected_frame_rows,
        "transforms": transform_records,
        "tps_nonpositive_jacobian_speakers": nonpositive,
        "sheets": validations,
        "csv": {
            name: {
                "path": str(path.resolve()),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for name, path in csv_paths.items()
        },
    }
    summary_path = output_dir / "render_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    summary = run(args)
    overall = {row["stage"]: row for row in summary["metric"]["overall"]}
    print(f"[done] {args.output_dir.resolve() / 'render_summary.json'}")
    print(
        "[P2CP RMS] affine={:.3f} mm, affine+TPS={:.3f} mm, delta={:+.3f} mm".format(
            overall["affine"]["p2cp_rms_equal_speaker_macro_mm"],
            overall["affine_tps"]["p2cp_rms_equal_speaker_macro_mm"],
            overall["affine_tps"]["delta_rms_vs_affine_mm"],
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
