from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from grid_transform.annotation_to_grid_workflow import build_transform_bundle
from grid_transform.cv2_config_helpers import resolve_disabled_landmarks_config
from grid_transform.cv2_shared import hex_to_bgr
from grid_transform.transform_helpers import LANDMARK_REPORT_ORDER, rms_point_error
from grid_transform.workspace_paths import sanitize_workspace_token
from tools.report.slide_p7_nnunet_config import (
    load_slide_p7_nnunet_config,
    stage_label_offset,
    stage_label_position,
    xy_tuple,
)
from grid_transform.config import DEFAULT_OUTPUT_DIR, DEFAULT_VTLN_DIR, VT_SEG_DATA_ROOT
from grid_transform.cv2_annotation_app_config import APP_CONFIG_PATH, load_cv2_app_config_defaults
from grid_transform.image_utils import as_grayscale_uint8
from grid_transform.io import load_frame_npy, load_frame_vtln
from grid_transform.source_annotation import load_source_annotation_json
from tools.report.common import subset_common_contours

from tools.report.export_p7_nnunet_common_label_images import NNUNET_FRAME_DEFAULT, P7_BASENAME


NNUNET_CASE_DIR_DEFAULT = VT_SEG_DATA_ROOT / "2008-003^01-1791" / "test"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export raw/affine/TPS grid comparison images for P7 -> nnUNet."
    )
    parser.add_argument("--vtln-dir", type=Path, default=DEFAULT_VTLN_DIR, help="Folder containing canonical VTLN PNG/ZIP pairs.")
    parser.add_argument("--p7-basename", default=P7_BASENAME, help="Canonical VTLN basename for P7.")
    parser.add_argument(
        "--nnunet-case-dir",
        type=Path,
        default=NNUNET_CASE_DIR_DEFAULT,
        help="nnUNet case folder containing PNG_MR and contours.",
    )
    parser.add_argument("--nnunet-frame", type=int, default=NNUNET_FRAME_DEFAULT, help="nnUNet frame number.")
    parser.add_argument(
        "--target-mode",
        choices=("auto", "workspace", "groundtruth"),
        default="auto",
        help="Which target contour source to use for the nnUNet grid.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=APP_CONFIG_PATH,
        help="Config file containing slide export settings.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR / "slide" / "images",
        help="Output folder for the generated PNG images.",
    )
    parser.add_argument(
        "--workspace-root",
        type=Path,
        default=DEFAULT_OUTPUT_DIR / "annotation_to_grid_transform",
        help="Workspace root used to look for saved target_annotation.latest.json files.",
    )
    return parser.parse_args(argv)


def workspace_dir_for_pair(
    *,
    p7_basename: str,
    nnunet_case: str,
    nnunet_frame: int,
    workspace_root: Path,
) -> Path:
    target_token = f"nnunet_{sanitize_workspace_token(nnunet_case)}_{int(nnunet_frame):06d}"
    workspace_id = sanitize_workspace_token(f"{p7_basename}__{target_token}")
    return workspace_root / workspace_id


def load_target_contours_for_export(
    *,
    p7_basename: str,
    nnunet_case: str,
    nnunet_frame: int,
    workspace_root: Path,
    target_mode: str,
    groundtruth_contours: dict[str, np.ndarray],
) -> tuple[dict[str, np.ndarray], str]:
    workspace_dir = workspace_dir_for_pair(
        p7_basename=p7_basename,
        nnunet_case=nnunet_case,
        nnunet_frame=nnunet_frame,
        workspace_root=workspace_root,
    )
    target_annotation_path = workspace_dir / "target_annotation.latest.json"

    if target_mode == "workspace" and target_annotation_path.is_file():
        payload = load_source_annotation_json(target_annotation_path)
        contours = {
            str(name): np.asarray(points, dtype=float).copy()
            for name, points in payload["contours"].items()
        }
        return contours, f"workspace:{target_annotation_path}"

    if target_mode == "workspace" and not target_annotation_path.is_file():
        raise FileNotFoundError(f"Workspace target annotation not found: {target_annotation_path}")

    contours = {
        str(name): np.asarray(points, dtype=float).copy()
        for name, points in groundtruth_contours.items()
    }
    return contours, "groundtruth:contours_zip"


def draw_polyline(panel: np.ndarray, pts: np.ndarray, color: tuple[int, int, int], *, thickness: int) -> None:
    array = np.round(np.asarray(pts, dtype=float)).astype(np.int32)
    if len(array) < 2:
        return
    cv2.polylines(panel, [array.reshape(-1, 1, 2)], False, color, thickness, cv2.LINE_AA)


def draw_grid_single_color(
    panel: np.ndarray,
    horiz_lines: list[np.ndarray],
    vert_lines: list[np.ndarray],
    color: tuple[int, int, int],
    *,
    outer_thickness: int,
    inner_thickness: int,
) -> None:
    for index, line in enumerate(horiz_lines):
        thickness = outer_thickness if index in (0, len(horiz_lines) - 1) else inner_thickness
        draw_polyline(panel, np.asarray(line, dtype=float), color, thickness=thickness)
    for index, line in enumerate(vert_lines):
        thickness = outer_thickness if index in (0, len(vert_lines) - 1) else inner_thickness
        draw_polyline(panel, np.asarray(line, dtype=float), color, thickness=thickness)


def flatten_grid_nodes(horiz_lines: list[np.ndarray]) -> np.ndarray:
    return np.vstack([np.asarray(line, dtype=float) for line in horiz_lines])


def visible_landmark_names(
    source_points: dict[str, np.ndarray | None],
    target_points: dict[str, np.ndarray | None],
) -> list[str]:
    return [
        name
        for name in LANDMARK_REPORT_ORDER
        if source_points.get(name) is not None and target_points.get(name) is not None
    ]


def landmark_arrays(
    source_points: dict[str, np.ndarray | None],
    target_points: dict[str, np.ndarray | None],
) -> tuple[list[str], np.ndarray, np.ndarray]:
    names = visible_landmark_names(source_points, target_points)
    src = np.vstack([np.asarray(source_points[name], dtype=float) for name in names]) if names else np.zeros((0, 2), dtype=float)
    dst = np.vstack([np.asarray(target_points[name], dtype=float) for name in names]) if names else np.zeros((0, 2), dtype=float)
    return names, src, dst


def draw_translucent_lines(
    panel: np.ndarray,
    pairs: list[tuple[tuple[int, int], tuple[int, int]]],
    color: tuple[int, int, int],
    *,
    alpha: float,
    thickness: int,
) -> None:
    if not pairs:
        return
    overlay = panel.copy()
    for start, end in pairs:
        cv2.line(overlay, start, end, color, thickness, cv2.LINE_AA)
    cv2.addWeighted(overlay, alpha, panel, 1.0 - alpha, 0.0, dst=panel)


def draw_landmarks(
    panel: np.ndarray,
    points: dict[str, np.ndarray | None],
    *,
    names: list[str],
    step1_labels: set[str],
    step2_extra_labels: set[str],
    slide_cfg: dict[str, object],
    stage_name: str,
    target_role: bool,
) -> None:
    palette = slide_cfg["palette"]
    seq_cfg = slide_cfg["transform_sequence"]
    default_offset = xy_tuple(seq_cfg.get("default_label_offset"), default=(7.0, -7.0))
    font_scale = float(seq_cfg.get("label_font_scale", 0.42))
    thickness = int(seq_cfg.get("label_thickness", 1))
    show_labels = bool(seq_cfg.get("show_labels", False))
    source_radius = int(seq_cfg.get("source_marker_radius", 3))
    target_radius = int(seq_cfg.get("target_marker_radius", 3))
    target_color = hex_to_bgr(str(palette.get("target_landmark", "#e5e7eb")))
    affine_color = hex_to_bgr(str(palette.get("affine_anchor", "#fb8500")))
    tps_color = hex_to_bgr(str(palette.get("tps_extra", "#80ed99")))
    other_color = hex_to_bgr(str(palette.get("other_landmark", "#ffd166")))

    for name in names:
        point = points.get(name)
        if point is None:
            continue
        x, y = np.round(np.asarray(point, dtype=float)).astype(int)
        if target_role:
            cv2.circle(panel, (x, y), target_radius + 1, target_color, 1, cv2.LINE_AA)
            cv2.circle(panel, (x, y), max(1, target_radius - 1), (255, 255, 255), 1, cv2.LINE_AA)
            color = target_color
        else:
            if name in step1_labels:
                color = affine_color
            elif name in step2_extra_labels:
                color = tps_color
            else:
                color = other_color
            cv2.circle(panel, (x, y), source_radius + 1, (255, 255, 255), -1, cv2.LINE_AA)
            cv2.circle(panel, (x, y), source_radius, color, -1, cv2.LINE_AA)
        if not show_labels or target_role:
            continue
        absolute_position = stage_label_position(slide_cfg, "transform_sequence", stage_name, name)
        if absolute_position is None:
            offset = stage_label_offset(
                slide_cfg,
                "transform_sequence",
                stage_name,
                name,
                default_offset=default_offset,
            )
            label_pos = (int(round(x + offset[0])), int(round(y + offset[1])))
        else:
            label_pos = (int(round(absolute_position[0])), int(round(absolute_position[1])))
        cv2.putText(panel, name, label_pos, cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness, cv2.LINE_AA)


def draw_metric_box(
    panel: np.ndarray,
    *,
    title: str,
    landmark_rmse: float,
    grid_rmse: float,
    n_points: int,
    color: tuple[int, int, int],
) -> None:
    overlay = panel.copy()
    cv2.rectangle(overlay, (10, 10), (250, 74), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.42, panel, 0.58, 0.0, dst=panel)
    cv2.putText(panel, title, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.56, color, 1, cv2.LINE_AA)
    cv2.putText(panel, f"Landmark RMSE: {landmark_rmse:.2f}px", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.46, color, 1, cv2.LINE_AA)
    cv2.putText(panel, f"Grid RMSE: {grid_rmse:.2f}px  N={n_points}", (20, 68), cv2.FONT_HERSHEY_SIMPLEX, 0.46, color, 1, cv2.LINE_AA)


def save_overlay_stage(
    *,
    background: np.ndarray,
    target_horiz: list[np.ndarray],
    target_vert: list[np.ndarray],
    target_landmarks: dict[str, np.ndarray | None],
    stage_horiz: list[np.ndarray],
    stage_vert: list[np.ndarray],
    stage_landmarks: dict[str, np.ndarray | None],
    step1_labels: set[str],
    step2_extra_labels: set[str],
    slide_cfg: dict[str, object],
    stage_name: str,
    output_path: Path,
) -> tuple[float, float]:
    seq_cfg = slide_cfg["transform_sequence"]
    palette = slide_cfg["palette"]
    panel = cv2.cvtColor(background, cv2.COLOR_GRAY2BGR)
    target_grid_color = hex_to_bgr(str(palette.get("target_grid", "#ffd166")))
    source_grid_color = hex_to_bgr(str(palette.get("source_grid", "#35c9ff")))
    connector_color = hex_to_bgr(str(palette.get("connector", "#f8fafc")))
    text_color = hex_to_bgr(str(palette.get("rmse_text", "#f8fafc")))

    draw_grid_single_color(
        panel,
        target_horiz,
        target_vert,
        target_grid_color,
        outer_thickness=int(seq_cfg.get("grid_thickness_outer", 3)),
        inner_thickness=int(seq_cfg.get("grid_thickness_inner", 1)),
    )
    draw_grid_single_color(
        panel,
        stage_horiz,
        stage_vert,
        source_grid_color,
        outer_thickness=int(seq_cfg.get("grid_thickness_outer", 3)),
        inner_thickness=int(seq_cfg.get("grid_thickness_inner", 1)),
    )

    names, src, dst = landmark_arrays(stage_landmarks, target_landmarks)
    pairs = [
        (
            tuple(np.round(src[index]).astype(int)),
            tuple(np.round(dst[index]).astype(int)),
        )
        for index in range(len(names))
    ]
    draw_translucent_lines(
        panel,
        pairs,
        connector_color,
        alpha=float(seq_cfg.get("connector_alpha", 0.22)),
        thickness=int(seq_cfg.get("connector_thickness", 1)),
    )
    draw_landmarks(
        panel,
        target_landmarks,
        names=names,
        step1_labels=step1_labels,
        step2_extra_labels=step2_extra_labels,
        slide_cfg=slide_cfg,
        stage_name=stage_name,
        target_role=True,
    )
    draw_landmarks(
        panel,
        stage_landmarks,
        names=names,
        step1_labels=step1_labels,
        step2_extra_labels=step2_extra_labels,
        slide_cfg=slide_cfg,
        stage_name=stage_name,
        target_role=False,
    )

    landmark_rmse = rms_point_error(src, dst) if len(names) else float("nan")
    grid_rmse = rms_point_error(flatten_grid_nodes(stage_horiz), flatten_grid_nodes(target_horiz))
    draw_metric_box(
        panel,
        title=stage_name.replace("_", " ").title(),
        landmark_rmse=landmark_rmse,
        grid_rmse=grid_rmse,
        n_points=len(names),
        color=text_color,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), panel)
    return landmark_rmse, grid_rmse


def save_source_only_stage(
    *,
    background: np.ndarray,
    target_horiz: list[np.ndarray],
    stage_horiz: list[np.ndarray],
    stage_vert: list[np.ndarray],
    stage_landmarks: dict[str, np.ndarray | None],
    target_landmarks: dict[str, np.ndarray | None],
    step1_labels: set[str],
    step2_extra_labels: set[str],
    slide_cfg: dict[str, object],
    stage_name: str,
    output_path: Path,
) -> tuple[float, float]:
    seq_cfg = slide_cfg["transform_sequence"]
    palette = slide_cfg["palette"]
    panel = cv2.cvtColor(background, cv2.COLOR_GRAY2BGR)
    source_grid_color = hex_to_bgr(str(palette.get("source_grid", "#35c9ff")))
    text_color = hex_to_bgr(str(palette.get("rmse_text", "#f8fafc")))
    draw_grid_single_color(
        panel,
        stage_horiz,
        stage_vert,
        source_grid_color,
        outer_thickness=int(seq_cfg.get("grid_thickness_outer", 3)),
        inner_thickness=int(seq_cfg.get("grid_thickness_inner", 1)),
    )
    names, src, dst = landmark_arrays(stage_landmarks, target_landmarks)
    draw_landmarks(
        panel,
        stage_landmarks,
        names=names,
        step1_labels=step1_labels,
        step2_extra_labels=step2_extra_labels,
        slide_cfg=slide_cfg,
        stage_name=stage_name,
        target_role=False,
    )
    landmark_rmse = rms_point_error(src, dst) if len(names) else float("nan")
    grid_rmse = rms_point_error(flatten_grid_nodes(stage_horiz), flatten_grid_nodes(target_horiz))
    draw_metric_box(
        panel,
        title=stage_name.replace("_", " ").title(),
        landmark_rmse=landmark_rmse,
        grid_rmse=grid_rmse,
        n_points=len(names),
        color=text_color,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), panel)
    return landmark_rmse, grid_rmse


def save_grids_only_stage(
    *,
    background: np.ndarray,
    target_horiz: list[np.ndarray],
    target_vert: list[np.ndarray],
    stage_horiz: list[np.ndarray],
    stage_vert: list[np.ndarray],
    slide_cfg: dict[str, object],
    output_path: Path,
) -> None:
    """Render only the two grids on background — no landmark markers, labels, or metric box."""
    seq_cfg = slide_cfg["transform_sequence"]
    palette = slide_cfg["palette"]
    panel = cv2.cvtColor(background, cv2.COLOR_GRAY2BGR)
    target_grid_color = hex_to_bgr(str(palette.get("target_grid", "#ffd166")))
    source_grid_color = hex_to_bgr(str(palette.get("source_grid", "#35c9ff")))
    draw_grid_single_color(
        panel,
        target_horiz,
        target_vert,
        target_grid_color,
        outer_thickness=int(seq_cfg.get("grid_thickness_outer", 3)),
        inner_thickness=int(seq_cfg.get("grid_thickness_inner", 1)),
    )
    draw_grid_single_color(
        panel,
        stage_horiz,
        stage_vert,
        source_grid_color,
        outer_thickness=int(seq_cfg.get("grid_thickness_outer", 3)),
        inner_thickness=int(seq_cfg.get("grid_thickness_inner", 1)),
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), panel)


def warp_source_image_affine(
    source_image,
    affine_spec: dict[str, object],
    target_shape: tuple[int, int],
) -> np.ndarray:
    """Warp the full source image into target image space using the affine transform.

    affine_spec: dict with "A" (2x2) and "t" (2,) as lists (from transform_spec["affine"]).
    target_shape: (height, width) of the target image canvas.
    Returns a grayscale uint8 warped image with the same size as the target.
    """
    source_gray = as_grayscale_uint8(source_image)
    A = np.array(affine_spec["A"], dtype=float)
    t = np.array(affine_spec["t"], dtype=float)
    M = np.hstack([A, t.reshape(2, 1)])  # 2x3 forward transform (source XY -> target XY)
    h, w = target_shape[:2]
    warped = cv2.warpAffine(source_gray, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return warped


def crop_to_valid_region(gray: np.ndarray) -> np.ndarray:
    """Crop out rows and columns that are entirely black (value == 0)."""
    mask = gray > 0
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    if not rows.any() or not cols.any():
        return gray
    r0, r1 = int(np.argmax(rows)), int(len(rows) - 1 - np.argmax(rows[::-1]))
    c0, c1 = int(np.argmax(cols)), int(len(cols) - 1 - np.argmax(cols[::-1]))
    return gray[r0:r1 + 1, c0:c1 + 1]


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    p7_image, p7_contours_all = load_frame_vtln(args.p7_basename, args.vtln_dir)
    nnunet_image, nnunet_groundtruth_contours = load_frame_npy(
        args.nnunet_frame,
        args.nnunet_case_dir,
        args.nnunet_case_dir / "contours",
    )
    nnunet_contours_all, target_source_label = load_target_contours_for_export(
        p7_basename=args.p7_basename,
        nnunet_case=args.nnunet_case_dir.relative_to(VT_SEG_DATA_ROOT).as_posix(),
        nnunet_frame=args.nnunet_frame,
        workspace_root=args.workspace_root,
        target_mode=args.target_mode,
        groundtruth_contours=nnunet_groundtruth_contours,
    )
    common_labels = subset_common_contours(p7_contours_all, nnunet_contours_all)
    if not common_labels:
        raise ValueError("No shared contours available for the requested P7/nnUNet pair.")

    p7_contours = {label: np.asarray(p7_contours_all[label], dtype=float).copy() for label in common_labels}
    nnunet_contours = {label: np.asarray(nnunet_contours_all[label], dtype=float).copy() for label in common_labels}
    defaults_payload, _ui_config = load_cv2_app_config_defaults(args.config)
    disabled_landmarks = set(
        resolve_disabled_landmarks_config(
            defaults_payload.get("disabled_landmarks"),
            config_path=args.config,
        )
    )
    top_axis_passes = int(defaults_payload.get("top_axis_smoothing_passes") or 4)
    slide_cfg = load_slide_p7_nnunet_config(args.config)

    bundle = build_transform_bundle(
        source_image=p7_image,
        source_contours=p7_contours,
        target_image=nnunet_image,
        target_contours=nnunet_contours,
        source_frame_number=int(args.p7_basename.split("_")[-1].lstrip("Ff")),
        target_frame_number=args.nnunet_frame,
        disabled_landmarks=disabled_landmarks,
        top_axis_passes=top_axis_passes,
    )
    step1_labels = set(bundle["transform_spec"]["step1_labels"])
    step2_extra_labels = set(bundle["transform_spec"]["step2_labels"]) - step1_labels
    target_gray = as_grayscale_uint8(nnunet_image)

    # Warp the full source image into target space using only the affine transform.
    affine_warped_source = warp_source_image_affine(
        p7_image,
        bundle["transform_spec"]["affine"],
        target_gray.shape,
    )

    outputs = {
        "raw_overlay": args.output_dir / "p7_to_nnunet_raw_overlay.png",
        "affine_overlay": args.output_dir / "p7_to_nnunet_affine_overlay.png",
        "affine_source_only": args.output_dir / "p7_to_nnunet_affine_source_only.png",
        "tps_overlay": args.output_dir / "p7_to_nnunet_tps_overlay.png",
        "tps_source_only": args.output_dir / "p7_to_nnunet_tps_source_only.png",
        "affine_warped_source_overlay": args.output_dir / "p7_to_nnunet_affine_warped_source_overlay.png",
        "affine_warped_source_full": args.output_dir / "p7_to_nnunet_affine_warped_source_full.png",
        "affine_warped_source_cropped": args.output_dir / "p7_to_nnunet_affine_warped_source_cropped.png",
        "affine_nnunet_bg_grids_only": args.output_dir / "p7_to_nnunet_affine_nnunet_bg_grids_only.png",
    }

    raw_landmark_rmse, raw_grid_rmse = save_overlay_stage(
        background=target_gray,
        target_horiz=bundle["target_grid"].horiz_lines,
        target_vert=bundle["target_grid"].vert_lines,
        target_landmarks=bundle["target_landmarks"],
        stage_horiz=bundle["source_grid"].horiz_lines,
        stage_vert=bundle["source_grid"].vert_lines,
        stage_landmarks=bundle["source_landmarks"],
        step1_labels=step1_labels,
        step2_extra_labels=step2_extra_labels,
        slide_cfg=slide_cfg,
        stage_name="raw_overlay",
        output_path=outputs["raw_overlay"],
    )
    affine_landmark_rmse, affine_grid_rmse = save_overlay_stage(
        background=target_gray,
        target_horiz=bundle["target_grid"].horiz_lines,
        target_vert=bundle["target_grid"].vert_lines,
        target_landmarks=bundle["target_landmarks"],
        stage_horiz=bundle["step1_horiz"],
        stage_vert=bundle["step1_vert"],
        stage_landmarks=bundle["step1_landmarks"],
        step1_labels=step1_labels,
        step2_extra_labels=step2_extra_labels,
        slide_cfg=slide_cfg,
        stage_name="affine_overlay",
        output_path=outputs["affine_overlay"],
    )
    save_source_only_stage(
        background=target_gray,
        target_horiz=bundle["target_grid"].horiz_lines,
        stage_horiz=bundle["step1_horiz"],
        stage_vert=bundle["step1_vert"],
        stage_landmarks=bundle["step1_landmarks"],
        target_landmarks=bundle["target_landmarks"],
        step1_labels=step1_labels,
        step2_extra_labels=step2_extra_labels,
        slide_cfg=slide_cfg,
        stage_name="affine_source_only",
        output_path=outputs["affine_source_only"],
    )
    tps_landmark_rmse, tps_grid_rmse = save_overlay_stage(
        background=target_gray,
        target_horiz=bundle["target_grid"].horiz_lines,
        target_vert=bundle["target_grid"].vert_lines,
        target_landmarks=bundle["target_landmarks"],
        stage_horiz=bundle["final_horiz"],
        stage_vert=bundle["final_vert"],
        stage_landmarks=bundle["final_landmarks"],
        step1_labels=step1_labels,
        step2_extra_labels=step2_extra_labels,
        slide_cfg=slide_cfg,
        stage_name="tps_overlay",
        output_path=outputs["tps_overlay"],
    )
    save_source_only_stage(
        background=target_gray,
        target_horiz=bundle["target_grid"].horiz_lines,
        stage_horiz=bundle["final_horiz"],
        stage_vert=bundle["final_vert"],
        stage_landmarks=bundle["final_landmarks"],
        target_landmarks=bundle["target_landmarks"],
        step1_labels=step1_labels,
        step2_extra_labels=step2_extra_labels,
        slide_cfg=slide_cfg,
        stage_name="tps_source_only",
        output_path=outputs["tps_source_only"],
    )

    # Affine-warped source image as overlay background (same grid colors as raw_overlay)
    save_overlay_stage(
        background=affine_warped_source,
        target_horiz=bundle["target_grid"].horiz_lines,
        target_vert=bundle["target_grid"].vert_lines,
        target_landmarks=bundle["target_landmarks"],
        stage_horiz=bundle["step1_horiz"],
        stage_vert=bundle["step1_vert"],
        stage_landmarks=bundle["step1_landmarks"],
        step1_labels=step1_labels,
        step2_extra_labels=step2_extra_labels,
        slide_cfg=slide_cfg,
        stage_name="affine_warped_source_overlay",
        output_path=outputs["affine_warped_source_overlay"],
    )

    # Plain affine-warped source: full canvas (may have black borders)
    outputs["affine_warped_source_full"].parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(outputs["affine_warped_source_full"]), affine_warped_source)

    # Plain affine-warped source: cropped to remove all-black rows/columns
    source_cropped = crop_to_valid_region(affine_warped_source)
    cv2.imwrite(str(outputs["affine_warped_source_cropped"]), source_cropped)

    # nnUNet background + 2 grids after affine (clean: no landmark markers or labels)
    save_grids_only_stage(
        background=target_gray,
        target_horiz=bundle["target_grid"].horiz_lines,
        target_vert=bundle["target_grid"].vert_lines,
        stage_horiz=bundle["step1_horiz"],
        stage_vert=bundle["step1_vert"],
        slide_cfg=slide_cfg,
        output_path=outputs["affine_nnunet_bg_grids_only"],
    )

    print(f"Common labels ({len(common_labels)}): {', '.join(common_labels)}")
    print(f"Target contour source: {target_source_label}")
    print(f"Disabled landmarks: {', '.join(sorted(disabled_landmarks)) if disabled_landmarks else '-'}")
    print(f"Raw overlay RMSE: landmark={raw_landmark_rmse:.2f}px grid={raw_grid_rmse:.2f}px")
    print(f"Affine overlay RMSE: landmark={affine_landmark_rmse:.2f}px grid={affine_grid_rmse:.2f}px")
    print(f"TPS overlay RMSE: landmark={tps_landmark_rmse:.2f}px grid={tps_grid_rmse:.2f}px")
    for name, path in outputs.items():
        print(f"{name}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
