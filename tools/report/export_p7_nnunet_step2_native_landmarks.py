from __future__ import annotations

import argparse
from argparse import Namespace
from pathlib import Path

import cv2
import numpy as np

from grid_transform.annotation_to_grid_workflow import TargetSelection, WorkspaceSelection
from grid_transform.apps.cv2_annotation_to_grid_transform_app import (
    Cv2TransformReviewWindow,
)
from grid_transform.curated_batch import BatchCase
from grid_transform.cv2_config_helpers import resolve_disabled_landmarks_config
from grid_transform.cv2_panels import draw_contours, draw_grid, render_panel_base, world_to_screen
from grid_transform.cv2_shared import hex_to_bgr
from tools.report.slide_p7_nnunet_config import (
    load_slide_p7_nnunet_config,
    stage_label_offset,
    stage_label_position,
    xy_tuple,
)
from grid_transform.cv2_annotation_app_config import APP_CONFIG_PATH, load_cv2_app_config_defaults
from grid_transform.config import DEFAULT_OUTPUT_DIR, DEFAULT_VTLN_DIR, VT_SEG_DATA_ROOT
from grid_transform.image_utils import as_grayscale_uint8
from grid_transform.io import load_frame_npy, load_frame_vtln
from tools.report.common import subset_common_contours

from tools.report.export_p7_nnunet_common_label_images import NNUNET_FRAME_DEFAULT, P7_BASENAME


NNUNET_CASE_DIR_DEFAULT = VT_SEG_DATA_ROOT / "2008-003^01-1791" / "test"
PANEL_SIZE = (720, 720)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Export Step-2-style native landmark/grid overlays for P7 and the bundled nnUNet case, "
            "restricted to the anatomical elements shared by both."
        )
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
        "--config",
        type=Path,
        default=APP_CONFIG_PATH,
        help="Config file used by the cv2 annotation-to-grid-transform app.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR / "slide" / "images",
        help="Output folder for the generated PNG images.",
    )
    parser.add_argument(
        "--use-predicted-masks",
        action="store_true",
        default=False,
        help="Load nnUNet contours from predicted mask PNGs instead of ground-truth ROI files.",
    )
    return parser.parse_args(argv)
def build_export_selection(
    *,
    vtln_dir: Path,
    p7_basename: str,
    nnunet_frame: int,
    output_dir: Path,
) -> WorkspaceSelection:
    case = BatchCase(
        output_basename=p7_basename,
        speaker="P7",
        raw_subject="1640",
        session="S2",
        frame_index_1based=int(p7_basename.split("_")[-1].lstrip("Ff")),
        annotation_status="export_only",
        annotation_source_path=None,
        reference_bundle_dir=vtln_dir,
        reference_bundle_name=p7_basename,
    )
    target = TargetSelection(
        target_type="nnunet",
        nnunet_case="2008-003^01-1791/test",
        nnunet_frame=nnunet_frame,
        vtln_reference=p7_basename,
        vtln_dir=vtln_dir,
    )
    return WorkspaceSelection(
        workspace_id=f"export_{p7_basename}_to_nnunet_{nnunet_frame}",
        workspace_dir=output_dir / "_tmp_step2_ui_export",
        case=case,
        artspeech_root=Path("."),
        dataset_root=None,
        reference_speaker="P7",
        reference_image_path=vtln_dir / f"{p7_basename}.png",
        reference_zip_path=vtln_dir / f"{p7_basename}.zip",
        reference_bundle_dir=vtln_dir,
        target=target,
        source_alias_note="export_only",
    )


def render_native_panel_without_header(review_window: Cv2TransformReviewWindow, pane_name: str) -> np.ndarray:
    geometry = review_window.pane_geometries[pane_name]
    base_image = review_window.source_image if pane_name == "source_native" else review_window.target_image
    panel, mapping = render_panel_base(base_image, geometry.width, geometry.height, review_window.pane_views[pane_name])
    review_window.last_panel_mappings[pane_name] = mapping
    step1, step2_extra = review_window._step_sets()
    slide_config = load_slide_p7_nnunet_config(review_window.args.config)
    step2_cfg = slide_config["step2_native"]
    palette = slide_config["palette"]
    marker_radius = int(step2_cfg.get("marker_radius", 4))
    marker_highlight_radius = int(step2_cfg.get("marker_highlight_radius", marker_radius + 1))
    default_label_offset = xy_tuple(step2_cfg.get("default_label_offset"), default=(8.0, -8.0))
    label_font_scale = float(step2_cfg.get("label_font_scale", 0.45))
    label_thickness = int(step2_cfg.get("label_thickness", 1))
    affine_color = hex_to_bgr(str(palette.get("affine_anchor", "#fb8500")))
    tps_color = hex_to_bgr(str(palette.get("tps_extra", "#80ed99")))
    other_color = hex_to_bgr(str(palette.get("other_landmark", "#ffd166")))

    def draw_points_compact(points: dict[str, np.ndarray | None]) -> None:
        for name, point in points.items():
            if point is None:
                continue
            point_arr = np.asarray(point, dtype=float)
            if point_arr.ndim != 1 or point_arr.shape[0] != 2:
                continue
            sx, sy = np.round(world_to_screen(mapping, point_arr.reshape(1, 2))[0]).astype(int)
            if name in step1:
                color = affine_color
                marker = "circle"
            elif name in step2_extra:
                color = tps_color
                marker = "diamond"
            else:
                color = other_color
                marker = "ring"
            radius = marker_highlight_radius if False else marker_radius
            if marker == "circle":
                cv2.circle(panel, (sx, sy), radius + 1, (255, 255, 255), -1, cv2.LINE_AA)
                cv2.circle(panel, (sx, sy), radius, color, -1, cv2.LINE_AA)
            elif marker == "diamond":
                diamond = np.array(
                    [[sx, sy - radius - 1], [sx + radius + 1, sy], [sx, sy + radius + 1], [sx - radius - 1, sy]],
                    dtype=np.int32,
                )
                cv2.fillConvexPoly(panel, diamond, (255, 255, 255), cv2.LINE_AA)
                diamond_inner = np.array(
                    [[sx, sy - radius], [sx + radius, sy], [sx, sy + radius], [sx - radius, sy]],
                    dtype=np.int32,
                )
                cv2.fillConvexPoly(panel, diamond_inner, color, cv2.LINE_AA)
            else:
                cv2.circle(panel, (sx, sy), radius + 1, color, 1, cv2.LINE_AA)
                cv2.circle(panel, (sx, sy), max(1, radius - 1), (255, 255, 255), 1, cv2.LINE_AA)
            absolute_position = stage_label_position(slide_config, "step2_native", pane_name, name)
            if absolute_position is None:
                label_offset = stage_label_offset(
                    slide_config,
                    "step2_native",
                    pane_name,
                    name,
                    default_offset=default_label_offset,
                )
                label_x = int(round(sx + label_offset[0]))
                label_y = int(round(sy + label_offset[1]))
            else:
                label_x = int(round(absolute_position[0]))
                label_y = int(round(absolute_position[1]))
            cv2.putText(
                panel,
                name,
                (label_x, label_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                label_font_scale,
                color,
                label_thickness,
                cv2.LINE_AA,
            )

    if pane_name == "source_native":
        draw_contours(panel, mapping, review_window.source_contours)
        draw_grid(
            panel,
            mapping,
            review_window.bundle["source_grid"].horiz_lines,
            review_window.bundle["source_grid"].vert_lines,
            color_h=(255, 190, 11),
            color_v=(60, 134, 255),
        )
        draw_points_compact(review_window.bundle["source_landmarks"])
    else:
        draw_contours(panel, mapping, review_window.target_contours)
        draw_grid(
            panel,
            mapping,
            review_window.bundle["target_grid"].horiz_lines,
            review_window.bundle["target_grid"].vert_lines,
            color_h=(255, 190, 11),
            color_v=(60, 134, 255),
        )
        draw_points_compact(review_window.bundle["target_landmarks"])

    trim = 1
    y0 = mapping.offset_y + trim
    y1 = mapping.offset_y + mapping.display_h - trim
    x0 = mapping.offset_x + trim
    x1 = mapping.offset_x + mapping.display_w - trim
    return panel[y0:y1, x0:x1].copy()


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    p7_image, p7_contours_all = load_frame_vtln(args.p7_basename, args.vtln_dir)
    nnunet_image, nnunet_contours_all = load_frame_npy(
        args.nnunet_frame,
        args.nnunet_case_dir,
        args.nnunet_case_dir / "contours",
        use_predicted_masks=args.use_predicted_masks,
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
    ui_args = Namespace(
        disabled_landmarks=tuple(sorted(disabled_landmarks)),
        top_axis_smoothing_passes=top_axis_passes,
        config=args.config,
    )
    selection = build_export_selection(
        vtln_dir=args.vtln_dir,
        p7_basename=args.p7_basename,
        nnunet_frame=args.nnunet_frame,
        output_dir=args.output_dir,
    )
    review_window = Cv2TransformReviewWindow(
        selection=selection,
        source_image=as_grayscale_uint8(p7_image),
        source_contours=p7_contours,
        target_image=as_grayscale_uint8(nnunet_image),
        target_contours=nnunet_contours,
        ui_config=_ui_config,
        args=ui_args,
    )

    p7_output = args.output_dir / "p7_step2_native_common_landmarks.png"
    nnunet_output = args.output_dir / "nnunet_step2_native_common_landmarks.png"
    p7_panel = render_native_panel_without_header(review_window, "source_native")
    nnunet_panel = render_native_panel_without_header(review_window, "target_native")
    p7_output.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(p7_output), p7_panel)
    cv2.imwrite(str(nnunet_output), nnunet_panel)

    print(f"Common labels ({len(common_labels)}): {', '.join(common_labels)}")
    print(f"Disabled landmarks from config: {', '.join(sorted(disabled_landmarks)) if disabled_landmarks else '-'}")
    print(f"Step1 labels: {', '.join(review_window.bundle['transform_spec']['step1_labels'])}")
    print(
        f"Step2 extra labels: {', '.join(sorted(set(review_window.bundle['transform_spec']['step2_labels']) - set(review_window.bundle['transform_spec']['step1_labels'])))}"
    )
    print(f"Saved P7 Step-2 native panel: {p7_output}")
    print(f"Saved nnUNet Step-2 native panel: {nnunet_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
