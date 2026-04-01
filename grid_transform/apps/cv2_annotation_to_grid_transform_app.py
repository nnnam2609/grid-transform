from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import cv2
import numpy as np

from grid_transform.annotation_to_grid_workflow import (
    DEFAULT_MAPPING_DOC,
    DEFAULT_NNUNET_TARGET_CASE,
    DEFAULT_NNUNET_TARGET_FRAME,
    DEFAULT_WORKSPACE_ROOT,
    MappingInfo,
    TargetSelection,
    WorkspaceSelection,
    apply_transform_from_spec,
    build_transform_bundle,
    build_workspace_selection,
    load_annotation_state_if_available,
    load_curated_cases,
    load_source_context,
    load_target_context,
    parse_mapping_doc,
    reference_name_for_case,
    save_annotation_state,
    save_json,
    serialize_points_map,
    source_annotation_metadata,
    step_file_paths,
    target_annotation_metadata,
    workspace_selection_to_payload,
)
from grid_transform.apps.edit_source_annotation import (
    CONTOUR_COLORS,
    WINDOW_MARGIN,
    clamp_point,
    hex_to_bgr,
    scaled_window_size,
    source_text_block,
)
from grid_transform.apps.run_curated_u_annotation_batch import DEFAULT_MANIFEST_CSV
from grid_transform.config import PROJECT_DIR
from grid_transform.source_annotation import LONG_CONTOUR_HANDLE_COUNTS
from grid_transform.warp import precompute_inverse_warp, warp_array_with_precomputed_inverse_warp


APP_CONFIG_PATH = PROJECT_DIR / "config.yaml"
APP_CONFIG_SECTION = "cv2_annotation_to_grid_transform"
VALID_CACHE_MODES = {"startup", "first-load"}
VALID_WINDOW_MODES = {"full-height", "fullscreen", "fixed"}

WINDOW_NAME_STEP0 = "Annotation To Grid Transform - Step 0"
WINDOW_NAME_STEP1 = "Annotation To Grid Transform - Step 1"
WINDOW_NAME_STEP2 = "Annotation To Grid Transform - Step 2"
WINDOW_NAME_STEP3 = "Annotation To Grid Transform - Step 3"

PANEL_BG = (24, 24, 24)
SIDEBAR_BG = (18, 18, 18)
FOOTER_BG = (10, 10, 10)
TEXT_COLOR = (235, 235, 235)
MUTED_TEXT = (180, 180, 180)
ACCENT = (120, 220, 255)
ROW_SELECTED = (64, 104, 164)
ROW_BG = (38, 38, 38)
AFFINE_COLOR = (0, 140, 255)
TPS_COLOR = (40, 235, 80)
OTHER_LANDMARK_COLOR = (255, 230, 0)

KEY_UP = 2490368
KEY_DOWN = 2621440
KEY_LEFT = 2424832
KEY_RIGHT = 2555904
KEY_TAB = 9
KEY_ENTER = 13
KEY_BACKSPACE = 8

EDITOR_PANEL_W = 960
EDITOR_PANEL_H = 960
EDITOR_SIDEBAR_W = 320
EDITOR_FOOTER_H = 88
EDITOR_LOUPE = 240

REVIEW_PANE_W = 480
REVIEW_PANE_H = 340
PANE_GAP = 12
REVIEW_FOOTER_H = 176

UI_CONFIG_KEYS = {
    "editor_panel_width": "EDITOR_PANEL_W",
    "editor_panel_height": "EDITOR_PANEL_H",
    "editor_sidebar_width": "EDITOR_SIDEBAR_W",
    "editor_footer_height": "EDITOR_FOOTER_H",
    "editor_loupe_size": "EDITOR_LOUPE",
    "review_pane_width": "REVIEW_PANE_W",
    "review_pane_height": "REVIEW_PANE_H",
    "pane_gap": "PANE_GAP",
    "review_footer_height": "REVIEW_FOOTER_H",
}


@dataclass
class ViewState:
    zoom: float
    center_x: float
    center_y: float


@dataclass
class PaneGeometry:
    x0: int
    y0: int
    width: int
    height: int


@dataclass
class PanelMapping:
    image_w: int
    image_h: int
    crop_x0: float
    crop_y0: float
    crop_w: float
    crop_h: float
    offset_x: int
    offset_y: int
    scale: float
    display_w: int
    display_h: int


def _read_yaml_config(path: Path) -> dict[str, object]:
    if not path.is_file():
        return {}
    try:
        import yaml
    except ModuleNotFoundError as exc:
        raise SystemExit(
            f"config file {path} requires PyYAML. Install dependencies from requirements.txt first."
        ) from exc

    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise SystemExit(f"config file {path} must contain a YAML mapping at the top level.")
    return payload


def _resolve_config_path(value: object, *, default: Path, config_path: Path) -> Path:
    if value in (None, ""):
        return default
    candidate = Path(str(value))
    if candidate.is_absolute():
        return candidate
    return (config_path.parent / candidate).resolve()


def _resolve_config_int(value: object, *, default: int) -> int:
    if value in (None, ""):
        return int(default)
    return int(value)


def _resolve_config_str(value: object, *, default: str) -> str:
    if value in (None, ""):
        return default
    return str(value)


def _resolve_choice(value: object, *, default: str, valid: set[str], field_name: str) -> str:
    resolved = _resolve_config_str(value, default=default)
    if resolved not in valid:
        raise SystemExit(f"Invalid {field_name}={resolved!r} in {APP_CONFIG_PATH.name}. Valid choices: {sorted(valid)}")
    return resolved


def _apply_ui_config(ui_payload: dict[str, object]) -> None:
    globals_ns = globals()
    for key, global_name in UI_CONFIG_KEYS.items():
        if key not in ui_payload or ui_payload[key] in (None, ""):
            continue
        globals_ns[global_name] = int(ui_payload[key])


def _load_app_config_defaults(config_path: Path) -> tuple[dict[str, object], dict[str, object]]:
    payload = _read_yaml_config(config_path)
    section = payload.get(APP_CONFIG_SECTION, {}) or {}
    if not isinstance(section, dict):
        raise SystemExit(f"Section {APP_CONFIG_SECTION!r} in {config_path} must be a mapping.")
    defaults_payload = section.get("defaults", {}) or {}
    ui_payload = section.get("ui", {}) or {}
    if not isinstance(defaults_payload, dict):
        raise SystemExit(f"Section {APP_CONFIG_SECTION}.defaults in {config_path} must be a mapping.")
    if not isinstance(ui_payload, dict):
        raise SystemExit(f"Section {APP_CONFIG_SECTION}.ui in {config_path} must be a mapping.")
    _apply_ui_config(ui_payload)
    return defaults_payload, ui_payload


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config", type=Path, default=APP_CONFIG_PATH)
    pre_args, _ = pre_parser.parse_known_args(argv)
    defaults_payload, _ui_payload = _load_app_config_defaults(pre_args.config)

    parser = argparse.ArgumentParser(description="CV2 multi-step annotation-to-grid-transform workflow.")
    parser.add_argument(
        "--config",
        type=Path,
        default=pre_args.config,
        help="YAML config file. CLI flags override YAML defaults.",
    )
    parser.add_argument(
        "--manifest-csv",
        type=Path,
        default=_resolve_config_path(defaults_payload.get("manifest_csv"), default=DEFAULT_MANIFEST_CSV, config_path=pre_args.config),
        help="Curated manifest CSV.",
    )
    parser.add_argument(
        "--artspeech-root",
        type=Path,
        default=_resolve_config_path(
            defaults_payload.get("artspeech_root"),
            default=PROJECT_DIR.parent / "Data" / "Artspeech_database",
            config_path=pre_args.config,
        ),
    )
    parser.add_argument(
        "--mapping-doc",
        type=Path,
        default=_resolve_config_path(defaults_payload.get("mapping_doc"), default=DEFAULT_MAPPING_DOC, config_path=pre_args.config),
    )
    parser.add_argument(
        "--workspace-root",
        type=Path,
        default=_resolve_config_path(defaults_payload.get("workspace_root"), default=DEFAULT_WORKSPACE_ROOT, config_path=pre_args.config),
    )
    parser.add_argument(
        "--vtln-dir",
        type=Path,
        default=_resolve_config_path(defaults_payload.get("vtln_dir"), default=PROJECT_DIR / "VTLN", config_path=pre_args.config),
    )
    parser.add_argument("--default-target-case", default=_resolve_config_str(defaults_payload.get("default_target_case"), default=DEFAULT_NNUNET_TARGET_CASE))
    parser.add_argument("--default-target-frame", type=int, default=_resolve_config_int(defaults_payload.get("default_target_frame"), default=DEFAULT_NNUNET_TARGET_FRAME))
    parser.add_argument("--render-workers", type=int, default=_resolve_config_int(defaults_payload.get("render_workers"), default=8))
    parser.add_argument("--render-prefetch", type=int, default=_resolve_config_int(defaults_payload.get("render_prefetch"), default=0))
    parser.add_argument(
        "--cache-mode",
        choices=("startup", "first-load"),
        default=_resolve_choice(
            defaults_payload.get("cache_mode"),
            default="startup",
            valid=VALID_CACHE_MODES,
            field_name="cache_mode",
        ),
        help=(
            "Cache behavior for source/target context loading. "
            "'startup' prewarms the current selection from step 0. "
            "'first-load' loads and caches only when the selection is opened the first time."
        ),
    )
    parser.add_argument(
        "--window-mode",
        choices=("full-height", "fullscreen", "fixed"),
        default=_resolve_choice(
            defaults_payload.get("window_mode"),
            default="full-height",
            valid=VALID_WINDOW_MODES,
            field_name="window_mode",
        ),
        help="Initial cv2 window mode for all steps.",
    )
    return parser.parse_args(argv)


def bgr_gray(image: np.ndarray) -> np.ndarray:
    array = np.asarray(image)
    if array.ndim == 3:
        array = array[..., 0]
    array = np.clip(array, 0, 255).astype(np.uint8)
    return cv2.cvtColor(array, cv2.COLOR_GRAY2BGR)


def ensure_view_state(view: ViewState | None, image_shape: tuple[int, int]) -> ViewState:
    height, width = image_shape[:2]
    if view is None:
        return ViewState(zoom=1.0, center_x=width / 2.0, center_y=height / 2.0)
    zoom = float(np.clip(view.zoom, 1.0, 12.0))
    crop_w = width / zoom
    crop_h = height / zoom
    cx = float(np.clip(view.center_x, crop_w / 2.0, max(crop_w / 2.0, width - crop_w / 2.0)))
    cy = float(np.clip(view.center_y, crop_h / 2.0, max(crop_h / 2.0, height - crop_h / 2.0)))
    return ViewState(zoom=zoom, center_x=cx, center_y=cy)


def build_panel_mapping(image_shape: tuple[int, int], panel_w: int, panel_h: int, view: ViewState) -> PanelMapping:
    image_h, image_w = image_shape[:2]
    view = ensure_view_state(view, image_shape)
    crop_w = image_w / view.zoom
    crop_h = image_h / view.zoom
    crop_x0 = float(np.clip(view.center_x - crop_w / 2.0, 0.0, max(0.0, image_w - crop_w)))
    crop_y0 = float(np.clip(view.center_y - crop_h / 2.0, 0.0, max(0.0, image_h - crop_h)))
    scale = min(panel_w / crop_w, panel_h / crop_h)
    display_w = max(1, int(round(crop_w * scale)))
    display_h = max(1, int(round(crop_h * scale)))
    offset_x = int(round((panel_w - display_w) / 2.0))
    offset_y = int(round((panel_h - display_h) / 2.0))
    return PanelMapping(
        image_w=image_w,
        image_h=image_h,
        crop_x0=crop_x0,
        crop_y0=crop_y0,
        crop_w=crop_w,
        crop_h=crop_h,
        offset_x=offset_x,
        offset_y=offset_y,
        scale=scale,
        display_w=display_w,
        display_h=display_h,
    )


def render_panel_base(image: np.ndarray, panel_w: int, panel_h: int, view: ViewState) -> tuple[np.ndarray, PanelMapping]:
    mapping = build_panel_mapping(tuple(image.shape[:2]), panel_w, panel_h, view)
    canvas = np.full((panel_h, panel_w, 3), PANEL_BG, dtype=np.uint8)
    x0 = int(np.floor(mapping.crop_x0))
    y0 = int(np.floor(mapping.crop_y0))
    x1 = int(np.ceil(mapping.crop_x0 + mapping.crop_w))
    y1 = int(np.ceil(mapping.crop_y0 + mapping.crop_h))
    x0 = int(np.clip(x0, 0, image.shape[1] - 1))
    y0 = int(np.clip(y0, 0, image.shape[0] - 1))
    x1 = int(np.clip(max(x1, x0 + 1), x0 + 1, image.shape[1]))
    y1 = int(np.clip(max(y1, y0 + 1), y0 + 1, image.shape[0]))
    crop = image[y0:y1, x0:x1]
    crop_bgr = bgr_gray(crop)
    resized = cv2.resize(crop_bgr, (mapping.display_w, mapping.display_h), interpolation=cv2.INTER_LINEAR)
    canvas[
        mapping.offset_y : mapping.offset_y + mapping.display_h,
        mapping.offset_x : mapping.offset_x + mapping.display_w,
    ] = resized
    cv2.rectangle(
        canvas,
        (mapping.offset_x, mapping.offset_y),
        (mapping.offset_x + mapping.display_w - 1, mapping.offset_y + mapping.display_h - 1),
        (70, 70, 70),
        1,
        lineType=cv2.LINE_AA,
    )
    return canvas, mapping


def world_to_screen(mapping: PanelMapping, pts: np.ndarray) -> np.ndarray:
    arr = np.asarray(pts, dtype=float)
    return np.column_stack(
        [
            mapping.offset_x + (arr[:, 0] - mapping.crop_x0) * mapping.scale,
            mapping.offset_y + (arr[:, 1] - mapping.crop_y0) * mapping.scale,
        ]
    )


def screen_to_world(mapping: PanelMapping, x: int, y: int) -> tuple[float, float] | None:
    if not (
        mapping.offset_x <= x < mapping.offset_x + mapping.display_w
        and mapping.offset_y <= y < mapping.offset_y + mapping.display_h
    ):
        return None
    wx = mapping.crop_x0 + (x - mapping.offset_x) / mapping.scale
    wy = mapping.crop_y0 + (y - mapping.offset_y) / mapping.scale
    return float(wx), float(wy)


def draw_polyline(
    panel: np.ndarray,
    mapping: PanelMapping,
    pts: np.ndarray,
    color: tuple[int, int, int],
    *,
    dashed: bool = False,
    thickness: int = 2,
) -> None:
    arr = np.asarray(pts, dtype=float)
    if len(arr) < 2:
        return
    scaled = np.round(world_to_screen(mapping, arr)).astype(np.int32)
    if dashed:
        for index in range(len(scaled) - 1):
            if index % 2 == 0:
                cv2.line(panel, tuple(scaled[index]), tuple(scaled[index + 1]), color, thickness, cv2.LINE_AA)
    else:
        cv2.polylines(panel, [scaled.reshape(-1, 1, 2)], False, color, thickness, cv2.LINE_AA)


def draw_points(
    panel: np.ndarray,
    mapping: PanelMapping,
    points: dict[str, np.ndarray | None],
    *,
    step1_labels: set[str],
    step2_extra_labels: set[str],
    highlight: str | None = None,
    draw_labels: bool = True,
) -> None:
    for name, point in points.items():
        if point is None:
            continue
        point_arr = np.asarray(point, dtype=float)
        if point_arr.ndim != 1 or point_arr.shape[0] != 2:
            continue
        point_arr = point_arr.reshape(1, 2)
        sx, sy = np.round(world_to_screen(mapping, point_arr)[0]).astype(int)
        if name in step1_labels:
            color = AFFINE_COLOR
            marker = "circle"
        elif name in step2_extra_labels:
            color = TPS_COLOR
            marker = "diamond"
        else:
            color = OTHER_LANDMARK_COLOR
            marker = "ring"
        radius = 7 if name == highlight else 6
        if marker == "circle":
            cv2.circle(panel, (sx, sy), radius + 2, (255, 255, 255), -1, cv2.LINE_AA)
            cv2.circle(panel, (sx, sy), radius, color, -1, cv2.LINE_AA)
        elif marker == "diamond":
            diamond = np.array(
                [[sx, sy - radius - 2], [sx + radius + 2, sy], [sx, sy + radius + 2], [sx - radius - 2, sy]],
                dtype=np.int32,
            )
            cv2.fillConvexPoly(panel, diamond, (255, 255, 255), cv2.LINE_AA)
            diamond_inner = np.array(
                [[sx, sy - radius], [sx + radius, sy], [sx, sy + radius], [sx - radius, sy]],
                dtype=np.int32,
            )
            cv2.fillConvexPoly(panel, diamond_inner, color, cv2.LINE_AA)
        else:
            cv2.circle(panel, (sx, sy), radius + 2, color, 2, cv2.LINE_AA)
            cv2.circle(panel, (sx, sy), max(1, radius - 1), (255, 255, 255), 1, cv2.LINE_AA)
        if draw_labels:
            cv2.putText(panel, name, (sx + 6, sy - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)


def draw_grid(
    panel: np.ndarray,
    mapping: PanelMapping,
    horiz_lines: list[np.ndarray],
    vert_lines: list[np.ndarray],
    *,
    color_h: tuple[int, int, int],
    color_v: tuple[int, int, int],
) -> None:
    for index, line in enumerate(horiz_lines):
        draw_polyline(panel, mapping, np.asarray(line, dtype=float), color_h, thickness=3 if index in (0, len(horiz_lines) - 1) else 1)
    for index, line in enumerate(vert_lines):
        draw_polyline(panel, mapping, np.asarray(line, dtype=float), color_v, thickness=3 if index in (0, len(vert_lines) - 1) else 1)


def draw_contours(panel: np.ndarray, mapping: PanelMapping, contours: dict[str, np.ndarray], *, dim: bool = False) -> None:
    for name, points in sorted(contours.items()):
        color = hex_to_bgr(CONTOUR_COLORS.get(name, "#00b4d8"))
        if dim:
            color = tuple(int(channel * 0.5) for channel in color)
        dashed = name in {"pharynx", "soft-palate-midline"}
        draw_polyline(panel, mapping, np.asarray(points, dtype=float), color, dashed=dashed, thickness=2)


def draw_header_text(panel: np.ndarray, lines: list[str], *, color: tuple[int, int, int] = TEXT_COLOR) -> None:
    overlay = panel.copy()
    height = 14 + 24 * len(lines)
    cv2.rectangle(overlay, (8, 8), (panel.shape[1] - 8, 8 + height), (0, 0, 0), -1)
    panel[:] = cv2.addWeighted(overlay, 0.32, panel, 0.68, 0.0)
    for row, line in enumerate(lines):
        cv2.putText(panel, line, (18, 30 + row * 24), cv2.FONT_HERSHEY_SIMPLEX, 0.58, color, 1, cv2.LINE_AA)


def draw_sidebar_block(panel: np.ndarray, title: str, lines: list[str], top: int, *, highlight: bool = False) -> int:
    block_h = 40 + 22 * len(lines)
    color = ACCENT if highlight else TEXT_COLOR
    cv2.rectangle(panel, (12, top), (panel.shape[1] - 12, top + block_h), (42, 42, 42), -1)
    cv2.rectangle(panel, (12, top), (panel.shape[1] - 12, top + block_h), (80, 80, 80), 1)
    cv2.putText(panel, title, (24, top + 26), cv2.FONT_HERSHEY_SIMPLEX, 0.58, color, 1, cv2.LINE_AA)
    for index, line in enumerate(lines):
        cv2.putText(panel, line, (24, top + 52 + 20 * index), cv2.FONT_HERSHEY_SIMPLEX, 0.47, TEXT_COLOR, 1, cv2.LINE_AA)
    return top + block_h + 12


def save_canvas_preview(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), image)


def format_label_list(prefix: str, labels: list[str], *, max_chars: int = 88) -> list[str]:
    if not labels:
        return [f"{prefix}: -"]
    lines: list[str] = []
    current = f"{prefix}:"
    for label in labels:
        candidate = f"{current} {label}" if current != f"{prefix}:" else f"{current} {label}"
        if len(candidate) > max_chars and current != f"{prefix}:":
            lines.append(current)
            current = f"  {label}"
        else:
            current = candidate
    lines.append(current)
    return lines


def draw_step2_marker_legend(canvas: np.ndarray, x: int, y: int) -> None:
    cv2.circle(canvas, (x + 10, y - 4), 7, (255, 255, 255), -1, cv2.LINE_AA)
    cv2.circle(canvas, (x + 10, y - 4), 5, AFFINE_COLOR, -1, cv2.LINE_AA)
    cv2.putText(canvas, "Affine anchors", (x + 26, y), cv2.FONT_HERSHEY_SIMPLEX, 0.46, AFFINE_COLOR, 1, cv2.LINE_AA)

    diamond = np.array([[x + 10, y + 20], [x + 18, y + 28], [x + 10, y + 36], [x + 2, y + 28]], dtype=np.int32)
    cv2.fillConvexPoly(canvas, diamond, TPS_COLOR, cv2.LINE_AA)
    cv2.putText(canvas, "TPS extra controls", (x + 26, y + 32), cv2.FONT_HERSHEY_SIMPLEX, 0.46, TPS_COLOR, 1, cv2.LINE_AA)

    cv2.circle(canvas, (x + 10, y + 60), 7, OTHER_LANDMARK_COLOR, 2, cv2.LINE_AA)
    cv2.circle(canvas, (x + 10, y + 60), 4, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(canvas, "Other visible landmarks", (x + 26, y + 64), cv2.FONT_HERSHEY_SIMPLEX, 0.46, OTHER_LANDMARK_COLOR, 1, cv2.LINE_AA)


class ContextLoadCache:
    def __init__(self) -> None:
        self._executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="app-context-cache")
        self._source_futures: dict[str, Future[dict[str, object]]] = {}
        self._target_futures: dict[str, Future[dict[str, object]]] = {}
        self._source_cache: dict[str, dict[str, object]] = {}
        self._target_cache: dict[str, dict[str, object]] = {}

    @staticmethod
    def _source_key(selection: WorkspaceSelection) -> str:
        return selection.workspace_id

    @staticmethod
    def _target_key(selection: WorkspaceSelection) -> str:
        return selection.workspace_id

    def prewarm(self, selection: WorkspaceSelection) -> None:
        source_key = self._source_key(selection)
        if source_key not in self._source_cache and source_key not in self._source_futures:
            self._source_futures[source_key] = self._executor.submit(load_source_context, selection)
        target_key = self._target_key(selection)
        if target_key not in self._target_cache and target_key not in self._target_futures:
            self._target_futures[target_key] = self._executor.submit(load_target_context, selection)

    def get_source(self, selection: WorkspaceSelection) -> dict[str, object]:
        key = self._source_key(selection)
        if key in self._source_cache:
            return self._source_cache[key]
        future = self._source_futures.pop(key, None)
        payload = future.result() if future is not None else load_source_context(selection)
        self._source_cache[key] = payload
        return payload

    def get_target(self, selection: WorkspaceSelection) -> dict[str, object]:
        key = self._target_key(selection)
        if key in self._target_cache:
            return self._target_cache[key]
        future = self._target_futures.pop(key, None)
        payload = future.result() if future is not None else load_target_context(selection)
        self._target_cache[key] = payload
        return payload

    def cache_counts(self) -> tuple[int, int]:
        return len(self._source_cache) + len(self._source_futures), len(self._target_cache) + len(self._target_futures)

    def close(self) -> None:
        self._executor.shutdown(wait=False, cancel_futures=True)


class Cv2SelectionWindow:
    def __init__(
        self,
        *,
        cases,
        mapping: MappingInfo,
        args: argparse.Namespace,
        prewarm_selection: Callable[[WorkspaceSelection], None] | None = None,
        cache_counts: Callable[[], tuple[int, int]] | None = None,
    ) -> None:
        self.cases = cases
        self.mapping = mapping
        self.args = args
        self.prewarm_selection = prewarm_selection
        self.cache_counts = cache_counts
        self.case_index = 0
        self.focus = "speaker"
        self.target_type = "nnunet"
        self.nnunet_case = args.default_target_case
        self.nnunet_frame = str(int(args.default_target_frame))
        self.vtln_reference = reference_name_for_case(self.cases[0]) if self.cases else ""
        self.vtln_dir = str(args.vtln_dir)
        self.click_targets: dict[str, tuple[int, int, int, int]] = {}
        self._queue_prewarm()

    def _current_case(self):
        return self.cases[self.case_index]

    def _reseed_vtln_reference_if_needed(self) -> None:
        if not self.vtln_reference:
            self.vtln_reference = reference_name_for_case(self._current_case())

    def _cycle_case(self, delta: int) -> None:
        self.case_index = int(np.clip(self.case_index + delta, 0, len(self.cases) - 1))
        if self.focus == "speaker":
            self.vtln_reference = reference_name_for_case(self._current_case())
        self._queue_prewarm()

    def _toggle_target_type(self) -> None:
        self.target_type = "vtln" if self.target_type == "nnunet" else "nnunet"
        if self.target_type == "vtln":
            self._reseed_vtln_reference_if_needed()
        self._queue_prewarm()

    def _active_field_order(self) -> list[str]:
        order = ["speaker", "target_type"]
        if self.target_type == "nnunet":
            order.extend(["nnunet_case", "nnunet_frame"])
        else:
            order.extend(["vtln_reference", "vtln_dir"])
        return order

    def _focus_next(self) -> None:
        order = self._active_field_order()
        index = order.index(self.focus) if self.focus in order else 0
        self.focus = order[(index + 1) % len(order)]

    def _handle_text_input(self, key: int) -> None:
        if self.focus not in {"nnunet_case", "nnunet_frame", "vtln_reference", "vtln_dir"}:
            return
        if key == KEY_BACKSPACE:
            current = getattr(self, self.focus)
            setattr(self, self.focus, current[:-1])
            return
        if 32 <= key <= 126:
            char = chr(key)
            if self.focus == "nnunet_frame" and not char.isdigit():
                return
            setattr(self, self.focus, getattr(self, self.focus) + char)

    def _queue_prewarm(self) -> None:
        if self.prewarm_selection is None:
            return
        try:
            self.prewarm_selection(self._build_selection())
        except Exception:
            return

    def _draw_field(self, canvas: np.ndarray, *, label: str, value: str, x0: int, y0: int, w: int, h: int, field_name: str) -> int:
        active = self.focus == field_name
        fill = (58, 58, 58) if active else (34, 34, 34)
        outline = ACCENT if active else (90, 90, 90)
        cv2.rectangle(canvas, (x0, y0), (x0 + w, y0 + h), fill, -1)
        cv2.rectangle(canvas, (x0, y0), (x0 + w, y0 + h), outline, 1)
        cv2.putText(canvas, label, (x0 + 10, y0 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.48, MUTED_TEXT, 1, cv2.LINE_AA)
        cv2.putText(canvas, value or "-", (x0 + 10, y0 + 46), cv2.FONT_HERSHEY_SIMPLEX, 0.52, TEXT_COLOR, 1, cv2.LINE_AA)
        self.click_targets[field_name] = (x0, y0, w, h)
        return y0 + h + 10

    def _render(self) -> np.ndarray:
        canvas = np.full((820, 1460, 3), PANEL_BG, dtype=np.uint8)
        self.click_targets = {}
        cv2.putText(canvas, "Step 0 - Select speaker, source, and target", (20, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.84, TEXT_COLOR, 2, cv2.LINE_AA)
        cv2.putText(
            canvas,
            "Enter continue | Up/Down speaker | Tab focus | Left/Right toggle target mode | X exit app",
            (20, 62),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            MUTED_TEXT,
            1,
            cv2.LINE_AA,
        )

        left = canvas[:, :420]
        middle = canvas[:, 430:930]
        right = canvas[:, 940:]
        left[:] = SIDEBAR_BG
        middle[:] = SIDEBAR_BG
        right[:] = SIDEBAR_BG

        cv2.putText(left, "Curated Speakers", (20, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.72, TEXT_COLOR, 1, cv2.LINE_AA)
        y = 60
        for index, case in enumerate(self.cases):
            row_h = 60
            row_rect = (12, y, 396, row_h)
            color = ROW_SELECTED if index == self.case_index else ROW_BG
            cv2.rectangle(left, (row_rect[0], row_rect[1]), (row_rect[0] + row_rect[2], row_rect[1] + row_rect[3]), color, -1)
            cv2.rectangle(left, (row_rect[0], row_rect[1]), (row_rect[0] + row_rect[2], row_rect[1] + row_rect[3]), (90, 90, 90), 1)
            cv2.putText(left, f"{case.speaker}  {case.session}  F{case.frame_index_1based:04d}", (24, y + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.62, TEXT_COLOR, 1, cv2.LINE_AA)
            cv2.putText(left, f"raw {case.raw_subject}   {case.output_basename}", (24, y + 46), cv2.FONT_HERSHEY_SIMPLEX, 0.46, MUTED_TEXT, 1, cv2.LINE_AA)
            self.click_targets[f"speaker:{index}"] = (row_rect[0], row_rect[1], row_rect[2], row_rect[3])
            y += row_h + 8

        case = self._current_case()
        reference_name = reference_name_for_case(case)
        alias = self.mapping.vtln_session_map.get(reference_name, (case.speaker, case.session))
        cv2.putText(middle, "Resolved Source", (20, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.72, TEXT_COLOR, 1, cv2.LINE_AA)
        info_lines = [
            f"speaker: {case.speaker}",
            f"session/frame: {case.session} / {case.frame_index_1based:04d}",
            f"raw subject: {case.raw_subject}",
            f"reference: {reference_name}",
            f"alias: {alias[0]}/{alias[1]}",
            f"annotation status: {case.annotation_status or '-'}",
        ]
        y = 70
        for line in info_lines:
            cv2.putText(middle, line, (22, y), cv2.FONT_HERSHEY_SIMPLEX, 0.54, TEXT_COLOR, 1, cv2.LINE_AA)
            y += 28

        paths = step_file_paths(self.args.workspace_root / "placeholder")
        cv2.putText(middle, "Latest-state files per workspace:", (20, y + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.58, ACCENT, 1, cv2.LINE_AA)
        y += 46
        for name in ("source_annotation", "target_annotation", "transform_spec"):
            cv2.putText(middle, f"- {paths[name].name}", (28, y), cv2.FONT_HERSHEY_SIMPLEX, 0.48, MUTED_TEXT, 1, cv2.LINE_AA)
            y += 22

        cv2.putText(right, "Target", (20, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.72, TEXT_COLOR, 1, cv2.LINE_AA)
        target_mode_text = "nnUNet" if self.target_type == "nnunet" else "VTLN"
        y = self._draw_field(right, label="Target Type", value=target_mode_text, x0=20, y0=60, w=470, h=58, field_name="target_type")
        if self.target_type == "nnunet":
            y = self._draw_field(right, label="Target Case", value=self.nnunet_case, x0=20, y0=y, w=470, h=58, field_name="nnunet_case")
            y = self._draw_field(right, label="Target Frame", value=self.nnunet_frame, x0=20, y0=y, w=470, h=58, field_name="nnunet_frame")
        else:
            y = self._draw_field(right, label="VTLN Reference", value=self.vtln_reference, x0=20, y0=y, w=470, h=58, field_name="vtln_reference")
            y = self._draw_field(right, label="VTLN Dir", value=self.vtln_dir, x0=20, y0=y, w=470, h=72, field_name="vtln_dir")

        cv2.putText(right, "Focus:", (22, y + 26), cv2.FONT_HERSHEY_SIMPLEX, 0.54, ACCENT, 1, cv2.LINE_AA)
        cv2.putText(right, self.focus, (90, y + 26), cv2.FONT_HERSHEY_SIMPLEX, 0.54, TEXT_COLOR, 1, cv2.LINE_AA)
        cv2.putText(
            right,
            f"cache mode: {self.args.cache_mode}",
            (22, y + 52),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            ACCENT,
            1,
            cv2.LINE_AA,
        )
        mode_note = (
            "prewarm during step 0"
            if self.args.cache_mode == "startup"
            else "load/cache on first open"
        )
        cv2.putText(
            right,
            mode_note,
            (22, y + 76),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.44,
            MUTED_TEXT,
            1,
            cv2.LINE_AA,
        )
        if self.cache_counts is not None:
            source_count, target_count = self.cache_counts()
            cv2.putText(
                right,
                f"cache state: source={source_count} target={target_count}",
                (22, y + 102),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.46,
                MUTED_TEXT,
                1,
                cv2.LINE_AA,
            )
        return canvas

    def _on_mouse(self, event: int, x: int, y: int, _flags: int, _param) -> None:
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        for key, rect in self.click_targets.items():
            x0, y0, w, h = rect
            if x0 <= x < x0 + w and y0 <= y < y0 + h:
                if key.startswith("speaker:"):
                    self.case_index = int(key.split(":")[1])
                    self.focus = "speaker"
                    self.vtln_reference = reference_name_for_case(self._current_case())
                else:
                    self.focus = key
                return

    def _build_selection(self) -> WorkspaceSelection:
        target = TargetSelection(
            target_type=self.target_type,
            nnunet_case=self.nnunet_case.strip() or self.args.default_target_case,
            nnunet_frame=max(1, int(self.nnunet_frame or self.args.default_target_frame)),
            vtln_reference=(self.vtln_reference.strip() or reference_name_for_case(self._current_case())),
            vtln_dir=Path(self.vtln_dir.strip() or str(self.args.vtln_dir)),
        )
        return build_workspace_selection(
            case=self._current_case(),
            artspeech_root=self.args.artspeech_root,
            target=target,
            workspace_root=self.args.workspace_root,
            mapping=self.mapping,
        )

    def run(self) -> WorkspaceSelection | None:
        canvas = self._render()
        cv2.namedWindow(WINDOW_NAME_STEP0, cv2.WINDOW_NORMAL | getattr(cv2, "WINDOW_KEEPRATIO", 0))
        if self.args.window_mode == "fullscreen":
            cv2.setWindowProperty(WINDOW_NAME_STEP0, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        else:
            width, height = scaled_window_size(canvas.shape[1], canvas.shape[0], margin=WINDOW_MARGIN)
            cv2.resizeWindow(WINDOW_NAME_STEP0, width, height)
        cv2.setMouseCallback(WINDOW_NAME_STEP0, self._on_mouse)
        try:
            while True:
                canvas = self._render()
                cv2.imshow(WINDOW_NAME_STEP0, canvas)
                key = cv2.waitKeyEx(20)
                if key == -1:
                    continue
                if key in (ord("x"), ord("X"), 27):
                    return None
                if key == KEY_ENTER:
                    selection = self._build_selection()
                    save_json(step_file_paths(selection.workspace_dir)["selection"], workspace_selection_to_payload(selection))
                    return selection
                if key == KEY_TAB:
                    self._focus_next()
                    continue
                if self.focus == "speaker":
                    if key == KEY_UP:
                        self._cycle_case(-1)
                        continue
                    if key == KEY_DOWN:
                        self._cycle_case(1)
                        continue
                if self.focus == "target_type" and key in (KEY_LEFT, KEY_RIGHT, ord(" ")):
                    self._toggle_target_type()
                    continue
                before = (
                    self.nnunet_case,
                    self.nnunet_frame,
                    self.vtln_reference,
                    self.vtln_dir,
                )
                self._handle_text_input(key)
                after = (
                    self.nnunet_case,
                    self.nnunet_frame,
                    self.vtln_reference,
                    self.vtln_dir,
                )
                if after != before:
                    self._queue_prewarm()
        finally:
            cv2.destroyWindow(WINDOW_NAME_STEP0)


class Cv2ContourEditorWindow:
    def __init__(
        self,
        *,
        window_name: str,
        image: np.ndarray,
        contours: dict[str, np.ndarray],
        save_path: Path,
        metadata: dict[str, object],
        title_lines: list[str],
        info_lines: list[str],
        allow_back: bool,
        args: argparse.Namespace,
    ) -> None:
        self.window_name = window_name
        self.image = np.asarray(image, dtype=np.uint8)
        self.save_path = save_path
        self.metadata = metadata
        self.title_lines = title_lines
        self.info_lines = info_lines
        self.allow_back = allow_back
        self.args = args
        self.saved_payload = load_annotation_state_if_available(save_path)
        starting_contours = self.saved_payload["contours"] if self.saved_payload is not None else contours
        self.original_contours = {name: np.asarray(points, dtype=float).copy() for name, points in starting_contours.items()}
        self.original_point_counts = {name: int(len(points)) for name, points in self.original_contours.items()}
        self.current_contours = {name: np.asarray(points, dtype=float).copy() for name, points in starting_contours.items()}
        self.density_mode = "dense"
        self.isolate_mode = False
        self.isolated_contour: str | None = None
        self.handle_points = self._build_handle_points(self.current_contours)
        self.active_handle: tuple[str, int] | None = None
        self.last_mouse_world: tuple[float, float] | None = None
        self.dragging_pan = False
        self.pan_anchor: tuple[int, int] | None = None
        self.view = ensure_view_state(None, tuple(self.image.shape[:2]))
        self.last_mapping: PanelMapping | None = None
        if self.allow_back:
            self.status_message = (
                "S = save only | N = save and go to the next step | "
                "G = go to the next step without saving | "
                "B = save and go back | R = reset | wheel = zoom | right-drag image = pan | X = exit"
            )
        else:
            self.status_message = (
                "S = save only | N = save and go to the next step | "
                "G = go to the next step without saving | "
                "R = reset | wheel = zoom | right-drag image = pan | X = exit"
            )

    def _base_handle_count(self, name: str, n_points: int) -> int:
        base = LONG_CONTOUR_HANDLE_COUNTS.get(name, min(n_points, 28))
        if self.density_mode == "dense":
            return max(4, min(n_points, base))
        if self.density_mode == "medium":
            return max(4, min(n_points, max(8, base // 2)))
        return max(4, min(n_points, max(6, base // 3)))

    def _build_handle_points(self, contours: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        handles: dict[str, np.ndarray] = {}
        for name, points in contours.items():
            pts = np.asarray(points, dtype=float)
            n_handles = self._base_handle_count(name, len(pts))
            if len(pts) <= n_handles:
                handles[name] = pts.copy()
                continue
            indices = np.linspace(0, len(pts) - 1, num=n_handles)
            indices = np.unique(np.round(indices).astype(int))
            handles[name] = pts[indices].copy()
        return handles

    @staticmethod
    def _resample_polyline(points: np.ndarray, n_samples: int) -> np.ndarray:
        pts = np.asarray(points, dtype=float)
        if len(pts) < 2:
            return np.repeat(pts[:1], n_samples, axis=0)
        segment = np.linalg.norm(np.diff(pts, axis=0), axis=1)
        keep = np.r_[True, segment > 1e-8]
        pts = pts[keep]
        if len(pts) < 2:
            return np.repeat(pts[:1], n_samples, axis=0)
        arc = np.cumsum(np.r_[0.0, np.linalg.norm(np.diff(pts, axis=0), axis=1)])
        arc /= max(float(arc[-1]), 1e-8)
        sample_t = np.linspace(0.0, 1.0, n_samples)
        return np.column_stack([np.interp(sample_t, arc, pts[:, 0]), np.interp(sample_t, arc, pts[:, 1])])

    def _rebuild_contours_from_handles(self) -> dict[str, np.ndarray]:
        contours: dict[str, np.ndarray] = {}
        for name, handles in self.handle_points.items():
            pts = np.asarray(handles, dtype=float)
            n_points = self.original_point_counts[name]
            if n_points > len(pts):
                contours[name] = self._resample_polyline(pts, n_points)
            else:
                contours[name] = pts.copy()
        return contours

    def _visible_contour_names(self) -> list[str]:
        if self.isolate_mode and self.isolated_contour is not None:
            return [self.isolated_contour]
        return sorted(self.current_contours.keys())

    def _nearest_handle(self, x: float, y: float, threshold_px: float = 10.0) -> tuple[str, int] | None:
        if self.last_mapping is None:
            return None
        threshold_world = threshold_px / max(self.last_mapping.scale, 1e-6)
        best: tuple[str, int] | None = None
        best_distance = float("inf")
        probe = np.array([x, y], dtype=float)
        for name in self._visible_contour_names():
            points = self.handle_points[name]
            distances = np.linalg.norm(points - probe, axis=1)
            if len(distances) == 0:
                continue
            index = int(np.argmin(distances))
            distance = float(distances[index])
            if distance < best_distance:
                best_distance = distance
                best = (name, index)
        if best is None or best_distance > threshold_world:
            return None
        return best

    def _cycle_density(self, delta: int) -> None:
        options = ["dense", "medium", "sparse"]
        current_index = options.index(self.density_mode)
        self.current_contours = self._rebuild_contours_from_handles()
        self.density_mode = options[(current_index + delta) % len(options)]
        self.handle_points = self._build_handle_points(self.current_contours)
        self.status_message = f"Handle density: {self.density_mode}"

    def _toggle_isolate(self) -> None:
        if self.active_handle is not None:
            self.isolated_contour = self.active_handle[0]
        elif self.isolated_contour is None:
            self.isolated_contour = sorted(self.current_contours.keys())[0]
        self.isolate_mode = not self.isolate_mode
        self.status_message = f"Isolate mode: {'on' if self.isolate_mode else 'off'}"

    def _render_main_panel(self) -> tuple[np.ndarray, PanelMapping]:
        panel, mapping = render_panel_base(self.image, EDITOR_PANEL_W, EDITOR_PANEL_H, self.view)
        draw_header_text(panel, self.title_lines)
        visible = self._visible_contour_names()
        for name in visible:
            draw_polyline(
                panel,
                mapping,
                self.current_contours[name],
                hex_to_bgr(CONTOUR_COLORS.get(name, "#00b4d8")),
                dashed=name in {"pharynx", "soft-palate-midline"},
                thickness=2,
            )
        for name in visible:
            pts = self.handle_points[name]
            scaled = np.round(world_to_screen(mapping, pts)).astype(int)
            for index, point in enumerate(scaled):
                is_active = self.active_handle == (name, index)
                radius = 6 if is_active else 4
                cv2.circle(panel, tuple(point), radius + 2, (255, 255, 255), -1, cv2.LINE_AA)
                cv2.circle(panel, tuple(point), radius, hex_to_bgr(CONTOUR_COLORS.get(name, "#ffffff")), -1, cv2.LINE_AA)
                if is_active:
                    cv2.putText(panel, f"{name}[{index}]", (point[0] + 6, point[1] - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.44, ACCENT, 1, cv2.LINE_AA)
        return panel, mapping

    def _render_loupe(self) -> np.ndarray:
        anchor = None
        if self.active_handle is not None:
            name, index = self.active_handle
            anchor = self.handle_points[name][index]
        elif self.last_mouse_world is not None:
            anchor = np.asarray(self.last_mouse_world, dtype=float)
        canvas = np.full((EDITOR_LOUPE, EDITOR_LOUPE, 3), PANEL_BG, dtype=np.uint8)
        if anchor is None:
            draw_header_text(canvas, ["Loupe", "Click or hover a point"])
            return canvas
        zoom = 7.0
        view = ViewState(zoom=zoom, center_x=float(anchor[0]), center_y=float(anchor[1]))
        panel, mapping = render_panel_base(self.image, EDITOR_LOUPE, EDITOR_LOUPE, view)
        draw_header_text(panel, ["Loupe", f"x{zoom:.1f}"])
        visible = self._visible_contour_names()
        for name in visible:
            draw_polyline(
                panel,
                mapping,
                self.current_contours[name],
                hex_to_bgr(CONTOUR_COLORS.get(name, "#00b4d8")),
                dashed=name in {"pharynx", "soft-palate-midline"},
                thickness=2,
            )
            pts = self.handle_points[name]
            scaled = np.round(world_to_screen(mapping, pts)).astype(int)
            for index, point in enumerate(scaled):
                cv2.circle(panel, tuple(point), 4, (255, 255, 255), -1, cv2.LINE_AA)
                cv2.circle(panel, tuple(point), 3, hex_to_bgr(CONTOUR_COLORS.get(name, "#ffffff")), -1, cv2.LINE_AA)
                if self.active_handle is not None and self.active_handle[0] == name and abs(index - self.active_handle[1]) <= 2:
                    cv2.putText(panel, str(index), (point[0] + 4, point[1] - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.34, ACCENT, 1, cv2.LINE_AA)
        return panel

    def _compose_canvas(self) -> np.ndarray:
        canvas = np.full((EDITOR_PANEL_H + EDITOR_FOOTER_H, EDITOR_PANEL_W + EDITOR_SIDEBAR_W, 3), PANEL_BG, dtype=np.uint8)
        main_panel, mapping = self._render_main_panel()
        self.last_mapping = mapping
        canvas[:EDITOR_PANEL_H, :EDITOR_PANEL_W] = main_panel

        sidebar = np.full((EDITOR_PANEL_H, EDITOR_SIDEBAR_W, 3), SIDEBAR_BG, dtype=np.uint8)
        loupe = self._render_loupe()
        sidebar[16 : 16 + EDITOR_LOUPE, 40 : 40 + EDITOR_LOUPE] = loupe
        top = 280
        top = draw_sidebar_block(sidebar, "Info", self.info_lines, top)
        top = draw_sidebar_block(sidebar, "Mode", [f"density: {self.density_mode}", f"isolate: {self.isolate_mode}"], top, highlight=True)
        if self.saved_payload is not None:
            top = draw_sidebar_block(sidebar, "Saved", ["Loaded latest saved annotation"], top)
        canvas[:EDITOR_PANEL_H, EDITOR_PANEL_W:] = sidebar

        footer_y = EDITOR_PANEL_H
        canvas[footer_y:, :] = FOOTER_BG
        help_line = (
            "Wheel zoom | Right-drag image to pan | Left drag point | "
            "S save only | N save and go to next step | G go to next step without saving | "
            "R reset | I isolate | [ ] density | X exit"
        )
        if self.allow_back:
            help_line = (
                "Wheel zoom | Right-drag image to pan | Left drag point | "
                "S save only | B save and go back | N save and go to next step | "
                "G go to next step without saving | "
                "R reset | I isolate | [ ] density | X exit"
            )
        cv2.putText(canvas, help_line, (12, footer_y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.52, TEXT_COLOR, 1, cv2.LINE_AA)
        cv2.putText(canvas, self.status_message[:180], (12, footer_y + 60), cv2.FONT_HERSHEY_SIMPLEX, 0.50, ACCENT, 1, cv2.LINE_AA)
        return canvas

    def _pan_by(self, dx_screen: int, dy_screen: int) -> None:
        if self.last_mapping is None:
            return
        self.view.center_x -= dx_screen / max(self.last_mapping.scale, 1e-6)
        self.view.center_y -= dy_screen / max(self.last_mapping.scale, 1e-6)
        self.view = ensure_view_state(self.view, tuple(self.image.shape[:2]))

    def _zoom(self, factor: float, anchor_world: tuple[float, float] | None = None) -> None:
        if anchor_world is None:
            anchor_world = (self.view.center_x, self.view.center_y)
        self.view.zoom = float(np.clip(self.view.zoom * factor, 1.0, 12.0))
        self.view.center_x = float(anchor_world[0])
        self.view.center_y = float(anchor_world[1])
        self.view = ensure_view_state(self.view, tuple(self.image.shape[:2]))

    def _on_mouse(self, event: int, x: int, y: int, flags: int, _param) -> None:
        if self.last_mapping is None:
            return
        world_xy = screen_to_world(self.last_mapping, x, y)
        if world_xy is not None:
            self.last_mouse_world = world_xy
        if event == cv2.EVENT_MOUSEWHEEL and world_xy is not None:
            self._zoom(1.18 if flags > 0 else 1.0 / 1.18, world_xy)
            return
        if event == cv2.EVENT_RBUTTONDOWN and world_xy is not None:
            self.dragging_pan = True
            self.pan_anchor = (x, y)
            return
        if event == cv2.EVENT_MOUSEMOVE and self.dragging_pan and self.pan_anchor is not None:
            self._pan_by(x - self.pan_anchor[0], y - self.pan_anchor[1])
            self.pan_anchor = (x, y)
            return
        if event == cv2.EVENT_RBUTTONUP:
            self.dragging_pan = False
            self.pan_anchor = None
            return
        if event == cv2.EVENT_LBUTTONDOWN and world_xy is not None:
            nearest = self._nearest_handle(*world_xy)
            if nearest is None:
                return
            self.active_handle = nearest
            self.isolated_contour = nearest[0]
            return
        if event == cv2.EVENT_MOUSEMOVE and self.active_handle is not None and world_xy is not None:
            contour_name, point_index = self.active_handle
            updated = clamp_point(np.array(world_xy, dtype=float), self.image.shape[1], self.image.shape[0])
            self.handle_points[contour_name][point_index] = updated
            self.current_contours = self._rebuild_contours_from_handles()
            return
        if event == cv2.EVENT_LBUTTONUP and self.active_handle is not None:
            if world_xy is not None:
                contour_name, point_index = self.active_handle
                updated = clamp_point(np.array(world_xy, dtype=float), self.image.shape[1], self.image.shape[0])
                self.handle_points[contour_name][point_index] = updated
                self.current_contours = self._rebuild_contours_from_handles()
            self.active_handle = None

    def _save(self) -> None:
        save_annotation_state(self.save_path, self.metadata, self.current_contours)
        self.status_message = f"Saved {self.save_path.name}"

    def run(self) -> tuple[str, dict[str, np.ndarray]]:
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL | getattr(cv2, "WINDOW_KEEPRATIO", 0))
        cv2.setMouseCallback(self.window_name, self._on_mouse)
        canvas = self._compose_canvas()
        if self.args.window_mode == "fullscreen":
            cv2.setWindowProperty(self.window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        else:
            width, height = scaled_window_size(canvas.shape[1], canvas.shape[0], margin=WINDOW_MARGIN)
            cv2.resizeWindow(self.window_name, width, height)
        try:
            while True:
                canvas = self._compose_canvas()
                cv2.imshow(self.window_name, canvas)
                key = cv2.waitKeyEx(20)
                if key == -1:
                    continue
                if key in (ord("x"), ord("X"), 27):
                    return "exit_all", self.current_contours
                if key == ord("s"):
                    self._save()
                elif key == ord("n"):
                    self._save()
                    return "next", self.current_contours
                elif key in (ord("g"), ord("G")):
                    self.status_message = "Continuing to the next step without saving to disk."
                    return "next_no_save", self.current_contours
                elif key == ord("b") and self.allow_back:
                    self._save()
                    return "back", self.current_contours
                elif key == ord("r"):
                    self.current_contours = {name: points.copy() for name, points in self.original_contours.items()}
                    self.handle_points = self._build_handle_points(self.current_contours)
                    self.status_message = "Reset to latest saved/starting contours."
                elif key == ord("i"):
                    self._toggle_isolate()
                elif key == ord("["):
                    self._cycle_density(-1)
                elif key == ord("]"):
                    self._cycle_density(1)
                elif key in (ord("+"), ord("=")):
                    self._zoom(1.15, self.last_mouse_world)
                elif key in (ord("-"), ord("_")):
                    self._zoom(1.0 / 1.15, self.last_mouse_world)
        finally:
            cv2.destroyWindow(self.window_name)


class Cv2TransformReviewWindow:
    def __init__(
        self,
        *,
        selection: WorkspaceSelection,
        source_image: np.ndarray,
        source_contours: dict[str, np.ndarray],
        target_image: np.ndarray,
        target_contours: dict[str, np.ndarray],
        args: argparse.Namespace,
    ) -> None:
        self.selection = selection
        self.source_image = np.asarray(source_image, dtype=np.uint8)
        self.target_image = np.asarray(target_image, dtype=np.uint8)
        self.source_contours = {name: np.asarray(points, dtype=float).copy() for name, points in source_contours.items()}
        self.target_contours = {name: np.asarray(points, dtype=float).copy() for name, points in target_contours.items()}
        self.args = args
        self.paths = step_file_paths(selection.workspace_dir)

        self.source_landmark_overrides: dict[str, np.ndarray | None] = {}
        self.target_landmark_overrides: dict[str, np.ndarray | None] = {}
        if self.paths["landmark_overrides"].is_file():
            try:
                overrides_payload = json.loads(self.paths["landmark_overrides"].read_text(encoding="utf-8"))
                for name, value in overrides_payload.get("source_landmark_overrides", {}).items():
                    self.source_landmark_overrides[name] = None if value is None else np.asarray(value, dtype=float)
                for name, value in overrides_payload.get("target_landmark_overrides", {}).items():
                    self.target_landmark_overrides[name] = None if value is None else np.asarray(value, dtype=float)
            except Exception:
                pass

        self.bundle = self._recompute_bundle()
        self.pane_views = {
            "source_native": ensure_view_state(None, tuple(self.source_image.shape[:2])),
            "source_affine": ensure_view_state(None, tuple(self.target_image.shape[:2])),
            "source_final": ensure_view_state(None, tuple(self.target_image.shape[:2])),
            "target_native": ensure_view_state(None, tuple(self.target_image.shape[:2])),
            "target_affine": ensure_view_state(None, tuple(self.target_image.shape[:2])),
            "target_final": ensure_view_state(None, tuple(self.target_image.shape[:2])),
        }
        self.pane_geometries = {
            "source_native": PaneGeometry(0, 0, REVIEW_PANE_W, REVIEW_PANE_H),
            "source_affine": PaneGeometry(REVIEW_PANE_W + PANE_GAP, 0, REVIEW_PANE_W, REVIEW_PANE_H),
            "source_final": PaneGeometry((REVIEW_PANE_W + PANE_GAP) * 2, 0, REVIEW_PANE_W, REVIEW_PANE_H),
            "target_native": PaneGeometry(0, REVIEW_PANE_H + PANE_GAP, REVIEW_PANE_W, REVIEW_PANE_H),
            "target_affine": PaneGeometry(REVIEW_PANE_W + PANE_GAP, REVIEW_PANE_H + PANE_GAP, REVIEW_PANE_W, REVIEW_PANE_H),
            "target_final": PaneGeometry((REVIEW_PANE_W + PANE_GAP) * 2, REVIEW_PANE_H + PANE_GAP, REVIEW_PANE_W, REVIEW_PANE_H),
        }
        self.last_panel_mappings: dict[str, PanelMapping] = {}
        self.active_landmark: tuple[str, str] | None = None
        self.active_landmark_backup: np.ndarray | None = None
        self.dragging_pan = False
        self.pan_anchor: tuple[str, int, int] | None = None
        self.status_message = (
            "Drag source/target landmarks in the native panels. Right-drag image to pan. "
            "S = save only | B = save and go back to step 1 | "
            "0 = save and return to step 0 | N = save and go to step 3 | X = exit"
        )

    def _warp_source_image(self, *, stage: str, spec: dict[str, object]) -> np.ndarray:
        target_shape = tuple(int(value) for value in spec["target_shape"])
        inverse_mapping = lambda pts: apply_transform_from_spec(spec, pts, direction="inverse", stage=stage)
        source_x, source_y, valid_mask = precompute_inverse_warp(target_shape, inverse_mapping, self.source_image.shape[:2])
        warped, _ = warp_array_with_precomputed_inverse_warp(self.source_image, source_x, source_y, valid_mask)
        return warped

    def _recompute_bundle(self) -> dict[str, object]:
        bundle = build_transform_bundle(
            source_image=self.source_image,
            source_contours=self.source_contours,
            target_image=self.target_image,
            target_contours=self.target_contours,
            source_frame_number=self.selection.case.frame_index_1based,
            target_frame_number=self.selection.target.nnunet_frame if self.selection.target.target_type == "nnunet" else 0,
            source_landmark_overrides=self.source_landmark_overrides,
            target_landmark_overrides=self.target_landmark_overrides,
        )
        bundle["affine_warped_image"] = self._warp_source_image(stage="affine", spec=bundle["transform_spec"])
        bundle["final_warped_image"] = self._warp_source_image(stage="final", spec=bundle["transform_spec"])
        return bundle

    def _step_sets(self) -> tuple[set[str], set[str]]:
        step1 = set(self.bundle["transform_spec"]["step1_labels"])
        step2 = set(self.bundle["transform_spec"]["step2_labels"])
        return step1, step2 - step1

    def _pane_name_at(self, x: int, y: int) -> str | None:
        for name, geometry in self.pane_geometries.items():
            if geometry.x0 <= x < geometry.x0 + geometry.width and geometry.y0 <= y < geometry.y0 + geometry.height:
                return name
        return None

    def _landmark_map_for_role(self, role: str) -> dict[str, np.ndarray | None]:
        return self.bundle["source_landmarks"] if role == "source" else self.bundle["target_landmarks"]

    def _nearest_landmark(self, role: str, world_xy: tuple[float, float], pane_name: str) -> str | None:
        mapping = self.last_panel_mappings.get(pane_name)
        if mapping is None:
            return None
        threshold_world = 12.0 / max(mapping.scale, 1e-6)
        probe = np.asarray(world_xy, dtype=float)
        best_name = None
        best_distance = float("inf")
        for name, point in self._landmark_map_for_role(role).items():
            if point is None:
                continue
            point_arr = np.asarray(point, dtype=float)
            if point_arr.ndim != 1 or point_arr.shape[0] != 2:
                continue
            distance = float(np.linalg.norm(point_arr - probe))
            if distance < best_distance:
                best_distance = distance
                best_name = name
        if best_distance > threshold_world:
            return None
        return best_name

    def _update_landmark(self, role: str, name: str, world_xy: tuple[float, float]) -> None:
        image_shape = self.source_image.shape[:2] if role == "source" else self.target_image.shape[:2]
        clamped = clamp_point(np.asarray(world_xy, dtype=float), image_shape[1], image_shape[0])
        if role == "source":
            self.source_landmark_overrides[name] = clamped
        else:
            self.target_landmark_overrides[name] = clamped

    def _save(self) -> None:
        save_json(
            self.paths["landmark_overrides"],
            {
                "source_landmark_overrides": serialize_points_map(self.source_landmark_overrides),
                "target_landmark_overrides": serialize_points_map(self.target_landmark_overrides),
            },
        )
        save_json(self.paths["transform_spec"], self.bundle["transform_spec"])
        save_json(self.paths["transform_review"], self.bundle["transform_review"])
        overview = self._compose_canvas()
        save_canvas_preview(self.paths["overview_preview"], overview)
        save_canvas_preview(self.paths["native_preview"], overview[:, :REVIEW_PANE_W].copy())
        save_canvas_preview(
            self.paths["affine_preview"],
            overview[:, REVIEW_PANE_W + PANE_GAP : REVIEW_PANE_W * 2 + PANE_GAP].copy(),
        )
        save_canvas_preview(
            self.paths["final_preview"],
            overview[:, REVIEW_PANE_W * 2 + PANE_GAP * 2 :].copy(),
        )
        self.status_message = "Saved transform_spec.latest.json and transform_review.latest.json"

    def _render_named_panel(self, pane_name: str) -> np.ndarray:
        geometry = self.pane_geometries[pane_name]
        if pane_name == "source_native":
            base_image = self.source_image
        elif pane_name == "target_native":
            base_image = self.target_image
        elif pane_name == "source_affine":
            base_image = self.bundle["affine_warped_image"]
        elif pane_name == "source_final":
            base_image = self.bundle["final_warped_image"]
        else:
            base_image = self.target_image
        panel, mapping = render_panel_base(base_image, geometry.width, geometry.height, self.pane_views[pane_name])
        self.last_panel_mappings[pane_name] = mapping
        step1, step2_extra = self._step_sets()

        if pane_name == "source_native":
            draw_contours(panel, mapping, self.source_contours)
            draw_grid(panel, mapping, self.bundle["source_grid"].horiz_lines, self.bundle["source_grid"].vert_lines, color_h=(255, 190, 11), color_v=(60, 134, 255))
            draw_points(
                panel,
                mapping,
                self.bundle["source_landmarks"],
                step1_labels=step1,
                step2_extra_labels=step2_extra,
                highlight=self.active_landmark[1] if self.active_landmark and self.active_landmark[0] == "source" else None,
            )
            draw_header_text(panel, ["Source native + editable landmarks", f"{self.selection.case.speaker}/{self.selection.case.session}"])
        elif pane_name == "target_native":
            draw_contours(panel, mapping, self.target_contours)
            draw_grid(panel, mapping, self.bundle["target_grid"].horiz_lines, self.bundle["target_grid"].vert_lines, color_h=(255, 190, 11), color_v=(60, 134, 255))
            draw_points(
                panel,
                mapping,
                self.bundle["target_landmarks"],
                step1_labels=step1,
                step2_extra_labels=step2_extra,
                highlight=self.active_landmark[1] if self.active_landmark and self.active_landmark[0] == "target" else None,
            )
            draw_header_text(panel, ["Target native + editable landmarks", self.selection.target.target_type])
        elif pane_name == "source_affine":
            draw_grid(panel, mapping, self.bundle["step1_horiz"], self.bundle["step1_vert"], color_h=(255, 190, 11), color_v=(60, 134, 255))
            draw_contours(panel, mapping, self.bundle["affine_contours"])
            draw_points(panel, mapping, self.bundle["step1_landmarks"], step1_labels=step1, step2_extra_labels=step2_extra, draw_labels=False)
            draw_header_text(panel, ["Affine stage", "Warped source in target space"])
        elif pane_name == "source_final":
            draw_grid(panel, mapping, self.bundle["final_horiz"], self.bundle["final_vert"], color_h=(255, 190, 11), color_v=(60, 134, 255))
            draw_contours(panel, mapping, self.bundle["final_contours"])
            draw_points(panel, mapping, self.bundle["final_landmarks"], step1_labels=step1, step2_extra_labels=step2_extra, draw_labels=False)
            draw_header_text(panel, ["Affine + TPS stage", "Warped source in target space"])
        elif pane_name == "target_affine":
            draw_contours(panel, mapping, self.target_contours, dim=True)
            draw_grid(panel, mapping, self.bundle["target_grid"].horiz_lines, self.bundle["target_grid"].vert_lines, color_h=(100, 100, 100), color_v=(80, 80, 80))
            draw_grid(panel, mapping, self.bundle["step1_horiz"], self.bundle["step1_vert"], color_h=(255, 190, 11), color_v=(60, 134, 255))
            draw_points(panel, mapping, self.bundle["target_landmarks"], step1_labels=set(), step2_extra_labels=set(), draw_labels=False)
            draw_points(panel, mapping, self.bundle["step1_landmarks"], step1_labels=step1, step2_extra_labels=step2_extra)
            draw_header_text(panel, ["Target space overlay", "Affine result"])
        else:
            draw_contours(panel, mapping, self.target_contours, dim=True)
            draw_grid(panel, mapping, self.bundle["target_grid"].horiz_lines, self.bundle["target_grid"].vert_lines, color_h=(100, 100, 100), color_v=(80, 80, 80))
            draw_grid(panel, mapping, self.bundle["final_horiz"], self.bundle["final_vert"], color_h=(255, 190, 11), color_v=(60, 134, 255))
            draw_points(panel, mapping, self.bundle["target_landmarks"], step1_labels=set(), step2_extra_labels=set(), draw_labels=False)
            draw_points(panel, mapping, self.bundle["final_landmarks"], step1_labels=step1, step2_extra_labels=step2_extra)
            draw_header_text(panel, ["Target space overlay", "Affine + TPS result"])
        return panel

    def _compose_canvas(self) -> np.ndarray:
        canvas_h = REVIEW_PANE_H * 2 + PANE_GAP + REVIEW_FOOTER_H
        canvas_w = REVIEW_PANE_W * 3 + PANE_GAP * 2
        canvas = np.full((canvas_h, canvas_w, 3), PANEL_BG, dtype=np.uint8)
        for name, geometry in self.pane_geometries.items():
            panel = self._render_named_panel(name)
            canvas[geometry.y0 : geometry.y0 + geometry.height, geometry.x0 : geometry.x0 + geometry.width] = panel
        footer_y = REVIEW_PANE_H * 2 + PANE_GAP
        canvas[footer_y:, :] = FOOTER_BG
        final_metrics = self.bundle["transform_review"]["metrics"]["final"]
        metric_line = (
            f"final horiz_axis_rms={final_metrics.get('horiz_axis_rms')} "
            f"vert_axis_rms={final_metrics.get('vert_axis_rms')} "
            f"spine_rms={final_metrics.get('spine_rms')}"
        )
        help_line = (
            "Drag only in native panels | wheel zoom | right-drag image to pan | "
            "S save only | B save and go back to step 1 | "
            "0 save and return to step 0 | N save and go to step 3 | X exit"
        )
        cv2.putText(canvas, help_line, (12, footer_y + 26), cv2.FONT_HERSHEY_SIMPLEX, 0.46, TEXT_COLOR, 1, cv2.LINE_AA)
        cv2.putText(canvas, metric_line[:180], (12, footer_y + 52), cv2.FONT_HERSHEY_SIMPLEX, 0.44, MUTED_TEXT, 1, cv2.LINE_AA)
        draw_step2_marker_legend(canvas, 12, footer_y + 78)
        affine_lines = format_label_list("Affine anchors", list(self.bundle["transform_spec"]["step1_labels"]))
        step1_set = set(self.bundle["transform_spec"]["step1_labels"])
        tps_extra_labels = [label for label in self.bundle["transform_spec"]["step2_labels"] if label not in step1_set]
        tps_lines = format_label_list("TPS extra controls", tps_extra_labels)
        label_lines = affine_lines + tps_lines
        start_x = 250
        for index, line in enumerate(label_lines):
            color = AFFINE_COLOR if index < len(affine_lines) else TPS_COLOR
            cv2.putText(canvas, line, (start_x, footer_y + 82 + 22 * index), cv2.FONT_HERSHEY_SIMPLEX, 0.44, color, 1, cv2.LINE_AA)
        cv2.putText(canvas, self.status_message[:180], (12, footer_y + REVIEW_FOOTER_H - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.48, ACCENT, 1, cv2.LINE_AA)
        return canvas

    def _on_mouse(self, event: int, x: int, y: int, flags: int, _param) -> None:
        pane_name = self._pane_name_at(x, y)
        if pane_name is None:
            return
        geometry = self.pane_geometries[pane_name]
        mapping = self.last_panel_mappings.get(pane_name)
        if mapping is None:
            return
        local_xy = screen_to_world(mapping, x - geometry.x0, y - geometry.y0)
        if event == cv2.EVENT_MOUSEWHEEL:
            if local_xy is None:
                return
            view = self.pane_views[pane_name]
            view.zoom = float(np.clip(view.zoom * (1.15 if flags > 0 else 1.0 / 1.15), 1.0, 12.0))
            view.center_x = float(local_xy[0])
            view.center_y = float(local_xy[1])
            shape = self.source_image.shape[:2] if pane_name == "source_native" else self.target_image.shape[:2]
            self.pane_views[pane_name] = ensure_view_state(view, shape)
            return
        if event == cv2.EVENT_RBUTTONDOWN and local_xy is not None:
            self.dragging_pan = True
            self.pan_anchor = (pane_name, x, y)
            return
        if event == cv2.EVENT_MOUSEMOVE and self.dragging_pan and self.pan_anchor is not None and self.pan_anchor[0] == pane_name:
            dx = x - self.pan_anchor[1]
            dy = y - self.pan_anchor[2]
            view = self.pane_views[pane_name]
            view.center_x -= dx / max(mapping.scale, 1e-6)
            view.center_y -= dy / max(mapping.scale, 1e-6)
            shape = self.source_image.shape[:2] if pane_name == "source_native" else self.target_image.shape[:2]
            self.pane_views[pane_name] = ensure_view_state(view, shape)
            self.pan_anchor = (pane_name, x, y)
            return
        if event == cv2.EVENT_RBUTTONUP:
            self.dragging_pan = False
            self.pan_anchor = None
            return
        if pane_name not in {"source_native", "target_native"}:
            return
        role = "source" if pane_name == "source_native" else "target"
        if event == cv2.EVENT_LBUTTONDOWN and local_xy is not None:
            landmark_name = self._nearest_landmark(role, local_xy, pane_name)
            if landmark_name is None:
                return
            self.active_landmark = (role, landmark_name)
            current_point = self._landmark_map_for_role(role).get(landmark_name)
            self.active_landmark_backup = None if current_point is None else np.asarray(current_point, dtype=float).copy()
            return
        if event == cv2.EVENT_MOUSEMOVE and self.active_landmark is not None and local_xy is not None:
            self._update_landmark(role, self.active_landmark[1], local_xy)
            try:
                self.bundle = self._recompute_bundle()
                self.status_message = "Updated transform preview."
            except Exception as exc:
                if role == "source":
                    if self.active_landmark_backup is None:
                        self.source_landmark_overrides.pop(self.active_landmark[1], None)
                    else:
                        self.source_landmark_overrides[self.active_landmark[1]] = self.active_landmark_backup.copy()
                else:
                    if self.active_landmark_backup is None:
                        self.target_landmark_overrides.pop(self.active_landmark[1], None)
                    else:
                        self.target_landmark_overrides[self.active_landmark[1]] = self.active_landmark_backup.copy()
                self.bundle = self._recompute_bundle()
                self.status_message = f"Invalid landmark edit reverted: {exc}"
            return
        if event == cv2.EVENT_LBUTTONUP and self.active_landmark is not None:
            self.active_landmark = None
            self.active_landmark_backup = None

    def run(self) -> str:
        cv2.namedWindow(WINDOW_NAME_STEP2, cv2.WINDOW_NORMAL | getattr(cv2, "WINDOW_KEEPRATIO", 0))
        cv2.setMouseCallback(WINDOW_NAME_STEP2, self._on_mouse)
        canvas = self._compose_canvas()
        if self.args.window_mode == "fullscreen":
            cv2.setWindowProperty(WINDOW_NAME_STEP2, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        else:
            width, height = scaled_window_size(canvas.shape[1], canvas.shape[0], margin=WINDOW_MARGIN)
            cv2.resizeWindow(WINDOW_NAME_STEP2, width, height)
        try:
            while True:
                canvas = self._compose_canvas()
                cv2.imshow(WINDOW_NAME_STEP2, canvas)
                key = cv2.waitKeyEx(20)
                if key == -1:
                    continue
                if key in (ord("x"), ord("X"), 27):
                    return "exit_all"
                if key == ord("s"):
                    self._save()
                elif key == ord("b"):
                    self._save()
                    return "back_step1"
                elif key == ord("0"):
                    self._save()
                    return "step0"
                elif key == ord("n"):
                    self._save()
                    return "step3"
        finally:
            cv2.destroyWindow(WINDOW_NAME_STEP2)


class Cv2ExportStepWindow:
    def __init__(self, *, selection: WorkspaceSelection, args: argparse.Namespace) -> None:
        self.selection = selection
        self.args = args
        self.paths = step_file_paths(selection.workspace_dir)
        self.export_dir = selection.workspace_dir / "background_export"

    def _render(self) -> np.ndarray:
        canvas = np.full((420, 980, 3), PANEL_BG, dtype=np.uint8)
        cv2.putText(canvas, "Step 3 - Background export", (20, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.84, TEXT_COLOR, 2, cv2.LINE_AA)
        lines = [
            f"workspace: {self.selection.workspace_id}",
            f"source: {self.selection.case.speaker}/{self.selection.case.session} frame {self.selection.case.frame_index_1based:04d}",
            f"target type: {self.selection.target.target_type}",
            f"transform spec: {self.paths['transform_spec'].name}",
            f"workers/prefetch: {self.args.render_workers}/{self.args.render_prefetch}",
            f"output dir: {self.export_dir}",
            "",
            "L = launch background export and return to step 0",
            "B = go back to step 2",
            "After launch, the app returns to step 0 by default.",
        ]
        y = 86
        for line in lines:
            cv2.putText(canvas, line, (24, y), cv2.FONT_HERSHEY_SIMPLEX, 0.58, TEXT_COLOR if line else MUTED_TEXT, 1, cv2.LINE_AA)
            y += 28
        return canvas

    def _launch_export(self) -> None:
        self.export_dir.mkdir(parents=True, exist_ok=True)
        module_name = "grid_transform.apps.export_saved_workspace_video_threaded"
        command = [
            sys.executable,
            "-m",
            module_name,
            "--workspace-dir",
            str(self.selection.workspace_dir),
            "--workers",
            str(int(self.args.render_workers)),
            "--prefetch",
            str(int(self.args.render_prefetch)),
        ]
        log_path = self.export_dir / "background_render.log"
        job_path = self.export_dir / "background_render_job.json"
        popen_kwargs: dict[str, object] = {
            "cwd": str(PROJECT_DIR),
            "stdin": subprocess.DEVNULL,
            "stdout": log_path.open("ab"),
            "stderr": subprocess.STDOUT,
        }
        if os.name == "nt":
            creationflags = 0
            creationflags |= getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
            creationflags |= getattr(subprocess, "DETACHED_PROCESS", 0)
            creationflags |= getattr(subprocess, "CREATE_NO_WINDOW", 0)
            popen_kwargs["creationflags"] = creationflags
        else:
            popen_kwargs["start_new_session"] = True
        process = subprocess.Popen(command, **popen_kwargs)
        if hasattr(popen_kwargs["stdout"], "close"):
            popen_kwargs["stdout"].close()
        job_payload = {
            "status": "started",
            "pid": int(process.pid),
            "module": module_name,
            "command": command,
            "log_path": str(log_path),
            "workspace_dir": str(self.selection.workspace_dir),
            "render_workers": int(self.args.render_workers),
            "render_prefetch": int(self.args.render_prefetch),
        }
        save_json(job_path, job_payload)

    def run(self) -> str:
        canvas = self._render()
        cv2.namedWindow(WINDOW_NAME_STEP3, cv2.WINDOW_NORMAL | getattr(cv2, "WINDOW_KEEPRATIO", 0))
        if self.args.window_mode == "fullscreen":
            cv2.setWindowProperty(WINDOW_NAME_STEP3, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        else:
            width, height = scaled_window_size(canvas.shape[1], canvas.shape[0], margin=WINDOW_MARGIN)
            cv2.resizeWindow(WINDOW_NAME_STEP3, width, height)
        try:
            while True:
                canvas = self._render()
                cv2.imshow(WINDOW_NAME_STEP3, canvas)
                key = cv2.waitKeyEx(20)
                if key == -1:
                    continue
                if key in (ord("x"), ord("X"), 27):
                    return "exit_all"
                if key == ord("b"):
                    return "back"
                if key == ord("l"):
                    self._launch_export()
                    return "step0"
        finally:
            cv2.destroyWindow(WINDOW_NAME_STEP3)


class Cv2AnnotationToGridTransformApp:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.mapping = parse_mapping_doc(args.mapping_doc)
        self.cases = load_curated_cases(args.manifest_csv, args.vtln_dir)
        self.context_cache = ContextLoadCache()
        self._selection_prewarm: Callable[[WorkspaceSelection], None] | None = None
        if self.args.cache_mode == "startup":
            self._selection_prewarm = self.context_cache.prewarm

    def _run_source_editor(self, selection: WorkspaceSelection, source_context: dict[str, object]) -> tuple[str, dict[str, np.ndarray]]:
        source_annotation_path = step_file_paths(selection.workspace_dir)["source_annotation"]
        metadata = source_annotation_metadata(selection, source_context["snapshot"], tuple(source_context["source_frame"].shape[:2]))
        editor = Cv2ContourEditorWindow(
            window_name=WINDOW_NAME_STEP1,
            image=source_context["source_frame"],
            contours=source_context["projected_contours"],
            save_path=source_annotation_path,
            metadata=metadata,
            title_lines=[
                "Step 1A - Source annotation",
                f"{selection.case.speaker}/{selection.case.session} frame {selection.case.frame_index_1based:04d}",
                "S = save only | N = save and continue to Step 1B | G = continue to Step 1B without saving",
                "No direct jump to Step 0 from Step 1",
            ],
            info_lines=source_text_block(source_context["snapshot"], selection.reference_speaker),
            allow_back=False,
            args=self.args,
        )
        return editor.run()

    def _run_target_editor(self, selection: WorkspaceSelection, target_context: dict[str, object]) -> tuple[str, dict[str, np.ndarray]]:
        target_annotation_path = step_file_paths(selection.workspace_dir)["target_annotation"]
        metadata = target_annotation_metadata(selection, tuple(target_context["target_image"].shape[:2]))
        info_lines = [
            f"target type: {selection.target.target_type}",
            f"label: {target_context['target_label']}",
            f"shape: {target_context['target_image'].shape[1]}x{target_context['target_image'].shape[0]}",
        ]
        editor = Cv2ContourEditorWindow(
            window_name=WINDOW_NAME_STEP1,
            image=target_context["target_image"],
            contours=target_context["target_contours"],
            save_path=target_annotation_path,
            metadata=metadata,
            title_lines=[
                "Step 1B - Target annotation",
                target_context["target_label"],
                "S = save only | B = save and go back to Step 1A | N = save and continue to Step 2",
                "G = continue to Step 2 without saving",
            ],
            info_lines=info_lines,
            allow_back=True,
            args=self.args,
        )
        return editor.run()

    def run(self) -> int:
        selection_window = Cv2SelectionWindow(
            cases=self.cases,
            mapping=self.mapping,
            args=self.args,
            prewarm_selection=self._selection_prewarm,
            cache_counts=self.context_cache.cache_counts,
        )
        try:
            while True:
                selection = selection_window.run()
                if selection is None:
                    return 0

                if self.args.cache_mode == "startup":
                    self.context_cache.prewarm(selection)
                source_context = self.context_cache.get_source(selection)
                target_context = self.context_cache.get_target(selection)

                source_action, source_contours = self._run_source_editor(selection, source_context)
                if source_action == "exit_all":
                    return 0

                while True:
                    target_action, target_contours = self._run_target_editor(selection, target_context)
                    if target_action == "exit_all":
                        return 0
                    if target_action == "back":
                        source_action, source_contours = self._run_source_editor(selection, source_context)
                        if source_action == "exit_all":
                            return 0
                        continue
                    break

                while True:
                    review_window = Cv2TransformReviewWindow(
                        selection=selection,
                        source_image=source_context["source_frame"],
                        source_contours=source_contours,
                        target_image=target_context["target_image"],
                        target_contours=target_contours,
                        args=self.args,
                    )
                    review_action = review_window.run()
                    if review_action == "exit_all":
                        return 0
                    if review_action == "step0":
                        break
                    if review_action == "back_step1":
                        target_action, target_contours = self._run_target_editor(selection, target_context)
                        if target_action == "exit_all":
                            return 0
                        if target_action == "back":
                            source_action, source_contours = self._run_source_editor(selection, source_context)
                            if source_action == "exit_all":
                                return 0
                            target_action, target_contours = self._run_target_editor(selection, target_context)
                            if target_action == "exit_all":
                                return 0
                        continue

                    export_window = Cv2ExportStepWindow(selection=selection, args=self.args)
                    export_action = export_window.run()
                    if export_action == "exit_all":
                        return 0
                    if export_action == "back":
                        continue
                    break
                continue
        finally:
            self.context_cache.close()


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    app = Cv2AnnotationToGridTransformApp(args)
    return app.run()


if __name__ == "__main__":
    raise SystemExit(main())
