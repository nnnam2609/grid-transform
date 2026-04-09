from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from grid_transform.cv2_shared import CONTOUR_COLORS, hex_to_bgr
from grid_transform.image_utils import as_grayscale_uint8


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


@dataclass(frozen=True)
class ClickTarget:
    x0: int
    y0: int
    width: int
    height: int

    def contains(self, x: int, y: int) -> bool:
        return self.x0 <= x < self.x0 + self.width and self.y0 <= y < self.y0 + self.height


def bgr_gray(image: np.ndarray) -> np.ndarray:
    array = as_grayscale_uint8(image)
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
        draw_polyline(
            panel,
            mapping,
            np.asarray(line, dtype=float),
            color_h,
            thickness=3 if index in (0, len(horiz_lines) - 1) else 1,
        )
    for index, line in enumerate(vert_lines):
        draw_polyline(
            panel,
            mapping,
            np.asarray(line, dtype=float),
            color_v,
            thickness=3 if index in (0, len(vert_lines) - 1) else 1,
        )


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


def draw_action_button(
    canvas: np.ndarray,
    *,
    rect: ClickTarget,
    title: str,
    subtitle: str,
    enabled: bool,
    accent_color: tuple[int, int, int] = ACCENT,
) -> None:
    fill = (42, 42, 42) if enabled else (28, 28, 28)
    outline = accent_color if enabled else (72, 72, 72)
    title_color = TEXT_COLOR if enabled else MUTED_TEXT
    subtitle_color = accent_color if enabled else (120, 120, 120)
    cv2.rectangle(canvas, (rect.x0, rect.y0), (rect.x0 + rect.width, rect.y0 + rect.height), fill, -1)
    cv2.rectangle(canvas, (rect.x0, rect.y0), (rect.x0 + rect.width, rect.y0 + rect.height), outline, 1)
    cv2.putText(canvas, title, (rect.x0 + 12, rect.y0 + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.52, title_color, 1, cv2.LINE_AA)
    cv2.putText(canvas, subtitle, (rect.x0 + 12, rect.y0 + 46), cv2.FONT_HERSHEY_SIMPLEX, 0.42, subtitle_color, 1, cv2.LINE_AA)


def _collect_xy_arrays(*collections: object) -> list[np.ndarray]:
    arrays: list[np.ndarray] = []
    stack = list(collections)
    while stack:
        value = stack.pop()
        if value is None:
            continue
        if isinstance(value, dict):
            stack.extend(value.values())
            continue
        if isinstance(value, (list, tuple)):
            stack.extend(value)
            continue
        arr = np.asarray(value, dtype=float)
        if arr.ndim == 1 and arr.shape[0] == 2:
            arrays.append(arr.reshape(1, 2))
        elif arr.ndim == 2 and arr.shape[1] == 2:
            arrays.append(arr)
    return arrays


def build_focus_view(
    image_shape: tuple[int, int],
    panel_w: int,
    panel_h: int,
    *collections: object,
    padding: float = 20.0,
) -> ViewState:
    arrays = _collect_xy_arrays(*collections)
    if not arrays:
        return ensure_view_state(None, image_shape)
    points = np.concatenate(arrays, axis=0)
    finite = np.isfinite(points).all(axis=1)
    points = points[finite]
    if len(points) == 0:
        return ensure_view_state(None, image_shape)

    image_h, image_w = image_shape[:2]
    x_min = float(np.clip(points[:, 0].min() - padding, 0.0, image_w))
    x_max = float(np.clip(points[:, 0].max() + padding, 0.0, image_w))
    y_min = float(np.clip(points[:, 1].min() - padding, 0.0, image_h))
    y_max = float(np.clip(points[:, 1].max() + padding, 0.0, image_h))

    crop_w = max(32.0, x_max - x_min)
    crop_h = max(32.0, y_max - y_min)
    panel_aspect = max(float(panel_w) / max(float(panel_h), 1.0), 1e-6)
    crop_aspect = crop_w / max(crop_h, 1e-6)
    if crop_aspect > panel_aspect:
        crop_h = crop_w / panel_aspect
    else:
        crop_w = crop_h * panel_aspect

    crop_w = min(crop_w, float(image_w))
    crop_h = min(crop_h, float(image_h))
    zoom = min(float(image_w) / max(crop_w, 1.0), float(image_h) / max(crop_h, 1.0))
    view = ViewState(
        zoom=max(1.0, zoom),
        center_x=float((x_min + x_max) * 0.5),
        center_y=float((y_min + y_max) * 0.5),
    )
    return ensure_view_state(view, image_shape)


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
    cv2.putText(
        canvas,
        "Other visible landmarks",
        (x + 26, y + 64),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.46,
        OTHER_LANDMARK_COLOR,
        1,
        cv2.LINE_AA,
    )
