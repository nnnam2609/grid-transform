from __future__ import annotations

import argparse
import shutil
import zipfile
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

try:
    import cv2
except ModuleNotFoundError:
    cv2 = None

import numpy as np
from roifile import ImagejRoi

from grid_transform.cv2_panels import (
    ACCENT,
    FOOTER_BG,
    KEY_DOWN,
    KEY_UP,
    MUTED_TEXT,
    PANEL_BG,
    ROW_BG,
    ROW_SELECTED,
    SIDEBAR_BG,
    TEXT_COLOR,
    ClickTarget,
    PanelMapping,
    ViewState,
    build_panel_mapping,
    draw_contours,
    draw_grid,
    draw_header_text,
    draw_polyline,
    draw_sidebar_block,
    ensure_view_state,
    render_panel_base,
    screen_to_world,
    world_to_screen,
)
from grid_transform.cv2_shared import (
    CONTOUR_COLORS,
    WINDOW_MARGIN,
    clamp_point,
    hex_to_bgr,
    scaled_window_size,
)
from grid_transform.config import DEFAULT_OUTPUT_DIR, DEFAULT_VTLN_DIR
from grid_transform.image_utils import as_grayscale_uint8
from grid_transform.io import load_frame_npy, load_frame_vtln
from grid_transform.source_annotation import LONG_CONTOUR_HANDLE_COUNTS
from grid_transform.transform_helpers import resample_polyline
from grid_transform.vt import build_grid


WINDOW_NAME = "VTLN Annotation Browser"

LIST_PANEL_WIDTH = 360
EDITOR_PANEL_WIDTH = 920
GRID_PANEL_WIDTH = 920
PANEL_HEIGHT = 940
PANEL_GAP = 12
FOOTER_HEIGHT = 96

LIST_ROW_HEIGHT = 50
LIST_ROW_GAP = 6
HANDLE_RADIUS = 4
ACTIVE_HANDLE_RADIUS = 6

SOURCE_HORIZ_COLOR = hex_to_bgr("#3a86ff")
SOURCE_VERT_COLOR = hex_to_bgr("#ffbe0b")
SOURCE_VT_COLOR = hex_to_bgr("#8338ec")
SOURCE_SPINE_COLOR = hex_to_bgr("#2a9d8f")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Browse every VTLN ROI zip bundle and bundled nnUNet contour case, edit one entry at a time, "
            "preview the live grid, then optionally overwrite the source zip after "
            "creating a timestamped backup."
        )
    )
    parser.add_argument(
        "--vtln-dir",
        type=Path,
        default=DEFAULT_VTLN_DIR,
        help="Folder containing VTLN PNG/TIF images, ROI zip bundles, and optional nnUNet cases.",
    )
    parser.add_argument(
        "--backup-root",
        type=Path,
        default=DEFAULT_OUTPUT_DIR / "vtln_annotation_browser" / "backups",
        help="Where timestamped zip backups are written before overwriting a speaker bundle.",
    )
    parser.add_argument(
        "--window-mode",
        choices=("full-height", "fullscreen", "fixed"),
        default="full-height",
        help="Initial cv2 window mode.",
    )
    return parser.parse_args(argv)


@dataclass(frozen=True)
class BrowserEntry:
    entry_id: str
    display_name: str
    image_name: str
    zip_path: Path
    load_kind: str
    data_dir: Path
    contours_dir: Path | None = None

    @property
    def backup_stem(self) -> str:
        safe = self.entry_id.replace("::", "__").replace("/", "_").replace("\\", "_").replace(":", "_")
        return safe.replace("^", "_")


def discover_vtln_entries(vtln_dir: Path) -> list[BrowserEntry]:
    vtln_dir = Path(vtln_dir)
    entries: list[BrowserEntry] = []

    for zip_path in sorted(vtln_dir.glob("*.zip")):
        speaker_id = zip_path.stem
        entries.append(
            BrowserEntry(
                entry_id=speaker_id,
                display_name=speaker_id,
                image_name=speaker_id,
                zip_path=zip_path,
                load_kind="vtln",
                data_dir=vtln_dir,
            )
        )

    nnunet_root = vtln_dir / "nnunet_data_80"
    if nnunet_root.is_dir():
        for zip_path in sorted(nnunet_root.glob("**/contours/*.zip")):
            contours_dir = zip_path.parent
            data_dir = contours_dir.parent
            if not (data_dir / "PNG_MR").is_dir():
                continue
            frame_name = zip_path.stem
            case_rel = data_dir.relative_to(nnunet_root).as_posix()
            entry_id = f"nnunet::{case_rel}::{frame_name}"
            display_name = f"nnunet/{case_rel}:{frame_name}"
            entries.append(
                BrowserEntry(
                    entry_id=entry_id,
                    display_name=display_name,
                    image_name=frame_name,
                    zip_path=zip_path,
                    load_kind="nnunet",
                    data_dir=data_dir,
                    contours_dir=contours_dir,
                )
            )

    return entries


def clone_contours(contours: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {
        name: np.asarray(points, dtype=float).copy()
        for name, points in contours.items()
    }


def contours_match(lhs: dict[str, np.ndarray], rhs: dict[str, np.ndarray], *, atol: float = 1e-6) -> bool:
    if lhs.keys() != rhs.keys():
        return False
    for name in lhs:
        left = np.asarray(lhs[name], dtype=float)
        right = np.asarray(rhs[name], dtype=float)
        if left.shape != right.shape:
            return False
        if not np.allclose(left, right, atol=atol):
            return False
    return True


def build_handle_points(contours: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    handles: dict[str, np.ndarray] = {}
    for name, points in contours.items():
        pts = np.asarray(points, dtype=float)
        n_handles = LONG_CONTOUR_HANDLE_COUNTS.get(name)
        if n_handles is None or len(pts) <= n_handles:
            handles[name] = pts.copy()
            continue
        indices = np.linspace(0, len(pts) - 1, num=n_handles)
        indices = np.unique(np.round(indices).astype(int))
        handles[name] = pts[indices].copy()
    return handles


def rebuild_contours_from_handles(
    handles: dict[str, np.ndarray],
    point_counts: dict[str, int],
) -> dict[str, np.ndarray]:
    contours: dict[str, np.ndarray] = {}
    for name, points in handles.items():
        pts = np.asarray(points, dtype=float)
        n_points = int(point_counts.get(name, len(pts)))
        if n_points > len(pts):
            contours[name] = resample_polyline(pts, n_points)
        else:
            contours[name] = pts.copy()
    return contours


def write_annotation_zip(path: Path, basename: str, contours: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for label, points in sorted(contours.items()):
            pts = np.asarray(points, dtype=float)
            if pts.ndim != 2 or pts.shape[1] != 2 or len(pts) == 0:
                continue
            roi_name = f"{basename}_{label}"
            roi = ImagejRoi.frompoints(pts, name=roi_name)
            zf.writestr(f"{roi_name}.roi", roi.tobytes())


@dataclass
class SpeakerState:
    speaker_id: str
    display_name: str
    image_name: str
    image: np.ndarray | None = None
    current_contours: dict[str, np.ndarray] = field(default_factory=dict)
    saved_contours: dict[str, np.ndarray] = field(default_factory=dict)
    handle_points: dict[str, np.ndarray] = field(default_factory=dict)
    point_counts: dict[str, int] = field(default_factory=dict)
    grid: object | None = None
    zip_path: Path | None = None
    dirty: bool = False
    load_error: str | None = None
    grid_error: str | None = None
    last_backup_path: Path | None = None


class VTLNAnnotationBrowser:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.entries = discover_vtln_entries(args.vtln_dir)
        if not self.entries:
            raise FileNotFoundError(f"No VTLN ROI zip files or nnUNet contour zips found in {args.vtln_dir}")
        self.entry_by_id = {entry.entry_id: entry for entry in self.entries}
        self.speaker_ids = [entry.entry_id for entry in self.entries]

        self.states: dict[str, SpeakerState] = {}
        self.selected_index = 0
        self.status_message = (
            "Ready. Drag handles in the middle panel. Save creates a backup before overwriting the source zip."
        )
        self.list_click_targets: dict[int, ClickTarget] = {}
        self.editor_rect = ClickTarget(
            x0=LIST_PANEL_WIDTH + PANEL_GAP,
            y0=0,
            width=EDITOR_PANEL_WIDTH,
            height=PANEL_HEIGHT,
        )
        self.grid_rect = ClickTarget(
            x0=LIST_PANEL_WIDTH + PANEL_GAP + EDITOR_PANEL_WIDTH + PANEL_GAP,
            y0=0,
            width=GRID_PANEL_WIDTH,
            height=PANEL_HEIGHT,
        )
        self.editor_mapping: PanelMapping | None = None
        self.grid_mapping: PanelMapping | None = None
        self.image_view: ViewState | None = None
        self.active_handle: tuple[str, int] | None = None
        self.active_handle_backup: np.ndarray | None = None
        self.dragging_pan = False
        self.pan_anchor: tuple[str, int, int] | None = None
        self.load_state(self.current_speaker_id)
        self._reset_view()

    @property
    def current_speaker_id(self) -> str:
        return self.speaker_ids[self.selected_index]

    @property
    def current_state(self) -> SpeakerState:
        return self.load_state(self.current_speaker_id)

    @property
    def current_entry(self) -> BrowserEntry:
        return self.entry_by_id[self.current_speaker_id]

    def load_state(self, speaker_id: str, *, force_reload: bool = False) -> SpeakerState:
        if not force_reload and speaker_id in self.states:
            return self.states[speaker_id]

        entry = self.entry_by_id[speaker_id]
        state = SpeakerState(
            speaker_id=speaker_id,
            display_name=entry.display_name,
            image_name=entry.image_name,
            zip_path=entry.zip_path,
        )
        try:
            if entry.load_kind == "nnunet":
                image, contours = load_frame_npy(int(entry.image_name), entry.data_dir, entry.contours_dir)
            else:
                image, contours = load_frame_vtln(entry.image_name, entry.data_dir)
            state.image = as_grayscale_uint8(image)
            state.current_contours = clone_contours(contours)
            state.saved_contours = clone_contours(contours)
            state.point_counts = {
                name: int(len(np.asarray(points, dtype=float)))
                for name, points in contours.items()
            }
            state.handle_points = build_handle_points(state.current_contours)
            try:
                state.grid = build_grid(
                    state.image,
                    state.current_contours,
                    n_vert=9,
                    n_points=250,
                    frame_number=0,
                )
            except Exception as exc:
                state.grid_error = str(exc)
        except Exception as exc:
            state.load_error = str(exc)

        self.states[speaker_id] = state
        return state

    def _current_preview_contours(self) -> dict[str, np.ndarray]:
        state = self.current_state
        if self.active_handle is None:
            return state.current_contours
        return rebuild_contours_from_handles(state.handle_points, state.point_counts)

    def _reset_view(self) -> None:
        state = self.current_state
        if state.image is None:
            self.image_view = None
            return
        self.image_view = ensure_view_state(None, tuple(state.image.shape[:2]))

    def _ensure_image_view(self) -> ViewState | None:
        state = self.current_state
        if state.image is None:
            self.image_view = None
            return None
        self.image_view = ensure_view_state(self.image_view, tuple(state.image.shape[:2]))
        return self.image_view

    def _speaker_state_summary(self, speaker_id: str) -> str:
        state = self.states.get(speaker_id)
        if state is None:
            return "lazy"
        if state.dirty:
            return "dirty"
        if state.load_error is not None:
            return "load error"
        if state.grid is None:
            return "grid issue"
        return "ready"

    def _status_prefix(self, speaker_id: str) -> str:
        state = self.states.get(speaker_id)
        if state is None:
            return " "
        if state.dirty:
            return "*"
        if state.load_error is not None or state.grid is None:
            return "!"
        return " "

    def _render_list_panel(self) -> np.ndarray:
        panel = np.full((PANEL_HEIGHT, LIST_PANEL_WIDTH, 3), SIDEBAR_BG, dtype=np.uint8)
        draw_header_text(
            panel,
            [
                "VTLN bundles + nnUNet",
                f"{len(self.speaker_ids)} entries discovered",
                "Up/Down or click to switch",
            ],
        )
        state = self.current_state
        top = 92
        top = draw_sidebar_block(
            panel,
            "Current",
            [
                f"entry: {self.current_entry.display_name}",
                f"state: {self._speaker_state_summary(self.current_speaker_id)}",
                "grid: H1 tail uses I5->C1 chord clamp",
                "midline: still contributes I6/I7 landmarks",
            ],
            top,
            highlight=True,
        )

        if state.last_backup_path is not None:
            top = draw_sidebar_block(
                panel,
                "Last Save",
                [
                    f"backup: {state.last_backup_path.name}",
                    f"zip: {state.zip_path.name if state.zip_path is not None else '-'}",
                ],
                top,
            )

        cv2.putText(panel, "Entry List", (18, top + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, TEXT_COLOR, 1, cv2.LINE_AA)
        list_top = top + 36
        visible_count = max(1, (PANEL_HEIGHT - list_top - 18) // (LIST_ROW_HEIGHT + LIST_ROW_GAP))
        start = min(
            max(0, self.selected_index - visible_count // 2),
            max(0, len(self.speaker_ids) - visible_count),
        )
        stop = min(len(self.speaker_ids), start + visible_count)
        self.list_click_targets = {}

        row_y = list_top
        for visible_index, speaker_id in enumerate(self.speaker_ids[start:stop], start=start):
            entry = self.entry_by_id[speaker_id]
            rect = ClickTarget(x0=12, y0=row_y, width=LIST_PANEL_WIDTH - 24, height=LIST_ROW_HEIGHT)
            fill = ROW_SELECTED if visible_index == self.selected_index else ROW_BG
            cv2.rectangle(
                panel,
                (rect.x0, rect.y0),
                (rect.x0 + rect.width, rect.y0 + rect.height),
                fill,
                -1,
            )
            cv2.rectangle(
                panel,
                (rect.x0, rect.y0),
                (rect.x0 + rect.width, rect.y0 + rect.height),
                (90, 90, 90),
                1,
            )
            prefix = self._status_prefix(speaker_id)
            cv2.putText(
                panel,
                f"{prefix} {entry.display_name[:34]}",
                (rect.x0 + 12, rect.y0 + 22),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.50,
                TEXT_COLOR,
                1,
                cv2.LINE_AA,
            )
            cv2.putText(
                panel,
                self._speaker_state_summary(speaker_id),
                (rect.x0 + 12, rect.y0 + 42),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.42,
                MUTED_TEXT,
                1,
                cv2.LINE_AA,
            )
            self.list_click_targets[visible_index] = rect
            row_y += LIST_ROW_HEIGHT + LIST_ROW_GAP
        return panel

    def _render_error_panel(self, width: int, title_lines: list[str], error_lines: list[str]) -> np.ndarray:
        panel = np.full((PANEL_HEIGHT, width, 3), PANEL_BG, dtype=np.uint8)
        draw_header_text(panel, title_lines)
        top = 120
        draw_sidebar_block(panel, "Issue", error_lines, top, highlight=True)
        return panel

    def _render_editor_panel(self) -> np.ndarray:
        state = self.current_state
        if state.load_error is not None or state.image is None:
            self.editor_mapping = None
            return self._render_error_panel(
                EDITOR_PANEL_WIDTH,
                [
                    "Editable annotation",
                    self.current_entry.display_name,
                    "Load failed",
                ],
                [state.load_error or "Unknown load error"],
            )

        view = self._ensure_image_view()
        if view is None:
            self.editor_mapping = None
            return self._render_error_panel(
                EDITOR_PANEL_WIDTH,
                [
                    "Editable annotation",
                    self.current_entry.display_name,
                    "View unavailable",
                ],
                ["Could not initialize image view."],
            )
        panel = np.full((PANEL_HEIGHT, EDITOR_PANEL_WIDTH, 3), PANEL_BG, dtype=np.uint8)
        panel, mapping = self._render_base_panel(
            image=state.image,
            panel=panel,
            width=EDITOR_PANEL_WIDTH,
            height=PANEL_HEIGHT,
            view=view,
        )
        self.editor_mapping = mapping

        contours = self._current_preview_contours()
        for name, points in sorted(contours.items()):
            draw_polyline(
                panel,
                mapping,
                np.asarray(points, dtype=float),
                hex_to_bgr(CONTOUR_COLORS.get(name, "#00b4d8")),
                dashed=name in {"pharynx", "soft-palate-midline"},
                thickness=2,
            )

        for name, points in sorted(state.handle_points.items()):
            scaled = np.round(world_to_screen(mapping, np.asarray(points, dtype=float))).astype(int)
            for index, point in enumerate(scaled):
                is_active = self.active_handle == (name, index)
                radius = ACTIVE_HANDLE_RADIUS if is_active else HANDLE_RADIUS
                cv2.circle(panel, tuple(point), radius + 2, (255, 255, 255), -1, cv2.LINE_AA)
                cv2.circle(
                    panel,
                    tuple(point),
                    radius,
                    hex_to_bgr(CONTOUR_COLORS.get(name, "#ffffff")),
                    -1,
                    cv2.LINE_AA,
                )
                if is_active:
                    cv2.putText(
                        panel,
                        f"{name}[{index}]",
                        (point[0] + 6, point[1] - 6),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.42,
                        ACCENT,
                        1,
                        cv2.LINE_AA,
                    )

        draw_header_text(
            panel,
            [
                "Editable VTLN annotation",
                self.current_entry.display_name,
                "Drag handles. Grid rebuild happens on mouse-up only.",
            ],
        )
        if state.grid_error is not None and state.grid is None:
            self._draw_footer_block(panel, [f"Current grid unavailable: {state.grid_error}"], accent=True)
        return panel

    def _render_grid_panel(self) -> np.ndarray:
        state = self.current_state
        if state.load_error is not None or state.image is None:
            self.grid_mapping = None
            return self._render_error_panel(
                GRID_PANEL_WIDTH,
                [
                    "Live grid preview",
                    self.current_entry.display_name,
                    "Load failed",
                ],
                [state.load_error or "Unknown load error"],
            )

        view = self._ensure_image_view()
        if view is None:
            self.grid_mapping = None
            return self._render_error_panel(
                GRID_PANEL_WIDTH,
                [
                    "Live grid preview",
                    self.current_entry.display_name,
                    "View unavailable",
                ],
                ["Could not initialize image view."],
            )
        panel = np.full((PANEL_HEIGHT, GRID_PANEL_WIDTH, 3), PANEL_BG, dtype=np.uint8)
        panel, mapping = self._render_base_panel(
            image=state.image,
            panel=panel,
            width=GRID_PANEL_WIDTH,
            height=PANEL_HEIGHT,
            view=view,
        )
        self.grid_mapping = mapping

        draw_contours(panel, mapping, state.current_contours, dim=True)
        if state.grid is not None:
            draw_grid(
                panel,
                mapping,
                state.grid.horiz_lines,
                state.grid.vert_lines,
                color_h=SOURCE_HORIZ_COLOR,
                color_v=SOURCE_VERT_COLOR,
            )
            if state.grid.vt_curve is not None:
                draw_polyline(panel, mapping, state.grid.vt_curve, SOURCE_VT_COLOR, thickness=3)
            if state.grid.spine_curve is not None:
                draw_polyline(panel, mapping, state.grid.spine_curve, SOURCE_SPINE_COLOR, thickness=3)

        draw_header_text(
            panel,
            [
                "Live no-mid-plane grid preview",
                self.current_entry.display_name,
                "H1 tail: I5 -> C1 chord clamp",
            ],
        )

        info_lines: list[str]
        if state.grid is None:
            info_lines = [f"Grid unavailable: {state.grid_error or 'unknown error'}"]
            self._draw_footer_block(panel, info_lines, accent=True)
            return panel

        info_lines = [
            f"h1_dev_ratio: {float(state.grid.h1_dev_ratio):.3f}",
            f"h1_max_clamp: {float(state.grid.h1_max_clamp):.3f}",
            f"contours: {len(state.current_contours)}",
        ]
        self._draw_footer_block(panel, info_lines, accent=False)
        return panel

    @staticmethod
    def _render_base_panel(
        *,
        image: np.ndarray,
        panel: np.ndarray,
        width: int,
        height: int,
        view,
    ) -> tuple[np.ndarray, PanelMapping]:
        rendered, mapping = render_panel_base(image, width, height, view)
        panel[:] = rendered
        return panel, mapping

    @staticmethod
    def _draw_footer_block(panel: np.ndarray, lines: list[str], *, accent: bool) -> None:
        block_h = 16 + 22 * len(lines)
        y1 = panel.shape[0] - 12
        y0 = max(12, y1 - block_h)
        overlay = panel.copy()
        cv2.rectangle(overlay, (12, y0), (panel.shape[1] - 12, y1), (0, 0, 0), -1)
        panel[:] = cv2.addWeighted(overlay, 0.34, panel, 0.66, 0.0)
        color = ACCENT if accent else TEXT_COLOR
        for row, line in enumerate(lines):
            cv2.putText(
                panel,
                line,
                (24, y0 + 24 + row * 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.46,
                color,
                1,
                cv2.LINE_AA,
            )

    def _compose_canvas(self) -> np.ndarray:
        canvas_h = PANEL_HEIGHT + FOOTER_HEIGHT
        canvas_w = LIST_PANEL_WIDTH + EDITOR_PANEL_WIDTH + GRID_PANEL_WIDTH + PANEL_GAP * 2
        canvas = np.full((canvas_h, canvas_w, 3), PANEL_BG, dtype=np.uint8)

        list_panel = self._render_list_panel()
        editor_panel = self._render_editor_panel()
        grid_panel = self._render_grid_panel()

        canvas[:PANEL_HEIGHT, :LIST_PANEL_WIDTH] = list_panel
        canvas[:PANEL_HEIGHT, self.editor_rect.x0 : self.editor_rect.x0 + self.editor_rect.width] = editor_panel
        canvas[:PANEL_HEIGHT, self.grid_rect.x0 : self.grid_rect.x0 + self.grid_rect.width] = grid_panel

        for rect in (self.editor_rect, self.grid_rect):
            cv2.rectangle(
                canvas,
                (rect.x0, rect.y0),
                (rect.x0 + rect.width - 1, rect.y0 + rect.height - 1),
                (70, 70, 70),
                1,
            )

        footer_y = PANEL_HEIGHT
        canvas[footer_y:, :] = FOOTER_BG
        help_line = "Up/Down or click speaker | Wheel zoom | Right-drag pan | Drag handle | S save | R reset | X exit"
        cv2.putText(canvas, help_line, (12, footer_y + 28), cv2.FONT_HERSHEY_SIMPLEX, 0.56, TEXT_COLOR, 1, cv2.LINE_AA)
        cv2.putText(
            canvas,
            self.status_message[:220],
            (12, footer_y + 62),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            ACCENT,
            1,
            cv2.LINE_AA,
        )
        return canvas

    def _nearest_handle(self, x: float, y: float, mapping: PanelMapping) -> tuple[str, int] | None:
        threshold_world = 10.0 / max(mapping.scale, 1e-6)
        query = np.array([x, y], dtype=float)
        best: tuple[str, int] | None = None
        best_distance = float("inf")
        for name, points in self.current_state.handle_points.items():
            pts = np.asarray(points, dtype=float)
            if len(pts) == 0:
                continue
            distances = np.linalg.norm(pts - query, axis=1)
            index = int(np.argmin(distances))
            distance = float(distances[index])
            if distance < best_distance:
                best_distance = distance
                best = (name, index)
        if best is None or best_distance > threshold_world:
            return None
        return best

    def _update_active_handle(self, x: float, y: float) -> None:
        if self.active_handle is None:
            return
        state = self.current_state
        contour_name, point_index = self.active_handle
        state.handle_points[contour_name][point_index] = clamp_point(
            np.array([x, y], dtype=float),
            state.image.shape[1],
            state.image.shape[0],
        )

    def _zoom(self, factor: float, anchor_world: tuple[float, float] | None) -> None:
        state = self.current_state
        if state.image is None:
            return
        view = self._ensure_image_view()
        if view is None:
            return
        if anchor_world is None:
            anchor_world = (view.center_x, view.center_y)
        view.zoom = float(np.clip(view.zoom * factor, 1.0, 12.0))
        view.center_x = float(anchor_world[0])
        view.center_y = float(anchor_world[1])
        self.image_view = ensure_view_state(view, tuple(state.image.shape[:2]))

    def _pan_by(self, dx_screen: int, dy_screen: int, mapping: PanelMapping) -> None:
        state = self.current_state
        if state.image is None:
            return
        view = self._ensure_image_view()
        if view is None:
            return
        view.center_x -= dx_screen / max(mapping.scale, 1e-6)
        view.center_y -= dy_screen / max(mapping.scale, 1e-6)
        self.image_view = ensure_view_state(view, tuple(state.image.shape[:2]))

    def _panel_context_at(self, x: int, y: int) -> tuple[str, PanelMapping, ClickTarget] | None:
        if self.editor_rect.contains(x, y) and self.editor_mapping is not None:
            return ("editor", self.editor_mapping, self.editor_rect)
        if self.grid_rect.contains(x, y) and self.grid_mapping is not None:
            return ("grid", self.grid_mapping, self.grid_rect)
        return None

    def _restore_active_handle(self) -> None:
        if self.active_handle is None or self.active_handle_backup is None:
            return
        state = self.current_state
        contour_name, _ = self.active_handle
        state.handle_points[contour_name] = self.active_handle_backup.copy()

    def _commit_active_handle(self) -> None:
        state = self.current_state
        if state.image is None:
            return
        preview_contours = rebuild_contours_from_handles(state.handle_points, state.point_counts)
        try:
            grid = build_grid(
                state.image,
                preview_contours,
                n_vert=9,
                n_points=250,
                frame_number=0,
            )
        except Exception as exc:
            self._restore_active_handle()
            self.status_message = f"Invalid edit reverted: {exc}"
            return

        state.current_contours = preview_contours
        state.grid = grid
        state.grid_error = None
        state.dirty = not contours_match(state.current_contours, state.saved_contours)
        self.status_message = (
            "Grid preview rebuilt. Save writes a backup first."
            if state.dirty
            else "Edit matches the saved bundle again."
        )

    def _try_select_index(self, index: int) -> None:
        if index < 0 or index >= len(self.speaker_ids) or index == self.selected_index:
            return
        if self.current_state.dirty:
            self.status_message = "Current speaker has unsaved edits. Save or reset before switching."
            return
        self.selected_index = index
        self.active_handle = None
        self.active_handle_backup = None
        self.dragging_pan = False
        self.pan_anchor = None
        self.load_state(self.current_speaker_id)
        self._reset_view()
        self.status_message = f"Loaded {self.current_entry.display_name}"

    def _save_current(self) -> None:
        state = self.current_state
        if state.load_error is not None or state.zip_path is None:
            self.status_message = "Cannot save because the current speaker failed to load."
            return
        if state.grid is None:
            self.status_message = "Cannot save while the current contour set does not build a valid grid."
            return
        if not state.dirty:
            self.status_message = "No unsaved changes for the current speaker."
            return

        backup_dir = self.args.backup_root / self.current_entry.backup_stem
        backup_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = backup_dir / f"{self.current_entry.backup_stem}_{timestamp}.zip"
        if state.zip_path.is_file():
            shutil.copy2(state.zip_path, backup_path)
        write_annotation_zip(state.zip_path, state.image_name, state.current_contours)
        state.saved_contours = clone_contours(state.current_contours)
        state.handle_points = build_handle_points(state.current_contours)
        state.dirty = False
        state.last_backup_path = backup_path
        self.status_message = f"Saved {state.display_name}. Backup: {backup_path.name}"

    def _reset_current(self) -> None:
        state = self.current_state
        if state.load_error is not None:
            self.status_message = "Nothing to reset because the current speaker failed to load."
            return
        state.current_contours = clone_contours(state.saved_contours)
        state.handle_points = build_handle_points(state.saved_contours)
        state.dirty = False
        if state.image is not None:
            try:
                state.grid = build_grid(
                    state.image,
                    state.current_contours,
                    n_vert=9,
                    n_points=250,
                    frame_number=0,
                )
                state.grid_error = None
            except Exception as exc:
                state.grid = None
                state.grid_error = str(exc)
        self.dragging_pan = False
        self.pan_anchor = None
        self.status_message = "Reset to the last loaded/saved contours."

    def _on_mouse(self, event: int, x: int, y: int, flags: int, _param) -> None:
        if event == cv2.EVENT_LBUTTONDOWN:
            for index, rect in self.list_click_targets.items():
                if rect.contains(x, y):
                    self._try_select_index(index)
                    return

        panel_context = self._panel_context_at(x, y)
        if event == cv2.EVENT_MOUSEWHEEL and panel_context is not None:
            _pane_name, mapping, rect = panel_context
            world_xy = screen_to_world(mapping, x - rect.x0, y - rect.y0)
            if world_xy is not None:
                self._zoom(1.18 if flags > 0 else 1.0 / 1.18, world_xy)
            return

        if event == cv2.EVENT_RBUTTONDOWN and panel_context is not None:
            pane_name, _mapping, _rect = panel_context
            self.dragging_pan = True
            self.pan_anchor = (pane_name, x, y)
            return

        if event == cv2.EVENT_MOUSEMOVE and self.dragging_pan and self.pan_anchor is not None:
            pane_name, anchor_x, anchor_y = self.pan_anchor
            active_context = self._panel_context_at(x, y)
            if active_context is not None and active_context[0] == pane_name:
                self._pan_by(x - anchor_x, y - anchor_y, active_context[1])
                self.pan_anchor = (pane_name, x, y)
            return

        if event == cv2.EVENT_RBUTTONUP:
            self.dragging_pan = False
            self.pan_anchor = None
            return

        if self.current_state.load_error is not None or self.editor_mapping is None:
            return

        local_x = x - self.editor_rect.x0
        local_y = y - self.editor_rect.y0
        world_xy = None
        if self.editor_rect.contains(x, y):
            world_xy = screen_to_world(self.editor_mapping, local_x, local_y)

        if event == cv2.EVENT_LBUTTONDOWN and world_xy is not None:
            nearest = self._nearest_handle(*world_xy, mapping=self.editor_mapping)
            if nearest is None:
                return
            contour_name, _ = nearest
            self.active_handle = nearest
            self.active_handle_backup = self.current_state.handle_points[contour_name].copy()
            self._update_active_handle(*world_xy)
            return

        if event == cv2.EVENT_MOUSEMOVE and self.active_handle is not None and world_xy is not None:
            self._update_active_handle(*world_xy)
            return

        if event == cv2.EVENT_LBUTTONUP and self.active_handle is not None:
            if world_xy is not None:
                self._update_active_handle(*world_xy)
            self._commit_active_handle()
            self.active_handle = None
            self.active_handle_backup = None

    def _apply_window_mode(self, canvas: np.ndarray) -> None:
        if self.args.window_mode == "fullscreen":
            cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
            return
        if self.args.window_mode == "fixed":
            width = min(1900, canvas.shape[1])
            height = min(1300, canvas.shape[0])
        else:
            width, height = scaled_window_size(canvas.shape[1], canvas.shape[0], margin=WINDOW_MARGIN)
        cv2.resizeWindow(WINDOW_NAME, width, height)

    def run(self) -> int:
        canvas = self._compose_canvas()
        window_flags = cv2.WINDOW_NORMAL
        if hasattr(cv2, "WINDOW_KEEPRATIO"):
            window_flags |= cv2.WINDOW_KEEPRATIO
        cv2.namedWindow(WINDOW_NAME, window_flags)
        cv2.setMouseCallback(WINDOW_NAME, self._on_mouse)
        self._apply_window_mode(canvas)

        try:
            while True:
                canvas = self._compose_canvas()
                cv2.imshow(WINDOW_NAME, canvas)
                key = cv2.waitKeyEx(20)
                if key == -1:
                    continue
                if key in (ord("x"), ord("X"), 27):
                    return 0
                if key == KEY_UP:
                    self._try_select_index(self.selected_index - 1)
                elif key == KEY_DOWN:
                    self._try_select_index(self.selected_index + 1)
                elif key in (ord("+"), ord("=")):
                    self._zoom(1.15, None)
                elif key in (ord("-"), ord("_")):
                    self._zoom(1.0 / 1.15, None)
                elif key in (ord("s"), ord("S")):
                    self._save_current()
                elif key in (ord("r"), ord("R")):
                    self._reset_current()
        finally:
            cv2.destroyWindow(WINDOW_NAME)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if cv2 is None:
        raise SystemExit(
            "cv2 is not installed in this environment. Install opencv-python in the repo venv first."
        )
    browser = VTLNAnnotationBrowser(args)
    return browser.run()


if __name__ == "__main__":
    raise SystemExit(main())
