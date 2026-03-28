from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

try:
    import cv2
except ModuleNotFoundError:
    cv2 = None

from grid_transform.annotation_workbench import (
    CONTOUR_COLORS,
    AnnotationWorkbenchConfig,
    AnnotationWorkbenchState,
    source_text_block,
)
from grid_transform.config import PROJECT_DIR
from grid_transform.source_annotation import LONG_CONTOUR_HANDLE_COUNTS


WINDOW_NAME = "Source Annotation Editor"
PANEL_SIZE = 560
PANEL_GAP = 16
FOOTER_HEIGHT = 78
PANEL_BG = (24, 24, 24)
FOOTER_BG = (16, 16, 16)
TEXT_COLOR = (235, 235, 235)

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Edit a projected source annotation on an ArtSpeech frame, then warp the session to the target."
    )
    parser.add_argument("--artspeech-speaker", default="P7", help="ArtSpeech speaker id, for example P7.")
    parser.add_argument("--session", default="S2", help="ArtSpeech session id, for example S2.")
    parser.add_argument("--reference-speaker", default="1640_s10_0829", help="VTNL reference speaker/image name.")
    parser.add_argument("--source-frame", type=int, help="Optional 1-based source frame override. Defaults to the best reference match.")
    parser.add_argument("--target-frame", type=int, default=143020, help="nnUNet target frame number.")
    parser.add_argument("--target-case", default="2008-003^01-1791/test", help="nnUNet target case relative path.")
    parser.add_argument("--dataset-root", type=Path, help="Optional explicit ArtSpeech dataset root.")
    parser.add_argument("--vtnl-dir", type=Path, default=PROJECT_DIR / "VTNL", help="Folder containing VTNL images and ROI zip files.")
    parser.add_argument("--output-dir", type=Path, help="Optional explicit output directory.")
    parser.add_argument("--output-mode", choices=("both", "warped", "review"), default="both", help="Which sequence outputs to render.")
    parser.add_argument("--max-output-frames", type=int, default=0, help="Optional debug limit for the sequence render triggered on save.")
    parser.add_argument("--skip-video-on-save", action="store_true", help="Headless mode only: save without rendering the output sequence.")
    parser.add_argument("--no-gui", action="store_true", help="Run the same initialization and save flow without opening the interactive window.")
    return parser.parse_args(argv)


def hex_to_bgr(color: str) -> tuple[int, int, int]:
    stripped = color.lstrip("#")
    if len(stripped) != 6:
        return (255, 255, 255)
    red = int(stripped[0:2], 16)
    green = int(stripped[2:4], 16)
    blue = int(stripped[4:6], 16)
    return blue, green, red


def clamp_point(point: np.ndarray, width: int, height: int) -> np.ndarray:
    return np.array(
        [
            np.clip(float(point[0]), 0.0, float(width - 1)),
            np.clip(float(point[1]), 0.0, float(height - 1)),
        ],
        dtype=float,
    )


class SourceAnnotationEditor:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.workbench = AnnotationWorkbenchState.from_projection(
            AnnotationWorkbenchConfig(
                artspeech_speaker=args.artspeech_speaker,
                session=args.session,
                reference_speaker=args.reference_speaker,
                source_frame=args.source_frame,
                target_frame=args.target_frame,
                target_case=args.target_case,
                dataset_root=args.dataset_root,
                vtnl_dir=args.vtnl_dir,
                output_dir=args.output_dir,
            )
        )
        self._sync_from_workbench()
        self.original_point_counts = {
            name: int(len(points))
            for name, points in self.original_contours.items()
        }
        self.handle_points = self._build_handle_points(self.original_contours)

        self.active_handle: tuple[str, int] | None = None
        self.active_handle_backup: np.ndarray | None = None
        self.status_message = "Ready. Drag handles in the source panel. Keys: s save only, v save+render, r reset, q quit."
        self.panel_images: dict[str, np.ndarray] = {}
        self.canvas_image: np.ndarray | None = None
        self.source_panel_origin = (0, 0)
        self.source_panel_scale_x = PANEL_SIZE / self.source_shape[1]
        self.source_panel_scale_y = PANEL_SIZE / self.source_shape[0]

    def _sync_from_workbench(self) -> None:
        self.dataset_root = self.workbench.dataset_root
        self.session_data = self.workbench.session_data
        self.reference_shape = self.workbench.reference_shape
        self.source_frame_index1 = self.workbench.source_frame_index1
        self.source_frame_index0 = self.workbench.source_frame_index0
        self.source_frame = self.workbench.source_frame
        self.source_shape = self.workbench.source_shape
        self.source_snapshot = self.workbench.source_snapshot
        self.output_dir = self.workbench.output_dir
        self.annotation_json_path = self.workbench.annotation_json_path
        self.target_frame = self.workbench.target_frame
        self.target_contours = self.workbench.target_contours
        self.target_grid = self.workbench.target_grid
        self.original_contours = self.workbench.original_contours
        self.current_contours = self.workbench.current_contours
        self.source_grid = self.workbench.source_grid
        self.articulators = self.workbench.articulators
        self.warped_source_contours = self.workbench.warped_source_contours
        self.warped_source_frame = self.workbench.warped_source_frame
        self.valid_ratio = self.workbench.valid_ratio

    def _build_handle_points(self, contours: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
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

    def _rebuild_contours_from_handles(self) -> dict[str, np.ndarray]:
        contours: dict[str, np.ndarray] = {}
        for name, handles in self.handle_points.items():
            pts = np.asarray(handles, dtype=float)
            if name in LONG_CONTOUR_HANDLE_COUNTS and self.original_point_counts[name] > len(pts):
                contours[name] = self._resample_polyline(pts, self.original_point_counts[name])
            else:
                contours[name] = pts.copy()
        return contours

    @staticmethod
    def _resample_polyline(points: np.ndarray, n_samples: int) -> np.ndarray:
        pts = np.asarray(points, dtype=float)
        if len(pts) == 1:
            return np.repeat(pts, n_samples, axis=0)
        segment = np.linalg.norm(np.diff(pts, axis=0), axis=1)
        keep = np.r_[True, segment > 1e-8]
        pts = pts[keep]
        if len(pts) < 2:
            return np.repeat(pts, n_samples, axis=0)
        arc = np.cumsum(np.r_[0.0, np.linalg.norm(np.diff(pts, axis=0), axis=1)])
        arc /= max(float(arc[-1]), 1e-8)
        t = np.linspace(0.0, 1.0, n_samples)
        return np.column_stack(
            [
                np.interp(t, arc, pts[:, 0]),
                np.interp(t, arc, pts[:, 1]),
            ]
        )

    def _recompute_state(self) -> None:
        contours = self._rebuild_contours_from_handles()
        self.workbench.set_contours(contours)
        self._sync_from_workbench()

    def _metadata_payload(self) -> dict[str, object]:
        return self.workbench.metadata_payload()

    def render_source_annotation(self, ax, *, show_handles: bool) -> None:
        self.workbench.render_source_annotation(ax)

        if show_handles:
            for name, handles in sorted(self.handle_points.items()):
                pts = np.asarray(handles, dtype=float)
                color = CONTOUR_COLORS.get(name, "#ffffff")
                ax.scatter(
                    pts[:, 0],
                    pts[:, 1],
                    s=34,
                    color=color,
                    edgecolors="white",
                    linewidths=0.9,
                    zorder=20,
                )

    def render_source_grid(self, ax) -> None:
        self.workbench.render_source_grid(ax)

    def render_target_panel(self, ax) -> None:
        self.workbench.render_target_panel(ax)

    def render_warped_preview(self, ax) -> None:
        self.workbench.render_warped_preview(ax)

    def _save_preview_images(self) -> dict[str, str]:
        return self.workbench._save_preview_images()

    def save_bundle(self, *, render_video: bool, max_output_frames: int, output_mode: str) -> dict[str, object]:
        result = self.workbench.save_bundle(
            render_video=render_video,
            max_output_frames=max_output_frames,
            output_mode=output_mode,
        )
        self._sync_from_workbench()
        return result

    def _render_source_panel_cv2(self, *, show_handles: bool) -> np.ndarray:
        image = cv2.cvtColor(
            cv2.resize(self.source_frame, (PANEL_SIZE, PANEL_SIZE), interpolation=cv2.INTER_NEAREST),
            cv2.COLOR_GRAY2BGR,
        )

        for name, points in sorted(self.current_contours.items()):
            pts = np.asarray(points, dtype=float)
            scaled = np.column_stack([pts[:, 0] * self.source_panel_scale_x, pts[:, 1] * self.source_panel_scale_y])
            rounded = np.round(scaled).astype(np.int32).reshape(-1, 1, 2)
            color = hex_to_bgr(CONTOUR_COLORS.get(name, "#00b4d8"))
            cv2.polylines(image, [rounded], False, color, 2, lineType=cv2.LINE_AA)
            anchor = np.round(scaled[len(scaled) // 2]).astype(int)
            cv2.putText(
                image,
                name,
                (int(anchor[0]) + 4, int(anchor[1]) - 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.34,
                color,
                1,
                cv2.LINE_AA,
            )

        if show_handles:
            for name, handles in sorted(self.handle_points.items()):
                color = hex_to_bgr(CONTOUR_COLORS.get(name, "#ffffff"))
                for handle in handles:
                    px = int(round(handle[0] * self.source_panel_scale_x))
                    py = int(round(handle[1] * self.source_panel_scale_y))
                    cv2.circle(image, (px, py), 5, (255, 255, 255), -1, lineType=cv2.LINE_AA)
                    cv2.circle(image, (px, py), 3, color, -1, lineType=cv2.LINE_AA)

        title_lines = [
            f"{self.args.artspeech_speaker}/{self.args.session} frame {self.source_frame_index1}",
            "Editable source annotation",
        ]
        for row, line in enumerate(title_lines):
            cv2.putText(
                image,
                line,
                (12, 24 + 20 * row),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

        lines = source_text_block(self.source_snapshot, self.args.reference_speaker)
        line_height = 18
        block_height = 10 + line_height * len(lines)
        overlay = image.copy()
        cv2.rectangle(
            overlay,
            (10, PANEL_SIZE - block_height - 10),
            (PANEL_SIZE - 10, PANEL_SIZE - 10),
            (0, 0, 0),
            -1,
        )
        image = cv2.addWeighted(overlay, 0.38, image, 0.62, 0.0)
        start_y = PANEL_SIZE - block_height + 8
        for row, line in enumerate(lines):
            cv2.putText(
                image,
                line,
                (18, start_y + row * line_height),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.42,
                TEXT_COLOR,
                1,
                cv2.LINE_AA,
            )
        return image

    def _render_matplotlib_panel(self, draw_fn) -> np.ndarray:
        fig = plt.figure(figsize=(5.6, 5.6), dpi=100)
        ax = fig.add_axes([0.03, 0.03, 0.94, 0.94])
        draw_fn(ax)
        fig.canvas.draw()
        rgb = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
        plt.close(fig)
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        return cv2.resize(bgr, (PANEL_SIZE, PANEL_SIZE), interpolation=cv2.INTER_AREA)

    def _update_panel_cache(self, *, source_only: bool = False) -> None:
        self.panel_images["source"] = self._render_source_panel_cv2(show_handles=True)
        if source_only:
            return
        self.panel_images["source_grid"] = self._render_matplotlib_panel(self.render_source_grid)
        self.panel_images["target"] = self._render_matplotlib_panel(self.render_target_panel)
        self.panel_images["warped"] = self._render_matplotlib_panel(self.render_warped_preview)

    def _compose_canvas(self) -> np.ndarray:
        canvas_h = PANEL_SIZE * 2 + PANEL_GAP + FOOTER_HEIGHT
        canvas_w = PANEL_SIZE * 2 + PANEL_GAP
        canvas = np.full((canvas_h, canvas_w, 3), PANEL_BG, dtype=np.uint8)

        positions = {
            "source": (0, 0),
            "source_grid": (PANEL_SIZE + PANEL_GAP, 0),
            "target": (0, PANEL_SIZE + PANEL_GAP),
            "warped": (PANEL_SIZE + PANEL_GAP, PANEL_SIZE + PANEL_GAP),
        }
        self.source_panel_origin = positions["source"]
        for name, (x0, y0) in positions.items():
            panel = self.panel_images[name]
            canvas[y0 : y0 + PANEL_SIZE, x0 : x0 + PANEL_SIZE] = panel
            cv2.rectangle(canvas, (x0, y0), (x0 + PANEL_SIZE - 1, y0 + PANEL_SIZE - 1), (70, 70, 70), 1)

        footer_y = PANEL_SIZE * 2 + PANEL_GAP
        canvas[footer_y:, :] = FOOTER_BG
        help_line = "Mouse: drag handles in top-left panel | Keys: s save only | v save + render | r reset | q quit"
        cv2.putText(canvas, help_line, (12, footer_y + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.48, TEXT_COLOR, 1, cv2.LINE_AA)
        cv2.putText(canvas, self.status_message[:145], (12, footer_y + 50), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (120, 220, 255), 1, cv2.LINE_AA)
        return canvas

    def _refresh_display(self, *, source_only: bool = False) -> None:
        self._update_panel_cache(source_only=source_only)
        self.canvas_image = self._compose_canvas()

    def _screen_to_source(self, x: int, y: int) -> tuple[float, float] | None:
        x0, y0 = self.source_panel_origin
        if not (x0 <= x < x0 + PANEL_SIZE and y0 <= y < y0 + PANEL_SIZE):
            return None
        source_x = (x - x0) / self.source_panel_scale_x
        source_y = (y - y0) / self.source_panel_scale_y
        return float(source_x), float(source_y)

    def _find_nearest_handle(self, x: float, y: float, threshold: float = 4.5) -> tuple[str, int] | None:
        best_name = None
        best_index = None
        best_distance = float("inf")
        query = np.array([x, y], dtype=float)
        for name, points in self.handle_points.items():
            pts = np.asarray(points, dtype=float)
            distances = np.linalg.norm(pts - query, axis=1)
            if len(distances) == 0:
                continue
            idx = int(np.argmin(distances))
            distance = float(distances[idx])
            if distance < best_distance:
                best_name = name
                best_index = idx
                best_distance = distance
        if best_name is None or best_distance > threshold:
            return None
        return best_name, int(best_index)

    def _update_active_handle(self, x: float, y: float) -> None:
        if self.active_handle is None:
            return
        contour_name, point_index = self.active_handle
        updated = clamp_point(np.array([x, y], dtype=float), self.source_shape[1], self.source_shape[0])
        self.handle_points[contour_name][point_index] = updated

    def _on_mouse(self, event: int, x: int, y: int, flags: int, _param) -> None:
        if cv2 is None:
            return
        source_xy = self._screen_to_source(x, y)
        if event == cv2.EVENT_LBUTTONDOWN and source_xy is not None:
            nearest = self._find_nearest_handle(*source_xy)
            if nearest is None:
                return
            self.active_handle = nearest
            contour_name, _ = nearest
            self.active_handle_backup = self.handle_points[contour_name].copy()
            self._update_active_handle(*source_xy)
            self._refresh_display(source_only=True)
            return

        if event == cv2.EVENT_MOUSEMOVE and self.active_handle is not None and source_xy is not None:
            self._update_active_handle(*source_xy)
            self._refresh_display(source_only=True)
            return

        if event == cv2.EVENT_LBUTTONUP and self.active_handle is not None:
            if source_xy is not None:
                self._update_active_handle(*source_xy)
            contour_name, _ = self.active_handle
            try:
                self._recompute_state()
                self.status_message = "Preview rebuilt. Keys: s save only, v save + render, r reset, q quit."
            except Exception as exc:
                if self.active_handle_backup is not None:
                    self.handle_points[contour_name] = self.active_handle_backup.copy()
                self._recompute_state()
                self.status_message = f"Invalid edit reverted: {exc}"
            finally:
                self.active_handle = None
                self.active_handle_backup = None
                self._refresh_display(source_only=False)

    def _handle_save(self, *, render_video: bool) -> None:
        action = "Saving annotation..." if not render_video else "Saving annotation and rendering sequence..."
        self.status_message = action
        self._refresh_display(source_only=False)
        result = self.save_bundle(
            render_video=render_video,
            max_output_frames=self.args.max_output_frames,
            output_mode=self.args.output_mode,
        )
        mode_text = "Saved annotation only." if not render_video else "Saved annotation and rendered sequence."
        self.status_message = f"{mode_text} Output dir: {self.output_dir}"
        self._refresh_display(source_only=False)
        print(json.dumps(result, indent=2, ensure_ascii=False))

    def _handle_reset(self) -> None:
        self.handle_points = self._build_handle_points(self.original_contours)
        self._recompute_state()
        self.status_message = "Reset to the initial projected annotation."
        self._refresh_display(source_only=False)

    def _run_cv2_editor(self) -> None:
        assert cv2 is not None
        self._refresh_display(source_only=False)
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WINDOW_NAME, min(1600, self.canvas_image.shape[1]), min(1200, self.canvas_image.shape[0]))
        cv2.setMouseCallback(WINDOW_NAME, self._on_mouse)

        try:
            while True:
                cv2.imshow(WINDOW_NAME, self.canvas_image)
                key = cv2.waitKey(20) & 0xFF
                if key == 255:
                    continue
                if key in (ord("q"), 27):
                    break
                if key == ord("r"):
                    self._handle_reset()
                elif key == ord("s"):
                    self._handle_save(render_video=False)
                elif key == ord("v"):
                    self._handle_save(render_video=True)
        finally:
            cv2.destroyWindow(WINDOW_NAME)

    def run(self) -> dict[str, object] | None:
        if self.args.no_gui:
            result = self.save_bundle(
                render_video=not self.args.skip_video_on_save,
                max_output_frames=self.args.max_output_frames,
                output_mode=self.args.output_mode,
            )
            print(json.dumps(result, indent=2, ensure_ascii=False))
            return result

        self._run_cv2_editor()
        return None


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if not args.no_gui and cv2 is None:
        raise SystemExit(
            "cv2 is not installed in this environment. Install opencv-python in the repo venv first, "
            "or use --no-gui for headless save/render only."
        )
    editor = SourceAnnotationEditor(args)
    editor.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
