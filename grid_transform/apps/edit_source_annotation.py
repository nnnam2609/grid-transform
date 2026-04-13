from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

try:
    import cv2
except ModuleNotFoundError:
    cv2 = None

from grid_transform.artspeech_video import (
    IntervalCursor,
    load_session_data,
    normalize_frame,
    resolve_default_dataset_root,
)
from grid_transform.config import DEFAULT_VTLN_DIR, PROJECT_DIR, TONGUE_COLOR, VT_SEG_CONTOURS_ROOT, VT_SEG_DATA_ROOT
from grid_transform.cv2_shared import (
    CONTOUR_COLORS,
    SOURCE_GRID_STYLE,
    TARGET_GRID_STYLE,
    WINDOW_MARGIN,
    clamp_point,
    hex_to_bgr,
    scaled_window_size,
    source_text_block,
)
from grid_transform.io import load_frame_npy, load_frame_vtln
from grid_transform.session_warp import run_session_warp_to_target
from grid_transform.session_warp_threaded import run_session_warp_to_target_threaded
from grid_transform.source_annotation import (
    LONG_CONTOUR_HANDLE_COUNTS,
    default_match_output_path,
    default_source_annotation_output_dir,
    frame_correlation,
    load_or_compute_reference_session_match,
    projected_reference_frame,
    project_reference_annotation_to_source,
)
from grid_transform.transfer import (
    DEFAULT_ARTICULATORS,
    build_two_step_transform,
    resolve_common_articulators,
    smooth_transformed_contours,
    transform_contours,
)
from grid_transform.vt import build_grid, visualize_grid
from grid_transform.vtln_annotation_sync import (
    annotation_shape_from_metadata,
    delete_temp_annotation_file,
    find_latest_source_annotation,
    promote_temp_annotation_to_vtln,
    resolve_vtln_data_dir,
    vtln_zip_path,
    write_source_annotation_to_vtln,
)
from grid_transform.warp import precompute_inverse_warp, warp_array_with_precomputed_inverse_warp


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
    parser.add_argument(
        "--reference-speaker",
        default="1640_P7_S2_F0829",
        help="VTLN reference speaker/image name. Defaults to the bundled 1640_P7_S2_F0829 example.",
    )
    parser.add_argument("--source-frame", type=int, help="Optional 1-based source frame override. Defaults to the best reference match.")
    parser.add_argument("--target-frame", type=int, default=143020, help="nnUNet target frame number.")
    parser.add_argument("--target-case", default="2008-003^01-1791/test", help="nnUNet target case relative path.")
    parser.add_argument("--dataset-root", type=Path, help="Optional explicit ArtSpeech dataset root.")
    parser.add_argument(
        "--vtln-dir",
        type=Path,
        default=DEFAULT_VTLN_DIR,
        help="Folder containing VTLN images and ROI zip files. Defaults to the bundled VTLN/data folder.",
    )
    parser.add_argument("--output-dir", type=Path, help="Optional explicit output directory.")
    parser.add_argument("--output-mode", choices=("both", "warped", "review"), default="both", help="Which sequence outputs to render.")
    parser.add_argument("--max-output-frames", type=int, default=0, help="Optional debug limit for the sequence render triggered on save.")
    parser.add_argument(
        "--render-workers",
        type=int,
        default=1,
        help="Number of threaded workers used for video rendering. Use 1 to keep the original sequential exporter.",
    )
    parser.add_argument(
        "--render-prefetch",
        type=int,
        default=0,
        help="Optional number of in-flight frames for the threaded renderer. Use 0 for the default heuristic.",
    )
    parser.add_argument(
        "--render-background",
        action="store_true",
        help="Save the annotation now and launch the video render as a detached background job.",
    )
    parser.add_argument("--skip-video-on-save", action="store_true", help="Headless mode only: save without rendering the output sequence.")
    parser.add_argument(
        "--window-mode",
        choices=("full-height", "fullscreen", "fixed"),
        default="full-height",
        help="Initial cv2 window mode. 'full-height' scales the canvas to the screen height, 'fullscreen' fills the screen, 'fixed' keeps the legacy size.",
    )
    parser.add_argument("--no-gui", action="store_true", help="Run the same initialization and save flow without opening the interactive window.")
    return parser.parse_args(argv)


class SourceAnnotationEditor:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.args.vtln_dir = resolve_vtln_data_dir(self.args.vtln_dir)
        self.dataset_root = args.dataset_root or resolve_default_dataset_root(args.artspeech_speaker)
        self.session_data = load_session_data(self.dataset_root, args.artspeech_speaker, args.session)

        self.reference_image, self.reference_contours = load_frame_vtln(args.reference_speaker, args.vtln_dir)
        self.reference_shape = tuple(int(value) for value in np.asarray(self.reference_image).shape[:2])
        self.projected_reference_frame = projected_reference_frame(
            self.reference_image,
            tuple(int(value) for value in self.session_data.images.shape[1:]),
        )
        self.match_payload = load_or_compute_reference_session_match(
            args.reference_speaker,
            self.reference_image,
            self.session_data,
            args.artspeech_speaker,
            args.session,
        )

        self.source_frame_index1 = int(args.source_frame or self.match_payload["best_match"]["frame1"])
        self.source_frame_index0 = self.source_frame_index1 - 1
        if self.source_frame_index0 < 0 or self.source_frame_index0 >= self.session_data.images.shape[0]:
            raise ValueError(
                f"source frame {self.source_frame_index1} is outside the session range 1..{self.session_data.images.shape[0]}"
            )
        self.source_frame = normalize_frame(
            self.session_data.images[self.source_frame_index0],
            self.session_data.frame_min,
            self.session_data.frame_max,
        )
        self.source_shape = tuple(int(value) for value in self.source_frame.shape[:2])
        self.source_snapshot = self._build_source_snapshot()

        self.output_dir = args.output_dir or default_source_annotation_output_dir(
            args.artspeech_speaker,
            args.session,
            self.source_frame_index1,
        )
        self.legacy_annotation_json_path = self.output_dir / "edited_annotation.json"
        self.vtln_zip_path = vtln_zip_path(args.vtln_dir, args.reference_speaker)

        self.target_image, self.target_contours = load_frame_npy(
            args.target_frame,
            VT_SEG_DATA_ROOT / args.target_case,
            VT_SEG_CONTOURS_ROOT / args.target_case,
        )
        self.target_frame = np.asarray(self.target_image)
        if self.target_frame.ndim == 3:
            self.target_frame = self.target_frame[..., 0]
        self.target_frame = np.clip(self.target_frame, 0, 255).astype(np.uint8)
        self.target_grid = build_grid(
            self.target_image,
            self.target_contours,
            n_vert=9,
            n_points=250,
            frame_number=args.target_frame,
        )

        self.loaded_saved_annotation = False
        self.saved_annotation_note = ""
        self._reconcile_temp_annotation_to_vtln()
        self.reference_image, self.reference_contours = load_frame_vtln(args.reference_speaker, args.vtln_dir)
        self.reference_shape = tuple(int(value) for value in np.asarray(self.reference_image).shape[:2])
        self.projected_reference_frame = projected_reference_frame(
            self.reference_image,
            tuple(int(value) for value in self.session_data.images.shape[1:]),
        )
        starting_contours = project_reference_annotation_to_source(
            self.reference_contours,
            self.reference_shape,
            self.source_shape,
        )
        self.original_contours = {
            name: np.asarray(points, dtype=float).copy()
            for name, points in starting_contours.items()
        }
        self.original_point_counts = {
            name: int(len(points))
            for name, points in self.original_contours.items()
        }
        self.handle_points = self._build_handle_points(starting_contours)

        self.current_contours: dict[str, np.ndarray] = {}
        self.source_grid = None
        self.articulators: list[str] = []
        self.warped_source_contours: dict[str, np.ndarray] = {}
        self.warped_source_frame = None
        self.valid_ratio = 0.0
        self._recompute_state()

        self.active_handle: tuple[str, int] | None = None
        self.active_handle_backup: np.ndarray | None = None
        if self.saved_annotation_note:
            self.status_message = f"{self.saved_annotation_note} Keys: s save, v save+render, r reset, q next, x exit all."
        else:
            self.status_message = "Ready. Drag handles in the source panel. Keys: s save, v save+render, r reset, q next, x exit all."
        self.panel_images: dict[str, np.ndarray] = {}
        self.canvas_image: np.ndarray | None = None
        self.source_panel_origin = (0, 0)
        self.source_panel_scale_x = PANEL_SIZE / self.source_shape[1]
        self.source_panel_scale_y = PANEL_SIZE / self.source_shape[0]

    def _build_source_snapshot(self) -> dict[str, object]:
        frame_times = (np.arange(self.session_data.images.shape[0], dtype=np.float64) + 0.5) / self.session_data.frame_rate
        current_time = float(frame_times[self.source_frame_index0])
        correlation = frame_correlation(self.projected_reference_frame, self.source_frame)
        word_cursor = IntervalCursor(self.session_data.tiers[0].intervals if len(self.session_data.tiers) >= 1 else [])
        phoneme_cursor = IntervalCursor(self.session_data.tiers[1].intervals if len(self.session_data.tiers) >= 2 else [])
        sentence_cursor = IntervalCursor(self.session_data.sentences)
        return {
            "frame1": self.source_frame_index1,
            "time_sec": current_time,
            "correlation": correlation,
            "word": word_cursor.at(current_time),
            "phoneme": phoneme_cursor.at(current_time),
            "sentence": sentence_cursor.at(current_time),
        }

    def _show_baseline_overlay(self) -> bool:
        return not self.loaded_saved_annotation

    def _source_overlay_legend_text(self) -> str:
        if self.loaded_saved_annotation:
            return "color = latest saved annotation"
        return "gray = baseline  |  color = current"

    def _reconcile_temp_annotation_to_vtln(self) -> None:
        latest_payload = find_latest_source_annotation(
            artspeech_speaker=self.args.artspeech_speaker,
            session=self.args.session,
            source_frame=self.source_frame_index1,
            current_source_shape=self.source_shape,
        )
        if latest_payload is None:
            return
        latest_path_raw = latest_payload.get("path")
        if latest_path_raw in (None, ""):
            return
        latest_path = Path(str(latest_path_raw))
        metadata = dict(latest_payload.get("metadata", {}))
        source_shape = annotation_shape_from_metadata(metadata, "source_shape", "reference_shape") or self.source_shape
        if promote_temp_annotation_to_vtln(
            path=latest_path,
            reference_name=self.args.reference_speaker,
            vtln_dir=self.args.vtln_dir,
            contours=latest_payload["contours"],
            source_shape=source_shape,
        ):
            self.saved_annotation_note = f"Promoted newer temp annotation into VTLN/data: {self.args.reference_speaker}"
        else:
            self.saved_annotation_note = "Deleted stale temp annotation and kept canonical VTLN/data annotation."

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
        source_grid = build_grid(
            self.source_frame,
            contours,
            n_vert=9,
            n_points=250,
            frame_number=self.source_frame_index1,
        )
        forward_transform = build_two_step_transform(source_grid, self.target_grid)
        inverse_transform = build_two_step_transform(self.target_grid, source_grid)
        articulators = resolve_common_articulators(
            self.target_contours,
            contours,
            defaults=DEFAULT_ARTICULATORS,
        )
        warped_contours = smooth_transformed_contours(
            transform_contours(contours, forward_transform["apply_two_step"], articulators)
        )
        source_x, source_y, valid_mask = precompute_inverse_warp(
            self.target_frame.shape,
            inverse_transform["apply_two_step"],
            self.source_shape,
        )
        warped_source_frame, _ = warp_array_with_precomputed_inverse_warp(
            self.source_frame,
            source_x,
            source_y,
            valid_mask,
        )

        self.current_contours = contours
        self.source_grid = source_grid
        self.articulators = articulators
        self.forward_transform = forward_transform
        self.inverse_transform = inverse_transform
        self.warped_source_contours = warped_contours
        self.warped_source_frame = warped_source_frame
        self.valid_ratio = float(valid_mask.mean())

    def _metadata_payload(self) -> dict[str, object]:
        return {
            "artspeech_speaker": self.args.artspeech_speaker,
            "session": self.args.session,
            "source_frame": self.source_frame_index1,
            "time_sec": self.source_snapshot["time_sec"],
            "reference_speaker": self.args.reference_speaker,
            "target_frame": self.args.target_frame,
            "target_case": self.args.target_case,
            "source_shape": list(self.source_shape),
            "reference_shape": list(self.reference_shape),
            "correlation": self.source_snapshot["correlation"],
            "word": self.source_snapshot["word"],
            "phoneme": self.source_snapshot["phoneme"],
            "sentence": self.source_snapshot["sentence"],
            "dataset_root": str(self.dataset_root),
            "match_json": str(
                default_match_output_path(
                    self.args.reference_speaker,
                    self.args.artspeech_speaker,
                    self.args.session,
                )
            ),
        }

    def render_source_annotation(self, ax, *, show_handles: bool) -> None:
        ax.clear()
        ax.imshow(self.source_frame, cmap="gray", vmin=0, vmax=255)
        ax.set_title(
            f"{self.args.artspeech_speaker}/{self.args.session} frame {self.source_frame_index1}\nEditable source annotation",
            fontsize=11,
            fontweight="bold",
        )
        ax.axis("off")
        ax.set_xlim(0, self.source_shape[1])
        ax.set_ylim(self.source_shape[0], 0)

        if self._show_baseline_overlay():
            for name, points in sorted(self.original_contours.items()):
                pts = np.asarray(points, dtype=float)
                ax.plot(pts[:, 0], pts[:, 1], linestyle="--", color="white", linewidth=1.2, alpha=0.5)

        for name, points in sorted(self.current_contours.items()):
            pts = np.asarray(points, dtype=float)
            color = CONTOUR_COLORS.get(name, "#00b4d8")
            linestyle = "--" if name in {"pharynx", "soft-palate-midline"} else "-"
            ax.plot(pts[:, 0], pts[:, 1], linestyle=linestyle, color=color, linewidth=2.0, alpha=0.95)
            anchor = pts[len(pts) // 2]
            ax.text(
                anchor[0],
                anchor[1],
                name,
                fontsize=7,
                color=CONTOUR_COLORS.get(name, "#ffffff"),
                bbox=dict(boxstyle="round,pad=0.15", fc="white", alpha=0.74, ec="none"),
            )

        ax.text(
            0.02,
            0.98,
            self._source_overlay_legend_text(),
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=8.5,
            family="monospace",
            color="white",
            bbox=dict(boxstyle="round,pad=0.25", fc="black", alpha=0.58, ec="none"),
        )

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

        ax.text(
            0.02,
            0.02,
            "\n".join(source_text_block(self.source_snapshot, self.args.reference_speaker)),
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=8.5,
            family="monospace",
            color="white",
            bbox=dict(boxstyle="round,pad=0.3", fc="black", alpha=0.72, ec="none"),
        )

    def render_source_grid(self, ax) -> None:
        ax.clear()
        visualize_grid(
            self.source_grid,
            ax=ax,
            show_contours=True,
            show_labels=True,
            style=SOURCE_GRID_STYLE,
        )
        ax.set_title(
            f"Source grid + notation\n{self.args.artspeech_speaker}/{self.args.session} frame {self.source_frame_index1}",
            fontsize=11,
            fontweight="bold",
        )

    def render_target_panel(self, ax) -> None:
        ax.clear()
        visualize_grid(
            self.target_grid,
            ax=ax,
            show_contours=True,
            show_labels=True,
            style=TARGET_GRID_STYLE,
        )
        ax.set_title(
            f"Target frame {self.args.target_frame}\nTarget contours + grid",
            fontsize=11,
            fontweight="bold",
        )

    def render_warped_preview(self, ax) -> None:
        ax.clear()
        ax.imshow(self.warped_source_frame, cmap="gray", vmin=0, vmax=255)
        ax.set_title("Warped source preview in target space", fontsize=11, fontweight="bold")
        ax.axis("off")
        ax.set_xlim(0, self.target_frame.shape[1])
        ax.set_ylim(self.target_frame.shape[0], 0)

        for name in self.articulators:
            target_pts = np.asarray(self.target_contours[name], dtype=float)
            source_pts = np.asarray(self.warped_source_contours[name], dtype=float)
            target_color = "#16a085"
            source_color = TONGUE_COLOR if name == "tongue" else "#ef476f"
            ax.plot(target_pts[:, 0], target_pts[:, 1], color=target_color, linewidth=2.0, alpha=0.92)
            ax.plot(source_pts[:, 0], source_pts[:, 1], "--", color=source_color, linewidth=2.0, alpha=0.92)

        ax.text(
            0.02,
            0.02,
            f"valid px: {self.valid_ratio:.4f}\narticulators: {', '.join(self.articulators)}",
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=8.5,
            family="monospace",
            color="white",
            bbox=dict(boxstyle="round,pad=0.3", fc="black", alpha=0.72, ec="none"),
        )

    def _save_preview_images(self) -> dict[str, str]:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        source_panel_path = self.output_dir / "source_full_head_annotation.png"
        source_grid_path = self.output_dir / "source_grid_annotation.png"
        warped_preview_path = self.output_dir / "source_to_target_preview.png"
        overview_path = self.output_dir / "editor_overview.png"

        fig, ax = plt.subplots(figsize=(7, 7))
        self.render_source_annotation(ax, show_handles=False)
        fig.savefig(source_panel_path, dpi=220, bbox_inches="tight")
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(7, 7))
        self.render_source_grid(ax)
        fig.savefig(source_grid_path, dpi=220, bbox_inches="tight")
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(7, 7))
        self.render_warped_preview(ax)
        fig.savefig(warped_preview_path, dpi=220, bbox_inches="tight")
        plt.close(fig)

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        self.render_source_annotation(axes[0, 0], show_handles=False)
        self.render_target_panel(axes[0, 1])
        self.render_source_grid(axes[1, 0])
        self.render_warped_preview(axes[1, 1])
        fig.tight_layout()
        fig.savefig(overview_path, dpi=220, bbox_inches="tight")
        plt.close(fig)

        return {
            "source_full_head_annotation": str(source_panel_path),
            "source_grid_annotation": str(source_grid_path),
            "source_to_target_preview": str(warped_preview_path),
            "editor_overview": str(overview_path),
        }

    def save_bundle(self, *, render_video: bool, max_output_frames: int, output_mode: str) -> dict[str, object]:
        metadata = self._metadata_payload()
        write_source_annotation_to_vtln(
            vtln_dir=self.args.vtln_dir,
            reference_name=self.args.reference_speaker,
            contours=self.current_contours,
            source_shape=self.source_shape,
        )
        delete_temp_annotation_file(self.legacy_annotation_json_path, stop_root=self.output_dir)
        preview_paths = self._save_preview_images()

        result: dict[str, object] = {
            "vtln_zip": str(self.vtln_zip_path),
            "preview_paths": preview_paths,
            "metadata": metadata,
        }

        if render_video:
            sequence_output_dir = self.output_dir / "sequence_warp"
            if self.args.render_background:
                warp_summary = self._launch_background_render(
                    sequence_output_dir=sequence_output_dir,
                    max_output_frames=max_output_frames,
                    output_mode=output_mode,
                )
            else:
                if int(self.args.render_workers) > 1:
                    warp_summary = run_session_warp_to_target_threaded(
                        annotation_speaker=self.args.reference_speaker,
                        source_annotation_json=None,
                        artspeech_speaker=self.args.artspeech_speaker,
                        session=self.args.session,
                        target_frame=self.args.target_frame,
                        target_case=self.args.target_case,
                        dataset_root=self.dataset_root,
                        vtln_dir=self.args.vtln_dir,
                        output_dir=sequence_output_dir,
                        max_frames=max_output_frames,
                        output_mode=output_mode,
                        workers=int(self.args.render_workers),
                        prefetch=int(self.args.render_prefetch),
                    )
                else:
                    warp_summary = run_session_warp_to_target(
                        annotation_speaker=self.args.reference_speaker,
                        source_annotation_json=None,
                        artspeech_speaker=self.args.artspeech_speaker,
                        session=self.args.session,
                        target_frame=self.args.target_frame,
                        target_case=self.args.target_case,
                        dataset_root=self.dataset_root,
                        vtln_dir=self.args.vtln_dir,
                        output_dir=sequence_output_dir,
                        max_frames=max_output_frames,
                        output_mode=output_mode,
                    )
            result["sequence_warp_summary"] = warp_summary

        summary_path = self.output_dir / "save_summary.json"
        summary_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
        result["save_summary_json"] = str(summary_path)
        return result

    def _launch_background_render(
        self,
        *,
        sequence_output_dir: Path,
        max_output_frames: int,
        output_mode: str,
    ) -> dict[str, object]:
        sequence_output_dir.mkdir(parents=True, exist_ok=True)
        use_threaded = int(self.args.render_workers) > 1
        module_name = (
            "grid_transform.apps.warp_artspeech_session_to_target_video_threaded"
            if use_threaded
            else "grid_transform.apps.warp_artspeech_session_to_target_video"
        )
        command = [
            sys.executable,
            "-m",
            module_name,
            "--annotation-speaker",
            self.args.reference_speaker,
            "--artspeech-speaker",
            self.args.artspeech_speaker,
            "--session",
            self.args.session,
            "--target-frame",
            str(self.args.target_frame),
            "--target-case",
            self.args.target_case,
            "--dataset-root",
            str(self.dataset_root),
            "--vtln-dir",
            str(self.args.vtln_dir),
            "--output-dir",
            str(sequence_output_dir),
            "--max-frames",
            str(max_output_frames),
            "--output-mode",
            output_mode,
        ]
        if use_threaded:
            command.extend(
                [
                    "--workers",
                    str(int(self.args.render_workers)),
                    "--prefetch",
                    str(int(self.args.render_prefetch)),
                ]
            )

        log_path = sequence_output_dir / "background_render.log"
        job_path = sequence_output_dir / "background_render_job.json"
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
            "mode": "background",
            "pid": int(process.pid),
            "module": module_name,
            "command": command,
            "log_path": str(log_path),
            "output_dir": str(sequence_output_dir),
            "render_workers": int(self.args.render_workers),
            "render_prefetch": int(self.args.render_prefetch),
        }
        job_path.write_text(json.dumps(job_payload, indent=2, ensure_ascii=False), encoding="utf-8")
        return {
            "background_render": job_payload,
            "job_json": str(job_path),
        }

    def _render_source_panel_cv2(self, *, show_handles: bool) -> np.ndarray:
        image = cv2.cvtColor(
            cv2.resize(self.source_frame, (PANEL_SIZE, PANEL_SIZE), interpolation=cv2.INTER_NEAREST),
            cv2.COLOR_GRAY2BGR,
        )

        if self._show_baseline_overlay():
            for _name, points in sorted(self.original_contours.items()):
                pts = np.asarray(points, dtype=float)
                scaled = np.column_stack([pts[:, 0] * self.source_panel_scale_x, pts[:, 1] * self.source_panel_scale_y])
                rounded = np.round(scaled).astype(np.int32).reshape(-1, 1, 2)
                cv2.polylines(image, [rounded], False, (220, 220, 220), 1, lineType=cv2.LINE_AA)

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
            self._source_overlay_legend_text(),
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
            "target": (PANEL_SIZE + PANEL_GAP, 0),
            "source_grid": (0, PANEL_SIZE + PANEL_GAP),
            "warped": (PANEL_SIZE + PANEL_GAP, PANEL_SIZE + PANEL_GAP),
        }
        self.source_panel_origin = positions["source"]
        for name, (x0, y0) in positions.items():
            panel = self.panel_images[name]
            canvas[y0 : y0 + PANEL_SIZE, x0 : x0 + PANEL_SIZE] = panel
            cv2.rectangle(canvas, (x0, y0), (x0 + PANEL_SIZE - 1, y0 + PANEL_SIZE - 1), (70, 70, 70), 1)

        footer_y = PANEL_SIZE * 2 + PANEL_GAP
        canvas[footer_y:, :] = FOOTER_BG
        help_line = "Mouse: drag handles | Keys: s save | v save+render | r reset | f fullscreen | q next | x exit all"
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
                self.status_message = "Preview rebuilt. Keys: s save, v save+render, r reset, q next, x exit all."
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
        if not render_video:
            mode_text = "Saved annotation only."
        elif self.args.render_background:
            mode_text = "Saved annotation and started background render."
        else:
            mode_text = "Saved annotation and rendered sequence."
        self.status_message = f"{mode_text} Output dir: {self.output_dir}"
        self._refresh_display(source_only=False)
        print(json.dumps(result, indent=2, ensure_ascii=False))

    def _handle_reset(self) -> None:
        self.handle_points = self._build_handle_points(self.original_contours)
        self._recompute_state()
        self.status_message = "Reset to the current VTLN/data annotation."
        self._refresh_display(source_only=False)

    def _window_mode_label(self, *, fullscreen: bool) -> str:
        if fullscreen:
            return "fullscreen"
        return "fixed" if self.args.window_mode == "fixed" else "full-height"

    def _apply_window_mode(self, *, fullscreen: bool | None = None) -> None:
        assert cv2 is not None
        if fullscreen is None:
            fullscreen = self.args.window_mode == "fullscreen"
        if fullscreen:
            cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
            return

        cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
        if self.args.window_mode == "fixed":
            window_width = min(1600, self.canvas_image.shape[1])
            window_height = min(1200, self.canvas_image.shape[0])
        else:
            window_width, window_height = scaled_window_size(self.canvas_image.shape[1], self.canvas_image.shape[0])
        cv2.resizeWindow(WINDOW_NAME, window_width, window_height)

    def _run_cv2_editor(self) -> str:
        assert cv2 is not None
        self._refresh_display(source_only=False)
        window_flags = cv2.WINDOW_NORMAL
        if hasattr(cv2, "WINDOW_KEEPRATIO"):
            window_flags |= cv2.WINDOW_KEEPRATIO
        cv2.namedWindow(WINDOW_NAME, window_flags)
        self._apply_window_mode()
        cv2.setMouseCallback(WINDOW_NAME, self._on_mouse)
        action = "next"
        fullscreen_enabled = self.args.window_mode == "fullscreen"

        try:
            while True:
                cv2.imshow(WINDOW_NAME, self.canvas_image)
                key = cv2.waitKey(20) & 0xFF
                if key == 255:
                    continue
                if key in (ord("q"), 27):
                    break
                if key in (ord("x"), ord("X")):
                    action = "exit_all"
                    break
                if key == ord("r"):
                    self._handle_reset()
                elif key in (ord("f"), ord("F")):
                    fullscreen_enabled = not fullscreen_enabled
                    self._apply_window_mode(fullscreen=fullscreen_enabled)
                    mode_label = self._window_mode_label(fullscreen=fullscreen_enabled)
                    self.status_message = f"Window mode: {mode_label}. Keys: s save, v save+render, r reset, f toggle fullscreen."
                    self._refresh_display(source_only=False)
                elif key == ord("s"):
                    self._handle_save(render_video=False)
                elif key == ord("v"):
                    self._handle_save(render_video=True)
        finally:
            cv2.destroyWindow(WINDOW_NAME)
        return action

    def run(self) -> dict[str, object] | None:
        if self.args.no_gui:
            result = self.save_bundle(
                render_video=not self.args.skip_video_on_save,
                max_output_frames=self.args.max_output_frames,
                output_mode=self.args.output_mode,
            )
            result["action"] = "next"
            print(json.dumps(result, indent=2, ensure_ascii=False))
            return result

        return {"action": self._run_cv2_editor()}


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
