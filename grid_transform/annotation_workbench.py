from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from grid_transform.artspeech_video import (
    IntervalCursor,
    load_session_data,
    normalize_frame,
    resolve_default_dataset_root,
)
from grid_transform.config import DEFAULT_VTNL_DIR, TONGUE_COLOR, VT_SEG_CONTOURS_ROOT, VT_SEG_DATA_ROOT
from grid_transform.io import load_frame_npy, load_frame_vtnl
from grid_transform.session_warp import run_session_warp_to_target
from grid_transform.source_annotation import (
    controls_to_grid_constraints,
    controls_to_grid_contours,
    default_match_output_path,
    default_source_annotation_output_dir,
    frame_correlation,
    load_or_compute_reference_session_match,
    load_source_annotation_json,
    normalize_grid_controls,
    projected_reference_frame,
    project_reference_annotation_to_source,
    save_source_annotation_json,
)
from grid_transform.transfer import (
    DEFAULT_ARTICULATORS,
    build_two_step_transform,
    resolve_common_articulators,
    smooth_transformed_contours,
    transform_contours,
)
from grid_transform.vt import build_grid, visualize_grid
from grid_transform.warp import precompute_inverse_warp, warp_array_with_precomputed_inverse_warp


CONTOUR_COLORS = {
    "c1": "#457b9d",
    "c2": "#4d7ea8",
    "c3": "#5c96bc",
    "c4": "#7aa6c2",
    "c5": "#90b8d8",
    "c6": "#b8d8f2",
    "incisior-hard-palate": "#ef476f",
    "mandible-incisior": "#f4a261",
    "pharynx": "#118ab2",
    "soft-palate": "#fb8500",
    "soft-palate-midline": "#8338ec",
    "tongue": TONGUE_COLOR,
}
SOURCE_GRID_STYLE = {
    "horiz_color": "#3a86ff",
    "vert_color": "#ffbe0b",
    "vt_color": "#8338ec",
    "spine_color": "#2a9d8f",
    "guide_color": "#fb5607",
    "i_color": "#8338ec",
    "c_color": "#2a9d8f",
    "p1_color": "#fb5607",
    "m1_color": "#8338ec",
    "l6_color": "#80ed99",
    "interp_color": "#3a86ff",
    "mandible_color": "#e76f51",
    "tongue_color": TONGUE_COLOR,
}
TARGET_GRID_STYLE = {
    "horiz_color": "#20c997",
    "vert_color": "#ffd166",
    "vt_color": "#ef476f",
    "spine_color": "#118ab2",
    "guide_color": "#ff006e",
    "i_color": "#ef476f",
    "c_color": "#118ab2",
    "p1_color": "#ff006e",
    "m1_color": "#ef476f",
    "l6_color": "#06d6a0",
    "interp_color": "#20c997",
    "mandible_color": "#f4a261",
    "tongue_color": TONGUE_COLOR,
}


@dataclass(slots=True)
class AnnotationWorkbenchConfig:
    artspeech_speaker: str = "P7"
    session: str = "S2"
    reference_speaker: str | None = "1640_s10_0829"
    source_frame: int | None = None
    target_frame: int = 143020
    target_case: str = "2008-003^01-1791/test"
    dataset_root: Path | None = None
    vtnl_dir: Path = DEFAULT_VTNL_DIR
    output_dir: Path | None = None


def source_text_block(snapshot: dict[str, object], reference_speaker: str | None) -> list[str]:
    sentence = " ".join(str(snapshot["sentence"]).split()) if snapshot.get("sentence") else "-"
    if len(sentence) > 56:
        sentence = sentence[:53] + "..."
    return [
        f"ref: {reference_speaker or '-'}",
        f"frame: {snapshot['frame1']}  t={snapshot['time_sec']:.4f}s",
        f"corr: {snapshot['correlation']:.4f}",
        f"word: {snapshot['word'] or '-'}  phoneme: {snapshot['phoneme'] or '-'}",
        f"sent: {sentence}",
    ]


def _target_frame_uint8(image) -> np.ndarray:
    frame = np.asarray(image)
    if frame.ndim == 3:
        frame = frame[..., 0]
    return np.clip(frame, 0, 255).astype(np.uint8)


class AnnotationWorkbenchState:
    def __init__(
        self,
        *,
        config: AnnotationWorkbenchConfig,
        dataset_root: Path,
        session_data,
        reference_shape: tuple[int, int],
        source_frame_index1: int,
        source_frame: np.ndarray,
        source_snapshot: dict[str, object],
        output_dir: Path,
        annotation_json_path: Path,
        target_frame_image: np.ndarray,
        target_contours: dict[str, np.ndarray],
        target_grid,
        original_contours: dict[str, np.ndarray],
        current_contours: dict[str, np.ndarray],
        grid_controls: dict[str, object] | None = None,
        match_payload: dict[str, object] | None = None,
    ) -> None:
        self.config = config
        self.dataset_root = Path(dataset_root)
        self.session_data = session_data
        self.reference_shape = tuple(int(value) for value in reference_shape)
        self.source_frame_index1 = int(source_frame_index1)
        self.source_frame_index0 = self.source_frame_index1 - 1
        self.source_frame = np.asarray(source_frame, dtype=np.uint8)
        self.source_shape = tuple(int(value) for value in self.source_frame.shape[:2])
        self.source_snapshot = dict(source_snapshot)
        self.output_dir = Path(output_dir)
        self.annotation_json_path = Path(annotation_json_path)
        self.target_frame = np.asarray(target_frame_image, dtype=np.uint8)
        self.target_contours = {
            name: np.asarray(points, dtype=float)
            for name, points in target_contours.items()
        }
        self.target_grid = target_grid
        self.original_contours = {
            name: np.asarray(points, dtype=float)
            for name, points in original_contours.items()
        }
        self.current_contours: dict[str, np.ndarray] = {}
        self.grid_controls = normalize_grid_controls(current_contours, grid_controls)
        self.match_payload = match_payload

        self.source_grid = None
        self.articulators: list[str] = []
        self.forward_transform: dict[str, object] = {}
        self.inverse_transform: dict[str, object] = {}
        self.warped_source_contours: dict[str, np.ndarray] = {}
        self.warped_source_frame: np.ndarray | None = None
        self.valid_ratio = 0.0
        self.set_contours(current_contours, grid_controls=self.grid_controls)

    @classmethod
    def from_projection(cls, config: AnnotationWorkbenchConfig) -> AnnotationWorkbenchState:
        dataset_root = Path(config.dataset_root) if config.dataset_root is not None else resolve_default_dataset_root(config.artspeech_speaker)
        session_data = load_session_data(dataset_root, config.artspeech_speaker, config.session)

        if not config.reference_speaker:
            raise ValueError("reference_speaker is required when initializing from projection.")

        reference_image, reference_contours = load_frame_vtnl(config.reference_speaker, config.vtnl_dir)
        reference_shape = tuple(int(value) for value in np.asarray(reference_image).shape[:2])
        projected_reference = projected_reference_frame(reference_image, tuple(int(value) for value in session_data.images.shape[1:]))
        match_payload = load_or_compute_reference_session_match(
            config.reference_speaker,
            reference_image,
            session_data,
            config.artspeech_speaker,
            config.session,
            output_path=default_match_output_path(config.reference_speaker, config.artspeech_speaker, config.session),
        )

        source_frame_index1 = int(config.source_frame or match_payload["best_match"]["frame1"])
        source_frame_index0 = source_frame_index1 - 1
        if source_frame_index0 < 0 or source_frame_index0 >= session_data.images.shape[0]:
            raise ValueError(
                f"source frame {source_frame_index1} is outside the session range 1..{session_data.images.shape[0]}"
            )

        source_frame = normalize_frame(
            session_data.images[source_frame_index0],
            session_data.frame_min,
            session_data.frame_max,
        )
        source_snapshot = cls._build_source_snapshot(
            session_data=session_data,
            source_frame_index0=source_frame_index0,
            source_frame=source_frame,
            projected_reference=projected_reference,
        )
        output_dir = config.output_dir or default_source_annotation_output_dir(
            config.artspeech_speaker,
            config.session,
            source_frame_index1,
        )
        annotation_json_path = Path(output_dir) / "edited_annotation.json"

        target_image, target_contours = load_frame_npy(
            config.target_frame,
            VT_SEG_DATA_ROOT / config.target_case,
            VT_SEG_CONTOURS_ROOT / config.target_case,
        )
        target_frame_image = _target_frame_uint8(target_image)
        target_grid = build_grid(
            target_image,
            target_contours,
            n_vert=9,
            n_points=250,
            frame_number=config.target_frame,
        )

        original_contours = project_reference_annotation_to_source(
            reference_contours,
            reference_shape,
            tuple(int(value) for value in source_frame.shape[:2]),
        )
        grid_controls = normalize_grid_controls(original_contours)

        return cls(
            config=config,
            dataset_root=dataset_root,
            session_data=session_data,
            reference_shape=reference_shape,
            source_frame_index1=source_frame_index1,
            source_frame=source_frame,
            source_snapshot=source_snapshot,
            output_dir=Path(output_dir),
            annotation_json_path=annotation_json_path,
            target_frame_image=target_frame_image,
            target_contours=target_contours,
            target_grid=target_grid,
            original_contours=original_contours,
            current_contours=original_contours,
            grid_controls=grid_controls,
            match_payload=match_payload,
        )

    @classmethod
    def from_saved_annotation(
        cls,
        annotation_json: Path | str,
        *,
        dataset_root: Path | None = None,
        vtnl_dir: Path = DEFAULT_VTNL_DIR,
        output_dir: Path | None = None,
        target_frame: int | None = None,
        target_case: str | None = None,
    ) -> AnnotationWorkbenchState:
        payload = load_source_annotation_json(annotation_json)
        metadata = payload["metadata"]
        annotation_json_path = Path(annotation_json)

        artspeech_speaker = str(metadata.get("artspeech_speaker") or "P7")
        session = str(metadata.get("session") or "S2")
        reference_speaker_value = metadata.get("reference_speaker")
        reference_speaker = str(reference_speaker_value) if reference_speaker_value else None
        source_frame_index1 = int(metadata.get("source_frame", 0))
        if source_frame_index1 <= 0:
            raise ValueError("Saved annotation metadata must include a positive source_frame.")

        resolved_dataset_root = Path(dataset_root or metadata.get("dataset_root") or resolve_default_dataset_root(artspeech_speaker))
        resolved_target_frame = int(target_frame or metadata.get("target_frame") or 143020)
        resolved_target_case = str(target_case or metadata.get("target_case") or "2008-003^01-1791/test")
        resolved_output_dir = Path(output_dir) if output_dir is not None else annotation_json_path.parent
        config = AnnotationWorkbenchConfig(
            artspeech_speaker=artspeech_speaker,
            session=session,
            reference_speaker=reference_speaker,
            source_frame=source_frame_index1,
            target_frame=resolved_target_frame,
            target_case=resolved_target_case,
            dataset_root=resolved_dataset_root,
            vtnl_dir=vtnl_dir,
            output_dir=resolved_output_dir,
        )

        session_data = load_session_data(resolved_dataset_root, artspeech_speaker, session)
        source_frame_index0 = source_frame_index1 - 1
        if source_frame_index0 < 0 or source_frame_index0 >= session_data.images.shape[0]:
            raise ValueError(
                f"Saved annotation source_frame {source_frame_index1} exceeds session frame count {session_data.images.shape[0]}."
            )

        source_frame = normalize_frame(
            session_data.images[source_frame_index0],
            session_data.frame_min,
            session_data.frame_max,
        )

        reference_shape = tuple(int(value) for value in metadata.get("reference_shape", metadata.get("source_shape", source_frame.shape[:2]))[:2])
        projected_reference = None
        if reference_speaker:
            try:
                reference_image, _ = load_frame_vtnl(reference_speaker, vtnl_dir)
                reference_shape = tuple(int(value) for value in np.asarray(reference_image).shape[:2])
                projected_reference = projected_reference_frame(reference_image, tuple(int(value) for value in source_frame.shape[:2]))
            except FileNotFoundError:
                projected_reference = None

        source_snapshot = cls._build_source_snapshot(
            session_data=session_data,
            source_frame_index0=source_frame_index0,
            source_frame=source_frame,
            projected_reference=projected_reference,
        )

        target_image, target_contours = load_frame_npy(
            resolved_target_frame,
            VT_SEG_DATA_ROOT / resolved_target_case,
            VT_SEG_CONTOURS_ROOT / resolved_target_case,
        )
        target_frame_image = _target_frame_uint8(target_image)
        target_grid = build_grid(
            target_image,
            target_contours,
            n_vert=9,
            n_points=250,
            frame_number=resolved_target_frame,
        )

        current_contours = payload["contours"]
        grid_controls = metadata.get("grid_controls")
        return cls(
            config=config,
            dataset_root=resolved_dataset_root,
            session_data=session_data,
            reference_shape=reference_shape,
            source_frame_index1=source_frame_index1,
            source_frame=source_frame,
            source_snapshot=source_snapshot,
            output_dir=resolved_output_dir,
            annotation_json_path=resolved_output_dir / annotation_json_path.name,
            target_frame_image=target_frame_image,
            target_contours=target_contours,
            target_grid=target_grid,
            original_contours=current_contours,
            current_contours=current_contours,
            grid_controls=grid_controls,
            match_payload=None,
        )

    @staticmethod
    def _build_source_snapshot(
        *,
        session_data,
        source_frame_index0: int,
        source_frame: np.ndarray,
        projected_reference: np.ndarray | None,
    ) -> dict[str, object]:
        frame_times = (np.arange(session_data.images.shape[0], dtype=np.float64) + 0.5) / session_data.frame_rate
        current_time = float(frame_times[source_frame_index0])
        correlation = frame_correlation(projected_reference, source_frame) if projected_reference is not None else 0.0
        word_cursor = IntervalCursor(session_data.tiers[0].intervals if len(session_data.tiers) >= 1 else [])
        phoneme_cursor = IntervalCursor(session_data.tiers[1].intervals if len(session_data.tiers) >= 2 else [])
        sentence_cursor = IntervalCursor(session_data.sentences)
        return {
            "frame1": int(source_frame_index0 + 1),
            "time_sec": current_time,
            "correlation": float(correlation),
            "word": word_cursor.at(current_time),
            "phoneme": phoneme_cursor.at(current_time),
            "sentence": sentence_cursor.at(current_time),
        }

    def _recompute_state(self) -> None:
        grid_contours = controls_to_grid_contours(self.current_contours, self.grid_controls)
        grid_constraints = controls_to_grid_constraints(self.current_contours, self.grid_controls)
        source_grid = build_grid(
            self.source_frame,
            grid_contours,
            n_vert=9,
            n_points=250,
            frame_number=self.source_frame_index1,
            grid_constraints=grid_constraints,
        )
        forward_transform = build_two_step_transform(source_grid, self.target_grid)
        inverse_transform = build_two_step_transform(self.target_grid, source_grid)
        articulators = resolve_common_articulators(
            self.target_contours,
            self.current_contours,
            defaults=DEFAULT_ARTICULATORS,
        )
        warped_contours = smooth_transformed_contours(
            transform_contours(self.current_contours, forward_transform["apply_two_step"], articulators)
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

        self.source_grid = source_grid
        self.articulators = articulators
        self.forward_transform = forward_transform
        self.inverse_transform = inverse_transform
        self.warped_source_contours = warped_contours
        self.warped_source_frame = warped_source_frame
        self.valid_ratio = float(valid_mask.mean())

    def set_contours(
        self,
        contours: dict[str, np.ndarray],
        *,
        grid_controls: dict[str, object] | None = None,
    ) -> None:
        self.current_contours = {
            name: np.asarray(points, dtype=float)
            for name, points in contours.items()
        }
        self.grid_controls = normalize_grid_controls(
            self.current_contours,
            self.grid_controls if grid_controls is None else grid_controls,
        )
        self._recompute_state()

    def set_grid_controls(self, grid_controls: dict[str, object]) -> None:
        self.grid_controls = normalize_grid_controls(self.current_contours, grid_controls)
        self._recompute_state()

    def metadata_payload(self) -> dict[str, object]:
        return {
            "artspeech_speaker": self.config.artspeech_speaker,
            "session": self.config.session,
            "source_frame": self.source_frame_index1,
            "time_sec": self.source_snapshot["time_sec"],
            "reference_speaker": self.config.reference_speaker,
            "target_frame": self.config.target_frame,
            "target_case": self.config.target_case,
            "source_shape": list(self.source_shape),
            "reference_shape": list(self.reference_shape),
            "correlation": self.source_snapshot["correlation"],
            "word": self.source_snapshot["word"],
            "phoneme": self.source_snapshot["phoneme"],
            "sentence": self.source_snapshot["sentence"],
            "dataset_root": str(self.dataset_root),
            "grid_controls": normalize_grid_controls(self.current_contours, self.grid_controls),
            "match_json": (
                str(default_match_output_path(self.config.reference_speaker, self.config.artspeech_speaker, self.config.session))
                if self.config.reference_speaker
                else ""
            ),
        }

    def render_source_annotation(self, ax, *, visible_contours: set[str] | None = None) -> None:
        visible = visible_contours or set(self.current_contours)
        ax.clear()
        ax.imshow(self.source_frame, cmap="gray", vmin=0, vmax=255)
        ax.set_title(
            f"{self.config.artspeech_speaker}/{self.config.session} frame {self.source_frame_index1}\nSource annotation",
            fontsize=11,
            fontweight="bold",
        )
        ax.axis("off")
        ax.set_xlim(0, self.source_shape[1])
        ax.set_ylim(self.source_shape[0], 0)

        for name, points in sorted(self.current_contours.items()):
            if name not in visible:
                continue
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
                color=color,
                bbox=dict(boxstyle="round,pad=0.15", fc="white", alpha=0.74, ec="none"),
            )

        ax.text(
            0.02,
            0.02,
            "\n".join(source_text_block(self.source_snapshot, self.config.reference_speaker)),
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
            f"Source grid\n{self.config.artspeech_speaker}/{self.config.session} frame {self.source_frame_index1}",
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
            f"Target frame {self.config.target_frame}\nTarget contours + grid",
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

    def _render_panel_image(self, draw_fn) -> np.ndarray:
        fig = plt.figure(figsize=(6.8, 6.8), dpi=120)
        ax = fig.add_axes([0.03, 0.03, 0.94, 0.94])
        draw_fn(ax)
        fig.canvas.draw()
        rgb = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
        plt.close(fig)
        return rgb

    def preview_images(self, *, visible_contours: set[str] | None = None) -> dict[str, np.ndarray]:
        return {
            "source": self._render_panel_image(lambda ax: self.render_source_annotation(ax, visible_contours=visible_contours)),
            "source_grid": self._render_panel_image(self.render_source_grid),
            "target": self._render_panel_image(self.render_target_panel),
            "warped": self._render_panel_image(self.render_warped_preview),
        }

    def _save_preview_images(self) -> dict[str, str]:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        source_panel_path = self.output_dir / "source_full_head_annotation.png"
        source_grid_path = self.output_dir / "source_grid_annotation.png"
        warped_preview_path = self.output_dir / "source_to_target_preview.png"
        overview_path = self.output_dir / "editor_overview.png"

        fig, ax = plt.subplots(figsize=(7, 7))
        self.render_source_annotation(ax)
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
        self.render_source_annotation(axes[0, 0])
        self.render_source_grid(axes[0, 1])
        self.render_target_panel(axes[1, 0])
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
        metadata = self.metadata_payload()
        payload = save_source_annotation_json(self.annotation_json_path, metadata, self.current_contours)
        preview_paths = self._save_preview_images()

        result: dict[str, object] = {
            "edited_annotation_json": str(self.annotation_json_path),
            "preview_paths": preview_paths,
            "metadata": payload["metadata"],
        }

        if render_video:
            sequence_output_dir = self.output_dir / "sequence_warp"
            warp_summary = run_session_warp_to_target(
                annotation_speaker=self.config.reference_speaker or "",
                source_annotation_json=self.annotation_json_path,
                artspeech_speaker=self.config.artspeech_speaker,
                session=self.config.session,
                target_frame=self.config.target_frame,
                target_case=self.config.target_case,
                dataset_root=self.dataset_root,
                vtnl_dir=self.config.vtnl_dir,
                output_dir=sequence_output_dir,
                max_frames=max_output_frames,
                output_mode=output_mode,
            )
            result["sequence_warp_summary"] = warp_summary

        summary_path = self.output_dir / "save_summary.json"
        summary_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
        result["save_summary_json"] = str(summary_path)
        return result
