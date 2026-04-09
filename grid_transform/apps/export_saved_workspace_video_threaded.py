from __future__ import annotations

import argparse
import json
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path

import imageio.v2 as imageio
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from grid_transform.annotation_to_grid_workflow import (
    apply_transform_from_spec,
    load_json,
    load_source_context,
    load_target_context,
    payload_to_workspace_selection,
    step_file_paths,
)
from grid_transform.artspeech_video import (
    IntervalCursor,
    downsample_waveform,
    load_session_data,
    normalize_frame,
)
from grid_transform.config import TONGUE_COLOR
from grid_transform.cv2_annotation_app_config import (
    RENDER_PREFETCH_RANGE,
    RENDER_WORKERS_RANGE,
    argparse_bounded_int,
    validate_render_settings,
)
from grid_transform.transfer import (
    DEFAULT_ARTICULATORS,
    resolve_common_articulators,
    smooth_transformed_contours,
    transform_contours,
)
from grid_transform.warp import precompute_inverse_warp, warp_array_with_precomputed_inverse_warp


_TARGET_COLOR = "#16a085"
_ASSUMED_COLOR = "#ef476f"


def _format_label_block(word: str, phoneme: str, sentence: str) -> str:
    sentence = " ".join(sentence.split()) if sentence else "-"
    if len(sentence) > 80:
        sentence = sentence[:77] + "..."
    return "\n".join([
        f"word: {word or '-'}",
        f"phoneme: {phoneme or '-'}",
        f"sentence: {sentence}",
    ])


def _gray_to_rgb(frame: np.ndarray) -> np.ndarray:
    return np.repeat(frame[..., None], 3, axis=2)


def _draw_contours(
    ax,
    contours: dict[str, np.ndarray],
    names: list[str],
    *,
    base_color: str,
    tongue_color: str,
    linestyle: str,
    linewidth: float = 1.6,
    alpha: float = 0.9,
) -> None:
    for name in names:
        pts = np.asarray(contours.get(name, []), dtype=float)
        if len(pts) < 2:
            continue
        color = tongue_color if name == "tongue" else base_color
        lw = 2.0 if name == "tongue" else linewidth
        ax.plot(pts[:, 0], pts[:, 1], linestyle=linestyle, color=color, linewidth=lw, alpha=alpha)


def _make_review_figure(
    first_source_frame: np.ndarray,
    target_frame: np.ndarray,
    first_warped_frame: np.ndarray,
    waveform_time: np.ndarray,
    waveform_values: np.ndarray,
    *,
    source_title: str,
    target_title: str,
    assumed_label: str,
) -> tuple:
    fig = plt.figure(figsize=(12.8, 10.08), dpi=100)
    grid = fig.add_gridspec(3, 2, height_ratios=[2.6, 2.6, 1.2], hspace=0.12, wspace=0.06)
    ax_raw = fig.add_subplot(grid[0, 0])
    ax_target = fig.add_subplot(grid[0, 1])
    ax_warped = fig.add_subplot(grid[1, :])
    ax_wave = fig.add_subplot(grid[2, :])
    for ax in (ax_raw, ax_target, ax_warped):
        ax.set_axis_off()
    raw_artist = ax_raw.imshow(first_source_frame, cmap="gray", vmin=0, vmax=255)
    target_artist = ax_target.imshow(target_frame, cmap="gray", vmin=0, vmax=255)
    warped_artist = ax_warped.imshow(first_warped_frame, cmap="gray", vmin=0, vmax=255)
    ax_raw.set_title(source_title, fontsize=10, fontweight="bold")
    ax_target.set_title(target_title, fontsize=10, fontweight="bold")
    ax_warped.set_title(f"Warped source in target space", fontsize=10, fontweight="bold")
    # Crop ax_warped to the valid (non-black) region of the first warped frame.
    _rows = np.any(first_warped_frame > 0, axis=1)
    _cols = np.any(first_warped_frame > 0, axis=0)
    if _rows.any() and _cols.any():
        _r0, _r1 = int(np.where(_rows)[0][0]), int(np.where(_rows)[0][-1])
        _c0, _c1 = int(np.where(_cols)[0][0]), int(np.where(_cols)[0][-1])
        _m = 8
        ax_warped.set_xlim(max(0, _c0 - _m) - 0.5, min(first_warped_frame.shape[1] - 1, _c1 + _m) + 0.5)
        ax_warped.set_ylim(min(first_warped_frame.shape[0] - 1, _r1 + _m) + 0.5, max(0, _r0 - _m) - 0.5)
    frame_text = ax_raw.text(0.02, 0.97, "", color="yellow", fontsize=10, va="top", transform=ax_raw.transAxes)
    label_text = ax_raw.text(
        0.02, 0.02, "", color="white", fontsize=8, family="monospace", va="bottom",
        transform=ax_raw.transAxes,
        bbox=dict(boxstyle="round,pad=0.3", fc="black", alpha=0.72, ec="none"),
    )
    valid_text = ax_warped.text(
        0.02, 0.02, "", color="white", fontsize=8, family="monospace", va="bottom",
        transform=ax_warped.transAxes,
        bbox=dict(boxstyle="round,pad=0.3", fc="black", alpha=0.72, ec="none"),
    )
    ax_wave.plot(waveform_time, waveform_values, color="black", linewidth=0.8)
    time_cursor = ax_wave.axvline(0.0, color="red", linewidth=1.2)
    ax_wave.set_ylim(-1.05, 1.05)
    ax_wave.set_ylabel("Amp.")
    ax_wave.set_xlabel("Time (s)")
    wave_title = ax_wave.set_title("", fontsize=10)
    artists = {
        "raw": raw_artist, "target": target_artist, "warped": warped_artist,
        "frame_text": frame_text, "label_text": label_text, "valid_text": valid_text,
        "time_cursor": time_cursor, "wave_title": wave_title, "ax_wave": ax_wave,
    }
    axes = {"raw": ax_raw, "target": ax_target, "warped": ax_warped}
    return fig, axes, artists


def _draw_review_frame(
    fig: plt.Figure,
    axes: dict,
    artists: dict,
    *,
    source_frame: np.ndarray,
    target_frame: np.ndarray,
    warped_frame: np.ndarray,
    source_space_contours: dict[str, np.ndarray],
    target_contours: dict[str, np.ndarray],
    target_space_ref_contours: dict[str, np.ndarray],
    articulators: list[str],
    frame_index: int,
    current_time: float,
    audio_duration: float,
    word: str,
    phoneme: str,
    sentence: str,
    valid_ratio: float,
) -> np.ndarray:
    artists["raw"].set_data(source_frame)
    artists["target"].set_data(target_frame)
    artists["warped"].set_data(warped_frame)
    for key in ("raw", "target"):
        for line in list(axes[key].lines):
            line.remove()
    _draw_contours(axes["raw"], source_space_contours, articulators, base_color=_ASSUMED_COLOR, tongue_color=TONGUE_COLOR, linestyle="--")
    _draw_contours(axes["target"], target_contours, articulators, base_color=_TARGET_COLOR, tongue_color=_TARGET_COLOR, linestyle="-")
    artists["frame_text"].set_text(f"frame {frame_index}")
    artists["label_text"].set_text(_format_label_block(word, phoneme, sentence))
    artists["valid_text"].set_text(f"valid px: {valid_ratio:.4f}")
    artists["time_cursor"].set_xdata([current_time, current_time])
    artists["wave_title"].set_text(" ".join(sentence.split()) if sentence else "")
    window = 1.0
    xmin = max(0.0, current_time - window)
    xmax = min(audio_duration, current_time + window)
    if xmax - xmin < 0.2:
        xmax = min(audio_duration, xmin + 0.2)
    artists["ax_wave"].set_xlim(xmin, xmax)
    fig.canvas.draw()
    return np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Background threaded export for a saved annotation-to-grid workspace.")
    parser.add_argument("--workspace-dir", type=Path, required=True)
    parser.add_argument(
        "--workers",
        type=argparse_bounded_int(
            "render_workers",
            min_value=RENDER_WORKERS_RANGE[0],
            max_value=RENDER_WORKERS_RANGE[1],
        ),
        default=8,
    )
    parser.add_argument(
        "--prefetch",
        type=argparse_bounded_int(
            "render_prefetch",
            min_value=RENDER_PREFETCH_RANGE[0],
            max_value=RENDER_PREFETCH_RANGE[1],
        ),
        default=0,
    )
    parser.add_argument("--max-frames", type=argparse_bounded_int("max_frames", min_value=0), default=0)
    parser.add_argument(
        "--output-mode",
        choices=("both", "warped", "review"),
        default="both",
        help="Whether to write the clean warped video, the review video, or both.",
    )
    return parser.parse_args(argv)


def _resolve_prefetch(workers: int, prefetch: int) -> int:
    if prefetch > 0:
        return prefetch
    return max(2, workers * 3)


def _render_single_frame(index: int, *, images: np.ndarray, frame_min: float, frame_max: float, source_x: np.ndarray, source_y: np.ndarray, valid_mask: np.ndarray) -> tuple[int, np.ndarray]:
    frame = normalize_frame(images[index], frame_min, frame_max)
    warped_frame, _ = warp_array_with_precomputed_inverse_warp(frame, source_x, source_y, valid_mask)
    return index, warped_frame


def run_workspace_export(*, workspace_dir: Path, workers: int, prefetch: int, max_frames: int, output_mode: str = "both") -> dict[str, object]:
    workers, prefetch = validate_render_settings(
        workers=workers,
        prefetch=prefetch,
        source="workspace export",
    )
    paths = step_file_paths(workspace_dir)
    selection = payload_to_workspace_selection(load_json(paths["selection"]))
    transform_spec = load_json(paths["transform_spec"])
    export_dir = workspace_dir / "background_export"
    export_dir.mkdir(parents=True, exist_ok=True)

    target_context = load_target_context(selection)
    source_context = load_source_context(selection)

    summary_path = export_dir / "workspace_export_summary.json"
    try:
        if selection.dataset_root is None:
            raise FileNotFoundError(
                f"No ArtSpeech dataset root is available for {selection.case.speaker}/{selection.case.session}."
            )
        session_data = load_session_data(selection.dataset_root, selection.case.speaker, selection.case.session)
    except Exception as exc:
        preview_path = export_dir / "preview_only.png"
        preview_frame = source_context["source_frame"]
        inverse_mapping = lambda pts: apply_transform_from_spec(transform_spec, pts, direction="inverse", stage="final")
        source_x, source_y, valid_mask = precompute_inverse_warp(target_context["target_image"].shape[:2], inverse_mapping, preview_frame.shape[:2])
        warped_preview, _ = warp_array_with_precomputed_inverse_warp(preview_frame, source_x, source_y, valid_mask)
        imageio.imwrite(preview_path, warped_preview)
        payload = {
            "mode": "preview_only",
            "reason": str(exc),
            "preview_png": str(preview_path),
        }
        summary_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        return payload

    speaker = selection.case.speaker
    session = selection.case.session
    target_label = target_context["target_label"]
    target_frame_img = target_context["target_image"]
    target_contours = target_context["target_contours"]
    reference_contours = source_context["reference_contours"]
    source_space_contours = source_context["projected_contours"]

    articulators = resolve_common_articulators(target_contours, reference_contours, defaults=DEFAULT_ARTICULATORS)

    def forward_mapping(pts):
        return apply_transform_from_spec(transform_spec, pts, direction="forward", stage="final")

    target_space_ref_contours = {
        name: np.asarray(forward_mapping(np.asarray(reference_contours[name], dtype=float)), dtype=float)
        for name in articulators
        if name in reference_contours
    }

    frame_count = session_data.images.shape[0] if max_frames <= 0 else min(max_frames, session_data.images.shape[0])
    inverse_mapping = lambda pts: apply_transform_from_spec(transform_spec, pts, direction="inverse", stage="final")
    target_shape = tuple(int(value) for value in transform_spec["target_shape"])
    source_x, source_y, valid_mask = precompute_inverse_warp(target_shape, inverse_mapping, session_data.images.shape[1:])
    valid_ratio = float(valid_mask.mean())

    write_warped = output_mode in {"both", "warped"}
    write_review = output_mode in {"both", "review"}

    warped_video_path = export_dir / f"{speaker}_{session}_to_{target_label.replace(' ', '_').replace('/', '-')[:60]}_warped.mp4"
    review_video_path = export_dir / f"{speaker}_{session}_to_{target_label.replace(' ', '_').replace('/', '-')[:60]}_review.mp4"
    mid_index = frame_count // 2
    warped_preview_paths = {
        0: export_dir / "warped_preview_frame_0001.png",
        mid_index: export_dir / f"warped_preview_frame_{mid_index + 1:04d}.png",
    } if write_warped else {}
    review_preview_paths = {
        0: export_dir / "review_preview_frame_0001.png",
        mid_index: export_dir / f"review_preview_frame_{mid_index + 1:04d}.png",
    } if write_review else {}
    prefetch_count = _resolve_prefetch(workers, prefetch)

    word_cursor = IntervalCursor(session_data.tiers[0].intervals if len(session_data.tiers) >= 1 else [])
    phoneme_cursor = IntervalCursor(session_data.tiers[1].intervals if len(session_data.tiers) >= 2 else [])
    sentence_cursor = IntervalCursor(session_data.sentences)
    frame_times = (np.arange(frame_count, dtype=np.float64) + 0.5) / session_data.frame_rate
    audio_duration = float(session_data.samples.size) / session_data.sample_rate
    waveform_time, waveform_values = downsample_waveform(session_data.samples, session_data.sample_rate)

    first_source_frame = normalize_frame(session_data.images[0], session_data.frame_min, session_data.frame_max)
    first_warped_frame, _ = warp_array_with_precomputed_inverse_warp(first_source_frame, source_x, source_y, valid_mask)

    review_fig = review_axes = review_artists = None
    if write_review:
        assumed_label = f"green: target ({target_label})\npink dashed: reference ({selection.reference_speaker})"
        review_fig, review_axes, review_artists = _make_review_figure(
            first_source_frame=first_source_frame,
            target_frame=target_frame_img,
            first_warped_frame=first_warped_frame,
            waveform_time=waveform_time,
            waveform_values=waveform_values,
            source_title=f"Source: {speaker}/{session}",
            target_title=f"Target: {target_label}",
            assumed_label=assumed_label,
        )

    from contextlib import ExitStack
    with ExitStack() as stack:
        warped_writer = None
        review_writer = None
        def _open_writer(path):
            return stack.enter_context(imageio.get_writer(
                path, fps=session_data.frame_rate, codec="libx264", pixelformat="yuv420p",
                quality=7, ffmpeg_log_level="warning",
                audio_path=str(session_data.paths.wav_path), audio_codec="aac",
            ))
        if write_warped:
            warped_writer = _open_writer(warped_video_path)
        if write_review:
            review_writer = _open_writer(review_video_path)

        pending: dict[int, Future[tuple[int, np.ndarray]]] = {}
        submission_order: deque[int] = deque()
        next_submit = 0
        with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="workspace-warp") as executor:
            for index in range(frame_count):
                while next_submit < frame_count and len(pending) < prefetch_count:
                    future = executor.submit(
                        _render_single_frame,
                        next_submit,
                        images=session_data.images,
                        frame_min=session_data.frame_min,
                        frame_max=session_data.frame_max,
                        source_x=source_x,
                        source_y=source_y,
                        valid_mask=valid_mask,
                    )
                    pending[next_submit] = future
                    submission_order.append(next_submit)
                    next_submit += 1

                future = pending.pop(index)
                if submission_order and submission_order[0] == index:
                    submission_order.popleft()
                _, warped_frame = future.result()

                if index % 100 == 0 or index == frame_count - 1:
                    print(f"[video] {workspace_dir.name} frame {index + 1}/{frame_count}")

                if warped_writer is not None:
                    warped_writer.append_data(_gray_to_rgb(warped_frame))
                    warped_preview_path = warped_preview_paths.get(index)
                    if warped_preview_path is not None:
                        imageio.imwrite(warped_preview_path, warped_frame)

                if write_review and review_writer is not None and review_fig is not None:
                    current_time = float(frame_times[index])
                    src_frame = normalize_frame(session_data.images[index], session_data.frame_min, session_data.frame_max)
                    review_rgb = _draw_review_frame(
                        review_fig, review_axes, review_artists,
                        source_frame=src_frame,
                        target_frame=target_frame_img,
                        warped_frame=warped_frame,
                        source_space_contours=source_space_contours,
                        target_contours=target_contours,
                        target_space_ref_contours=target_space_ref_contours,
                        articulators=articulators,
                        frame_index=index,
                        current_time=current_time,
                        audio_duration=audio_duration,
                        word=word_cursor.at(current_time),
                        phoneme=phoneme_cursor.at(current_time),
                        sentence=sentence_cursor.at(current_time),
                        valid_ratio=valid_ratio,
                    )
                    review_writer.append_data(review_rgb)
                    review_preview_path = review_preview_paths.get(index)
                    if review_preview_path is not None:
                        imageio.imwrite(review_preview_path, review_rgb)

    if review_fig is not None:
        plt.close(review_fig)

    payload = {
        "mode": "full_session",
        "workspace_dir": str(workspace_dir),
        "export_dir": str(export_dir),
        "warped_video": str(warped_video_path) if write_warped else None,
        "review_video": str(review_video_path) if write_review else None,
        "valid_ratio": valid_ratio,
        "frame_count": int(frame_count),
        "workers": int(workers),
        "prefetch": int(prefetch_count),
        "target_label": target_label,
    }
    summary_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return payload


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    payload = run_workspace_export(
        workspace_dir=args.workspace_dir,
        workers=int(args.workers),
        prefetch=int(args.prefetch),
        max_frames=int(args.max_frames),
        output_mode=args.output_mode,
    )
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
