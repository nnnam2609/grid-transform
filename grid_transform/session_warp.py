from __future__ import annotations

import json
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path

import imageio.v2 as imageio
import matplotlib.pyplot as plt
import numpy as np

from grid_transform.annotation_projection import (
    build_resize_affine,
    transform_reference_contours,
)
from grid_transform.apps.method4_transform import apply_transform
from grid_transform.artspeech_video import (
    IntervalCursor,
    downsample_waveform,
    load_session_data,
    normalize_frame,
    resolve_default_dataset_root,
)
from grid_transform.config import (
    DEFAULT_VTNL_DIR,
    TONGUE_COLOR,
    VIDEO_OUTPUT_DIR,
    VT_SEG_CONTOURS_ROOT,
    VT_SEG_DATA_ROOT,
)
from grid_transform.io import load_frame_npy, load_frame_vtnl
from grid_transform.source_annotation import (
    controls_to_grid_constraints,
    controls_to_grid_contours,
    load_source_annotation_json,
)
from grid_transform.transfer import (
    DEFAULT_ARTICULATORS,
    build_two_step_transform,
    resolve_common_articulators,
    smooth_transformed_contours,
    transform_contours,
)
from grid_transform.vt import build_grid
from grid_transform.warp import (
    precompute_inverse_warp,
    warp_array_with_precomputed_inverse_warp,
)


TARGET_COLOR = "#16a085"
ASSUMED_COLOR = "#ef476f"


@dataclass
class PreparedSessionWarp:
    source_mode: str
    annotation_source: str
    reference_speaker: str | None
    source_annotation_json: str | None
    source_frame_index0: int | None
    source_frame_index1: int | None
    source_time_sec: float | None
    source_shape: tuple[int, int]
    target_shape: tuple[int, int]
    articulators: list[str]
    source_space_assumed_contours: dict[str, np.ndarray]
    target_space_assumed_contours: dict[str, np.ndarray]
    target_contours: dict[str, np.ndarray]
    target_to_source_mapping: object
    forward_transform: dict[str, object]
    review_raw_title: str
    review_overlay_legend: str
    assumption_text: str
    annotation_reference_shape: tuple[int, int]


def default_output_dir(
    artspeech_speaker: str,
    session: str,
    annotation_speaker: str,
    target_frame: int,
) -> Path:
    return VIDEO_OUTPUT_DIR / (
        f"{artspeech_speaker.lower()}_{session.lower()}_assume_{annotation_speaker.lower()}_to_{target_frame}"
    )


def default_output_dir_for_saved_annotation(
    artspeech_speaker: str,
    session: str,
    source_frame: int | None,
    target_frame: int,
) -> Path:
    frame_text = f"edited_frame_{source_frame:04d}" if source_frame is not None else "edited_annotation"
    return VIDEO_OUTPUT_DIR / f"{artspeech_speaker.lower()}_{session.lower()}_{frame_text}_to_{target_frame}"


def as_grayscale_uint8(image) -> np.ndarray:
    array = np.asarray(image)
    if array.ndim == 3:
        array = array[..., 0]
    return np.clip(array, 0, 255).astype(np.uint8)


def gray_to_rgb(frame: np.ndarray) -> np.ndarray:
    return np.repeat(frame[..., None], 3, axis=2)


def format_label_block(word: str, phoneme: str, sentence: str) -> str:
    sentence = " ".join(sentence.split()) if sentence else "-"
    if len(sentence) > 80:
        sentence = sentence[:77] + "..."
    return "\n".join(
        [
            f"word: {word or '-'}",
            f"phoneme: {phoneme or '-'}",
            f"sentence: {sentence}",
        ]
    )


def resolve_articulators(target_contours: dict, annotation_contours: dict) -> list[str]:
    return resolve_common_articulators(
        target_contours,
        annotation_contours,
        defaults=DEFAULT_ARTICULATORS,
    )


def build_target_to_source_mapping(target_to_annotation_fn, resize_affine: dict[str, np.ndarray]):
    def mapping(target_pts):
        annotation_pts = np.asarray(target_to_annotation_fn(target_pts), dtype=float)
        return np.asarray(apply_transform(resize_affine, annotation_pts), dtype=float)

    return mapping


def draw_contours(
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
        pts = np.asarray(contours[name], dtype=float)
        if name == "tongue":
            ax.plot(
                pts[:, 0],
                pts[:, 1],
                linestyle=linestyle,
                color=tongue_color,
                linewidth=2.0,
                alpha=alpha,
            )
        else:
            ax.plot(
                pts[:, 0],
                pts[:, 1],
                linestyle=linestyle,
                color=base_color,
                linewidth=linewidth,
                alpha=alpha,
            )


def make_review_figure(
    first_source_frame: np.ndarray,
    target_frame: np.ndarray,
    first_warped_frame: np.ndarray,
    waveform_time: np.ndarray,
    waveform_values: np.ndarray,
    *,
    raw_title: str,
    overlay_legend_text: str,
):
    fig = plt.figure(figsize=(12.8, 10.08), dpi=100)
    grid = fig.add_gridspec(3, 2, height_ratios=[2.6, 2.6, 1.2], hspace=0.12, wspace=0.06)

    ax_raw = fig.add_subplot(grid[0, 0])
    ax_target = fig.add_subplot(grid[0, 1])
    ax_warped = fig.add_subplot(grid[1, 0])
    ax_overlay = fig.add_subplot(grid[1, 1])
    ax_wave = fig.add_subplot(grid[2, :])

    for ax in (ax_raw, ax_target, ax_warped, ax_overlay):
        ax.set_axis_off()

    raw_artist = ax_raw.imshow(first_source_frame, cmap="gray", vmin=0, vmax=255)
    target_artist = ax_target.imshow(target_frame, cmap="gray", vmin=0, vmax=255)
    warped_artist = ax_warped.imshow(first_warped_frame, cmap="gray", vmin=0, vmax=255)
    overlay_artist = ax_overlay.imshow(first_warped_frame, cmap="gray", vmin=0, vmax=255)

    ax_raw.set_title(raw_title, fontsize=11, fontweight="bold")
    ax_target.set_title("Target 143020 frame + true contours", fontsize=11, fontweight="bold")
    ax_warped.set_title("Warped source frame in target space", fontsize=11, fontweight="bold")
    ax_overlay.set_title("Warped frame + target vs source contours", fontsize=11, fontweight="bold")

    frame_text = ax_raw.text(0.02, 0.97, "", color="yellow", fontsize=11, va="top", transform=ax_raw.transAxes)
    label_text = ax_raw.text(
        0.02,
        0.02,
        "",
        color="white",
        fontsize=9,
        family="monospace",
        va="bottom",
        transform=ax_raw.transAxes,
        bbox=dict(boxstyle="round,pad=0.3", fc="black", alpha=0.72, ec="none"),
    )
    valid_text = ax_warped.text(
        0.02,
        0.02,
        "",
        color="white",
        fontsize=9,
        family="monospace",
        va="bottom",
        transform=ax_warped.transAxes,
        bbox=dict(boxstyle="round,pad=0.3", fc="black", alpha=0.72, ec="none"),
    )
    overlay_legend = ax_overlay.text(
        0.02,
        0.98,
        overlay_legend_text,
        color="white",
        fontsize=9,
        family="monospace",
        va="top",
        transform=ax_overlay.transAxes,
        bbox=dict(boxstyle="round,pad=0.3", fc="black", alpha=0.72, ec="none"),
    )

    ax_wave.plot(waveform_time, waveform_values, color="black", linewidth=0.8)
    time_cursor = ax_wave.axvline(0.0, color="red", linewidth=1.2)
    ax_wave.set_ylim(-1.05, 1.05)
    ax_wave.set_ylabel("Amp.")
    ax_wave.set_xlabel("Time (s)")
    wave_title = ax_wave.set_title("", fontsize=10)

    artists = {
        "raw": raw_artist,
        "target": target_artist,
        "warped": warped_artist,
        "overlay": overlay_artist,
        "frame_text": frame_text,
        "label_text": label_text,
        "valid_text": valid_text,
        "overlay_legend": overlay_legend,
        "time_cursor": time_cursor,
        "wave_title": wave_title,
        "ax_wave": ax_wave,
    }
    axes = {
        "raw": ax_raw,
        "target": ax_target,
        "warped": ax_warped,
        "overlay": ax_overlay,
    }
    return fig, axes, artists


def draw_review_frame(
    fig: plt.Figure,
    axes: dict[str, plt.Axes],
    artists: dict[str, object],
    *,
    source_frame: np.ndarray,
    target_frame: np.ndarray,
    warped_frame: np.ndarray,
    source_space_assumed_contours: dict[str, np.ndarray],
    target_contours: dict[str, np.ndarray],
    target_space_assumed_contours: dict[str, np.ndarray],
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
    artists["overlay"].set_data(warped_frame)

    for key in ("raw", "target", "overlay"):
        for line in list(axes[key].lines):
            line.remove()

    draw_contours(
        axes["raw"],
        source_space_assumed_contours,
        articulators,
        base_color=ASSUMED_COLOR,
        tongue_color=TONGUE_COLOR,
        linestyle="--",
    )
    draw_contours(
        axes["target"],
        target_contours,
        articulators,
        base_color=TARGET_COLOR,
        tongue_color=TARGET_COLOR,
        linestyle="-",
    )
    draw_contours(
        axes["overlay"],
        target_contours,
        articulators,
        base_color=TARGET_COLOR,
        tongue_color=TARGET_COLOR,
        linestyle="-",
    )
    draw_contours(
        axes["overlay"],
        target_space_assumed_contours,
        articulators,
        base_color=ASSUMED_COLOR,
        tongue_color=TONGUE_COLOR,
        linestyle="--",
    )

    artists["frame_text"].set_text(f"frame {frame_index}")
    artists["label_text"].set_text(format_label_block(word, phoneme, sentence))
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


def _prepare_reference_annotation(
    *,
    annotation_speaker: str,
    artspeech_speaker: str,
    session: str,
    vtnl_dir: Path,
    session_data,
    target_image,
    target_contours: dict[str, np.ndarray],
    target_frame: int,
) -> PreparedSessionWarp:
    annotation_image, annotation_contours = load_frame_vtnl(annotation_speaker, vtnl_dir)
    annotation_frame = as_grayscale_uint8(annotation_image)

    source_shape = tuple(int(value) for value in session_data.images.shape[1:])
    target_shape = tuple(int(value) for value in np.asarray(target_image).shape[:2])
    articulators = resolve_articulators(target_contours, annotation_contours)

    annotation_grid = build_grid(annotation_image, annotation_contours, n_vert=9, n_points=250, frame_number=0)
    target_grid = build_grid(target_image, target_contours, n_vert=9, n_points=250, frame_number=target_frame)
    forward_transform = build_two_step_transform(annotation_grid, target_grid)
    inverse_transform = build_two_step_transform(target_grid, annotation_grid)
    resize_affine = build_resize_affine(tuple(annotation_frame.shape[:2]), source_shape)

    source_space_assumed_contours = transform_reference_contours(annotation_contours, resize_affine)
    target_space_assumed_contours = smooth_transformed_contours(
        transform_contours(annotation_contours, forward_transform["apply_two_step"], articulators)
    )
    target_to_source_mapping = build_target_to_source_mapping(inverse_transform["apply_two_step"], resize_affine)

    return PreparedSessionWarp(
        source_mode="vtnl_reference",
        annotation_source=annotation_speaker,
        reference_speaker=annotation_speaker,
        source_annotation_json=None,
        source_frame_index0=None,
        source_frame_index1=None,
        source_time_sec=None,
        source_shape=source_shape,
        target_shape=target_shape,
        articulators=articulators,
        source_space_assumed_contours=source_space_assumed_contours,
        target_space_assumed_contours=target_space_assumed_contours,
        target_contours=target_contours,
        target_to_source_mapping=target_to_source_mapping,
        forward_transform=forward_transform,
        review_raw_title=f"Raw {artspeech_speaker}/{session} frame + assumed {annotation_speaker} contours",
        review_overlay_legend=f"green: target {target_frame}\npink dashed: assumed {annotation_speaker}",
        assumption_text=f"All {artspeech_speaker}/{session} frames use VTNL annotation from {annotation_speaker}",
        annotation_reference_shape=tuple(int(value) for value in annotation_frame.shape[:2]),
    )


def _prepare_saved_source_annotation(
    *,
    source_annotation_json: Path,
    artspeech_speaker: str,
    session: str,
    session_data,
    target_image,
    target_contours: dict[str, np.ndarray],
    target_frame: int,
) -> PreparedSessionWarp:
    payload = load_source_annotation_json(source_annotation_json)
    metadata = payload["metadata"]
    if metadata.get("artspeech_speaker") and metadata["artspeech_speaker"] != artspeech_speaker:
        raise ValueError(
            f"Saved annotation speaker mismatch: expected {artspeech_speaker}, got {metadata['artspeech_speaker']}"
        )
    if metadata.get("session") and metadata["session"] != session:
        raise ValueError(f"Saved annotation session mismatch: expected {session}, got {metadata['session']}")

    source_frame_index1 = int(metadata.get("source_frame", 0))
    if source_frame_index1 <= 0:
        raise ValueError("Saved annotation metadata must include a positive source_frame value.")
    source_frame_index0 = source_frame_index1 - 1
    if source_frame_index0 >= session_data.images.shape[0]:
        raise ValueError(
            f"Saved annotation source_frame {source_frame_index1} exceeds session frame count {session_data.images.shape[0]}."
        )

    source_shape = tuple(int(value) for value in session_data.images.shape[1:])
    saved_shape = metadata.get("source_shape")
    if saved_shape:
        saved_shape_tuple = tuple(int(value) for value in saved_shape[:2])
        if saved_shape_tuple != source_shape:
            raise ValueError(f"Saved annotation shape {saved_shape_tuple} does not match session shape {source_shape}.")

    source_frame = normalize_frame(
        session_data.images[source_frame_index0],
        session_data.frame_min,
        session_data.frame_max,
    )
    source_contours = payload["contours"]
    grid_controls = metadata.get("grid_controls")
    target_shape = tuple(int(value) for value in np.asarray(target_image).shape[:2])
    articulators = resolve_articulators(target_contours, source_contours)

    source_grid = build_grid(
        source_frame,
        controls_to_grid_contours(source_contours, grid_controls),
        n_vert=9,
        n_points=250,
        frame_number=source_frame_index1,
        grid_constraints=controls_to_grid_constraints(source_contours, grid_controls),
    )
    target_grid = build_grid(target_image, target_contours, n_vert=9, n_points=250, frame_number=target_frame)
    forward_transform = build_two_step_transform(source_grid, target_grid)
    inverse_transform = build_two_step_transform(target_grid, source_grid)

    target_space_assumed_contours = smooth_transformed_contours(
        transform_contours(source_contours, forward_transform["apply_two_step"], articulators)
    )
    reference_shape = metadata.get("reference_shape") or saved_shape or list(source_shape)
    reference_shape_tuple = tuple(int(value) for value in reference_shape[:2])

    return PreparedSessionWarp(
        source_mode="saved_source_annotation",
        annotation_source=f"saved_annotation_frame_{source_frame_index1:04d}",
        reference_speaker=str(metadata.get("reference_speaker") or ""),
        source_annotation_json=str(source_annotation_json),
        source_frame_index0=source_frame_index0,
        source_frame_index1=source_frame_index1,
        source_time_sec=float(metadata["time_sec"]) if metadata.get("time_sec") is not None else None,
        source_shape=source_shape,
        target_shape=target_shape,
        articulators=articulators,
        source_space_assumed_contours=source_contours,
        target_space_assumed_contours=target_space_assumed_contours,
        target_contours=target_contours,
        target_to_source_mapping=inverse_transform["apply_two_step"],
        forward_transform=forward_transform,
        review_raw_title=f"Raw {artspeech_speaker}/{session} frame + edited frame {source_frame_index1} contours",
        review_overlay_legend=f"green: target {target_frame}\npink dashed: edited source",
        assumption_text=(
            f"All {artspeech_speaker}/{session} frames use saved source annotation from frame {source_frame_index1}"
        ),
        annotation_reference_shape=reference_shape_tuple,
    )


def prepare_session_warp(
    *,
    annotation_speaker: str,
    source_annotation_json: Path | None,
    artspeech_speaker: str,
    session: str,
    target_frame: int,
    target_case: str,
    dataset_root: Path | str | None,
    vtnl_dir: Path,
):
    dataset_root = Path(dataset_root) if dataset_root is not None else resolve_default_dataset_root(artspeech_speaker)

    target_image, target_contours = load_frame_npy(
        target_frame,
        VT_SEG_DATA_ROOT / target_case,
        VT_SEG_CONTOURS_ROOT / target_case,
    )
    target_frame_image = as_grayscale_uint8(target_image)
    session_data = load_session_data(dataset_root, artspeech_speaker, session)

    if source_annotation_json is not None:
        prepared = _prepare_saved_source_annotation(
            source_annotation_json=Path(source_annotation_json),
            artspeech_speaker=artspeech_speaker,
            session=session,
            session_data=session_data,
            target_image=target_image,
            target_contours=target_contours,
            target_frame=target_frame,
        )
    else:
        prepared = _prepare_reference_annotation(
            annotation_speaker=annotation_speaker,
            artspeech_speaker=artspeech_speaker,
            session=session,
            vtnl_dir=vtnl_dir,
            session_data=session_data,
            target_image=target_image,
            target_contours=target_contours,
            target_frame=target_frame,
        )

    return dataset_root, session_data, target_frame_image, prepared


def run_session_warp_to_target(
    *,
    annotation_speaker: str = "1640_s10_0829",
    source_annotation_json: Path | str | None = None,
    artspeech_speaker: str = "P7",
    session: str = "S10",
    target_frame: int = 143020,
    target_case: str = "2008-003^01-1791/test",
    dataset_root: Path | str | None = None,
    vtnl_dir: Path = DEFAULT_VTNL_DIR,
    output_dir: Path | str | None = None,
    max_frames: int = 0,
    output_mode: str = "both",
) -> dict[str, object]:
    dataset_root, session_data, target_frame_image, prepared = prepare_session_warp(
        annotation_speaker=annotation_speaker,
        source_annotation_json=Path(source_annotation_json) if source_annotation_json is not None else None,
        artspeech_speaker=artspeech_speaker,
        session=session,
        target_frame=target_frame,
        target_case=target_case,
        dataset_root=dataset_root,
        vtnl_dir=vtnl_dir,
    )

    if output_dir is None:
        if source_annotation_json is not None:
            output_dir = default_output_dir_for_saved_annotation(
                artspeech_speaker,
                session,
                prepared.source_frame_index1,
                target_frame,
            )
        else:
            output_dir = default_output_dir(
                artspeech_speaker,
                session,
                annotation_speaker,
                target_frame,
            )
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    frame_count = session_data.images.shape[0] if max_frames <= 0 else min(max_frames, session_data.images.shape[0])
    source_x, source_y, valid_mask = precompute_inverse_warp(
        prepared.target_shape,
        prepared.target_to_source_mapping,
        prepared.source_shape,
    )
    valid_ratio = float(valid_mask.mean())

    word_cursor = IntervalCursor(session_data.tiers[0].intervals if len(session_data.tiers) >= 1 else [])
    phoneme_cursor = IntervalCursor(session_data.tiers[1].intervals if len(session_data.tiers) >= 2 else [])
    sentence_cursor = IntervalCursor(session_data.sentences)
    frame_times = (np.arange(frame_count, dtype=np.float64) + 0.5) / session_data.frame_rate
    audio_duration = session_data.samples.size / session_data.sample_rate
    waveform_time, waveform_values = downsample_waveform(session_data.samples, session_data.sample_rate)

    first_source_frame = normalize_frame(session_data.images[0], session_data.frame_min, session_data.frame_max)
    first_warped_frame, warped_mask = warp_array_with_precomputed_inverse_warp(
        first_source_frame,
        source_x,
        source_y,
        valid_mask,
    )
    warped_video_path = None
    review_video_path = None
    review_figure = None
    review_axes = None
    review_artists = None

    write_warped = output_mode in {"both", "warped"}
    write_review = output_mode in {"both", "review"}

    if write_warped:
        warped_video_path = output_dir / f"{artspeech_speaker}_{session}_warped_to_{target_frame}.mp4"
    if write_review:
        review_video_path = output_dir / f"{artspeech_speaker}_{session}_warped_to_{target_frame}_review.mp4"
        review_figure, review_axes, review_artists = make_review_figure(
            first_source_frame=first_source_frame,
            target_frame=target_frame_image,
            first_warped_frame=first_warped_frame,
            waveform_time=waveform_time,
            waveform_values=waveform_values,
            raw_title=prepared.review_raw_title,
            overlay_legend_text=prepared.review_overlay_legend,
        )

    mid_index = frame_count // 2
    warped_preview_paths = (
        {
            0: output_dir / "warped_preview_frame_0001.png",
            mid_index: output_dir / f"warped_preview_frame_{mid_index + 1:04d}.png",
        }
        if write_warped
        else {}
    )
    clean_warp_alias_paths = (
        {
            0: output_dir / "source_speaker_warped_to_target.png",
            mid_index: output_dir / "source_speaker_warped_to_target_mid.png",
        }
        if write_warped
        else {}
    )
    review_preview_paths = (
        {
            0: output_dir / "review_preview_frame_0001.png",
            mid_index: output_dir / f"review_preview_frame_{mid_index + 1:04d}.png",
        }
        if write_review
        else {}
    )

    print("[video] writing requested outputs")
    with ExitStack() as stack:
        warped_writer = None
        review_writer = None
        if write_warped and warped_video_path is not None:
            warped_writer = stack.enter_context(
                imageio.get_writer(
                    warped_video_path,
                    fps=session_data.frame_rate,
                    codec="libx264",
                    pixelformat="yuv420p",
                    quality=7,
                    ffmpeg_log_level="warning",
                    audio_path=str(session_data.paths.wav_path),
                    audio_codec="aac",
                )
            )
        if write_review and review_video_path is not None:
            review_writer = stack.enter_context(
                imageio.get_writer(
                    review_video_path,
                    fps=session_data.frame_rate,
                    codec="libx264",
                    pixelformat="yuv420p",
                    quality=7,
                    ffmpeg_log_level="warning",
                    audio_path=str(session_data.paths.wav_path),
                    audio_codec="aac",
                )
            )

        for index in range(frame_count):
            if index % 100 == 0 or index == frame_count - 1:
                print(f"[video] rendering frame {index + 1}/{frame_count}")
            current_time = frame_times[index]
            source_frame = normalize_frame(session_data.images[index], session_data.frame_min, session_data.frame_max)
            warped_frame, _ = warp_array_with_precomputed_inverse_warp(
                source_frame,
                source_x,
                source_y,
                valid_mask,
            )
            word = word_cursor.at(current_time)
            phoneme = phoneme_cursor.at(current_time)
            sentence = sentence_cursor.at(current_time)

            if warped_writer is not None:
                warped_rgb = gray_to_rgb(warped_frame)
                warped_writer.append_data(warped_rgb)
                preview_path = warped_preview_paths.get(index)
                if preview_path is not None:
                    imageio.imwrite(preview_path, warped_frame)
                clean_alias_path = clean_warp_alias_paths.get(index)
                if clean_alias_path is not None:
                    imageio.imwrite(clean_alias_path, warped_frame)

            if review_writer is not None and review_figure is not None and review_axes is not None and review_artists is not None:
                review_rgb = draw_review_frame(
                    review_figure,
                    review_axes,
                    review_artists,
                    source_frame=source_frame,
                    target_frame=target_frame_image,
                    warped_frame=warped_frame,
                    source_space_assumed_contours=prepared.source_space_assumed_contours,
                    target_contours=prepared.target_contours,
                    target_space_assumed_contours=prepared.target_space_assumed_contours,
                    articulators=prepared.articulators,
                    frame_index=index + 1,
                    current_time=current_time,
                    audio_duration=audio_duration,
                    word=word,
                    phoneme=phoneme,
                    sentence=sentence,
                    valid_ratio=valid_ratio,
                )
                review_writer.append_data(review_rgb)
                preview_path = review_preview_paths.get(index)
                if preview_path is not None:
                    imageio.imwrite(preview_path, review_rgb)

    if review_figure is not None:
        plt.close(review_figure)

    summary = {
        "workflow": "artspeech_session_warp_to_target",
        "transform_mode": (
            "fixed_saved_source_annotation_affine_tps"
            if prepared.source_mode == "saved_source_annotation"
            else "fixed_affine_tps_plus_resize_assumption"
        ),
        "assumption": prepared.assumption_text,
        "annotation_source": prepared.annotation_source,
        "reference_speaker": prepared.reference_speaker,
        "source_annotation_mode": prepared.source_mode,
        "source_annotation_json": prepared.source_annotation_json,
        "source_annotation_frame": prepared.source_frame_index1,
        "source_annotation_time_sec": prepared.source_time_sec,
        "artspeech_speaker": artspeech_speaker,
        "session": session,
        "target_frame": target_frame,
        "target_case": target_case,
        "dataset_root": str(dataset_root),
        "source_shape": list(prepared.source_shape),
        "annotation_reference_shape": list(prepared.annotation_reference_shape),
        "target_shape": list(prepared.target_shape),
        "frame_count": frame_count,
        "frame_rate_fps": session_data.frame_rate,
        "step1_labels": list(prepared.forward_transform["step1_labels"]),
        "step2_labels": list(prepared.forward_transform["step2_labels"]),
        "valid_pixel_ratio_mean": valid_ratio,
        "valid_pixel_ratio_min": valid_ratio,
        "valid_pixel_ratio_max": valid_ratio,
        "warped_video_path": None if warped_video_path is None else str(warped_video_path),
        "review_video_path": None if review_video_path is None else str(review_video_path),
        "warped_preview_frame_0001": None if 0 not in warped_preview_paths else str(warped_preview_paths[0]),
        "warped_preview_frame_mid": None if mid_index not in warped_preview_paths else str(warped_preview_paths[mid_index]),
        "source_speaker_warped_to_target_preview": None if 0 not in clean_warp_alias_paths else str(clean_warp_alias_paths[0]),
        "source_speaker_warped_to_target_preview_mid": None if mid_index not in clean_warp_alias_paths else str(clean_warp_alias_paths[mid_index]),
        "review_preview_frame_0001": None if 0 not in review_preview_paths else str(review_preview_paths[0]),
        "review_preview_frame_mid": None if mid_index not in review_preview_paths else str(review_preview_paths[mid_index]),
        "output_mode": output_mode,
        "articulators": prepared.articulators,
        "target_mask_valid_pixel_count": int(np.count_nonzero(warped_mask)),
    }
    summary_path = output_dir / "warp_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary
