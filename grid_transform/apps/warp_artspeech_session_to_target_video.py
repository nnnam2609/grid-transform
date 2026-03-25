from __future__ import annotations

import argparse
import json
from contextlib import ExitStack
from pathlib import Path

import imageio.v2 as imageio
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

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


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Warp a full ArtSpeech session into one nnUNet target frame space using a fixed "
            "1640-derived annotation assumption plus the existing affine + TPS transform."
        )
    )
    parser.add_argument(
        "--annotation-speaker",
        default="1640_s10_0654",
        help="VTNL annotation/reference speaker image name.",
    )
    parser.add_argument("--artspeech-speaker", default="P7", help="ArtSpeech speaker id, for example P7.")
    parser.add_argument("--session", default="S10", help="ArtSpeech session id, for example S10.")
    parser.add_argument("--target-frame", type=int, default=143020, help="nnUNet target frame number.")
    parser.add_argument(
        "--target-case",
        default="2008-003^01-1791/test",
        help="nnUNet target case relative path.",
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        help="Optional explicit ArtSpeech dataset root. Defaults to an auto-detected local path.",
    )
    parser.add_argument("--vtnl-dir", type=Path, default=DEFAULT_VTNL_DIR, help="Folder containing VTNL images and ROI zip files.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Optional explicit output directory. Defaults to outputs/videos/<speaker>_<session>_assume_<annotation>_to_<frame>/.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=0,
        help="Optional debug limit. Use 0 for the full session.",
    )
    parser.add_argument(
        "--output-mode",
        choices=("both", "warped", "review"),
        default="both",
        help="Whether to write the clean warped video, the review video, or both.",
    )
    return parser.parse_args(argv)


def default_output_dir(
    artspeech_speaker: str,
    session: str,
    annotation_speaker: str,
    target_frame: int,
) -> Path:
    return VIDEO_OUTPUT_DIR / (
        f"{artspeech_speaker.lower()}_{session.lower()}_assume_{annotation_speaker.lower()}_to_{target_frame}"
    )


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

    ax_raw.set_title("Raw P7/S10 frame + assumed 1640 contours", fontsize=11, fontweight="bold")
    ax_target.set_title("Target 143020 frame + true contours", fontsize=11, fontweight="bold")
    ax_warped.set_title("Warped P7/S10 frame in target space", fontsize=11, fontweight="bold")
    ax_overlay.set_title("Warped frame + target vs assumed contours", fontsize=11, fontweight="bold")

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
        "green: target 143020\npink dashed: assumed 1640",
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


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    dataset_root = args.dataset_root or resolve_default_dataset_root(args.artspeech_speaker)
    output_dir = args.output_dir or default_output_dir(
        args.artspeech_speaker,
        args.session,
        args.annotation_speaker,
        args.target_frame,
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    print("[load] reading VTNL annotation source")
    annotation_image, annotation_contours = load_frame_vtnl(args.annotation_speaker, args.vtnl_dir)
    annotation_frame = as_grayscale_uint8(annotation_image)

    print("[load] reading nnUNet target frame")
    target_image, target_contours = load_frame_npy(
        args.target_frame,
        VT_SEG_DATA_ROOT / args.target_case,
        VT_SEG_CONTOURS_ROOT / args.target_case,
    )
    target_frame = as_grayscale_uint8(target_image)

    print("[load] reading ArtSpeech session")
    session_data = load_session_data(dataset_root, args.artspeech_speaker, args.session)
    frame_count = session_data.images.shape[0] if args.max_frames <= 0 else min(args.max_frames, session_data.images.shape[0])
    source_shape = tuple(int(value) for value in session_data.images.shape[1:])
    target_shape = tuple(int(value) for value in target_frame.shape[:2])

    articulators = resolve_articulators(target_contours, annotation_contours)
    annotation_grid = build_grid(annotation_image, annotation_contours, n_vert=9, n_points=250, frame_number=0)
    target_grid = build_grid(target_image, target_contours, n_vert=9, n_points=250, frame_number=args.target_frame)
    forward_transform = build_two_step_transform(annotation_grid, target_grid)
    inverse_transform = build_two_step_transform(target_grid, annotation_grid)
    resize_affine = build_resize_affine(tuple(annotation_frame.shape[:2]), source_shape)

    source_space_assumed_contours = transform_reference_contours(annotation_contours, resize_affine)
    target_space_assumed_contours = smooth_transformed_contours(
        transform_contours(annotation_contours, forward_transform["apply_two_step"], articulators)
    )
    target_to_source_mapping = build_target_to_source_mapping(inverse_transform["apply_two_step"], resize_affine)
    source_x, source_y, valid_mask = precompute_inverse_warp(target_shape, target_to_source_mapping, source_shape)
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

    write_warped = args.output_mode in {"both", "warped"}
    write_review = args.output_mode in {"both", "review"}

    if write_warped:
        warped_video_path = output_dir / f"{args.artspeech_speaker}_{args.session}_warped_to_{args.target_frame}.mp4"
    if write_review:
        review_video_path = output_dir / f"{args.artspeech_speaker}_{args.session}_warped_to_{args.target_frame}_review.mp4"
        review_figure, review_axes, review_artists = make_review_figure(
            first_source_frame=first_source_frame,
            target_frame=target_frame,
            first_warped_frame=first_warped_frame,
            waveform_time=waveform_time,
            waveform_values=waveform_values,
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
                    target_frame=target_frame,
                    warped_frame=warped_frame,
                    source_space_assumed_contours=source_space_assumed_contours,
                    target_contours=target_contours,
                    target_space_assumed_contours=target_space_assumed_contours,
                    articulators=articulators,
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
        "transform_mode": "fixed_affine_tps_plus_resize_assumption",
        "assumption": f"All {args.artspeech_speaker}/{args.session} frames use VTNL annotation from {args.annotation_speaker}",
        "annotation_source": args.annotation_speaker,
        "artspeech_speaker": args.artspeech_speaker,
        "session": args.session,
        "target_frame": args.target_frame,
        "target_case": args.target_case,
        "dataset_root": str(dataset_root),
        "source_shape": list(source_shape),
        "annotation_reference_shape": list(annotation_frame.shape[:2]),
        "target_shape": list(target_shape),
        "frame_count": frame_count,
        "frame_rate_fps": session_data.frame_rate,
        "step1_labels": list(forward_transform["step1_labels"]),
        "step2_labels": list(forward_transform["step2_labels"]),
        "resize_scale_x": float(resize_affine["A"][0, 0]),
        "resize_scale_y": float(resize_affine["A"][1, 1]),
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
        "output_mode": args.output_mode,
        "articulators": articulators,
        "target_mask_valid_pixel_count": int(np.count_nonzero(warped_mask)),
    }
    summary_path = output_dir / "warp_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
