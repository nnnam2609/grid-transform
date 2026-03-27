from __future__ import annotations

import argparse
import json
from pathlib import Path

import imageio.v2 as imageio
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from grid_transform.annotation_projection import (
    build_resize_affine,
    resize_inverse_mapping,
    transform_reference_contours,
)
from grid_transform.artspeech_video import (
    IntervalCursor,
    default_output_dir,
    downsample_waveform,
    load_session_data,
    normalize_frame,
    resolve_default_dataset_root,
)
from grid_transform.config import DEFAULT_VTNL_DIR, TONGUE_COLOR
from grid_transform.io import load_frame_vtnl
from grid_transform.warp import warp_image_to_target_space


REFERENCE_COLORS = {
    "tongue": TONGUE_COLOR,
    "incisior-hard-palate": "#ef476f",
    "soft-palate-midline": "#8338ec",
    "soft-palate": "#ff7f50",
    "mandible-incisior": "#f4a261",
    "pharynx": "#118ab2",
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Project one VTNL annotation/reference speaker onto a full ArtSpeech session video "
            "using a static resize affine assumption."
        )
    )
    parser.add_argument(
        "--target-speaker",
        "--vtnl-speaker",
        dest="target_speaker",
        default="1640_s10_0829",
        help="VTNL annotation/reference speaker image name.",
    )
    parser.add_argument("--vtnl-dir", type=Path, default=DEFAULT_VTNL_DIR, help="Folder containing VTNL images and ROI zip files.")
    parser.add_argument("--artspeech-speaker", default="P7", help="ArtSpeech speaker id, for example P7.")
    parser.add_argument("--session", default="S10", help="ArtSpeech session id, for example S10.")
    parser.add_argument(
        "--dataset-root",
        type=Path,
        help="Optional explicit ArtSpeech dataset root. Defaults to an auto-detected local path.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Optional output directory. Defaults to outputs/videos/<target>_on_<speaker>_<session>/.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=0,
        help="Optional debug limit. Use 0 for the full session.",
    )
    return parser.parse_args(argv)


def default_projection_output_dir(target_speaker: str, artspeech_speaker: str, session: str) -> Path:
    return default_output_dir(f"{target_speaker}_on_{artspeech_speaker}", session)


def frame_correlation(reference_frame: np.ndarray, session_frame: np.ndarray) -> float:
    ref = reference_frame.astype(np.float32).ravel()
    cur = session_frame.astype(np.float32).ravel()
    ref_std = float(ref.std())
    cur_std = float(cur.std())
    if ref_std < 1e-8 or cur_std < 1e-8:
        return 0.0
    return float(np.corrcoef(ref, cur)[0, 1])


def make_projection_figure(reference_frame: np.ndarray, waveform_time: np.ndarray, waveform_values: np.ndarray):
    fig = plt.figure(figsize=(12.8, 6.4), dpi=100)
    grid = fig.add_gridspec(2, 3, height_ratios=[3.2, 1.2], hspace=0.16, wspace=0.08)

    ax_raw = fig.add_subplot(grid[0, 0])
    ax_reference = fig.add_subplot(grid[0, 1])
    ax_overlay = fig.add_subplot(grid[0, 2])
    ax_wave = fig.add_subplot(grid[1, :])

    for ax in (ax_raw, ax_reference, ax_overlay):
        ax.set_axis_off()

    raw_artist = ax_raw.imshow(reference_frame, cmap="gray", vmin=0, vmax=255)
    reference_artist = ax_reference.imshow(reference_frame, cmap="gray", vmin=0, vmax=255)
    overlay_artist = ax_overlay.imshow(reference_frame, cmap="gray", vmin=0, vmax=255)

    ax_raw.set_title("ArtSpeech frame", fontsize=12, fontweight="bold")
    ax_reference.set_title("Projected VTNL reference", fontsize=12, fontweight="bold")
    ax_overlay.set_title("Reference contours on frame", fontsize=12, fontweight="bold")

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
    corr_text = ax_overlay.text(
        0.02,
        0.02,
        "",
        color="white",
        fontsize=9,
        family="monospace",
        va="bottom",
        transform=ax_overlay.transAxes,
        bbox=dict(boxstyle="round,pad=0.3", fc="black", alpha=0.72, ec="none"),
    )

    ax_wave.plot(waveform_time, waveform_values, color="black", linewidth=0.8)
    time_cursor = ax_wave.axvline(0.0, color="red", linewidth=1.2)
    ax_wave.set_ylim(-1.05, 1.05)
    ax_wave.set_ylabel("Amp.")
    ax_wave.set_xlabel("Time (s)")
    ax_wave.set_title("", fontsize=10)

    artists = {
        "raw": raw_artist,
        "reference": reference_artist,
        "overlay": overlay_artist,
        "frame_text": frame_text,
        "label_text": label_text,
        "corr_text": corr_text,
        "time_cursor": time_cursor,
        "ax_wave": ax_wave,
    }
    return fig, (ax_raw, ax_reference, ax_overlay), artists


def draw_reference_contours(ax, contours: dict[str, np.ndarray]) -> None:
    for name, points in sorted(contours.items()):
        pts = np.asarray(points, dtype=float)
        color = REFERENCE_COLORS.get(name, "#00b4d8")
        linestyle = "--" if name == "tongue" else "-"
        linewidth = 2.0 if name == "tongue" else 1.6
        ax.plot(pts[:, 0], pts[:, 1], linestyle=linestyle, color=color, linewidth=linewidth, alpha=0.92)


def format_label_block(word: str, phoneme: str, sentence: str) -> str:
    sentence = " ".join(sentence.split()) if sentence else "-"
    if len(sentence) > 64:
        sentence = sentence[:61] + "..."
    return "\n".join(
        [
            f"word: {word or '-'}",
            f"phoneme: {phoneme or '-'}",
            f"sentence: {sentence}",
        ]
    )


def draw_projection_frame(
    fig: plt.Figure,
    axes,
    artists: dict[str, object],
    session_frame: np.ndarray,
    projected_reference_frame: np.ndarray,
    projected_reference_contours: dict[str, np.ndarray],
    frame_index: int,
    current_time: float,
    audio_duration: float,
    correlation: float,
    word: str,
    phoneme: str,
    sentence: str,
) -> np.ndarray:
    ax_raw, ax_reference, ax_overlay = axes
    artists["raw"].set_data(session_frame)
    artists["reference"].set_data(projected_reference_frame)
    artists["overlay"].set_data(session_frame)

    for line in list(ax_overlay.lines):
        line.remove()
    draw_reference_contours(ax_overlay, projected_reference_contours)

    artists["frame_text"].set_text(f"frame {frame_index}")
    artists["label_text"].set_text(format_label_block(word, phoneme, sentence))
    artists["corr_text"].set_text(f"ref corr: {correlation:.4f}")
    artists["time_cursor"].set_xdata([current_time, current_time])

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
    output_dir = args.output_dir or default_projection_output_dir(args.target_speaker, args.artspeech_speaker, args.session)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("[load] reading VTNL reference")
    reference_image, reference_contours = load_frame_vtnl(args.target_speaker, args.vtnl_dir)

    print("[load] reading ArtSpeech session")
    session_data = load_session_data(dataset_root, args.artspeech_speaker, args.session)
    frame_count = session_data.images.shape[0] if args.max_frames <= 0 else min(args.max_frames, session_data.images.shape[0])

    target_shape = session_data.images.shape[1:]
    affine = build_resize_affine(np.asarray(reference_image).shape[:2], target_shape)
    projected_reference_contours = transform_reference_contours(reference_contours, affine)
    projected_reference_frame, _ = warp_image_to_target_space(
        reference_image,
        target_shape,
        resize_inverse_mapping(np.asarray(reference_image).shape[:2], target_shape),
    )

    word_cursor = IntervalCursor(session_data.tiers[0].intervals if len(session_data.tiers) >= 1 else [])
    phoneme_cursor = IntervalCursor(session_data.tiers[1].intervals if len(session_data.tiers) >= 2 else [])
    sentence_cursor = IntervalCursor(session_data.sentences)
    frame_times = (np.arange(frame_count, dtype=np.float64) + 0.5) / session_data.frame_rate
    waveform_time, waveform_values = downsample_waveform(session_data.samples, session_data.sample_rate)
    first_frame = normalize_frame(session_data.images[0], session_data.frame_min, session_data.frame_max)
    fig, axes, artists = make_projection_figure(first_frame, waveform_time, waveform_values)

    correlations: list[float] = []
    best_frame = {"index": 0, "correlation": -1.0, "time_sec": 0.0, "word": "", "phoneme": "", "sentence": ""}
    audio_duration = session_data.samples.size / session_data.sample_rate
    output_video = output_dir / f"{args.target_speaker}_projected_on_{args.artspeech_speaker}_{args.session}.mp4"
    preview_indices = {0: output_dir / "preview_frame_0001.png"}
    preview_indices[frame_count // 2] = output_dir / f"preview_frame_{frame_count // 2 + 1:04d}.png"

    print("[video] writing projection video")
    with imageio.get_writer(
        output_video,
        fps=session_data.frame_rate,
        codec="libx264",
        pixelformat="yuv420p",
        quality=7,
        ffmpeg_log_level="warning",
        audio_path=str(session_data.paths.wav_path),
        audio_codec="aac",
    ) as writer:
        for index in range(frame_count):
            if index % 100 == 0 or index == frame_count - 1:
                print(f"[video] rendering frame {index + 1}/{frame_count}")
            current_time = frame_times[index]
            session_frame = normalize_frame(session_data.images[index], session_data.frame_min, session_data.frame_max)
            correlation = frame_correlation(projected_reference_frame, session_frame)
            correlations.append(correlation)
            word = word_cursor.at(current_time)
            phoneme = phoneme_cursor.at(current_time)
            sentence = sentence_cursor.at(current_time)
            if correlation > best_frame["correlation"]:
                best_frame = {
                    "index": index,
                    "correlation": correlation,
                    "time_sec": float(current_time),
                    "word": word,
                    "phoneme": phoneme,
                    "sentence": sentence,
                }
            rgb = draw_projection_frame(
                fig=fig,
                axes=axes,
                artists=artists,
                session_frame=session_frame,
                projected_reference_frame=projected_reference_frame,
                projected_reference_contours=projected_reference_contours,
                frame_index=index + 1,
                current_time=current_time,
                audio_duration=audio_duration,
                correlation=correlation,
                word=word,
                phoneme=phoneme,
                sentence=sentence,
            )
            writer.append_data(rgb)
            preview_path = preview_indices.get(index)
            if preview_path is not None and not preview_path.exists():
                imageio.imwrite(preview_path, rgb)
    plt.close(fig)

    summary = {
        "workflow": "annotation_projection",
        "transform_mode": "static_affine_resize",
        "assumption": f"ArtSpeech session uses VTNL annotation from {args.target_speaker}",
        "video_path": str(output_video),
        "dataset_root": str(dataset_root),
        "annotation_source": args.target_speaker,
        "target_speaker": args.target_speaker,
        "vtnl_speaker": args.target_speaker,
        "artspeech_speaker": args.artspeech_speaker,
        "session": args.session,
        "frame_count": frame_count,
        "frame_rate_fps": session_data.frame_rate,
        "target_shape": list(target_shape),
        "reference_shape": list(np.asarray(reference_image).shape[:2]),
        "affine_scale_x": float(affine["A"][0, 0]),
        "affine_scale_y": float(affine["A"][1, 1]),
        "correlation_mean": float(np.mean(correlations)),
        "correlation_median": float(np.median(correlations)),
        "correlation_min": float(np.min(correlations)),
        "correlation_max": float(np.max(correlations)),
        "best_frame": best_frame,
        "preview_frame_0001": str(output_dir / "preview_frame_0001.png"),
        "preview_frame_mid": str(output_dir / f"preview_frame_{frame_count // 2 + 1:04d}.png"),
    }
    summary_path = output_dir / "projection_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
