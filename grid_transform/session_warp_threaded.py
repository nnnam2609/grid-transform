from __future__ import annotations

import json
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import ExitStack
from pathlib import Path

import imageio.v2 as imageio
import matplotlib.pyplot as plt
import numpy as np

from grid_transform.artspeech_video import IntervalCursor, downsample_waveform, normalize_frame
from grid_transform.config import DEFAULT_VTLN_DIR
from grid_transform.session_warp import (
    default_output_dir,
    default_output_dir_for_saved_annotation,
    draw_review_frame,
    gray_to_rgb,
    make_review_figure,
    prepare_session_warp,
)
from grid_transform.warp import precompute_inverse_warp, warp_array_with_precomputed_inverse_warp


def _precompute_labels(session_data, frame_count: int) -> tuple[np.ndarray, list[tuple[str, str, str]], float, tuple[np.ndarray, np.ndarray]]:
    word_cursor = IntervalCursor(session_data.tiers[0].intervals if len(session_data.tiers) >= 1 else [])
    phoneme_cursor = IntervalCursor(session_data.tiers[1].intervals if len(session_data.tiers) >= 2 else [])
    sentence_cursor = IntervalCursor(session_data.sentences)
    frame_times = (np.arange(frame_count, dtype=np.float64) + 0.5) / session_data.frame_rate
    labels = [
        (
            word_cursor.at(float(current_time)),
            phoneme_cursor.at(float(current_time)),
            sentence_cursor.at(float(current_time)),
        )
        for current_time in frame_times
    ]
    audio_duration = session_data.samples.size / session_data.sample_rate
    waveform = downsample_waveform(session_data.samples, session_data.sample_rate)
    return frame_times, labels, audio_duration, waveform


def _render_single_frame(
    index: int,
    *,
    images: np.ndarray,
    frame_min: float,
    frame_max: float,
    source_x: np.ndarray,
    source_y: np.ndarray,
    valid_mask: np.ndarray,
) -> tuple[int, np.ndarray, np.ndarray]:
    source_frame = normalize_frame(images[index], frame_min, frame_max)
    warped_frame, _ = warp_array_with_precomputed_inverse_warp(
        source_frame,
        source_x,
        source_y,
        valid_mask,
    )
    return index, source_frame, warped_frame


def _resolve_prefetch(workers: int, prefetch: int) -> int:
    if prefetch > 0:
        return prefetch
    return max(2, workers * 3)


def run_session_warp_to_target_threaded(
    *,
    annotation_speaker: str = "1640_s10_0829",
    source_annotation_json: Path | str | None = None,
    artspeech_speaker: str = "P7",
    session: str = "S10",
    target_frame: int = 143020,
    target_case: str = "2008-003^01-1791/test",
    dataset_root: Path | str | None = None,
    vtln_dir: Path = DEFAULT_VTLN_DIR,
    output_dir: Path | str | None = None,
    max_frames: int = 0,
    output_mode: str = "both",
    workers: int = 4,
    prefetch: int = 0,
) -> dict[str, object]:
    dataset_root, session_data, target_frame_image, prepared = prepare_session_warp(
        annotation_speaker=annotation_speaker,
        source_annotation_json=Path(source_annotation_json) if source_annotation_json is not None else None,
        artspeech_speaker=artspeech_speaker,
        session=session,
        target_frame=target_frame,
        target_case=target_case,
        dataset_root=dataset_root,
        vtln_dir=vtln_dir,
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

    frame_times, labels, audio_duration, (waveform_time, waveform_values) = _precompute_labels(session_data, frame_count)

    first_source_frame = normalize_frame(session_data.images[0], session_data.frame_min, session_data.frame_max)
    first_warped_frame, warped_mask = warp_array_with_precomputed_inverse_warp(
        first_source_frame,
        source_x,
        source_y,
        valid_mask,
    )

    write_warped = output_mode in {"both", "warped"}
    write_review = output_mode in {"both", "review"}
    warped_video_path = output_dir / f"{artspeech_speaker}_{session}_warped_to_{target_frame}.mp4" if write_warped else None
    review_video_path = (
        output_dir / f"{artspeech_speaker}_{session}_warped_to_{target_frame}_review.mp4" if write_review else None
    )
    review_figure = None
    review_axes = None
    review_artists = None
    if write_review and review_video_path is not None:
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

    prefetch_count = _resolve_prefetch(workers=max(1, workers), prefetch=prefetch)
    print(f"[video-threaded] writing requested outputs with workers={workers}, prefetch={prefetch_count}")

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

        pending: dict[int, Future[tuple[int, np.ndarray, np.ndarray]]] = {}
        submission_order: deque[int] = deque()
        next_submit = 0

        with ThreadPoolExecutor(max_workers=max(1, workers), thread_name_prefix="warp-frame") as executor:
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

                _, source_frame, warped_frame = future.result()
                word, phoneme, sentence = labels[index]
                current_time = float(frame_times[index])

                if index % 100 == 0 or index == frame_count - 1:
                    print(f"[video-threaded] rendering frame {index + 1}/{frame_count}")

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
        "workflow": "artspeech_session_warp_to_target_threaded",
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
        "parallel_execution": {
            "backend": "threadpool",
            "workers": int(max(1, workers)),
            "prefetch": int(prefetch_count),
        },
    }
    summary_path = output_dir / "warp_summary_threaded.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary
