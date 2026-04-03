from __future__ import annotations

import argparse
import json
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path

import imageio.v2 as imageio
import numpy as np

from grid_transform.annotation_to_grid_workflow import (
    apply_transform_from_spec,
    load_json,
    load_source_context,
    load_target_context,
    payload_to_workspace_selection,
    step_file_paths,
)
from grid_transform.artspeech_video import load_session_data, normalize_frame
from grid_transform.cv2_annotation_app_config import (
    RENDER_PREFETCH_RANGE,
    RENDER_WORKERS_RANGE,
    argparse_bounded_int,
    validate_render_settings,
)
from grid_transform.warp import precompute_inverse_warp, warp_array_with_precomputed_inverse_warp


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
    return parser.parse_args(argv)


def _resolve_prefetch(workers: int, prefetch: int) -> int:
    if prefetch > 0:
        return prefetch
    return max(2, workers * 3)


def _render_single_frame(index: int, *, images: np.ndarray, frame_min: float, frame_max: float, source_x: np.ndarray, source_y: np.ndarray, valid_mask: np.ndarray) -> tuple[int, np.ndarray]:
    frame = normalize_frame(images[index], frame_min, frame_max)
    warped_frame, _ = warp_array_with_precomputed_inverse_warp(frame, source_x, source_y, valid_mask)
    return index, warped_frame


def run_workspace_export(*, workspace_dir: Path, workers: int, prefetch: int, max_frames: int) -> dict[str, object]:
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

    frame_count = session_data.images.shape[0] if max_frames <= 0 else min(max_frames, session_data.images.shape[0])
    inverse_mapping = lambda pts: apply_transform_from_spec(transform_spec, pts, direction="inverse", stage="final")
    target_shape = tuple(int(value) for value in transform_spec["target_shape"])
    source_x, source_y, valid_mask = precompute_inverse_warp(target_shape, inverse_mapping, session_data.images.shape[1:])
    valid_ratio = float(valid_mask.mean())

    warped_video_path = export_dir / f"{selection.case.speaker}_{selection.case.session}_workspace_warped.mp4"
    preview_paths = {
        0: export_dir / "warped_preview_frame_0001.png",
        frame_count // 2: export_dir / f"warped_preview_frame_{frame_count // 2 + 1:04d}.png",
    }
    prefetch_count = _resolve_prefetch(workers, prefetch)

    with imageio.get_writer(
        warped_video_path,
        fps=session_data.frame_rate,
        codec="libx264",
        pixelformat="yuv420p",
        quality=7,
        ffmpeg_log_level="warning",
        audio_path=str(session_data.paths.wav_path),
        audio_codec="aac",
    ) as writer:
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
                writer.append_data(np.repeat(warped_frame[..., None], 3, axis=2))
                preview_path = preview_paths.get(index)
                if preview_path is not None:
                    imageio.imwrite(preview_path, warped_frame)

    payload = {
        "mode": "full_session",
        "workspace_dir": str(workspace_dir),
        "export_dir": str(export_dir),
        "warped_video": str(warped_video_path),
        "valid_ratio": valid_ratio,
        "frame_count": int(frame_count),
        "workers": int(workers),
        "prefetch": int(prefetch_count),
        "target_label": target_context["target_label"],
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
    )
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
