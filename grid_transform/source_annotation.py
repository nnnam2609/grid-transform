from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from grid_transform.annotation_projection import (
    build_resize_affine,
    resize_inverse_mapping,
    transform_reference_contours,
)
from grid_transform.artspeech_video import IntervalCursor, normalize_frame
from grid_transform.config import DEFAULT_OUTPUT_DIR
from grid_transform.warp import warp_image_to_target_space


DEFAULT_SOURCE_ANNOTATION_ROOT = DEFAULT_OUTPUT_DIR / "source_annotation_edits"
GRID_CONTROL_REQUIRED_CONTOURS = (
    "incisior-hard-palate",
    "mandible-incisior",
    "c1",
    "c2",
    "c3",
    "c4",
    "c5",
    "c6",
)
GRID_CONSTRAINT_GROUPS = ("none", "velum", "pharynx")
LONG_CONTOUR_HANDLE_COUNTS = {
    "pharynx": 32,
    "soft-palate-midline": 24,
}


def as_grayscale_uint8(image) -> np.ndarray:
    array = np.asarray(image)
    if array.ndim == 3:
        array = array[..., 0]
    return np.clip(array, 0, 255).astype(np.uint8)


def frame_correlation(reference_frame: np.ndarray, session_frame: np.ndarray) -> float:
    ref = np.asarray(reference_frame, dtype=np.float32).ravel()
    cur = np.asarray(session_frame, dtype=np.float32).ravel()
    ref_std = float(ref.std())
    cur_std = float(cur.std())
    if ref_std < 1e-8 or cur_std < 1e-8:
        return 0.0
    return float(np.corrcoef(ref, cur)[0, 1])


def default_match_output_path(reference_speaker: str, artspeech_speaker: str, session: str) -> Path:
    return DEFAULT_OUTPUT_DIR / "comparisons" / f"{reference_speaker}_vs_{artspeech_speaker}_{session}_match.json"


def default_source_annotation_output_dir(speaker: str, session: str, frame_number: int) -> Path:
    return DEFAULT_SOURCE_ANNOTATION_ROOT / f"{speaker.lower()}_{session.lower()}_frame_{frame_number:04d}"


def project_reference_annotation_to_source(
    reference_contours: dict[str, np.ndarray],
    reference_shape: tuple[int, int],
    source_shape: tuple[int, int],
) -> dict[str, np.ndarray]:
    affine = build_resize_affine(reference_shape, source_shape)
    return transform_reference_contours(reference_contours, affine)


def projected_reference_frame(reference_image, source_shape: tuple[int, int]) -> np.ndarray:
    projected, _ = warp_image_to_target_space(
        reference_image,
        source_shape,
        resize_inverse_mapping(np.asarray(reference_image).shape[:2], source_shape),
    )
    return np.asarray(projected, dtype=np.uint8)


def _snapshot_payload(
    frame_index0: int,
    frame_times: np.ndarray,
    correlations: np.ndarray,
    word_cursor: IntervalCursor,
    phoneme_cursor: IntervalCursor,
    sentence_cursor: IntervalCursor,
) -> dict[str, object]:
    current_time = float(frame_times[frame_index0])
    return {
        "index0": int(frame_index0),
        "frame1": int(frame_index0 + 1),
        "time_sec": current_time,
        "correlation": float(correlations[frame_index0]),
        "word": word_cursor.at(current_time),
        "phoneme": phoneme_cursor.at(current_time),
        "sentence": sentence_cursor.at(current_time),
    }


def compute_reference_session_match(
    reference_speaker: str,
    reference_image,
    session_data,
    artspeech_speaker: str,
    session: str,
) -> dict[str, object]:
    frame_count = int(session_data.images.shape[0])
    source_shape = tuple(int(value) for value in session_data.images.shape[1:])
    projected_ref = projected_reference_frame(reference_image, source_shape)

    correlations = np.zeros(frame_count, dtype=np.float64)
    for index in range(frame_count):
        frame = normalize_frame(session_data.images[index], session_data.frame_min, session_data.frame_max)
        correlations[index] = frame_correlation(projected_ref, frame)

    frame_times = (np.arange(frame_count, dtype=np.float64) + 0.5) / session_data.frame_rate
    word_cursor = IntervalCursor(session_data.tiers[0].intervals if len(session_data.tiers) >= 1 else [])
    phoneme_cursor = IntervalCursor(session_data.tiers[1].intervals if len(session_data.tiers) >= 2 else [])
    sentence_cursor = IntervalCursor(session_data.sentences)

    best_index0 = int(np.argmax(correlations))
    ranked = np.argsort(correlations)[::-1]
    top5 = [
        _snapshot_payload(int(index), frame_times, correlations, word_cursor, phoneme_cursor, sentence_cursor)
        for index in ranked[:5]
    ]

    payload: dict[str, object] = {
        "vtnl_reference": reference_speaker,
        "artspeech_speaker": artspeech_speaker,
        "session": session,
        "frame_count": frame_count,
        "best_match": _snapshot_payload(best_index0, frame_times, correlations, word_cursor, phoneme_cursor, sentence_cursor),
        "top5_matches": top5,
    }

    frame_829_exists = frame_count >= 829
    payload["frame_829_exists"] = frame_829_exists
    if frame_829_exists:
        frame_829_index0 = 828
        payload["frame_829"] = _snapshot_payload(
            frame_829_index0,
            frame_times,
            correlations,
            word_cursor,
            phoneme_cursor,
            sentence_cursor,
        )
        payload["frame_829_rank"] = int(np.where(ranked == frame_829_index0)[0][0]) + 1
        payload["correlation_gap_to_best"] = float(correlations[best_index0] - correlations[frame_829_index0])

    return payload


def load_or_compute_reference_session_match(
    reference_speaker: str,
    reference_image,
    session_data,
    artspeech_speaker: str,
    session: str,
    *,
    output_path: Path | None = None,
) -> dict[str, object]:
    output_path = output_path or default_match_output_path(reference_speaker, artspeech_speaker, session)
    if output_path.is_file():
        payload = json.loads(output_path.read_text(encoding="utf-8"))
        if (
            payload.get("vtnl_reference") == reference_speaker
            and payload.get("artspeech_speaker") == artspeech_speaker
            and payload.get("session") == session
            and int(payload.get("frame_count", -1)) == int(session_data.images.shape[0])
        ):
            return payload

    payload = compute_reference_session_match(
        reference_speaker=reference_speaker,
        reference_image=reference_image,
        session_data=session_data,
        artspeech_speaker=artspeech_speaker,
        session=session,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return payload


def serialize_contours(contours: dict[str, np.ndarray]) -> dict[str, list[list[float]]]:
    return {
        name: np.asarray(points, dtype=float).tolist()
        for name, points in sorted(contours.items())
    }


def deserialize_contours(contours_payload: dict[str, object]) -> dict[str, np.ndarray]:
    contours: dict[str, np.ndarray] = {}
    for name, points in contours_payload.items():
        array = np.asarray(points, dtype=float)
        if array.ndim != 2 or array.shape[1] != 2:
            raise ValueError(f"Contour {name!r} must have shape (N, 2), got {array.shape}")
        contours[str(name)] = array
    return contours


def normalize_grid_controls(
    contours: dict[str, np.ndarray],
    grid_controls: dict[str, object] | None = None,
) -> dict[str, dict[str, object]]:
    grid_controls = dict(grid_controls or {})
    include_payload = dict(grid_controls.get("include_in_grid", {}))
    constraint_payload = dict(grid_controls.get("constraint_group", {}))

    include_in_grid: dict[str, bool] = {}
    constraint_group: dict[str, str] = {}
    for name in sorted(contours):
        include = bool(include_payload.get(name, True))
        if name in GRID_CONTROL_REQUIRED_CONTOURS:
            include = True
        include_in_grid[name] = include

        raw_group = str(constraint_payload.get(name, "none")).strip().lower()
        group = raw_group if raw_group in GRID_CONSTRAINT_GROUPS else "none"
        if not include:
            group = "none"
        constraint_group[name] = group

    return {
        "include_in_grid": include_in_grid,
        "constraint_group": constraint_group,
    }


def controls_to_grid_contours(
    contours: dict[str, np.ndarray],
    grid_controls: dict[str, object] | None = None,
) -> dict[str, np.ndarray]:
    controls = normalize_grid_controls(contours, grid_controls)
    return {
        name: np.asarray(points, dtype=float)
        for name, points in contours.items()
        if controls["include_in_grid"].get(name, True) or name in GRID_CONTROL_REQUIRED_CONTOURS
    }


def controls_to_grid_constraints(
    contours: dict[str, np.ndarray],
    grid_controls: dict[str, object] | None = None,
) -> dict[str, list[np.ndarray]]:
    controls = normalize_grid_controls(contours, grid_controls)
    constraints = {
        "velum_paths": [],
        "pharynx_paths": [],
    }
    for name, points in contours.items():
        if not controls["include_in_grid"].get(name, True):
            continue
        group = controls["constraint_group"].get(name, "none")
        if group == "velum":
            constraints["velum_paths"].append(np.asarray(points, dtype=float))
        elif group == "pharynx":
            constraints["pharynx_paths"].append(np.asarray(points, dtype=float))
    return constraints


def save_source_annotation_json(
    path: Path,
    metadata: dict[str, object],
    contours: dict[str, np.ndarray],
) -> dict[str, object]:
    payload_metadata = dict(metadata)
    if "grid_controls" in payload_metadata:
        payload_metadata["grid_controls"] = normalize_grid_controls(contours, payload_metadata.get("grid_controls"))
    payload = {
        "metadata": payload_metadata,
        "contours": serialize_contours(contours),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return payload


def load_source_annotation_json(path: Path | str) -> dict[str, object]:
    path = Path(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    metadata = dict(payload.get("metadata", {}))
    contours = deserialize_contours(payload.get("contours", {}))
    metadata["grid_controls"] = normalize_grid_controls(contours, metadata.get("grid_controls"))
    return {
        "path": path,
        "metadata": metadata,
        "contours": contours,
    }
