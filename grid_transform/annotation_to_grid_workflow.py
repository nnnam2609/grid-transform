from __future__ import annotations

import json
import re
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

from grid_transform.apps.method4_transform import (
    LANDMARK_REPORT_ORDER,
    apply_tps,
    apply_transform,
    build_step1_anchors,
    build_step2_controls,
    compute_grid_line_errors,
    compute_metrics,
    compute_named_point_errors,
    extract_true_landmarks,
    fit_tps,
    map_landmarks,
    smooth_transformed_grid,
)
from grid_transform.apps.run_curated_u_annotation_batch import (
    BatchCase,
    CURATED_VTLN_DIR,
    DEFAULT_MANIFEST_CSV,
    load_cases as load_batch_cases,
    resolve_reference_bundle,
    resolve_speaker_root,
)
from grid_transform.artspeech_video import IntervalCursor, load_session_data, normalize_frame
from grid_transform.config import DEFAULT_OUTPUT_DIR, PROJECT_DIR, VT_SEG_CONTOURS_ROOT, VT_SEG_DATA_ROOT
from grid_transform.image_utils import as_grayscale_uint8
from grid_transform.io import load_frame_npy, load_frame_vtln
from grid_transform.source_annotation import (
    frame_correlation,
    load_source_annotation_json,
    project_reference_annotation_to_source,
    projected_reference_frame,
    save_source_annotation_json,
)
from grid_transform.transfer import DEFAULT_ARTICULATORS, resolve_common_articulators, smooth_transformed_contours, transform_contours
from grid_transform.vt import build_grid


DEFAULT_WORKSPACE_ROOT = DEFAULT_OUTPUT_DIR / "annotation_to_grid_transform"
DEFAULT_MAPPING_DOC = PROJECT_DIR / "docs" / "ARTSPEECH_MAPPING.md"
DEFAULT_NNUNET_TARGET_CASE = "2008-003^01-1791/test"
DEFAULT_NNUNET_TARGET_FRAME = 143020

SPEAKER_ROW_RE = re.compile(r"\|\s*`(?P<raw>\d+)`\s*\|\s*`(?P<speaker>P\d+)`\s*\|")
VTLN_SESSION_ROW_RE = re.compile(
    r"\|\s*`(?P<vtln>[0-9]+_s[0-9]+_[0-9]+)`\s*\|\s*`(?P<speaker>P\d+)`\s*\|\s*`(?P<session>S\d+)`\s*\|"
)
WORKSPACE_SAFE_RE = re.compile(r"[^A-Za-z0-9._-]+")


@dataclass(frozen=True)
class MappingInfo:
    speaker_map: dict[str, str]
    vtln_session_map: dict[str, tuple[str, str]]


@dataclass(frozen=True)
class TargetSelection:
    target_type: str
    nnunet_case: str
    nnunet_frame: int
    vtln_reference: str
    vtln_dir: Path


@dataclass(frozen=True)
class WorkspaceSelection:
    workspace_id: str
    workspace_dir: Path
    case: BatchCase
    artspeech_root: Path
    dataset_root: Path | None
    reference_speaker: str
    reference_image_path: Path
    reference_zip_path: Path
    reference_bundle_dir: Path
    target: TargetSelection
    source_alias_note: str


def sanitize_workspace_token(value: str) -> str:
    cleaned = WORKSPACE_SAFE_RE.sub("_", value.strip())
    return cleaned.strip("._-") or "workspace"


def parse_mapping_doc(path: Path = DEFAULT_MAPPING_DOC) -> MappingInfo:
    text = path.read_text(encoding="utf-8")
    speaker_map: dict[str, str] = {}
    vtln_session_map: dict[str, tuple[str, str]] = {}
    for line in text.splitlines():
        speaker_match = SPEAKER_ROW_RE.search(line)
        if speaker_match:
            speaker_map[speaker_match.group("raw")] = speaker_match.group("speaker")
        vtln_match = VTLN_SESSION_ROW_RE.search(line)
        if vtln_match:
            vtln_session_map[vtln_match.group("vtln")] = (
                vtln_match.group("speaker"),
                vtln_match.group("session"),
            )
    return MappingInfo(speaker_map=speaker_map, vtln_session_map=vtln_session_map)


def reference_name_for_case(case: BatchCase) -> str:
    if case.reference_bundle_name:
        return case.reference_bundle_name
    if case.annotation_source_path is not None:
        return case.annotation_source_path.stem
    return case.output_basename


def load_curated_cases(
    manifest_csv: Path = DEFAULT_MANIFEST_CSV,
    vtln_dir: Path = CURATED_VTLN_DIR,
) -> list[BatchCase]:
    return load_batch_cases(manifest_csv, vtln_dir)


def workspace_selection_to_payload(selection: WorkspaceSelection) -> dict[str, object]:
    return {
        "workspace_id": selection.workspace_id,
        "workspace_dir": str(selection.workspace_dir),
        "artspeech_root": str(selection.artspeech_root),
        "dataset_root": None if selection.dataset_root is None else str(selection.dataset_root),
        "reference_speaker": selection.reference_speaker,
        "reference_image_path": str(selection.reference_image_path),
        "reference_zip_path": str(selection.reference_zip_path),
        "reference_bundle_dir": str(selection.reference_bundle_dir),
        "source_alias_note": selection.source_alias_note,
        "case": {
            "output_basename": selection.case.output_basename,
            "speaker": selection.case.speaker,
            "raw_subject": selection.case.raw_subject,
            "session": selection.case.session,
            "frame_index_1based": selection.case.frame_index_1based,
            "annotation_status": selection.case.annotation_status,
        },
        "target": {
            "target_type": selection.target.target_type,
            "nnunet_case": selection.target.nnunet_case,
            "nnunet_frame": int(selection.target.nnunet_frame),
            "vtln_reference": selection.target.vtln_reference,
            "vtln_dir": str(selection.target.vtln_dir),
        },
    }


def payload_to_workspace_selection(payload: dict[str, object]) -> WorkspaceSelection:
    case_payload = dict(payload["case"])
    case = BatchCase(
        output_basename=str(case_payload["output_basename"]),
        speaker=str(case_payload["speaker"]),
        raw_subject=str(case_payload["raw_subject"]),
        session=str(case_payload["session"]),
        frame_index_1based=int(case_payload["frame_index_1based"]),
        annotation_status=str(case_payload.get("annotation_status", "")),
        annotation_source_path=None,
        reference_bundle_dir=None,
        reference_bundle_name=None,
    )
    target_payload = dict(payload["target"])
    target = TargetSelection(
        target_type=str(target_payload["target_type"]),
        nnunet_case=str(target_payload["nnunet_case"]),
        nnunet_frame=int(target_payload["nnunet_frame"]),
        vtln_reference=str(target_payload["vtln_reference"]),
        vtln_dir=Path(str(target_payload["vtln_dir"])),
    )
    return WorkspaceSelection(
        workspace_id=str(payload["workspace_id"]),
        workspace_dir=Path(str(payload["workspace_dir"])),
        case=case,
        artspeech_root=Path(str(payload["artspeech_root"])),
        dataset_root=(
            Path(str(payload["dataset_root"]))
            if payload.get("dataset_root") not in (None, "")
            else None
        ),
        reference_speaker=str(payload["reference_speaker"]),
        reference_image_path=Path(str(payload["reference_image_path"])),
        reference_zip_path=Path(str(payload["reference_zip_path"])),
        reference_bundle_dir=Path(str(payload["reference_bundle_dir"])),
        target=target,
        source_alias_note=str(payload.get("source_alias_note", "")),
    )


def save_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def load_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def source_snapshot_for_frame(session_data, frame_index1: int, reference_image) -> dict[str, object]:
    frame_index0 = frame_index1 - 1
    frame = normalize_frame(
        session_data.images[frame_index0],
        session_data.frame_min,
        session_data.frame_max,
    )
    source_shape = tuple(int(value) for value in session_data.images.shape[1:])
    projected_ref = projected_reference_frame(reference_image, source_shape)
    frame_times = (np.arange(session_data.images.shape[0], dtype=np.float64) + 0.5) / session_data.frame_rate
    current_time = float(frame_times[frame_index0])
    word_cursor = IntervalCursor(session_data.tiers[0].intervals if len(session_data.tiers) >= 1 else [])
    phoneme_cursor = IntervalCursor(session_data.tiers[1].intervals if len(session_data.tiers) >= 2 else [])
    sentence_cursor = IntervalCursor(session_data.sentences)
    return {
        "frame1": int(frame_index1),
        "time_sec": current_time,
        "correlation": frame_correlation(projected_ref, frame),
        "word": word_cursor.at(current_time),
        "phoneme": phoneme_cursor.at(current_time),
        "sentence": sentence_cursor.at(current_time),
    }


def build_workspace_selection(
    *,
    case: BatchCase,
    artspeech_root: Path,
    target: TargetSelection,
    workspace_root: Path = DEFAULT_WORKSPACE_ROOT,
    mapping: MappingInfo | None = None,
) -> WorkspaceSelection:
    try:
        dataset_root = resolve_speaker_root(artspeech_root, case.speaker)
    except FileNotFoundError:
        dataset_root = None
    reference_speaker, reference_bundle_dir, reference_image_path, reference_zip_path = resolve_reference_bundle(case)
    mapping = mapping or parse_mapping_doc()
    alias = mapping.vtln_session_map.get(reference_speaker)
    alias_note = (
        f"{reference_speaker} -> {alias[0]}/{alias[1]}"
        if alias is not None
        else f"{reference_speaker} -> {case.speaker}/{case.session}"
    )
    target_token = (
        f"nnunet_{sanitize_workspace_token(target.nnunet_case)}_{int(target.nnunet_frame):06d}"
        if target.target_type == "nnunet"
        else f"vtln_{sanitize_workspace_token(target.vtln_reference)}"
    )
    workspace_id = sanitize_workspace_token(f"{case.output_basename}__{target_token}")
    workspace_dir = workspace_root / workspace_id
    return WorkspaceSelection(
        workspace_id=workspace_id,
        workspace_dir=workspace_dir,
        case=case,
        artspeech_root=artspeech_root,
        dataset_root=dataset_root,
        reference_speaker=reference_speaker,
        reference_image_path=reference_image_path,
        reference_zip_path=reference_zip_path,
        reference_bundle_dir=reference_bundle_dir,
        target=target,
        source_alias_note=alias_note,
    )


def serialize_points_map(points: dict[str, np.ndarray | None]) -> dict[str, list[float] | None]:
    return {
        name: None if value is None else np.asarray(value, dtype=float).tolist()
        for name, value in sorted(points.items())
    }


def deserialize_points_map(payload: dict[str, object]) -> dict[str, np.ndarray | None]:
    result: dict[str, np.ndarray | None] = {}
    for name, value in payload.items():
        if value is None:
            result[str(name)] = None
        else:
            result[str(name)] = np.asarray(value, dtype=float)
    return result


def _clone_landmarks(landmarks: dict[str, np.ndarray | None]) -> dict[str, np.ndarray | None]:
    return {
        name: None if value is None else np.asarray(value, dtype=float).copy()
        for name, value in landmarks.items()
    }


def _apply_overrides(
    landmarks: dict[str, np.ndarray | None],
    overrides: dict[str, np.ndarray | None] | None,
) -> dict[str, np.ndarray | None]:
    merged = _clone_landmarks(landmarks)
    if overrides is None:
        return merged
    for name, value in overrides.items():
        merged[name] = None if value is None else np.asarray(value, dtype=float).copy()
    if "C" in merged:
        c_points = [merged.get(f"C{index}") for index in range(1, 7)]
        if all(point is not None for point in c_points):
            merged["C"] = np.vstack(c_points)
    if merged.get("L1") is None and merged.get("I1") is not None:
        merged["L1"] = np.asarray(merged["I1"], dtype=float)
    return merged


def _disable_landmarks(
    landmarks: dict[str, np.ndarray | None],
    disabled_landmarks: set[str] | None,
) -> dict[str, np.ndarray | None]:
    if not disabled_landmarks:
        return _clone_landmarks(landmarks)
    masked = _clone_landmarks(landmarks)
    for name in disabled_landmarks:
        if name in masked:
            masked[name] = None
    return masked


def _single_point_landmark_pairs(
    src_landmarks: dict[str, np.ndarray | None],
    dst_landmarks: dict[str, np.ndarray | None],
) -> list[tuple[str, np.ndarray, np.ndarray]]:
    pairs_by_source_key: dict[tuple[float, float], tuple[str, np.ndarray, np.ndarray]] = {}
    for name, src_value in src_landmarks.items():
        dst_value = dst_landmarks.get(name)
        if src_value is None or dst_value is None:
            continue
        src_arr = np.asarray(src_value, dtype=float)
        dst_arr = np.asarray(dst_value, dtype=float)
        if src_arr.ndim != 1 or dst_arr.ndim != 1 or src_arr.shape != (2,) or dst_arr.shape != (2,):
            continue
        key = tuple(np.round(src_arr, 6))
        displacement = float(np.linalg.norm(dst_arr - src_arr))
        current = pairs_by_source_key.get(key)
        if current is None:
            pairs_by_source_key[key] = (name, src_arr, dst_arr)
            continue
        _, current_src, current_dst = current
        current_displacement = float(np.linalg.norm(current_dst - current_src))
        if displacement >= current_displacement:
            pairs_by_source_key[key] = (name, src_arr, dst_arr)
    return list(pairs_by_source_key.values())


def _deform_grid_from_landmark_overrides(
    grid,
    raw_landmarks: dict[str, np.ndarray | None],
    updated_landmarks: dict[str, np.ndarray | None],
):
    pairs = _single_point_landmark_pairs(raw_landmarks, updated_landmarks)
    if not pairs:
        return grid
    if all(np.allclose(src_arr, dst_arr, atol=1e-6) for _, src_arr, dst_arr in pairs):
        return grid

    src_points = np.vstack([src_arr for _, src_arr, _ in pairs])
    dst_points = np.vstack([dst_arr for _, _, dst_arr in pairs])

    if len(src_points) >= 3:
        deformation = fit_tps(src_points, dst_points, smoothing=0.0)

        def apply_deformation(points):
            return apply_tps(deformation, points)

    else:
        mean_delta = np.mean(dst_points - src_points, axis=0)

        def apply_deformation(points):
            return np.asarray(points, dtype=float) + mean_delta

    deformed_horiz_raw = [apply_deformation(line) for line in grid.horiz_lines]
    deformed_horiz, deformed_vert = smooth_transformed_grid(
        deformed_horiz_raw,
        updated_landmarks,
        grid.n_vert,
        top_axis_passes=4,
    )

    left_pts = np.asarray([line[0] for line in deformed_horiz], dtype=float) if deformed_horiz else None
    right_pts = np.asarray([line[-1] for line in deformed_horiz], dtype=float) if deformed_horiz else None
    i_points = {
        name: None if updated_landmarks.get(name) is None else np.asarray(updated_landmarks[name], dtype=float).copy()
        for name in (f"I{index}" for index in range(1, 8))
    }
    cervical_centers = {
        f"c{index}": None if updated_landmarks.get(f"C{index}") is None else tuple(np.asarray(updated_landmarks[f"C{index}"], dtype=float))
        for index in range(1, 7)
    }

    return replace(
        grid,
        horiz_lines=[np.asarray(line, dtype=float).copy() for line in deformed_horiz],
        vert_lines=[np.asarray(line, dtype=float).copy() for line in deformed_vert],
        left_pts=left_pts,
        right_pts=right_pts,
        I1=None if updated_landmarks.get("I1") is None else np.asarray(updated_landmarks["I1"], dtype=float).copy(),
        L6=None if updated_landmarks.get("L6") is None else np.asarray(updated_landmarks["L6"], dtype=float).copy(),
        I_points=i_points,
        cervical_centers=cervical_centers,
        P1_point=None if updated_landmarks.get("P1") is None else np.asarray(updated_landmarks["P1"], dtype=float).copy(),
        M1_point=None if updated_landmarks.get("M1") is None else np.asarray(updated_landmarks["M1"], dtype=float).copy(),
        vt_curve=None if grid.vt_curve is None else np.asarray(apply_deformation(grid.vt_curve), dtype=float),
        spine_curve=None if grid.spine_curve is None else np.asarray(apply_deformation(grid.spine_curve), dtype=float),
    )


def _serialize_affine(transform: dict[str, np.ndarray]) -> dict[str, object]:
    return {
        "A": np.asarray(transform["A"], dtype=float).tolist(),
        "t": np.asarray(transform["t"], dtype=float).tolist(),
    }


def _deserialize_affine(payload: dict[str, object]) -> dict[str, np.ndarray]:
    return {
        "A": np.asarray(payload["A"], dtype=float),
        "t": np.asarray(payload["t"], dtype=float),
    }


def apply_transform_from_spec(spec: dict[str, object], pts, *, direction: str = "forward", stage: str = "final"):
    affine_key = "affine" if direction == "forward" else "inverse_affine"
    tps_key = "tps" if direction == "forward" else "inverse_tps"
    affine = _deserialize_affine(dict(spec[affine_key]))
    pts_arr = np.asarray(pts, dtype=float)
    affine_pts = apply_transform(affine, pts_arr)
    if stage == "affine":
        return affine_pts
    tps_payload = dict(spec[tps_key])
    tps = fit_tps(np.asarray(tps_payload["src_points"], dtype=float), np.asarray(tps_payload["dst_points"], dtype=float))
    return apply_tps(tps, affine_pts)


def build_transform_bundle(
    *,
    source_image,
    source_contours: dict[str, np.ndarray],
    target_image,
    target_contours: dict[str, np.ndarray],
    source_frame_number: int,
    target_frame_number: int,
    source_landmark_overrides: dict[str, np.ndarray | None] | None = None,
    target_landmark_overrides: dict[str, np.ndarray | None] | None = None,
    disabled_landmarks: set[str] | None = None,
    top_axis_passes: int = 4,
) -> dict[str, object]:
    source_grid_base = build_grid(source_image, source_contours, n_vert=9, n_points=250, frame_number=source_frame_number)
    target_grid_base = build_grid(target_image, target_contours, n_vert=9, n_points=250, frame_number=target_frame_number)

    disabled = {str(name) for name in (disabled_landmarks or set())}
    source_landmarks_raw = _disable_landmarks(extract_true_landmarks(source_grid_base), disabled)
    target_landmarks_raw = _disable_landmarks(extract_true_landmarks(target_grid_base), disabled)
    source_landmarks = _disable_landmarks(_apply_overrides(source_landmarks_raw, source_landmark_overrides), disabled)
    target_landmarks = _disable_landmarks(_apply_overrides(target_landmarks_raw, target_landmark_overrides), disabled)
    source_grid = _deform_grid_from_landmark_overrides(source_grid_base, source_landmarks_raw, source_landmarks)
    target_grid = _deform_grid_from_landmark_overrides(target_grid_base, target_landmarks_raw, target_landmarks)

    step0_errors = compute_named_point_errors(source_landmarks, target_landmarks, LANDMARK_REPORT_ORDER)
    step0_metrics = compute_metrics(source_landmarks, target_landmarks)
    step0_grid_errors = compute_grid_line_errors(source_grid.horiz_lines, source_grid.vert_lines, target_grid)

    step1_src, step1_dst, step1_labels = build_step1_anchors(source_landmarks, target_landmarks)
    step1_affine = {
        "A": np.asarray(np.eye(2), dtype=float),
        "t": np.asarray(np.zeros(2), dtype=float),
    }
    if len(step1_src) >= 3:
        from grid_transform.apps.method4_transform import estimate_affine

        step1_affine = estimate_affine(step1_src, step1_dst)

    def apply_affine_only(points):
        return apply_transform(step1_affine, points)

    step1_landmarks = map_landmarks(apply_affine_only, source_landmarks)
    step1_horiz_raw = [apply_affine_only(h) for h in source_grid.horiz_lines]
    step1_horiz, step1_vert = smooth_transformed_grid(
        step1_horiz_raw,
        step1_landmarks,
        source_grid.n_vert,
        top_axis_passes=top_axis_passes,
    )
    step1_errors = compute_named_point_errors(step1_landmarks, target_landmarks, LANDMARK_REPORT_ORDER)
    step1_metrics = compute_metrics(step1_landmarks, target_landmarks)
    step1_grid_errors = compute_grid_line_errors(step1_horiz, step1_vert, target_grid)

    step2_src, step2_dst, step2_labels = build_step2_controls(step1_landmarks, target_landmarks)
    step2_tps = fit_tps(step2_src, step2_dst, smoothing=0.0) if len(step2_src) >= 3 else None

    def apply_two_step(points):
        affine_pts = apply_affine_only(points)
        if step2_tps is None:
            return affine_pts
        return apply_tps(step2_tps, affine_pts)

    mapped_final = map_landmarks(apply_two_step, source_landmarks)
    final_horiz_raw = [apply_two_step(h) for h in source_grid.horiz_lines]
    final_horiz, final_vert = smooth_transformed_grid(
        final_horiz_raw,
        mapped_final,
        source_grid.n_vert,
        top_axis_passes=top_axis_passes,
    )
    final_errors = compute_named_point_errors(mapped_final, target_landmarks, LANDMARK_REPORT_ORDER)
    final_metrics = compute_metrics(mapped_final, target_landmarks)
    final_grid_errors = compute_grid_line_errors(final_horiz, final_vert, target_grid)

    articulators = resolve_common_articulators(target_contours, source_contours, defaults=DEFAULT_ARTICULATORS)
    affine_contours = smooth_transformed_contours(transform_contours(source_contours, apply_affine_only, articulators))
    final_contours = smooth_transformed_contours(transform_contours(source_contours, apply_two_step, articulators))

    inverse_bundle = build_inverse_transform_spec(source_landmarks, target_landmarks)

    transform_spec = {
        "source_shape": list(np.asarray(source_image).shape[:2]),
        "target_shape": list(np.asarray(target_image).shape[:2]),
        "affine": _serialize_affine(step1_affine),
        "inverse_affine": inverse_bundle["inverse_affine"],
        "tps": {
            "src_points": np.asarray(step2_src, dtype=float).tolist(),
            "dst_points": np.asarray(step2_dst, dtype=float).tolist(),
            "labels": list(step2_labels),
        },
        "inverse_tps": inverse_bundle["inverse_tps"],
        "step1_labels": list(step1_labels),
        "step2_labels": list(step2_labels),
        "source_landmarks": serialize_points_map(source_landmarks),
        "target_landmarks": serialize_points_map(target_landmarks),
        "source_landmark_overrides": serialize_points_map(source_landmark_overrides or {}),
        "target_landmark_overrides": serialize_points_map(target_landmark_overrides or {}),
    }
    transform_review = {
        "landmark_errors": {
            "raw": step0_errors,
            "affine": step1_errors,
            "final": final_errors,
        },
        "metrics": {
            "raw": step0_metrics,
            "affine": step1_metrics,
            "final": final_metrics,
        },
        "grid_errors": {
            "raw": {
                "horizontal": step0_grid_errors[0],
                "vertical": step0_grid_errors[1],
            },
            "affine": {
                "horizontal": step1_grid_errors[0],
                "vertical": step1_grid_errors[1],
            },
            "final": {
                "horizontal": final_grid_errors[0],
                "vertical": final_grid_errors[1],
            },
        },
        "step1_labels": list(step1_labels),
        "step2_labels": list(step2_labels),
    }

    return {
        "source_grid": source_grid,
        "target_grid": target_grid,
        "source_landmarks": source_landmarks,
        "target_landmarks": target_landmarks,
        "step1_landmarks": step1_landmarks,
        "final_landmarks": mapped_final,
        "step1_horiz": step1_horiz,
        "step1_vert": step1_vert,
        "final_horiz": final_horiz,
        "final_vert": final_vert,
        "affine_contours": affine_contours,
        "final_contours": final_contours,
        "articulators": articulators,
        "transform_spec": transform_spec,
        "transform_review": transform_review,
        "apply_affine_only": apply_affine_only,
        "apply_two_step": apply_two_step,
        "apply_inverse_two_step": inverse_bundle["apply_inverse_two_step"],
    }


def build_inverse_transform_spec(
    source_landmarks: dict[str, np.ndarray | None],
    target_landmarks: dict[str, np.ndarray | None],
) -> dict[str, object]:
    inverse_src, inverse_dst, _ = build_step1_anchors(target_landmarks, source_landmarks)
    from grid_transform.apps.method4_transform import estimate_affine

    inverse_affine = estimate_affine(inverse_src, inverse_dst) if len(inverse_src) >= 3 else {
        "A": np.eye(2, dtype=float),
        "t": np.zeros(2, dtype=float),
    }

    def apply_inverse_affine(points):
        return apply_transform(inverse_affine, points)

    inverse_step1_landmarks = map_landmarks(apply_inverse_affine, target_landmarks)
    inverse_step2_src, inverse_step2_dst, inverse_step2_labels = build_step2_controls(inverse_step1_landmarks, source_landmarks)
    inverse_tps = (
        fit_tps(inverse_step2_src, inverse_step2_dst, smoothing=0.0)
        if len(inverse_step2_src) >= 3
        else None
    )

    def apply_inverse_two_step(points):
        affine_pts = apply_inverse_affine(points)
        if inverse_tps is None:
            return affine_pts
        return apply_tps(inverse_tps, affine_pts)

    return {
        "inverse_affine": _serialize_affine(inverse_affine),
        "inverse_tps": {
            "src_points": np.asarray(inverse_step2_src, dtype=float).tolist(),
            "dst_points": np.asarray(inverse_step2_dst, dtype=float).tolist(),
            "labels": list(inverse_step2_labels),
        },
        "apply_inverse_two_step": apply_inverse_two_step,
    }


def source_annotation_metadata(selection: WorkspaceSelection, snapshot: dict[str, object], source_shape: tuple[int, int]) -> dict[str, object]:
    return {
        "role": "source",
        "workspace_id": selection.workspace_id,
        "artspeech_speaker": selection.case.speaker,
        "session": selection.case.session,
        "source_frame": selection.case.frame_index_1based,
        "time_sec": snapshot["time_sec"],
        "reference_speaker": selection.reference_speaker,
        "source_shape": list(source_shape),
        "correlation": snapshot["correlation"],
        "word": snapshot["word"],
        "phoneme": snapshot["phoneme"],
        "sentence": snapshot["sentence"],
        "dataset_root": None if selection.dataset_root is None else str(selection.dataset_root),
    }


def target_annotation_metadata(selection: WorkspaceSelection, target_shape: tuple[int, int]) -> dict[str, object]:
    return {
        "role": "target",
        "workspace_id": selection.workspace_id,
        "target_type": selection.target.target_type,
        "target_case": selection.target.nnunet_case,
        "target_frame": int(selection.target.nnunet_frame),
        "vtln_reference": selection.target.vtln_reference,
        "vtln_dir": str(selection.target.vtln_dir),
        "target_shape": list(target_shape),
    }


def _reference_only_source_snapshot(frame_index1: int, reference_image) -> dict[str, object]:
    frame = as_grayscale_uint8(reference_image)
    projected_ref = projected_reference_frame(reference_image, tuple(frame.shape[:2]))
    return {
        "frame1": int(frame_index1),
        "time_sec": 0.0,
        "correlation": frame_correlation(projected_ref, frame),
        "word": "",
        "phoneme": "",
        "sentence": "Reference-only source (no ArtSpeech session video available).",
    }


def load_source_context(selection: WorkspaceSelection) -> dict[str, object]:
    reference_image, reference_contours = load_frame_vtln(
        selection.reference_speaker,
        selection.reference_bundle_dir,
        validate_triplet_bundle=True,
    )
    reference_image_arr = as_grayscale_uint8(reference_image)
    reference_shape = tuple(int(value) for value in reference_image_arr.shape[:2])
    session_data = None
    source_video_error: str | None = None

    if selection.dataset_root is not None:
        try:
            session_data = load_session_data(selection.dataset_root, selection.case.speaker, selection.case.session)
        except Exception as exc:
            source_video_error = str(exc)
    else:
        source_video_error = (
            f"No ArtSpeech dataset root found for {selection.case.speaker}/{selection.case.session} "
            f"under {selection.artspeech_root}."
        )

    if session_data is None:
        projected_contours = project_reference_annotation_to_source(reference_contours, reference_shape, reference_shape)
        snapshot = _reference_only_source_snapshot(selection.case.frame_index_1based, reference_image_arr)
        return {
            "session_data": None,
            "source_frame": reference_image_arr,
            "reference_image": reference_image_arr.copy(),
            "reference_contours": reference_contours,
            "projected_contours": projected_contours,
            "snapshot": snapshot,
            "source_has_video": False,
            "source_video_error": source_video_error,
        }

    frame_index0 = selection.case.frame_index_1based - 1
    source_frame = normalize_frame(
        session_data.images[frame_index0],
        session_data.frame_min,
        session_data.frame_max,
    )
    projected_contours = project_reference_annotation_to_source(reference_contours, reference_shape, tuple(source_frame.shape[:2]))
    snapshot = source_snapshot_for_frame(session_data, selection.case.frame_index_1based, reference_image_arr)
    return {
        "session_data": session_data,
        "source_frame": source_frame,
        "reference_image": reference_image_arr.copy(),
        "reference_contours": reference_contours,
        "projected_contours": projected_contours,
        "snapshot": snapshot,
        "source_has_video": True,
        "source_video_error": None,
    }


def load_target_context(selection: WorkspaceSelection) -> dict[str, object]:
    if selection.target.target_type == "nnunet":
        image, contours = load_frame_npy(
            int(selection.target.nnunet_frame),
            VT_SEG_DATA_ROOT / selection.target.nnunet_case,
            VT_SEG_CONTOURS_ROOT / selection.target.nnunet_case,
        )
        label = f"nnUNet {selection.target.nnunet_case} frame {int(selection.target.nnunet_frame)}"
        frame_number = int(selection.target.nnunet_frame)
    else:
        image, contours = load_frame_vtln(
            selection.target.vtln_reference,
            selection.target.vtln_dir,
            validate_triplet_bundle=True,
        )
        label = f"VTLN {selection.target.vtln_reference}"
        frame_number = 0
    image_arr = as_grayscale_uint8(image)
    return {
        "target_image": image_arr,
        "target_contours": contours,
        "target_label": label,
        "target_frame_number": frame_number,
    }


def save_annotation_state(
    path: Path,
    metadata: dict[str, object],
    contours: dict[str, np.ndarray],
) -> dict[str, object]:
    return save_source_annotation_json(path, metadata, contours)


def load_annotation_state_if_available(path: Path) -> dict[str, object] | None:
    if not path.is_file():
        return None
    try:
        return load_source_annotation_json(path)
    except Exception:
        return None


def step_file_paths(workspace_dir: Path) -> dict[str, Path]:
    return {
        "selection": workspace_dir / "workspace_selection.latest.json",
        "source_annotation": workspace_dir / "source_annotation.latest.json",
        "target_annotation": workspace_dir / "target_annotation.latest.json",
        "landmark_overrides": workspace_dir / "landmark_overrides.latest.json",
        "transform_spec": workspace_dir / "transform_spec.latest.json",
        "transform_review": workspace_dir / "transform_review.latest.json",
        "native_preview": workspace_dir / "step2_native.latest.png",
        "affine_preview": workspace_dir / "step2_affine.latest.png",
        "final_preview": workspace_dir / "step2_final.latest.png",
        "overview_preview": workspace_dir / "step2_overview.latest.png",
    }


def load_workspace_selection_if_available(path: Path) -> WorkspaceSelection | None:
    if not path.is_file():
        return None
    return payload_to_workspace_selection(load_json(path))
