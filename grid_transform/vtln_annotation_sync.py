from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from grid_transform.config import DEFAULT_OUTPUT_DIR, DEFAULT_VTLN_DIR
from grid_transform.io import VTLN_TRIPLET_SHAPE
from grid_transform.source_annotation import load_source_annotation_json
from grid_transform.vtln_bundle import scale_contours_to_triplet_space, write_annotation_zip


@dataclass(frozen=True)
class TempAnnotationSyncSpec:
    path: Path
    role: str
    reference_name: str
    vtln_dir: Path
    source_shape: tuple[int, int]
    metadata: dict[str, object]
    contours: dict[str, np.ndarray]


def resolve_vtln_data_dir(vtln_dir: Path) -> Path:
    vtln_dir = Path(vtln_dir)
    if vtln_dir.name == "data":
        return vtln_dir
    nested = vtln_dir / "data"
    if nested.is_dir():
        return nested
    return vtln_dir


def parse_shape2d(shape_raw: object) -> tuple[int, int] | None:
    try:
        source_h = int(shape_raw[0])
        source_w = int(shape_raw[1])
    except (TypeError, ValueError, IndexError):
        return None
    if source_h <= 0 or source_w <= 0:
        return None
    return source_h, source_w


def annotation_shape_from_metadata(
    metadata: dict[str, object],
    *field_names: str,
) -> tuple[int, int] | None:
    for field_name in field_names:
        shape = parse_shape2d(metadata.get(field_name))
        if shape is not None:
            return shape
    return None


def contour_bounds(contours: dict[str, np.ndarray] | dict[str, object]) -> tuple[float, float, float, float] | None:
    min_x = float("inf")
    min_y = float("inf")
    max_x = float("-inf")
    max_y = float("-inf")
    found = False
    for points in contours.values():
        array = np.asarray(points, dtype=float)
        if array.ndim != 2 or array.shape[1] != 2 or len(array) == 0:
            continue
        found = True
        min_x = min(min_x, float(array[:, 0].min()))
        min_y = min(min_y, float(array[:, 1].min()))
        max_x = max(max_x, float(array[:, 0].max()))
        max_y = max(max_y, float(array[:, 1].max()))
    if not found:
        return None
    return min_x, min_y, max_x, max_y


def source_annotation_payload_is_plausible(
    payload: dict[str, object] | None,
    *,
    current_source_shape: tuple[int, int] | None = None,
) -> bool:
    if not payload:
        return False
    metadata = payload.get("metadata", {})
    resolved_shape = parse_shape2d(current_source_shape)
    if resolved_shape is None:
        resolved_shape = parse_shape2d(metadata.get("source_shape") or metadata.get("reference_shape"))
    if resolved_shape is None:
        return True
    source_h, source_w = resolved_shape
    bounds = contour_bounds(payload.get("contours", {}))
    if bounds is None:
        return True
    min_x, min_y, max_x, max_y = bounds
    tolerance_px = 6.0
    tolerance_scale = 1.15
    return (
        min_x >= -tolerance_px
        and min_y >= -tolerance_px
        and max_x <= source_w * tolerance_scale + tolerance_px
        and max_y <= source_h * tolerance_scale + tolerance_px
    )


def discover_saved_source_annotation_paths(
    *,
    output_root: Path = DEFAULT_OUTPUT_DIR,
) -> list[Path]:
    patterns = (
        (output_root / "annotation_to_grid_transform", "source_annotation.latest.json"),
        (output_root / "source_annotation_edits", "edited_annotation.json"),
    )
    paths: list[Path] = []
    for root, filename in patterns:
        if not root.is_dir():
            continue
        paths.extend(root.rglob(filename))
    return sorted(set(paths))


def discover_saved_target_annotation_paths(
    *,
    output_root: Path = DEFAULT_OUTPUT_DIR,
) -> list[Path]:
    root = output_root / "annotation_to_grid_transform"
    if not root.is_dir():
        return []
    paths = list(root.rglob("target_annotation.latest.json"))
    promotion_root = root / "_vtln_promotions" / "target"
    if promotion_root.is_dir():
        paths.extend(promotion_root.rglob("source_annotation.latest.json"))
    return sorted(set(paths))


def load_annotation_state_if_available(path: Path) -> dict[str, object] | None:
    if not path.is_file():
        return None
    try:
        return load_source_annotation_json(path)
    except Exception:
        return None


def find_latest_source_annotation(
    *,
    artspeech_speaker: str,
    session: str,
    source_frame: int,
    current_source_shape: tuple[int, int] | None = None,
    output_root: Path = DEFAULT_OUTPUT_DIR,
) -> dict[str, object] | None:
    best_payload: dict[str, object] | None = None
    best_mtime = float("-inf")
    expected_key = (
        str(artspeech_speaker),
        str(session),
        int(source_frame),
    )
    for path in discover_saved_source_annotation_paths(output_root=output_root):
        payload = load_annotation_state_if_available(path)
        if payload is None:
            continue
        metadata = payload.get("metadata", {})
        if str(metadata.get("promoted_from_role") or "").strip().lower() == "target":
            continue
        if not source_annotation_payload_is_plausible(payload, current_source_shape=current_source_shape):
            continue
        payload_key = (
            str(metadata.get("artspeech_speaker") or ""),
            str(metadata.get("session") or ""),
            int(metadata.get("source_frame") or 0),
        )
        if payload_key != expected_key:
            continue
        try:
            mtime = path.stat().st_mtime
        except OSError:
            continue
        if mtime > best_mtime:
            best_payload = dict(payload)
            best_payload["path"] = str(path)
            best_mtime = mtime
    return best_payload


def vtln_zip_path(vtln_dir: Path, reference_name: str) -> Path:
    return resolve_vtln_data_dir(vtln_dir) / f"{reference_name}.zip"


def write_source_annotation_to_vtln(
    *,
    vtln_dir: Path,
    reference_name: str,
    contours: dict[str, np.ndarray],
    source_shape: tuple[int, int],
) -> Path:
    vtln_dir = resolve_vtln_data_dir(vtln_dir)
    scaled_contours = scale_contours_to_triplet_space(
        contours,
        source_shape,
        tuple(int(value) for value in VTLN_TRIPLET_SHAPE[:2]),
    )
    zip_path = vtln_zip_path(vtln_dir, reference_name)
    write_annotation_zip(
        zip_path,
        reference_name,
        scaled_contours,
        dry_run=False,
    )
    return zip_path


def delete_temp_annotation_file(path: Path, *, stop_root: Path = DEFAULT_OUTPUT_DIR) -> None:
    try:
        path.unlink(missing_ok=True)
    except OSError:
        return
    try:
        stop_root = Path(stop_root).resolve()
    except OSError:
        return
    parent = path.parent
    while parent.exists():
        try:
            if parent.resolve() == stop_root:
                break
        except OSError:
            break
        try:
            parent.rmdir()
        except OSError:
            break
        parent = parent.parent


def promote_temp_annotation_to_vtln(
    *,
    path: Path,
    reference_name: str,
    vtln_dir: Path,
    contours: dict[str, np.ndarray],
    source_shape: tuple[int, int],
    stop_root: Path = DEFAULT_OUTPUT_DIR,
) -> bool:
    if not path.is_file():
        return False
    zip_path = vtln_zip_path(vtln_dir, reference_name)
    try:
        json_mtime = path.stat().st_mtime
    except OSError:
        return False
    try:
        zip_mtime = zip_path.stat().st_mtime
    except OSError:
        zip_mtime = float("-inf")
    promoted = json_mtime > zip_mtime
    if promoted:
        write_source_annotation_to_vtln(
            vtln_dir=vtln_dir,
            reference_name=reference_name,
            contours=contours,
            source_shape=source_shape,
        )
    delete_temp_annotation_file(path, stop_root=stop_root)
    return promoted


def build_temp_annotation_sync_spec(
    path: Path,
    *,
    default_source_vtln_dir: Path = DEFAULT_VTLN_DIR,
) -> TempAnnotationSyncSpec | None:
    payload = load_annotation_state_if_available(path)
    if payload is None:
        return None
    metadata = dict(payload.get("metadata", {}))
    role = str(metadata.get("role") or "").strip().lower()
    promoted_from_role = str(metadata.get("promoted_from_role") or "").strip().lower()
    effective_role = promoted_from_role or role or "source"
    if effective_role == "target":
        target_type = str(metadata.get("target_type") or "vtln").strip().lower()
        reference_name = str(
            metadata.get("vtln_reference")
            or metadata.get("target_reference")
            or metadata.get("reference_speaker")
            or ""
        ).strip()
        if target_type != "vtln" or not reference_name:
            return None
        source_shape = annotation_shape_from_metadata(metadata, "target_shape", "source_shape", "reference_shape")
        if source_shape is None:
            return None
        vtln_dir = Path(str(metadata.get("vtln_dir") or default_source_vtln_dir))
        return TempAnnotationSyncSpec(
            path=path,
            role="target",
            reference_name=reference_name,
            vtln_dir=resolve_vtln_data_dir(vtln_dir),
            source_shape=source_shape,
            metadata=metadata,
            contours=payload["contours"],
        )

    reference_name = str(metadata.get("reference_speaker") or "").strip()
    if not reference_name:
        return None
    if not source_annotation_payload_is_plausible(payload):
        return None
    source_shape = annotation_shape_from_metadata(metadata, "source_shape", "reference_shape")
    if source_shape is None:
        return None
    return TempAnnotationSyncSpec(
        path=path,
        role="source",
        reference_name=reference_name,
        vtln_dir=resolve_vtln_data_dir(default_source_vtln_dir),
        source_shape=source_shape,
        metadata=metadata,
        contours=payload["contours"],
    )
