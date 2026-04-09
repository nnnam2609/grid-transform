from __future__ import annotations

from copy import deepcopy
from pathlib import Path

from grid_transform.cv2_annotation_app_config import APP_CONFIG_PATH, read_yaml_config


SECTION_NAME = "slide_p7_nnunet_exports"
DEFAULT_CONFIG: dict[str, object] = {
    "palette": {
        "source_grid": "#35c9ff",
        "target_grid": "#ffd166",
        "connector": "#f8fafc",
        "target_landmark": "#e5e7eb",
        "affine_anchor": "#fb8500",
        "tps_extra": "#80ed99",
        "other_landmark": "#ffd166",
        "rmse_text": "#f8fafc",
    },
    "grid_build_landmarks": {
        "font_size": 9.2,
        "marker_edge_width": 1.2,
        "default_offset": [10.0, -10.0],
        "point_offsets": {
            "I1": [-18.0, -8.0],
            "I2": [-12.0, -10.0],
            "I3": [0.0, -12.0],
            "I4": [8.0, -10.0],
            "I5": [10.0, -2.0],
            "I6": [10.0, 6.0],
            "I7": [10.0, -6.0],
            "P1": [10.0, 10.0],
            "M1": [10.0, -8.0],
            "L6": [10.0, 10.0],
            "L3": [-12.0, 2.0],
            "L4": [-12.0, 2.0],
            "L5": [-12.0, 2.0],
        },
        "point_sizes": {
            "default": 62.0,
            "P1": 78.0,
            "L6": 72.0,
            "L3": 54.0,
            "L4": 54.0,
            "L5": 54.0,
        },
    },
    "step2_native": {
        "marker_radius": 3,
        "marker_highlight_radius": 4,
        "label_font_scale": 0.45,
        "label_thickness": 1,
        "default_label_offset": [8, -8],
        "label_positions": {
            "source_native": {},
            "target_native": {},
        },
        "label_offsets": {
            "source_native": {
                "I1": [-16, -14],
                "I2": [-6, -16],
                "I3": [4, -16],
                "I4": [6, -16],
                "I5": [6, -16],
                "M1": [-24, -2],
                "L3": [-24, 4],
                "L4": [-24, 4],
                "L5": [-24, 4],
                "L6": [-24, 4],
            },
            "target_native": {
                "I1": [-16, -14],
                "I2": [-6, -16],
                "I3": [4, -16],
                "I4": [6, -16],
                "I5": [6, -16],
                "M1": [-24, -2],
                "L3": [-24, 4],
                "L4": [-24, 4],
                "L5": [-24, 4],
                "L6": [-24, 4],
            },
        },
    },
    "transform_sequence": {
        "connector_alpha": 0.22,
        "connector_thickness": 1,
        "grid_thickness_outer": 3,
        "grid_thickness_inner": 1,
        "target_marker_radius": 3,
        "source_marker_radius": 3,
        "show_labels": True,
        "label_font_scale": 0.42,
        "label_thickness": 1,
        "default_label_offset": [7, -7],
        "label_positions": {
            "raw_overlay": {},
            "affine_overlay": {},
            "tps_overlay": {},
            "affine_source_only": {},
            "tps_source_only": {},
        },
        "label_offsets": {
            "raw_overlay": {
                "I1": [-16, -12],
                "I2": [-6, -14],
                "I3": [4, -14],
                "I4": [6, -14],
                "I5": [6, -14],
                "M1": [-22, -2],
                "L3": [-22, 4],
                "L4": [-22, 4],
                "L5": [-22, 4],
                "L6": [-22, 4],
            },
            "affine_overlay": {
                "I1": [-16, -12],
                "I2": [-6, -14],
                "I3": [4, -14],
                "I4": [6, -14],
                "I5": [6, -14],
                "M1": [-22, -2],
                "L3": [-22, 4],
                "L4": [-22, 4],
                "L5": [-22, 4],
                "L6": [-22, 4],
            },
            "tps_overlay": {
                "I1": [-16, -12],
                "I2": [-6, -14],
                "I3": [4, -14],
                "I4": [6, -14],
                "I5": [6, -14],
                "M1": [-22, -2],
                "L3": [-22, 4],
                "L4": [-22, 4],
                "L5": [-22, 4],
                "L6": [-22, 4],
            },
            "affine_source_only": {
                "I1": [-16, -12],
                "I2": [-6, -14],
                "I3": [4, -14],
                "I4": [6, -14],
                "I5": [6, -14],
                "M1": [-22, -2],
                "L3": [-22, 4],
                "L4": [-22, 4],
                "L5": [-22, 4],
                "L6": [-22, 4],
            },
            "tps_source_only": {
                "I1": [-16, -12],
                "I2": [-6, -14],
                "I3": [4, -14],
                "I4": [6, -14],
                "I5": [6, -14],
                "M1": [-22, -2],
                "L3": [-22, 4],
                "L4": [-22, 4],
                "L5": [-22, 4],
                "L6": [-22, 4],
            },
        },
    },
}


def _deep_merge(base: dict[str, object], override: dict[str, object]) -> dict[str, object]:
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_merge(base[key], value)  # type: ignore[index]
        else:
            base[key] = value
    return base


def load_slide_p7_nnunet_config(config_path: Path = APP_CONFIG_PATH) -> dict[str, object]:
    payload = read_yaml_config(config_path)
    section = payload.get(SECTION_NAME, {}) or {}
    if not isinstance(section, dict):
        raise SystemExit(f"Section {SECTION_NAME!r} in {config_path} must be a mapping.")
    merged = deepcopy(DEFAULT_CONFIG)
    return _deep_merge(merged, section)


def nested_dict(payload: dict[str, object], *keys: str) -> dict[str, object]:
    current: object = payload
    for key in keys:
        if not isinstance(current, dict):
            return {}
        current = current.get(key, {})
    return current if isinstance(current, dict) else {}


def nested_value(payload: dict[str, object], *keys: str, default):
    current: object = payload
    for key in keys:
        if not isinstance(current, dict):
            return default
        current = current.get(key, default)
    return current


def xy_tuple(value: object, default: tuple[float, float]) -> tuple[float, float]:
    if isinstance(value, (list, tuple)) and len(value) >= 2:
        try:
            return float(value[0]), float(value[1])
        except (TypeError, ValueError):
            return default
    return default


def stage_label_position(
    config: dict[str, object],
    section_name: str,
    stage_name: str,
    label_name: str,
) -> tuple[float, float] | None:
    raw = nested_value(config, section_name, "label_positions", stage_name, label_name, default=None)
    if raw is None:
        return None
    position = xy_tuple(raw, default=(float("nan"), float("nan")))
    if any(value != value for value in position):
        return None
    return position


def stage_label_offset(
    config: dict[str, object],
    section_name: str,
    stage_name: str,
    label_name: str,
    *,
    default_offset: tuple[float, float],
) -> tuple[float, float]:
    raw = nested_value(config, section_name, "label_offsets", stage_name, label_name, default=None)
    if raw is None:
        return default_offset
    return xy_tuple(raw, default=default_offset)
