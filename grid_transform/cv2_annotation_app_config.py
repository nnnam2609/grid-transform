from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

from grid_transform.config import PROJECT_DIR


APP_CONFIG_PATH = PROJECT_DIR / "config.yaml"
APP_CONFIG_SECTION = "cv2_annotation_to_grid_transform"
VALID_CACHE_MODES = {"startup", "first-load"}
VALID_WINDOW_MODES = {"full-height", "fullscreen", "fixed"}

RENDER_WORKERS_RANGE = (1, 64)
RENDER_PREFETCH_RANGE = (0, 1024)

UI_LIMITS: dict[str, tuple[int, int]] = {
    "editor_panel_width": (240, 4096),
    "editor_panel_height": (240, 4096),
    "review_pane_width": (240, 4096),
    "review_pane_height": (240, 4096),
    "editor_sidebar_width": (160, 2048),
    "editor_footer_height": (48, 2048),
    "review_footer_height": (48, 2048),
    "editor_loupe_size": (96, 1024),
    "pane_gap": (0, 256),
}


@dataclass(frozen=True)
class UIConfig:
    editor_panel_width: int = 960
    editor_panel_height: int = 960
    editor_sidebar_width: int = 320
    editor_footer_height: int = 88
    editor_loupe_size: int = 240
    review_pane_width: int = 480
    review_pane_height: int = 340
    pane_gap: int = 12
    review_footer_height: int = 176

    @classmethod
    def from_payload(cls, payload: dict[str, object], *, config_path: Path) -> "UIConfig":
        defaults = cls()
        return cls(
            editor_panel_width=resolve_config_int(
                payload.get("editor_panel_width"),
                default=defaults.editor_panel_width,
                field_name="editor_panel_width",
                config_path=config_path,
                min_value=UI_LIMITS["editor_panel_width"][0],
                max_value=UI_LIMITS["editor_panel_width"][1],
            ),
            editor_panel_height=resolve_config_int(
                payload.get("editor_panel_height"),
                default=defaults.editor_panel_height,
                field_name="editor_panel_height",
                config_path=config_path,
                min_value=UI_LIMITS["editor_panel_height"][0],
                max_value=UI_LIMITS["editor_panel_height"][1],
            ),
            editor_sidebar_width=resolve_config_int(
                payload.get("editor_sidebar_width"),
                default=defaults.editor_sidebar_width,
                field_name="editor_sidebar_width",
                config_path=config_path,
                min_value=UI_LIMITS["editor_sidebar_width"][0],
                max_value=UI_LIMITS["editor_sidebar_width"][1],
            ),
            editor_footer_height=resolve_config_int(
                payload.get("editor_footer_height"),
                default=defaults.editor_footer_height,
                field_name="editor_footer_height",
                config_path=config_path,
                min_value=UI_LIMITS["editor_footer_height"][0],
                max_value=UI_LIMITS["editor_footer_height"][1],
            ),
            editor_loupe_size=resolve_config_int(
                payload.get("editor_loupe_size"),
                default=defaults.editor_loupe_size,
                field_name="editor_loupe_size",
                config_path=config_path,
                min_value=UI_LIMITS["editor_loupe_size"][0],
                max_value=UI_LIMITS["editor_loupe_size"][1],
            ),
            review_pane_width=resolve_config_int(
                payload.get("review_pane_width"),
                default=defaults.review_pane_width,
                field_name="review_pane_width",
                config_path=config_path,
                min_value=UI_LIMITS["review_pane_width"][0],
                max_value=UI_LIMITS["review_pane_width"][1],
            ),
            review_pane_height=resolve_config_int(
                payload.get("review_pane_height"),
                default=defaults.review_pane_height,
                field_name="review_pane_height",
                config_path=config_path,
                min_value=UI_LIMITS["review_pane_height"][0],
                max_value=UI_LIMITS["review_pane_height"][1],
            ),
            pane_gap=resolve_config_int(
                payload.get("pane_gap"),
                default=defaults.pane_gap,
                field_name="pane_gap",
                config_path=config_path,
                min_value=UI_LIMITS["pane_gap"][0],
                max_value=UI_LIMITS["pane_gap"][1],
            ),
            review_footer_height=resolve_config_int(
                payload.get("review_footer_height"),
                default=defaults.review_footer_height,
                field_name="review_footer_height",
                config_path=config_path,
                min_value=UI_LIMITS["review_footer_height"][0],
                max_value=UI_LIMITS["review_footer_height"][1],
            ),
        )


def _source_label(source: str | Path | None) -> str:
    if source is None:
        return "value"
    return str(source)


def bounded_int(
    name: str,
    value: object,
    *,
    min_value: int | None = None,
    max_value: int | None = None,
    source: str | Path | None = None,
) -> int:
    source_label = _source_label(source)
    if isinstance(value, bool):
        raise ValueError(f"{name} in {source_label} must be an integer, got {value!r}.")
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} in {source_label} must be an integer, got {value!r}.") from exc
    if min_value is not None and parsed < min_value:
        raise ValueError(f"{name} in {source_label} must be >= {min_value}, got {parsed}.")
    if max_value is not None and parsed > max_value:
        raise ValueError(f"{name} in {source_label} must be <= {max_value}, got {parsed}.")
    return parsed


def positive_int(name: str, value: object, *, max_value: int | None = None, source: str | Path | None = None) -> int:
    return bounded_int(name, value, min_value=1, max_value=max_value, source=source)


def non_negative_int(name: str, value: object, *, max_value: int | None = None, source: str | Path | None = None) -> int:
    return bounded_int(name, value, min_value=0, max_value=max_value, source=source)


def resolve_config_int(
    value: object,
    *,
    default: int,
    field_name: str,
    config_path: Path,
    min_value: int | None = None,
    max_value: int | None = None,
) -> int:
    raw_value = default if value in (None, "") else value
    return bounded_int(
        field_name,
        raw_value,
        min_value=min_value,
        max_value=max_value,
        source=f"config file {config_path}",
    )


def argparse_bounded_int(name: str, *, min_value: int | None = None, max_value: int | None = None):
    def _parse(raw: str) -> int:
        try:
            return bounded_int(name, raw, min_value=min_value, max_value=max_value, source="command line")
        except ValueError as exc:
            raise argparse.ArgumentTypeError(str(exc)) from exc

    return _parse


def resolve_config_path(value: object, *, default: Path, config_path: Path) -> Path:
    if value in (None, ""):
        return default
    candidate = Path(str(value))
    if candidate.is_absolute():
        return candidate
    return (config_path.parent / candidate).resolve()


def resolve_config_str(value: object, *, default: str) -> str:
    if value in (None, ""):
        return default
    return str(value)


def resolve_choice(
    value: object,
    *,
    default: str,
    valid: set[str],
    field_name: str,
    config_path: Path,
) -> str:
    resolved = resolve_config_str(value, default=default)
    if resolved not in valid:
        raise SystemExit(f"Invalid {field_name}={resolved!r} in config file {config_path}. Valid choices: {sorted(valid)}")
    return resolved


def read_yaml_config(path: Path) -> dict[str, object]:
    if not path.is_file():
        return {}
    try:
        import yaml
    except ModuleNotFoundError as exc:
        raise SystemExit(
            f"config file {path} requires PyYAML. Install dependencies from requirements.txt first."
        ) from exc

    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise SystemExit(f"config file {path} must contain a YAML mapping at the top level.")
    return payload


def load_cv2_app_config_defaults(config_path: Path) -> tuple[dict[str, object], UIConfig]:
    payload = read_yaml_config(config_path)
    section = payload.get(APP_CONFIG_SECTION, {}) or {}
    if not isinstance(section, dict):
        raise SystemExit(f"Section {APP_CONFIG_SECTION!r} in {config_path} must be a mapping.")
    defaults_payload = section.get("defaults", {}) or {}
    ui_payload = section.get("ui", {}) or {}
    if not isinstance(defaults_payload, dict):
        raise SystemExit(f"Section {APP_CONFIG_SECTION}.defaults in {config_path} must be a mapping.")
    if not isinstance(ui_payload, dict):
        raise SystemExit(f"Section {APP_CONFIG_SECTION}.ui in {config_path} must be a mapping.")
    return defaults_payload, UIConfig.from_payload(ui_payload, config_path=config_path)


def render_validation_errors(*, workers: object, prefetch: object, source: str | Path | None) -> list[str]:
    errors: list[str] = []
    try:
        positive_int("render_workers", workers, max_value=RENDER_WORKERS_RANGE[1], source=source)
    except ValueError as exc:
        errors.append(str(exc))
    try:
        non_negative_int("render_prefetch", prefetch, max_value=RENDER_PREFETCH_RANGE[1], source=source)
    except ValueError as exc:
        errors.append(str(exc))
    return errors


def validate_render_settings(*, workers: object, prefetch: object, source: str | Path | None) -> tuple[int, int]:
    errors = render_validation_errors(workers=workers, prefetch=prefetch, source=source)
    if errors:
        raise ValueError("\n".join(errors))
    return (
        positive_int("render_workers", workers, max_value=RENDER_WORKERS_RANGE[1], source=source),
        non_negative_int("render_prefetch", prefetch, max_value=RENDER_PREFETCH_RANGE[1], source=source),
    )
