from __future__ import annotations

from pathlib import Path


VALID_CONFIG_LANDMARK_NAMES = frozenset(
    {
        *(f"I{i}" for i in range(1, 8)),
        "P1",
        *(f"C{i}" for i in range(1, 7)),
        "M1",
        "L6",
    }
)


def resolve_disabled_landmarks_config(value: object, *, config_path: Path) -> tuple[str, ...]:
    if value in (None, "", []):
        return ()
    if isinstance(value, str):
        raw_items = [item.strip() for item in value.replace(",", " ").split()]
    elif isinstance(value, (list, tuple, set)):
        raw_items = [str(item).strip() for item in value]
    else:
        raise SystemExit(
            f"disabled_landmarks in config file {config_path} must be a string or a list of landmark names."
        )
    cleaned = [item.upper() for item in raw_items if item]
    invalid = sorted({item for item in cleaned if item not in VALID_CONFIG_LANDMARK_NAMES})
    if invalid:
        raise SystemExit(
            f"Invalid disabled_landmarks in config file {config_path}: {invalid}. "
            f"Valid names: {sorted(VALID_CONFIG_LANDMARK_NAMES)}"
        )
    return tuple(dict.fromkeys(cleaned))
