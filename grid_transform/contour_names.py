from __future__ import annotations

from collections.abc import Iterable


def normalize_contour_name(name: str, known_labels: Iterable[str]) -> str:
    labels = tuple(sorted({str(label) for label in known_labels}, key=len, reverse=True))
    if name in labels:
        return name
    for label in labels:
        if name.endswith(f"_{label}"):
            return label
    return name
