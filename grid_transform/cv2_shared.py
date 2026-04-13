from __future__ import annotations

import ctypes
import ctypes.wintypes
import os

import numpy as np

from grid_transform.config import TONGUE_COLOR


WINDOW_MARGIN = 36

CONTOUR_COLORS = {
    "c1": "#457b9d",
    "c2": "#4d7ea8",
    "c3": "#5c96bc",
    "c4": "#7aa6c2",
    "c5": "#90b8d8",
    "c6": "#b8d8f2",
    "incisior-hard-palate": "#ef476f",
    "mandible-incisior": "#f4a261",
    "pharynx": "#118ab2",
    "soft-palate": "#fb8500",
    "soft-palate-midline": "#8338ec",
    "tongue": TONGUE_COLOR,
}

SOURCE_GRID_STYLE = {
    "horiz_color": "#3a86ff",
    "vert_color": "#ffbe0b",
    "vt_color": "#8338ec",
    "spine_color": "#2a9d8f",
    "guide_color": "#fb5607",
    "i_color": "#8338ec",
    "c_color": "#2a9d8f",
    "p1_color": "#fb5607",
    "m1_color": "#8338ec",
    "l6_color": "#80ed99",
    "interp_color": "#3a86ff",
    "mandible_color": "#e76f51",
    "tongue_color": TONGUE_COLOR,
}

TARGET_GRID_STYLE = {
    "horiz_color": "#20c997",
    "vert_color": "#ffd166",
    "vt_color": "#ef476f",
    "spine_color": "#118ab2",
    "guide_color": "#ff006e",
    "i_color": "#ef476f",
    "c_color": "#118ab2",
    "p1_color": "#ff006e",
    "m1_color": "#ef476f",
    "l6_color": "#06d6a0",
    "interp_color": "#20c997",
    "mandible_color": "#f4a261",
    "tongue_color": TONGUE_COLOR,
}


def source_text_block(snapshot: dict[str, object], reference_speaker: str) -> list[str]:
    sentence = " ".join(str(snapshot["sentence"]).split()) if snapshot.get("sentence") else "-"
    if len(sentence) > 56:
        sentence = sentence[:53] + "..."
    return [
        f"ref: {reference_speaker}",
        f"frame: {snapshot['frame1']}  t={snapshot['time_sec']:.4f}s",
        f"corr: {snapshot['correlation']:.4f}",
        f"word: {snapshot['word'] or '-'}  phoneme: {snapshot['phoneme'] or '-'}",
        f"sent: {sentence}",
    ]


def hex_to_bgr(color: str) -> tuple[int, int, int]:
    stripped = color.lstrip("#")
    if len(stripped) != 6:
        return (255, 255, 255)
    red = int(stripped[0:2], 16)
    green = int(stripped[2:4], 16)
    blue = int(stripped[4:6], 16)
    return blue, green, red


def clamp_point(point: np.ndarray, width: int, height: int) -> np.ndarray:
    return np.array(
        [
            np.clip(float(point[0]), 0.0, float(width - 1)),
            np.clip(float(point[1]), 0.0, float(height - 1)),
        ],
        dtype=float,
    )


def screen_work_area() -> tuple[int, int] | None:
    if os.name == "nt":
        try:
            user32 = ctypes.windll.user32
            rect = ctypes.wintypes.RECT()
            if user32.SystemParametersInfoW(0x0030, 0, ctypes.byref(rect), 0):
                return int(rect.right - rect.left), int(rect.bottom - rect.top)
            return int(user32.GetSystemMetrics(0)), int(user32.GetSystemMetrics(1))
        except Exception:
            pass

    try:
        import tkinter as tk

        root = tk.Tk()
        root.withdraw()
        width = int(root.winfo_screenwidth())
        height = int(root.winfo_screenheight())
        root.destroy()
        return width, height
    except Exception:
        return None


def scaled_window_size(canvas_width: int, canvas_height: int, *, margin: int = WINDOW_MARGIN) -> tuple[int, int]:
    available = screen_work_area()
    if available is None:
        return min(1600, canvas_width), min(1200, canvas_height)

    available_width = max(640, int(available[0] - margin))
    available_height = max(640, int(available[1] - margin))
    scale = min(available_width / max(canvas_width, 1), available_height / max(canvas_height, 1))
    scale = max(scale, 0.5)
    return max(640, int(round(canvas_width * scale))), max(640, int(round(canvas_height * scale)))
