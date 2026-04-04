from __future__ import annotations

import sys
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]
VTLN_DATA_DIR = PROJECT_DIR / "VTLN" / "data"
VT_SEG_DIR = VTLN_DATA_DIR
VT_SEG_DATA_ROOT = VTLN_DATA_DIR / "nnunet_data_80"
VT_SEG_CONTOURS_ROOT = VT_SEG_DATA_ROOT
NOTEBOOK_VTLN_DIR = VTLN_DATA_DIR
DEFAULT_VTLN_DIR = VTLN_DATA_DIR
DEFAULT_OUTPUT_DIR = PROJECT_DIR / "outputs"
REPORT_OUTPUT_DIR = DEFAULT_OUTPUT_DIR / "reports"
VIDEO_OUTPUT_DIR = DEFAULT_OUTPUT_DIR / "videos"
P4_SAMPLE_OUTPUT_DIR = VIDEO_OUTPUT_DIR / "p4_s10_video_sample"
TONGUE_COLOR = "#ff006e"


def ensure_vt_grid_import_path() -> None:
    """Legacy no-op kept for compatibility after moving vt_grid into the package."""
    return None
