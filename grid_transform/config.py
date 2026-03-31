from __future__ import annotations

import sys
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]
VT_SEG_DIR = PROJECT_DIR / "vocal-tract-seg"
VT_SEG_DATA_ROOT = VT_SEG_DIR / "data_80"
VT_SEG_CONTOURS_ROOT = VT_SEG_DIR / "results" / "nnunet_080" / "inference_contours"
NOTEBOOK_VTLN_DIR = PROJECT_DIR / "VTLN"
DEFAULT_VTLN_DIR = PROJECT_DIR / "VTLN" / "iadi_replace_merge"
DEFAULT_OUTPUT_DIR = PROJECT_DIR / "outputs"
REPORT_OUTPUT_DIR = DEFAULT_OUTPUT_DIR / "reports"
VIDEO_OUTPUT_DIR = DEFAULT_OUTPUT_DIR / "videos"
P4_SAMPLE_OUTPUT_DIR = VIDEO_OUTPUT_DIR / "p4_s10_video_sample"
TONGUE_COLOR = "#ff006e"


def ensure_vt_grid_import_path() -> None:
    """Make the vendored vt_grid module importable from the workspace root."""
    vt_seg_dir = str(VT_SEG_DIR)
    if vt_seg_dir not in sys.path:
        sys.path.insert(0, vt_seg_dir)
