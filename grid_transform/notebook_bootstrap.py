from __future__ import annotations

import sys
from pathlib import Path

from grid_transform.config import NOTEBOOK_VTNL_DIR, VT_SEG_CONTOURS_ROOT, VT_SEG_DATA_ROOT
from grid_transform.io import load_frame_npy, load_frame_vtnl
from grid_transform.vt import build_grid, print_grid_summary, visualize_grid


def find_repo_root(start: Path | None = None) -> Path:
    """Search upward for the workspace root from a starting directory."""
    start_path = (start or Path.cwd()).resolve()
    for candidate in (start_path, *start_path.parents):
        if (candidate / "requirements.txt").is_file() and (candidate / "grid_transform").is_dir():
            return candidate
    raise RuntimeError(f"Could not locate repo root from {start_path}")


def discover_vtnl_speakers(vtnl_dir: Path) -> list[str]:
    """Return sorted VTNL stems that have ROI zip files."""
    return sorted(path.stem for path in Path(vtnl_dir).glob("*.zip"))


def bootstrap_notebook(project_dir: Path | None = None) -> dict[str, object]:
    """Return a notebook-friendly context rooted at the repo."""
    project_dir = find_repo_root(project_dir)
    if str(project_dir) not in sys.path:
        sys.path.insert(0, str(project_dir))

    vt_seg_dir = project_dir / "vocal-tract-seg"
    vtnl_dir = project_dir / "VTNL"

    return {
        "PROJECT_DIR": project_dir,
        "VT_SEG_DIR": vt_seg_dir,
        "VT_SEG_DATA_ROOT": project_dir / VT_SEG_DATA_ROOT.relative_to(project_dir),
        "VT_SEG_CONTOURS_ROOT": project_dir / VT_SEG_CONTOURS_ROOT.relative_to(project_dir),
        "VTNL_DIR": project_dir / NOTEBOOK_VTNL_DIR.relative_to(project_dir),
        "build_grid": build_grid,
        "print_grid_summary": print_grid_summary,
        "visualize_grid": visualize_grid,
        "load_frame_npy": load_frame_npy,
        "load_frame_vtnl": load_frame_vtnl,
        "discover_vtnl_speakers": discover_vtnl_speakers,
    }
