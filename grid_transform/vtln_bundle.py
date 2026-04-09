from __future__ import annotations

import zipfile
from pathlib import Path

import numpy as np
from roifile import ImagejRoi

from grid_transform.annotation_projection import build_resize_affine, transform_reference_contours


def write_annotation_zip(
    path: Path,
    basename: str,
    contours: dict[str, np.ndarray],
    *,
    dry_run: bool = False,
) -> None:
    if dry_run:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for label, points in sorted(contours.items()):
            pts = np.asarray(points, dtype=float)
            if pts.ndim != 2 or pts.shape[1] != 2 or len(pts) == 0:
                continue
            roi_name = f"{basename}_{label}"
            roi = ImagejRoi.frompoints(pts, name=roi_name)
            zf.writestr(f"{roi_name}.roi", roi.tobytes())


def scale_contours_to_triplet_space(
    contours: dict[str, np.ndarray],
    source_shape: tuple[int, int],
    target_shape: tuple[int, int],
) -> dict[str, np.ndarray]:
    affine = build_resize_affine(source_shape, target_shape)
    return transform_reference_contours(contours, affine)
