from __future__ import annotations

import hashlib
from pathlib import Path
import zipfile

import numpy as np
from roifile import ImagejRoi, ROI_TYPE

from grid_transform.apps.update_geometry_reference_lower_incisor import replace_lower_member


def payload(points: np.ndarray, name: str) -> bytes:
    return ImagejRoi.frompoints(points, name=name).tobytes()


def test_replace_changes_only_closed_lower_member(tmp_path: Path) -> None:
    basename = "1612_P1_S16_F0952"
    zip_path = tmp_path / f"{basename}.zip"
    lower_name = f"{basename}_mandible-incisior.roi"
    other_name = f"{basename}_tongue.roi"
    old_lower = payload(np.array([[1, 1], [2, 1], [2, 2]], dtype=float), Path(lower_name).stem)
    other = payload(np.array([[4, 4], [5, 4], [5, 5]], dtype=float), Path(other_name).stem)
    with zipfile.ZipFile(zip_path, "w") as archive:
        archive.writestr(lower_name, old_lower)
        archive.writestr(other_name, other)
    points = np.column_stack((np.linspace(10, 20, 50), np.linspace(30, 40, 50)))

    result = replace_lower_member(
        zip_path,
        points,
        backup_path=tmp_path / "backup" / zip_path.name,
        run_id="test",
    )

    with zipfile.ZipFile(zip_path) as archive:
        assert archive.read(other_name) == other
        roi = ImagejRoi.frombytes(archive.read(lower_name))
    assert result["status"] == "PASS"
    assert roi.roitype == ROI_TYPE.FREEHAND
    assert roi.coordinates().shape == (50, 2)
    assert hashlib.sha256(old_lower).hexdigest() != result["lower_roi_sha256_after"]
