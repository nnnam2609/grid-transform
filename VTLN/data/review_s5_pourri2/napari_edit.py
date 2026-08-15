from __future__ import annotations

"""Self-contained Napari editor copied into the S5 pourri #2 review workspace."""

import argparse
import json
import os
import shutil
import tempfile
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image
from roifile import ImagejRoi, ROI_TYPE


CONTOUR_ORDER = (
    "arytenoid-cartilage",
    "epiglottis",
    "lower-incisor",
    "lower-lip",
    "pharynx",
    "soft-palate-midline",
    "thyroid-cartilage",
    "tongue",
    "upper-incisor",
    "upper-lip",
    "vocal-folds",
)
COLORS = {
    "arytenoid-cartilage": "#ff595e",
    "epiglottis": "#ff924c",
    "lower-incisor": "#ffca3a",
    "lower-lip": "#c5e327",
    "pharynx": "#52b788",
    "soft-palate-midline": "#2ec4b6",
    "thyroid-cartilage": "#00b4d8",
    "tongue": "#4361ee",
    "upper-incisor": "#9b5de5",
    "upper-lip": "#f15bb5",
    "vocal-folds": "#f8f9fa",
}


def validate_contour(points: np.ndarray, label: str) -> np.ndarray:
    array = np.asarray(points, dtype=float)
    if array.shape != (50, 2):
        raise ValueError(f"{label}: expected exactly one 50-point path, got {array.shape}")
    if not np.isfinite(array).all():
        raise ValueError(f"{label}: non-finite coordinates")
    if np.any(array < 0.0) or np.any(array > 135.0):
        raise ValueError(f"{label}: coordinates leave native 136x136")
    return array


def load_case(case_dir: Path) -> dict[str, np.ndarray]:
    result = {}
    for label in CONTOUR_ORDER:
        path = case_dir / "contours_npy" / f"{label}.npy"
        result[label] = validate_contour(np.load(path, allow_pickle=False), label)
    return result


def write_imagej_zip(path: Path, contours: dict[str, np.ndarray]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for label in CONTOUR_ORDER:
            points = validate_contour(contours[label], label)
            roi = ImagejRoi.frompoints(points, name=label)
            roi.roitype = ROI_TYPE.POLYLINE
            archive.writestr(f"{label}.roi", roi.tobytes())
    os.replace(temporary, path)


def save_case(case_dir: Path, contours: dict[str, np.ndarray]) -> dict[str, object]:
    validated = {label: validate_contour(contours[label], label) for label in CONTOUR_ORDER}
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    backup_dir = case_dir / "backups" / timestamp
    backup_dir.mkdir(parents=True)
    for label in CONTOUR_ORDER:
        source = case_dir / "contours_npy" / f"{label}.npy"
        shutil.copy2(source, backup_dir / source.name)
    shutil.copy2(case_dir / "imagej_rois.zip", backup_dir / "imagej_rois.zip")

    for label in CONTOUR_ORDER:
        target = case_dir / "contours_npy" / f"{label}.npy"
        with tempfile.NamedTemporaryFile(
            mode="wb",
            prefix=f".{label}.",
            suffix=".npy.tmp",
            dir=target.parent,
            delete=False,
        ) as handle:
            np.save(handle, validated[label], allow_pickle=False)
            temporary = Path(handle.name)
        os.replace(temporary, target)
    write_imagej_zip(case_dir / "imagej_rois.zip", validated)
    record = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "case": case_dir.name,
        "operation": "napari_backup_then_atomic_save",
        "backup_dir": str(backup_dir.resolve()),
        "contour_count": len(CONTOUR_ORDER),
        "point_count_per_contour": 50,
        "coordinate_space": "native_136x136_xy",
    }
    with (case_dir / "edits.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")
    return record


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", required=True, help="Case folder name, e.g. P1_S5_F0970")
    args = parser.parse_args()
    workspace = Path(__file__).resolve().parent
    case_dir = workspace / "cases" / args.case
    if not case_dir.is_dir():
        raise SystemExit(f"Unknown case: {case_dir}")

    try:
        import napari
    except ModuleNotFoundError as exc:
        raise SystemExit('Install Napari first: python -m pip install "napari[all]"') from exc

    image = np.asarray(Image.open(case_dir / "mri_native_136.png").convert("L"))
    source_contours = load_case(case_dir)
    viewer = napari.Viewer(title=f"S5 pourri #2 - {args.case}")
    viewer.add_image(image, name="MRI native 136", colormap="gray")
    for label in CONTOUR_ORDER:
        # Napari coordinates are (row, column) = (y, x).
        viewer.add_shapes(
            [source_contours[label][:, ::-1]],
            shape_type="path",
            name=f"contour:{label}",
            edge_color=COLORS[label],
            edge_width=1.5,
            face_color="transparent",
        )

    @viewer.bind_key("Control-Shift-S")
    def save_from_viewer(_viewer):
        current = {}
        for label in CONTOUR_ORDER:
            layer = viewer.layers[f"contour:{label}"]
            if len(layer.data) != 1:
                raise ValueError(f"{label}: expected one path, got {len(layer.data)}")
            current[label] = np.asarray(layer.data[0], dtype=float)[:, ::-1]
        record = save_case(case_dir, current)
        viewer.status = f"Saved with backup: {record['backup_dir']}"

    viewer.status = "Edit existing vertices; Ctrl+Shift+S saves with backup."
    napari.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
