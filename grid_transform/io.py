from __future__ import annotations

import zipfile
from os import PathLike
from pathlib import Path

import numpy as np
from PIL import Image
from roifile import ImagejRoi


VTNL_CONTOUR_NAME_MAP = {
    "incisor-hard-palate": "incisior-hard-palate",
    "mandible-incisor": "mandible-incisior",
}


def normalize_vtnl_contour_name(label: str) -> str:
    """Map VTNL/iADI contour labels to the canonical names used in the repo."""
    return VTNL_CONTOUR_NAME_MAP.get(label, label)


def load_frame_npy(frame_number: int, data_dir: PathLike[str] | str, contours_dir: PathLike[str] | str):
    data_dir = Path(data_dir)
    contours_dir = Path(contours_dir)
    image_path = data_dir / "PNG_MR" / f"{frame_number}.png"
    if not image_path.is_file():
        raise FileNotFoundError(f"Image not found: {image_path}")

    image = Image.open(image_path)
    contours = {}
    for npy_path in sorted(contours_dir.glob(f"{frame_number}_*.npy")):
        name = npy_path.stem.replace(f"{frame_number}_", "")
        points = np.load(npy_path)
        if len(points) > 0:
            contours[name] = points

    if not contours:
        raise FileNotFoundError(f"No contour files found in {contours_dir} for frame {frame_number}")

    return image, contours


def load_frame_vtnl(image_name: str, vtnl_dir: PathLike[str] | str):
    vtnl_dir = Path(vtnl_dir)
    image = None
    for ext in (".png", ".tif", ".tiff"):
        image_path = vtnl_dir / f"{image_name}{ext}"
        if image_path.is_file():
            image = Image.open(image_path).convert("L")
            break

    if image is None:
        raise FileNotFoundError(f"No VTNL image found for {image_name} in {vtnl_dir}")

    zip_path = vtnl_dir / f"{image_name}.zip"
    if not zip_path.is_file():
        raise FileNotFoundError(f"ROI zip not found: {zip_path}")

    contours = {}
    with zipfile.ZipFile(zip_path) as zf:
        for name in sorted(zf.namelist()):
            if not name.endswith(".roi"):
                continue
            roi = ImagejRoi.frombytes(zf.read(name))
            coords = roi.coordinates()
            if coords is None or len(coords) == 0:
                continue
            label = Path(name).stem.replace(f"{image_name}_", "")
            contours[normalize_vtnl_contour_name(label)] = np.array(coords, dtype=float)

    if not contours:
        raise FileNotFoundError(f"No ROI contours found in {zip_path}")

    return image, contours
