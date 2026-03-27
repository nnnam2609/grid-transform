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
KNOWN_VTNL_CONTOUR_NAMES = {
    "c1",
    "c2",
    "c3",
    "c4",
    "c5",
    "c6",
    "incisor-hard-palate",
    "incisior-hard-palate",
    "mandible-incisor",
    "mandible-incisior",
    "pharynx",
    "soft-palate",
    "soft-palate-midline",
    "tongue",
}


def normalize_vtnl_contour_name(label: str) -> str:
    """Map VTNL/iADI contour labels to the canonical names used in the repo."""
    return VTNL_CONTOUR_NAME_MAP.get(label, label)


def extract_vtnl_contour_name(stem: str, image_name: str) -> str:
    """Recover a canonical contour label even if the ROI stem carries a stale prefix."""
    candidate = stem
    if stem.startswith(f"{image_name}_"):
        candidate = stem[len(image_name) + 1 :]

    normalized = normalize_vtnl_contour_name(candidate)
    if normalized in KNOWN_VTNL_CONTOUR_NAMES:
        return normalized

    fallback = stem.rsplit("_", 1)[-1]
    fallback_normalized = normalize_vtnl_contour_name(fallback)
    if fallback_normalized in KNOWN_VTNL_CONTOUR_NAMES:
        return fallback_normalized

    return normalized


def candidate_vtnl_dirs(vtnl_dir: Path) -> list[Path]:
    """Search the requested VTNL folder first, then its parent as a compatibility fallback."""
    candidates = [vtnl_dir]
    parent = vtnl_dir.parent
    if parent != vtnl_dir and parent.exists():
        candidates.append(parent)
    return candidates


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
    resolved_dir = None
    for candidate_dir in candidate_vtnl_dirs(vtnl_dir):
        for ext in (".png", ".tif", ".tiff"):
            image_path = candidate_dir / f"{image_name}{ext}"
            if image_path.is_file():
                image = Image.open(image_path).convert("L")
                resolved_dir = candidate_dir
                break
        if image is not None:
            break

    if image is None:
        raise FileNotFoundError(f"No VTNL image found for {image_name} in {vtnl_dir}")

    zip_path = (resolved_dir or vtnl_dir) / f"{image_name}.zip"
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
            label = extract_vtnl_contour_name(Path(name).stem, image_name)
            contours[normalize_vtnl_contour_name(label)] = np.array(coords, dtype=float)

    if not contours:
        raise FileNotFoundError(f"No ROI contours found in {zip_path}")

    return image, contours
