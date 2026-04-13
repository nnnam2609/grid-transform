from __future__ import annotations

import csv
import zipfile
from functools import lru_cache
from os import PathLike
from pathlib import Path

import numpy as np
from PIL import Image
from roifile import ImagejRoi


VTLN_CONTOUR_NAME_MAP = {
    "incisor-hard-palate": "incisior-hard-palate",
    "mandible-incisor": "mandible-incisior",
}
KNOWN_VTLN_CONTOUR_NAMES = {
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
VTLN_TRIPLET_SHAPE = (480, 480)
VTLN_TRIPLET_CHANNEL_ORDER = "R=t-1,G=t,B=t+1"


def normalize_vtln_contour_name(label: str) -> str:
    """Map VTLN/iADI contour labels to the canonical names used in the repo."""
    return VTLN_CONTOUR_NAME_MAP.get(label, label)


def extract_vtln_contour_name(stem: str, image_name: str) -> str:
    """Recover a canonical contour label even if the ROI stem carries a stale prefix."""
    candidate = stem
    if stem.startswith(f"{image_name}_"):
        candidate = stem[len(image_name) + 1 :]

    normalized = normalize_vtln_contour_name(candidate)
    if normalized in KNOWN_VTLN_CONTOUR_NAMES:
        return normalized

    fallback = stem.rsplit("_", 1)[-1]
    fallback_normalized = normalize_vtln_contour_name(fallback)
    if fallback_normalized in KNOWN_VTLN_CONTOUR_NAMES:
        return fallback_normalized

    return normalized


def candidate_vtln_dirs(vtln_dir: Path) -> list[Path]:
    """Use only the explicitly requested VTLN folder."""
    return [vtln_dir]


def _contour_bounds(contours: dict[str, np.ndarray]) -> tuple[float, float, float, float] | None:
    min_x = float("inf")
    min_y = float("inf")
    max_x = float("-inf")
    max_y = float("-inf")
    found = False
    for points in contours.values():
        pts = np.asarray(points, dtype=float)
        if pts.ndim != 2 or pts.shape[1] != 2 or len(pts) == 0:
            continue
        found = True
        min_x = min(min_x, float(pts[:, 0].min()))
        min_y = min(min_y, float(pts[:, 1].min()))
        max_x = max(max_x, float(pts[:, 0].max()))
        max_y = max(max_y, float(pts[:, 1].max()))
    if not found:
        return None
    return min_x, min_y, max_x, max_y


@lru_cache(maxsize=16)
def _load_vtln_manifest_rows(vtln_dir_str: str) -> dict[str, dict[str, str]]:
    manifest_path = Path(vtln_dir_str) / "selection_manifest.csv"
    if not manifest_path.is_file():
        return {}
    with manifest_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return {
            str(row.get("output_basename") or "").strip(): {str(key): str(value) for key, value in row.items()}
            for row in reader
            if str(row.get("output_basename") or "").strip()
        }


def _validate_vtln_triplet_bundle(
    image_name: str,
    *,
    vtln_dir: Path,
    image,
    contours: dict[str, np.ndarray],
) -> None:
    array = np.asarray(image)
    if array.ndim != 3 or array.shape[2] != 3:
        raise ValueError(
            f"VTLN image {image_name} in {vtln_dir} must be an RGB triplet, got shape {array.shape}."
        )
    if tuple(int(value) for value in array.shape[:2]) != VTLN_TRIPLET_SHAPE:
        raise ValueError(
            f"VTLN image {image_name} in {vtln_dir} must be {VTLN_TRIPLET_SHAPE[0]}x{VTLN_TRIPLET_SHAPE[1]}, got {array.shape[:2]}."
        )

    row = _load_vtln_manifest_rows(str(vtln_dir.resolve())).get(image_name)
    if row:
        output_size = str(row.get("output_size") or "").strip()
        channel_order = str(row.get("channel_order") or "").strip()
        if output_size and output_size != "480x480":
            raise ValueError(
                f"VTLN manifest row {image_name} in {vtln_dir} must declare output_size=480x480, got {output_size!r}."
            )
        if channel_order and channel_order != VTLN_TRIPLET_CHANNEL_ORDER:
            raise ValueError(
                f"VTLN manifest row {image_name} in {vtln_dir} must declare channel_order={VTLN_TRIPLET_CHANNEL_ORDER!r}, got {channel_order!r}."
            )

    bounds = _contour_bounds(contours)
    if bounds is None:
        return
    min_x, min_y, max_x, max_y = bounds
    tolerance_px = 4.0
    if (
        min_x < -tolerance_px
        or min_y < -tolerance_px
        or max_x > VTLN_TRIPLET_SHAPE[1] + tolerance_px
        or max_y > VTLN_TRIPLET_SHAPE[0] + tolerance_px
    ):
        raise ValueError(
            f"VTLN annotation {image_name} in {vtln_dir} is outside 480x480 triplet space: "
            f"bbox=({min_x:.2f}, {min_y:.2f}, {max_x:.2f}, {max_y:.2f})."
        )


def _load_roi_contours_from_zip(zip_path: Path, *, image_name: str) -> dict[str, np.ndarray]:
    contours: dict[str, np.ndarray] = {}
    with zipfile.ZipFile(zip_path) as zf:
        for name in sorted(zf.namelist()):
            if not name.endswith(".roi"):
                continue
            roi = ImagejRoi.frombytes(zf.read(name))
            coords = roi.coordinates()
            if coords is None or len(coords) == 0:
                continue
            label = extract_vtln_contour_name(Path(name).stem, image_name)
            contours[normalize_vtln_contour_name(label)] = np.array(coords, dtype=float)
    return contours


def _load_roi_contours_from_directory(contours_dir: Path, *, image_name: str) -> dict[str, np.ndarray]:
    zip_path = contours_dir / f"{image_name}.zip"
    if zip_path.is_file():
        return _load_roi_contours_from_zip(zip_path, image_name=image_name)

    contours: dict[str, np.ndarray] = {}
    for roi_path in sorted(contours_dir.glob(f"{image_name}_*.roi")):
        roi = ImagejRoi.fromfile(roi_path)
        coords = roi.coordinates()
        if coords is None or len(coords) == 0:
            continue
        label = extract_vtln_contour_name(roi_path.stem, image_name)
        contours[normalize_vtln_contour_name(label)] = np.array(coords, dtype=float)
    return contours


def _load_contours_from_masks(masks_dir: Path, frame_number: int) -> dict[str, np.ndarray]:
    import cv2

    contours: dict[str, np.ndarray] = {}
    for mask_path in sorted(masks_dir.glob(f"{frame_number}_*.png")):
        name = mask_path.stem.replace(f"{frame_number}_", "")
        name = normalize_vtln_contour_name(name)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            continue
        found, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if not found:
            continue
        largest = max(found, key=cv2.contourArea)
        pts = largest.squeeze(1).astype(float)  # (N, 2) as x, y
        if len(pts) > 0:
            contours[name] = pts
    return contours


def load_frame_npy(
    frame_number: int,
    data_dir: PathLike[str] | str,
    contours_dir: PathLike[str] | str,
    *,
    use_predicted_masks: bool = False,
):
    data_dir = Path(data_dir)
    contours_dir = Path(contours_dir)
    image_path = data_dir / "PNG_MR" / f"{frame_number}.png"
    if not image_path.is_file():
        raise FileNotFoundError(f"Image not found: {image_path}")

    image = Image.open(image_path)

    if use_predicted_masks:
        masks_dir = data_dir / "masks"
        contours = _load_contours_from_masks(masks_dir, frame_number)
        if contours:
            return image, contours
        raise FileNotFoundError(f"No mask files found in {masks_dir} for frame {frame_number}")

    contours = {}
    for npy_path in sorted(contours_dir.glob(f"{frame_number}_*.npy")):
        name = npy_path.stem.replace(f"{frame_number}_", "")
        points = np.load(npy_path)
        if len(points) > 0:
            contours[name] = points

    if contours:
        return image, contours

    roi_dirs = [
        contours_dir / "contours",
        data_dir / "contours",
        contours_dir,
    ]
    for roi_dir in roi_dirs:
        if not roi_dir.is_dir():
            continue
        contours = _load_roi_contours_from_directory(roi_dir, image_name=str(frame_number))
        if contours:
            return image, contours

    if not contours:
        raise FileNotFoundError(f"No contour files found in {contours_dir} for frame {frame_number}")
    return image, contours


def load_frame_vtln(
    image_name: str,
    vtln_dir: PathLike[str] | str,
    *,
    validate_triplet_bundle: bool = False,
):
    vtln_dir = Path(vtln_dir)
    image = None
    resolved_dir = None
    for candidate_dir in candidate_vtln_dirs(vtln_dir):
        for ext in (".png", ".tif", ".tiff"):
            image_path = candidate_dir / f"{image_name}{ext}"
            if image_path.is_file():
                image = Image.open(image_path)
                resolved_dir = candidate_dir
                break
        if image is not None:
            break

    if image is None:
        raise FileNotFoundError(f"No VTLN image found for {image_name} in {vtln_dir}")

    zip_path = (resolved_dir or vtln_dir) / f"{image_name}.zip"
    if not zip_path.is_file():
        raise FileNotFoundError(f"ROI zip not found: {zip_path}")

    contours = _load_roi_contours_from_zip(zip_path, image_name=image_name)

    if not contours:
        raise FileNotFoundError(f"No ROI contours found in {zip_path}")

    if validate_triplet_bundle:
        _validate_vtln_triplet_bundle(
            image_name,
            vtln_dir=(resolved_dir or vtln_dir),
            image=image,
            contours=contours,
        )

    return image, contours
