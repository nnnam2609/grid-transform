from __future__ import annotations

"""Add the reviewed ASD2 F2812 reference to a GTGRD bundle candidate."""

import argparse
import csv
import hashlib
import json
import os
import shutil
import tempfile
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from roifile import ImagejRoi, ROI_TYPE

from grid_transform.config import DEFAULT_VTLN_DIR, PROJECT_DIR
from grid_transform.vt import build_grid
from grid_transform.vtln_bundle import scale_contours_to_triplet_space


BASE_VERSION = "0.1.18"
VERSION = "0.1.19"
SPEAKER = "ASD2"
RAW_SUBJECT = "1791"
SESSION = "S29"
FRAME = 2812
BASENAME = "1791_ASD2_S29_F2812"
SOURCE_SHAPE = (136, 136)
TARGET_SHAPE = (480, 480)
FRAME_IDS = (FRAME - 1, FRAME, FRAME + 1)
CHANNEL_ORDER = "R=t-1,G=t,B=t+1"
ZIP_TIMESTAMP = (1980, 1, 1, 0, 0, 0)
ZIP_FILE_MODE = 0o100644 << 16

DYNAMIC_LABELS = (
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
CERVICAL_LABELS = tuple(f"C{index}" for index in range(1, 7))
EXPECTED_LABELS = set(DYNAMIC_LABELS) | set(CERVICAL_LABELS)
CANONICAL_LABEL_MAP = {
    "upper-incisor": "incisior-hard-palate",
    "lower-incisor": "mandible-incisior",
    **{label: label for label in DYNAMIC_LABELS if "incisor" not in label},
    **{label: label.lower() for label in CERVICAL_LABELS},
}

DEFAULT_ASD2_SESSION_DIR = PROJECT_DIR.parent / "Data" / "ArtSpeech_Database_2" / RAW_SUBJECT / SESSION
DEFAULT_MRI_DIR = DEFAULT_ASD2_SESSION_DIR / "NPY_MR_registered"
DEFAULT_ROI_ZIP = (
    PROJECT_DIR
    / "outputs"
    / "asd2_1791_stable_tongue_search_20260824_v2"
    / "S29_F2812_imagej_manual_edit"
    / "RoiSet_update.zip"
)
DEFAULT_OUTPUT_ROOT = PROJECT_DIR / "outputs" / "geometry_reference_candidates" / "v0.1.19_asd2_f2812"
DEFAULT_BACKUP_ROOT = PROJECT_DIR / "outputs" / "geometry_reference_backups" / "v0.1.19_before_asd2"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-vtln-dir", type=Path, default=DEFAULT_VTLN_DIR)
    parser.add_argument("--mri-dir", type=Path, default=DEFAULT_MRI_DIR)
    parser.add_argument("--roi-zip", type=Path, default=DEFAULT_ROI_ZIP)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--backup-root", type=Path, default=DEFAULT_BACKUP_ROOT)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--promote",
        action="store_true",
        help="After candidate validation, atomically copy its five changed/new files into VTLN/data.",
    )
    return parser


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def normalize_registered_triplet(frames: list[np.ndarray]) -> tuple[np.ndarray, dict[str, float]]:
    if len(frames) != 3:
        raise ValueError(f"Expected three registered frames, got {len(frames)}")
    stack = np.stack([np.asarray(frame) for frame in frames], axis=0)
    if stack.shape != (3, *SOURCE_SHAPE):
        raise ValueError(f"Expected registered triplet {(3, *SOURCE_SHAPE)}, got {stack.shape}")
    if not np.isfinite(stack).all():
        raise ValueError("Registered MRI triplet contains non-finite values")
    value_min = float(stack.min())
    value_max = float(stack.max())
    if value_max <= value_min:
        raise ValueError("Registered MRI triplet has no intensity range")
    scaled = np.rint((stack.astype(np.float64) - value_min) * 255.0 / (value_max - value_min))
    scaled = np.clip(scaled, 0.0, 255.0).astype(np.uint8)
    resized = [cv2.resize(frame, TARGET_SHAPE[::-1], interpolation=cv2.INTER_CUBIC) for frame in scaled]
    triplet = np.stack(resized, axis=2).astype(np.uint8)
    return triplet, {
        "source_min": value_min,
        "source_max": value_max,
        "output_min": float(triplet.min()),
        "output_max": float(triplet.max()),
    }


def load_reviewed_rois(
    roi_zip_path: Path,
) -> tuple[dict[str, np.ndarray], dict[str, str], dict[str, str]]:
    contours: dict[str, np.ndarray] = {}
    roi_types: dict[str, str] = {}
    member_hashes: dict[str, str] = {}
    with zipfile.ZipFile(roi_zip_path) as archive:
        for member in sorted(archive.namelist()):
            if not member.lower().endswith(".roi"):
                continue
            payload = archive.read(member)
            roi = ImagejRoi.frombytes(payload)
            label = Path(member).stem.split("_", 1)[-1]
            if label in contours:
                raise ValueError(f"Duplicate reviewed ASD2 label: {label}")
            points = np.asarray(roi.coordinates(), dtype=float)
            contours[label] = points
            roi_types[label] = roi.roitype.name
            member_hashes[label] = sha256_bytes(payload)
    if set(contours) != EXPECTED_LABELS:
        raise ValueError(
            f"Reviewed ASD2 ROI labels differ: missing={sorted(EXPECTED_LABELS - set(contours))}, "
            f"extra={sorted(set(contours) - EXPECTED_LABELS)}"
        )
    for label in DYNAMIC_LABELS:
        if contours[label].shape != (50, 2) or roi_types[label] != "POLYLINE":
            raise ValueError(f"Invalid dynamic ROI {label}: {contours[label].shape}/{roi_types[label]}")
    for label in CERVICAL_LABELS:
        points = contours[label]
        if points.ndim != 2 or points.shape[1] != 2 or len(points) < 3:
            raise ValueError(f"Invalid cervical ROI {label}: {points.shape}")
        if roi_types[label] != "FREEHAND":
            raise ValueError(f"Expected FREEHAND {label}, got {roi_types[label]}")
    for label, points in contours.items():
        if not np.isfinite(points).all():
            raise ValueError(f"Non-finite reviewed ASD2 ROI: {label}")
        if np.any(points < 0.0) or np.any(points[:, 0] > 135.0) or np.any(points[:, 1] > 135.0):
            raise ValueError(f"Out-of-bounds reviewed ASD2 ROI: {label}")
    return contours, roi_types, member_hashes


def canonicalize_reviewed_rois(contours: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    scaled = scale_contours_to_triplet_space(contours, SOURCE_SHAPE, TARGET_SHAPE)
    return {CANONICAL_LABEL_MAP[label]: np.asarray(points, dtype=float) for label, points in scaled.items()}


def write_annotation_zip(
    path: Path,
    contours: dict[str, np.ndarray],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for label, points in sorted(contours.items()):
            array = np.asarray(points, dtype=float)
            roi_name = f"{BASENAME}_{label}"
            roi = ImagejRoi.frompoints(array, name=roi_name)
            roi.roitype = ROI_TYPE.FREEHAND if label.startswith("c") else ROI_TYPE.POLYLINE
            info = zipfile.ZipInfo(f"{roi_name}.roi", ZIP_TIMESTAMP)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.create_system = 3
            info.external_attr = ZIP_FILE_MODE
            archive.writestr(info, roi.tobytes())


def read_manifest(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        return list(reader.fieldnames or []), list(reader)


def append_manifest_row(
    path: Path,
    *,
    source_dir: Path,
    roi_zip: Path,
    reference_bundle_dir: Path,
) -> None:
    fieldnames, rows = read_manifest(path)
    if any(row.get("speaker") == SPEAKER or row.get("output_basename") == BASENAME for row in rows):
        raise ValueError(f"Manifest already contains {SPEAKER}/{BASENAME}")
    row = {
        "output_basename": BASENAME,
        "speaker": SPEAKER,
        "raw_subject": RAW_SUBJECT,
        "session": SESSION,
        "selected_source": f"{SPEAKER}/{SESSION}/F{FRAME:04d}",
        "image_source": f"{source_dir.resolve()}#frames={FRAME_IDS[0]},{FRAME_IDS[1]},{FRAME_IDS[2]};shared_minmax_uint8",
        "annotation_source": str(roi_zip.resolve()),
        "annotation_status": "user_reviewed_11_dynamic_plus_C1_C6",
        "annotation_origin_path": str(roi_zip.resolve()),
        "reference_bundle_dir": str(reference_bundle_dir.resolve()),
        "reference_bundle_name": BASENAME,
        "prev_frame_1based": str(FRAME_IDS[0]),
        "center_frame_1based": str(FRAME_IDS[1]),
        "next_frame_1based": str(FRAME_IDS[2]),
        "channel_order": CHANNEL_ORDER,
        "output_size": "480x480",
        "annotation_space": "480x480_scaled_from_registered_136x136",
    }
    if set(row) != set(fieldnames):
        raise ValueError(f"Manifest schema differs: expected={fieldnames}, row={list(row)}")
    rows.append(row)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def candidate_readme() -> str:
    return f"""# GTGRD v{VERSION}

GTGRD means Grid Transform Geometry Reference Data. `VTLN/data` remains the
runtime compatibility path.

- The ten ASD1 references P1-P10 are byte-identical to GTGRD v{BASE_VERSION} and use the midpoint frame of S5 `pourri #2 /u/`.
- ASD2 adds `1791/S29/F2812`, the user-selected exact `/u/` shape reference with the final ImageJ-reviewed `RoiSet_update.zip` geometry.
- RGB triplets use `R=t-1, G=t, B=t+1` and are 480x480. ASD1 triplets retain their original review-AVI conversion. ASD2 uses registered uint16 MRI frames F2811-F2813, one shared linear min/max conversion to uint8, and cubic resize from 136x136 to 480x480.
- ASD2 contains 11 dynamic contours plus C1-C6. The current server-`bf` incisors and final manually updated cervical contours are retained exactly before the common 136-to-480 resize.
- Dynamic contours are stored as open ImageJ polylines; C1-C6 retain FREEHAND topology.
- Historical labels `incisior-hard-palate` and `mandible-incisior` remain the canonical stored names for upper and lower incisors.
- P2 still has ten observed dynamic contours; `vocal-folds` is absent and is not imputed.
- GTGRD v{VERSION} changes data only. It does not change affine/TPS controls, transform ordering, landmark definitions, or P2CP metrics.
"""


def update_build_summary(
    path: Path,
    *,
    png_path: Path,
    zip_path: Path,
    mri_paths: list[Path],
    roi_zip: Path,
    member_hashes: dict[str, str],
    intensity: dict[str, float],
    grid: object,
    max_round_trip_px: float,
) -> None:
    summary = json.loads(path.read_text(encoding="utf-8"))
    if summary.get("version") != BASE_VERSION or summary.get("status") != "PASS":
        raise ValueError(
            f"Expected PASS GTGRD {BASE_VERSION} base, got {summary.get('version')}/{summary.get('status')}"
        )
    speakers = list(summary.get("speaker_order", []))
    if speakers != [f"P{index}" for index in range(1, 11)]:
        raise ValueError(f"Unexpected base speaker order: {speakers}")
    cases = list(summary.get("cases", []))
    if len(cases) != 10 or any(case.get("speaker") == SPEAKER for case in cases):
        raise ValueError("Unexpected base case inventory")
    now = datetime.now(timezone.utc).isoformat()
    case = {
        "speaker": SPEAKER,
        "raw_subject": RAW_SUBJECT,
        "basename": BASENAME,
        "session": SESSION,
        "selected_frame": FRAME,
        "phone_selection": "exact /u/; user-selected shape reference",
        "image_coordinate_space": "registered 136x136",
        "image_intensity_policy": "shared F2811-F2813 linear min/max to uint8 before cubic 480x480 resize",
        "image_intensity_range": intensity,
        "source_mri_sha256": {f"F{frame:04d}": sha256_file(mri_path) for frame, mri_path in zip(FRAME_IDS, mri_paths)},
        "source_roi_zip": str(roi_zip.resolve()),
        "source_roi_zip_sha256": sha256_file(roi_zip),
        "source_roi_member_sha256": member_hashes,
        "dynamic_contour_count": len(DYNAMIC_LABELS),
        "canonical_contour_count": len(DYNAMIC_LABELS) + len(CERVICAL_LABELS),
        "missing_dynamic_labels": [],
        "incisor_policy": "latest server-bf upper/lower incisors retained from reviewed RoiSet_update",
        "cervical_policy": "final user-reviewed C1-C6 from RoiSet_update",
        "png_sha256": sha256_file(png_path),
        "zip_sha256": sha256_file(zip_path),
        "roi_round_trip_max_px_480": max_round_trip_px,
        "grid_horiz_count": len(grid.horiz_lines),
        "grid_vert_count": len(grid.vert_lines),
        "grid_warnings": list(grid.warnings),
    }
    summary.update(
        {
            "schema_version": "gtgrd-v0.1.19-asd2-extension-v1",
            "version": VERSION,
            "created_at_utc": now,
            "status": "PASS",
            "scope": "canonical GTGRD v0.1.18 ASD1 references plus reviewed ASD2 1791/S29/F2812",
            "speaker_order": [*speakers, SPEAKER],
            "cases": [*cases, case],
            "asd2_reference_policy": {
                "role": "additional transform/reference geometry; does not replace P10",
                "reference_speaker_for_historical_v018_diagnostics": summary.get("reference_speaker"),
                "transforms_to_p10_recomputed": False,
                "reason": "v0.1.19 is an additive data release; prior P1-P10 transform diagnostics remain unchanged",
            },
            "base_v018_nonmetadata_files_unchanged": True,
        }
    )
    path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def validate_annotation_round_trip(
    zip_path: Path,
    expected: dict[str, np.ndarray],
) -> float:
    decoded: dict[str, np.ndarray] = {}
    with zipfile.ZipFile(zip_path) as archive:
        members = sorted(name for name in archive.namelist() if name.endswith(".roi"))
        if len(members) != 17:
            raise ValueError(f"Expected 17 ASD2 ROI members, got {len(members)}")
        for member in members:
            roi = ImagejRoi.frombytes(archive.read(member))
            label = Path(member).stem[len(BASENAME) + 1 :]
            expected_type = ROI_TYPE.FREEHAND if label.startswith("c") else ROI_TYPE.POLYLINE
            if roi.roitype != expected_type:
                raise ValueError(f"Unexpected ROI type {label}: {roi.roitype}")
            decoded[label] = np.asarray(roi.coordinates(), dtype=float)
    if set(decoded) != set(expected):
        raise ValueError("ASD2 canonical ROI round-trip label mismatch")
    errors = []
    for label, points in expected.items():
        delta = decoded[label] - np.asarray(points, dtype=float)
        errors.append(float(np.sqrt(np.mean(np.sum(delta * delta, axis=1)))))
    maximum = max(errors)
    if maximum >= 1e-3:
        raise ValueError(f"ASD2 ROI round-trip error is too large: {maximum}")
    return maximum


def nonmetadata_hashes(data_dir: Path) -> dict[str, str]:
    excluded = {"README.md", "selection_manifest.csv", "build_summary.json"}
    return {
        path.relative_to(data_dir).as_posix(): sha256_file(path)
        for path in sorted(data_dir.rglob("*"))
        if path.is_file() and path.name not in excluded
    }


def build_candidate(args: argparse.Namespace) -> dict[str, object]:
    base_dir = Path(args.base_vtln_dir).resolve()
    output_root = Path(args.output_root).resolve()
    if output_root.exists():
        if not args.overwrite:
            raise FileExistsError(f"Candidate exists: {output_root}; use --overwrite")
        shutil.rmtree(output_root)
    candidate_dir = output_root / "VTLN" / "data"
    candidate_dir.parent.mkdir(parents=True)
    shutil.copytree(base_dir, candidate_dir)

    base_hashes = nonmetadata_hashes(base_dir)
    png_path = candidate_dir / f"{BASENAME}.png"
    zip_path = candidate_dir / f"{BASENAME}.zip"
    if png_path.exists() or zip_path.exists():
        raise FileExistsError(f"ASD2 reference already exists under {candidate_dir}")

    mri_paths = [Path(args.mri_dir).resolve() / f"{frame:04d}.npy" for frame in FRAME_IDS]
    frames = [np.load(path, allow_pickle=False) for path in mri_paths]
    triplet, intensity = normalize_registered_triplet(frames)
    Image.fromarray(triplet, mode="RGB").save(png_path)

    reviewed, _, member_hashes = load_reviewed_rois(Path(args.roi_zip).resolve())
    canonical = canonicalize_reviewed_rois(reviewed)
    write_annotation_zip(zip_path, canonical)
    max_round_trip_px = validate_annotation_round_trip(zip_path, canonical)
    grid = build_grid(triplet, canonical, n_vert=9, n_points=250, frame_number=FRAME)
    if len(grid.horiz_lines) != 6 or len(grid.vert_lines) != 9 or grid.warnings:
        raise ValueError(f"Invalid ASD2 grid: {len(grid.horiz_lines)}x{len(grid.vert_lines)}, warnings={grid.warnings}")

    append_manifest_row(
        candidate_dir / "selection_manifest.csv",
        source_dir=Path(args.mri_dir),
        roi_zip=Path(args.roi_zip),
        reference_bundle_dir=base_dir,
    )
    (candidate_dir / "README.md").write_text(candidate_readme(), encoding="utf-8")
    update_build_summary(
        candidate_dir / "build_summary.json",
        png_path=png_path,
        zip_path=zip_path,
        mri_paths=mri_paths,
        roi_zip=Path(args.roi_zip),
        member_hashes=member_hashes,
        intensity=intensity,
        grid=grid,
        max_round_trip_px=max_round_trip_px,
    )

    candidate_hashes = nonmetadata_hashes(candidate_dir)
    added = {f"{BASENAME}.png", f"{BASENAME}.zip"}
    retained = {key: value for key, value in candidate_hashes.items() if key not in added}
    if retained != base_hashes:
        raise RuntimeError("A pre-v0.1.19 non-metadata GTGRD file changed")
    _, manifest_rows = read_manifest(candidate_dir / "selection_manifest.csv")
    if len(manifest_rows) != 11 or manifest_rows[-1]["speaker"] != SPEAKER:
        raise RuntimeError("Candidate manifest does not contain ordered P1-P10 plus ASD2")

    result: dict[str, object] = {
        "status": "PASS",
        "version": VERSION,
        "candidate_dir": str(candidate_dir),
        "basename": BASENAME,
        "png_sha256": sha256_file(png_path),
        "zip_sha256": sha256_file(zip_path),
        "roi_count": 17,
        "roi_round_trip_max_px_480": max_round_trip_px,
        "grid_horiz_count": len(grid.horiz_lines),
        "grid_vert_count": len(grid.vert_lines),
        "grid_warnings": list(grid.warnings),
        "base_nonmetadata_files_unchanged": True,
        "promoted": False,
    }
    if args.promote:
        promote_candidate(
            candidate_dir=candidate_dir,
            target_dir=base_dir,
            backup_root=Path(args.backup_root).resolve(),
            overwrite=args.overwrite,
        )
        result["promoted"] = True
    (output_root / "candidate_summary.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return result


def promote_candidate(
    *,
    candidate_dir: Path,
    target_dir: Path,
    backup_root: Path,
    overwrite: bool,
) -> None:
    if backup_root.exists():
        if not overwrite:
            raise FileExistsError(f"Backup exists: {backup_root}; use --overwrite")
        shutil.rmtree(backup_root)
    backup_root.mkdir(parents=True)
    metadata = ("README.md", "selection_manifest.csv", "build_summary.json")
    for name in metadata:
        shutil.copy2(target_dir / name, backup_root / name)

    changed_names = (f"{BASENAME}.png", f"{BASENAME}.zip", *metadata)
    staged: list[tuple[Path, Path]] = []
    for name in changed_names:
        source = candidate_dir / name
        fd, temporary_name = tempfile.mkstemp(prefix=f".{name}.", dir=target_dir)
        os.close(fd)
        temporary = Path(temporary_name)
        shutil.copy2(source, temporary)
        if sha256_file(source) != sha256_file(temporary):
            raise RuntimeError(f"Promotion staging hash mismatch: {name}")
        staged.append((temporary, target_dir / name))
    for temporary, destination in staged:
        os.replace(temporary, destination)

    for name in changed_names:
        if sha256_file(candidate_dir / name) != sha256_file(target_dir / name):
            raise RuntimeError(f"Promoted file hash mismatch: {name}")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = build_candidate(args)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
