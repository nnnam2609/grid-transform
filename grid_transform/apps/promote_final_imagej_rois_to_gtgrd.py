from __future__ import annotations

"""Build and optionally promote GTGRD v0.1.20 from final ImageJ RoiSet files.

The per-slice ``RoiSet.zip`` files are treated as authoritative reviewed
geometry.  P10 is intentionally stored in the accepted stage-1 registered
coordinate system in this version.  ASD2 is intentionally changed from
S29/F2812 to S29/F2180, with a matching registered MRI triplet.
"""

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
from typing import Any

import cv2
import matplotlib
import numpy as np
from PIL import Image
from roifile import ImagejRoi, ROI_TYPE

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from grid_transform.analysis_shared import load_curated_speakers
from grid_transform.config import DEFAULT_VTLN_DIR, PROJECT_DIR
from grid_transform.vtln_bundle import scale_contours_to_triplet_space


BASE_VERSION = "0.1.19"
VERSION = "0.1.20"
SOURCE_SHAPE = (136, 136)
TARGET_SHAPE = (480, 480)
SCALE = TARGET_SHAPE[0] / SOURCE_SHAPE[0]
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
EDIT16_LABELS = tuple(label for label in DYNAMIC_LABELS if label != "vocal-folds") + tuple(
    f"c{index}" for index in range(1, 7)
)
FULL17_LABELS = DYNAMIC_LABELS + tuple(f"c{index}" for index in range(1, 7))
DISPLAY_TO_CANONICAL = {
    "upper-incisor": "incisior-hard-palate",
    "lower-incisor": "mandible-incisior",
    **{
        label: label
        for label in DYNAMIC_LABELS
        if label not in {"upper-incisor", "lower-incisor"}
    },
    **{f"c{index}": f"c{index}" for index in range(1, 7)},
}
CANONICAL_TO_DISPLAY = {value: key for key, value in DISPLAY_TO_CANONICAL.items()}

CASE_INFO = {
    "P7": ("007_P7", "1640_P7_S5_F0864"),
    "P8": ("008_P8", "1653_P8_S5_F0792"),
    "P9": ("009_P9", "1659_P9_S5_F0704"),
    "P10": ("010_P10", "1662_P10_S5_F0859"),
}
ASD2_SLICE = "001_ASD2"
ASD2_OLD_BASENAME = "1791_ASD2_S29_F2812"
ASD2_NEW_BASENAME = "1791_ASD2_S29_F2180"
ASD2_FRAME = 2180

DEFAULT_IMAGEJ_ROOT = (
    PROJECT_DIR
    / "outputs"
    / "reference-to-target"
    / "asd2-reference-to-target_P10-stage1-registered"
    / "imagej_sequence"
)
DEFAULT_ASD2_MRI_DIR = (
    PROJECT_DIR.parent
    / "Data"
    / "ArtSpeech_Database_2"
    / "1791"
    / "S29"
    / "NPY_MR_registered"
)
DEFAULT_P10_MATRIX = (
    PROJECT_DIR
    / "registration1"
    / "output"
    / "matrices"
    / "01_stage1_mask_rigid_P10_to_1775_2x3.csv"
)
DEFAULT_OUTPUT_ROOT = (
    PROJECT_DIR
    / "outputs"
    / "geometry_reference_candidates"
    / "v0.1.20_final_imagej_rois"
)
DEFAULT_BACKUP_ROOT = (
    PROJECT_DIR
    / "outputs"
    / "geometry_reference_backups"
    / "v0.1.20_before_final_imagej_rois"
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-vtln-dir", type=Path, default=DEFAULT_VTLN_DIR)
    parser.add_argument("--imagej-root", type=Path, default=DEFAULT_IMAGEJ_ROOT)
    parser.add_argument("--asd2-mri-dir", type=Path, default=DEFAULT_ASD2_MRI_DIR)
    parser.add_argument("--p10-matrix", type=Path, default=DEFAULT_P10_MATRIX)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--backup-root", type=Path, default=DEFAULT_BACKUP_ROOT)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--promote", action="store_true")
    return parser


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def native_matrix_to_triplet(matrix: np.ndarray) -> np.ndarray:
    value = np.asarray(matrix, dtype=float)
    if value.shape != (2, 3):
        raise ValueError(f"Expected P10 2x3 matrix, got {value.shape}")
    converted = value.copy()
    converted[:, 2] *= SCALE
    return converted


def validate_proper_rigid(matrix: np.ndarray) -> dict[str, float]:
    linear = np.asarray(matrix, dtype=float)[:, :2]
    singular_values = np.linalg.svd(linear, compute_uv=False)
    determinant = float(np.linalg.det(linear))
    orthogonality_error = float(np.max(np.abs(linear.T @ linear - np.eye(2))))
    scale_error = float(np.max(np.abs(singular_values - 1.0)))
    if determinant <= 0.0 or orthogonality_error > 1e-8 or scale_error > 1e-8:
        raise ValueError(
            "P10 stage-1 matrix is not proper rigid: "
            f"det={determinant}, orthogonality={orthogonality_error}, "
            f"scale_error={scale_error}"
        )
    return {
        "determinant": determinant,
        "orthogonality_max_abs_error": orthogonality_error,
        "scale_max_error_from_one": scale_error,
        "rotation_deg_image_xy": float(
            np.degrees(np.arctan2(linear[1, 0], linear[0, 0]))
        ),
    }


def identify_display_label(member: str) -> str:
    stem = Path(member).stem.lower()
    for label in sorted(FULL17_LABELS, key=len, reverse=True):
        if stem == label or stem.endswith(f"_{label}"):
            return label
    raise ValueError(f"Unknown ROI label: {member}")


def read_display_roi_zip(path: Path) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    contours: dict[str, np.ndarray] = {}
    metadata: dict[str, Any] = {"members": {}, "zip_sha256": sha256_file(path)}
    with zipfile.ZipFile(path) as archive:
        for member in sorted(archive.namelist()):
            if not member.lower().endswith(".roi"):
                continue
            payload = archive.read(member)
            roi = ImagejRoi.frombytes(payload)
            label = identify_display_label(member)
            if label in contours:
                raise ValueError(f"Duplicate {label} in {path}")
            points = np.asarray(roi.coordinates(), dtype=float)
            contours[label] = points
            metadata["members"][label] = {
                "member": member,
                "sha256": sha256_bytes(payload),
                "roi_type": roi.roitype.name,
                "position": int(roi.position),
                "point_count": int(len(points)),
            }
    return contours, metadata


def read_canonical_zip(path: Path, basename: str) -> dict[str, np.ndarray]:
    contours: dict[str, np.ndarray] = {}
    prefix = f"{basename}_"
    with zipfile.ZipFile(path) as archive:
        for member in archive.namelist():
            if not member.lower().endswith(".roi"):
                continue
            stem = Path(member).stem
            if not stem.startswith(prefix):
                raise ValueError(f"Unexpected canonical member: {member}")
            label = stem[len(prefix) :].lower()
            contours[label] = np.asarray(
                ImagejRoi.frombytes(archive.read(member)).coordinates(), dtype=float
            )
    return contours


def validate_full17(contours: dict[str, np.ndarray], *, allow_missing_vocal: bool = False) -> None:
    expected = {DISPLAY_TO_CANONICAL[label] for label in FULL17_LABELS}
    if allow_missing_vocal:
        expected.remove("vocal-folds")
    if set(contours) != expected:
        raise ValueError(
            f"Canonical labels differ: missing={sorted(expected - set(contours))}, "
            f"extra={sorted(set(contours) - expected)}"
        )
    for canonical, points in contours.items():
        display = CANONICAL_TO_DISPLAY[canonical]
        value = np.asarray(points, dtype=float)
        if value.ndim != 2 or value.shape[1] != 2 or not np.isfinite(value).all():
            raise ValueError(f"Invalid contour {canonical}: {value.shape}")
        if display in DYNAMIC_LABELS and value.shape != (50, 2):
            raise ValueError(f"Dynamic contour must be 50x2: {canonical}/{value.shape}")
        if display.startswith("c") and len(value) < 3:
            raise ValueError(f"Cervical contour is too short: {canonical}/{value.shape}")
        if np.any(value < 0.0) or np.any(value > 479.999):
            raise ValueError(f"Out-of-bounds contour: {canonical}")


def validate_edit16(contours: dict[str, np.ndarray], path: Path) -> None:
    if set(contours) != set(EDIT16_LABELS):
        raise ValueError(
            f"{path}: expected Edit16 labels; "
            f"missing={sorted(set(EDIT16_LABELS) - set(contours))}, "
            f"extra={sorted(set(contours) - set(EDIT16_LABELS))}"
        )
    canonical = {DISPLAY_TO_CANONICAL[label]: points for label, points in contours.items()}
    validate_full17(canonical, allow_missing_vocal=True)


def write_canonical_zip(
    path: Path,
    basename: str,
    contours: dict[str, np.ndarray],
) -> float:
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(
        path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9
    ) as archive:
        for label in sorted(contours):
            points = np.asarray(contours[label], dtype=float)
            name = f"{basename}_{label}"
            roi = ImagejRoi.frompoints(points, name=name)
            roi.roitype = ROI_TYPE.FREEHAND if label.startswith("c") else ROI_TYPE.POLYLINE
            info = zipfile.ZipInfo(f"{name}.roi", ZIP_TIMESTAMP)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.create_system = 3
            info.external_attr = ZIP_FILE_MODE
            archive.writestr(info, roi.tobytes())

    decoded = read_canonical_zip(path, basename)
    errors = []
    for label, expected in contours.items():
        actual = decoded[label]
        if actual.shape != expected.shape:
            raise ValueError(f"ROI shape changed for {label}")
        errors.append(float(np.max(np.linalg.norm(actual - expected, axis=1))))
    maximum = max(errors, default=0.0)
    if maximum >= 1e-3:
        raise ValueError(f"ROI round-trip error too large: {maximum}")
    return maximum


def normalize_registered_triplet(frames: list[np.ndarray]) -> tuple[np.ndarray, dict[str, float]]:
    stack = np.stack([np.asarray(frame) for frame in frames], axis=0)
    if stack.shape != (3, *SOURCE_SHAPE) or not np.isfinite(stack).all():
        raise ValueError(f"Invalid ASD2 triplet: {stack.shape}")
    value_min = float(stack.min())
    value_max = float(stack.max())
    if value_max <= value_min:
        raise ValueError("ASD2 triplet has no intensity range")
    scaled = np.rint(
        (stack.astype(np.float64) - value_min) * 255.0 / (value_max - value_min)
    )
    scaled = np.clip(scaled, 0.0, 255.0).astype(np.uint8)
    resized = [
        cv2.resize(frame, TARGET_SHAPE[::-1], interpolation=cv2.INTER_CUBIC)
        for frame in scaled
    ]
    return np.stack(resized, axis=2), {
        "source_min": value_min,
        "source_max": value_max,
    }


def merge_reviewed_edit16(
    *,
    user_zip: Path,
    full17_zip: Path,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    reviewed, metadata = read_display_roi_zip(user_zip)
    validate_edit16(reviewed, user_zip)
    fallback, fallback_metadata = read_display_roi_zip(full17_zip)
    if "vocal-folds" not in fallback:
        raise ValueError(f"Missing vocal-folds fallback: {full17_zip}")
    display_full17 = {**reviewed, "vocal-folds": fallback["vocal-folds"]}
    canonical = {
        DISPLAY_TO_CANONICAL[label]: np.asarray(points, dtype=float)
        for label, points in display_full17.items()
    }
    validate_full17(canonical)
    metadata["vocal_folds_fallback"] = {
        "path": str(full17_zip.resolve()),
        "zip_sha256": fallback_metadata["zip_sha256"],
        "member": fallback_metadata["members"]["vocal-folds"],
    }
    return canonical, metadata


def read_manifest(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        return list(reader.fieldnames or []), list(reader)


def write_manifest(path: Path, fields: list[str], rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def update_selection_manifest(
    path: Path,
    *,
    imagej_root: Path,
    asd2_mri_dir: Path,
    p10_matrix: Path,
) -> None:
    fields, rows = read_manifest(path)
    for row in rows:
        speaker = row["speaker"]
        if speaker in CASE_INFO:
            slice_name, _ = CASE_INFO[speaker]
            roi_path = imagej_root / "per_slice" / slice_name / "RoiSet.zip"
            row["annotation_source"] = str(roi_path.resolve())
            row["annotation_origin_path"] = str(roi_path.resolve())
            row["annotation_status"] = "final_user_reviewed_ImageJ_RoiSet"
        if speaker == "P10":
            row["image_source"] = (
                f"{row['image_source']};stage1_registered_with={p10_matrix.resolve()}"
            )
            row["annotation_space"] = "480x480_stage1_registered_default"
        if speaker == "ASD2":
            roi_path = imagej_root / "per_slice" / ASD2_SLICE / "RoiSet.zip"
            row.update(
                {
                    "output_basename": ASD2_NEW_BASENAME,
                    "selected_source": f"ASD2/S29/F{ASD2_FRAME:04d}",
                    "image_source": (
                        f"{asd2_mri_dir.resolve()}#frames="
                        f"{ASD2_FRAME - 1},{ASD2_FRAME},{ASD2_FRAME + 1};"
                        "shared_minmax_uint8"
                    ),
                    "annotation_source": str(roi_path.resolve()),
                    "annotation_status": "final_user_reviewed_ImageJ_RoiSet_plus_full17_vocal_folds",
                    "annotation_origin_path": str(roi_path.resolve()),
                    "reference_bundle_name": ASD2_NEW_BASENAME,
                    "prev_frame_1based": str(ASD2_FRAME - 1),
                    "center_frame_1based": str(ASD2_FRAME),
                    "next_frame_1based": str(ASD2_FRAME + 1),
                    "annotation_space": "480x480_scaled_from_registered_136x136",
                }
            )
    if [row["speaker"] for row in rows] != [
        "P1",
        "P2",
        "P3",
        "P4",
        "P5",
        "P6",
        "P7",
        "P8",
        "P9",
        "P10",
        "ASD2",
    ]:
        raise ValueError("Unexpected GTGRD speaker order")
    write_manifest(path, fields, rows)


def update_build_summary(
    path: Path,
    *,
    candidate_dir: Path,
    import_audit: dict[str, Any],
    matrix_path: Path,
    matrix_diagnostics: dict[str, float],
) -> None:
    summary = json.loads(path.read_text(encoding="utf-8"))
    if summary.get("version") != BASE_VERSION or summary.get("status") != "PASS":
        raise ValueError("GTGRD base is not the expected PASS v0.1.19")
    for case in summary.get("cases", []):
        speaker = case.get("speaker")
        if speaker == "ASD2":
            case.update(
                {
                    "basename": ASD2_NEW_BASENAME,
                    "selected_frame": ASD2_FRAME,
                    "phone_selection": "exact /u/ in cou; final ImageJ review",
                    "source_roi_zip": import_audit["ASD2"]["user_roi_zip"],
                    "source_roi_zip_sha256": import_audit["ASD2"]["user_roi_zip_sha256"],
                    "incisor_policy": "latest grele-13 BF snapshot plus final lower-incisor ImageJ edit",
                    "cervical_policy": "retained same-session C1-C6 with final ImageJ RoiSet coordinates",
                }
            )
        if speaker in {*CASE_INFO, "ASD2"}:
            basename = ASD2_NEW_BASENAME if speaker == "ASD2" else CASE_INFO[speaker][1]
            case["png_sha256"] = sha256_file(candidate_dir / f"{basename}.png")
            case["zip_sha256"] = sha256_file(candidate_dir / f"{basename}.zip")
            case["final_imagej_import"] = import_audit[speaker]

    summary.update(
        {
            "schema_version": "gtgrd-v0.1.20-final-imagej-registered-p10-v1",
            "version": VERSION,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "status": "PASS",
            "scope": (
                "GTGRD v0.1.19 plus final ImageJ edits for ASD2/P7/P8/P9/P10; "
                "ASD2 uses S29/F2180 and P10 is registered by default"
            ),
            "p10_coordinate_policy": {
                "default": "stage1_registered",
                "matrix": str(matrix_path.resolve()),
                "matrix_sha256": sha256_file(matrix_path),
                **matrix_diagnostics,
                "double_registration_forbidden": True,
            },
            "asd2_reference_policy": {
                "role": "additional transform/reference geometry; does not replace P10",
                "speaker_session_frame": "1791/S29/F2180",
                "word_phone": "cou /u/",
                "matching_mri_triplet": [2179, 2180, 2181],
            },
            "final_imagej_import": import_audit,
            "base_v019_untouched_speakers": ["P1", "P2", "P3", "P4", "P5", "P6"],
        }
    )
    path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")


def readme_text() -> str:
    return f"""# GTGRD v{VERSION}

GTGRD means Grid Transform Geometry Reference Data. `VTLN/data` remains the
runtime compatibility path.

- The authoritative final ImageJ `RoiSet.zip` edits are imported for ASD2, P7,
  P8, P9, and P10. P1-P6 and P2 remain unchanged from v{BASE_VERSION}.
- P10/S5/F0859 is stored in the accepted stage-1 registered coordinate system
  by default. Its RGB MRI triplet, all reviewed contours, `vocal-folds`, and
  C1-C6 share that coordinate system. Downstream code must not apply the rigid
  registration a second time.
- ASD2 is `1791/S29/F2180`, the exact `/u/` in `cou`. Its registered MRI
  triplet is F2179-F2181. The 16 edited ROIs come from the per-slice
  `RoiSet.zip`; `vocal-folds` is retained from the matching F2180 full17 set.
- P7-P10 use their final per-slice ImageJ review coordinates. Editing packages
  omit `vocal-folds`, so the matching full17 target value is preserved.
- Dynamic contours are open polylines; C1-C6 are stored as closed FREEHAND
  contours. Historical canonical names `incisior-hard-palate` and
  `mandible-incisior` remain unchanged.
- No smoothing, reordering, landmark redefinition, metric change, affine/TPS
  protocol change, or imputation is introduced. P2 still lacks `vocal-folds`.
"""


def export_review_snapshot(candidate_dir: Path, output_root: Path) -> Path:
    snapshot = output_root / "gtgrd_v0.1.20_metric_contours_native136"
    snapshot.mkdir(parents=True)
    speakers = ("P1", "P3", "P4", "P5", "P6", "P7", "P8", "P9", "P10")
    loaded = load_curated_speakers(candidate_dir, list(speakers))
    for speaker, item in loaded.items():
        frame = int(item.spec.frame or 0)
        case_dir = snapshot / "cases" / f"{speaker}_S5_F{frame:04d}"
        contour_dir = case_dir / "contours_npy"
        contour_dir.mkdir(parents=True)
        display = {
            CANONICAL_TO_DISPLAY[label]: np.asarray(points, dtype=float) / SCALE
            for label, points in item.contours.items()
            if label in CANONICAL_TO_DISPLAY
            and CANONICAL_TO_DISPLAY[label] in DYNAMIC_LABELS
        }
        if speaker == "P2":
            continue
        if set(display) != set(DYNAMIC_LABELS):
            raise ValueError(f"Incomplete dynamic GTGRD snapshot for {speaker}")
        for label, points in display.items():
            np.save(contour_dir / f"{label}.npy", points)
        (case_dir / "metadata.json").write_text(
            json.dumps(
                {
                    "speaker": speaker,
                    "session": "S5",
                    "frame": frame,
                    "coordinate_space": "native136 values derived exactly from GTGRD v0.1.20",
                    "source_zip": str(item.spec.zip_path),
                    "source_zip_sha256": sha256_file(Path(item.spec.zip_path)),
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
    return snapshot


def write_asd2_native_source_package(
    *,
    output_root: Path,
    contours_480: dict[str, np.ndarray],
    mri_path: Path,
) -> dict[str, str]:
    root = output_root / "ASD2_1791_S29_F2180_native136_source"
    root.mkdir(parents=True)
    native_image = np.asarray(np.load(mri_path, allow_pickle=False))
    if native_image.shape != SOURCE_SHAPE:
        raise ValueError(f"Unexpected ASD2 native MRI: {native_image.shape}")
    image_path = root / "1791_S29_F2180_registered_MRI.tif"
    Image.fromarray(native_image).save(image_path)
    display_native = {
        CANONICAL_TO_DISPLAY[label]: np.asarray(points, dtype=float) / SCALE
        for label, points in contours_480.items()
    }
    roi_path = root / "RoiSet_F2180_final_full17.zip"
    with zipfile.ZipFile(
        roi_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9
    ) as archive:
        for display in DYNAMIC_LABELS + tuple(f"c{i}" for i in range(1, 7)):
            output_label = display.upper() if display.startswith("c") else display
            name = f"ASD2_{output_label}"
            roi = ImagejRoi.frompoints(display_native[display], name=name)
            roi.roitype = ROI_TYPE.FREEHAND if display.startswith("c") else ROI_TYPE.POLYLINE
            info = zipfile.ZipInfo(f"{name}.roi", ZIP_TIMESTAMP)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.create_system = 3
            info.external_attr = ZIP_FILE_MODE
            archive.writestr(info, roi.tobytes())
    return {
        "image": str(image_path.resolve()),
        "image_sha256": sha256_file(image_path),
        "roi_zip": str(roi_path.resolve()),
        "roi_zip_sha256": sha256_file(roi_path),
    }


def render_annotation_checks(candidate_dir: Path, output_root: Path) -> dict[str, Any]:
    speakers = tuple(f"P{index}" for index in range(1, 11)) + ("ASD2",)
    loaded = load_curated_speakers(candidate_dir, list(speakers))
    per_speaker_root = output_root / "annotation_review" / "per_speaker"
    per_speaker_root.mkdir(parents=True)
    colors = plt.get_cmap("tab20")(np.linspace(0.0, 1.0, len(FULL17_LABELS)))
    paths: dict[str, str] = {}

    def draw(axis: plt.Axes, speaker: str) -> None:
        item = loaded[speaker]
        axis.imshow(item.image)
        for color, display in zip(colors, FULL17_LABELS):
            canonical = DISPLAY_TO_CANONICAL[display]
            if canonical not in item.contours:
                continue
            points = np.asarray(item.contours[canonical], dtype=float)
            shown = (
                np.vstack((points, points[0]))
                if display.startswith("c") and not np.allclose(points[0], points[-1])
                else points
            )
            axis.plot(shown[:, 0], shown[:, 1], color=color, linewidth=1.25)
        registered = " | registered default" if speaker == "P10" else ""
        axis.set_title(f"{speaker} | {item.spec.session}/F{item.spec.frame:04d}{registered}")
        axis.set_xlim(0, 480)
        axis.set_ylim(480, 0)
        axis.set_aspect("equal")
        axis.axis("off")

    for speaker in speakers:
        path = per_speaker_root / f"{speaker}_GTGRD_v0.1.20_contours_over_MRI.png"
        figure, axis = plt.subplots(figsize=(6.2, 6.2), constrained_layout=True)
        draw(axis, speaker)
        figure.savefig(path, dpi=180, facecolor="white")
        plt.close(figure)
        paths[speaker] = str(path.resolve())

    sheet_path = output_root / "annotation_review" / "GTGRD_v0.1.20_all_speakers_sheet.png"
    figure, axes = plt.subplots(3, 4, figsize=(16, 12), constrained_layout=True)
    for axis, speaker in zip(axes.flat, speakers):
        draw(axis, speaker)
    axes.flat[-1].axis("off")
    figure.suptitle(
        "GTGRD v0.1.20 | final reviewed contours | P10 registered by default",
        fontsize=16,
        fontweight="bold",
    )
    figure.savefig(sheet_path, dpi=180, facecolor="white")
    plt.close(figure)
    return {"per_speaker": paths, "sheet": str(sheet_path.resolve())}


def tree_hashes(root: Path) -> dict[str, str]:
    return {
        path.relative_to(root).as_posix(): sha256_file(path)
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def promote_candidate(candidate_dir: Path, target_dir: Path, backup_root: Path) -> None:
    if backup_root.exists():
        raise FileExistsError(f"Backup already exists: {backup_root}")
    backup_data = backup_root / "VTLN" / "data"
    backup_data.parent.mkdir(parents=True)
    shutil.copytree(target_dir, backup_data)
    if tree_hashes(backup_data) != tree_hashes(target_dir):
        raise RuntimeError("GTGRD backup hash verification failed")

    current = tree_hashes(target_dir)
    candidate = tree_hashes(candidate_dir)
    removed = sorted(set(current) - set(candidate))
    changed = sorted(
        relative
        for relative, digest in candidate.items()
        if current.get(relative) != digest
    )
    for relative in removed:
        target = target_dir / relative
        target.unlink()
    for relative in changed:
        source = candidate_dir / relative
        target = target_dir / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        handle, temporary_name = tempfile.mkstemp(prefix=f".{target.name}.", dir=target.parent)
        os.close(handle)
        temporary = Path(temporary_name)
        try:
            shutil.copy2(source, temporary)
            os.replace(temporary, target)
        finally:
            if temporary.exists():
                temporary.unlink()
    if tree_hashes(target_dir) != candidate:
        raise RuntimeError("Promoted GTGRD does not match validated candidate")


def run(args: argparse.Namespace) -> dict[str, Any]:
    base_dir = args.base_vtln_dir.resolve()
    imagej_root = args.imagej_root.resolve()
    output_root = args.output_root.resolve()
    backup_root = args.backup_root.resolve()
    if output_root.exists():
        if not args.overwrite:
            raise FileExistsError(f"Candidate exists: {output_root}")
        shutil.rmtree(output_root)
    candidate_dir = output_root / "VTLN" / "data"
    candidate_dir.parent.mkdir(parents=True)
    shutil.copytree(base_dir, candidate_dir)

    base_summary = json.loads((base_dir / "build_summary.json").read_text(encoding="utf-8"))
    if base_summary.get("version") != BASE_VERSION or base_summary.get("status") != "PASS":
        raise ValueError("Expected canonical PASS GTGRD v0.1.19 input")

    matrix_native = np.loadtxt(args.p10_matrix, delimiter=",")
    matrix_diagnostics = validate_proper_rigid(matrix_native)
    matrix_triplet = native_matrix_to_triplet(matrix_native)
    import_audit: dict[str, Any] = {}
    roundtrip_errors: dict[str, float] = {}

    for speaker, (slice_name, basename) in CASE_INFO.items():
        slice_dir = imagej_root / "per_slice" / slice_name
        user_zip = slice_dir / "RoiSet.zip"
        full17_zip = slice_dir / f"{slice_name}_existing_target_full17_RoiSet.zip"
        canonical, audit = merge_reviewed_edit16(
            user_zip=user_zip,
            full17_zip=full17_zip,
        )
        output_zip = candidate_dir / f"{basename}.zip"
        roundtrip_errors[speaker] = write_canonical_zip(output_zip, basename, canonical)
        import_audit[speaker] = {
            "user_roi_zip": str(user_zip.resolve()),
            "user_roi_zip_sha256": sha256_file(user_zip),
            "full17_vocal_folds_source": str(full17_zip.resolve()),
            "coordinate_policy": (
                "stage1_registered_default_no_inverse"
                if speaker == "P10"
                else "unchanged_GTGRD_480_space"
            ),
            "roi_metadata": audit,
        }

    p10_basename = CASE_INFO["P10"][1]
    p10_image_path = candidate_dir / f"{p10_basename}.png"
    p10_image = np.asarray(Image.open(p10_image_path))
    if p10_image.shape != (*TARGET_SHAPE, 3):
        raise ValueError(f"Unexpected P10 RGB triplet: {p10_image.shape}")
    p10_registered = cv2.warpAffine(
        p10_image,
        matrix_triplet,
        TARGET_SHAPE[::-1],
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    Image.fromarray(p10_registered).save(p10_image_path)
    p10_review_mri = np.asarray(
        Image.open(imagej_root / "per_slice" / "010_P10" / "010_P10_MRI.tif")
    )
    p10_mri_error = int(
        np.max(np.abs(p10_registered[..., 1].astype(int) - p10_review_mri.astype(int)))
    )
    if p10_mri_error != 0:
        raise ValueError(f"P10 registered MRI differs from reviewed sequence: {p10_mri_error}")

    asd2_slice = imagej_root / "per_slice" / ASD2_SLICE
    asd2_user_zip = asd2_slice / "RoiSet.zip"
    asd2_full17_zip = asd2_slice / f"{ASD2_SLICE}_ASD2_adapted_full17_RoiSet.zip"
    asd2_canonical, asd2_audit = merge_reviewed_edit16(
        user_zip=asd2_user_zip,
        full17_zip=asd2_full17_zip,
    )
    old_png = candidate_dir / f"{ASD2_OLD_BASENAME}.png"
    old_zip = candidate_dir / f"{ASD2_OLD_BASENAME}.zip"
    if not old_png.is_file() or not old_zip.is_file():
        raise FileNotFoundError("Expected v0.1.19 ASD2 F2812 files")
    old_png.unlink()
    old_zip.unlink()
    frames = [
        np.load(args.asd2_mri_dir.resolve() / f"{frame:04d}.npy", allow_pickle=False)
        for frame in (ASD2_FRAME - 1, ASD2_FRAME, ASD2_FRAME + 1)
    ]
    asd2_triplet, asd2_intensity = normalize_registered_triplet(frames)
    Image.fromarray(asd2_triplet).save(candidate_dir / f"{ASD2_NEW_BASENAME}.png")
    roundtrip_errors["ASD2"] = write_canonical_zip(
        candidate_dir / f"{ASD2_NEW_BASENAME}.zip",
        ASD2_NEW_BASENAME,
        asd2_canonical,
    )
    import_audit["ASD2"] = {
        "user_roi_zip": str(asd2_user_zip.resolve()),
        "user_roi_zip_sha256": sha256_file(asd2_user_zip),
        "full17_vocal_folds_source": str(asd2_full17_zip.resolve()),
        "coordinate_policy": "registered_S29_F2180_480_space",
        "mri_triplet_frames": [2179, 2180, 2181],
        "mri_intensity": asd2_intensity,
        "roi_metadata": asd2_audit,
    }

    update_selection_manifest(
        candidate_dir / "selection_manifest.csv",
        imagej_root=imagej_root,
        asd2_mri_dir=args.asd2_mri_dir,
        p10_matrix=args.p10_matrix,
    )
    (candidate_dir / "README.md").write_text(readme_text(), encoding="utf-8")
    update_build_summary(
        candidate_dir / "build_summary.json",
        candidate_dir=candidate_dir,
        import_audit=import_audit,
        matrix_path=args.p10_matrix,
        matrix_diagnostics=matrix_diagnostics,
    )

    loaded = load_curated_speakers(
        candidate_dir, [*(f"P{index}" for index in range(1, 11)), "ASD2"]
    )
    grid_checks = {
        speaker: {
            "horizontal": len(item.grid.horiz_lines),
            "vertical": len(item.grid.vert_lines),
            "warnings": list(item.grid.warnings),
            "hard_errors": list(item.grid.hard_errors),
        }
        for speaker, item in loaded.items()
    }
    invalid_grids = [
        speaker
        for speaker, check in grid_checks.items()
        if check["horizontal"] != 6
        or check["vertical"] != 9
        or check["warnings"]
        or check["hard_errors"]
    ]
    if invalid_grids:
        raise ValueError(f"Invalid GTGRD grids: {invalid_grids}")

    snapshot_root = export_review_snapshot(candidate_dir, output_root)
    source_package = write_asd2_native_source_package(
        output_root=output_root,
        contours_480=asd2_canonical,
        mri_path=args.asd2_mri_dir.resolve() / f"{ASD2_FRAME:04d}.npy",
    )
    review_images = render_annotation_checks(candidate_dir, output_root)
    base_hashes = tree_hashes(base_dir)
    candidate_hashes = tree_hashes(candidate_dir)
    changed = sorted(
        relative
        for relative, digest in candidate_hashes.items()
        if base_hashes.get(relative) != digest
    )
    removed = sorted(set(base_hashes) - set(candidate_hashes))
    added = sorted(set(candidate_hashes) - set(base_hashes))
    summary: dict[str, Any] = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS",
        "base_version": BASE_VERSION,
        "version": VERSION,
        "candidate_dir": str(candidate_dir.resolve()),
        "promoted": False,
        "changed_files": changed,
        "removed_files": removed,
        "added_files": added,
        "imported_speakers": ["ASD2", "P7", "P8", "P9", "P10"],
        "untouched_speakers": ["P1", "P2", "P3", "P4", "P5", "P6"],
        "p10_registered_default": True,
        "p10_registered_mri_review_max_error_uint8": p10_mri_error,
        "roi_roundtrip_max_error_px_480": max(roundtrip_errors.values()),
        "roi_roundtrip_by_speaker_px_480": roundtrip_errors,
        "grid_checks": grid_checks,
        "metric_contour_snapshot": str(snapshot_root.resolve()),
        "asd2_native_source_package": source_package,
        "annotation_review": review_images,
    }
    if args.promote:
        promote_candidate(candidate_dir, base_dir, backup_root)
        summary["promoted"] = True
        summary["backup_root"] = str(backup_root)
    (output_root / "candidate_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def main(argv: list[str] | None = None) -> int:
    summary = run(build_parser().parse_args(argv))
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
