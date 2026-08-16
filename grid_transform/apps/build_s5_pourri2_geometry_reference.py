from __future__ import annotations

"""Build a non-destructive ten-speaker S5 pourri #2 geometry candidate."""

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

from grid_transform.analysis_shared import load_curated_speakers
from grid_transform.apps.render_s5_pourri2_p7_affine_tps_sheets import (
    CONTOUR_ORDER,
    build_stage_geometry,
    render_all_stage_contour_overlay,
    render_tps_m1_l6_match_sheet,
)
from grid_transform.config import DEFAULT_VTLN_DIR, PROJECT_DIR
from grid_transform.vt import build_grid
from grid_transform.vtln_bundle import scale_contours_to_triplet_space


VERSION = "0.1.18"
SESSION = "S5"
REFERENCE_SPEAKER = "P10"
SPEAKERS = tuple(f"P{index}" for index in range(1, 11))
SOURCE_SHAPE = (136, 136)
TARGET_SHAPE = (480, 480)
MRI_CROP = (90, 92, 270, 270)
CERVICAL_LABELS = tuple(f"c{index}" for index in range(1, 7))
CANONICAL_LABEL_MAP = {
    "upper-incisor": "incisior-hard-palate",
    "lower-incisor": "mandible-incisior",
}
P2_MISSING_LABEL = "vocal-folds"

DEFAULT_SELECTION_CSV = (
    PROJECT_DIR
    / "outputs"
    / "s5_pourri_repetitions_p1_p10_20260814"
    / "pourri_three_repetitions_p1_p10.csv"
)
DEFAULT_INFERENCE_ROOT = PROJECT_DIR.parent / "Preprocess" / "inference"
DEFAULT_VIDEO_ROOT = PROJECT_DIR.parent / "Data" / "Need-to-verify-alignment"
DEFAULT_REVIEW_ROOT = DEFAULT_VTLN_DIR / "review_s5_pourri2"
DEFAULT_OUTPUT_ROOT = (
    PROJECT_DIR / "outputs" / "geometry_reference_candidates" / f"v{VERSION}_s5_pourri2"
)
DEFAULT_OVERLAY_OUTPUT_DIR = (
    PROJECT_DIR / "outputs" / "s5_pourri2_p10_affine_tps_imagej_edited_20260815"
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection-csv", type=Path, default=DEFAULT_SELECTION_CSV)
    parser.add_argument("--inference-root", type=Path, default=DEFAULT_INFERENCE_ROOT)
    parser.add_argument("--video-root", type=Path, default=DEFAULT_VIDEO_ROOT)
    parser.add_argument(
        "--review-root",
        type=Path,
        default=DEFAULT_REVIEW_ROOT,
        help=(
            "S5 pourri #2 ImageJ review workspace. Its exact manually reviewed "
            "lower-incisor prototype is authoritative for P1 and P3-P10."
        ),
    )
    parser.add_argument("--current-vtln-dir", type=Path, default=DEFAULT_VTLN_DIR)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--overlay-output-dir",
        type=Path,
        default=DEFAULT_OVERLAY_OUTPUT_DIR,
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def load_selection(path: Path) -> dict[str, dict[str, str]]:
    with Path(path).open("r", encoding="utf-8", newline="") as stream:
        rows = [row for row in csv.DictReader(stream) if row["candidate"] == "pourri/r2"]
    rows_by_speaker = {row["speaker"]: row for row in rows}
    if tuple(rows_by_speaker) != SPEAKERS:
        raise ValueError(f"Expected ordered {SPEAKERS}, got {tuple(rows_by_speaker)}")
    for speaker, row in rows_by_speaker.items():
        expected = "old_textgrid_fallback" if speaker == "P10" else "incoming_*_c"
        if row["alignment_source"] != expected:
            raise ValueError(
                f"Unexpected alignment source for {speaker}: {row['alignment_source']}"
            )
    return rows_by_speaker


def load_native_contours(
    inference_root: Path,
    speaker: str,
    frame: int,
) -> tuple[dict[str, np.ndarray], dict[str, Path]]:
    contour_dir = Path(inference_root) / speaker / SESSION / "contours"
    prefix = f"{frame:04d}_"
    paths = sorted(contour_dir.glob(f"{prefix}*.npy"))
    by_label = {path.stem[len(prefix) :]: path for path in paths}
    expected = set(CONTOUR_ORDER)
    actual = set(by_label)
    allowed_missing = {P2_MISSING_LABEL} if speaker == "P2" else set()
    if expected - actual != allowed_missing or actual - expected:
        raise ValueError(
            f"{speaker}/S5/F{frame:04d}: missing={sorted(expected - actual)}, "
            f"extra={sorted(actual - expected)}"
        )

    contours: dict[str, np.ndarray] = {}
    for label, path in by_label.items():
        points = np.asarray(np.load(path, allow_pickle=False), dtype=float)
        if points.shape != (50, 2) or not np.isfinite(points).all():
            raise ValueError(f"Invalid {speaker}/S5/F{frame:04d}/{label}: {points.shape}")
        if (
            np.any(points[:, 0] < 0.0)
            or np.any(points[:, 0] > SOURCE_SHAPE[1] - 1.0)
            or np.any(points[:, 1] < 0.0)
            or np.any(points[:, 1] > SOURCE_SHAPE[0] - 1.0)
        ):
            raise ValueError(f"Out-of-bounds {speaker}/S5/F{frame:04d}/{label}")
        contours[label] = points
    return contours, by_label


def load_reviewed_lower_incisor(
    review_root: Path,
    speaker: str,
    frame: int,
) -> tuple[np.ndarray, Path]:
    path = (
        Path(review_root)
        / "cases"
        / f"{speaker}_{SESSION}_F{frame:04d}"
        / "contours_npy"
        / "lower-incisor.npy"
    )
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing manually reviewed lower-incisor prototype for {speaker}: {path}"
        )
    points = np.asarray(np.load(path, allow_pickle=False), dtype=float)
    if points.shape != (50, 2) or not np.isfinite(points).all():
        raise ValueError(f"Invalid reviewed lower-incisor for {speaker}: {points.shape}")
    if (
        np.any(points[:, 0] < 0.0)
        or np.any(points[:, 0] > SOURCE_SHAPE[1] - 1.0)
        or np.any(points[:, 1] < 0.0)
        or np.any(points[:, 1] > SOURCE_SHAPE[0] - 1.0)
    ):
        raise ValueError(f"Out-of-bounds reviewed lower-incisor for {speaker}")
    return points, path


def read_video_gray_frame(video_path: Path, frame_1based: int) -> np.ndarray:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise FileNotFoundError(video_path)
    capture.set(cv2.CAP_PROP_POS_FRAMES, frame_1based - 1)
    ok, image = capture.read()
    capture.release()
    if not ok or image is None:
        raise RuntimeError(f"Could not decode {video_path} frame {frame_1based}")
    if image.shape[:2] != (600, 450):
        raise ValueError(f"Unexpected video frame shape {image.shape}: {video_path}")
    x0, y0, width, height = MRI_CROP
    crop = image[y0 : y0 + height, x0 : x0 + width]
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    return cv2.resize(gray, TARGET_SHAPE[::-1], interpolation=cv2.INTER_CUBIC)


def load_rgb_triplet(video_path: Path, center_frame_1based: int) -> np.ndarray:
    frames = [
        read_video_gray_frame(video_path, frame)
        for frame in (
            center_frame_1based - 1,
            center_frame_1based,
            center_frame_1based + 1,
        )
    ]
    triplet = np.stack(frames, axis=2).astype(np.uint8)
    if triplet.shape != (480, 480, 3):
        raise ValueError(f"Invalid triplet shape: {triplet.shape}")
    return triplet


def canonicalize_contours(
    native_contours: dict[str, np.ndarray],
    cervical_contours: dict[str, np.ndarray],
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    scaled_dynamic = scale_contours_to_triplet_space(
        native_contours,
        SOURCE_SHAPE,
        TARGET_SHAPE,
    )
    canonical = {
        CANONICAL_LABEL_MAP.get(label, label): np.asarray(points, dtype=float)
        for label, points in scaled_dynamic.items()
    }
    canonical.update(
        {
            label: np.asarray(cervical_contours[label], dtype=float).copy()
            for label in CERVICAL_LABELS
        }
    )
    return canonical, scaled_dynamic


def write_annotation_zip(
    path: Path,
    basename: str,
    contours: dict[str, np.ndarray],
) -> None:
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for label, points in sorted(contours.items()):
            array = np.asarray(points, dtype=float)
            if array.ndim != 2 or array.shape[1] != 2 or len(array) < 2:
                raise ValueError(f"Invalid canonical contour {label}: {array.shape}")
            roi_name = f"{basename}_{label}"
            roi = ImagejRoi.frompoints(array, name=roi_name)
            roi.roitype = (
                ROI_TYPE.FREEHAND if label in CERVICAL_LABELS else ROI_TYPE.POLYLINE
            )
            archive.writestr(f"{roi_name}.roi", roi.tobytes())


def write_manifest(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def image_record(path: Path) -> dict[str, object]:
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise RuntimeError(f"Unreadable image: {path}")
    return {
        "relative_path": path.name,
        "width": int(image.shape[1]),
        "height": int(image.shape[0]),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def candidate_readme() -> str:
    return f"""# GTGRD v{VERSION}

GTGRD means Grid Transform Geometry Reference Data. `VTLN/data` remains the
runtime compatibility path.

- Ten top-level references P1-P10 use the midpoint frame of S5 `pourri #2 /u/`.
- RGB triplets are `R=t-1, G=t, B=t+1`, cropped from the matching review AVI and resized to 480x480.
- Dynamic contours are the latest available native 136x136 inference annotations, scaled directly to 480x480.
- For P1 and P3-P10, `lower-incisor` is the exact manually reviewed ImageJ prototype from the S5 pourri #2 review workspace. This deliberately takes precedence over the later all-frame propagated copy, which preserves placement but not the exact reviewed 50-point prototype.
- P2 has no manually reviewed lower-incisor prototype, so its observed inference contour is retained.
- `upper-incisor` and `lower-incisor` are stored under the historical canonical labels `incisior-hard-palate` and `mandible-incisior`.
- C1-C6 remain the fixed speaker-specific auxiliary contours from the preceding canonical geometry release.
- P2 has ten observed dynamic contours; `vocal-folds` is absent at its selected interval and is not imputed.
- All dynamic ROIs are stored as open ImageJ polylines; C1-C6 retain closed FREEHAND topology.
"""


def build_candidate(args: argparse.Namespace) -> dict[str, object]:
    output_root = Path(args.output_root).resolve()
    if output_root.exists() and not args.overwrite:
        raise FileExistsError(f"Candidate exists: {output_root}; use --overwrite")
    output_root.parent.mkdir(parents=True, exist_ok=True)
    stage_root = Path(
        tempfile.mkdtemp(prefix=f".v{VERSION}-candidate-", dir=output_root.parent)
    )
    data_dir = stage_root / "VTLN" / "data"
    diagnostics_dir = stage_root / "diagnostics"
    data_dir.mkdir(parents=True)
    diagnostics_dir.mkdir()

    current_vtln_dir = Path(args.current_vtln_dir).resolve()
    selection_rows = load_selection(args.selection_csv)
    current_speakers = load_curated_speakers(current_vtln_dir, list(SPEAKERS))
    current_files = sorted(
        path for path in current_vtln_dir.iterdir() if path.is_file()
    )
    current_hashes_before = {path.name: sha256_file(path) for path in current_files}
    dynamic_by_speaker: dict[str, dict[str, np.ndarray]] = {}
    manifest_rows: list[dict[str, object]] = []
    case_records: list[dict[str, object]] = []

    try:
        for speaker in SPEAKERS:
            selection = selection_rows[speaker]
            frame = int(selection["selected_frame"])
            spec = current_speakers[speaker].spec
            if spec.raw_subject is None:
                raise ValueError(f"Missing raw subject for {speaker}")
            basename = f"{spec.raw_subject}_{speaker}_{SESSION}_F{frame:04d}"
            video_path = (
                Path(args.video_root) / speaker / SESSION / f"VIDEO_{speaker}_{SESSION}.avi"
            )
            native, native_paths = load_native_contours(
                args.inference_root,
                speaker,
                frame,
            )
            reviewed_lower_path: Path | None = None
            propagated_lower_rms_px = 0.0
            propagated_lower_max_px = 0.0
            if speaker != "P2":
                propagated_lower = native["lower-incisor"].copy()
                reviewed_lower, reviewed_lower_path = load_reviewed_lower_incisor(
                    args.review_root,
                    speaker,
                    frame,
                )
                displacement = np.linalg.norm(
                    propagated_lower - reviewed_lower,
                    axis=1,
                )
                propagated_lower_rms_px = float(
                    np.sqrt(np.mean(np.square(displacement)))
                )
                propagated_lower_max_px = float(np.max(displacement))
                native["lower-incisor"] = reviewed_lower
                native_paths["lower-incisor"] = reviewed_lower_path
            canonical, scaled_dynamic = canonicalize_contours(
                native,
                current_speakers[speaker].contours,
            )
            triplet = load_rgb_triplet(video_path, frame)
            png_path = data_dir / f"{basename}.png"
            zip_path = data_dir / f"{basename}.zip"
            Image.fromarray(triplet, mode="RGB").save(png_path)
            write_annotation_zip(zip_path, basename, canonical)
            grid = build_grid(
                triplet,
                canonical,
                n_vert=9,
                n_points=250,
                frame_number=frame,
            )

            display_dynamic = {
                label: np.asarray(points, dtype=float).copy()
                for label, points in scaled_dynamic.items()
            }
            display_dynamic.setdefault(P2_MISSING_LABEL, np.empty((0, 2), dtype=float))
            dynamic_by_speaker[speaker] = display_dynamic
            missing = sorted(set(CONTOUR_ORDER) - set(native))
            annotation_hashes = {
                label: sha256_file(path) for label, path in sorted(native_paths.items())
            }
            case_records.append(
                {
                    "speaker": speaker,
                    "basename": basename,
                    "session": SESSION,
                    "selected_frame": frame,
                    "phone_selection": "pourri/r2/sole_u/midpoint",
                    "alignment_source": selection["alignment_source"],
                    "dynamic_contour_count": len(native),
                    "canonical_contour_count": len(canonical),
                    "missing_dynamic_labels": missing,
                    "missing_policy": "observed absence; no imputation" if missing else "none",
                    "lower_incisor_source_policy": (
                        "observed_inference_no_manual_review"
                        if speaker == "P2"
                        else "exact_imagej_reviewed_pourri2_prototype"
                    ),
                    "lower_incisor_source_path": str(
                        (
                            Path(args.inference_root)
                            / speaker
                            / SESSION
                            / "contours"
                            / f"{frame:04d}_lower-incisor.npy"
                        ).resolve()
                        if reviewed_lower_path is None
                        else reviewed_lower_path.resolve()
                    ),
                    "propagated_vs_reviewed_lower_incisor_rms_px": propagated_lower_rms_px,
                    "propagated_vs_reviewed_lower_incisor_max_px": propagated_lower_max_px,
                    "source_annotation_sha256": annotation_hashes,
                    "png_sha256": sha256_file(png_path),
                    "zip_sha256": sha256_file(zip_path),
                    "grid_horiz_count": len(grid.horiz_lines),
                    "grid_vert_count": len(grid.vert_lines),
                    "grid_warnings": list(grid.warnings),
                }
            )
            manifest_rows.append(
                {
                    "output_basename": basename,
                    "speaker": speaker,
                    "raw_subject": spec.raw_subject,
                    "session": SESSION,
                    "selected_source": f"{speaker}/{SESSION}/F{frame:04d}",
                    "image_source": f"{video_path.resolve()}#frames={frame - 1},{frame},{frame + 1}",
                    "annotation_source": str(
                        (Path(args.inference_root) / speaker / SESSION / "contours").resolve()
                    ),
                    "annotation_status": (
                        "latest_available_10_of_11_missing_vocal_folds"
                        if speaker == "P2"
                        else "latest_available_with_exact_imagej_reviewed_lower_incisor"
                    ),
                    "annotation_origin_path": str(
                        (Path(args.inference_root) / speaker / SESSION / "contours").resolve()
                    ),
                    "reference_bundle_dir": str(current_vtln_dir),
                    "reference_bundle_name": basename,
                    "prev_frame_1based": frame - 1,
                    "center_frame_1based": frame,
                    "next_frame_1based": frame + 1,
                    "channel_order": "R=t-1,G=t,B=t+1",
                    "output_size": "480x480",
                    "annotation_space": "480x480_scaled_from_native_136x136",
                }
            )

        write_manifest(data_dir / "selection_manifest.csv", manifest_rows)
        (data_dir / "README.md").write_text(candidate_readme(), encoding="utf-8")
        loaded_candidate = load_curated_speakers(data_dir, list(SPEAKERS))
        stage_contours, stage_landmarks, transform_records, _ = build_stage_geometry(
            dynamic_by_speaker,
            loaded_candidate,
            reference_speaker=REFERENCE_SPEAKER,
            speakers=SPEAKERS,
            lower_shape_controls=False,
        )
        affine_path = diagnostics_dir / "all_affine_contours_overlay.png"
        tps_path = diagnostics_dir / "all_tps_contours_overlay.png"
        m1_l6_path = diagnostics_dir / "tps_m1_l6_match_sheet.png"
        for stage, path in (("affine", affine_path), ("affine_tps", tps_path)):
            render_all_stage_contour_overlay(
                path,
                stage=stage,
                stage_contours=stage_contours,
                loaded_speakers=loaded_candidate,
                reference_speaker=REFERENCE_SPEAKER,
                speakers=SPEAKERS,
            )
        render_tps_m1_l6_match_sheet(
            m1_l6_path,
            stage_contours=stage_contours,
            stage_landmarks=stage_landmarks,
            reference_speaker=REFERENCE_SPEAKER,
        )

        nonpositive = [
            speaker
            for speaker, record in transform_records.items()
            if float(record["jacobian_nonpositive_fraction"]) > 0.0
        ]
        overlay_records = {
            "affine": image_record(affine_path),
            "affine_tps": image_record(tps_path),
            "tps_m1_l6_match": image_record(m1_l6_path),
        }
        current_hashes_after = {path.name: sha256_file(path) for path in current_files}
        if current_hashes_before != current_hashes_after:
            raise RuntimeError("Current VTLN files changed during candidate construction")

        summary = {
            "schema_version": "s5-pourri2-geometry-reference-candidate-v1",
            "version": VERSION,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "status": "PASS" if not nonpositive else "PASS_WITH_TPS_FOLDING_WARNING",
            "scope": "validated candidate; canonical when promoted into VTLN/data",
            "reference_speaker": REFERENCE_SPEAKER,
            "speaker_order": list(SPEAKERS),
            "phone_selection": "S5 pourri repetition 2, sole /u/, midpoint frame",
            "dynamic_roi_topology": "open POLYLINE",
            "cervical_roi_topology": "closed FREEHAND",
            "cervical_policy": "copy current fixed speaker-specific C1-C6 without modification",
            "lower_incisor_policy": (
                "exact ImageJ-reviewed pourri #2 prototype for P1/P3-P10; "
                "observed current inference for P2 because no reviewed prototype exists"
            ),
            "tps_lower_incisor_shape_controls": {
                "enabled": False,
                "labels": [],
                "definition": "disabled; canonical TPS uses only M1 and L6 for the lower incisor",
                "scope": "canonical candidate validation and release transform",
            },
            "p2_missing_policy": "vocal-folds absent and not imputed; empty only in visualization payload",
            "source": {
                "selection_csv": str(Path(args.selection_csv).resolve()),
                "selection_csv_sha256": sha256_file(args.selection_csv),
                "inference_root": str(Path(args.inference_root).resolve()),
                "video_root": str(Path(args.video_root).resolve()),
                "review_root": str(Path(args.review_root).resolve()),
                "current_vtln_dir": str(current_vtln_dir),
            },
            "cases": case_records,
            "transforms_to_p10": transform_records,
            "tps_nonpositive_jacobian_speakers": nonpositive,
            "overlays": overlay_records,
            "current_top_level_file_hashes_unchanged": True,
        }
        (data_dir / "build_summary.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        (stage_root / "candidate_summary.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

        if output_root.exists():
            shutil.rmtree(output_root)
        try:
            os.replace(stage_root, output_root)
        except PermissionError:
            # Windows may refuse to rename a tempfile-created directory even
            # after every image/ZIP handle is closed. Copying the validated
            # tree preserves the same candidate bytes.
            shutil.copytree(stage_root, output_root)
            shutil.rmtree(stage_root, ignore_errors=True)
        stage_root = output_root

        overlay_output_dir = Path(args.overlay_output_dir).resolve()
        overlay_output_dir.mkdir(parents=True, exist_ok=True)
        external_paths = {
            "affine": overlay_output_dir
            / "08_v018_candidate_all_affine_contours_overlay.png",
            "affine_tps": overlay_output_dir
            / "09_v018_candidate_all_tps_contours_overlay.png",
        }
        shutil.copy2(
            output_root / "diagnostics" / "all_affine_contours_overlay.png",
            external_paths["affine"],
        )
        shutil.copy2(
            output_root / "diagnostics" / "all_tps_contours_overlay.png",
            external_paths["affine_tps"],
        )
        summary["external_overlay_paths"] = {
            stage: str(path) for stage, path in external_paths.items()
        }
        (output_root / "candidate_summary.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        return summary
    except Exception:
        if stage_root.exists() and stage_root != output_root:
            shutil.rmtree(stage_root, ignore_errors=True)
        raise


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    summary = build_candidate(args)
    print(f"[status] {summary['status']}")
    print(f"[candidate] {Path(args.output_root).resolve()}")
    for stage, path in summary["external_overlay_paths"].items():
        print(f"[{stage}] {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
