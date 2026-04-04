from __future__ import annotations

import argparse
import csv
import json
import shutil
import zipfile
from dataclasses import dataclass
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from roifile import ImagejRoi

from grid_transform.annotation_projection import build_resize_affine, transform_reference_contours
from grid_transform.config import DEFAULT_OUTPUT_DIR, PROJECT_DIR
from grid_transform.io import load_frame_vtln
from grid_transform.source_annotation import load_source_annotation_json


TRIPLET_DIR_DEFAULT = PROJECT_DIR / "VTLN" / "u_curated_selection_20260330_rgb480_triplets"
CURATED_DIR_DEFAULT = PROJECT_DIR / "VTLN" / "u_curated_selection_20260330"
OUTPUT_DIR_DEFAULT = PROJECT_DIR / "VTLN" / "data"
TARGET_CASE_DEFAULT = "2008-003^01-1791/test"
TARGET_DATA_ROOT_DEFAULT = PROJECT_DIR / "vocal-tract-seg" / "data_80"
SUMMARY_PATH_DEFAULT = OUTPUT_DIR_DEFAULT / "build_summary.json"
BACKUP_ROOT_DEFAULT = PROJECT_DIR.parent / "backup"


@dataclass(frozen=True)
class TripletRow:
    output_basename: str
    speaker: str
    raw_subject: str
    session: str
    selected_source: str
    prev_frame_1based: int
    center_frame_1based: int
    next_frame_1based: int
    channel_order: str
    output_size: str
    annotation_status: str


@dataclass(frozen=True)
class LatestAnnotation:
    json_path: Path
    metadata: dict[str, object]
    contours: dict[str, np.ndarray]
    mtime: float


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build the canonical VTLN/data bundle: 480x480 RGB triplet PNGs, scaled ROI zips, "
            "and the bundled nnUNet target case with groundtruth contours used by the current pipeline."
        )
    )
    parser.add_argument("--triplet-dir", type=Path, default=TRIPLET_DIR_DEFAULT, help="Source folder with 480x480 RGB triplet PNGs.")
    parser.add_argument("--curated-dir", type=Path, default=CURATED_DIR_DEFAULT, help="Curated source-shape PNG/ZIP folder used as contour fallback.")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR_DEFAULT, help="Destination canonical VTLN/data folder.")
    parser.add_argument(
        "--manifest-csv",
        type=Path,
        default=TRIPLET_DIR_DEFAULT / "selection_manifest_rgb_triplets.csv",
        help="Triplet manifest CSV used to drive the bundle build.",
    )
    parser.add_argument("--target-case", default=TARGET_CASE_DEFAULT, help="nnUNet case path to copy into VTLN/data.")
    parser.add_argument("--target-data-root", type=Path, default=TARGET_DATA_ROOT_DEFAULT, help="Source root that contains data_80/<case>.")
    parser.add_argument("--summary-path", type=Path, default=SUMMARY_PATH_DEFAULT, help="Where to write the build summary JSON.")
    parser.add_argument("--dry-run", action="store_true", help="Report actions without writing files.")
    return parser.parse_args(argv)


def parse_manifest_row(row: dict[str, str]) -> TripletRow:
    return TripletRow(
        output_basename=row["output_basename"].strip(),
        speaker=row["speaker"].strip(),
        raw_subject=row["raw_subject"].strip(),
        session=row["session"].strip(),
        selected_source=row["selected_source"].strip(),
        prev_frame_1based=int(row["prev_frame_1based"]),
        center_frame_1based=int(row["center_frame_1based"]),
        next_frame_1based=int(row["next_frame_1based"]),
        channel_order=row.get("channel_order", "R=t-1,G=t,B=t+1").strip(),
        output_size=row.get("output_size", "480x480").strip(),
        annotation_status=row.get("annotation_status", "").strip(),
    )


def load_manifest(path: Path) -> list[TripletRow]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return [parse_manifest_row(row) for row in reader]


def resolve_existing_dir(path: Path, *, leaf_name: str) -> Path:
    if path.is_dir():
        return path
    if BACKUP_ROOT_DEFAULT.is_dir():
        matches = [candidate for candidate in BACKUP_ROOT_DEFAULT.rglob(leaf_name) if candidate.is_dir()]
        if matches:
            return max(matches, key=lambda candidate: candidate.stat().st_mtime)
    return path


def discover_saved_annotation_paths() -> list[Path]:
    candidates: list[Path] = []
    search_specs = (
        (DEFAULT_OUTPUT_DIR / "annotation_to_grid_transform", "source_annotation.latest.json"),
        (DEFAULT_OUTPUT_DIR / "source_annotation_edits", "edited_annotation.json"),
    )
    for root, filename in search_specs:
        if root.is_dir():
            candidates.extend(root.rglob(filename))
    return sorted(set(candidates))


def build_latest_annotation_map() -> dict[tuple[str, str, int], LatestAnnotation]:
    latest: dict[tuple[str, str, int], LatestAnnotation] = {}
    for json_path in discover_saved_annotation_paths():
        payload = load_source_annotation_json(json_path)
        metadata = payload["metadata"]
        key = (
            str(metadata.get("artspeech_speaker") or ""),
            str(metadata.get("session") or ""),
            int(metadata.get("source_frame") or 0),
        )
        if not key[0] or not key[1] or key[2] <= 0:
            continue
        mtime = json_path.stat().st_mtime
        current = latest.get(key)
        if current is None or mtime > current.mtime:
            latest[key] = LatestAnnotation(
                json_path=json_path,
                metadata=metadata,
                contours=payload["contours"],
                mtime=mtime,
            )
    return latest


def write_annotation_zip(path: Path, basename: str, contours: dict[str, np.ndarray], *, dry_run: bool) -> None:
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


def contour_payload_for_row(
    row: TripletRow,
    *,
    latest_map: dict[tuple[str, str, int], LatestAnnotation],
    curated_dir: Path,
) -> tuple[dict[str, np.ndarray] | None, tuple[int, int] | None, str, str]:
    latest = latest_map.get((row.speaker, row.session, row.center_frame_1based))
    if latest is not None:
        source_shape_raw = latest.metadata.get("source_shape") or latest.metadata.get("reference_shape")
        if source_shape_raw is None:
            raise ValueError(f"Latest annotation {latest.json_path} is missing source/reference shape metadata.")
        source_shape = tuple(int(value) for value in source_shape_raw[:2])
        return latest.contours, source_shape, "scaled_latest_annotation", str(latest.json_path)

    fallback_zip = curated_dir / f"{row.output_basename}.zip"
    if fallback_zip.is_file():
        fallback_image, contours = load_frame_vtln(row.output_basename, curated_dir)
        source_shape = tuple(int(value) for value in np.asarray(fallback_image).shape[:2])
        return contours, source_shape, "scaled_curated_zip", str(fallback_zip)

    return None, None, "missing", ""


def scale_contours_to_triplet_space(
    contours: dict[str, np.ndarray],
    source_shape: tuple[int, int],
    target_shape: tuple[int, int],
) -> dict[str, np.ndarray]:
    affine = build_resize_affine(source_shape, target_shape)
    return transform_reference_contours(contours, affine)


def write_manifest(rows: list[dict[str, str]], path: Path, *, dry_run: bool) -> None:
    if dry_run or not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_readme(path: Path, *, dry_run: bool) -> None:
    if dry_run:
        return
    lines = [
        "# VTLN/data",
        "",
        "Canonical bundle for the current pipeline.",
        "",
        "- `*.png`: 480x480 RGB triplets with channel order `R=t-1, G=t, B=t+1`.",
        "- `*.zip`: ROI contours scaled into the same 480x480 coordinate space.",
        "- `nnunet_data_80/`: bundled target MRI image case and groundtruth contours used by the apps.",
        "",
        "All grayscale computations should use the center channel `G=t`.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def copytree_overwrite(src: Path, dst: Path, *, dry_run: bool) -> None:
    if dry_run:
        return
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    args.triplet_dir = resolve_existing_dir(args.triplet_dir, leaf_name=TRIPLET_DIR_DEFAULT.name)
    args.curated_dir = resolve_existing_dir(args.curated_dir, leaf_name=CURATED_DIR_DEFAULT.name)
    args.target_data_root = resolve_existing_dir(args.target_data_root, leaf_name=TARGET_DATA_ROOT_DEFAULT.name)
    if not args.manifest_csv.is_file():
        candidate_manifest = args.triplet_dir / "selection_manifest_rgb_triplets.csv"
        if candidate_manifest.is_file():
            args.manifest_csv = candidate_manifest
    rows = load_manifest(args.manifest_csv)
    latest_map = build_latest_annotation_map()
    report_rows: list[dict[str, object]] = []
    manifest_rows: list[dict[str, str]] = []

    if not args.dry_run:
        args.output_dir.mkdir(parents=True, exist_ok=True)

    for row in rows:
        png_src = args.triplet_dir / f"{row.output_basename}.png"
        png_dst = args.output_dir / png_src.name
        if not png_src.is_file():
            raise FileNotFoundError(f"Triplet PNG not found: {png_src}")

        triplet_image = imageio.imread(png_src)
        target_shape = tuple(int(value) for value in np.asarray(triplet_image).shape[:2])

        if not args.dry_run:
            shutil.copy2(png_src, png_dst)

        contours, source_shape, annotation_status, annotation_origin = contour_payload_for_row(
            row,
            latest_map=latest_map,
            curated_dir=args.curated_dir,
        )

        zip_dst = args.output_dir / f"{row.output_basename}.zip"
        scaled_count = 0
        if contours is not None and source_shape is not None:
            scaled_contours = scale_contours_to_triplet_space(contours, source_shape, target_shape)
            write_annotation_zip(zip_dst, row.output_basename, scaled_contours, dry_run=args.dry_run)
            scaled_count = len(scaled_contours)

        manifest_rows.append(
            {
                "output_basename": row.output_basename,
                "speaker": row.speaker,
                "raw_subject": row.raw_subject,
                "session": row.session,
                "selected_source": row.selected_source,
                "image_source": str(png_src),
                "annotation_source": str(zip_dst) if scaled_count > 0 else "",
                "annotation_status": annotation_status,
                "annotation_origin_path": annotation_origin,
                "reference_bundle_dir": str(args.output_dir),
                "reference_bundle_name": row.output_basename,
                "prev_frame_1based": str(row.prev_frame_1based),
                "center_frame_1based": str(row.center_frame_1based),
                "next_frame_1based": str(row.next_frame_1based),
                "channel_order": row.channel_order,
                "output_size": row.output_size,
                "annotation_space": f"{target_shape[1]}x{target_shape[0]}",
            }
        )
        report_rows.append(
            {
                "output_basename": row.output_basename,
                "png_path": str(png_dst),
                "zip_path": str(zip_dst) if scaled_count > 0 else "",
                "annotation_status": annotation_status,
                "annotation_origin_path": annotation_origin,
                "source_shape": None if source_shape is None else list(source_shape),
                "target_shape": list(target_shape),
                "scaled_contour_count": scaled_count,
            }
        )

    source_manifest_copy = args.output_dir / "selection_manifest_rgb_triplets.source.csv"
    if not args.dry_run:
        shutil.copy2(args.manifest_csv, source_manifest_copy)

    write_manifest(manifest_rows, args.output_dir / "selection_manifest.csv", dry_run=args.dry_run)
    write_readme(args.output_dir / "README.md", dry_run=args.dry_run)

    nnunet_data_src = args.target_data_root / args.target_case
    nnunet_data_dst = args.output_dir / "nnunet_data_80" / args.target_case
    if not nnunet_data_src.is_dir():
        raise FileNotFoundError(f"Target data case not found: {nnunet_data_src}")
    copytree_overwrite(nnunet_data_src, nnunet_data_dst, dry_run=args.dry_run)

    summary = {
        "output_dir": str(args.output_dir),
        "triplet_dir": str(args.triplet_dir),
        "curated_dir": str(args.curated_dir),
        "target_case": args.target_case,
        "speaker_rows": len(report_rows),
        "zip_rows": sum(1 for row in report_rows if row["zip_path"]),
        "missing_annotation_rows": sum(1 for row in report_rows if row["annotation_status"] == "missing"),
        "nnunet_data_dst": str(nnunet_data_dst),
        "nnunet_groundtruth_dir": str(nnunet_data_dst / "contours"),
        "rows": report_rows,
    }
    if not args.dry_run:
        args.summary_path.parent.mkdir(parents=True, exist_ok=True)
        args.summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
