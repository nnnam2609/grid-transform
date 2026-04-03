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

from grid_transform.artspeech_video import load_session_data, normalize_frame
from grid_transform.config import DEFAULT_OUTPUT_DIR, PROJECT_DIR
from grid_transform.source_annotation import load_source_annotation_json


CURATED_VTLN_DIR_DEFAULT = PROJECT_DIR / "VTLN" / "u_curated_selection_20260330"
MANIFEST_CSV_DEFAULT = CURATED_VTLN_DIR_DEFAULT / "selection_manifest.csv"
ARCHIVE_ROOT_DEFAULT = DEFAULT_OUTPUT_DIR / "vtln_annotation_archive"
BACKUP_ROOT_DEFAULT = PROJECT_DIR.parent / "backup"


@dataclass(frozen=True)
class CuratedRow:
    output_basename: str
    speaker: str
    session: str
    frame_index_1based: int


@dataclass(frozen=True)
class LatestAnnotation:
    json_path: Path
    metadata: dict[str, object]
    contours: dict[str, np.ndarray]
    mtime: float


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Sync the newest saved edited annotations into the canonical curated VTLN folder. "
            "Images are rebuilt from the matching ArtSpeech source frame, and ROI zip files are regenerated."
        )
    )
    parser.add_argument(
        "--curated-vtln-dir",
        type=Path,
        default=CURATED_VTLN_DIR_DEFAULT,
        help="Canonical curated VTLN folder to update in place.",
    )
    parser.add_argument(
        "--manifest-csv",
        type=Path,
        default=MANIFEST_CSV_DEFAULT,
        help="Selection manifest describing the curated basenames.",
    )
    parser.add_argument(
        "--archive-dir",
        type=Path,
        default=ARCHIVE_ROOT_DEFAULT / "latest_sync_backup",
        help="Where to copy the previous PNG/ZIP files before overwrite.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report planned updates without writing files.",
    )
    return parser.parse_args(argv)


def parse_manifest_row(row: dict[str, str]) -> CuratedRow:
    output_basename = row["output_basename"].strip()
    parts = output_basename.split("_")
    if len(parts) != 4 or not parts[3].startswith("F"):
        raise ValueError(f"Unexpected output_basename format: {output_basename}")
    return CuratedRow(
        output_basename=output_basename,
        speaker=row["speaker"].strip(),
        session=parts[2].strip(),
        frame_index_1based=int(parts[3][1:]),
    )


def load_manifest(path: Path) -> list[CuratedRow]:
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


def load_source_frame(annotation: LatestAnnotation) -> np.ndarray:
    metadata = annotation.metadata
    dataset_root = metadata.get("dataset_root")
    speaker = metadata.get("artspeech_speaker")
    session = metadata.get("session")
    source_frame = int(metadata.get("source_frame") or 0)
    if not dataset_root or not speaker or not session or source_frame <= 0:
        raise ValueError(
            f"Annotation {annotation.json_path} is missing dataset_root, artspeech_speaker, session, or source_frame."
        )

    session_data = load_session_data(Path(str(dataset_root)), str(speaker), str(session))
    frame_index0 = source_frame - 1
    if frame_index0 < 0 or frame_index0 >= session_data.images.shape[0]:
        raise ValueError(
            f"Annotation {annotation.json_path} frame {source_frame} is outside session range 1..{session_data.images.shape[0]}."
        )
    frame = normalize_frame(
        session_data.images[frame_index0],
        session_data.frame_min,
        session_data.frame_max,
    )
    return np.asarray(frame, dtype=np.uint8)


def archive_file_if_present(path: Path, archive_dir: Path, *, dry_run: bool) -> Path | None:
    if not path.is_file():
        return None
    archive_dir.mkdir(parents=True, exist_ok=True)
    archive_path = archive_dir / path.name
    if dry_run:
        return archive_path
    shutil.copy2(path, archive_path)
    return archive_path


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


def write_summary(
    curated_vtln_dir: Path,
    *,
    rows: list[dict[str, object]],
    dry_run: bool,
) -> Path | None:
    if dry_run:
        return None
    summary_path = curated_vtln_dir / "latest_annotation_sync_summary.json"
    payload = {
        "curated_vtln_dir": str(curated_vtln_dir),
        "updated_count": sum(1 for row in rows if row["status"] == "updated"),
        "missing_count": sum(1 for row in rows if row["status"] == "missing_latest_annotation"),
        "rows": rows,
    }
    summary_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary_path


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    args.curated_vtln_dir = resolve_existing_dir(args.curated_vtln_dir, leaf_name=CURATED_VTLN_DIR_DEFAULT.name)
    if not args.manifest_csv.is_file():
        candidate_manifest = args.curated_vtln_dir / "selection_manifest.csv"
        if candidate_manifest.is_file():
            args.manifest_csv = candidate_manifest
    curated_rows = load_manifest(args.manifest_csv)
    latest_map = build_latest_annotation_map()

    report_rows: list[dict[str, object]] = []
    updated = 0

    for row in curated_rows:
        key = (row.speaker, row.session, row.frame_index_1based)
        annotation = latest_map.get(key)
        png_path = args.curated_vtln_dir / f"{row.output_basename}.png"
        zip_path = args.curated_vtln_dir / f"{row.output_basename}.zip"
        archive_dir = args.archive_dir / row.output_basename

        if annotation is None:
            report_rows.append(
                {
                    "output_basename": row.output_basename,
                    "status": "missing_latest_annotation",
                    "png_path": str(png_path),
                    "zip_path": str(zip_path),
                }
            )
            continue

        source_frame = load_source_frame(annotation)
        archived_png = archive_file_if_present(png_path, archive_dir, dry_run=args.dry_run)
        archived_zip = archive_file_if_present(zip_path, archive_dir, dry_run=args.dry_run)

        if not args.dry_run:
            imageio.imwrite(png_path, source_frame)
        write_annotation_zip(zip_path, row.output_basename, annotation.contours, dry_run=args.dry_run)

        report_rows.append(
            {
                "output_basename": row.output_basename,
                "status": "updated",
                "annotation_json": str(annotation.json_path),
                "png_path": str(png_path),
                "zip_path": str(zip_path),
                "source_shape": list(source_frame.shape[:2]),
                "archived_png": None if archived_png is None else str(archived_png),
                "archived_zip": None if archived_zip is None else str(archived_zip),
            }
        )
        updated += 1

    summary_path = write_summary(args.curated_vtln_dir, rows=report_rows, dry_run=args.dry_run)

    print(
        json.dumps(
            {
                "curated_vtln_dir": str(args.curated_vtln_dir),
                "updated": updated,
                "missing_latest_annotation": sum(
                    1 for row in report_rows if row["status"] == "missing_latest_annotation"
                ),
                "dry_run": bool(args.dry_run),
                "summary_json": None if summary_path is None else str(summary_path),
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
