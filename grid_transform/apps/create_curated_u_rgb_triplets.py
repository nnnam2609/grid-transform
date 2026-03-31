from __future__ import annotations

import argparse
import csv
import json
import shutil
from dataclasses import dataclass
from pathlib import Path

import cv2
import imageio.v2 as imageio
import numpy as np

from grid_transform.artspeech_video import load_session_data, normalize_frame
from grid_transform.config import PROJECT_DIR


ARTSPEECH_ROOT_DEFAULT = PROJECT_DIR.parent / "Data" / "Artspeech_database"
INPUT_DIR_DEFAULT = PROJECT_DIR / "VTLN" / "u_curated_selection_20260330"
OUTPUT_DIR_DEFAULT = PROJECT_DIR / "VTLN" / "u_curated_selection_20260330_rgb480_triplets"
MANIFEST_CSV_DEFAULT = INPUT_DIR_DEFAULT / "selection_manifest.csv"

INTERPOLATION_MAP = {
    "nearest": cv2.INTER_NEAREST,
    "linear": cv2.INTER_LINEAR,
    "cubic": cv2.INTER_CUBIC,
    "area": cv2.INTER_AREA,
    "lanczos": cv2.INTER_LANCZOS4,
}


@dataclass(frozen=True)
class CuratedRow:
    output_basename: str
    speaker: str
    raw_subject: str
    session: str
    frame_index_1based: int
    annotation_status: str
    selected_source: str


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create 480x480 RGB triplets from the curated /u/ ArtSpeech frame selection. "
            "The output PNG uses channel order R=t-1, G=t, B=t+1."
        )
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=INPUT_DIR_DEFAULT,
        help="Existing curated folder containing selection_manifest.csv and optional annotation zips.",
    )
    parser.add_argument(
        "--manifest-csv",
        type=Path,
        default=MANIFEST_CSV_DEFAULT,
        help="Manifest CSV created for the curated selection.",
    )
    parser.add_argument(
        "--artspeech-root",
        type=Path,
        default=ARTSPEECH_ROOT_DEFAULT,
        help="Root of Artspeech_database.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR_DEFAULT,
        help="Destination folder for the resized RGB triplets.",
    )
    parser.add_argument(
        "--size",
        type=int,
        default=480,
        help="Square output size. Default: 480.",
    )
    parser.add_argument(
        "--interpolation",
        choices=tuple(INTERPOLATION_MAP),
        default="cubic",
        help="Resize interpolation used by cv2.",
    )
    parser.add_argument(
        "--skip-annotation-copy",
        action="store_true",
        help="Do not copy existing annotation zip files into the output folder.",
    )
    return parser.parse_args(argv)


def resolve_speaker_root(artspeech_root: Path, speaker: str) -> Path:
    direct = artspeech_root / speaker
    nested = artspeech_root / speaker / speaker
    if (direct / "DCM_2D").exists() and (direct / "OTHER").exists():
        return direct
    if (nested / "DCM_2D").exists() and (nested / "OTHER").exists():
        return nested
    raise FileNotFoundError(f"Could not resolve ArtSpeech root for {speaker} under {artspeech_root}")


def parse_manifest_row(row: dict[str, str]) -> CuratedRow:
    output_basename = row["output_basename"].strip()
    parts = output_basename.split("_")
    if len(parts) != 4 or not parts[3].startswith("F"):
        raise ValueError(f"Unexpected output_basename format: {output_basename}")
    return CuratedRow(
        output_basename=output_basename,
        speaker=row["speaker"].strip(),
        raw_subject=row["raw_subject"].strip(),
        session=parts[2].strip(),
        frame_index_1based=int(parts[3][1:]),
        annotation_status=row.get("annotation_status", "").strip(),
        selected_source=row.get("selected_source", "").strip(),
    )


def load_manifest(manifest_csv: Path) -> list[CuratedRow]:
    with manifest_csv.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return [parse_manifest_row(row) for row in reader]


def resize_frame(frame: np.ndarray, size: int, interpolation: int) -> np.ndarray:
    return cv2.resize(frame, (size, size), interpolation=interpolation)


def build_rgb_triplet(session_data, frame_index_1based: int, size: int, interpolation: int) -> tuple[np.ndarray, dict[str, int]]:
    frame_count = int(session_data.images.shape[0])
    center_index0 = frame_index_1based - 1
    if center_index0 < 0 or center_index0 >= frame_count:
        raise IndexError(f"Frame {frame_index_1based} is out of range 1..{frame_count}")

    prev_index0 = max(0, center_index0 - 1)
    next_index0 = min(frame_count - 1, center_index0 + 1)

    prev_frame = normalize_frame(session_data.images[prev_index0], session_data.frame_min, session_data.frame_max)
    center_frame = normalize_frame(session_data.images[center_index0], session_data.frame_min, session_data.frame_max)
    next_frame = normalize_frame(session_data.images[next_index0], session_data.frame_min, session_data.frame_max)

    resized_prev = resize_frame(prev_frame, size, interpolation)
    resized_center = resize_frame(center_frame, size, interpolation)
    resized_next = resize_frame(next_frame, size, interpolation)

    rgb = np.stack([resized_prev, resized_center, resized_next], axis=-1).astype(np.uint8)
    return rgb, {
        "prev_frame_1based": prev_index0 + 1,
        "center_frame_1based": center_index0 + 1,
        "next_frame_1based": next_index0 + 1,
    }


def copy_annotation_if_available(input_dir: Path, output_dir: Path, basename: str) -> str:
    source_zip = input_dir / f"{basename}.zip"
    if not source_zip.is_file():
        return "missing"
    shutil.copy2(source_zip, output_dir / source_zip.name)
    return "copied"


def write_manifest(rows: list[dict[str, str]], output_path: Path) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_readme(rows: list[dict[str, str]], output_path: Path) -> None:
    lines = [
        "# Curated /u/ RGB Triplets",
        "",
        "Each PNG is resized to 480x480 and saved as RGB with channel order:",
        "- R = t-1",
        "- G = t",
        "- B = t+1",
        "",
        "| Output basename | Speaker | Session | Source frame | R channel | G channel | B channel | Annotation zip |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        lines.append(
            f"| {row['output_basename']} | {row['speaker']} | {row['session']} | "
            f"{row['center_frame_1based']} | {row['prev_frame_1based']} | {row['center_frame_1based']} | "
            f"{row['next_frame_1based']} | {row['annotation_zip_status']} |"
        )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows = load_manifest(args.manifest_csv)
    interpolation = INTERPOLATION_MAP[args.interpolation]
    session_cache: dict[tuple[str, str], object] = {}
    output_rows: list[dict[str, str]] = []

    for row in rows:
        cache_key = (row.speaker, row.session)
        if cache_key not in session_cache:
            speaker_root = resolve_speaker_root(args.artspeech_root, row.speaker)
            session_cache[cache_key] = load_session_data(speaker_root, row.speaker, row.session)
        session_data = session_cache[cache_key]

        rgb, indices = build_rgb_triplet(
            session_data=session_data,
            frame_index_1based=row.frame_index_1based,
            size=args.size,
            interpolation=interpolation,
        )

        output_png = args.output_dir / f"{row.output_basename}.png"
        imageio.imwrite(output_png, rgb)

        annotation_zip_status = "skipped"
        if not args.skip_annotation_copy:
            annotation_zip_status = copy_annotation_if_available(args.input_dir, args.output_dir, row.output_basename)

        output_rows.append(
            {
                "output_basename": row.output_basename,
                "speaker": row.speaker,
                "raw_subject": row.raw_subject,
                "session": row.session,
                "selected_source": row.selected_source,
                "prev_frame_1based": str(indices["prev_frame_1based"]),
                "center_frame_1based": str(indices["center_frame_1based"]),
                "next_frame_1based": str(indices["next_frame_1based"]),
                "channel_order": "R=t-1,G=t,B=t+1",
                "output_size": f"{args.size}x{args.size}",
                "annotation_status": row.annotation_status,
                "annotation_zip_status": annotation_zip_status,
            }
        )

    shutil.copy2(args.manifest_csv, args.output_dir / "selection_manifest_original.csv")
    write_manifest(output_rows, args.output_dir / "selection_manifest_rgb_triplets.csv")
    write_readme(output_rows, args.output_dir / "README.md")

    print(
        json.dumps(
            {
                "input_dir": str(args.input_dir),
                "output_dir": str(args.output_dir),
                "items": len(output_rows),
                "channel_order": "R=t-1,G=t,B=t+1",
                "output_size": args.size,
                "interpolation": args.interpolation,
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
