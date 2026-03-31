from __future__ import annotations

import argparse
import csv
import json
import shutil
from dataclasses import dataclass
from pathlib import Path

import imageio.v2 as imageio

from grid_transform.artspeech_video import load_session_data, normalize_frame
from grid_transform.config import DEFAULT_VTLN_DIR, PROJECT_DIR
from grid_transform.io import candidate_vtln_dirs


ARTSPEECH_ROOT_DEFAULT = PROJECT_DIR.parent / "Data" / "Artspeech_database"
OUTPUT_DIR_DEFAULT = PROJECT_DIR / "VTLN" / "u_curated_selection_20260330"
REFERENCE_IMAGE_NAME = "1640_s10_0829"


@dataclass(frozen=True)
class Selection:
    speaker: str
    raw_subject: str
    session: str
    frame_index_1based: int
    annotation_source_basename: str | None
    use_reference_image: bool = False

    @property
    def output_basename(self) -> str:
        return f"{self.raw_subject}_{self.speaker}_{self.session}_F{self.frame_index_1based:04d}"


SELECTIONS: tuple[Selection, ...] = (
    Selection("P1", "1612", "S16", 952, "1612_s10_0654"),
    Selection("P2", "1617", "S9", 1478, "1617_s10_0822"),
    Selection("P3", "1618", "S14", 1556, "1618_s10_0757"),
    Selection("P4", "1628", "S4", 196, "1628_s10_0943"),
    Selection("P5", "1635", "S6", 324, "1635_s10_0898"),
    Selection("P6", "1638", "S8", 138, "1638_s10_1167"),
    Selection("P7", "1640", "S2", 829, "1640_s10_0829", use_reference_image=True),
    Selection("P8", "1653", "S2", 159, "1653_s10_1729"),
    Selection("P9", "1659", "S5", 196, "1659_s10_0654"),
    Selection("P10", "1662", "S14", 110, None),
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create a curated VTLN folder from selected ArtSpeech /u/ frames and copy the "
            "currently available old VTLN annotations when they exist."
        )
    )
    parser.add_argument(
        "--artspeech-root",
        type=Path,
        default=ARTSPEECH_ROOT_DEFAULT,
        help="Root of Artspeech_database.",
    )
    parser.add_argument(
        "--vtln-dir",
        type=Path,
        default=DEFAULT_VTLN_DIR,
        help="Primary VTLN directory. The parent and Data/VTLN are also searched.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR_DEFAULT,
        help="Destination folder inside VTLN.",
    )
    return parser.parse_args(argv)


def resolve_speaker_root(artspeech_root: Path, speaker: str) -> Path:
    direct = artspeech_root / speaker
    nested = artspeech_root / speaker / speaker
    if (direct / "OTHER").exists() and (direct / "DCM_2D").exists():
        return direct
    if (nested / "OTHER").exists() and (nested / "DCM_2D").exists():
        return nested
    raise FileNotFoundError(f"Could not resolve ArtSpeech root for {speaker} under {artspeech_root}")


def vtln_search_roots(vtln_dir: Path) -> list[Path]:
    roots = candidate_vtln_dirs(vtln_dir)
    data_vtln = PROJECT_DIR.parent / "Data" / "VTLN"
    if data_vtln.exists():
        roots.append(data_vtln)
    unique: list[Path] = []
    seen: set[Path] = set()
    for root in roots:
        if root not in seen:
            unique.append(root)
            seen.add(root)
    return unique


def find_existing_file(basename: str, extensions: tuple[str, ...], roots: list[Path]) -> Path | None:
    for root in roots:
        for ext in extensions:
            candidate = root / f"{basename}{ext}"
            if candidate.is_file():
                return candidate
    return None


def export_selected_frame(selection: Selection, artspeech_root: Path, output_path: Path) -> Path:
    speaker_root = resolve_speaker_root(artspeech_root, selection.speaker)
    session_data = load_session_data(speaker_root, selection.speaker, selection.session)
    frame_index_0based = selection.frame_index_1based - 1
    if frame_index_0based < 0 or frame_index_0based >= session_data.images.shape[0]:
        raise IndexError(
            f"Frame {selection.frame_index_1based} is out of bounds for {selection.speaker}/{selection.session}"
        )
    frame = normalize_frame(
        session_data.images[frame_index_0based],
        session_data.frame_min,
        session_data.frame_max,
    )
    imageio.imwrite(output_path, frame)
    return speaker_root


def copy_reference_image(reference_path: Path, output_path: Path) -> None:
    shutil.copy2(reference_path, output_path)


def copy_annotation_zip(source_path: Path, output_path: Path) -> None:
    shutil.copy2(source_path, output_path)


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
        "# Curated VTLN /u/ Selection",
        "",
        "This folder contains the frames manually chosen from ArtSpeech for each encoded speaker.",
        "When an old VTLN annotation zip existed for the corresponding raw speaker, it was copied and renamed to match the new image basename.",
        "",
        "| Output basename | Speaker | Raw subject | Selected source | Image source | Old annotation source | Annotation status |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        lines.append(
            f"| {row['output_basename']} | {row['speaker']} | {row['raw_subject']} | "
            f"{row['selected_source']} | {row['image_source']} | {row['annotation_source']} | {row['annotation_status']} |"
        )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    roots = vtln_search_roots(args.vtln_dir)
    reference_image_path = find_existing_file(REFERENCE_IMAGE_NAME, (".png", ".tif", ".tiff"), roots)
    if reference_image_path is None:
        raise FileNotFoundError(f"Reference image {REFERENCE_IMAGE_NAME} was not found in {roots}")

    manifest_rows: list[dict[str, str]] = []

    for selection in SELECTIONS:
        output_basename = selection.output_basename
        output_image_path = output_dir / f"{output_basename}.png"
        output_zip_path = output_dir / f"{output_basename}.zip"

        if selection.use_reference_image:
            copy_reference_image(reference_image_path, output_image_path)
            image_source = str(reference_image_path)
        else:
            speaker_root = export_selected_frame(selection, args.artspeech_root, output_image_path)
            image_source = str(speaker_root / "DCM_2D" / selection.session)

        annotation_source = ""
        annotation_status = "missing"
        if selection.annotation_source_basename:
            source_zip = find_existing_file(selection.annotation_source_basename, (".zip",), roots)
            annotation_source = selection.annotation_source_basename
            if source_zip is not None:
                copy_annotation_zip(source_zip, output_zip_path)
                annotation_status = "copied"
                annotation_source = str(source_zip)
            else:
                annotation_status = "not_found"
        else:
            annotation_status = "not_available"

        manifest_rows.append(
            {
                "output_basename": output_basename,
                "speaker": selection.speaker,
                "raw_subject": selection.raw_subject,
                "selected_source": f"{selection.speaker}/{selection.session}/F{selection.frame_index_1based:04d}",
                "image_source": image_source,
                "annotation_source": annotation_source,
                "annotation_status": annotation_status,
            }
        )

    write_manifest(manifest_rows, output_dir / "selection_manifest.csv")
    write_readme(manifest_rows, output_dir / "README.md")

    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "manifest_csv": str(output_dir / "selection_manifest.csv"),
                "readme": str(output_dir / "README.md"),
                "items": len(manifest_rows),
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
