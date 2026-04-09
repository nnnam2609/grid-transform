from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from pathlib import Path

from grid_transform.config import DEFAULT_OUTPUT_DIR, PROJECT_DIR


CURATED_VTLN_DIR = PROJECT_DIR / "VTLN" / "data"
DEFAULT_MANIFEST_CSV = CURATED_VTLN_DIR / "selection_manifest.csv"
DEFAULT_BATCH_OUTPUT_ROOT = DEFAULT_OUTPUT_DIR / "source_annotation_edits" / "curated_u_batch"
ARTSPEECH_ROOT_DEFAULT = PROJECT_DIR.parent / "Data" / "Artspeech_database"
TARGET_CASE_DEFAULT = "2008-003^01-1791/test"
TARGET_FRAME_DEFAULT = 143020

OUTPUT_BASENAME_RE = re.compile(
    r"^(?P<raw_subject>\d+)_(?P<speaker>P\d+)_(?P<session>S\d+)_F(?P<frame>\d+)$",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class BatchCase:
    output_basename: str
    speaker: str
    raw_subject: str
    session: str
    frame_index_1based: int
    annotation_status: str
    annotation_source_path: Path | None
    reference_bundle_dir: Path | None
    reference_bundle_name: str | None


def parse_manifest_row(row: dict[str, str], _vtln_dir: Path | None = None) -> BatchCase:
    output_basename = row["output_basename"].strip()
    match = OUTPUT_BASENAME_RE.match(output_basename)
    if match is None:
        raise ValueError(f"Could not parse output_basename={output_basename!r}")
    annotation_source_raw = row.get("annotation_source", "").strip()
    annotation_source_path = None
    if annotation_source_raw.lower().endswith(".zip"):
        annotation_source_path = Path(annotation_source_raw)
    reference_bundle_dir_raw = row.get("reference_bundle_dir", "").strip()
    reference_bundle_dir = Path(reference_bundle_dir_raw) if reference_bundle_dir_raw else None
    reference_bundle_name = row.get("reference_bundle_name", "").strip() or None
    return BatchCase(
        output_basename=output_basename,
        speaker=match.group("speaker").upper(),
        raw_subject=match.group("raw_subject"),
        session=match.group("session").upper(),
        frame_index_1based=int(match.group("frame")),
        annotation_status=row.get("annotation_status", "").strip(),
        annotation_source_path=annotation_source_path,
        reference_bundle_dir=reference_bundle_dir,
        reference_bundle_name=reference_bundle_name,
    )


def load_cases(manifest_csv: Path, vtln_dir: Path) -> list[BatchCase]:
    with manifest_csv.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return [parse_manifest_row(row, vtln_dir) for row in reader]


def build_case_output_dir(output_root: Path, case: BatchCase) -> Path:
    return output_root / case.output_basename


def resolve_reference_bundle(case: BatchCase) -> tuple[str, Path, Path, Path]:
    if case.reference_bundle_dir is not None and case.reference_bundle_name is not None:
        reference_speaker = case.reference_bundle_name
        bundle_dir = case.reference_bundle_dir
        zip_path = bundle_dir / f"{reference_speaker}.zip"
        if not zip_path.is_file():
            raise FileNotFoundError(f"Reference zip not found for override bundle: {zip_path}")
        for ext in (".png", ".tif", ".tiff"):
            image_path = bundle_dir / f"{reference_speaker}{ext}"
            if image_path.is_file():
                return reference_speaker, bundle_dir, image_path, zip_path
        raise FileNotFoundError(
            f"Reference image for override bundle {reference_speaker} was not found in {bundle_dir}."
        )

    if case.annotation_source_path is None:
        raise FileNotFoundError(f"No annotation zip is recorded in the manifest for {case.output_basename}")

    zip_path = case.annotation_source_path
    if not zip_path.is_file():
        raise FileNotFoundError(f"Annotation zip not found: {zip_path}")

    reference_speaker = zip_path.stem
    search_dirs = [zip_path.parent]
    if zip_path.parent.parent.exists() and zip_path.parent.parent not in search_dirs:
        search_dirs.append(zip_path.parent.parent)

    for directory in search_dirs:
        for ext in (".png", ".tif", ".tiff"):
            image_path = directory / f"{reference_speaker}{ext}"
            if image_path.is_file():
                return reference_speaker, zip_path.parent, image_path, zip_path

    raise FileNotFoundError(
        f"Reference image for {reference_speaker} was not found next to {zip_path} or in its parent directory."
    )


def resolve_speaker_root(artspeech_root: Path, speaker: str) -> Path:
    direct = artspeech_root / speaker
    nested = artspeech_root / speaker / speaker
    if (direct / "DCM_2D").exists() and (direct / "OTHER").exists():
        return direct
    if (nested / "DCM_2D").exists() and (nested / "OTHER").exists():
        return nested
    raise FileNotFoundError(
        f"Could not resolve ArtSpeech root for {speaker} under {artspeech_root}. "
        f"Tried {direct} and {nested}."
    )
