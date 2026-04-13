from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from grid_transform.config import TONGUE_COLOR
from grid_transform.transform_helpers import resample_polyline


CURATED_SPEAKER_GENDER = {
    "P1": "male",
    "P2": "male",
    "P3": "male",
    "P4": "female",
    "P5": "male",
    "P6": "male",
    "P7": "female",
    "P8": "female",
    "P9": "female",
    "P10": "female",
}
NNUNET_AVERAGE_SPEAKER_ID = "nnUNet_A"
CURATED_PLUS_NNUNET_GENDER = {
    **CURATED_SPEAKER_GENDER,
    # The bundled average-speaker nnUNet case is treated as a female cohort item.
    NNUNET_AVERAGE_SPEAKER_ID: "female",
}
IMAGE_SUFFIXES = (".png", ".tif", ".tiff")
CURATED_BASENAME_RE = re.compile(
    r"^(?P<raw_subject>\d+)_(?P<speaker>P\d+)_(?P<session>S\d+)_F(?P<frame>\d+)$",
    re.IGNORECASE,
)
DEFAULT_N_VERT = 9
DEFAULT_N_POINTS = 250


@dataclass(frozen=True)
class CuratedSpeakerSpec:
    speaker_id: str
    basename: str
    raw_subject: str | None
    session: str | None
    frame: int | None
    gender: str
    image_path: Path | None
    zip_path: Path | None


@dataclass
class LoadedSpeaker:
    spec: CuratedSpeakerSpec
    image: object
    contours: dict[str, np.ndarray]
    grid: object


def speaker_id_sort_key(speaker_id: str) -> tuple[int, str]:
    match = re.fullmatch(r"P(\d+)", speaker_id, re.IGNORECASE)
    if match is None:
        return (10_000, speaker_id)
    return (int(match.group(1)), speaker_id)


def resolve_image_path(vtln_dir: Path, basename: str) -> Path | None:
    for ext in IMAGE_SUFFIXES:
        candidate = vtln_dir / f"{basename}{ext}"
        if candidate.is_file():
            return candidate
    return None


def parse_selected_source(selected_source: str) -> tuple[str | None, int | None]:
    parts = [part.strip() for part in selected_source.split("/") if part.strip()]
    if len(parts) != 3:
        return None, None
    session = parts[1].upper()
    frame_text = parts[2].upper()
    if not frame_text.startswith("F") or not frame_text[1:].isdigit():
        return session, None
    return session, int(frame_text[1:])


def parse_curated_basename(basename: str) -> tuple[str | None, str | None, str | None, int | None]:
    match = CURATED_BASENAME_RE.match(basename)
    if match is None:
        return None, None, None, None
    return (
        match.group("speaker").upper(),
        match.group("raw_subject"),
        match.group("session").upper(),
        int(match.group("frame")),
    )


def read_curated_specs_map(vtln_dir: Path) -> dict[str, CuratedSpeakerSpec]:
    manifest_path = vtln_dir / "selection_manifest.csv"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Curated manifest not found: {manifest_path}")

    specs: dict[str, CuratedSpeakerSpec] = {}
    with manifest_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            speaker_id = row["speaker"].strip().upper()
            if speaker_id not in CURATED_SPEAKER_GENDER:
                continue
            if speaker_id in specs:
                raise ValueError(f"Expected one curated row per speaker, found duplicate for {speaker_id}.")

            basename = row["output_basename"].strip()
            session, frame = parse_selected_source(row.get("selected_source", ""))
            specs[speaker_id] = CuratedSpeakerSpec(
                speaker_id=speaker_id,
                basename=basename,
                raw_subject=row.get("raw_subject", "").strip() or None,
                session=session,
                frame=frame,
                gender=CURATED_SPEAKER_GENDER[speaker_id],
                image_path=resolve_image_path(vtln_dir, basename),
                zip_path=vtln_dir / f"{basename}.zip",
            )
    return specs


def read_curated_specs_list(vtln_dir: Path) -> list[CuratedSpeakerSpec]:
    manifest_path = vtln_dir / "selection_manifest.csv"
    specs: list[CuratedSpeakerSpec] = []

    if manifest_path.is_file():
        with manifest_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                basename = row["output_basename"].strip()
                speaker_id = row["speaker"].strip().upper()
                if speaker_id not in CURATED_SPEAKER_GENDER:
                    continue
                session, frame = parse_selected_source(row.get("selected_source", ""))
                image_path = resolve_image_path(vtln_dir, basename)
                zip_path = vtln_dir / f"{basename}.zip"
                specs.append(
                    CuratedSpeakerSpec(
                        speaker_id=speaker_id,
                        basename=basename,
                        raw_subject=row.get("raw_subject", "").strip() or None,
                        session=session,
                        frame=frame,
                        gender=CURATED_SPEAKER_GENDER[speaker_id],
                        image_path=image_path,
                        zip_path=zip_path if zip_path.is_file() else None,
                    )
                )
        return sorted(specs, key=lambda item: item.basename)

    stems = sorted({path.stem for ext in IMAGE_SUFFIXES + (".zip",) for path in vtln_dir.glob(f"*{ext}")})
    for stem in stems:
        speaker_id, raw_subject, session, frame = parse_curated_basename(stem)
        if speaker_id is None or speaker_id not in CURATED_SPEAKER_GENDER:
            continue
        zip_path = vtln_dir / f"{stem}.zip"
        specs.append(
            CuratedSpeakerSpec(
                speaker_id=speaker_id,
                basename=stem,
                raw_subject=raw_subject,
                session=session,
                frame=frame,
                gender=CURATED_SPEAKER_GENDER[speaker_id],
                image_path=resolve_image_path(vtln_dir, stem),
                zip_path=zip_path if zip_path.is_file() else None,
            )
        )
    return specs


def load_curated_speakers(vtln_dir: Path, requested_speakers: list[str]) -> dict[str, LoadedSpeaker]:
    from grid_transform.io import load_frame_vtln
    from grid_transform.vt import build_grid

    specs = read_curated_specs_map(vtln_dir)
    missing = [speaker for speaker in requested_speakers if speaker not in specs]
    if missing:
        raise ValueError(f"Missing curated manifest rows for: {', '.join(missing)}")

    loaded: dict[str, LoadedSpeaker] = {}
    for speaker_id in requested_speakers:
        spec = specs[speaker_id]
        if spec.image_path is None or not spec.image_path.is_file():
            raise FileNotFoundError(f"Curated image not found for {speaker_id}: {spec.image_path}")
        if spec.zip_path is None or not spec.zip_path.is_file():
            raise FileNotFoundError(f"Curated ROI zip not found for {speaker_id}: {spec.zip_path}")

        image, contours = load_frame_vtln(spec.basename, vtln_dir)
        grid = build_grid(
            image,
            contours,
            n_vert=DEFAULT_N_VERT,
            n_points=DEFAULT_N_POINTS,
            frame_number=0,
        )
        loaded[speaker_id] = LoadedSpeaker(spec=spec, image=image, contours=contours, grid=grid)
    return loaded


def load_available_speakers(specs: list[CuratedSpeakerSpec], vtln_dir: Path) -> tuple[dict[str, LoadedSpeaker], list[str]]:
    from grid_transform.io import load_frame_vtln
    from grid_transform.vt import build_grid

    loaded: dict[str, LoadedSpeaker] = {}
    skipped: list[str] = []

    for spec in specs:
        if spec.image_path is None:
            skipped.append(f"{spec.basename}: missing image")
            continue
        if spec.zip_path is None:
            skipped.append(f"{spec.basename}: missing ROI zip")
            continue

        try:
            image, contours = load_frame_vtln(spec.basename, vtln_dir)
            grid = build_grid(
                image,
                contours,
                n_vert=DEFAULT_N_VERT,
                n_points=DEFAULT_N_POINTS,
                frame_number=0,
            )
        except Exception as exc:
            skipped.append(f"{spec.basename}: {type(exc).__name__}: {exc}")
            continue

        loaded[spec.basename] = LoadedSpeaker(spec=spec, image=image, contours=contours, grid=grid)

    return loaded, skipped


def compute_global_common_labels(loaded_speakers: dict[str, LoadedSpeaker]) -> list[str]:
    if not loaded_speakers:
        raise ValueError("No speakers were loaded successfully.")
    label_sets = [set(speaker.contours.keys()) for speaker in loaded_speakers.values()]
    labels = sorted(set.intersection(*label_sets))
    if not labels:
        raise ValueError("No common contour labels were shared across the available speakers.")
    return labels


def stack_resampled_contours(contours: dict[str, np.ndarray], labels: list[str], nc_template: int) -> np.ndarray:
    return np.vstack([resample_polyline(contours[label], nc_template) for label in labels])


def flatten_resampled_contours(contours: dict[str, np.ndarray], labels: list[str], nc_template: int) -> np.ndarray:
    return stack_resampled_contours(contours, labels, nc_template).reshape(-1)


def choose_label_colors(labels):
    from grid_transform.transform_helpers import choose_tab20_label_colors

    return choose_tab20_label_colors(labels, tongue_color=TONGUE_COLOR)
