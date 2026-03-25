from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import imageio.v2 as imageio

from grid_transform.artspeech_video import (
    frame_index_for_time,
    load_session_data,
    normalize_frame,
    parse_textgrid,
)
from grid_transform.config import DEFAULT_OUTPUT_DIR, PROJECT_DIR


DEFAULT_VOWELS = ("a", "e", "i", "o", "u")


@dataclass
class VowelSelection:
    speaker: str
    vowel: str
    session: str
    frame_index_0based: int
    frame_index_1based: int
    time_s: float
    interval_start_s: float
    interval_end_s: float
    dataset_root: str
    textgrid_path: str
    image_path: str


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Extract one representative ArtSpeech frame per speaker for the five core vowels "
            "a/e/i/o/u and organize them into one folder per vowel."
        )
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=PROJECT_DIR.parent / "Data" / "Artspeech_database",
        help="Root directory containing ArtSpeech speaker folders.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR / "vowel_samples_aeiou",
        help="Output directory containing one subfolder per vowel.",
    )
    parser.add_argument(
        "--vowels",
        default=",".join(DEFAULT_VOWELS),
        help="Comma-separated vowel labels to extract. Defaults to a,e,i,o,u.",
    )
    return parser.parse_args(argv)


def parse_vowels(text: str) -> list[str]:
    vowels = [value.strip() for value in text.split(",") if value.strip()]
    if not vowels:
        raise ValueError("No vowels were provided.")
    return vowels


def discover_speaker_dirs(dataset_root: Path) -> list[Path]:
    return sorted(path for path in dataset_root.iterdir() if path.is_dir())


def first_interval_for_vowel(textgrid_path: Path, vowel: str) -> tuple[float, float, float] | None:
    tiers = parse_textgrid(textgrid_path)
    if len(tiers) < 2:
        return None
    for interval in tiers[1].intervals:
        label = interval.text.strip()
        if label == vowel:
            time_s = 0.5 * (interval.start + interval.end)
            return interval.start, interval.end, time_s
    return None


def select_textgrid_for_vowel(speaker_dir: Path, vowel: str) -> tuple[Path, str, float, float, float] | None:
    textgrids = sorted(speaker_dir.rglob("TEXT_ALIGNMENT_*.textgrid"))
    for textgrid_path in textgrids:
        match = first_interval_for_vowel(textgrid_path, vowel)
        if match is None:
            continue
        interval_start_s, interval_end_s, time_s = match
        session = textgrid_path.stem.split("_")[-1]
        return textgrid_path, session, interval_start_s, interval_end_s, time_s
    return None


def dataset_root_from_textgrid(textgrid_path: Path) -> Path:
    return textgrid_path.parents[2]


def extract_frame_image(session_data, frame_index_0based: int):
    return normalize_frame(
        session_data.images[frame_index_0based],
        session_data.frame_min,
        session_data.frame_max,
    )


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    vowels = parse_vowels(args.vowels)
    dataset_root = args.dataset_root
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    for vowel in vowels:
        (output_dir / vowel).mkdir(parents=True, exist_ok=True)

    speaker_dirs = discover_speaker_dirs(dataset_root)
    if not speaker_dirs:
        raise FileNotFoundError(f"No speaker directories found in {dataset_root}")

    session_cache: dict[tuple[str, str, str], object] = {}
    selections: list[VowelSelection] = []
    missing: dict[str, list[str]] = {vowel: [] for vowel in vowels}

    for vowel in vowels:
        print(f"[extract] vowel {vowel}")
        for speaker_dir in speaker_dirs:
            speaker = speaker_dir.name
            chosen = select_textgrid_for_vowel(speaker_dir, vowel)
            if chosen is None:
                missing[vowel].append(speaker)
                print(f"  - {speaker}: missing")
                continue

            textgrid_path, session, interval_start_s, interval_end_s, time_s = chosen
            resolved_dataset_root = dataset_root_from_textgrid(textgrid_path)
            cache_key = (speaker, session, str(resolved_dataset_root))
            if cache_key not in session_cache:
                session_cache[cache_key] = load_session_data(resolved_dataset_root, speaker, session)
            session_data = session_cache[cache_key]
            frame_index_0based = frame_index_for_time(time_s, session_data.frame_rate, session_data.images.shape[0])
            frame_index_1based = frame_index_0based + 1
            frame = extract_frame_image(session_data, frame_index_0based)

            image_path = output_dir / vowel / f"{speaker}_{session}_frame_{frame_index_1based:04d}.png"
            imageio.imwrite(image_path, frame)
            selections.append(
                VowelSelection(
                    speaker=speaker,
                    vowel=vowel,
                    session=session,
                    frame_index_0based=frame_index_0based,
                    frame_index_1based=frame_index_1based,
                    time_s=float(time_s),
                    interval_start_s=float(interval_start_s),
                    interval_end_s=float(interval_end_s),
                    dataset_root=str(resolved_dataset_root),
                    textgrid_path=str(textgrid_path),
                    image_path=str(image_path),
                )
            )
            print(f"  - {speaker}: {session} frame {frame_index_1based}")

    summary = {
        "dataset_root": str(dataset_root),
        "output_dir": str(output_dir),
        "vowels": vowels,
        "speaker_count": len(speaker_dirs),
        "selections": [asdict(selection) for selection in selections],
        "missing": missing,
    }
    summary_path = output_dir / "selection_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[done] saved samples in {output_dir}")
    print(f"[done] summary: {summary_path}")


if __name__ == "__main__":
    main()
