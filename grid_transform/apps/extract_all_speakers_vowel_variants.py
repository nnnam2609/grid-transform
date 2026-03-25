from __future__ import annotations

import argparse
import json
from pathlib import Path

from grid_transform.apps.extract_speaker_vowel_variants import main as extract_speaker_main
from grid_transform.config import DEFAULT_OUTPUT_DIR, PROJECT_DIR


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the within-speaker vowel variant extraction workflow for multiple ArtSpeech speakers "
            "and place all speaker folders under one shared output directory."
        )
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=PROJECT_DIR.parent / "Data" / "Artspeech_database",
        help="Root directory containing ArtSpeech speaker folders.",
    )
    parser.add_argument(
        "--speakers",
        default="P1,P2,P3,P4,P5,P6,P7,P8,P9,P10",
        help="Comma-separated speaker ids. Defaults to P1..P10.",
    )
    parser.add_argument(
        "--vowels",
        default="a,e,i,o,u",
        help="Comma-separated vowel labels to extract. Defaults to a,e,i,o,u.",
    )
    parser.add_argument(
        "--samples-per-vowel",
        type=int,
        default=10,
        help="Maximum number of samples per vowel for each speaker. Defaults to 10.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Optional batch output directory. Defaults to outputs/speaker_vowel_variants/all_speakers_<N>_samples/.",
    )
    return parser.parse_args(argv)


def parse_speakers(text: str) -> list[str]:
    speakers = [value.strip() for value in text.split(",") if value.strip()]
    if not speakers:
        raise ValueError("No speakers were provided.")
    return speakers


def default_output_dir(samples_per_vowel: int) -> Path:
    return DEFAULT_OUTPUT_DIR / "speaker_vowel_variants" / f"all_speakers_{samples_per_vowel}_samples"


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    speakers = parse_speakers(args.speakers)
    batch_output_dir = args.output_dir or default_output_dir(args.samples_per_vowel)
    batch_output_dir.mkdir(parents=True, exist_ok=True)

    speaker_outputs: list[dict[str, object]] = []
    for speaker in speakers:
        speaker_output_dir = batch_output_dir / speaker
        print(f"[batch] running speaker {speaker} -> {speaker_output_dir}")
        extract_speaker_main(
            [
                "--speaker",
                speaker,
                "--dataset-root",
                str(args.dataset_root),
                "--output-dir",
                str(speaker_output_dir),
                "--vowels",
                args.vowels,
                "--samples-per-vowel",
                str(args.samples_per_vowel),
            ]
        )
        speaker_outputs.append(
            {
                "speaker": speaker,
                "output_dir": str(speaker_output_dir),
                "contact_sheets_dir": str(speaker_output_dir / "contact_sheets"),
                "summary_path": str(speaker_output_dir / "speaker_vowel_variants_summary.json"),
            }
        )

    summary = {
        "dataset_root": str(args.dataset_root),
        "batch_output_dir": str(batch_output_dir),
        "speakers": speakers,
        "vowels": [value.strip() for value in args.vowels.split(",") if value.strip()],
        "samples_per_vowel": args.samples_per_vowel,
        "speaker_outputs": speaker_outputs,
    }
    summary_path = batch_output_dir / "batch_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[done] batch output: {batch_output_dir}")
    print(f"[done] batch summary: {summary_path}")


if __name__ == "__main__":
    main()
