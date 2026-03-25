from __future__ import annotations

import argparse
import json
from pathlib import Path

from grid_transform.artspeech_video import DEFAULT_SESSION, resolve_default_dataset_root, run_session_video
from grid_transform.config import P4_SAMPLE_OUTPUT_DIR


DEFAULT_SPEAKER = "P4"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a P4/S10 sample video from DICOM + audio + TextGrid/TRS."
    )
    parser.add_argument("--dataset-root", type=Path, default=resolve_default_dataset_root(DEFAULT_SPEAKER))
    parser.add_argument("--speaker", default=DEFAULT_SPEAKER)
    parser.add_argument("--session", default=DEFAULT_SESSION)
    parser.add_argument("--output-dir", type=Path, default=P4_SAMPLE_OUTPUT_DIR)
    parser.add_argument(
        "--max-frames",
        type=int,
        default=0,
        help="Optional debug limit. Use 0 for the full session.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    result = run_session_video(
        speaker=args.speaker,
        session=args.session,
        dataset_root=args.dataset_root,
        output_dir=args.output_dir,
        max_frames=args.max_frames,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
