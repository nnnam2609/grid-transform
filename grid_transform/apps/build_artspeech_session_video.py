from __future__ import annotations

import argparse
import json
from pathlib import Path

from grid_transform.artspeech_video import default_output_dir, resolve_default_dataset_root, run_session_video


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a reusable ArtSpeech session video from DICOM + audio + TextGrid/TRS."
    )
    parser.add_argument("--speaker", required=True, help="ArtSpeech speaker id, for example P7.")
    parser.add_argument("--session", default="S10", help="ArtSpeech session id, for example S10.")
    parser.add_argument(
        "--dataset-root",
        type=Path,
        help="Optional explicit dataset root. Defaults to an auto-detected local path.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Optional explicit output directory. Defaults to outputs/videos/<speaker>_<session>/.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=0,
        help="Optional debug limit. Use 0 for the full session.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    dataset_root = args.dataset_root or resolve_default_dataset_root(args.speaker)
    output_dir = args.output_dir or default_output_dir(args.speaker, args.session)
    result = run_session_video(
        speaker=args.speaker,
        session=args.session,
        dataset_root=dataset_root,
        output_dir=output_dir,
        max_frames=args.max_frames,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
