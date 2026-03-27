from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

from grid_transform.config import DEFAULT_VTNL_DIR
from grid_transform.session_warp import run_session_warp_to_target


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Warp a full ArtSpeech session into one nnUNet target frame space using either "
            "a fixed VTNL reference annotation or a saved edited source annotation."
        )
    )
    parser.add_argument(
        "--annotation-speaker",
        default="1640_s10_0829",
        help="VTNL annotation/reference speaker image name.",
    )
    parser.add_argument(
        "--source-annotation-json",
        type=Path,
        help="Optional saved edited source annotation JSON. Overrides --annotation-speaker when provided.",
    )
    parser.add_argument("--artspeech-speaker", default="P7", help="ArtSpeech speaker id, for example P7.")
    parser.add_argument("--session", default="S10", help="ArtSpeech session id, for example S10.")
    parser.add_argument("--target-frame", type=int, default=143020, help="nnUNet target frame number.")
    parser.add_argument(
        "--target-case",
        default="2008-003^01-1791/test",
        help="nnUNet target case relative path.",
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        help="Optional explicit ArtSpeech dataset root. Defaults to an auto-detected local path.",
    )
    parser.add_argument("--vtnl-dir", type=Path, default=DEFAULT_VTNL_DIR, help="Folder containing VTNL images and ROI zip files.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Optional explicit output directory.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=0,
        help="Optional debug limit. Use 0 for the full session.",
    )
    parser.add_argument(
        "--output-mode",
        choices=("both", "warped", "review"),
        default="both",
        help="Whether to write the clean warped video, the review video, or both.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    summary = run_session_warp_to_target(
        annotation_speaker=args.annotation_speaker,
        source_annotation_json=args.source_annotation_json,
        artspeech_speaker=args.artspeech_speaker,
        session=args.session,
        target_frame=args.target_frame,
        target_case=args.target_case,
        dataset_root=args.dataset_root,
        vtnl_dir=args.vtnl_dir,
        output_dir=args.output_dir,
        max_frames=args.max_frames,
        output_mode=args.output_mode,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
