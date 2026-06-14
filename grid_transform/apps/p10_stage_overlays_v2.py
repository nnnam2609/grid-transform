from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

from grid_transform.analysis_shared import speaker_id_sort_key
from grid_transform.config import DEFAULT_OUTPUT_DIR, DEFAULT_VTLN_DIR
from grid_transform.fixed_target_stage_overlays import (
    DEFAULT_OVERLAP_METRIC,
    DEFAULT_TARGET_BASENAME,
    DEFAULT_TARGET_SPEAKER_ID,
    SUPPORTED_OVERLAP_METRICS,
    export_fixed_target_stage_overlays,
    load_fixed_target_stage_overlay_speakers,
)


def default_output_dir() -> Path:
    return DEFAULT_OUTPUT_DIR


def parse_csv_values(text: str) -> list[str]:
    return [value.strip() for value in text.split(",") if value.strip()]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Export stage_overlays_v2 with fixed target P10. "
            "All speakers are rendered in P10 space for affine/TPS stages, "
            "and every non-pharynx contour is visually closed by connecting its last point back to its first point."
        )
    )
    parser.add_argument(
        "--vtln-dir",
        type=Path,
        default=DEFAULT_VTLN_DIR,
        help="Canonical curated VTLN folder. Defaults to VTLN/data.",
    )
    parser.add_argument(
        "--speakers",
        default=None,
        help="Optional comma-separated curated speaker ids. Default uses all manifest speakers and ensures P10 is included.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=default_output_dir(),
        help="Root output directory where the stage overlay folder will be created.",
    )
    parser.add_argument(
        "--overlap-metric",
        choices=SUPPORTED_OVERLAP_METRICS,
        default=DEFAULT_OVERLAP_METRIC,
        help="Overlap metric used for stage variance. Defaults to Dice.",
    )
    parser.add_argument(
        "--stage-dir-name",
        default=None,
        help="Optional output folder name under --output-dir. Defaults to stage_overlays_v2_<overlap-metric>.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    requested_speakers = None if args.speakers is None else [speaker.upper() for speaker in parse_csv_values(args.speakers)]
    speakers = load_fixed_target_stage_overlay_speakers(
        args.vtln_dir,
        requested_speakers=requested_speakers,
        target_speaker_id=DEFAULT_TARGET_SPEAKER_ID,
    )
    summary = export_fixed_target_stage_overlays(
        speakers,
        args.output_dir,
        target_speaker_id=DEFAULT_TARGET_SPEAKER_ID,
        overlap_metric=args.overlap_metric,
        stage_dir_name=args.stage_dir_name,
    )
    print("Fixed-target stage_overlays_v2 workflow")
    print("=======================================")
    print("Target:", f"{DEFAULT_TARGET_SPEAKER_ID} ({DEFAULT_TARGET_BASENAME})")
    print("Overlap metric:", args.overlap_metric)
    print("Speakers used:", ", ".join(sorted(speakers, key=speaker_id_sort_key)))
    print(f"Saved: {summary['summary_json']}")
    print(f"Saved: {summary['summary_txt']}")
    print(f"Saved: {summary['workshop_summary_md']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
