from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

from grid_transform.config import DEFAULT_OUTPUT_DIR, DEFAULT_VTLN_DIR
from grid_transform.roi_average_speaker import (
    DEFAULT_EXCLUDE_LABELS,
    default_curated_speaker_ids,
    load_roi_average_speakers,
    rank_affine_roi_average_speakers,
    resolve_roi_average_labels,
    write_affine_roi_average_speaker_report,
)


def default_output_dir() -> Path:
    return DEFAULT_OUTPUT_DIR / "average_speaker_roi_affine"


def parse_csv_values(text: str) -> list[str]:
    return [value.strip() for value in text.split(",") if value.strip()]


def parse_speakers(text: str | None, vtln_dir: Path) -> list[str]:
    if text is None:
        return default_curated_speaker_ids(vtln_dir)
    speakers = [speaker.upper() for speaker in parse_csv_values(text)]
    if not speakers:
        raise ValueError("Expected at least one speaker id in --speakers.")
    seen: set[str] = set()
    ordered: list[str] = []
    for speaker in speakers:
        if speaker not in seen:
            ordered.append(speaker)
            seen.add(speaker)
    return ordered


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Rank curated P1..P10 speakers as affine-only ROI-overlap average-speaker candidates. "
            "The workflow closes selected contours into ROI polygons, scores ordered source->target "
            "pairs with Dice/IoU overlap, and picks the target speaker with the highest mean Dice."
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
        help="Comma-separated curated speaker ids. Default uses all manifest speakers in P-order.",
    )
    parser.add_argument(
        "--exclude-labels",
        default=",".join(DEFAULT_EXCLUDE_LABELS),
        help="Comma-separated contour labels to exclude before global ROI-overlap scoring.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=default_output_dir(),
        help="Directory where reports and figures will be written.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    speaker_ids = parse_speakers(args.speakers, args.vtln_dir)
    exclude_labels = parse_csv_values(args.exclude_labels)

    speakers = load_roi_average_speakers(args.vtln_dir, speaker_ids)
    labels, excluded_rows = resolve_roi_average_labels(speakers, exclude_labels=exclude_labels)
    ranking = rank_affine_roi_average_speakers(speakers, labels)
    written = write_affine_roi_average_speaker_report(
        speakers,
        ranking,
        args.output_dir,
        config={
            "vtln_dir": str(args.vtln_dir),
            "requested_speakers": speaker_ids,
            "exclude_labels": exclude_labels,
            "primary_metric": "Dice",
            "secondary_metric": "IoU",
            "transform_stage": "affine_only",
        },
        excluded_labels=excluded_rows,
    )

    winner = ranking["winner"]
    print("ROI-Dice affine average-speaker workflow")
    print("=======================================")
    print("Speakers used:", ", ".join(speaker_ids))
    print("Labels used:", ", ".join(labels))
    print(
        "Winner:",
        f"{winner.target_speaker_id} ({winner.target_basename})",
        f"| mean Dice = {winner.mean_pair_dice:.4f}",
        f"| mean IoU = {winner.mean_pair_iou:.4f}",
    )
    print(f"Saved: {written['summary_json']}")
    print(f"Saved: {written['summary_md']}")
    print(f"Saved: {written['candidate_scores_csv']}")
    print(f"Saved: {written['pair_scores_csv']}")
    print(f"Saved: {written['label_scores_csv']}")
    print(f"Saved figures under: {written['targets_dir']}")
    print(f"Saved figures under: {written['pairs_dir']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
