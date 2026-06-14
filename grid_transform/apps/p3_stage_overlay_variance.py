from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")

from grid_transform.analysis_shared import speaker_id_sort_key
from grid_transform.config import DEFAULT_OUTPUT_DIR, DEFAULT_VTLN_DIR
from grid_transform.p3_stage_overlay_variance import (
    DEFAULT_TARGET_BASENAME,
    DEFAULT_TARGET_SPEAKER_ID,
    build_stage_masks,
    compute_source_to_target_stage_metrics,
    compute_stage_variance,
    load_p3_stage_overlay_speakers,
    map_speakers_to_p3_stages,
    resolve_closed_roi_labels,
    write_stage_overlay_v1_report,
)
from grid_transform.roi_average_speaker import DEFAULT_EXCLUDE_LABELS, default_curated_speaker_ids


def default_output_dir() -> Path:
    return DEFAULT_OUTPUT_DIR / "stage_overlays_v1"


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

    if DEFAULT_TARGET_SPEAKER_ID not in seen:
        ordered.append(DEFAULT_TARGET_SPEAKER_ID)
    return ordered


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Render fixed-P3 contour overlays and Dice-based ROI variance summaries "
            "for before_affine, after_affine, and after_tps stages."
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
        help="Comma-separated contour labels to exclude before closed-ROI Dice scoring.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=default_output_dir(),
        help="Directory where stage_overlays_v1 figures and reports will be written.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    speaker_ids = parse_speakers(args.speakers, args.vtln_dir)
    exclude_labels = parse_csv_values(args.exclude_labels)

    speakers = load_p3_stage_overlay_speakers(args.vtln_dir, speaker_ids)
    labels, excluded_rows = resolve_closed_roi_labels(speakers, exclude_labels=exclude_labels)
    stage_contours = map_speakers_to_p3_stages(speakers, labels, target_speaker_id=DEFAULT_TARGET_SPEAKER_ID)
    target_shape = tuple(np.asarray(speakers[DEFAULT_TARGET_SPEAKER_ID].image).shape)
    stage_masks = build_stage_masks(stage_contours, labels, target_shape)
    stage_metrics = compute_stage_variance(stage_masks, labels)
    source_to_target_rows = compute_source_to_target_stage_metrics(
        stage_masks,
        labels,
        target_speaker_id=DEFAULT_TARGET_SPEAKER_ID,
    )
    written = write_stage_overlay_v1_report(
        speakers,
        labels,
        excluded_rows,
        stage_contours,
        stage_metrics,
        source_to_target_rows,
        args.output_dir,
    )

    summaries = stage_metrics["stage_summaries"]
    variance_reductions = written["variance_reductions"]

    print("P3 fixed-target stage overlay variance workflow")
    print("===============================================")
    print(
        "Target:",
        f"{DEFAULT_TARGET_SPEAKER_ID} ({DEFAULT_TARGET_BASENAME})",
    )
    print("Speakers used:", ", ".join(sorted(speakers, key=speaker_id_sort_key)))
    print("Labels used:", ", ".join(labels))
    for stage_name in ("before_affine", "after_affine", "after_tps"):
        summary = summaries[stage_name]
        print(
            f"{stage_name}:",
            f"variance={summary.overall_variance:.6f}",
            f"| mean Dice={summary.overall_mean_pairwise_dice:.4f}",
            f"| mean Dice distance={summary.overall_mean_pairwise_dice_distance:.4f}",
        )
    print(
        "Variance reduction:",
        f"affine vs before={variance_reductions['affine_vs_before']:.2f}%" if variance_reductions["affine_vs_before"] is not None else "affine vs before=n/a",
        f"| tps vs before={variance_reductions['tps_vs_before']:.2f}%" if variance_reductions["tps_vs_before"] is not None else "| tps vs before=n/a",
        f"| tps vs affine={variance_reductions['tps_vs_affine']:.2f}%" if variance_reductions["tps_vs_affine"] is not None else "| tps vs affine=n/a",
    )
    print(f"Saved: {written['summary_json']}")
    print(f"Saved: {written['summary_md']}")
    print(f"Saved: {written['stage_variance_csv']}")
    print(f"Saved: {written['stage_label_variance_csv']}")
    print(f"Saved: {written['pairwise_csv']}")
    print(f"Saved: {written['per_source_to_target_csv']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
