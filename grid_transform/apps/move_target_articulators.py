from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

from grid_transform.articulators import (
    compute_articulator_errors,
    parse_articulators,
    save_comparison_figure,
)
from grid_transform.config import (
    DEFAULT_VTNL_DIR,
    DEFAULT_OUTPUT_DIR,
    VT_SEG_CONTOURS_ROOT,
    VT_SEG_DATA_ROOT,
)
from grid_transform.io import load_frame_npy, load_frame_vtnl
from grid_transform.transfer import (
    build_two_step_transform,
    smooth_transformed_contours,
    transform_contours,
)
from grid_transform.vt import build_grid


def build_parser():
    parser = argparse.ArgumentParser(
        description="Move source articulators into target space with the same affine + TPS pipeline.",
    )
    parser.add_argument("--target-speaker", default="1640_s10_0829", help="VTNL target speaker/image name.")
    parser.add_argument("--source-frame", type=int, default=143020, help="nnUNet source frame number.")
    parser.add_argument("--case", default="2008-003^01-1791/test", help="nnUNet case relative path.")
    parser.add_argument("--vtnl-dir", type=Path, default=DEFAULT_VTNL_DIR, help="Folder containing VTNL images and ROI zip files.")
    parser.add_argument(
        "--articulators",
        help="Comma-separated articulator list. Default uses the common moving articulators.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_DIR / "source_moved_to_target_articulators.png",
        help="Output comparison image path.",
    )
    return parser


def main(argv: list[str] | None = None):
    args = build_parser().parse_args(argv)

    target_image, target_contours = load_frame_vtnl(args.target_speaker, args.vtnl_dir)
    source_image, source_contours = load_frame_npy(
        args.source_frame,
        VT_SEG_DATA_ROOT / args.case,
        VT_SEG_CONTOURS_ROOT / args.case,
    )

    articulators = parse_articulators(args.articulators, target_contours, source_contours)

    # Fit source -> target so the moved source can be compared directly on top of target.
    target_grid = build_grid(target_image, target_contours, n_vert=9, n_points=250, frame_number=0)
    source_grid = build_grid(source_image, source_contours, n_vert=9, n_points=250, frame_number=args.source_frame)
    transform = build_two_step_transform(source_grid, target_grid)

    moved_source_contours = transform_contours(source_contours, transform["apply_two_step"], articulators)
    moved_source_contours = smooth_transformed_contours(moved_source_contours)
    errors = compute_articulator_errors(moved_source_contours, target_contours)

    save_comparison_figure(
        target_image,
        source_image,
        target_contours,
        source_contours,
        moved_source_contours,
        articulators,
        errors,
        args.output,
    )

    print("Step 1 anchors:", ", ".join(transform["step1_labels"]))
    print("Step 2 controls:", ", ".join(transform["step2_labels"]))
    print("Articulators:", ", ".join(articulators))
    for name in articulators:
        print(f"{name}: RMS = {errors[name]:.2f} px")
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
