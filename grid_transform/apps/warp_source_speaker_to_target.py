from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import numpy as np
from PIL import Image

from grid_transform.config import (
    DEFAULT_VTNL_DIR,
    DEFAULT_OUTPUT_DIR,
    VT_SEG_CONTOURS_ROOT,
    VT_SEG_DATA_ROOT,
)
from grid_transform.io import load_frame_npy, load_frame_vtnl
from grid_transform.transfer import (
    DEFAULT_ARTICULATORS,
    build_two_step_transform,
    resolve_common_articulators,
    smooth_transformed_contours,
    transform_contours,
)
from grid_transform.warp import save_comparison_figure, warp_image_to_target_space
from grid_transform.vt import build_grid


def resolve_articulators(target_contours, source_contours, requested):
    """Pick a stable shared articulator set for overlay visualization."""
    return resolve_common_articulators(
        target_contours,
        source_contours,
        requested,
        defaults=DEFAULT_ARTICULATORS,
    )


def build_parser():
    parser = argparse.ArgumentParser(
        description="Warp the whole source speaker image into target speaker space with affine + TPS.",
    )
    parser.add_argument("--target-speaker", default="1640_s10_0654", help="VTNL target speaker/image name.")
    parser.add_argument("--source-frame", type=int, default=143020, help="nnUNet source frame number.")
    parser.add_argument("--case", default="2008-003^01-1791/test", help="nnUNet case relative path.")
    parser.add_argument("--vtnl-dir", type=Path, default=DEFAULT_VTNL_DIR, help="Folder containing VTNL images and ROI zip files.")
    parser.add_argument(
        "--articulators",
        help="Comma-separated articulators for overlay visualization.",
    )
    parser.add_argument(
        "--warped-image-output",
        type=Path,
        default=DEFAULT_OUTPUT_DIR / "source_speaker_warped_to_target.png",
        help="Where to save the warped source image.",
    )
    parser.add_argument(
        "--mask-output",
        type=Path,
        default=DEFAULT_OUTPUT_DIR / "source_speaker_warped_mask.png",
        help="Where to save the valid warp mask.",
    )
    parser.add_argument(
        "--figure-output",
        type=Path,
        default=DEFAULT_OUTPUT_DIR / "source_speaker_warped_to_target_comparison.png",
        help="Where to save the comparison figure.",
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

    articulators = resolve_articulators(target_contours, source_contours, args.articulators)

    target_grid = build_grid(target_image, target_contours, n_vert=9, n_points=250, frame_number=0)
    source_grid = build_grid(source_image, source_contours, n_vert=9, n_points=250, frame_number=args.source_frame)

    forward_transform = build_two_step_transform(source_grid, target_grid)
    inverse_transform = build_two_step_transform(target_grid, source_grid)

    warped_source_arr, warped_mask_arr = warp_image_to_target_space(
        source_image,
        np.asarray(target_image).shape,
        inverse_transform["apply_two_step"],
    )
    warped_source_contours = transform_contours(
        source_contours,
        forward_transform["apply_two_step"],
        articulators,
    )
    warped_source_contours = smooth_transformed_contours(warped_source_contours)

    args.warped_image_output.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(warped_source_arr).save(args.warped_image_output)
    Image.fromarray(warped_mask_arr).save(args.mask_output)

    save_comparison_figure(
        target_image,
        source_image,
        warped_source_arr,
        target_contours,
        source_contours,
        warped_source_contours,
        articulators,
        args.figure_output,
    )

    print("Forward Step 1 anchors:", ", ".join(forward_transform["step1_labels"]))
    print("Forward Step 2 controls:", ", ".join(forward_transform["step2_labels"]))
    print("Articulators shown:", ", ".join(articulators))
    print(f"Saved warped image: {args.warped_image_output}")
    print(f"Saved warp mask: {args.mask_output}")
    print(f"Saved figure: {args.figure_output}")


if __name__ == "__main__":
    main()
