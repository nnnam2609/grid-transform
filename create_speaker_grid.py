from __future__ import annotations

import argparse
import os
import sys
import zipfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from roifile import ImagejRoi


PROJECT_DIR = Path(__file__).resolve().parent
VT_SEG_DIR = PROJECT_DIR / "vocal-tract-seg"
DEFAULT_VTNL_DIR = PROJECT_DIR / "VTNL" / "iadi_replace_merge"
TONGUE_COLOR = "#ff006e"
if str(VT_SEG_DIR) not in sys.path:
    sys.path.insert(0, str(VT_SEG_DIR))

from vt_grid import build_grid, print_grid_summary, visualize_grid


VIS_STYLE_PRESETS = {
    "target": {
        "horiz_color": "#20c997",
        "vert_color": "#ffd166",
        "vt_color": "#ef476f",
        "spine_color": "#118ab2",
        "guide_color": "#ff006e",
        "i_color": "#ef476f",
        "c_color": "#118ab2",
        "p1_color": "#ff006e",
        "m1_color": "#ef476f",
        "l6_color": "#06d6a0",
        "interp_color": "#20c997",
        "mandible_color": "#f4a261",
        "tongue_color": TONGUE_COLOR,
    },
    "source": {
        "horiz_color": "#3a86ff",
        "vert_color": "#ffbe0b",
        "vt_color": "#8338ec",
        "spine_color": "#2a9d8f",
        "guide_color": "#fb5607",
        "i_color": "#8338ec",
        "c_color": "#2a9d8f",
        "p1_color": "#fb5607",
        "m1_color": "#8338ec",
        "l6_color": "#80ed99",
        "interp_color": "#3a86ff",
        "mandible_color": "#e76f51",
        "tongue_color": TONGUE_COLOR,
    },
}


VTNL_CONTOUR_NAME_MAP = {
    "incisor-hard-palate": "incisior-hard-palate",
    "mandible-incisor": "mandible-incisior",
}


def normalize_vtnl_contour_name(label: str) -> str:
    """Map VTNL/iADI contour labels to the canonical names used in the repo."""
    return VTNL_CONTOUR_NAME_MAP.get(label, label)


def load_frame_npy(frame_number: int, data_dir: Path, contours_dir: Path):
    image_path = data_dir / "PNG_MR" / f"{frame_number}.png"
    if not image_path.is_file():
        raise FileNotFoundError(f"Image not found: {image_path}")

    image = Image.open(image_path)
    contours = {}
    for npy_path in sorted(contours_dir.glob(f"{frame_number}_*.npy")):
        name = npy_path.stem.replace(f"{frame_number}_", "")
        points = np.load(npy_path)
        if len(points) > 0:
            contours[name] = points

    if not contours:
        raise FileNotFoundError(f"No contour files found in {contours_dir} for frame {frame_number}")

    return image, contours


def load_frame_vtnl(image_name: str, vtnl_dir: Path):
    image = None
    for ext in (".png", ".tif", ".tiff"):
        image_path = vtnl_dir / f"{image_name}{ext}"
        if image_path.is_file():
            image = Image.open(image_path).convert("L")
            break

    if image is None:
        raise FileNotFoundError(f"No VTNL image found for {image_name} in {vtnl_dir}")

    zip_path = vtnl_dir / f"{image_name}.zip"
    if not zip_path.is_file():
        raise FileNotFoundError(f"ROI zip not found: {zip_path}")

    contours = {}
    with zipfile.ZipFile(zip_path) as zf:
        for name in sorted(zf.namelist()):
            if not name.endswith(".roi"):
                continue
            roi = ImagejRoi.frombytes(zf.read(name))
            coords = roi.coordinates()
            if coords is None or len(coords) == 0:
                continue
            label = Path(name).stem.replace(f"{image_name}_", "")
            label = normalize_vtnl_contour_name(label)
            contours[label] = np.array(coords, dtype=float)

    if not contours:
        raise FileNotFoundError(f"No ROI contours found in {zip_path}")

    return image, contours


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build a vocal-tract grid overlay for a speaker and save it as an image."
    )
    parser.add_argument(
        "--source",
        choices=("vtnl", "nnunet"),
        required=True,
        help="Input data source.",
    )
    parser.add_argument(
        "--speaker",
        help="VTNL speaker/image name, for example 1640_s10_0654.",
    )
    parser.add_argument(
        "--frame",
        type=int,
        default=143020,
        help="nnUNet frame number to load.",
    )
    parser.add_argument(
        "--case",
        default="2008-003^01-1791/test",
        help="nnUNet case path relative to vocal-tract-seg/data_80 and results/.../inference_contours.",
    )
    parser.add_argument(
        "--vtnl-dir",
        type=Path,
        default=DEFAULT_VTNL_DIR,
        help="Folder containing VTNL images and ROI zip files.",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=VT_SEG_DIR / "data_80",
        help="Root folder for nnUNet image data.",
    )
    parser.add_argument(
        "--contours-root",
        type=Path,
        default=VT_SEG_DIR / "results" / "nnunet_080" / "inference_contours",
        help="Root folder for nnUNet contour .npy files.",
    )
    parser.add_argument(
        "--n-vert",
        type=int,
        default=9,
        help="Number of vertical grid lines.",
    )
    parser.add_argument(
        "--n-points",
        type=int,
        default=250,
        help="Number of sample points per grid polyline.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output image path. Defaults to outputs/<speaker>_grid.png.",
    )
    parser.add_argument(
        "--hide-labels",
        action="store_true",
        help="Do not draw landmark labels on the saved overlay.",
    )
    parser.add_argument(
        "--hide-contours",
        action="store_true",
        help="Do not draw faint contour lines behind the grid.",
    )
    parser.add_argument(
        "--speaker-role",
        choices=("target", "source"),
        help="Visual style preset for the speaker overlay.",
    )
    return parser


def resolve_output_path(args: argparse.Namespace) -> Path:
    if args.output is not None:
        return args.output

    if args.source == "vtnl":
        stem = args.speaker
    else:
        stem = f"{args.case.replace('/', '_')}_frame_{args.frame}"

    return PROJECT_DIR / "outputs" / f"{stem}_grid.png"


def resolve_speaker_role(args: argparse.Namespace) -> str:
    if args.speaker_role is not None:
        return args.speaker_role
    return "target" if args.source == "vtnl" else "source"


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if args.source == "vtnl" and not args.speaker:
        parser.error("--speaker is required when --source vtnl")

    if args.source == "vtnl":
        image, contours = load_frame_vtnl(args.speaker, args.vtnl_dir)
        frame_number = 0
        title_name = args.speaker
    else:
        data_dir = args.data_root / args.case
        contours_dir = args.contours_root / args.case
        image, contours = load_frame_npy(args.frame, data_dir, contours_dir)
        frame_number = args.frame
        title_name = f"{args.case} frame {args.frame}"

    grid = build_grid(
        image,
        contours,
        n_vert=args.n_vert,
        n_points=args.n_points,
        frame_number=frame_number,
    )

    output_path = resolve_output_path(args)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    speaker_role = resolve_speaker_role(args)
    style = VIS_STYLE_PRESETS[speaker_role]

    fig = visualize_grid(
        grid,
        figsize=(14, 14),
        show_contours=not args.hide_contours,
        show_labels=not args.hide_labels,
        style=style,
    )
    ax = fig.axes[0]
    ax.set_title(
        f"{title_name}\nGrid overlay ({speaker_role})",
        fontsize=14,
        fontweight="bold",
    )
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    print_grid_summary(grid)
    print(f"Saved overlay image to: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
