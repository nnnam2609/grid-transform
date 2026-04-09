from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from grid_transform.contour_names import normalize_contour_name
from grid_transform.config import DEFAULT_OUTPUT_DIR, DEFAULT_VTLN_DIR, VT_SEG_DATA_ROOT
from grid_transform.image_utils import as_grayscale_uint8
from grid_transform.io import load_frame_npy, load_frame_vtln


SLIDE_CONTOUR_COLORS = {
    "arytenoid-cartilage": "#ff4fa3",
    "brain-stem": "#8ecae6",
    "c1": "#457b9d",
    "c2": "#4d7ea8",
    "c3": "#5c96bc",
    "c4": "#7aa6c2",
    "c5": "#90b8d8",
    "c6": "#b8d8f2",
    "cerebellum": "#5bc0eb",
    "chin": "#7bd389",
    "epiglottis": "#00c853",
    "frontal-sinus": "#bde0fe",
    "geniohyoid-muscle": "#2ec4b6",
    "hyoid": "#ffbe0b",
    "incisior-hard-palate": "#5aa9ff",
    "lower-lip": "#39d353",
    "mandible-incisior": "#ff7f11",
    "nasal-root": "#cdb4db",
    "nose-tip": "#ffafcc",
    "occipital-crest": "#80ed99",
    "pharynx": "#ffd60a",
    "soft-palate": "#22d3ee",
    "soft-palate-midline": "#a855f7",
    "sphenoid": "#ade8f4",
    "thyroid-cartilage": "#9d4edd",
    "tongue": "#ff9f1c",
    "tongue-floor": "#2dd4bf",
    "upper-lip": "#d946ef",
    "vocal-folds": "#ff4d6d",
    "vomer": "#48cae4",
}
FALLBACK_COLORS = [
    "#ff6b6b",
    "#4dabf7",
    "#51cf66",
    "#ffd43b",
    "#845ef7",
    "#ffa94d",
    "#2ec4b6",
    "#f06595",
]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export slide-ready contour overlays for the canonical VTLN references and nnUNet target frame."
    )
    parser.add_argument("--vtln-dir", type=Path, default=DEFAULT_VTLN_DIR, help="Folder containing canonical VTLN PNG/ZIP pairs.")
    parser.add_argument(
        "--nnunet-case-dir",
        type=Path,
        default=VT_SEG_DATA_ROOT / "2008-003^01-1791" / "test",
        help="nnUNet case folder containing PNG_MR and contours.",
    )
    parser.add_argument("--nnunet-frame", type=int, default=143020, help="nnUNet frame number to export.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR / "slide" / "images",
        help="Output folder for slide-ready PNG overlays.",
    )
    parser.add_argument("--line-thickness", type=int, default=3, help="Contour stroke width in pixels.")
    parser.add_argument("--outline-thickness", type=int, default=5, help="Dark outline width under each contour.")
    return parser.parse_args(argv)


def hex_to_bgr(color: str) -> tuple[int, int, int]:
    value = color.lstrip("#")
    if len(value) != 6:
        raise ValueError(f"Expected a 6-digit hex color, got {color!r}")
    return int(value[4:6], 16), int(value[2:4], 16), int(value[0:2], 16)


def normalize_display_label(name: str) -> str:
    return normalize_contour_name(name, SLIDE_CONTOUR_COLORS)


def contour_bgr(name: str) -> tuple[int, int, int]:
    label = normalize_display_label(name)
    if label in SLIDE_CONTOUR_COLORS:
        return hex_to_bgr(SLIDE_CONTOUR_COLORS[label])
    color = FALLBACK_COLORS[sum(ord(ch) for ch in label) % len(FALLBACK_COLORS)]
    return hex_to_bgr(color)


def render_overlay(
    image,
    contours: dict[str, np.ndarray],
    *,
    line_thickness: int,
    outline_thickness: int,
) -> np.ndarray:
    gray = as_grayscale_uint8(image)
    panel = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    for name in sorted(contours):
        points = np.asarray(contours[name], dtype=float)
        if points.ndim != 2 or points.shape[1] != 2 or len(points) < 2:
            continue
        polyline = np.round(points).astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(
            panel,
            [polyline],
            isClosed=False,
            color=(18, 18, 18),
            thickness=outline_thickness,
            lineType=cv2.LINE_AA,
        )
        cv2.polylines(
            panel,
            [polyline],
            isClosed=False,
            color=contour_bgr(name),
            thickness=line_thickness,
            lineType=cv2.LINE_AA,
        )
    return panel


def export_nnunet_overlay(case_dir: Path, *, frame_number: int, output_dir: Path, line_thickness: int, outline_thickness: int) -> Path:
    image, contours = load_frame_npy(frame_number, case_dir, case_dir / "contours")
    overlay = render_overlay(
        image,
        contours,
        line_thickness=line_thickness,
        outline_thickness=outline_thickness,
    )
    case_token = case_dir.parent.name.replace("^", "_")
    output_path = output_dir / f"nnunet_{case_token}_{frame_number}_overlay.png"
    cv2.imwrite(str(output_path), overlay)
    return output_path


def export_vtln_overlays(vtln_dir: Path, *, output_dir: Path, line_thickness: int, outline_thickness: int) -> list[Path]:
    output_paths: list[Path] = []
    for zip_path in sorted(vtln_dir.glob("*.zip")):
        basename = zip_path.stem
        image, contours = load_frame_vtln(basename, vtln_dir)
        overlay = render_overlay(
            image,
            contours,
            line_thickness=line_thickness,
            outline_thickness=outline_thickness,
        )
        output_path = output_dir / f"{basename}_overlay.png"
        cv2.imwrite(str(output_path), overlay)
        output_paths.append(output_path)
    return output_paths


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    nnunet_output = export_nnunet_overlay(
        args.nnunet_case_dir,
        frame_number=args.nnunet_frame,
        output_dir=args.output_dir,
        line_thickness=args.line_thickness,
        outline_thickness=args.outline_thickness,
    )
    vtln_outputs = export_vtln_overlays(
        args.vtln_dir,
        output_dir=args.output_dir,
        line_thickness=args.line_thickness,
        outline_thickness=args.outline_thickness,
    )

    print(f"Saved nnUNet overlay: {nnunet_output}")
    print(f"Saved VTLN overlays: {len(vtln_outputs)}")
    for path in vtln_outputs:
        print(f"  {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
