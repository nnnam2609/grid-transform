from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from grid_transform.config import DEFAULT_OUTPUT_DIR, VT_SEG_DATA_ROOT
from grid_transform.io import load_frame_npy


DEFAULT_OUTPUT_DIR_REPORT = DEFAULT_OUTPUT_DIR / "slide" / "images"
NNUNET_CASE_DIR_DEFAULT = VT_SEG_DATA_ROOT / "2008-003^01-1791" / "test"
NNUNET_FRAME_DEFAULT = 143020


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export a simple legend and/or centroid overlay for all nnUNet elements."
    )
    parser.add_argument(
        "--mode",
        choices=("legend", "overlay", "both"),
        default="both",
        help="Which export(s) to generate.",
    )
    parser.add_argument(
        "--nnunet-case-dir",
        type=Path,
        default=NNUNET_CASE_DIR_DEFAULT,
        help="nnUNet case folder containing PNG_MR and contours.",
    )
    parser.add_argument(
        "--nnunet-frame",
        type=int,
        default=NNUNET_FRAME_DEFAULT,
        help="nnUNet frame number.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR_REPORT,
        help="Output folder for the generated files.",
    )
    return parser.parse_args(argv)


def _load_fonts() -> tuple[ImageFont.FreeTypeFont | ImageFont.ImageFont, ImageFont.FreeTypeFont | ImageFont.ImageFont]:
    try:
        return ImageFont.truetype("arial.ttf", 28), ImageFont.truetype("arial.ttf", 20)
    except Exception:
        fallback = ImageFont.load_default()
        return fallback, fallback


def export_legend(*, case_dir: Path, frame_number: int, output_dir: Path) -> tuple[Path, Path]:
    _image, contours = load_frame_npy(frame_number, case_dir, case_dir / "contours")
    labels = sorted(contours)
    output_dir.mkdir(parents=True, exist_ok=True)

    txt_path = output_dir / "nnunet_elements_list.txt"
    txt_path.write_text("\n".join(labels) + "\n", encoding="utf-8")

    title_font, item_font = _load_fonts()
    image_width, image_height = 1000, 1200
    panel = Image.new("RGB", (image_width, image_height), color=(255, 255, 255))
    draw = ImageDraw.Draw(panel)
    padding = 30
    x_pos = padding
    y_pos = padding

    draw.text((x_pos, y_pos), "nnUNet Ground-Truth Elements", fill=(10, 70, 160), font=title_font)
    y_pos += 48
    draw.text((x_pos, y_pos), f"Case: {case_dir.name}   Frame: {frame_number}", fill=(20, 20, 20), font=item_font)
    y_pos += 36
    draw.line([(x_pos, y_pos), (image_width - padding, y_pos)], fill=(200, 200, 200))
    y_pos += 16

    column_x = [x_pos, image_width // 2 + 20]
    header_y = y_pos
    row_height = 26
    rows_per_column = max(1, (image_height - y_pos - padding) // row_height)
    for index, name in enumerate(labels):
        column_index = min(index // rows_per_column, len(column_x) - 1)
        row_index = index % rows_per_column
        draw.text(
            (column_x[column_index], header_y + row_index * row_height),
            f"- {name}",
            fill=(20, 20, 20),
            font=item_font,
        )

    output_path = output_dir / "nnunet_elements_list.png"
    panel.save(output_path)
    return txt_path, output_path


def export_overlay(*, case_dir: Path, frame_number: int, output_dir: Path) -> Path:
    image, contours = load_frame_npy(frame_number, case_dir, case_dir / "contours")
    panel = image.convert("RGBA")
    draw = ImageDraw.Draw(panel)
    try:
        label_font = ImageFont.truetype("arial.ttf", 18)
        title_font = ImageFont.truetype("arial.ttf", 26)
    except Exception:
        label_font = ImageFont.load_default()
        title_font = ImageFont.load_default()

    width, height = panel.size
    draw.rectangle([(0, 0), (width, 58)], fill=(0, 0, 0, 120))
    draw.text((12, 8), f"nnUNet: {case_dir.name}  frame: {frame_number}", font=title_font, fill=(255, 255, 255, 255))

    marker_color = (255, 170, 30, 255)
    text_color = (255, 255, 255, 255)
    outline_color = (0, 0, 0, 200)
    padding = 12

    for name in sorted(contours):
        points = np.asarray(contours[name], dtype=float)
        if points.size == 0:
            continue
        center_x, center_y = points.mean(axis=0)
        cx = int(round(center_x))
        cy = int(round(center_y))
        radius = 6
        draw.ellipse([(cx - radius, cy - radius), (cx + radius, cy + radius)], fill=marker_color, outline=outline_color)
        label = name.replace("-", " ")
        try:
            text_width, text_height = draw.textbbox((0, 0), label, font=label_font)[2:]
        except Exception:
            text_width, text_height = label_font.getsize(label)
        label_x = cx + 10
        label_y = cy - 10
        if label_x + text_width + padding > width:
            label_x = cx - 10 - text_width
        if label_x < padding:
            label_x = padding
        if label_y < 60:
            label_y = 60
        if label_y + text_height + padding > height:
            label_y = height - text_height - padding
        for offset_x, offset_y in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            draw.text((label_x + offset_x, label_y + offset_y), label, font=label_font, fill=outline_color)
        draw.text((label_x, label_y), label, font=label_font, fill=text_color)

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "nnunet_elements_overlay.png"
    panel.save(output_path)
    return output_path


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    outputs: list[Path] = []
    if args.mode in ("legend", "both"):
        outputs.extend(export_legend(case_dir=args.nnunet_case_dir, frame_number=args.nnunet_frame, output_dir=args.output_dir))
    if args.mode in ("overlay", "both"):
        outputs.append(export_overlay(case_dir=args.nnunet_case_dir, frame_number=args.nnunet_frame, output_dir=args.output_dir))
    for path in outputs:
        print(f"Wrote: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
