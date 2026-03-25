"""Convert selected experiment PNGs into print-friendly PDFs."""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image

from grid_transform.config import DEFAULT_OUTPUT_DIR


DEFAULT_PNG_NAMES = (
    "source_speaker_warped_to_target_comparison.png",
    "source_moved_to_target_articulators.png",
    "target_moved_to_source_articulators.png",
    "method4_step_by_step.png",
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert selected experiment PNGs into print-friendly PDFs."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory containing the PNG outputs to convert.",
    )
    return parser.parse_args(argv)


def to_pdf_compatible(image: Image.Image) -> Image.Image:
    """Convert a PIL image to RGB for PDF export."""
    if image.mode == "RGB":
        return image

    if image.mode in ("RGBA", "LA", "P"):
        background = Image.new("RGB", image.size, (255, 255, 255))
        alpha = image.split()[-1] if image.mode == "RGBA" else None
        background.paste(image, mask=alpha)
        return background

    return image.convert("RGB")


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    png_files = [args.output_dir / name for name in DEFAULT_PNG_NAMES]

    for png_file in png_files:
        if not png_file.exists():
            print(f"[missing] {png_file}")
            continue

        with Image.open(png_file) as image:
            pdf_image = to_pdf_compatible(image)
            pdf_file = png_file.with_suffix(".pdf")
            pdf_image.save(pdf_file, "PDF", quality=95)
        print(f"[ok] {png_file.name} -> {pdf_file.name}")

    print(f"\nSaved PDF files in: {args.output_dir}")


if __name__ == "__main__":
    main()
