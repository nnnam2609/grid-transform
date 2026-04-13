from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image

from grid_transform.config import DEFAULT_OUTPUT_DIR


DEFAULT_PNG_FILES = [
    DEFAULT_OUTPUT_DIR / "source_speaker_warped_to_target_comparison.png",
    DEFAULT_OUTPUT_DIR / "source_moved_to_target_articulators.png",
    DEFAULT_OUTPUT_DIR / "target_moved_to_source_articulators.png",
    DEFAULT_OUTPUT_DIR / "method4_step_by_step.png",
]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert selected PNG report images to PDF files.")
    parser.add_argument(
        "png_files",
        nargs="*",
        type=Path,
        default=DEFAULT_PNG_FILES,
        help="PNG files to convert. Defaults to the canonical report outputs.",
    )
    return parser.parse_args(argv)


def _pdf_safe_image(path: Path) -> Image.Image:
    image = Image.open(path)
    if image.mode in ("RGBA", "LA", "P"):
        background = Image.new("RGB", image.size, (255, 255, 255))
        background.paste(image, mask=image.split()[-1] if image.mode == "RGBA" else None)
        return background
    if image.mode != "RGB":
        return image.convert("RGB")
    return image


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    converted = 0
    for png_file in args.png_files:
        if not png_file.exists():
            print(f"Missing: {png_file}")
            continue
        image = _pdf_safe_image(png_file)
        pdf_file = png_file.with_suffix(".pdf")
        image.save(pdf_file, "PDF", quality=95)
        converted += 1
        print(f"Converted: {png_file.name} -> {pdf_file.name}")
    print(f"Done. Converted {converted} file(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
