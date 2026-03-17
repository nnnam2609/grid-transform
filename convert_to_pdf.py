"""Convert PNG images to PDF for printing."""

from pathlib import Path
from PIL import Image

# Files to convert
output_dir = Path("outputs")
png_files = [
    output_dir / "source_speaker_warped_to_target_comparison.png",
    output_dir / "source_moved_to_target_articulators.png",
    output_dir / "target_moved_to_source_articulators.png",
    output_dir / "method4_step_by_step.png",
]

for png_file in png_files:
    if png_file.exists():
        # Convert PNG to PDF
        img = Image.open(png_file)
        
        # Convert RGBA to RGB if needed (for PDF compatibility)
        if img.mode in ('RGBA', 'LA', 'P'):
            # Create a white background
            background = Image.new('RGB', img.size, (255, 255, 255))
            background.paste(img, mask=img.split()[-1] if img.mode == 'RGBA' else None)
            img = background
        elif img.mode != 'RGB':
            img = img.convert('RGB')
        
        # Save as PDF
        pdf_file = png_file.with_suffix('.pdf')
        img.save(pdf_file, 'PDF', quality=95)
        print(f"✓ Converted: {png_file.name} → {pdf_file.name}")
    else:
        print(f"✗ Not found: {png_file}")

print("\nDone! PDF files are saved in the outputs/ directory.")
