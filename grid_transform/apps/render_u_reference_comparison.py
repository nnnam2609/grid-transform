from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path

import imageio.v2 as imageio
import matplotlib
import numpy as np
from PIL import Image

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from grid_transform.config import DEFAULT_OUTPUT_DIR, DEFAULT_VTLN_DIR
from grid_transform.image_utils import as_grayscale_uint8
from grid_transform.io import candidate_vtln_dirs


RAW_SUBJECT_BY_SPEAKER = {
    "P1": "1612",
    "P2": "1617",
    "P3": "1618",
    "P4": "1628",
    "P5": "1635",
    "P6": "1638",
    "P7": "1640",
    "P8": "1653",
    "P9": "1659",
    "P10": "1662",
}


@dataclass
class SampleRow:
    speaker: str
    session: str
    frame_index_1based: int
    sample_rank: int
    image_path: Path
    correlation: float


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Render one side-by-side comparison sheet per speaker using the extracted "
            "/u/ samples and a VTLN reference frame."
        )
    )
    parser.add_argument(
        "--reference-name",
        default="1640_s10_0829",
        help="VTLN reference image name without extension.",
    )
    parser.add_argument(
        "--vtln-dir",
        type=Path,
        default=DEFAULT_VTLN_DIR,
        help="Primary VTLN directory. The parent is also searched as a fallback.",
    )
    parser.add_argument(
        "--samples-root",
        type=Path,
        default=DEFAULT_OUTPUT_DIR / "speaker_vowel_variants" / "all_speakers_10_samples",
        help="Root directory containing the extracted speaker vowel summaries.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR / "reports" / "u_vs_1640_s10_0829",
        help="Directory for the generated PNGs and CSV manifest.",
    )
    return parser.parse_args(argv)


def speaker_sort_key(speaker: str) -> tuple[int, str]:
    return (int(speaker[1:]), speaker)


def session_sort_key(session: str) -> tuple[int, str]:
    return (int(session[1:]), session)


def find_reference_image(reference_name: str, vtln_dir: Path) -> Path:
    for base_dir in candidate_vtln_dirs(vtln_dir):
        for ext in (".png", ".tif", ".tiff"):
            candidate = base_dir / f"{reference_name}{ext}"
            if candidate.is_file():
                return candidate
    raise FileNotFoundError(f"Reference image not found for {reference_name} under {vtln_dir}")


def load_grayscale(path: Path) -> np.ndarray:
    return np.asarray(as_grayscale_uint8(imageio.imread(path)), dtype=np.float32)


def resize_to_match(image: np.ndarray, target_shape: tuple[int, int]) -> np.ndarray:
    target_h, target_w = target_shape
    pil_image = Image.fromarray(np.clip(image, 0, 255).astype(np.uint8))
    resized = pil_image.resize((target_w, target_h), resample=Image.BILINEAR)
    return np.asarray(resized, dtype=np.float32)


def pearson_corr(a: np.ndarray, b: np.ndarray) -> float:
    a_flat = a.astype(np.float32).ravel()
    b_flat = b.astype(np.float32).ravel()
    a_std = float(a_flat.std())
    b_std = float(b_flat.std())
    if a_std < 1e-8 or b_std < 1e-8:
        return 0.0
    return float(np.corrcoef(a_flat, b_flat)[0, 1])


def load_u_samples(summary_path: Path, reference_image: np.ndarray) -> list[SampleRow]:
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    speaker = str(payload["speaker"])
    rows: list[SampleRow] = []
    for row in payload["selections"]:
        if row["vowel"] != "u":
            continue
        image_path = Path(str(row["image_path"]))
        image = load_grayscale(image_path)
        reference_resized = resize_to_match(reference_image, image.shape)
        rows.append(
            SampleRow(
                speaker=speaker,
                session=str(row["session"]),
                frame_index_1based=int(row["frame_index_1based"]),
                sample_rank=int(row["sample_rank"]),
                image_path=image_path,
                correlation=pearson_corr(reference_resized, image),
            )
        )
    rows.sort(key=lambda row: (-row.correlation, row.sample_rank, session_sort_key(row.session)))
    return rows


def render_sheet(
    speaker: str,
    rows: list[SampleRow],
    reference_name: str,
    reference_image: np.ndarray,
    output_path: Path,
) -> None:
    fig = plt.figure(figsize=(17.5, 7.4), dpi=180)
    grid = fig.add_gridspec(2, 6, width_ratios=[1.2, 1, 1, 1, 1, 1], wspace=0.12, hspace=0.22)

    ref_ax = fig.add_subplot(grid[:, 0])
    ref_ax.imshow(reference_image, cmap="gray", vmin=0, vmax=255)
    ref_ax.set_title(f"Reference\n{reference_name}", fontsize=12, fontweight="bold")
    ref_ax.axis("off")

    sample_axes = [fig.add_subplot(grid[row, col]) for row in range(2) for col in range(1, 6)]
    for display_rank, (ax, row) in enumerate(zip(sample_axes, rows), start=1):
        image = load_grayscale(row.image_path)
        ax.imshow(image, cmap="gray", vmin=0, vmax=255)
        ax.set_title(
            f"#{display_rank}  {row.session}  f{row.frame_index_1based:04d}\n"
            f"corr={row.correlation:.4f}",
            fontsize=9.4,
            fontweight="bold",
        )
        ax.axis("off")

    for ax in sample_axes[len(rows) :]:
        ax.axis("off")

    raw_subject = RAW_SUBJECT_BY_SPEAKER.get(speaker)
    speaker_label = f"{speaker} (raw {raw_subject})" if raw_subject else speaker
    fig.suptitle(
        f"{speaker_label} /u/ candidates vs {reference_name}",
        fontsize=15,
        fontweight="bold",
        y=0.985,
    )
    fig.text(
        0.015,
        0.02,
        "Samples are sorted by image correlation to the VTLN reference.",
        fontsize=10,
        family="monospace",
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def write_csv(rows_by_speaker: dict[str, list[SampleRow]], output_path: Path) -> None:
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "speaker",
                "raw_subject",
                "display_rank",
                "session",
                "frame_index_1based",
                "source_sample_rank",
                "correlation_to_reference",
                "image_path",
            ]
        )
        for speaker in sorted(rows_by_speaker, key=speaker_sort_key):
            for display_rank, row in enumerate(rows_by_speaker[speaker], start=1):
                writer.writerow(
                    [
                        speaker,
                        RAW_SUBJECT_BY_SPEAKER.get(speaker, ""),
                        display_rank,
                        row.session,
                        row.frame_index_1based,
                        row.sample_rank,
                        f"{row.correlation:.6f}",
                        str(row.image_path),
                    ]
                )


def write_readme(
    rows_by_speaker: dict[str, list[SampleRow]],
    output_dir: Path,
    reference_name: str,
    reference_path: Path,
) -> None:
    lines = [
        "# /u/ Comparison Sheets",
        "",
        f"Reference image: `{reference_name}`",
        f"Reference path: `{reference_path}`",
        "",
        "Each PNG shows the VTLN reference on the left and 10 extracted `/u/` images from one encoded speaker.",
        "The 10 samples are sorted by Pearson correlation to the reference image.",
        "",
        "| Speaker | Raw subject | PNG | Top-3 sessions |",
        "| --- | --- | --- | --- |",
    ]
    for speaker in sorted(rows_by_speaker, key=speaker_sort_key):
        png_path = output_dir / f"{speaker}_u_vs_{reference_name}.png"
        top3 = ", ".join(
            f"{row.session}/f{row.frame_index_1based:04d} ({row.correlation:.3f})"
            for row in rows_by_speaker[speaker][:3]
        )
        lines.append(
            f"| {speaker} | {RAW_SUBJECT_BY_SPEAKER.get(speaker, '-')} | {png_path.name} | {top3} |"
        )
    (output_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    reference_path = find_reference_image(args.reference_name, args.vtln_dir)
    reference_image = load_grayscale(reference_path)

    summary_paths = sorted(
        args.samples_root.glob("P*/speaker_vowel_variants_summary.json"),
        key=lambda path: speaker_sort_key(path.parent.name),
    )
    if not summary_paths:
        raise FileNotFoundError(f"No speaker summaries found under {args.samples_root}")

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    rows_by_speaker: dict[str, list[SampleRow]] = {}
    for summary_path in summary_paths:
        rows = load_u_samples(summary_path, reference_image)
        if not rows:
            continue
        speaker = rows[0].speaker
        rows_by_speaker[speaker] = rows
        render_sheet(
            speaker=speaker,
            rows=rows,
            reference_name=args.reference_name,
            reference_image=reference_image,
            output_path=output_dir / f"{speaker}_u_vs_{args.reference_name}.png",
        )

    if not rows_by_speaker:
        raise ValueError("No /u/ samples were found in the provided summaries.")

    write_csv(rows_by_speaker, output_dir / "u_reference_scores.csv")
    write_readme(rows_by_speaker, output_dir, args.reference_name, reference_path)

    print(
        json.dumps(
            {
                "reference_image": str(reference_path),
                "output_dir": str(output_dir),
                "speaker_count": len(rows_by_speaker),
                "csv": str(output_dir / "u_reference_scores.csv"),
                "readme": str(output_dir / "README.md"),
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
