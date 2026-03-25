from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import imageio.v2 as imageio
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from grid_transform.config import DEFAULT_OUTPUT_DIR


@dataclass
class VowelStats:
    vowel: str
    speaker_count: int
    pairwise_correlation_mean: float
    pairwise_correlation_min: float
    pairwise_correlation_max: float
    mean_pixel_std: float
    max_pixel_std: float


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build one contact sheet per vowel from the extracted ArtSpeech samples and "
            "write a simple cross-speaker variability summary."
        )
    )
    parser.add_argument(
        "--samples-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR / "vowel_samples_aeiou",
        help="Directory containing the extracted vowel sample folders and selection_summary.json.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Optional output directory. Defaults to <samples-dir>/contact_sheets.",
    )
    return parser.parse_args(argv)


def speaker_sort_key(name: str) -> tuple[int, str]:
    suffix = name[1:] if name.startswith("P") else name
    try:
        return int(suffix), name
    except ValueError:
        return 10**9, name


def load_summary(summary_path: Path) -> dict:
    return json.loads(summary_path.read_text(encoding="utf-8"))


def group_by_vowel(summary: dict) -> dict[str, list[dict]]:
    grouped: dict[str, list[dict]] = {vowel: [] for vowel in summary["vowels"]}
    for row in summary["selections"]:
        grouped.setdefault(row["vowel"], []).append(row)
    for vowel, rows in grouped.items():
        rows.sort(key=lambda row: speaker_sort_key(row["speaker"]))
    return grouped


def load_grayscale(path: Path) -> np.ndarray:
    image = imageio.imread(path)
    if image.ndim == 3:
        image = image[..., 0]
    return np.asarray(image, dtype=np.float32)


def compute_vowel_stats(vowel: str, rows: list[dict]) -> VowelStats:
    stack = np.stack([load_grayscale(Path(row["image_path"])) for row in rows], axis=0)
    flat = stack.reshape(stack.shape[0], -1)

    correlations: list[float] = []
    for i in range(flat.shape[0]):
        for j in range(i + 1, flat.shape[0]):
            a = flat[i]
            b = flat[j]
            a_std = float(a.std())
            b_std = float(b.std())
            if a_std < 1e-8 or b_std < 1e-8:
                corr = 0.0
            else:
                corr = float(np.corrcoef(a, b)[0, 1])
            correlations.append(corr)

    pixel_std = stack.std(axis=0)
    return VowelStats(
        vowel=vowel,
        speaker_count=len(rows),
        pairwise_correlation_mean=float(np.mean(correlations)),
        pairwise_correlation_min=float(np.min(correlations)),
        pairwise_correlation_max=float(np.max(correlations)),
        mean_pixel_std=float(np.mean(pixel_std)),
        max_pixel_std=float(np.max(pixel_std)),
    )


def save_contact_sheet(vowel: str, rows: list[dict], stats: VowelStats, output_path: Path) -> None:
    n_items = len(rows)
    ncols = 5
    nrows = int(np.ceil(n_items / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(15, 3.2 * nrows), dpi=140)
    axes_arr = np.atleast_1d(axes).ravel()

    for ax in axes_arr:
        ax.set_axis_off()

    for ax, row in zip(axes_arr, rows):
        image = load_grayscale(Path(row["image_path"]))
        ax.imshow(image, cmap="gray", vmin=0, vmax=255)
        ax.set_title(
            f"{row['speaker']} | {row['session']}\nframe {row['frame_index_1based']}",
            fontsize=10,
            fontweight="bold",
        )

    fig.suptitle(
        (
            f"Vowel /{vowel}/ across speakers\n"
            f"pairwise corr mean={stats.pairwise_correlation_mean:.3f} | "
            f"min={stats.pairwise_correlation_min:.3f} | "
            f"pixel std mean={stats.mean_pixel_std:.2f}"
        ),
        fontsize=14,
        fontweight="bold",
        y=0.98,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def write_summary_files(stats: list[VowelStats], output_dir: Path) -> None:
    summary_json = {
        "notes": [
            "These variability scores are computed on raw midsagittal image frames.",
            "They reflect both articulation differences and speaker-specific anatomy/framing.",
            "To isolate articulation more cleanly, the frames should be aligned into a common space first.",
        ],
        "vowel_stats": [
            {
                "vowel": row.vowel,
                "speaker_count": row.speaker_count,
                "pairwise_correlation_mean": row.pairwise_correlation_mean,
                "pairwise_correlation_min": row.pairwise_correlation_min,
                "pairwise_correlation_max": row.pairwise_correlation_max,
                "mean_pixel_std": row.mean_pixel_std,
                "max_pixel_std": row.max_pixel_std,
            }
            for row in stats
        ],
    }
    (output_dir / "vowel_variability_summary.json").write_text(
        json.dumps(summary_json, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    lines = [
        "# Vowel Variability Summary",
        "",
        "These scores come from raw frame images, so they mix articulation differences with anatomy/framing differences.",
        "",
        "| vowel | speakers | pairwise corr mean | pairwise corr min | pairwise corr max | mean pixel std | max pixel std |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in stats:
        lines.append(
            f"| {row.vowel} | {row.speaker_count} | {row.pairwise_correlation_mean:.3f} | "
            f"{row.pairwise_correlation_min:.3f} | {row.pairwise_correlation_max:.3f} | "
            f"{row.mean_pixel_std:.2f} | {row.max_pixel_std:.2f} |"
        )
    (output_dir / "vowel_variability_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    samples_dir = args.samples_dir
    output_dir = args.output_dir or (samples_dir / "contact_sheets")
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = load_summary(samples_dir / "selection_summary.json")
    grouped = group_by_vowel(summary)

    stats: list[VowelStats] = []
    for vowel, rows in grouped.items():
        if not rows:
            continue
        vowel_stats = compute_vowel_stats(vowel, rows)
        stats.append(vowel_stats)
        save_contact_sheet(vowel, rows, vowel_stats, output_dir / f"{vowel}_contact_sheet.png")
        print(f"[sheet] {vowel}: {len(rows)} speakers")

    write_summary_files(stats, output_dir)
    print(f"[done] contact sheets: {output_dir}")


if __name__ == "__main__":
    main()
