from __future__ import annotations

import argparse
import json
from pathlib import Path

import imageio.v2 as imageio
import matplotlib
import numpy as np
from scipy import signal
from scipy.ndimage import shift as ndimage_shift

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from grid_transform.config import DEFAULT_OUTPUT_DIR


DEFAULT_VOWELS = ("a", "e", "i", "o", "u")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Align multiple same-vowel images from one ArtSpeech speaker, then export "
            "aligned contact sheets and summary statistics to inspect within-speaker variability."
        )
    )
    parser.add_argument("--speaker", default="P7", help="Speaker id, for example P7.")
    parser.add_argument(
        "--samples-per-vowel",
        type=int,
        default=10,
        help="Used only to infer the default samples directory name. Defaults to 10.",
    )
    parser.add_argument(
        "--samples-dir",
        type=Path,
        help="Directory produced by extract_speaker_vowel_variants.py.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Optional output directory. Defaults to <samples-dir>/aligned_analysis.",
    )
    parser.add_argument(
        "--vowels",
        default=",".join(DEFAULT_VOWELS),
        help="Comma-separated vowel labels to analyze. Defaults to a,e,i,o,u.",
    )
    parser.add_argument(
        "--max-shift",
        type=int,
        default=24,
        help="Maximum absolute translation in pixels during alignment. Defaults to 24.",
    )
    parser.add_argument(
        "--alignment-passes",
        type=int,
        default=2,
        help="Number of alignment passes. First to reference, then to running mean. Defaults to 2.",
    )
    return parser.parse_args(argv)


def parse_vowels(text: str) -> list[str]:
    vowels = [value.strip() for value in text.split(",") if value.strip()]
    if not vowels:
        raise ValueError("No vowels were provided.")
    return vowels


def default_samples_dir(speaker: str, samples_per_vowel: int) -> Path:
    return DEFAULT_OUTPUT_DIR / "speaker_vowel_variants" / f"{speaker}_{samples_per_vowel}_samples"


def load_summary(samples_dir: Path) -> dict[str, object]:
    summary_path = samples_dir / "speaker_vowel_variants_summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"Summary file not found: {summary_path}")
    return json.loads(summary_path.read_text(encoding="utf-8"))


def load_grayscale_image(path: Path) -> np.ndarray:
    image = imageio.imread(path)
    if image.ndim == 3:
        image = image[..., 0]
    return np.asarray(image, dtype=np.float32)


def save_grayscale_image(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    imageio.imwrite(path, np.clip(image, 0, 255).astype(np.uint8))


def corrcoef_1d(a: np.ndarray, b: np.ndarray) -> float:
    if a.size < 2 or b.size < 2:
        return 0.0
    a_std = float(a.std())
    b_std = float(b.std())
    if a_std < 1e-8 or b_std < 1e-8:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def compute_pairwise_correlations(
    images: list[np.ndarray],
    masks: list[np.ndarray] | None = None,
) -> tuple[float, float, float]:
    if len(images) <= 1:
        return 1.0, 1.0, 1.0

    correlations: list[float] = []
    for i in range(len(images)):
        for j in range(i + 1, len(images)):
            if masks is None:
                a = images[i].ravel()
                b = images[j].ravel()
            else:
                valid = masks[i] & masks[j]
                if int(valid.sum()) < 16:
                    correlations.append(0.0)
                    continue
                a = images[i][valid]
                b = images[j][valid]
            correlations.append(corrcoef_1d(a, b))

    return float(np.mean(correlations)), float(np.min(correlations)), float(np.max(correlations))


def choose_reference_index(images: list[np.ndarray]) -> int:
    if len(images) == 1:
        return 0
    scores = []
    for i, image in enumerate(images):
        corr_sum = 0.0
        corr_count = 0
        for j, other in enumerate(images):
            if i == j:
                continue
            corr_sum += corrcoef_1d(image.ravel(), other.ravel())
            corr_count += 1
        scores.append(corr_sum / max(corr_count, 1))
    return int(np.argmax(scores))


def estimate_translation(reference: np.ndarray, moving: np.ndarray, max_shift: int) -> tuple[float, float]:
    reference_zm = reference - float(reference.mean())
    moving_zm = moving - float(moving.mean())
    corr = signal.fftconvolve(moving_zm, reference_zm[::-1, ::-1], mode="full")

    center_y = reference.shape[0] - 1
    center_x = reference.shape[1] - 1
    y0 = max(0, center_y - max_shift)
    y1 = min(corr.shape[0], center_y + max_shift + 1)
    x0 = max(0, center_x - max_shift)
    x1 = min(corr.shape[1], center_x + max_shift + 1)
    local_corr = corr[y0:y1, x0:x1]

    peak_y, peak_x = np.unravel_index(np.argmax(local_corr), local_corr.shape)
    raw_shift_y = (peak_y + y0) - center_y
    raw_shift_x = (peak_x + x0) - center_x
    return float(-raw_shift_y), float(-raw_shift_x)


def apply_translation(image: np.ndarray, dy: float, dx: float, order: int) -> np.ndarray:
    return ndimage_shift(image, shift=(dy, dx), order=order, mode="constant", cval=0.0, prefilter=False)


def compute_mean_and_std(images: list[np.ndarray], masks: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    stack = np.stack(images, axis=0)
    mask_stack = np.stack(masks, axis=0).astype(np.float32)
    counts = mask_stack.sum(axis=0)

    mean = np.divide(
        (stack * mask_stack).sum(axis=0),
        counts,
        out=np.zeros_like(stack[0], dtype=np.float32),
        where=counts > 0,
    )
    variance = np.divide(
        (((stack - mean) ** 2) * mask_stack).sum(axis=0),
        counts,
        out=np.zeros_like(stack[0], dtype=np.float32),
        where=counts > 0,
    )
    return mean, np.sqrt(np.clip(variance, 0.0, None))


def compute_residual_metrics(
    entries: list[dict[str, object]],
    aligned_images: list[np.ndarray],
    aligned_masks: list[np.ndarray],
    mean_image: np.ndarray,
) -> list[dict[str, object]]:
    metrics: list[dict[str, object]] = []
    for entry, image, mask in zip(entries, aligned_images, aligned_masks):
        valid = mask.astype(bool)
        if int(valid.sum()) < 16:
            mae = 0.0
            rmse = 0.0
            corr = 0.0
        else:
            diff = image[valid] - mean_image[valid]
            mae = float(np.mean(np.abs(diff)))
            rmse = float(np.sqrt(np.mean(diff**2)))
            corr = corrcoef_1d(image[valid], mean_image[valid])
        metric = dict(entry)
        metric["mae_to_mean"] = mae
        metric["rmse_to_mean"] = rmse
        metric["corr_to_mean"] = corr
        metrics.append(metric)
    return metrics


def normalize_to_uint8(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image, dtype=np.float32)
    min_value = float(image.min())
    max_value = float(image.max())
    if max_value <= min_value + 1e-8:
        return np.zeros_like(image, dtype=np.uint8)
    scaled = (image - min_value) / (max_value - min_value)
    return np.clip(scaled * 255.0, 0, 255).astype(np.uint8)


def align_images(
    images: list[np.ndarray],
    max_shift: int,
    alignment_passes: int,
) -> tuple[list[np.ndarray], list[np.ndarray], list[dict[str, float]], int]:
    reference_index = choose_reference_index(images)
    template = images[reference_index]
    cumulative_shifts = [{"shift_y": 0.0, "shift_x": 0.0} for _ in images]
    aligned_images = [image.copy() for image in images]
    aligned_masks = [np.ones(image.shape, dtype=bool) for image in images]

    for _ in range(max(1, alignment_passes)):
        next_images: list[np.ndarray] = []
        next_masks: list[np.ndarray] = []
        for index, image in enumerate(aligned_images):
            if index == reference_index:
                aligned = image
                mask = np.ones(image.shape, dtype=bool)
                dy = 0.0
                dx = 0.0
            else:
                dy, dx = estimate_translation(template, image, max_shift=max_shift)
                aligned = apply_translation(image, dy, dx, order=1)
                mask = apply_translation(np.ones(image.shape, dtype=np.float32), dy, dx, order=0) > 0.5
                cumulative_shifts[index]["shift_y"] += dy
                cumulative_shifts[index]["shift_x"] += dx
            next_images.append(aligned)
            next_masks.append(mask)
        aligned_images = next_images
        aligned_masks = next_masks
        template, _ = compute_mean_and_std(aligned_images, aligned_masks)

    return aligned_images, aligned_masks, cumulative_shifts, reference_index


def save_contact_sheet(
    speaker: str,
    vowel: str,
    entries: list[dict[str, object]],
    output_path: Path,
    image_key: str,
    title_prefix: str,
    stats: dict[str, float],
    show_shift: bool,
) -> None:
    n_items = len(entries)
    ncols = min(5, max(1, n_items))
    nrows = int(np.ceil(n_items / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.2 * ncols, 3.5 * nrows), dpi=140)
    axes_arr = np.atleast_1d(axes).ravel()
    for ax in axes_arr:
        ax.set_axis_off()

    for ax, entry in zip(axes_arr, entries):
        image = load_grayscale_image(Path(str(entry[image_key])))
        ax.imshow(image, cmap="gray", vmin=0, vmax=255)
        if show_shift:
            shift_text = f"\ndy={entry['shift_y']:+.1f}, dx={entry['shift_x']:+.1f}"
        else:
            shift_text = ""
        ax.set_title(
            f"{entry['session']} | frame {entry['frame_index_1based']}{shift_text}",
            fontsize=9,
            fontweight="bold",
        )

    fig.suptitle(
        (
            f"{speaker} vowel /{vowel}/ {title_prefix}\n"
            f"pairwise corr mean={stats['mean']:.3f} | min={stats['min']:.3f} | max={stats['max']:.3f}"
        ),
        fontsize=13,
        fontweight="bold",
        y=0.98,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def save_mean_std_figure(
    speaker: str,
    vowel: str,
    mean_image: np.ndarray,
    std_image: np.ndarray,
    output_path: Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.6), dpi=160)
    axes[0].imshow(mean_image, cmap="gray", vmin=0, vmax=255)
    axes[0].set_title("Aligned mean image", fontweight="bold")
    axes[0].set_axis_off()

    axes[1].imshow(std_image, cmap="inferno")
    axes[1].set_title("Aligned std map", fontweight="bold")
    axes[1].set_axis_off()

    fig.suptitle(f"{speaker} vowel /{vowel}/ aligned variability summary", fontweight="bold")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def save_ranked_variants_figure(
    speaker: str,
    vowel: str,
    metrics: list[dict[str, object]],
    output_path: Path,
    top_k: int = 3,
) -> None:
    if not metrics:
        return

    ranked = sorted(metrics, key=lambda row: float(row["rmse_to_mean"]))
    closest = ranked[: min(top_k, len(ranked))]
    farthest = list(reversed(ranked[-min(top_k, len(ranked)) :]))
    groups = [("Closest To Mean", closest), ("Farthest From Mean", farthest)]

    ncols = max(len(group) for _, group in groups)
    fig, axes = plt.subplots(2, ncols, figsize=(3.3 * ncols, 6.8), dpi=150)
    axes_arr = np.asarray(axes, dtype=object).reshape(2, ncols)

    for row_axes in axes_arr:
        for ax in row_axes:
            ax.set_axis_off()

    for row_index, (label, group) in enumerate(groups):
        for col_index, metric in enumerate(group):
            ax = axes_arr[row_index, col_index]
            image = load_grayscale_image(Path(str(metric["aligned_image_path"])))
            ax.imshow(image, cmap="gray", vmin=0, vmax=255)
            ax.set_title(
                (
                    f"{label}\n"
                    f"{metric['session']} | frame {metric['frame_index_1based']}\n"
                    f"rmse={metric['rmse_to_mean']:.2f}, corr={metric['corr_to_mean']:.3f}"
                ),
                fontsize=9,
                fontweight="bold",
            )

    fig.suptitle(f"{speaker} vowel /{vowel}/ prototype vs outliers after alignment", fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def build_markdown(summary: dict[str, object]) -> str:
    lines = [
        f"# Within-Speaker Aligned Vowel Variability: {summary['speaker']}",
        "",
        f"- Samples directory: `{summary['samples_dir']}`",
        f"- Alignment output: `{summary['output_dir']}`",
        f"- Alignment method: translation-only intensity registration",
        f"- Max shift: `{summary['max_shift']}` px",
        f"- Alignment passes: `{summary['alignment_passes']}`",
        "",
        "| Vowel | Samples | Mean Corr Before | Mean Corr After | Gain | Mean Std | Mean Abs Shift | Prototype | Outlier |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |",
    ]

    for vowel, row in summary["aligned_variability_summary"].items():
        if row["selected_samples"] == 0:
            lines.append(f"| {vowel} | 0 | - | - | - | - |")
            continue
        lines.append(
            "| "
            + f"{vowel} | {row['selected_samples']} | "
            + f"{row['pairwise_correlation_before_mean']:.3f} | "
            + f"{row['pairwise_correlation_after_mean']:.3f} | "
            + f"{row['pairwise_correlation_gain_mean']:.3f} | "
            + f"{row['aligned_std_mean']:.2f} | "
            + f"{row['mean_abs_l1_shift_px']:.2f} | "
            + f"{row['prototype_session']}#{row['prototype_frame_index_1based']} | "
            + f"{row['outlier_session']}#{row['outlier_frame_index_1based']} |"
        )
    lines.append("")
    lines.append("Interpretation: higher correlation after alignment means some raw variability was due to positioning/framing; the remaining std map and prototype-vs-outlier spread better reflect repeat-to-repeat articulation differences.")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    vowels = parse_vowels(args.vowels)
    samples_dir = args.samples_dir or default_samples_dir(args.speaker, args.samples_per_vowel)
    summary = load_summary(samples_dir)
    speaker = str(summary["speaker"])
    output_dir = args.output_dir or (samples_dir / "aligned_analysis")
    aligned_images_dir = output_dir / "aligned_images"
    contact_sheets_dir = output_dir / "contact_sheets"
    rankings_dir = output_dir / "rankings"
    stats_dir = output_dir / "stats"
    output_dir.mkdir(parents=True, exist_ok=True)
    aligned_images_dir.mkdir(parents=True, exist_ok=True)
    contact_sheets_dir.mkdir(parents=True, exist_ok=True)
    rankings_dir.mkdir(parents=True, exist_ok=True)
    stats_dir.mkdir(parents=True, exist_ok=True)

    entries_by_vowel: dict[str, list[dict[str, object]]] = {vowel: [] for vowel in vowels}
    for row in summary["selections"]:
        vowel = str(row["vowel"])
        if vowel in entries_by_vowel:
            entries_by_vowel[vowel].append(dict(row))

    aligned_summary: dict[str, dict[str, object]] = {}

    for vowel in vowels:
        entries = sorted(entries_by_vowel[vowel], key=lambda row: int(row["sample_rank"]))
        print(f"[align] {speaker} vowel {vowel}: {len(entries)} samples")
        if not entries:
            aligned_summary[vowel] = {
                "selected_samples": 0,
                "pairwise_correlation_before_mean": None,
                "pairwise_correlation_before_min": None,
                "pairwise_correlation_before_max": None,
                "pairwise_correlation_after_mean": None,
                "pairwise_correlation_after_min": None,
                "pairwise_correlation_after_max": None,
                "pairwise_correlation_gain_mean": None,
                "aligned_std_mean": None,
                "aligned_std_max": None,
                "reference_sample_rank": None,
            }
            continue

        raw_images = [load_grayscale_image(Path(str(entry["image_path"]))) for entry in entries]
        before_mean, before_min, before_max = compute_pairwise_correlations(raw_images)

        aligned_images, aligned_masks, shifts, reference_index = align_images(
            raw_images,
            max_shift=args.max_shift,
            alignment_passes=args.alignment_passes,
        )
        after_mean, after_min, after_max = compute_pairwise_correlations(aligned_images, aligned_masks)
        mean_image, std_image = compute_mean_and_std(aligned_images, aligned_masks)

        aligned_entries: list[dict[str, object]] = []
        for entry, aligned_image, shift_values in zip(entries, aligned_images, shifts):
            aligned_path = aligned_images_dir / vowel / Path(str(entry["image_path"])).name
            save_grayscale_image(aligned_path, aligned_image)
            updated = dict(entry)
            updated["aligned_image_path"] = str(aligned_path)
            updated["shift_y"] = float(shift_values["shift_y"])
            updated["shift_x"] = float(shift_values["shift_x"])
            aligned_entries.append(updated)

        shift_magnitudes = [abs(entry["shift_y"]) + abs(entry["shift_x"]) for entry in aligned_entries]
        residual_metrics = compute_residual_metrics(aligned_entries, aligned_images, aligned_masks, mean_image)
        ranked_metrics = sorted(residual_metrics, key=lambda row: float(row["rmse_to_mean"]))
        prototype = ranked_metrics[0]
        outlier = ranked_metrics[-1]

        before_stats = {"mean": before_mean, "min": before_min, "max": before_max}
        after_stats = {"mean": after_mean, "min": after_min, "max": after_max}

        save_contact_sheet(
            speaker=speaker,
            vowel=vowel,
            entries=aligned_entries,
            output_path=contact_sheets_dir / f"{speaker}_{vowel}_raw_contact_sheet.png",
            image_key="image_path",
            title_prefix="raw variants",
            stats=before_stats,
            show_shift=False,
        )
        save_contact_sheet(
            speaker=speaker,
            vowel=vowel,
            entries=aligned_entries,
            output_path=contact_sheets_dir / f"{speaker}_{vowel}_aligned_contact_sheet.png",
            image_key="aligned_image_path",
            title_prefix="aligned variants",
            stats=after_stats,
            show_shift=True,
        )
        save_mean_std_figure(
            speaker=speaker,
            vowel=vowel,
            mean_image=np.clip(mean_image, 0, 255).astype(np.uint8),
            std_image=std_image,
            output_path=stats_dir / f"{speaker}_{vowel}_aligned_mean_std.png",
        )
        save_ranked_variants_figure(
            speaker=speaker,
            vowel=vowel,
            metrics=residual_metrics,
            output_path=rankings_dir / f"{speaker}_{vowel}_prototype_vs_outliers.png",
        )
        save_grayscale_image(stats_dir / f"{speaker}_{vowel}_aligned_mean.png", mean_image)
        save_grayscale_image(stats_dir / f"{speaker}_{vowel}_aligned_std.png", normalize_to_uint8(std_image))

        aligned_summary[vowel] = {
            "selected_samples": len(entries),
            "pairwise_correlation_before_mean": before_mean,
            "pairwise_correlation_before_min": before_min,
            "pairwise_correlation_before_max": before_max,
            "pairwise_correlation_after_mean": after_mean,
            "pairwise_correlation_after_min": after_min,
            "pairwise_correlation_after_max": after_max,
            "pairwise_correlation_gain_mean": after_mean - before_mean,
            "aligned_std_mean": float(std_image.mean()),
            "aligned_std_max": float(std_image.max()),
            "mean_abs_l1_shift_px": float(np.mean(shift_magnitudes)),
            "max_abs_l1_shift_px": float(np.max(shift_magnitudes)),
            "reference_sample_rank": int(entries[reference_index]["sample_rank"]),
            "reference_image_path": str(entries[reference_index]["image_path"]),
            "prototype_sample_rank": int(prototype["sample_rank"]),
            "prototype_session": str(prototype["session"]),
            "prototype_frame_index_1based": int(prototype["frame_index_1based"]),
            "prototype_rmse_to_mean": float(prototype["rmse_to_mean"]),
            "prototype_corr_to_mean": float(prototype["corr_to_mean"]),
            "prototype_image_path": str(prototype["aligned_image_path"]),
            "outlier_sample_rank": int(outlier["sample_rank"]),
            "outlier_session": str(outlier["session"]),
            "outlier_frame_index_1based": int(outlier["frame_index_1based"]),
            "outlier_rmse_to_mean": float(outlier["rmse_to_mean"]),
            "outlier_corr_to_mean": float(outlier["corr_to_mean"]),
            "outlier_image_path": str(outlier["aligned_image_path"]),
            "aligned_entries": aligned_entries,
            "residual_metrics": residual_metrics,
        }

    output_summary = {
        "speaker": speaker,
        "samples_dir": str(samples_dir),
        "output_dir": str(output_dir),
        "vowels": vowels,
        "max_shift": args.max_shift,
        "alignment_passes": args.alignment_passes,
        "alignment_method": "translation_only_intensity_registration",
        "aligned_variability_summary": aligned_summary,
    }
    summary_path = output_dir / "aligned_variability_summary.json"
    markdown_path = output_dir / "aligned_variability_summary.md"
    summary_path.write_text(json.dumps(output_summary, indent=2, ensure_ascii=False), encoding="utf-8")
    markdown_path.write_text(build_markdown(output_summary), encoding="utf-8")
    print(f"[done] aligned variability summary: {summary_path}")
    print(f"[done] markdown summary: {markdown_path}")


if __name__ == "__main__":
    main()
