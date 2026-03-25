from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import imageio.v2 as imageio
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from grid_transform.artspeech_video import (
    evenly_spaced_indices,
    frame_index_for_time,
    load_session_data,
    normalize_frame,
    parse_textgrid,
)
from grid_transform.config import DEFAULT_OUTPUT_DIR, PROJECT_DIR


DEFAULT_VOWELS = ("a", "e", "i", "o", "u")


@dataclass
class VariantSelection:
    speaker: str
    vowel: str
    sample_rank: int
    session: str
    frame_index_0based: int
    frame_index_1based: int
    time_s: float
    interval_start_s: float
    interval_end_s: float
    dataset_root: str
    textgrid_path: str
    image_path: str


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Extract multiple occurrences per vowel for one ArtSpeech speaker, organize them "
            "into vowel folders, and build contact sheets to inspect within-speaker variability."
        )
    )
    parser.add_argument("--speaker", default="P7", help="Speaker id, for example P7.")
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=PROJECT_DIR.parent / "Data" / "Artspeech_database",
        help="Root directory containing ArtSpeech speaker folders.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Optional output directory. Defaults to outputs/speaker_vowel_variants/<speaker>_<N>_samples/.",
    )
    parser.add_argument(
        "--vowels",
        default=",".join(DEFAULT_VOWELS),
        help="Comma-separated vowel labels to extract. Defaults to a,e,i,o,u.",
    )
    parser.add_argument(
        "--samples-per-vowel",
        type=int,
        default=10,
        help="Maximum number of samples to save for each vowel. Defaults to 10.",
    )
    return parser.parse_args(argv)


def parse_vowels(text: str) -> list[str]:
    vowels = [value.strip() for value in text.split(",") if value.strip()]
    if not vowels:
        raise ValueError("No vowels were provided.")
    return vowels


def default_output_dir(speaker: str, samples_per_vowel: int) -> Path:
    return DEFAULT_OUTPUT_DIR / "speaker_vowel_variants" / f"{speaker}_{samples_per_vowel}_samples"


def dataset_root_from_textgrid(textgrid_path: Path) -> Path:
    return textgrid_path.parents[2]


def load_grayscale(path: Path) -> np.ndarray:
    image = imageio.imread(path)
    if image.ndim == 3:
        image = image[..., 0]
    return np.asarray(image, dtype=np.float32)


def speaker_dir(dataset_root: Path, speaker: str) -> Path:
    path = dataset_root / speaker
    if not path.exists():
        raise FileNotFoundError(f"Speaker directory not found: {path}")
    return path


def collect_occurrences(speaker_root: Path, vowels: list[str]) -> dict[str, list[dict[str, object]]]:
    occurrences: dict[str, list[dict[str, object]]] = {vowel: [] for vowel in vowels}
    for textgrid_path in sorted(speaker_root.rglob("TEXT_ALIGNMENT_*.textgrid")):
        tiers = parse_textgrid(textgrid_path)
        if len(tiers) < 2:
            continue
        session = textgrid_path.stem.split("_")[-1]
        for interval in tiers[1].intervals:
            label = interval.text.strip()
            if label not in occurrences:
                continue
            time_s = 0.5 * (interval.start + interval.end)
            occurrences[label].append(
                {
                    "session": session,
                    "time_s": float(time_s),
                    "interval_start_s": float(interval.start),
                    "interval_end_s": float(interval.end),
                    "textgrid_path": str(textgrid_path),
                    "dataset_root": str(dataset_root_from_textgrid(textgrid_path)),
                }
            )
    return occurrences


def select_occurrences(rows: list[dict[str, object]], max_samples: int) -> list[dict[str, object]]:
    if len(rows) <= max_samples:
        return list(rows)
    indices = evenly_spaced_indices(len(rows), max_samples)
    return [rows[index] for index in indices]


def extract_frame(session_data, frame_index_0based: int) -> np.ndarray:
    return normalize_frame(
        session_data.images[frame_index_0based],
        session_data.frame_min,
        session_data.frame_max,
    )


def compute_pairwise_correlations(image_paths: list[Path]) -> tuple[float, float, float]:
    stack = np.stack([load_grayscale(path) for path in image_paths], axis=0)
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
    if not correlations:
        return 1.0, 1.0, 1.0
    return float(np.mean(correlations)), float(np.min(correlations)), float(np.max(correlations))


def save_contact_sheet(speaker: str, vowel: str, rows: list[VariantSelection], output_path: Path) -> None:
    n_items = len(rows)
    ncols = min(5, max(1, n_items))
    nrows = int(np.ceil(n_items / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.2 * ncols, 3.4 * nrows), dpi=140)
    axes_arr = np.atleast_1d(axes).ravel()
    for ax in axes_arr:
        ax.set_axis_off()

    image_paths = [Path(row.image_path) for row in rows]
    corr_mean, corr_min, corr_max = compute_pairwise_correlations(image_paths)

    for ax, row in zip(axes_arr, rows):
        image = load_grayscale(Path(row.image_path))
        ax.imshow(image, cmap="gray", vmin=0, vmax=255)
        ax.set_title(
            f"{row.session} | frame {row.frame_index_1based}\nt={row.time_s:.3f}s",
            fontsize=9.5,
            fontweight="bold",
        )

    fig.suptitle(
        (
            f"{speaker} vowel /{vowel}/ variants ({len(rows)} samples)\n"
            f"pairwise corr mean={corr_mean:.3f} | min={corr_min:.3f} | max={corr_max:.3f}"
        ),
        fontsize=13,
        fontweight="bold",
        y=0.98,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    vowels = parse_vowels(args.vowels)
    output_dir = args.output_dir or default_output_dir(args.speaker, args.samples_per_vowel)
    output_dir.mkdir(parents=True, exist_ok=True)
    for vowel in vowels:
        (output_dir / vowel).mkdir(parents=True, exist_ok=True)
    contact_sheets_dir = output_dir / "contact_sheets"
    contact_sheets_dir.mkdir(parents=True, exist_ok=True)

    root = speaker_dir(args.dataset_root, args.speaker)
    occurrences = collect_occurrences(root, vowels)
    session_cache: dict[tuple[str, str], object] = {}
    selections: list[VariantSelection] = []
    variability_summary: dict[str, dict[str, object]] = {}

    for vowel in vowels:
        rows = select_occurrences(occurrences[vowel], args.samples_per_vowel)
        print(f"[extract] {args.speaker} vowel {vowel}: {len(rows)} / {len(occurrences[vowel])} samples")
        vowel_selections: list[VariantSelection] = []
        for rank, row in enumerate(rows, start=1):
            session = str(row["session"])
            dataset_root = Path(str(row["dataset_root"]))
            cache_key = (str(dataset_root), session)
            if cache_key not in session_cache:
                session_cache[cache_key] = load_session_data(dataset_root, args.speaker, session)
            session_data = session_cache[cache_key]
            time_s = float(row["time_s"])
            frame_index_0based = frame_index_for_time(time_s, session_data.frame_rate, session_data.images.shape[0])
            frame_index_1based = frame_index_0based + 1
            frame = extract_frame(session_data, frame_index_0based)
            image_path = output_dir / vowel / f"{rank:02d}_{args.speaker}_{session}_frame_{frame_index_1based:04d}.png"
            imageio.imwrite(image_path, frame)
            selection = VariantSelection(
                speaker=args.speaker,
                vowel=vowel,
                sample_rank=rank,
                session=session,
                frame_index_0based=frame_index_0based,
                frame_index_1based=frame_index_1based,
                time_s=time_s,
                interval_start_s=float(row["interval_start_s"]),
                interval_end_s=float(row["interval_end_s"]),
                dataset_root=str(dataset_root),
                textgrid_path=str(row["textgrid_path"]),
                image_path=str(image_path),
            )
            selections.append(selection)
            vowel_selections.append(selection)

        if vowel_selections:
            save_contact_sheet(
                args.speaker,
                vowel,
                vowel_selections,
                contact_sheets_dir / f"{args.speaker}_{vowel}_contact_sheet.png",
            )
            corr_mean, corr_min, corr_max = compute_pairwise_correlations([Path(row.image_path) for row in vowel_selections])
            variability_summary[vowel] = {
                "available_occurrences": len(occurrences[vowel]),
                "selected_samples": len(vowel_selections),
                "pairwise_correlation_mean": corr_mean,
                "pairwise_correlation_min": corr_min,
                "pairwise_correlation_max": corr_max,
            }
        else:
            variability_summary[vowel] = {
                "available_occurrences": len(occurrences[vowel]),
                "selected_samples": 0,
                "pairwise_correlation_mean": None,
                "pairwise_correlation_min": None,
                "pairwise_correlation_max": None,
            }

    summary = {
        "speaker": args.speaker,
        "dataset_root": str(args.dataset_root),
        "output_dir": str(output_dir),
        "vowels": vowels,
        "samples_per_vowel": args.samples_per_vowel,
        "variability_summary": variability_summary,
        "selections": [asdict(selection) for selection in selections],
    }
    summary_path = output_dir / "speaker_vowel_variants_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[done] saved variants in {output_dir}")
    print(f"[done] summary: {summary_path}")


if __name__ == "__main__":
    main()
