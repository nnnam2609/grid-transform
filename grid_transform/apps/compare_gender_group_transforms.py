from __future__ import annotations

import argparse
import csv
import re
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from grid_transform.apps.average_speaker_grid_transform import choose_label_colors
from grid_transform.apps.method4_transform import format_target_frame, resample_polyline
from grid_transform.config import DEFAULT_OUTPUT_DIR, TONGUE_COLOR
from grid_transform.io import load_frame_vtln
from grid_transform.transfer import build_two_step_transform, smooth_transformed_contours, transform_contours
from grid_transform.vt import build_grid
from grid_transform.warp import warp_image_to_target_space


CURATED_SPEAKER_DIRNAME = "data"
CURATED_SPEAKER_GENDER = {
    "P1": "male",
    "P2": "male",
    "P3": "male",
    "P4": "female",
    "P5": "male",
    "P6": "male",
    "P7": "female",
    "P8": "female",
    "P9": "female",
    "P10": "female",
}
IMAGE_SUFFIXES = (".png", ".tif", ".tiff")
CURATED_BASENAME_RE = re.compile(
    r"^(?P<raw_subject>\d+)_(?P<speaker>P\d+)_(?P<session>S\d+)_F(?P<frame>\d+)$",
    re.IGNORECASE,
)
DEFAULT_N_VERT = 9
DEFAULT_N_POINTS = 250
DEFAULT_TEMPLATE_POINTS = 80


@dataclass(frozen=True)
class CuratedSpeakerSpec:
    basename: str
    speaker_id: str
    raw_subject: str | None
    session: str | None
    frame: int | None
    gender: str
    image_path: Path | None
    zip_path: Path | None


@dataclass
class LoadedSpeaker:
    spec: CuratedSpeakerSpec
    image: object
    contours: dict[str, np.ndarray]
    grid: object


@dataclass(frozen=True)
class PairTransformResult:
    cohort: str
    source_basename: str
    target_basename: str
    source_speaker_id: str
    target_speaker_id: str
    source_gender: str
    target_gender: str
    rms: float
    per_label_rms: dict[str, float]
    mapped_contours: dict[str, np.ndarray]
    warped_source_image: np.ndarray | None = None


@dataclass(frozen=True)
class SourceSelectionSummary:
    source_basename: str
    source_speaker_id: str
    source_gender: str
    same_gender_mean_rms: float
    all_target_mean_rms: float

    @property
    def improvement(self) -> float:
        return self.all_target_mean_rms - self.same_gender_mean_rms


def default_curated_vtln_dir() -> Path:
    return DEFAULT_OUTPUT_DIR.parent / "VTLN" / CURATED_SPEAKER_DIRNAME


def default_output_dir() -> Path:
    return DEFAULT_OUTPUT_DIR / "compare_males_and_females"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare same-sex grid transforms across the curated ArtSpeech/VTLN speaker set. "
            "The experiment generates all ordered source->target pairs with source!=target "
            "for male->male and female->female."
        )
    )
    parser.add_argument(
        "--vtln-dir",
        type=Path,
        default=default_curated_vtln_dir(),
        help=f"Canonical curated VTLN speaker folder. Defaults to VTLN/{CURATED_SPEAKER_DIRNAME}.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=default_output_dir(),
        help="Directory where the experiment PNG outputs will be written.",
    )
    parser.add_argument(
        "--nc-template",
        type=int,
        default=DEFAULT_TEMPLATE_POINTS,
        help="Resample count used for pointwise RMS metrics.",
    )
    return parser.parse_args(argv)


def parse_curated_basename(basename: str) -> tuple[str | None, str | None, str | None, int | None]:
    match = CURATED_BASENAME_RE.match(basename)
    if not match:
        return None, None, None, None
    return (
        match.group("speaker").upper(),
        match.group("raw_subject"),
        match.group("session").upper(),
        int(match.group("frame")),
    )


def resolve_image_path(vtln_dir: Path, basename: str) -> Path | None:
    for ext in IMAGE_SUFFIXES:
        candidate = vtln_dir / f"{basename}{ext}"
        if candidate.is_file():
            return candidate
    return None


def read_curated_specs(vtln_dir: Path) -> list[CuratedSpeakerSpec]:
    manifest_path = vtln_dir / "selection_manifest.csv"
    specs: list[CuratedSpeakerSpec] = []

    if manifest_path.is_file():
        with manifest_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                basename = row["output_basename"].strip()
                speaker_id = row["speaker"].strip().upper()
                raw_subject = row.get("raw_subject", "").strip() or None
                selected_source = row.get("selected_source", "").strip()
                session = None
                frame = None
                if selected_source:
                    parts = selected_source.split("/")
                    if len(parts) == 3:
                        session = parts[1].strip().upper()
                        frame_text = parts[2].strip().upper()
                        if frame_text.startswith("F") and frame_text[1:].isdigit():
                            frame = int(frame_text[1:])
                image_path = resolve_image_path(vtln_dir, basename)
                zip_path = vtln_dir / f"{basename}.zip"
                specs.append(
                    CuratedSpeakerSpec(
                        basename=basename,
                        speaker_id=speaker_id,
                        raw_subject=raw_subject,
                        session=session,
                        frame=frame,
                        gender=CURATED_SPEAKER_GENDER[speaker_id],
                        image_path=image_path,
                        zip_path=zip_path if zip_path.is_file() else None,
                    )
                )
        return sorted(specs, key=lambda item: item.basename)

    stems = sorted(
        {
            path.stem
            for ext in IMAGE_SUFFIXES + (".zip",)
            for path in vtln_dir.glob(f"*{ext}")
        }
    )
    for stem in stems:
        speaker_id, raw_subject, session, frame = parse_curated_basename(stem)
        if speaker_id is None or speaker_id not in CURATED_SPEAKER_GENDER:
            continue
        zip_path = vtln_dir / f"{stem}.zip"
        specs.append(
            CuratedSpeakerSpec(
                basename=stem,
                speaker_id=speaker_id,
                raw_subject=raw_subject,
                session=session,
                frame=frame,
                gender=CURATED_SPEAKER_GENDER[speaker_id],
                image_path=resolve_image_path(vtln_dir, stem),
                zip_path=zip_path if zip_path.is_file() else None,
            )
        )
    return specs


def load_available_speakers(specs: list[CuratedSpeakerSpec], vtln_dir: Path) -> tuple[dict[str, LoadedSpeaker], list[str]]:
    loaded: dict[str, LoadedSpeaker] = {}
    skipped: list[str] = []

    for spec in specs:
        if spec.image_path is None:
            skipped.append(f"{spec.basename}: missing image")
            continue
        if spec.zip_path is None:
            skipped.append(f"{spec.basename}: missing ROI zip")
            continue

        try:
            image, contours = load_frame_vtln(spec.basename, vtln_dir)
            grid = build_grid(
                image,
                contours,
                n_vert=DEFAULT_N_VERT,
                n_points=DEFAULT_N_POINTS,
                frame_number=0,
            )
        except Exception as exc:
            skipped.append(f"{spec.basename}: {type(exc).__name__}: {exc}")
            continue

        loaded[spec.basename] = LoadedSpeaker(
            spec=spec,
            image=image,
            contours=contours,
            grid=grid,
        )

    return loaded, skipped


def compute_global_common_labels(loaded_speakers: dict[str, LoadedSpeaker]) -> list[str]:
    if not loaded_speakers:
        raise ValueError("No speakers were loaded successfully.")
    label_sets = [set(speaker.contours.keys()) for speaker in loaded_speakers.values()]
    labels = sorted(set.intersection(*label_sets))
    if not labels:
        raise ValueError("No common contour labels were shared across the available speakers.")
    return labels


def speaker_id_sort_key(speaker_id: str) -> tuple[int, str]:
    match = re.fullmatch(r"P(\d+)", speaker_id, re.IGNORECASE)
    if match is None:
        return (10_000, speaker_id)
    return (int(match.group(1)), speaker_id)


def plot_contours(ax, contours: dict[str, np.ndarray], labels: list[str], colors: dict[str, object], *, lw: float, alpha: float, linestyle: str = "-") -> None:
    for label in labels:
        pts = np.asarray(contours[label], dtype=float)
        ax.plot(
            pts[:, 0],
            pts[:, 1],
            color=colors.get(label, TONGUE_COLOR),
            lw=lw,
            alpha=alpha,
            linestyle=linestyle,
        )


def stack_resampled_contours(contours: dict[str, np.ndarray], labels: list[str], nc_template: int) -> np.ndarray:
    return np.vstack([resample_polyline(contours[label], nc_template) for label in labels])


def compute_pair_transform(
    cohort: str,
    source: LoadedSpeaker,
    target: LoadedSpeaker,
    common_labels: list[str],
    nc_template: int,
    *,
    include_warped_image: bool = False,
) -> PairTransformResult:
    forward_transform = build_two_step_transform(source.grid, target.grid)
    mapped_contours = transform_contours(source.contours, forward_transform["apply_two_step"], common_labels)
    mapped_contours = smooth_transformed_contours(mapped_contours)

    source_shape = stack_resampled_contours(mapped_contours, common_labels, nc_template)
    target_shape = stack_resampled_contours(target.contours, common_labels, nc_template)
    rms = float(np.sqrt(np.mean((source_shape - target_shape) ** 2)))

    per_label_rms = {}
    for label in common_labels:
        src_pts = resample_polyline(mapped_contours[label], nc_template)
        tgt_pts = resample_polyline(target.contours[label], nc_template)
        per_label_rms[label] = float(np.sqrt(np.mean((src_pts - tgt_pts) ** 2)))

    warped_source_image = None
    if include_warped_image:
        inverse_transform = build_two_step_transform(target.grid, source.grid)
        warped_source_image, _ = warp_image_to_target_space(
            source.image,
            np.asarray(target.image).shape,
            inverse_transform["apply_two_step"],
        )

    return PairTransformResult(
        cohort=cohort,
        source_basename=source.spec.basename,
        target_basename=target.spec.basename,
        source_speaker_id=source.spec.speaker_id,
        target_speaker_id=target.spec.speaker_id,
        source_gender=source.spec.gender,
        target_gender=target.spec.gender,
        rms=rms,
        per_label_rms=per_label_rms,
        mapped_contours=mapped_contours,
        warped_source_image=warped_source_image,
    )


def save_pair_figure(
    result: PairTransformResult,
    source: LoadedSpeaker,
    target: LoadedSpeaker,
    common_labels: list[str],
    colors: dict[str, object],
    output_path: Path,
) -> None:
    if result.warped_source_image is None:
        raise ValueError("Pair figure rendering requires a warped source image.")

    fig, axes = plt.subplots(2, 2, figsize=(15.5, 13))
    axes = axes.ravel()

    format_target_frame(axes[0], source.image, "Source image + contours")
    plot_contours(axes[0], source.contours, common_labels, colors, lw=2.0, alpha=0.95)

    format_target_frame(axes[1], target.image, "Target image + contours")
    plot_contours(axes[1], target.contours, common_labels, colors, lw=2.0, alpha=0.95)

    format_target_frame(axes[2], target.image, "Mapped contours in target space")
    plot_contours(axes[2], target.contours, common_labels, colors, lw=1.6, alpha=0.35)
    plot_contours(axes[2], result.mapped_contours, common_labels, colors, lw=2.4, alpha=0.98, linestyle="--")
    axes[2].text(
        0.02,
        0.02,
        "\n".join(
            [
                "solid = target contours",
                "dashed = mapped source contours",
                f"shared labels = {len(common_labels)}",
            ]
        ),
        transform=axes[2].transAxes,
        ha="left",
        va="bottom",
        fontsize=9.5,
        bbox=dict(boxstyle="round,pad=0.35", fc="white", alpha=0.94),
    )

    format_target_frame(axes[3], result.warped_source_image, "Warped full source image")
    axes[3].text(
        0.02,
        0.02,
        "\n".join(
            [
                "full source image warped",
                "into target geometry",
                "center channel G=t",
            ]
        ),
        transform=axes[3].transAxes,
        ha="left",
        va="bottom",
        fontsize=9.5,
        bbox=dict(boxstyle="round,pad=0.35", fc="white", alpha=0.94),
    )

    fig.text(
        0.5,
        0.985,
        result.cohort.replace("_", " "),
        ha="center",
        va="top",
        fontsize=16,
        fontweight="bold",
    )
    fig.text(
        0.5,
        0.955,
        (
            f"{result.source_basename} -> {result.target_basename} | "
            f"{result.source_speaker_id} -> {result.target_speaker_id} | "
            f"RMS = {result.rms:.2f} px"
        ),
        ha="center",
        va="top",
        fontsize=11.5,
        fontweight="bold",
    )

    plt.tight_layout(rect=(0, 0, 1, 0.92))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_target_overview_figure(
    cohort: str,
    target: LoadedSpeaker,
    pair_results: list[PairTransformResult],
    common_labels: list[str],
    colors: dict[str, object],
    output_path: Path,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(22, 7.5))

    ax = axes[0]
    format_target_frame(ax, target.image, f"{cohort} target\n{target.spec.basename}")
    plot_contours(ax, target.contours, common_labels, colors, lw=2.2, alpha=0.7)
    for pair in pair_results:
        plot_contours(ax, pair.mapped_contours, common_labels, colors, lw=1.3, alpha=0.18, linestyle="--")
    ax.text(
        0.02,
        0.02,
        f"mapped sources: {len(pair_results)}",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=10,
        bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.94),
    )

    ax = axes[1]
    source_labels = [pair.source_speaker_id for pair in pair_results]
    source_values = [pair.rms for pair in pair_results]
    y_pos = np.arange(len(pair_results))
    ax.barh(y_pos, source_values, color="#4c78a8", edgecolor="black", alpha=0.82)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(source_labels, fontsize=9)
    ax.set_xlabel("Pair RMS (px)")
    ax.set_title("Mapped source -> target RMS", fontsize=13, fontweight="bold")
    ax.invert_yaxis()
    ax.grid(True, axis="x", alpha=0.3)
    for yi, val in zip(y_pos, source_values):
        ax.text(val + 0.12, yi, f"{val:.2f}", va="center", fontsize=8.5)

    ax = axes[2]
    label_means = []
    for label in common_labels:
        label_means.append(float(np.mean([pair.per_label_rms[label] for pair in pair_results])))
    order = np.argsort(label_means)[::-1]
    ordered_labels = [common_labels[idx] for idx in order]
    ordered_values = [label_means[idx] for idx in order]
    bar_colors = [colors[label] for label in ordered_labels]
    y_pos = np.arange(len(ordered_labels))
    ax.barh(y_pos, ordered_values, color=bar_colors, edgecolor="black", alpha=0.85)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(ordered_labels, fontsize=9)
    ax.set_xlabel("Mean label RMS (px)")
    ax.set_title("Per-label mean RMS to target", fontsize=13, fontweight="bold")
    ax.invert_yaxis()
    ax.grid(True, axis="x", alpha=0.3)

    pair_mean = float(np.mean([pair.rms for pair in pair_results]))
    pair_median = float(np.median([pair.rms for pair in pair_results]))
    fig.suptitle(
        f"{cohort}: all same-sex sources mapped to {target.spec.basename} | mean={pair_mean:.2f} px | median={pair_median:.2f} px",
        fontsize=14,
        fontweight="bold",
    )
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def grouped_pair_results_by_target(pair_results: list[PairTransformResult]) -> dict[str, list[PairTransformResult]]:
    grouped: dict[str, list[PairTransformResult]] = {}
    for pair in pair_results:
        grouped.setdefault(pair.target_basename, []).append(pair)
    return grouped


def grouped_pair_results_by_source(pair_results: list[PairTransformResult]) -> dict[str, list[PairTransformResult]]:
    grouped: dict[str, list[PairTransformResult]] = {}
    for pair in pair_results:
        grouped.setdefault(pair.source_basename, []).append(pair)
    return grouped


def cohort_target_mean_rms(pair_results: list[PairTransformResult]) -> dict[str, float]:
    grouped = grouped_pair_results_by_target(pair_results)
    return {
        target_id: float(np.mean([pair.rms for pair in target_pairs]))
        for target_id, target_pairs in grouped.items()
    }


def cohort_label_mean_rms(pair_results: list[PairTransformResult], common_labels: list[str]) -> dict[str, float]:
    return {
        label: float(np.mean([pair.per_label_rms[label] for pair in pair_results]))
        for label in common_labels
    }


def compute_source_selection_summaries(
    same_gender_results: list[PairTransformResult],
    all_target_results: list[PairTransformResult],
    loaded_speakers: dict[str, LoadedSpeaker],
) -> list[SourceSelectionSummary]:
    same_by_source = grouped_pair_results_by_source(same_gender_results)
    all_by_source = grouped_pair_results_by_source(all_target_results)

    summaries: list[SourceSelectionSummary] = []
    for source_basename in sorted(
        loaded_speakers,
        key=lambda name: speaker_id_sort_key(loaded_speakers[name].spec.speaker_id),
    ):
        if source_basename not in same_by_source:
            raise ValueError(f"Missing same-gender results for source {source_basename}")
        if source_basename not in all_by_source:
            raise ValueError(f"Missing all-target results for source {source_basename}")

        source_spec = loaded_speakers[source_basename].spec
        same_gender_mean_rms = float(np.mean([pair.rms for pair in same_by_source[source_basename]]))
        all_target_mean_rms = float(np.mean([pair.rms for pair in all_by_source[source_basename]]))
        summaries.append(
            SourceSelectionSummary(
                source_basename=source_basename,
                source_speaker_id=source_spec.speaker_id,
                source_gender=source_spec.gender,
                same_gender_mean_rms=same_gender_mean_rms,
                all_target_mean_rms=all_target_mean_rms,
            )
        )

    return summaries


def save_same_gender_vs_all_targets_summary(
    source_summaries: list[SourceSelectionSummary],
    common_labels: list[str],
    output_path: Path,
) -> None:
    ordered = sorted(
        source_summaries,
        key=lambda item: (
            0 if item.source_gender == "male" else 1,
            speaker_id_sort_key(item.source_speaker_id),
        ),
    )
    speaker_ids = [item.source_speaker_id for item in ordered]
    same_gender_means = [item.same_gender_mean_rms for item in ordered]
    all_target_means = [item.all_target_mean_rms for item in ordered]
    improvements = [item.improvement for item in ordered]
    source_colors = ["#4c78a8" if item.source_gender == "male" else "#f58518" for item in ordered]

    male_summaries = [item for item in ordered if item.source_gender == "male"]
    female_summaries = [item for item in ordered if item.source_gender == "female"]
    male_improvements = [item.improvement for item in male_summaries]
    female_improvements = [item.improvement for item in female_summaries]

    fig, axes = plt.subplots(2, 2, figsize=(18, 12))

    ax = axes[0, 0]
    x = np.arange(len(ordered))
    width = 0.38
    ax.bar(x - width / 2, same_gender_means, width=width, color="#16a085", edgecolor="black", alpha=0.88, label="same-gender")
    ax.bar(x + width / 2, all_target_means, width=width, color="#7f8c8d", edgecolor="black", alpha=0.86, label="all-targets")
    ax.set_xticks(x)
    ax.set_xticklabels(speaker_ids, rotation=25, ha="right")
    ax.set_ylabel("Mean RMS per source (px)")
    ax.set_title("Per-source mean RMS", fontsize=13, fontweight="bold")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(framealpha=0.95)
    for tick, color in zip(ax.get_xticklabels(), source_colors):
        tick.set_color(color)

    ax = axes[0, 1]
    ax.bar(x, improvements, color=source_colors, edgecolor="black", alpha=0.86)
    ax.axhline(0.0, color="black", lw=1.0, alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(speaker_ids, rotation=25, ha="right")
    ax.set_ylabel("Improvement (px)")
    ax.set_title("All-target mean - same-gender mean", fontsize=13, fontweight="bold")
    ax.grid(True, axis="y", alpha=0.25)
    for tick, color in zip(ax.get_xticklabels(), source_colors):
        tick.set_color(color)

    ax = axes[1, 0]
    cohort_names = ["male sources", "female sources"]
    cohort_same = [
        float(np.mean([item.same_gender_mean_rms for item in male_summaries])),
        float(np.mean([item.same_gender_mean_rms for item in female_summaries])),
    ]
    cohort_all = [
        float(np.mean([item.all_target_mean_rms for item in male_summaries])),
        float(np.mean([item.all_target_mean_rms for item in female_summaries])),
    ]
    cohort_x = np.arange(len(cohort_names))
    ax.bar(cohort_x - width / 2, cohort_same, width=width, color="#16a085", edgecolor="black", alpha=0.88, label="same-gender")
    ax.bar(cohort_x + width / 2, cohort_all, width=width, color="#7f8c8d", edgecolor="black", alpha=0.86, label="all-targets")
    ax.set_xticks(cohort_x)
    ax.set_xticklabels(cohort_names)
    ax.set_ylabel("Mean RMS (px)")
    ax.set_title("Cohort mean RMS by target policy", fontsize=13, fontweight="bold")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(framealpha=0.95)

    ax = axes[1, 1]
    ax.boxplot([male_improvements, female_improvements], labels=["male sources", "female sources"])
    ax.scatter(np.full(len(male_improvements), 1.0), male_improvements, color="#4c78a8", alpha=0.82, zorder=4)
    ax.scatter(np.full(len(female_improvements), 2.0), female_improvements, color="#f58518", alpha=0.82, zorder=4)
    ax.axhline(0.0, color="black", lw=1.0, alpha=0.7)
    ax.set_ylabel("Improvement (px)")
    ax.set_title("Improvement distribution by source gender", fontsize=13, fontweight="bold")
    ax.grid(True, axis="y", alpha=0.25)
    ax.text(
        0.02,
        0.02,
        "\n".join(
            [
                f"male mean/median: {np.mean(male_improvements):.2f} / {np.median(male_improvements):.2f} px",
                f"female mean/median: {np.mean(female_improvements):.2f} / {np.median(female_improvements):.2f} px",
                f"all sources mean: {np.mean(improvements):.2f} px",
                f"all sources median: {np.median(improvements):.2f} px",
            ]
        ),
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=9.5,
        bbox=dict(boxstyle="round,pad=0.35", fc="white", alpha=0.96),
    )

    fig.suptitle("Same-gender target selection vs all-target baseline", fontsize=16, fontweight="bold")
    fig.text(
        0.02,
        0.01,
        "\n".join(
            [
                "Baseline policy: exhaustive mean over every valid non-self target.",
                "Improvement = all-target mean RMS - same-gender mean RMS. Positive means same-gender is better.",
                f"Common labels ({len(common_labels)}): {', '.join(common_labels)}",
            ]
        ),
        ha="left",
        va="bottom",
        fontsize=9.5,
        bbox=dict(boxstyle="round,pad=0.35", fc="white", alpha=0.96),
    )

    plt.tight_layout(rect=(0, 0.08, 1, 0.96))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_cross_cohort_summary(
    male_results: list[PairTransformResult],
    female_results: list[PairTransformResult],
    common_labels: list[str],
    loaded_speakers: dict[str, LoadedSpeaker],
    skipped_messages: list[str],
    output_path: Path,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))

    male_rms = [pair.rms for pair in male_results]
    female_rms = [pair.rms for pair in female_results]

    ax = axes[0, 0]
    bins = max(4, min(10, len(male_rms) + len(female_rms)))
    ax.hist(male_rms, bins=bins, alpha=0.68, color="#4c78a8", label="male->male")
    ax.hist(female_rms, bins=bins, alpha=0.68, color="#f58518", label="female->female")
    ax.set_xlabel("Pair RMS (px)")
    ax.set_ylabel("Count")
    ax.set_title("Pairwise RMS distribution", fontsize=13, fontweight="bold")
    ax.grid(True, alpha=0.25)
    ax.legend(framealpha=0.95)

    ax = axes[0, 1]
    ax.boxplot([male_rms, female_rms], labels=["male->male", "female->female"])
    ax.set_ylabel("Pair RMS (px)")
    ax.set_title("Pairwise RMS summary", fontsize=13, fontweight="bold")
    ax.grid(True, axis="y", alpha=0.25)

    ax = axes[1, 0]
    male_target_means = cohort_target_mean_rms(male_results)
    female_target_means = cohort_target_mean_rms(female_results)
    names = list(male_target_means) + list(female_target_means)
    values = list(male_target_means.values()) + list(female_target_means.values())
    colors = ["#4c78a8"] * len(male_target_means) + ["#f58518"] * len(female_target_means)
    x = np.arange(len(names))
    ax.bar(x, values, color=colors, edgecolor="black", alpha=0.84)
    ax.set_xticks(x)
    ax.set_xticklabels([loaded_speakers[name].spec.speaker_id for name in names], rotation=25, ha="right")
    ax.set_ylabel("Mean target RMS (px)")
    ax.set_title("Target-wise mean RMS", fontsize=13, fontweight="bold")
    ax.grid(True, axis="y", alpha=0.25)

    ax = axes[1, 1]
    male_label_means = cohort_label_mean_rms(male_results, common_labels)
    female_label_means = cohort_label_mean_rms(female_results, common_labels)
    x = np.arange(len(common_labels))
    width = 0.42
    ax.bar(x - width / 2, [male_label_means[label] for label in common_labels], width=width, color="#4c78a8", alpha=0.85, label="male->male")
    ax.bar(x + width / 2, [female_label_means[label] for label in common_labels], width=width, color="#f58518", alpha=0.85, label="female->female")
    ax.set_xticks(x)
    ax.set_xticklabels(common_labels, rotation=30, ha="right")
    ax.set_ylabel("Mean label RMS (px)")
    ax.set_title("Per-label cohort comparison", fontsize=13, fontweight="bold")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(framealpha=0.95)

    available_counts = {"male": 0, "female": 0}
    for speaker in loaded_speakers.values():
        available_counts[speaker.spec.gender] += 1

    fig.suptitle("Male vs female same-sex transform comparison", fontsize=16, fontweight="bold")
    fig.text(
        0.02,
        0.01,
        "\n".join(
            [
                f"Available speakers: male={available_counts['male']} | female={available_counts['female']}",
                f"Pairs: male->male={len(male_results)} | female->female={len(female_results)}",
                f"Common labels ({len(common_labels)}): {', '.join(common_labels)}",
                "Skipped curated speakers:",
                "  " + ("; ".join(skipped_messages) if skipped_messages else "none"),
            ]
        ),
        ha="left",
        va="bottom",
        fontsize=9.5,
        bbox=dict(boxstyle="round,pad=0.35", fc="white", alpha=0.96),
    )

    plt.tight_layout(rect=(0, 0.08, 1, 0.96))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def run_cohort(
    cohort: str,
    speakers: list[LoadedSpeaker],
    common_labels: list[str],
    output_dir: Path,
    nc_template: int,
) -> list[PairTransformResult]:
    colors = choose_label_colors(common_labels)
    pair_results: list[PairTransformResult] = []

    for target in speakers:
        target_pair_results: list[PairTransformResult] = []
        for source in speakers:
            if source.spec.basename == target.spec.basename:
                continue

            result = compute_pair_transform(
                cohort=cohort,
                source=source,
                target=target,
                common_labels=common_labels,
                nc_template=nc_template,
                include_warped_image=True,
            )
            pair_results.append(result)
            target_pair_results.append(result)

            pair_output = output_dir / cohort / "pairs" / f"{result.source_basename}_to_{result.target_basename}.png"
            save_pair_figure(result, source, target, common_labels, colors, pair_output)

        target_output = output_dir / cohort / "targets" / f"{target.spec.basename}_overview.png"
        save_target_overview_figure(
            cohort=cohort,
            target=target,
            pair_results=target_pair_results,
            common_labels=common_labels,
            colors=colors,
            output_path=target_output,
        )

    return pair_results


def run_all_targets_metric_pass(
    speakers: list[LoadedSpeaker],
    common_labels: list[str],
    nc_template: int,
) -> list[PairTransformResult]:
    pair_results: list[PairTransformResult] = []

    for source in speakers:
        for target in speakers:
            if source.spec.basename == target.spec.basename:
                continue
            pair_results.append(
                compute_pair_transform(
                    cohort="all_targets",
                    source=source,
                    target=target,
                    common_labels=common_labels,
                    nc_template=nc_template,
                    include_warped_image=False,
                )
            )

    return pair_results


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if not args.vtln_dir.is_dir():
        raise FileNotFoundError(
            f"Curated VTLN directory not found: {args.vtln_dir}. "
            "Pass --vtln-dir explicitly if the curated set lives outside this branch worktree."
        )

    specs = read_curated_specs(args.vtln_dir)
    if not specs:
        raise ValueError(f"No curated speaker specs were found in {args.vtln_dir}")

    loaded_speakers, skipped_messages = load_available_speakers(specs, args.vtln_dir)
    common_labels = compute_global_common_labels(loaded_speakers)
    all_speakers = sorted(loaded_speakers.values(), key=lambda item: item.spec.basename)

    male_speakers = sorted(
        [speaker for speaker in all_speakers if speaker.spec.gender == "male"],
        key=lambda item: item.spec.basename,
    )
    female_speakers = sorted(
        [speaker for speaker in all_speakers if speaker.spec.gender == "female"],
        key=lambda item: item.spec.basename,
    )
    if len(male_speakers) < 2 or len(female_speakers) < 2:
        raise ValueError(
            "Need at least two available speakers per cohort. "
            f"Loaded male={len(male_speakers)}, female={len(female_speakers)}"
        )

    male_results = run_cohort("male_to_male", male_speakers, common_labels, args.output_dir, args.nc_template)
    female_results = run_cohort("female_to_female", female_speakers, common_labels, args.output_dir, args.nc_template)
    all_target_results = run_all_targets_metric_pass(all_speakers, common_labels, args.nc_template)
    source_selection_summaries = compute_source_selection_summaries(
        male_results + female_results,
        all_target_results,
        loaded_speakers,
    )

    summary_path = args.output_dir / "male_vs_female_summary.png"
    save_cross_cohort_summary(
        male_results=male_results,
        female_results=female_results,
        common_labels=common_labels,
        loaded_speakers=loaded_speakers,
        skipped_messages=skipped_messages,
        output_path=summary_path,
    )
    same_gender_vs_all_targets_summary_path = args.output_dir / "same_gender_vs_all_targets_summary.png"
    save_same_gender_vs_all_targets_summary(
        source_summaries=source_selection_summaries,
        common_labels=common_labels,
        output_path=same_gender_vs_all_targets_summary_path,
    )

    male_mean = float(np.mean([pair.rms for pair in male_results]))
    male_median = float(np.median([pair.rms for pair in male_results]))
    female_mean = float(np.mean([pair.rms for pair in female_results]))
    female_median = float(np.median([pair.rms for pair in female_results]))
    overall_improvement_mean = float(np.mean([item.improvement for item in source_selection_summaries]))
    overall_improvement_median = float(np.median([item.improvement for item in source_selection_summaries]))

    print("Loaded speakers:")
    for speaker in all_speakers:
        print(f"  {speaker.spec.basename} ({speaker.spec.speaker_id}, {speaker.spec.gender})")
    print("Skipped speakers:")
    if skipped_messages:
        for message in skipped_messages:
            print(f"  {message}")
    else:
        print("  none")
    print(f"Common labels ({len(common_labels)}): {', '.join(common_labels)}")
    print(f"male->male pairs: {len(male_results)}")
    print(f"male->male RMS mean/median: {male_mean:.2f} / {male_median:.2f} px")
    print(f"female->female pairs: {len(female_results)}")
    print(f"female->female RMS mean/median: {female_mean:.2f} / {female_median:.2f} px")
    print("Same-gender vs all-target baseline:")
    for item in source_selection_summaries:
        print(
            "  "
            f"{item.source_speaker_id} ({item.source_gender}): "
            f"same-gender={item.same_gender_mean_rms:.2f} px | "
            f"all-targets={item.all_target_mean_rms:.2f} px | "
            f"improvement={item.improvement:.2f} px"
        )
    print(
        "Overall improvement mean/median "
        f"(all-target mean - same-gender mean): {overall_improvement_mean:.2f} / {overall_improvement_median:.2f} px"
    )
    print(f"Saved summary: {summary_path}")
    print(f"Saved summary: {same_gender_vs_all_targets_summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
