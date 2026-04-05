from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from grid_transform.apps.analyze_aligned_speaker_vowel_variants import (
    main as analyze_aligned_speaker_vowel_variants_main,
)
from grid_transform.apps.average_speaker_grid_transform import (
    choose_label_colors,
    compute_notebook_style_mean,
    compute_transformed_population,
    plot_contour_dict,
    save_notebook_style_figure,
    save_transformed_average_figure,
)
from grid_transform.apps.extract_speaker_vowel_variants import (
    main as extract_speaker_vowel_variants_main,
)
from grid_transform.apps.method4_transform import format_target_frame, resample_polyline
from grid_transform.config import DEFAULT_OUTPUT_DIR, PROJECT_DIR
from grid_transform.io import load_frame_vtln
from grid_transform.vt import build_grid


DEFAULT_SPEAKERS = ("P1", "P2", "P3", "P4", "P5", "P6", "P7", "P8", "P9", "P10")
DEFAULT_VOWELS = ("a", "e", "i", "o", "u")
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
DEFAULT_N_VERT = 9
DEFAULT_N_POINTS = 250
DEFAULT_RESAMPLE_POINTS = 80
DEFAULT_TEMPLATE_POINTS = 80
DEFAULT_MAX_SHIFT = 24
DEFAULT_ALIGNMENT_PASSES = 2
EPSILON = 1e-8


@dataclass(frozen=True)
class CuratedSpeakerSpec:
    speaker_id: str
    basename: str
    raw_subject: str | None
    session: str | None
    frame: int | None
    gender: str
    image_path: Path
    zip_path: Path


@dataclass
class LoadedSpeaker:
    spec: CuratedSpeakerSpec
    image: object
    contours: dict[str, np.ndarray]
    grid: object


def default_vtln_dir() -> Path:
    return PROJECT_DIR / "VTLN" / "data"


def default_dataset_root() -> Path:
    return PROJECT_DIR.parent / "Data" / "Artspeech_database"


def default_output_dir() -> Path:
    return DEFAULT_OUTPUT_DIR / "average_speaker_and_variability"


def speaker_sort_key(name: str) -> tuple[int, str]:
    suffix = name[1:] if name.startswith("P") else name
    try:
        return int(suffix), name
    except ValueError:
        return 10**9, name


def parse_csv_values(text: str) -> list[str]:
    values = [value.strip() for value in text.split(",") if value.strip()]
    if not values:
        raise ValueError("Expected a non-empty comma-separated list.")
    return values


def parse_speakers(text: str) -> list[str]:
    speakers = [value.upper() for value in parse_csv_values(text)]
    invalid = [speaker for speaker in speakers if speaker not in CURATED_SPEAKER_GENDER]
    if invalid:
        raise ValueError(f"Unknown curated speakers: {', '.join(invalid)}")
    seen: set[str] = set()
    deduped: list[str] = []
    for speaker in speakers:
        if speaker not in seen:
            deduped.append(speaker)
            seen.add(speaker)
    return deduped


def parse_vowels(text: str) -> list[str]:
    return parse_csv_values(text)


def parse_selected_source(selected_source: str) -> tuple[str | None, int | None]:
    parts = [part.strip() for part in selected_source.split("/") if part.strip()]
    if len(parts) != 3:
        return None, None
    session = parts[1].upper()
    frame_text = parts[2].upper()
    if not frame_text.startswith("F") or not frame_text[1:].isdigit():
        return session, None
    return session, int(frame_text[1:])


def resolve_image_path(vtln_dir: Path, basename: str) -> Path:
    for suffix in IMAGE_SUFFIXES:
        candidate = vtln_dir / f"{basename}{suffix}"
        if candidate.is_file():
            return candidate
    return vtln_dir / f"{basename}.png"


def read_curated_specs(vtln_dir: Path) -> dict[str, CuratedSpeakerSpec]:
    manifest_path = vtln_dir / "selection_manifest.csv"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Curated manifest not found: {manifest_path}")

    specs: dict[str, CuratedSpeakerSpec] = {}
    with manifest_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            speaker_id = row["speaker"].strip().upper()
            if speaker_id not in CURATED_SPEAKER_GENDER:
                continue
            if speaker_id in specs:
                raise ValueError(f"Expected one curated row per speaker, found duplicate for {speaker_id}.")

            basename = row["output_basename"].strip()
            session, frame = parse_selected_source(row.get("selected_source", ""))
            specs[speaker_id] = CuratedSpeakerSpec(
                speaker_id=speaker_id,
                basename=basename,
                raw_subject=row.get("raw_subject", "").strip() or None,
                session=session,
                frame=frame,
                gender=CURATED_SPEAKER_GENDER[speaker_id],
                image_path=resolve_image_path(vtln_dir, basename),
                zip_path=vtln_dir / f"{basename}.zip",
            )
    return specs


def load_curated_speakers(vtln_dir: Path, requested_speakers: list[str]) -> dict[str, LoadedSpeaker]:
    specs = read_curated_specs(vtln_dir)
    missing = [speaker for speaker in requested_speakers if speaker not in specs]
    if missing:
        raise ValueError(f"Missing curated manifest rows for: {', '.join(missing)}")

    loaded: dict[str, LoadedSpeaker] = {}
    for speaker_id in requested_speakers:
        spec = specs[speaker_id]
        if not spec.image_path.is_file():
            raise FileNotFoundError(f"Curated image not found for {speaker_id}: {spec.image_path}")
        if not spec.zip_path.is_file():
            raise FileNotFoundError(f"Curated ROI zip not found for {speaker_id}: {spec.zip_path}")

        image, contours = load_frame_vtln(spec.basename, vtln_dir)
        grid = build_grid(
            image,
            contours,
            n_vert=DEFAULT_N_VERT,
            n_points=DEFAULT_N_POINTS,
            frame_number=0,
        )
        loaded[speaker_id] = LoadedSpeaker(spec=spec, image=image, contours=contours, grid=grid)
    return loaded


def stack_resampled_contours(contours: dict[str, np.ndarray], labels: list[str], nc_template: int) -> np.ndarray:
    return np.vstack([resample_polyline(contours[label], nc_template) for label in labels])


def flatten_resampled_contours(contours: dict[str, np.ndarray], labels: list[str], nc_template: int) -> np.ndarray:
    return stack_resampled_contours(contours, labels, nc_template).reshape(-1)


def serialize_contours(contours: dict[str, np.ndarray]) -> dict[str, list[list[float]]]:
    return {
        label: np.asarray(points, dtype=float).round(6).tolist()
        for label, points in contours.items()
    }


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})


def save_population_overlay_figure(
    speakers: dict[str, LoadedSpeaker],
    labels: list[str],
    transformed: dict[str, object],
    output_path: Path,
) -> None:
    reference_id = str(transformed["reference_id"])
    reference = speakers[reference_id]
    label_colors = choose_label_colors(labels)

    fig, ax = plt.subplots(1, 1, figsize=(10, 10), dpi=180)
    format_target_frame(ax, reference.image, f"Population mean/median contours in reference space: {reference_id}")

    for label in labels:
        ref_pts = np.asarray(reference.contours[label], dtype=float)
        ax.plot(ref_pts[:, 0], ref_pts[:, 1], color="0.55", lw=1.0, alpha=0.35)

    for speaker_id in sorted(transformed["mapped_contours"], key=speaker_sort_key):
        plot_contour_dict(
            ax,
            transformed["mapped_contours"][speaker_id],
            labels,
            label_colors,
            alpha=0.05,
            lw=0.9,
            linestyle="--",
        )

    plot_contour_dict(
        ax,
        transformed["mean_contours"],
        labels,
        label_colors,
        alpha=0.98,
        lw=3.0,
        linestyle="-",
    )
    plot_contour_dict(
        ax,
        transformed["median_contours"],
        labels,
        label_colors,
        alpha=0.98,
        lw=2.2,
        linestyle=":",
    )
    ax.set_title("Reference contours (gray), transformed mean (solid), transformed median (dotted)", fontweight="bold")
    handles = [
        plt.Line2D([0], [0], color="black", lw=3.0, linestyle="-"),
        plt.Line2D([0], [0], color="black", lw=2.2, linestyle=":"),
        plt.Line2D([0], [0], color="0.55", lw=1.3, linestyle="-"),
    ]
    ax.legend(handles, ["Transformed mean", "Transformed median", "Reference speaker"], loc="upper right", framealpha=0.95)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def compute_pairwise_rms_stats(
    mapped_contours: dict[str, dict[str, np.ndarray]],
    labels: list[str],
    nc_template: int,
    gender_by_speaker: dict[str, str],
) -> tuple[list[dict[str, object]], dict[str, dict[str, float | None]], dict[str, float | None]]:
    speaker_ids = sorted(mapped_contours, key=speaker_sort_key)
    resampled_shapes = {
        speaker_id: stack_resampled_contours(mapped_contours[speaker_id], labels, nc_template)
        for speaker_id in speaker_ids
    }

    pairwise_rows: list[dict[str, object]] = []
    per_speaker_values = {
        speaker_id: {"all": [], "within": [], "between": []}
        for speaker_id in speaker_ids
    }
    within_gender_rms: list[float] = []
    between_gender_rms: list[float] = []

    for index, speaker_i in enumerate(speaker_ids):
        for speaker_j in speaker_ids[index + 1 :]:
            rms = float(np.sqrt(np.mean((resampled_shapes[speaker_i] - resampled_shapes[speaker_j]) ** 2)))
            same_gender = gender_by_speaker[speaker_i] == gender_by_speaker[speaker_j]
            pair_type = "within_gender" if same_gender else "between_gender"

            pairwise_rows.append(
                {
                    "speaker_i": speaker_i,
                    "gender_i": gender_by_speaker[speaker_i],
                    "speaker_j": speaker_j,
                    "gender_j": gender_by_speaker[speaker_j],
                    "pair_type": pair_type,
                    "rms": rms,
                }
            )

            per_speaker_values[speaker_i]["all"].append(rms)
            per_speaker_values[speaker_j]["all"].append(rms)
            if same_gender:
                per_speaker_values[speaker_i]["within"].append(rms)
                per_speaker_values[speaker_j]["within"].append(rms)
                within_gender_rms.append(rms)
            else:
                per_speaker_values[speaker_i]["between"].append(rms)
                per_speaker_values[speaker_j]["between"].append(rms)
                between_gender_rms.append(rms)

    per_speaker_summary: dict[str, dict[str, float | None]] = {}
    for speaker_id, buckets in per_speaker_values.items():
        pairwise_all = buckets["all"]
        pairwise_within = buckets["within"]
        pairwise_between = buckets["between"]
        per_speaker_summary[speaker_id] = {
            "pairwise_rms_sum": float(np.sum(pairwise_all)) if pairwise_all else 0.0,
            "pairwise_rms_mean": float(np.mean(pairwise_all)) if pairwise_all else 0.0,
            "within_gender_mean_rms": float(np.mean(pairwise_within)) if pairwise_within else None,
            "between_gender_mean_rms": float(np.mean(pairwise_between)) if pairwise_between else None,
        }

    overall_summary = {
        "within_gender_mean_rms": float(np.mean(within_gender_rms)) if within_gender_rms else None,
        "between_gender_mean_rms": float(np.mean(between_gender_rms)) if between_gender_rms else None,
    }
    return pairwise_rows, per_speaker_summary, overall_summary


def compute_guided_gender_pca(
    mapped_contours: dict[str, dict[str, np.ndarray]],
    labels: list[str],
    nc_template: int,
    gender_by_speaker: dict[str, str],
) -> dict[str, object]:
    speaker_ids = sorted(mapped_contours, key=speaker_sort_key)
    features = np.stack(
        [flatten_resampled_contours(mapped_contours[speaker_id], labels, nc_template) for speaker_id in speaker_ids],
        axis=0,
    )
    centered = features - features.mean(axis=0, keepdims=True)
    male_mask = np.asarray([gender_by_speaker[speaker_id] == "male" for speaker_id in speaker_ids], dtype=bool)
    female_mask = np.asarray([gender_by_speaker[speaker_id] == "female" for speaker_id in speaker_ids], dtype=bool)
    if int(male_mask.sum()) == 0 or int(female_mask.sum()) == 0:
        raise ValueError("Guided gender PCA requires at least one male and one female speaker.")

    guided_axis_raw = centered[male_mask].mean(axis=0) - centered[female_mask].mean(axis=0)
    axis_norm = float(np.linalg.norm(guided_axis_raw))
    if axis_norm < EPSILON:
        guided_axis = np.zeros_like(guided_axis_raw)
        guided_scores = np.zeros(len(speaker_ids), dtype=float)
    else:
        guided_axis = guided_axis_raw / axis_norm
        guided_scores = centered @ guided_axis

    residual = centered - np.outer(guided_scores, guided_axis)
    residual_scores = np.zeros((len(speaker_ids), 2), dtype=float)
    explained_variance_ratio = [0.0, 0.0]
    if residual.size > 0:
        u_mat, singular_vals, _ = np.linalg.svd(residual, full_matrices=False)
        max_components = min(2, len(singular_vals))
        if max_components > 0:
            residual_scores[:, :max_components] = u_mat[:, :max_components] * singular_vals[:max_components]
            if residual.shape[0] > 1:
                residual_variance = (singular_vals ** 2) / max(residual.shape[0] - 1, 1)
                total_residual_variance = float(residual_variance.sum())
                if total_residual_variance > EPSILON:
                    for index in range(max_components):
                        explained_variance_ratio[index] = float(residual_variance[index] / total_residual_variance)

    score_rows: list[dict[str, object]] = []
    for index, speaker_id in enumerate(speaker_ids):
        score_rows.append(
            {
                "speaker": speaker_id,
                "gender": gender_by_speaker[speaker_id],
                "guided_axis_score": float(guided_scores[index]),
                "residual_pc1_score": float(residual_scores[index, 0]),
                "residual_pc2_score": float(residual_scores[index, 1]),
            }
        )

    male_scores = [row["guided_axis_score"] for row in score_rows if row["gender"] == "male"]
    female_scores = [row["guided_axis_score"] for row in score_rows if row["gender"] == "female"]
    return {
        "factor": "gender",
        "axis_direction": "male_minus_female",
        "axis_norm": axis_norm,
        "explained_variance_ratio": explained_variance_ratio,
        "score_rows": score_rows,
        "male_guided_axis_mean": float(np.mean(male_scores)) if male_scores else None,
        "female_guided_axis_mean": float(np.mean(female_scores)) if female_scores else None,
    }


def save_guided_pca_scatter(guided_pca: dict[str, object], output_path: Path) -> None:
    rows = list(guided_pca["score_rows"])
    explained = guided_pca["explained_variance_ratio"]
    colors = {"male": "#1f77b4", "female": "#d62728"}
    panels = [
        ("residual_pc1_score", "Residual PC1", explained[0]),
        ("residual_pc2_score", "Residual PC2", explained[1]),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), dpi=180)
    for ax, (score_key, ylabel, explained_ratio) in zip(axes, panels):
        for gender in ("male", "female"):
            gender_rows = [row for row in rows if row["gender"] == gender]
            if not gender_rows:
                continue
            x_values = [float(row["guided_axis_score"]) for row in gender_rows]
            y_values = [float(row[score_key]) for row in gender_rows]
            ax.scatter(x_values, y_values, s=60, alpha=0.9, color=colors[gender], label=gender.title())
            for row, x_value, y_value in zip(gender_rows, x_values, y_values):
                ax.annotate(
                    str(row["speaker"]),
                    xy=(x_value, y_value),
                    xytext=(4, 4),
                    textcoords="offset points",
                    fontsize=9,
                )

        ax.axvline(0.0, color="0.6", lw=1.0, linestyle="--")
        ax.axhline(0.0, color="0.6", lw=1.0, linestyle="--")
        ax.set_xlabel("Guided gender axis score", fontweight="bold")
        ax.set_ylabel(ylabel, fontweight="bold")
        ax.set_title(f"{ylabel} vs guided gender axis\nresidual explained variance={explained_ratio:.3f}", fontweight="bold")
        ax.grid(True, alpha=0.25)
        ax.margins(x=0.08, y=0.12)

    axes[0].legend(loc="best", framealpha=0.95)
    fig.subplots_adjust(left=0.08, right=0.98, bottom=0.12, top=0.86, wspace=0.28)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)


def run_shape_analysis(
    vtln_dir: Path,
    speaker_ids: list[str],
    output_dir: Path,
    reference_speaker: str,
    nc_resample: int,
    nc_template: int,
) -> dict[str, object]:
    stage_dir = output_dir / "shape_analysis"
    stage_dir.mkdir(parents=True, exist_ok=True)

    speakers = load_curated_speakers(vtln_dir, speaker_ids)
    gender_by_speaker = {speaker_id: speakers[speaker_id].spec.gender for speaker_id in speaker_ids}
    contours_by_speaker = {
        speaker_id: speakers[speaker_id].contours
        for speaker_id in speaker_ids
    }
    ms = compute_notebook_style_mean(contours_by_speaker, nc_resample=nc_resample)

    if reference_speaker == "auto":
        reference_id = str(ms["geometric_median"])
    else:
        reference_id = reference_speaker.upper()
        if reference_id not in speakers:
            raise ValueError(f"Reference speaker '{reference_speaker}' not found in the curated population.")

    speaker_payload = {
        speaker_id: {
            "image": speakers[speaker_id].image,
            "contours": speakers[speaker_id].contours,
            "grid": speakers[speaker_id].grid,
        }
        for speaker_id in speaker_ids
    }
    transformed = compute_transformed_population(
        speaker_payload,
        ms["common_labels"],
        reference_id,
        nc_template=nc_template,
    )

    pairwise_rows, pairwise_summary_by_speaker, pairwise_overall = compute_pairwise_rms_stats(
        transformed["mapped_contours"],
        ms["common_labels"],
        nc_template,
        gender_by_speaker,
    )
    guided_pca = compute_guided_gender_pca(
        transformed["mapped_contours"],
        ms["common_labels"],
        nc_template,
        gender_by_speaker,
    )
    guided_scores_by_speaker = {
        row["speaker"]: row
        for row in guided_pca["score_rows"]
    }

    notebook_figure_path = stage_dir / "average_speaker_notebook_style_tongue.png"
    transformed_figure_path = stage_dir / "average_speaker_after_grid_transform_tongue.png"
    overlay_figure_path = stage_dir / "population_mean_median_overlay.png"
    guided_pca_figure_path = stage_dir / "guided_gender_pca_scatter.png"
    summary_text_path = stage_dir / "average_speaker_grid_transform_summary.txt"
    contours_json_path = stage_dir / "synthetic_population_contours.json"
    summary_json_path = stage_dir / "summary.json"
    per_speaker_metrics_path = stage_dir / "per_speaker_metrics.csv"
    pairwise_rms_path = stage_dir / "pairwise_rms.csv"
    guided_scores_path = stage_dir / "guided_pca_scores.csv"

    save_notebook_style_figure(ms, notebook_figure_path)
    save_transformed_average_figure(speaker_payload, ms, transformed, transformed_figure_path)
    save_population_overlay_figure(speakers, ms["common_labels"], transformed, overlay_figure_path)
    save_guided_pca_scatter(guided_pca, guided_pca_figure_path)

    lines = [
        "Average Speaker + Speaker Variability Summary",
        "=============================================",
        f"Speakers used ({len(speaker_ids)}): {', '.join(speaker_ids)}",
        f"Common labels ({len(ms['common_labels'])}): {', '.join(ms['common_labels'])}",
        f"Geometric median speaker: {ms['geometric_median']}",
        f"Closest to rigid-GPA mean: {ms['closest_to_mean']}",
        f"Reference speaker for grid transform: {reference_id}",
        f"Within-gender RMS mean: {pairwise_overall['within_gender_mean_rms']:.3f}" if pairwise_overall["within_gender_mean_rms"] is not None else "Within-gender RMS mean: n/a",
        f"Between-gender RMS mean: {pairwise_overall['between_gender_mean_rms']:.3f}" if pairwise_overall["between_gender_mean_rms"] is not None else "Between-gender RMS mean: n/a",
        "",
        "Per-label variability after grid transform (RMS to transformed median):",
    ]
    for label, value in sorted(transformed["label_variability_to_median"].items(), key=lambda item: item[1], reverse=True):
        lines.append(f"  {label:20s} {value:7.3f} px")
    lines.append("")
    lines.append("Guided PCA (gender):")
    lines.append(f"  axis_direction: {guided_pca['axis_direction']}")
    lines.append(f"  male_guided_axis_mean: {guided_pca['male_guided_axis_mean']:.3f}")
    lines.append(f"  female_guided_axis_mean: {guided_pca['female_guided_axis_mean']:.3f}")
    lines.append(f"  residual_pc1_explained_variance_ratio: {guided_pca['explained_variance_ratio'][0]:.3f}")
    lines.append(f"  residual_pc2_explained_variance_ratio: {guided_pca['explained_variance_ratio'][1]:.3f}")
    summary_text_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    contours_payload = {
        "reference_speaker": reference_id,
        "common_labels": list(ms["common_labels"]),
        "nc_resample": int(nc_resample),
        "nc_template": int(nc_template),
        "rigid_gpa_mean_contours": serialize_contours(ms["template_per_label"]),
        "rigid_gpa_median_contours": serialize_contours(ms["median_per_label"]),
        "transformed_mean_contours": serialize_contours(transformed["mean_contours"]),
        "transformed_median_contours": serialize_contours(transformed["median_contours"]),
    }
    contours_json_path.write_text(json.dumps(contours_payload, indent=2, ensure_ascii=False), encoding="utf-8")

    per_speaker_rows: list[dict[str, object]] = []
    for speaker_id in speaker_ids:
        spec = speakers[speaker_id].spec
        pairwise_stats = pairwise_summary_by_speaker[speaker_id]
        guided_row = guided_scores_by_speaker[speaker_id]
        per_speaker_rows.append(
            {
                "speaker": speaker_id,
                "basename": spec.basename,
                "gender": spec.gender,
                "raw_subject": spec.raw_subject,
                "session": spec.session,
                "frame": spec.frame,
                "rigid_gpa_rms_to_mean": float(ms["rms_to_template"][speaker_id]),
                "rigid_gpa_rms_to_median": float(ms["speaker_rms_to_median"][speaker_id]),
                "rms_to_population_mean": float(transformed["mean_rms"][speaker_id]),
                "rms_to_population_median": float(transformed["median_rms"][speaker_id]),
                "pairwise_rms_sum": pairwise_stats["pairwise_rms_sum"],
                "pairwise_rms_mean": pairwise_stats["pairwise_rms_mean"],
                "within_gender_mean_rms": pairwise_stats["within_gender_mean_rms"],
                "between_gender_mean_rms": pairwise_stats["between_gender_mean_rms"],
                "guided_axis_score": guided_row["guided_axis_score"],
                "residual_pc1_score": guided_row["residual_pc1_score"],
                "residual_pc2_score": guided_row["residual_pc2_score"],
            }
        )

    write_csv(
        per_speaker_metrics_path,
        [
            "speaker",
            "basename",
            "gender",
            "raw_subject",
            "session",
            "frame",
            "rigid_gpa_rms_to_mean",
            "rigid_gpa_rms_to_median",
            "rms_to_population_mean",
            "rms_to_population_median",
            "pairwise_rms_sum",
            "pairwise_rms_mean",
            "within_gender_mean_rms",
            "between_gender_mean_rms",
            "guided_axis_score",
            "residual_pc1_score",
            "residual_pc2_score",
        ],
        per_speaker_rows,
    )
    write_csv(
        pairwise_rms_path,
        ["speaker_i", "gender_i", "speaker_j", "gender_j", "pair_type", "rms"],
        pairwise_rows,
    )
    write_csv(
        guided_scores_path,
        ["speaker", "gender", "guided_axis_score", "residual_pc1_score", "residual_pc2_score"],
        list(guided_pca["score_rows"]),
    )

    summary_payload = {
        "stage": "shape_analysis",
        "speaker_count": len(speaker_ids),
        "speakers": speaker_ids,
        "common_labels": list(ms["common_labels"]),
        "reference_speaker": reference_id,
        "geometric_median_speaker": str(ms["geometric_median"]),
        "closest_to_mean_speaker": str(ms["closest_to_mean"]),
        "within_gender_mean_rms": pairwise_overall["within_gender_mean_rms"],
        "between_gender_mean_rms": pairwise_overall["between_gender_mean_rms"],
        "per_label_variability_to_median": {
            label: float(value)
            for label, value in transformed["label_variability_to_median"].items()
        },
        "guided_pca": {
            "factor": guided_pca["factor"],
            "axis_direction": guided_pca["axis_direction"],
            "male_guided_axis_mean": guided_pca["male_guided_axis_mean"],
            "female_guided_axis_mean": guided_pca["female_guided_axis_mean"],
            "explained_variance_ratio": list(guided_pca["explained_variance_ratio"]),
        },
        "paths": {
            "notebook_figure": str(notebook_figure_path),
            "transformed_figure": str(transformed_figure_path),
            "overlay_figure": str(overlay_figure_path),
            "guided_pca_figure": str(guided_pca_figure_path),
            "summary_text": str(summary_text_path),
            "contours_json": str(contours_json_path),
            "per_speaker_metrics_csv": str(per_speaker_metrics_path),
            "pairwise_rms_csv": str(pairwise_rms_path),
            "guided_pca_scores_csv": str(guided_scores_path),
        },
    }
    summary_json_path.write_text(json.dumps(summary_payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return {
        "stage_dir": stage_dir,
        "summary": summary_payload,
        "per_speaker_rows": per_speaker_rows,
    }


def summarize_vowel_metrics(
    speaker: str,
    gender: str,
    vowel: str,
    vowel_summary: dict[str, object],
) -> dict[str, object]:
    residual_metrics = list(vowel_summary.get("residual_metrics") or [])
    if residual_metrics:
        mean_rmse = float(np.mean([float(row["rmse_to_mean"]) for row in residual_metrics]))
        mean_mae = float(np.mean([float(row["mae_to_mean"]) for row in residual_metrics]))
        mean_corr = float(np.mean([float(row["corr_to_mean"]) for row in residual_metrics]))
    else:
        mean_rmse = None
        mean_mae = None
        mean_corr = None

    return {
        "speaker": speaker,
        "gender": gender,
        "vowel": vowel,
        "selected_samples": int(vowel_summary.get("selected_samples") or 0),
        "pairwise_correlation_before_mean": vowel_summary.get("pairwise_correlation_before_mean"),
        "pairwise_correlation_after_mean": vowel_summary.get("pairwise_correlation_after_mean"),
        "pairwise_correlation_gain_mean": vowel_summary.get("pairwise_correlation_gain_mean"),
        "aligned_std_mean": vowel_summary.get("aligned_std_mean"),
        "aligned_std_max": vowel_summary.get("aligned_std_max"),
        "mean_abs_l1_shift_px": vowel_summary.get("mean_abs_l1_shift_px"),
        "mean_rmse_to_mean": mean_rmse,
        "mean_mae_to_mean": mean_mae,
        "mean_corr_to_mean": mean_corr,
        "prototype_session": vowel_summary.get("prototype_session"),
        "prototype_frame_index_1based": vowel_summary.get("prototype_frame_index_1based"),
        "outlier_session": vowel_summary.get("outlier_session"),
        "outlier_frame_index_1based": vowel_summary.get("outlier_frame_index_1based"),
    }


def run_within_speaker_variability(
    dataset_root: Path,
    speaker_ids: list[str],
    vowels: list[str],
    samples_per_vowel: int,
    max_shift: int,
    alignment_passes: int,
    output_dir: Path,
) -> dict[str, object]:
    if not dataset_root.exists():
        raise FileNotFoundError(f"ArtSpeech dataset root not found: {dataset_root}")

    stage_dir = output_dir / "token_variability"
    stage_dir.mkdir(parents=True, exist_ok=True)

    per_speaker_rows: list[dict[str, object]] = []
    per_vowel_rows: list[dict[str, object]] = []
    vowels_text = ",".join(vowels)

    for speaker_id in speaker_ids:
        speaker_dir = stage_dir / speaker_id
        samples_dir = speaker_dir / "samples"
        aligned_dir = speaker_dir / "aligned_analysis"

        extract_speaker_vowel_variants_main(
            [
                "--speaker",
                speaker_id,
                "--dataset-root",
                str(dataset_root),
                "--output-dir",
                str(samples_dir),
                "--vowels",
                vowels_text,
                "--samples-per-vowel",
                str(samples_per_vowel),
            ]
        )
        analyze_aligned_speaker_vowel_variants_main(
            [
                "--speaker",
                speaker_id,
                "--samples-dir",
                str(samples_dir),
                "--output-dir",
                str(aligned_dir),
                "--vowels",
                vowels_text,
                "--max-shift",
                str(max_shift),
                "--alignment-passes",
                str(alignment_passes),
            ]
        )

        summary_path = aligned_dir / "aligned_variability_summary.json"
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        aligned_summary = dict(summary["aligned_variability_summary"])
        gender = CURATED_SPEAKER_GENDER[speaker_id]

        all_residual_metrics: list[dict[str, object]] = []
        vowel_rows_for_speaker: list[dict[str, object]] = []
        for vowel in vowels:
            vowel_summary = dict(aligned_summary.get(vowel, {}))
            summarized_row = summarize_vowel_metrics(speaker_id, gender, vowel, vowel_summary)
            per_vowel_rows.append(summarized_row)
            vowel_rows_for_speaker.append(summarized_row)
            all_residual_metrics.extend(list(vowel_summary.get("residual_metrics") or []))

        if all_residual_metrics:
            token_variability_score = float(np.mean([float(row["rmse_to_mean"]) for row in all_residual_metrics]))
            mean_mae_to_mean = float(np.mean([float(row["mae_to_mean"]) for row in all_residual_metrics]))
            mean_corr_to_mean = float(np.mean([float(row["corr_to_mean"]) for row in all_residual_metrics]))
        else:
            token_variability_score = None
            mean_mae_to_mean = None
            mean_corr_to_mean = None

        valid_vowel_rows = [row for row in vowel_rows_for_speaker if row["selected_samples"] > 0]
        per_speaker_rows.append(
            {
                "speaker": speaker_id,
                "gender": gender,
                "analyzed_vowels": len(valid_vowel_rows),
                "total_selected_samples": int(sum(row["selected_samples"] for row in vowel_rows_for_speaker)),
                "token_variability_score": token_variability_score,
                "mean_mae_to_mean": mean_mae_to_mean,
                "mean_corr_to_mean": mean_corr_to_mean,
                "mean_pairwise_correlation_gain": float(np.mean([row["pairwise_correlation_gain_mean"] for row in valid_vowel_rows])) if valid_vowel_rows else None,
                "mean_aligned_std": float(np.mean([row["aligned_std_mean"] for row in valid_vowel_rows])) if valid_vowel_rows else None,
                "speaker_output_dir": str(speaker_dir),
                "samples_dir": str(samples_dir),
                "aligned_summary_path": str(summary_path),
            }
        )

    per_speaker_path = stage_dir / "within_speaker_variability.csv"
    per_vowel_path = stage_dir / "within_speaker_variability_by_vowel.csv"
    summary_json_path = stage_dir / "within_speaker_variability.json"

    write_csv(
        per_speaker_path,
        [
            "speaker",
            "gender",
            "analyzed_vowels",
            "total_selected_samples",
            "token_variability_score",
            "mean_mae_to_mean",
            "mean_corr_to_mean",
            "mean_pairwise_correlation_gain",
            "mean_aligned_std",
            "speaker_output_dir",
            "samples_dir",
            "aligned_summary_path",
        ],
        per_speaker_rows,
    )
    write_csv(
        per_vowel_path,
        [
            "speaker",
            "gender",
            "vowel",
            "selected_samples",
            "pairwise_correlation_before_mean",
            "pairwise_correlation_after_mean",
            "pairwise_correlation_gain_mean",
            "aligned_std_mean",
            "aligned_std_max",
            "mean_abs_l1_shift_px",
            "mean_rmse_to_mean",
            "mean_mae_to_mean",
            "mean_corr_to_mean",
            "prototype_session",
            "prototype_frame_index_1based",
            "outlier_session",
            "outlier_frame_index_1based",
        ],
        per_vowel_rows,
    )

    summary_payload = {
        "stage": "token_variability",
        "dataset_root": str(dataset_root),
        "speakers": speaker_ids,
        "vowels": vowels,
        "samples_per_vowel": int(samples_per_vowel),
        "max_shift": int(max_shift),
        "alignment_passes": int(alignment_passes),
        "per_speaker_rows": per_speaker_rows,
        "per_vowel_rows": per_vowel_rows,
    }
    summary_json_path.write_text(json.dumps(summary_payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return {
        "stage_dir": stage_dir,
        "summary": summary_payload,
        "per_speaker_rows": per_speaker_rows,
        "per_vowel_rows": per_vowel_rows,
    }


def save_shape_vs_token_scatter(rows: list[dict[str, object]], output_path: Path) -> None:
    colors = {"male": "#1f77b4", "female": "#d62728"}
    fig, ax = plt.subplots(1, 1, figsize=(8, 6), dpi=180)

    for gender in ("male", "female"):
        gender_rows = [
            row for row in rows
            if row["gender"] == gender
            and row["shape_rms_to_population_median"] is not None
            and row["token_variability_score"] is not None
        ]
        if not gender_rows:
            continue
        x_values = [float(row["shape_rms_to_population_median"]) for row in gender_rows]
        y_values = [float(row["token_variability_score"]) for row in gender_rows]
        ax.scatter(x_values, y_values, s=70, alpha=0.92, color=colors[gender], label=gender.title())
        for row, x_value, y_value in zip(gender_rows, x_values, y_values):
            ax.annotate(
                str(row["speaker"]),
                xy=(x_value, y_value),
                xytext=(4, 4),
                textcoords="offset points",
                fontsize=9,
            )

    ax.set_xlabel("Shape RMS to population median", fontweight="bold")
    ax.set_ylabel("Token variability score (mean RMSE to aligned mean)", fontweight="bold")
    ax.set_title("Shape variability vs within-speaker token variability", fontweight="bold")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best", framealpha=0.95)
    ax.margins(x=0.08, y=0.12)
    fig.subplots_adjust(left=0.12, right=0.98, bottom=0.12, top=0.9)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)


def run_integration(
    shape_result: dict[str, object],
    token_result: dict[str, object],
    output_dir: Path,
) -> dict[str, object]:
    stage_dir = output_dir / "integration"
    stage_dir.mkdir(parents=True, exist_ok=True)

    shape_rows = {
        row["speaker"]: row
        for row in shape_result["per_speaker_rows"]
    }
    token_rows = {
        row["speaker"]: row
        for row in token_result["per_speaker_rows"]
    }

    joined_speakers = sorted(set(shape_rows) & set(token_rows), key=speaker_sort_key)
    integrated_rows: list[dict[str, object]] = []
    for speaker_id in joined_speakers:
        shape_row = shape_rows[speaker_id]
        token_row = token_rows[speaker_id]
        integrated_rows.append(
            {
                "speaker": speaker_id,
                "basename": shape_row["basename"],
                "gender": shape_row["gender"],
                "shape_rms_to_population_mean": shape_row["rms_to_population_mean"],
                "shape_rms_to_population_median": shape_row["rms_to_population_median"],
                "pairwise_rms_sum": shape_row["pairwise_rms_sum"],
                "guided_axis_score": shape_row["guided_axis_score"],
                "residual_pc1_score": shape_row["residual_pc1_score"],
                "residual_pc2_score": shape_row["residual_pc2_score"],
                "token_variability_score": token_row["token_variability_score"],
                "mean_mae_to_mean": token_row["mean_mae_to_mean"],
                "mean_corr_to_mean": token_row["mean_corr_to_mean"],
                "mean_pairwise_correlation_gain": token_row["mean_pairwise_correlation_gain"],
            }
        )

    integrated_csv_path = stage_dir / "integrated_speaker_metrics.csv"
    integrated_json_path = stage_dir / "summary.json"
    comparison_figure_path = stage_dir / "shape_vs_token_variability.png"

    write_csv(
        integrated_csv_path,
        [
            "speaker",
            "basename",
            "gender",
            "shape_rms_to_population_mean",
            "shape_rms_to_population_median",
            "pairwise_rms_sum",
            "guided_axis_score",
            "residual_pc1_score",
            "residual_pc2_score",
            "token_variability_score",
            "mean_mae_to_mean",
            "mean_corr_to_mean",
            "mean_pairwise_correlation_gain",
        ],
        integrated_rows,
    )
    save_shape_vs_token_scatter(integrated_rows, comparison_figure_path)

    summary_payload = {
        "stage": "integration",
        "speaker_count": len(integrated_rows),
        "speakers": joined_speakers,
        "paths": {
            "integrated_metrics_csv": str(integrated_csv_path),
            "comparison_figure": str(comparison_figure_path),
        },
    }
    integrated_json_path.write_text(json.dumps(summary_payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return {
        "stage_dir": stage_dir,
        "summary": summary_payload,
        "integrated_rows": integrated_rows,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the combined curated-speaker average/variability analysis and the within-speaker "
            "ArtSpeech vowel variability workflow, then join both result tables."
        )
    )
    parser.add_argument("--vtln-dir", type=Path, default=default_vtln_dir(), help="Curated VTLN speaker directory. Defaults to VTLN/data.")
    parser.add_argument("--dataset-root", type=Path, default=default_dataset_root(), help="ArtSpeech dataset root. Defaults to ../Data/Artspeech_database.")
    parser.add_argument("--speakers", default=",".join(DEFAULT_SPEAKERS), help="Comma-separated curated speakers. Defaults to P1..P10.")
    parser.add_argument("--vowels", default=",".join(DEFAULT_VOWELS), help="Comma-separated vowels for token variability. Defaults to a,e,i,o,u.")
    parser.add_argument("--samples-per-vowel", type=int, default=10, help="Maximum number of extracted ArtSpeech samples per vowel and speaker. Defaults to 10.")
    parser.add_argument("--output-dir", type=Path, default=default_output_dir(), help="Root output directory for this combined experiment.")
    parser.add_argument("--reference-speaker", default="auto", help="Reference speaker for the grid-transform stage. Default 'auto' uses the rigid-GPA geometric median.")
    parser.add_argument("--guided-factor", default="gender", choices=["gender"], help="Guided factor for the supervised axis. Currently only 'gender' is supported.")
    parser.add_argument("--nc-resample", type=int, default=DEFAULT_RESAMPLE_POINTS, help="Resample count for the rigid-GPA stage. Defaults to 80.")
    parser.add_argument("--nc-template", type=int, default=DEFAULT_TEMPLATE_POINTS, help="Resample count for transformed contour metrics and guided PCA. Defaults to 80.")
    parser.add_argument("--max-shift", type=int, default=DEFAULT_MAX_SHIFT, help="Maximum alignment shift in pixels for within-speaker ArtSpeech analysis. Defaults to 24.")
    parser.add_argument("--alignment-passes", type=int, default=DEFAULT_ALIGNMENT_PASSES, help="Number of alignment passes for within-speaker ArtSpeech analysis. Defaults to 2.")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    speaker_ids = parse_speakers(args.speakers)
    vowels = parse_vowels(args.vowels)
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    shape_result = run_shape_analysis(
        vtln_dir=args.vtln_dir,
        speaker_ids=speaker_ids,
        output_dir=output_dir,
        reference_speaker=args.reference_speaker,
        nc_resample=args.nc_resample,
        nc_template=args.nc_template,
    )
    token_result = run_within_speaker_variability(
        dataset_root=args.dataset_root,
        speaker_ids=speaker_ids,
        vowels=vowels,
        samples_per_vowel=args.samples_per_vowel,
        max_shift=args.max_shift,
        alignment_passes=args.alignment_passes,
        output_dir=output_dir,
    )
    integration_result = run_integration(shape_result, token_result, output_dir)

    root_summary = {
        "experiment": "average_speaker_and_speaker_variability",
        "guided_factor": args.guided_factor,
        "output_dir": str(output_dir),
        "speakers": speaker_ids,
        "vowels": vowels,
        "samples_per_vowel": int(args.samples_per_vowel),
        "reference_speaker": shape_result["summary"]["reference_speaker"],
        "geometric_median_speaker": shape_result["summary"]["geometric_median_speaker"],
        "closest_to_mean_speaker": shape_result["summary"]["closest_to_mean_speaker"],
        "paths": {
            "shape_analysis_dir": str(shape_result["stage_dir"]),
            "token_variability_dir": str(token_result["stage_dir"]),
            "integration_dir": str(integration_result["stage_dir"]),
            "integrated_metrics_csv": str(integration_result["stage_dir"] / "integrated_speaker_metrics.csv"),
        },
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(root_summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Shape analysis dir: {shape_result['stage_dir']}")
    print(f"Token variability dir: {token_result['stage_dir']}")
    print(f"Integration dir: {integration_result['stage_dir']}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
