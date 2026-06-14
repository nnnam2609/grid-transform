from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Collection

import numpy as np

from grid_transform.analysis_shared import LoadedSpeaker, choose_label_colors, speaker_id_sort_key
from grid_transform.roi_average_speaker import (
    DEFAULT_EXCLUDE_LABELS,
    build_affine_only_transform,
    compute_mask_overlap_metrics,
    load_roi_average_speakers,
    polygon_to_mask,
)
from grid_transform.transfer import build_two_step_transform, smooth_transformed_contours, transform_contours
from grid_transform.transform_helpers import format_target_frame


DEFAULT_TARGET_SPEAKER_ID = "P3"
DEFAULT_TARGET_BASENAME = "1618_P3_S14_F1556"
DEFAULT_CLOSED_ROI_LABELS = (
    "c1",
    "c2",
    "c3",
    "c4",
    "c5",
    "c6",
    "incisior-hard-palate",
    "mandible-incisior",
    "soft-palate",
)
DEFAULT_STAGE_ORDER = ("before_affine", "after_affine", "after_tps")


def resolve_closed_roi_labels(
    speakers: dict[str, LoadedSpeaker],
    exclude_labels: Collection[str] = DEFAULT_EXCLUDE_LABELS,
    allowed_labels: Collection[str] = DEFAULT_CLOSED_ROI_LABELS,
) -> tuple[list[str], list[dict[str, object]]]:
    """Resolve the closed ROI labels used by the P3 stage-variance workflow."""
    if not speakers:
        raise ValueError("No speakers were loaded.")

    exclude_set = {label.strip() for label in exclude_labels if label.strip()}
    allowed_set = {label.strip() for label in allowed_labels if label.strip()}
    all_labels = sorted({label for speaker in speakers.values() for label in speaker.contours})
    ordered_speakers = sorted(speakers, key=speaker_id_sort_key)

    included: list[str] = []
    excluded_rows: list[dict[str, object]] = []
    for label in all_labels:
        reasons: list[dict[str, object]] = []
        if label in exclude_set:
            reasons.append({"type": "explicit_exclude"})
        if label not in allowed_set:
            reasons.append({"type": "not_in_closed_roi_allowlist"})
        missing_speakers = [speaker_id for speaker_id in ordered_speakers if label not in speakers[speaker_id].contours]
        if missing_speakers:
            reasons.append({"type": "missing_from_global_intersection", "speakers": missing_speakers})
        short_speakers = [
            speaker_id
            for speaker_id in ordered_speakers
            if label in speakers[speaker_id].contours and len(np.asarray(speakers[speaker_id].contours[label], dtype=float)) < 3
        ]
        if short_speakers:
            reasons.append({"type": "too_few_points", "speakers": short_speakers})

        if reasons:
            excluded_rows.append({"label": label, "reasons": reasons})
        else:
            included.append(label)

    included = [label for label in DEFAULT_CLOSED_ROI_LABELS if label in included]
    if not included:
        raise ValueError("No closed ROI labels remained after filtering.")
    return included, excluded_rows


def load_p3_stage_overlay_speakers(
    vtln_dir: Path,
    requested_speakers: list[str] | None = None,
) -> dict[str, LoadedSpeaker]:
    """Load the curated speaker set and verify that the fixed P3 reference is present."""
    speakers = load_roi_average_speakers(vtln_dir, requested_speakers)
    if DEFAULT_TARGET_SPEAKER_ID not in speakers:
        raise ValueError(f"Fixed target speaker {DEFAULT_TARGET_SPEAKER_ID} was not loaded.")
    target = speakers[DEFAULT_TARGET_SPEAKER_ID]
    if target.spec.basename != DEFAULT_TARGET_BASENAME:
        raise ValueError(
            f"Expected {DEFAULT_TARGET_SPEAKER_ID} basename {DEFAULT_TARGET_BASENAME}, got {target.spec.basename}."
        )
    return speakers


def contour_to_mask(points: np.ndarray, image_shape: tuple[int, ...]) -> np.ndarray:
    """Rasterize one contour as a closed ROI mask on the shared 480x480 canvas."""
    return polygon_to_mask(points, image_shape)


def dice_score(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    """Return Dice overlap between two ROI masks."""
    return float(compute_mask_overlap_metrics(mask_a, mask_b)["dice"])


def dice_distance(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    """Return Dice distance between two ROI masks."""
    return float(1.0 - dice_score(mask_a, mask_b))


def map_speakers_to_p3_stages(
    speakers: dict[str, LoadedSpeaker],
    labels: list[str],
    *,
    target_speaker_id: str = DEFAULT_TARGET_SPEAKER_ID,
) -> dict[str, dict[str, dict[str, np.ndarray]]]:
    """Build contour dictionaries for the three requested stages in P3 space."""
    target = speakers[target_speaker_id]
    stage_contours: dict[str, dict[str, dict[str, np.ndarray]]] = {
        "before_affine": {},
        "after_affine": {},
        "after_tps": {},
    }

    for speaker_id, speaker in speakers.items():
        raw_contours = {label: np.asarray(speaker.contours[label], dtype=float) for label in labels}
        stage_contours["before_affine"][speaker_id] = raw_contours

        if speaker_id == target_speaker_id:
            stage_contours["after_affine"][speaker_id] = {label: np.asarray(points, dtype=float).copy() for label, points in raw_contours.items()}
            stage_contours["after_tps"][speaker_id] = {label: np.asarray(points, dtype=float).copy() for label, points in raw_contours.items()}
            continue

        affine_transform = build_affine_only_transform(speaker.grid, target.grid)
        affine_contours = {
            label: affine_transform["apply_affine"](np.asarray(speaker.contours[label], dtype=float))
            for label in labels
        }
        stage_contours["after_affine"][speaker_id] = {
            label: np.asarray(points, dtype=float)
            for label, points in affine_contours.items()
        }

        two_step = build_two_step_transform(speaker.grid, target.grid)
        tps_contours = transform_contours(speaker.contours, two_step["apply_two_step"], labels)
        stage_contours["after_tps"][speaker_id] = smooth_transformed_contours(tps_contours)

    return stage_contours


def build_stage_masks(
    stage_contours: dict[str, dict[str, dict[str, np.ndarray]]],
    labels: list[str],
    image_shape: tuple[int, ...],
) -> dict[str, dict[str, dict[str, np.ndarray]]]:
    """Rasterize all stage contours into binary masks."""
    masks: dict[str, dict[str, dict[str, np.ndarray]]] = {}
    for stage_name, speaker_contours in stage_contours.items():
        masks[stage_name] = {}
        for speaker_id, contours in speaker_contours.items():
            masks[stage_name][speaker_id] = {
                label: contour_to_mask(np.asarray(contours[label], dtype=float), image_shape)
                for label in labels
            }
    return masks


@dataclass(frozen=True)
class StageLabelVariance:
    stage: str
    label: str
    variance: float
    mean_pairwise_dice: float
    mean_pairwise_dice_distance: float

    def to_row(self) -> dict[str, object]:
        return {
            "stage": self.stage,
            "label": self.label,
            "variance": self.variance,
            "mean_pairwise_dice": self.mean_pairwise_dice,
            "mean_pairwise_dice_distance": self.mean_pairwise_dice_distance,
        }


@dataclass(frozen=True)
class StageSummary:
    stage: str
    overall_variance: float
    overall_mean_pairwise_dice: float
    overall_mean_pairwise_dice_distance: float
    num_speakers: int
    num_labels: int

    def to_row(self) -> dict[str, object]:
        return {
            "stage": self.stage,
            "overall_variance": self.overall_variance,
            "overall_mean_pairwise_dice": self.overall_mean_pairwise_dice,
            "overall_mean_pairwise_dice_distance": self.overall_mean_pairwise_dice_distance,
            "num_speakers": self.num_speakers,
            "num_labels": self.num_labels,
        }


def compute_pairwise_stage_dice(
    stage: str,
    stage_masks: dict[str, dict[str, np.ndarray]],
    labels: list[str],
) -> tuple[list[dict[str, object]], list[StageLabelVariance], StageSummary]:
    """Compute pairwise Dice details plus label/stage summaries for one stage."""
    ordered_ids = sorted(stage_masks, key=speaker_id_sort_key)
    pairwise_rows: list[dict[str, object]] = []
    label_rows: list[StageLabelVariance] = []

    for label in labels:
        label_pair_rows: list[dict[str, object]] = []
        for index, speaker_i in enumerate(ordered_ids):
            for speaker_j in ordered_ids[index + 1 :]:
                metrics = compute_mask_overlap_metrics(stage_masks[speaker_i][label], stage_masks[speaker_j][label])
                dice = float(metrics["dice"])
                distance = float(1.0 - dice)
                row = {
                    "stage": stage,
                    "label": label,
                    "speaker_i": speaker_i,
                    "speaker_j": speaker_j,
                    "dice": dice,
                    "dice_distance": distance,
                    "dice_distance_squared": float(distance**2),
                    "intersection": int(metrics["intersection"]),
                    "area_i": int(metrics["source_area"]),
                    "area_j": int(metrics["target_area"]),
                    "union": int(metrics["union"]),
                }
                pairwise_rows.append(row)
                label_pair_rows.append(row)

        label_rows.append(
            StageLabelVariance(
                stage=stage,
                label=label,
                variance=float(np.mean([row["dice_distance_squared"] for row in label_pair_rows])),
                mean_pairwise_dice=float(np.mean([row["dice"] for row in label_pair_rows])),
                mean_pairwise_dice_distance=float(np.mean([row["dice_distance"] for row in label_pair_rows])),
            )
        )

    summary = StageSummary(
        stage=stage,
        overall_variance=float(np.mean([row.variance for row in label_rows])),
        overall_mean_pairwise_dice=float(np.mean([row.mean_pairwise_dice for row in label_rows])),
        overall_mean_pairwise_dice_distance=float(np.mean([row.mean_pairwise_dice_distance for row in label_rows])),
        num_speakers=len(ordered_ids),
        num_labels=len(labels),
    )
    return pairwise_rows, label_rows, summary


def compute_stage_variance(
    stage_masks: dict[str, dict[str, dict[str, np.ndarray]]],
    labels: list[str],
) -> dict[str, object]:
    """Compute all pairwise stage Dice metrics for before_affine / after_affine / after_tps."""
    pairwise_rows: list[dict[str, object]] = []
    label_rows: list[dict[str, object]] = []
    stage_summaries: dict[str, StageSummary] = {}

    for stage in DEFAULT_STAGE_ORDER:
        stage_pair_rows, stage_label_rows, stage_summary = compute_pairwise_stage_dice(stage, stage_masks[stage], labels)
        pairwise_rows.extend(stage_pair_rows)
        label_rows.extend(row.to_row() for row in stage_label_rows)
        stage_summaries[stage] = stage_summary

    return {
        "pairwise_rows": pairwise_rows,
        "label_rows": label_rows,
        "stage_summaries": stage_summaries,
    }


def compute_source_to_target_stage_metrics(
    stage_masks: dict[str, dict[str, dict[str, np.ndarray]]],
    labels: list[str],
    *,
    target_speaker_id: str = DEFAULT_TARGET_SPEAKER_ID,
) -> list[dict[str, object]]:
    """Compute per-source-to-P3 Dice summaries for each stage."""
    rows: list[dict[str, object]] = []
    ordered_ids = sorted(stage_masks["before_affine"], key=speaker_id_sort_key)
    for stage in DEFAULT_STAGE_ORDER:
        for source_id in ordered_ids:
            if source_id == target_speaker_id:
                continue
            dice_values: list[float] = []
            distance_values: list[float] = []
            squared_values: list[float] = []
            for label in labels:
                metrics = compute_mask_overlap_metrics(
                    stage_masks[stage][source_id][label],
                    stage_masks[stage][target_speaker_id][label],
                )
                dice = float(metrics["dice"])
                distance = float(1.0 - dice)
                dice_values.append(dice)
                distance_values.append(distance)
                squared_values.append(distance**2)
            rows.append(
                {
                    "stage": stage,
                    "source_speaker": source_id,
                    "target_speaker": target_speaker_id,
                    "mean_dice": float(np.mean(dice_values)),
                    "mean_dice_distance": float(np.mean(distance_values)),
                    "mean_squared_dice_distance": float(np.mean(squared_values)),
                }
            )
    return rows


def _choose_speaker_colors(speakers: dict[str, LoadedSpeaker]) -> dict[str, object]:
    import matplotlib.pyplot as plt

    cmap = plt.get_cmap("tab20")
    ordered_ids = sorted(speakers, key=speaker_id_sort_key)
    return {speaker_id: cmap(index % 20) for index, speaker_id in enumerate(ordered_ids)}


def _stage_display_name(stage: str) -> str:
    if stage == "before_affine":
        return "Before affine"
    if stage == "after_affine":
        return "After affine"
    if stage == "after_tps":
        return "After TPS"
    return stage


def _set_overlay_view_limits(ax, image, point_sets: list[np.ndarray], *, padding: float) -> None:
    image_arr = np.asarray(image)
    height, width = image_arr.shape[:2]
    all_points = np.vstack([np.asarray(points, dtype=float) for points in point_sets if len(points)])
    min_x = float(np.min(all_points[:, 0]))
    max_x = float(np.max(all_points[:, 0]))
    min_y = float(np.min(all_points[:, 1]))
    max_y = float(np.max(all_points[:, 1]))
    center_x = 0.5 * (min_x + max_x)
    center_y = 0.5 * (min_y + max_y)
    half_extent = 0.5 * max(max_x - min_x, max_y - min_y) + padding
    x0 = max(0.0, center_x - half_extent)
    x1 = min(float(width), center_x + half_extent)
    y0 = max(0.0, center_y - half_extent)
    y1 = min(float(height), center_y + half_extent)
    ax.set_xlim(x0, x1)
    ax.set_ylim(y1, y0)


def _create_stage_overlay_figure(title: str):
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(11.4, 12.7), dpi=220, facecolor="white")
    grid = fig.add_gridspec(4, 1, height_ratios=(0.09, 0.76, 0.08, 0.07))
    title_ax = fig.add_subplot(grid[0, 0])
    image_ax = fig.add_subplot(grid[1, 0])
    note_ax = fig.add_subplot(grid[2, 0])
    legend_ax = fig.add_subplot(grid[3, 0])
    for ax in (title_ax, note_ax, legend_ax):
        ax.axis("off")
    title_ax.text(0.5, 0.52, title, ha="center", va="center", fontsize=19, fontweight="bold")
    fig.subplots_adjust(left=0.02, right=0.98, top=0.985, bottom=0.03, hspace=0.03)
    return fig, image_ax, note_ax, legend_ax


def _add_stage_note(note_ax, note_text: str) -> None:
    note_ax.text(
        0.01,
        0.97,
        note_text,
        ha="left",
        va="top",
        fontsize=12.5,
        bbox=dict(boxstyle="round,pad=0.45", fc="white", ec="0.80", alpha=0.98),
    )


def _add_stage_legend(legend_ax, ordered_ids: list[str], speaker_colors: dict[str, object], target_speaker_id: str) -> None:
    import matplotlib.pyplot as plt

    handles = []
    labels = []
    for speaker_id in ordered_ids:
        is_target = speaker_id == target_speaker_id
        handles.append(
            plt.Line2D(
                [0],
                [0],
                color=speaker_colors[speaker_id],
                lw=3.2 if is_target else 1.8,
                alpha=0.98 if is_target else 0.82,
            )
        )
        labels.append(f"{speaker_id} (target)" if is_target else speaker_id)
    legend_ax.legend(
        handles,
        labels,
        loc="center",
        ncol=min(6, len(handles)),
        fontsize=10.5,
        framealpha=0.96,
        handlelength=2.6,
        columnspacing=1.25,
    )


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})


def _percent_reduction(before: float, after: float) -> float | None:
    if abs(before) < 1e-12:
        return None
    return float(((before - after) / before) * 100.0)


def _percent_change(before: float, after: float) -> float | None:
    if abs(before) < 1e-12:
        return None
    return float(((after - before) / before) * 100.0)


def _save_stage_overlay_figure(
    stage: str,
    target: LoadedSpeaker,
    labels: list[str],
    stage_contours: dict[str, dict[str, np.ndarray]],
    stage_summary: StageSummary,
    speaker_colors: dict[str, object],
    output_path: Path,
    *,
    target_speaker_id: str = DEFAULT_TARGET_SPEAKER_ID,
) -> None:
    import matplotlib.pyplot as plt

    ordered_ids = sorted(stage_contours, key=speaker_id_sort_key)
    draw_order = [speaker_id for speaker_id in ordered_ids if speaker_id != target_speaker_id] + [target_speaker_id]
    fig, image_ax, note_ax, legend_ax = _create_stage_overlay_figure(
        f"{_stage_display_name(stage)}: contour overlay (all speakers, target P3)"
    )
    format_target_frame(image_ax, target.image, "")
    image_ax.set_title("")

    all_points: list[np.ndarray] = []
    for speaker_id in draw_order:
        is_target = speaker_id == target_speaker_id
        for label in labels:
            pts = np.asarray(stage_contours[speaker_id][label], dtype=float)
            all_points.append(pts)
            image_ax.plot(
                pts[:, 0],
                pts[:, 1],
                color=speaker_colors[speaker_id],
                lw=2.8 if is_target else 1.15,
                alpha=0.98 if is_target else 0.24,
                zorder=5 if is_target else 2,
            )

    _set_overlay_view_limits(image_ax, target.image, all_points, padding=26.0)
    note_text = "\n".join(
        [
            f"Target: {DEFAULT_TARGET_SPEAKER_ID} / {target.spec.basename}",
            f"Speakers: {stage_summary.num_speakers}",
            f"Labels: {stage_summary.num_labels}",
            f"Overall variance: {stage_summary.overall_variance:.5f}",
            f"Mean pairwise Dice: {stage_summary.overall_mean_pairwise_dice:.4f}",
            f"Mean Dice distance: {stage_summary.overall_mean_pairwise_dice_distance:.4f}",
        ]
    )
    _add_stage_note(note_ax, note_text)
    _add_stage_legend(legend_ax, ordered_ids, speaker_colors, target_speaker_id)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight", pad_inches=0.04, facecolor="white")
    plt.close(fig)


def _save_stage_label_variance_figure(
    stage: str,
    label_rows: list[dict[str, object]],
    output_path: Path,
) -> None:
    import matplotlib.pyplot as plt

    stage_rows = [row for row in label_rows if row["stage"] == stage]
    stage_rows = sorted(stage_rows, key=lambda row: float(row["variance"]), reverse=True)
    label_names = [str(row["label"]) for row in stage_rows]
    variance_values = [float(row["variance"]) for row in stage_rows]
    dice_values = [float(row["mean_pairwise_dice"]) for row in stage_rows]
    colors = choose_label_colors(label_names)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6.8))
    y_pos = np.arange(len(stage_rows))

    axes[0].barh(y_pos, variance_values, color=[colors[label] for label in label_names], edgecolor="black", alpha=0.86)
    axes[0].set_yticks(y_pos)
    axes[0].set_yticklabels(label_names, fontsize=9)
    axes[0].set_xlabel("Variance = mean squared Dice distance")
    axes[0].set_title(f"{_stage_display_name(stage)}: label variance", fontsize=14, fontweight="bold")
    axes[0].invert_yaxis()
    axes[0].grid(True, axis="x", alpha=0.25)

    axes[1].barh(y_pos, dice_values, color=[colors[label] for label in label_names], edgecolor="black", alpha=0.86)
    axes[1].set_yticks(y_pos)
    axes[1].set_yticklabels(label_names, fontsize=9)
    axes[1].set_xlabel("Mean pairwise Dice")
    axes[1].set_title(f"{_stage_display_name(stage)}: label mean Dice", fontsize=14, fontweight="bold")
    axes[1].invert_yaxis()
    axes[1].set_xlim(0.0, 1.0)
    axes[1].grid(True, axis="x", alpha=0.25)

    fig.suptitle(f"{_stage_display_name(stage)} ROI label summary (target fixed to P3 space)", fontsize=16, fontweight="bold")
    plt.tight_layout(rect=(0, 0, 1, 0.95))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _save_stage_variance_summary_figure(
    stage_rows: list[dict[str, object]],
    reductions: dict[str, float | None],
    output_path: Path,
) -> None:
    import matplotlib.pyplot as plt

    ordered = [row for stage in DEFAULT_STAGE_ORDER for row in stage_rows if row["stage"] == stage]
    stages = [str(row["stage"]) for row in ordered]
    stage_labels = [_stage_display_name(stage) for stage in stages]
    variances = [float(row["overall_variance"]) for row in ordered]
    mean_dice = [float(row["overall_mean_pairwise_dice"]) for row in ordered]
    mean_distance = [float(row["overall_mean_pairwise_dice_distance"]) for row in ordered]
    colors = ["#9b2226", "#2a9d8f", "#1d3557"]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6.5))
    x = np.arange(len(ordered))

    axes[0].bar(x, variances, color=colors, edgecolor="black", alpha=0.86)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(stage_labels, rotation=18, ha="right")
    axes[0].set_ylabel("Overall variance")
    axes[0].set_title("Mean pairwise squared Dice distance", fontsize=13, fontweight="bold")
    axes[0].grid(True, axis="y", alpha=0.25)

    axes[1].bar(x, mean_dice, color=colors, edgecolor="black", alpha=0.86)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(stage_labels, rotation=18, ha="right")
    axes[1].set_ylim(0.0, 1.0)
    axes[1].set_ylabel("Overall mean pairwise Dice")
    axes[1].set_title("ROI similarity by stage", fontsize=13, fontweight="bold")
    axes[1].grid(True, axis="y", alpha=0.25)

    axes[2].bar(x, mean_distance, color=colors, edgecolor="black", alpha=0.86)
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(stage_labels, rotation=18, ha="right")
    axes[2].set_ylabel("Overall mean Dice distance")
    axes[2].set_title("ROI distance by stage", fontsize=13, fontweight="bold")
    axes[2].grid(True, axis="y", alpha=0.25)
    axes[2].text(
        0.02,
        0.02,
        "\n".join(
            [
                f"Affine vs before: {reductions['affine_vs_before']:.2f}%" if reductions["affine_vs_before"] is not None else "Affine vs before: n/a",
                f"TPS vs before: {reductions['tps_vs_before']:.2f}%" if reductions["tps_vs_before"] is not None else "TPS vs before: n/a",
                f"TPS vs affine: {reductions['tps_vs_affine']:.2f}%" if reductions["tps_vs_affine"] is not None else "TPS vs affine: n/a",
            ]
        ),
        transform=axes[2].transAxes,
        ha="left",
        va="bottom",
        fontsize=9.5,
        bbox=dict(boxstyle="round,pad=0.35", fc="white", alpha=0.96),
    )

    fig.suptitle("P3 fixed-target stage variance summary", fontsize=16, fontweight="bold")
    plt.tight_layout(rect=(0, 0, 1, 0.95))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def build_p3_stage_overlays(
    speakers: dict[str, LoadedSpeaker],
    stage_contours: dict[str, dict[str, dict[str, np.ndarray]]],
    labels: list[str],
    stage_summaries: dict[str, StageSummary],
    label_rows: list[dict[str, object]],
    output_dir: Path,
) -> dict[str, Path]:
    """Generate V1 overlay figures and summary plots under outputs/stage_overlays_v1."""
    target = speakers[DEFAULT_TARGET_SPEAKER_ID]
    speaker_colors = _choose_speaker_colors(speakers)
    paths = {
        "before_overlay": output_dir / "001_all_before_affine_contours_overlay.png",
        "affine_overlay": output_dir / "002_all_after_affine_contours_overlay.png",
        "tps_overlay": output_dir / "003_all_after_tps_contours_overlay.png",
        "variance_summary": output_dir / "004_stage_variance_summary.png",
        "before_label": output_dir / "005_before_affine_label_variance.png",
        "affine_label": output_dir / "006_after_affine_label_variance.png",
        "tps_label": output_dir / "007_after_tps_label_variance.png",
    }

    _save_stage_overlay_figure(
        "before_affine",
        target,
        labels,
        stage_contours["before_affine"],
        stage_summaries["before_affine"],
        speaker_colors,
        paths["before_overlay"],
    )
    _save_stage_overlay_figure(
        "after_affine",
        target,
        labels,
        stage_contours["after_affine"],
        stage_summaries["after_affine"],
        speaker_colors,
        paths["affine_overlay"],
    )
    _save_stage_overlay_figure(
        "after_tps",
        target,
        labels,
        stage_contours["after_tps"],
        stage_summaries["after_tps"],
        speaker_colors,
        paths["tps_overlay"],
    )

    _save_stage_label_variance_figure("before_affine", label_rows, paths["before_label"])
    _save_stage_label_variance_figure("after_affine", label_rows, paths["affine_label"])
    _save_stage_label_variance_figure("after_tps", label_rows, paths["tps_label"])

    stage_rows = [stage_summaries[stage].to_row() for stage in DEFAULT_STAGE_ORDER]
    reductions = {
        "affine_vs_before": _percent_reduction(stage_summaries["before_affine"].overall_variance, stage_summaries["after_affine"].overall_variance),
        "tps_vs_before": _percent_reduction(stage_summaries["before_affine"].overall_variance, stage_summaries["after_tps"].overall_variance),
        "tps_vs_affine": _percent_reduction(stage_summaries["after_affine"].overall_variance, stage_summaries["after_tps"].overall_variance),
    }
    _save_stage_variance_summary_figure(stage_rows, reductions, paths["variance_summary"])
    return paths


def write_stage_overlay_v1_report(
    speakers: dict[str, LoadedSpeaker],
    labels: list[str],
    excluded_labels: list[dict[str, object]],
    stage_contours: dict[str, dict[str, dict[str, np.ndarray]]],
    stage_metrics: dict[str, object],
    source_to_target_rows: list[dict[str, object]],
    output_dir: Path,
) -> dict[str, object]:
    """Write figures plus detailed CSV/JSON/Markdown outputs for the fixed-P3 stage-variance workflow."""
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_json_path = output_dir / "summary.json"
    summary_md_path = output_dir / "summary.md"
    stage_variance_csv_path = output_dir / "stage_variance.csv"
    stage_label_variance_csv_path = output_dir / "stage_label_variance.csv"
    pairwise_csv_path = output_dir / "pairwise_dice_details.csv"
    source_target_csv_path = output_dir / "per_source_to_target.csv"

    stage_summaries: dict[str, StageSummary] = stage_metrics["stage_summaries"]
    stage_rows = [stage_summaries[stage].to_row() for stage in DEFAULT_STAGE_ORDER]
    label_rows = list(stage_metrics["label_rows"])
    pairwise_rows = list(stage_metrics["pairwise_rows"])

    _write_csv(
        stage_variance_csv_path,
        [
            "stage",
            "overall_variance",
            "overall_mean_pairwise_dice",
            "overall_mean_pairwise_dice_distance",
            "num_speakers",
            "num_labels",
        ],
        stage_rows,
    )
    _write_csv(
        stage_label_variance_csv_path,
        [
            "stage",
            "label",
            "variance",
            "mean_pairwise_dice",
            "mean_pairwise_dice_distance",
        ],
        label_rows,
    )
    _write_csv(
        pairwise_csv_path,
        [
            "stage",
            "label",
            "speaker_i",
            "speaker_j",
            "dice",
            "dice_distance",
            "dice_distance_squared",
            "intersection",
            "area_i",
            "area_j",
            "union",
        ],
        pairwise_rows,
    )
    _write_csv(
        source_target_csv_path,
        [
            "stage",
            "source_speaker",
            "target_speaker",
            "mean_dice",
            "mean_dice_distance",
            "mean_squared_dice_distance",
        ],
        source_to_target_rows,
    )

    figure_paths = build_p3_stage_overlays(speakers, stage_contours, labels, stage_summaries, label_rows, output_dir)

    before = stage_summaries["before_affine"]
    affine = stage_summaries["after_affine"]
    tps = stage_summaries["after_tps"]
    variance_reductions = {
        "affine_vs_before": _percent_reduction(before.overall_variance, affine.overall_variance),
        "tps_vs_before": _percent_reduction(before.overall_variance, tps.overall_variance),
        "tps_vs_affine": _percent_reduction(affine.overall_variance, tps.overall_variance),
    }
    dice_distance_reductions = {
        "affine_vs_before": _percent_reduction(
            before.overall_mean_pairwise_dice_distance,
            affine.overall_mean_pairwise_dice_distance,
        ),
        "tps_vs_before": _percent_reduction(
            before.overall_mean_pairwise_dice_distance,
            tps.overall_mean_pairwise_dice_distance,
        ),
        "tps_vs_affine": _percent_reduction(
            affine.overall_mean_pairwise_dice_distance,
            tps.overall_mean_pairwise_dice_distance,
        ),
    }
    dice_changes = {
        "affine_vs_before": _percent_change(
            before.overall_mean_pairwise_dice,
            affine.overall_mean_pairwise_dice,
        ),
        "tps_vs_before": _percent_change(
            before.overall_mean_pairwise_dice,
            tps.overall_mean_pairwise_dice,
        ),
        "tps_vs_affine": _percent_change(
            affine.overall_mean_pairwise_dice,
            tps.overall_mean_pairwise_dice,
        ),
    }

    target = speakers[DEFAULT_TARGET_SPEAKER_ID]
    summary_payload = {
        "target_speaker": {
            "speaker_id": DEFAULT_TARGET_SPEAKER_ID,
            "basename": target.spec.basename,
            "gender": target.spec.gender,
        },
        "labels_used": labels,
        "labels_excluded": excluded_labels,
        "speaker_list_used": [
            {
                "speaker_id": speaker_id,
                "basename": speakers[speaker_id].spec.basename,
                "gender": speakers[speaker_id].spec.gender,
            }
            for speaker_id in sorted(speakers, key=speaker_id_sort_key)
        ],
        "overall_variance": {stage: stage_summaries[stage].overall_variance for stage in DEFAULT_STAGE_ORDER},
        "overall_mean_pairwise_dice": {stage: stage_summaries[stage].overall_mean_pairwise_dice for stage in DEFAULT_STAGE_ORDER},
        "overall_mean_pairwise_dice_distance": {stage: stage_summaries[stage].overall_mean_pairwise_dice_distance for stage in DEFAULT_STAGE_ORDER},
        "reduction_percent": {
            "variance": variance_reductions,
            "mean_pairwise_dice_distance": dice_distance_reductions,
        },
        "change_percent": {
            "mean_pairwise_dice": dice_changes,
        },
        "paths": {
            "summary_md": str(summary_md_path),
            "stage_variance_csv": str(stage_variance_csv_path),
            "stage_label_variance_csv": str(stage_label_variance_csv_path),
            "pairwise_dice_details_csv": str(pairwise_csv_path),
            "per_source_to_target_csv": str(source_target_csv_path),
            **{key: str(value) for key, value in figure_paths.items()},
        },
    }
    summary_json_path.write_text(json.dumps(summary_payload, indent=2, ensure_ascii=False), encoding="utf-8")

    lines = [
        "# P3 Stage Overlay Variance Summary",
        "",
        "Target/reference speaker is fixed to `P3 / 1618_P3_S14_F1556`.",
        "",
        "## Metric definition",
        "",
        "- Binary mask per contour label on the shared `480x480` canvas.",
        "- Dice(A,B) = `2 * intersection / (|A| + |B|)`.",
        "- Dice distance = `1 - Dice`.",
        "- Label variance = mean pairwise squared Dice distance over all speaker pairs.",
        "- Overall stage variance = mean of the label variances with equal label weight.",
        "",
        f"Labels used ({len(labels)}): {', '.join(labels)}",
        "",
        "## Stage summary",
        "",
        "| stage | overall variance | mean pairwise Dice | mean Dice distance |",
        "| --- | ---: | ---: | ---: |",
    ]
    for stage in DEFAULT_STAGE_ORDER:
        summary = stage_summaries[stage]
        lines.append(
            f"| {stage} | {summary.overall_variance:.6f} | "
            f"{summary.overall_mean_pairwise_dice:.4f} | {summary.overall_mean_pairwise_dice_distance:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Variance reduction",
            "",
            f"- affine vs before: {variance_reductions['affine_vs_before']:.2f}%" if variance_reductions["affine_vs_before"] is not None else "- affine vs before: n/a",
            f"- tps vs before: {variance_reductions['tps_vs_before']:.2f}%" if variance_reductions["tps_vs_before"] is not None else "- tps vs before: n/a",
            f"- tps vs affine: {variance_reductions['tps_vs_affine']:.2f}%" if variance_reductions["tps_vs_affine"] is not None else "- tps vs affine: n/a",
            "",
            "## Dice summary change",
            "",
            f"- mean Dice change, affine vs before: {dice_changes['affine_vs_before']:.2f}%" if dice_changes["affine_vs_before"] is not None else "- mean Dice change, affine vs before: n/a",
            f"- mean Dice change, tps vs before: {dice_changes['tps_vs_before']:.2f}%" if dice_changes["tps_vs_before"] is not None else "- mean Dice change, tps vs before: n/a",
            f"- mean Dice change, tps vs affine: {dice_changes['tps_vs_affine']:.2f}%" if dice_changes["tps_vs_affine"] is not None else "- mean Dice change, tps vs affine: n/a",
            "",
            "## Key findings",
            "",
            "- `before_affine` uses the raw shared 480x480 canvas without speaker-to-P3 mapping.",
            "- `after_affine` maps every non-P3 speaker into P3 space with Step 1 affine only.",
            "- `after_tps` maps every non-P3 speaker into P3 space with full affine + TPS.",
        ]
    )
    summary_md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    return {
        "summary_json": summary_json_path,
        "summary_md": summary_md_path,
        "stage_variance_csv": stage_variance_csv_path,
        "stage_label_variance_csv": stage_label_variance_csv_path,
        "pairwise_csv": pairwise_csv_path,
        "per_source_to_target_csv": source_target_csv_path,
        "figure_paths": figure_paths,
        "variance_reductions": variance_reductions,
        "dice_distance_reductions": dice_distance_reductions,
        "dice_changes": dice_changes,
    }


__all__ = [
    "DEFAULT_TARGET_SPEAKER_ID",
    "DEFAULT_TARGET_BASENAME",
    "DEFAULT_CLOSED_ROI_LABELS",
    "DEFAULT_STAGE_ORDER",
    "resolve_closed_roi_labels",
    "load_p3_stage_overlay_speakers",
    "contour_to_mask",
    "dice_score",
    "dice_distance",
    "map_speakers_to_p3_stages",
    "build_stage_masks",
    "compute_pairwise_stage_dice",
    "compute_stage_variance",
    "compute_source_to_target_stage_metrics",
    "build_p3_stage_overlays",
    "write_stage_overlay_v1_report",
]
