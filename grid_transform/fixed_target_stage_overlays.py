from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from grid_transform.analysis_shared import LoadedSpeaker, load_curated_speakers, speaker_id_sort_key
from grid_transform.roi_average_speaker import (
    compute_mask_overlap_metrics,
    default_curated_speaker_ids,
    polygon_to_mask,
)
from grid_transform.transfer import build_two_step_transform, smooth_transformed_contours, transform_contours
from grid_transform.transform_helpers import apply_transform, resample_polyline


DEFAULT_TARGET_SPEAKER_ID = "P10"
DEFAULT_TARGET_BASENAME = "1662_P10_S14_F0110"
DEFAULT_COHORTS = ("all", "male", "female")
DEFAULT_STAGE_ORDER = ("init", "affine", "tps")
DEFAULT_STAGE_EXPORT_COHORTS = ("all", "female", "male")
DEFAULT_STAGE_EXPORT_ORDER = ("affine", "init", "tps")
DEFAULT_STAGE_ALPHA = 0.28
DEFAULT_STAGE_GRID_ALPHA = 0.24
DEFAULT_STAGE_LINEWIDTH = 1.15
DEFAULT_STAGE_HIGHLIGHT_LINEWIDTH = 2.8
DEFAULT_STAGE_FIGURE_SIZE = (11.4, 12.7)
DEFAULT_STAGE_TITLE_SIZE = 23
DEFAULT_STAGE_NOTE_SIZE = 13.5
DEFAULT_STAGE_LEGEND_SIZE = 12.5
DEFAULT_STAGE_LAYOUT_VERSION = "V2"
DEFAULT_OPEN_VISUAL_LABELS = ("pharynx",)
DEFAULT_VARIANCE_EXCLUDE_LABELS = ("pharynx",)
DEFAULT_VISUAL_CONTOUR_POLICY = "connect_last_point_to_first_point_for_all_labels_except_pharynx"
DEFAULT_STAGE_DIR_PREFIX = "stage_overlays_v2"
DEFAULT_OVERLAP_METRIC = "dice"
SUPPORTED_OVERLAP_METRICS = ("dice", "iou")


def normalize_overlap_metric(overlap_metric: str) -> str:
    metric = overlap_metric.strip().lower()
    if metric not in SUPPORTED_OVERLAP_METRICS:
        raise ValueError(f"Unsupported overlap metric '{overlap_metric}'. Expected one of: {', '.join(SUPPORTED_OVERLAP_METRICS)}")
    return metric


def metric_display_name(overlap_metric: str) -> str:
    metric = normalize_overlap_metric(overlap_metric)
    return "Dice" if metric == "dice" else "IoU"


def default_stage_dir_name(overlap_metric: str) -> str:
    return f"{DEFAULT_STAGE_DIR_PREFIX}_{normalize_overlap_metric(overlap_metric)}"


def metric_column_names(overlap_metric: str) -> dict[str, str]:
    metric = normalize_overlap_metric(overlap_metric)
    return {
        "metric": metric,
        "distance": f"{metric}_distance",
        "squared_distance": f"{metric}_distance_squared",
        "mean": f"mean_pairwise_{metric}",
        "mean_distance": f"mean_pairwise_{metric}_distance",
        "mean_squared": f"mean_squared_{metric}_distance",
        "overall_mean": f"overall_mean_pairwise_{metric}",
        "overall_mean_distance": f"overall_mean_pairwise_{metric}_distance",
        "overall_mean_squared": f"overall_mean_squared_{metric}_distance_per_speaker",
    }


def stage_display_name(stage_name: str) -> str:
    if stage_name == "init":
        return "Before affine (init)"
    if stage_name == "affine":
        return "Affine only"
    if stage_name == "tps":
        return "After TPS"
    return stage_name


def stage_compact_label(stage_name: str) -> str:
    if stage_name == "init":
        return "Init"
    if stage_name == "affine":
        return "Affine"
    if stage_name == "tps":
        return "TPS"
    return stage_name


def speaker_display_name(speakers: dict[str, LoadedSpeaker], speaker_id: str) -> str:
    return speakers[speaker_id].spec.speaker_id


def speaker_order_key(speakers: dict[str, LoadedSpeaker], speaker_id: str) -> tuple[int, str]:
    return speaker_id_sort_key(speaker_display_name(speakers, speaker_id))


def choose_speaker_colors(speakers: dict[str, LoadedSpeaker]) -> dict[str, tuple[float, float, float, float]]:
    import matplotlib.pyplot as plt

    cmap = plt.get_cmap("tab20")
    ordered_ids = sorted(speakers, key=lambda speaker_id: speaker_order_key(speakers, speaker_id))
    return {speaker_id: cmap(index % 20) for index, speaker_id in enumerate(ordered_ids)}


def normalize_requested_speakers(
    vtln_dir: Path,
    requested_speakers: list[str] | None,
    *,
    target_speaker_id: str,
) -> list[str]:
    if requested_speakers is None:
        speakers = default_curated_speaker_ids(vtln_dir)
    else:
        speakers = [speaker.upper().strip() for speaker in requested_speakers if speaker.strip()]
        if not speakers:
            raise ValueError("Expected at least one requested speaker.")

    seen: set[str] = set()
    ordered: list[str] = []
    for speaker in speakers:
        if speaker not in seen:
            ordered.append(speaker)
            seen.add(speaker)
    if target_speaker_id not in seen:
        ordered.append(target_speaker_id)
    return ordered


def load_fixed_target_stage_overlay_speakers(
    vtln_dir: Path,
    requested_speakers: list[str] | None = None,
    *,
    target_speaker_id: str = DEFAULT_TARGET_SPEAKER_ID,
) -> dict[str, LoadedSpeaker]:
    speaker_ids = normalize_requested_speakers(vtln_dir, requested_speakers, target_speaker_id=target_speaker_id)
    speakers = load_curated_speakers(vtln_dir, speaker_ids)
    if target_speaker_id not in speakers:
        raise ValueError(f"Fixed target speaker {target_speaker_id} was not loaded.")
    target = speakers[target_speaker_id]
    if target.spec.basename != DEFAULT_TARGET_BASENAME:
        raise ValueError(
            f"Expected {target_speaker_id} basename {DEFAULT_TARGET_BASENAME}, got {target.spec.basename}."
        )
    return speakers


def resolve_cohort_member_ids(
    speakers: dict[str, LoadedSpeaker],
    cohort_name: str,
) -> list[str]:
    ordered_ids = sorted(speakers, key=lambda speaker_id: speaker_order_key(speakers, speaker_id))
    if cohort_name == "all":
        return ordered_ids
    return [speaker_id for speaker_id in ordered_ids if speakers[speaker_id].spec.gender == cohort_name]


def resolve_overlay_ids(
    speakers: dict[str, LoadedSpeaker],
    cohort_name: str,
    *,
    target_speaker_id: str,
) -> tuple[list[str], list[str]]:
    cohort_ids = resolve_cohort_member_ids(speakers, cohort_name)
    if not cohort_ids:
        raise ValueError(f"No speakers available for cohort '{cohort_name}'.")
    overlay_ids = list(cohort_ids)
    if target_speaker_id not in overlay_ids:
        overlay_ids.append(target_speaker_id)
    overlay_ids = sorted(set(overlay_ids), key=lambda speaker_id: speaker_order_key(speakers, speaker_id))
    return cohort_ids, overlay_ids


def resolve_common_labels(
    speakers: dict[str, LoadedSpeaker],
    overlay_ids: list[str],
) -> list[str]:
    label_sets = [set(speakers[speaker_id].contours.keys()) for speaker_id in overlay_ids]
    labels = sorted(set.intersection(*label_sets))
    if not labels:
        raise ValueError("No common contour labels were shared across the selected overlay speakers.")
    return labels


def resolve_variance_labels(
    common_labels: list[str],
    exclude_labels: tuple[str, ...] = DEFAULT_VARIANCE_EXCLUDE_LABELS,
) -> list[str]:
    exclude_set = {label.strip() for label in exclude_labels if label.strip()}
    variance_labels = [label for label in common_labels if label not in exclude_set]
    if not variance_labels:
        raise ValueError("No labels remained for variance after applying exclusions.")
    return variance_labels


def close_visual_contour(points: np.ndarray, label: str | None = None) -> np.ndarray:
    pts = np.asarray(points, dtype=float)
    if len(pts) == 0:
        return pts.copy()
    if label in DEFAULT_OPEN_VISUAL_LABELS:
        return pts.copy()
    if np.allclose(pts[0], pts[-1]):
        return pts.copy()
    return np.vstack([pts, pts[0]])


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})


def map_grid_lines(grid, mapping_fn) -> dict[str, list[np.ndarray]]:
    return {
        "horiz_lines": [np.asarray(mapping_fn(np.asarray(line, dtype=float)), dtype=float) for line in grid.horiz_lines],
        "vert_lines": [np.asarray(mapping_fn(np.asarray(line, dtype=float)), dtype=float) for line in grid.vert_lines],
    }


def summarize_stage_population(
    mapped_contours: dict[str, dict[str, np.ndarray]],
    mapped_grids: dict[str, dict[str, list[np.ndarray]]],
    common_labels: list[str],
    reference_id: str,
    nc_template: int,
) -> dict[str, object]:
    mean_contours = {}
    median_contours = {}
    label_variability_to_median = {}
    for label in common_labels:
        stack = np.stack(
            [resample_polyline(mapped_contours[speaker_id][label], nc_template) for speaker_id in sorted(mapped_contours)],
            axis=0,
        )
        mean_contours[label] = np.mean(stack, axis=0)
        median_contours[label] = np.median(stack, axis=0)
        label_variability_to_median[label] = float(
            np.mean(
                np.sqrt(
                    np.mean(
                        np.sum((stack - median_contours[label][None, :, :]) ** 2, axis=2),
                        axis=1,
                    )
                )
            )
        )

    mean_shape = np.vstack([mean_contours[label] for label in common_labels])
    median_shape = np.vstack([median_contours[label] for label in common_labels])
    mean_rms = {}
    median_rms = {}
    for speaker_id in sorted(mapped_contours):
        speaker_shape = np.vstack(
            [resample_polyline(mapped_contours[speaker_id][label], nc_template) for label in common_labels]
        )
        mean_rms[speaker_id] = float(np.sqrt(np.mean((speaker_shape - mean_shape) ** 2)))
        median_rms[speaker_id] = float(np.sqrt(np.mean((speaker_shape - median_shape) ** 2)))

    median_speaker = min(median_rms, key=median_rms.get)
    closest_to_mean = min(mean_rms, key=mean_rms.get)
    return {
        "reference_id": reference_id,
        "mapped_contours": mapped_contours,
        "mapped_grids": mapped_grids,
        "mean_contours": mean_contours,
        "median_contours": median_contours,
        "mean_rms": mean_rms,
        "median_rms": median_rms,
        "median_speaker": median_speaker,
        "closest_to_mean": closest_to_mean,
        "label_variability_to_median": label_variability_to_median,
        "nc_template": nc_template,
    }


def compute_stage_populations(
    speakers: dict[str, LoadedSpeaker],
    overlay_ids: list[str],
    common_labels: list[str],
    reference_id: str,
    *,
    nc_template: int,
) -> dict[str, dict[str, object]]:
    reference = speakers[reference_id]
    mapped_payload = {
        "init": {"contours": {}, "grids": {}},
        "affine": {"contours": {}, "grids": {}},
        "tps": {"contours": {}, "grids": {}},
    }

    for speaker_id in overlay_ids:
        speaker = speakers[speaker_id]
        identity_map = lambda pts: np.asarray(pts, dtype=float)
        init_contours = {label: np.asarray(speaker.contours[label], dtype=float) for label in common_labels}
        init_grid = map_grid_lines(speaker.grid, identity_map)

        if speaker_id == reference_id:
            affine_contours = {label: np.asarray(speaker.contours[label], dtype=float) for label in common_labels}
            tps_contours = {label: np.asarray(speaker.contours[label], dtype=float) for label in common_labels}
            affine_grid = map_grid_lines(speaker.grid, identity_map)
            tps_grid = map_grid_lines(speaker.grid, identity_map)
        else:
            transform = build_two_step_transform(speaker.grid, reference.grid)

            def apply_affine_only(points, *, affine=transform["step1_affine"]):
                return np.asarray(apply_transform(affine, np.asarray(points, dtype=float)), dtype=float)

            affine_contours = transform_contours(speaker.contours, apply_affine_only, common_labels)
            tps_contours = smooth_transformed_contours(
                transform_contours(speaker.contours, transform["apply_two_step"], common_labels)
            )
            affine_grid = map_grid_lines(speaker.grid, apply_affine_only)
            tps_grid = map_grid_lines(speaker.grid, transform["apply_two_step"])

        mapped_payload["init"]["contours"][speaker_id] = init_contours
        mapped_payload["init"]["grids"][speaker_id] = init_grid
        mapped_payload["affine"]["contours"][speaker_id] = affine_contours
        mapped_payload["affine"]["grids"][speaker_id] = affine_grid
        mapped_payload["tps"]["contours"][speaker_id] = tps_contours
        mapped_payload["tps"]["grids"][speaker_id] = tps_grid

    return {
        stage_name: summarize_stage_population(
            stage_payload["contours"],
            stage_payload["grids"],
            common_labels,
            reference_id,
            nc_template,
        )
        for stage_name, stage_payload in mapped_payload.items()
    }


def set_overlay_view_limits(ax, image, point_sets: list[np.ndarray], *, padding: float) -> None:
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


def format_white_canvas(ax) -> None:
    ax.set_facecolor("white")
    ax.set_aspect("equal")
    ax.axis("off")


def create_stage_overlay_figure(title: str, *, include_note: bool = True):
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=DEFAULT_STAGE_FIGURE_SIZE, dpi=220, facecolor="white")
    if include_note:
        grid = fig.add_gridspec(4, 1, height_ratios=(0.09, 0.76, 0.08, 0.07))
    else:
        grid = fig.add_gridspec(3, 1, height_ratios=(0.09, 0.83, 0.08))
    title_ax = fig.add_subplot(grid[0, 0])
    image_ax = fig.add_subplot(grid[1, 0])
    note_ax = fig.add_subplot(grid[2, 0]) if include_note else None
    legend_ax = fig.add_subplot(grid[3, 0] if include_note else grid[2, 0])
    aux_axes = [title_ax, legend_ax]
    if note_ax is not None:
        aux_axes.append(note_ax)
    for ax in aux_axes:
        ax.axis("off")
    title_ax.text(0.5, 0.52, title, ha="center", va="center", fontsize=DEFAULT_STAGE_TITLE_SIZE, fontweight="bold")
    format_white_canvas(image_ax)
    fig.subplots_adjust(left=0.02, right=0.98, top=0.985, bottom=0.03, hspace=0.03)
    return fig, image_ax, note_ax, legend_ax


def add_stage_note(note_ax, note_text: str) -> None:
    note_ax.text(
        0.01,
        0.97,
        note_text,
        ha="left",
        va="top",
        fontsize=DEFAULT_STAGE_NOTE_SIZE,
        bbox=dict(boxstyle="round,pad=0.45", fc="white", ec="0.80", alpha=0.98),
    )


def add_stage_legend(
    legend_ax,
    speakers: dict[str, LoadedSpeaker],
    ordered_ids: list[str],
    speaker_colors: dict[str, tuple[float, float, float, float]],
    *,
    highlight_speaker_id: str,
    highlight_label: str,
) -> None:
    import matplotlib.pyplot as plt

    handles = []
    labels = []
    for speaker_id in ordered_ids:
        is_highlight = speaker_id == highlight_speaker_id
        line_width = 3.2 if is_highlight else 2.5
        alpha = 0.98 if is_highlight else 0.90
        linestyle = "-"
        handles.append(
            plt.Line2D(
                [0],
                [0],
                color=speaker_colors[speaker_id],
                lw=line_width,
                alpha=alpha,
                linestyle=linestyle,
            )
        )
        label = speaker_display_name(speakers, speaker_id)
        if is_highlight:
            label += f" ({highlight_label})"
        labels.append(label)
    legend_ax.legend(
        handles,
        labels,
        loc="center",
        ncol=min(5, len(handles)),
        fontsize=DEFAULT_STAGE_LEGEND_SIZE,
        framealpha=0.96,
        handlelength=3.0,
        columnspacing=1.05,
    )


def stage_overlay_output_path(stage_dir: Path, cohort_name: str, stage_name: str, overlay_kind: str) -> Path:
    cohort_index = DEFAULT_STAGE_EXPORT_COHORTS.index(cohort_name)
    stage_index = DEFAULT_STAGE_EXPORT_ORDER.index(stage_name)
    kind_index = 0 if overlay_kind == "contours" else 1
    ordinal = cohort_index * 6 + stage_index * 2 + kind_index + 1
    return stage_dir / f"{ordinal:03d}_{cohort_name}_{stage_name}_{overlay_kind}_overlay.png"


def build_stage_note_text(
    speakers: dict[str, LoadedSpeaker],
    cohort_ids: list[str],
    overlay_ids: list[str],
    stage_payload: dict[str, object],
    *,
    cohort_name: str,
    target_speaker_id: str,
) -> str:
    target = speakers[target_speaker_id]
    label_count = len(stage_payload["mapped_contours"][target_speaker_id])
    return "\n".join(
        [
            f"{cohort_name} | N={len(overlay_ids)} | labels={label_count}",
            f"Target: {target_speaker_id}",
            f"Average speaker: {speaker_display_name(speakers, str(stage_payload['median_speaker']))}",
            "Visual: closed except pharynx",
        ]
    )


def plot_single_speaker_contours(
    ax,
    contour_dict: dict[str, np.ndarray],
    labels: list[str],
    color,
    *,
    alpha: float,
    lw: float,
    zorder: int,
    linestyle: str = "-",
    outline: bool = False,
) -> None:
    for label in labels:
        pts = close_visual_contour(np.asarray(contour_dict[label], dtype=float), label)
        if outline:
            ax.plot(
                pts[:, 0],
                pts[:, 1],
                color="black",
                lw=lw + 1.6,
                alpha=min(0.42, alpha),
                zorder=max(1, zorder - 1),
                linestyle=linestyle,
            )
        ax.plot(
            pts[:, 0],
            pts[:, 1],
            color=color,
            lw=lw * (1.12 if label == "tongue" else 1.0),
            alpha=alpha,
            zorder=zorder,
            linestyle=linestyle,
        )


def plot_single_speaker_grid(
    ax,
    grid_payload: dict[str, list[np.ndarray]],
    color,
    *,
    alpha: float,
    lw: float,
    zorder: int,
    outline: bool = False,
) -> None:
    horiz_lines = list(grid_payload["horiz_lines"])
    vert_lines = list(grid_payload["vert_lines"])
    for index, line in enumerate(horiz_lines):
        boundary = index in {0, len(horiz_lines) - 1}
        line_lw = lw * (1.2 if boundary else 1.0)
        if outline:
            ax.plot(line[:, 0], line[:, 1], color="black", lw=line_lw + 1.0, alpha=min(0.34, alpha), zorder=max(1, zorder - 1))
        ax.plot(line[:, 0], line[:, 1], color=color, lw=line_lw, alpha=alpha, zorder=zorder)
    for index, line in enumerate(vert_lines):
        boundary = index in {0, len(vert_lines) - 1}
        line_lw = lw * (1.15 if boundary else 0.92)
        if outline:
            ax.plot(line[:, 0], line[:, 1], color="black", lw=line_lw + 1.0, alpha=min(0.34, alpha), zorder=max(1, zorder - 1))
        ax.plot(line[:, 0], line[:, 1], color=color, lw=line_lw, alpha=alpha, zorder=zorder)


def stage_title(cohort_name: str, stage_name: str, overlay_kind: str, *, target_speaker_id: str, target_injected: bool) -> str:
    if cohort_name == "all" and overlay_kind == "contours":
        return f"{stage_compact_label(stage_name)} contours"
    cohort_label = "All" if cohort_name == "all" else cohort_name.capitalize()
    if target_injected:
        cohort_label = f"{cohort_label} + {target_speaker_id}"
    kind_label = "contours" if overlay_kind == "contours" else "grid"
    return f"{cohort_label} | {stage_compact_label(stage_name)} {kind_label}"


def save_stage_contour_overlay_figure(
    speakers: dict[str, LoadedSpeaker],
    cohort_ids: list[str],
    overlay_ids: list[str],
    labels: list[str],
    stage_payload: dict[str, object],
    speaker_colors: dict[str, tuple[float, float, float, float]],
    *,
    stage_name: str,
    cohort_name: str,
    target_speaker_id: str,
    output_path: Path,
) -> None:
    import matplotlib.pyplot as plt

    reference = speakers[str(stage_payload["reference_id"])]
    median_speaker_id = str(stage_payload["median_speaker"])
    ordered_ids = sorted(stage_payload["mapped_contours"], key=lambda speaker_id: speaker_order_key(speakers, speaker_id))
    target_injected = target_speaker_id not in cohort_ids
    use_reference_highlight = cohort_name == "all"
    highlight_speaker_id = target_speaker_id if use_reference_highlight else median_speaker_id
    highlight_label = "reference speaker" if use_reference_highlight else "average speaker"
    draw_order = [speaker_id for speaker_id in ordered_ids if speaker_id != highlight_speaker_id]
    draw_order.append(highlight_speaker_id)
    include_note = not use_reference_highlight

    fig, ax, note_ax, legend_ax = create_stage_overlay_figure(
        stage_title(cohort_name, stage_name, "contours", target_speaker_id=target_speaker_id, target_injected=target_injected),
        include_note=include_note,
    )

    all_points: list[np.ndarray] = []
    for speaker_id in draw_order:
        is_highlight = speaker_id == highlight_speaker_id
        speaker_contours = stage_payload["mapped_contours"][speaker_id]
        for label in labels:
            all_points.append(close_visual_contour(np.asarray(speaker_contours[label], dtype=float), label))
        plot_single_speaker_contours(
            ax,
            speaker_contours,
            labels,
            speaker_colors[speaker_id],
            alpha=0.98 if is_highlight else 0.24,
            lw=3.2 if is_highlight else 1.55,
            zorder=5 if is_highlight else 2,
            linestyle="-",
            outline=is_highlight,
        )

    set_overlay_view_limits(ax, reference.image, all_points, padding=26.0)
    if note_ax is not None:
        add_stage_note(
            note_ax,
            build_stage_note_text(
                speakers,
                cohort_ids,
                overlay_ids,
                stage_payload,
                cohort_name=cohort_name,
                target_speaker_id=target_speaker_id,
            ),
        )
    add_stage_legend(
        legend_ax,
        speakers,
        ordered_ids,
        speaker_colors,
        highlight_speaker_id=highlight_speaker_id,
        highlight_label=highlight_label,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight", pad_inches=0.04, facecolor="white")
    plt.close(fig)


def save_stage_grid_overlay_figure(
    speakers: dict[str, LoadedSpeaker],
    cohort_ids: list[str],
    overlay_ids: list[str],
    stage_payload: dict[str, object],
    speaker_colors: dict[str, tuple[float, float, float, float]],
    *,
    stage_name: str,
    cohort_name: str,
    target_speaker_id: str,
    output_path: Path,
) -> None:
    import matplotlib.pyplot as plt

    reference = speakers[str(stage_payload["reference_id"])]
    median_speaker_id = str(stage_payload["median_speaker"])
    ordered_ids = sorted(stage_payload["mapped_grids"], key=lambda speaker_id: speaker_order_key(speakers, speaker_id))
    draw_order = [speaker_id for speaker_id in ordered_ids if speaker_id != median_speaker_id]
    draw_order.append(median_speaker_id)
    target_injected = target_speaker_id not in cohort_ids

    fig, ax, note_ax, legend_ax = create_stage_overlay_figure(
        stage_title(cohort_name, stage_name, "grid", target_speaker_id=target_speaker_id, target_injected=target_injected)
    )

    for speaker_id in draw_order:
        is_median = speaker_id == median_speaker_id
        plot_single_speaker_grid(
            ax,
            stage_payload["mapped_grids"][speaker_id],
            speaker_colors[speaker_id],
            alpha=0.94 if is_median else 0.24,
            lw=2.3 if is_median else 1.12,
            zorder=5 if is_median else 2,
            outline=is_median,
        )

    set_overlay_view_limits(
        ax,
        reference.image,
        [
            line
            for speaker_id in draw_order
            for line in (
                list(stage_payload["mapped_grids"][speaker_id]["horiz_lines"])
                + list(stage_payload["mapped_grids"][speaker_id]["vert_lines"])
            )
        ],
        padding=22.0,
    )
    add_stage_note(
        note_ax,
        build_stage_note_text(
            speakers,
            cohort_ids,
            overlay_ids,
            stage_payload,
            cohort_name=cohort_name,
            target_speaker_id=target_speaker_id,
        ),
    )
    add_stage_legend(
        legend_ax,
        speakers,
        ordered_ids,
        speaker_colors,
        highlight_speaker_id=median_speaker_id,
        highlight_label="average speaker",
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight", pad_inches=0.04, facecolor="white")
    plt.close(fig)


def compute_stage_variance_details(
    speakers: dict[str, LoadedSpeaker],
    variance_labels: list[str],
    stage_populations: dict[str, dict[str, object]],
    *,
    cohort_name: str,
    target_speaker_id: str,
    overlap_metric: str = DEFAULT_OVERLAP_METRIC,
) -> dict[str, list[dict[str, object]]]:
    overlap_metric = normalize_overlap_metric(overlap_metric)
    columns = metric_column_names(overlap_metric)
    secondary_metric = "iou" if overlap_metric == "dice" else "dice"
    summary_rows: list[dict[str, object]] = []
    label_rows: list[dict[str, object]] = []
    speaker_rows: list[dict[str, object]] = []
    pair_rows: list[dict[str, object]] = []

    for stage_name in DEFAULT_STAGE_ORDER:
        payload = stage_populations[stage_name]
        ordered_ids = sorted(payload["mapped_contours"], key=lambda speaker_id: speaker_order_key(speakers, speaker_id))
        reference_id = str(payload.get("reference_id", target_speaker_id if target_speaker_id in speakers else ordered_ids[0]))
        reference = speakers[reference_id]
        reference_image = getattr(reference, "image", np.zeros((480, 480), dtype=np.uint8))
        image_shape = np.asarray(reference_image).shape
        speaker_overlap_map = {speaker_id: [] for speaker_id in ordered_ids}
        speaker_distance_map = {speaker_id: [] for speaker_id in ordered_ids}
        speaker_squared_distance_map = {speaker_id: [] for speaker_id in ordered_ids}
        label_variances: list[float] = []
        label_mean_overlap_values: list[float] = []
        label_mean_distance_values: list[float] = []

        for label in variance_labels:
            masks = {
                speaker_id: polygon_to_mask(np.asarray(payload["mapped_contours"][speaker_id][label], dtype=float), image_shape)
                for speaker_id in ordered_ids
            }
            label_overlaps: list[float] = []
            label_distances: list[float] = []
            label_squared_distances: list[float] = []
            pair_count = 0

            for index, speaker_i in enumerate(ordered_ids):
                for speaker_j in ordered_ids[index + 1 :]:
                    metrics = compute_mask_overlap_metrics(masks[speaker_i], masks[speaker_j])
                    overlap = float(metrics[overlap_metric])
                    distance = float(1.0 - overlap)
                    squared_distance = float(distance * distance)
                    label_overlaps.append(overlap)
                    label_distances.append(distance)
                    label_squared_distances.append(squared_distance)
                    pair_count += 1

                    speaker_overlap_map[speaker_i].append(overlap)
                    speaker_overlap_map[speaker_j].append(overlap)
                    speaker_distance_map[speaker_i].append(distance)
                    speaker_distance_map[speaker_j].append(distance)
                    speaker_squared_distance_map[speaker_i].append(squared_distance)
                    speaker_squared_distance_map[speaker_j].append(squared_distance)

                    pair_rows.append(
                        {
                            "cohort": cohort_name,
                            "stage": stage_name,
                            "label": label,
                            "speaker_i": speaker_display_name(speakers, speaker_i),
                            "speaker_j": speaker_display_name(speakers, speaker_j),
                            columns["metric"]: overlap,
                            columns["distance"]: distance,
                            columns["squared_distance"]: squared_distance,
                            secondary_metric: float(metrics[secondary_metric]),
                            "intersection": int(metrics["intersection"]),
                            "area_i": int(metrics["source_area"]),
                            "area_j": int(metrics["target_area"]),
                            "union": int(metrics["union"]),
                        }
                    )

            label_variance = float(np.mean(label_squared_distances)) if label_squared_distances else 0.0
            label_mean_overlap = float(np.mean(label_overlaps)) if label_overlaps else 1.0
            label_mean_distance = float(np.mean(label_distances)) if label_distances else 0.0
            label_variances.append(label_variance)
            label_mean_overlap_values.append(label_mean_overlap)
            label_mean_distance_values.append(label_mean_distance)
            label_rows.append(
                {
                    "cohort": cohort_name,
                    "stage": stage_name,
                    "label": label,
                    "label_variance": label_variance,
                    columns["mean"]: label_mean_overlap,
                    columns["mean_distance"]: label_mean_distance,
                    "pair_count": pair_count,
                    "speaker_count": len(ordered_ids),
                }
            )

        speaker_stage_rows: list[dict[str, object]] = []
        for speaker_id in ordered_ids:
            mean_overlap = float(np.mean(speaker_overlap_map[speaker_id])) if speaker_overlap_map[speaker_id] else 1.0
            mean_distance = (
                float(np.mean(speaker_distance_map[speaker_id])) if speaker_distance_map[speaker_id] else 0.0
            )
            mean_squared_distance = (
                float(np.mean(speaker_squared_distance_map[speaker_id]))
                if speaker_squared_distance_map[speaker_id]
                else 0.0
            )
            row = {
                "cohort": cohort_name,
                "stage": stage_name,
                "speaker": speaker_display_name(speakers, speaker_id),
                "is_target": speaker_id == target_speaker_id,
                columns["mean"]: mean_overlap,
                columns["mean_distance"]: mean_distance,
                columns["mean_squared"]: mean_squared_distance,
                "pair_count": len(speaker_overlap_map[speaker_id]),
                "label_count": len(variance_labels),
            }
            speaker_rows.append(row)
            speaker_stage_rows.append(row)

        summary_rows.append(
            {
                "cohort": cohort_name,
                "stage": stage_name,
                "speaker_count": len(ordered_ids),
                "label_count": len(variance_labels),
                "overall_variance": float(np.mean(label_variances)),
                columns["overall_mean"]: float(np.mean(label_mean_overlap_values)),
                columns["overall_mean_distance"]: float(np.mean(label_mean_distance_values)),
                columns["overall_mean_squared"]: float(
                    np.mean([row[columns["mean_squared"]] for row in speaker_stage_rows])
                ),
                "pair_count_per_label": int(len(ordered_ids) * (len(ordered_ids) - 1) / 2),
                "average_speaker": speaker_display_name(speakers, str(payload["median_speaker"])),
                "closest_to_mean": speaker_display_name(speakers, str(payload["closest_to_mean"])),
            }
        )

    return {
        "summary_rows": summary_rows,
        "label_rows": label_rows,
        "speaker_rows": speaker_rows,
        "pair_rows": pair_rows,
    }


def _read_reference_candidate_rows(output_dir: Path, limit: int = 3) -> list[dict[str, str]]:
    candidates_csv = output_dir / "average_speaker_roi_affine_c1_c6_incisior_hard_palate" / "candidate_scores.csv"
    if not candidates_csv.is_file():
        return []
    with candidates_csv.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))[:limit]


def _reduction_percent(before: float, after: float) -> float:
    if abs(before) < 1e-12:
        return 0.0
    return 100.0 * (before - after) / before


def write_workshop_stage_overlay_summary(
    stage_dir: Path,
    output_dir: Path,
    summary_payload: dict[str, object],
    variance_summary_rows: list[dict[str, object]],
    *,
    overlap_metric: str,
) -> Path:
    overlap_metric = normalize_overlap_metric(overlap_metric)
    columns = metric_column_names(overlap_metric)
    metric_label = metric_display_name(overlap_metric)
    target = dict(summary_payload["target_speaker"])
    candidate_rows = _read_reference_candidate_rows(output_dir)
    rows_by_key = {(str(row["cohort"]), str(row["stage"])): row for row in variance_summary_rows}
    all_init = float(rows_by_key[("all", "init")]["overall_variance"])
    all_affine = float(rows_by_key[("all", "affine")]["overall_variance"])
    all_tps = float(rows_by_key[("all", "tps")]["overall_variance"])

    lines = [
        f"# Workshop 2026 Summary: Reference-Speaker Selection and {metric_label} Stage-Overlay Results",
        "",
        "## 1. Objective",
        "",
        "This workflow answers two linked questions:",
        "",
        "1. Which curated speaker should be used as the fixed reference speaker?",
        f"2. After fixing that reference speaker, how much do the affine and TPS stages reduce cross-speaker variability under {metric_label} distance?",
        "",
        "The final fixed reference used for the overlay workflow is:",
        "",
        f"- `{target['speaker_id']} / {target['basename']}`",
        "",
        "## 2. How the Reference Speaker Was Selected",
        "",
        "Reference-speaker selection was done with the affine ROI-overlap workflow in:",
        "",
        "- `outputs/average_speaker_roi_affine_c1_c6_incisior_hard_palate/`",
        "",
        "Only affine was used at this stage. For every ordered pair `source -> target`, source contours were affine-mapped into target space, straight-closed into ROI masks, and scored with Dice overlap.",
        "",
        "```text",
        "Dice(A, B) = 2 |A intersect B| / (|A| + |B|)",
        "```",
        "",
        "The chosen reference speaker is the target with the largest `mean_pair_dice`.",
        "",
    ]
    if candidate_rows:
        lines.extend(
            [
                "Top candidates from `candidate_scores.csv`:",
                "",
                "| rank | speaker | basename | mean pair Dice | mean pair IoU |",
                "| ---- | ------- | -------- | -------------: | ------------: |",
            ]
        )
        for row in candidate_rows:
            lines.append(
                f"| {row['rank']} | `{row['target_speaker']}` | `{row['target_basename']}` | "
                f"`{float(row['mean_pair_dice']):.4f}` | `{float(row['mean_pair_iou']):.4f}` |"
            )
        lines.append("")

    lines.extend(
        [
            "Conclusion:",
            "",
            f"- `{target['speaker_id']}` was selected because it gives the highest mean affine ROI-Dice overlap across the other speakers on the chosen closed labels.",
            "",
            "## 3. Fixed-Reference Stage Workflow",
            "",
            f"After selecting `{target['speaker_id']}`, the fixed-target overlay workflow was run in:",
            "",
            f"- `outputs/{stage_dir.name}/`",
            "",
            "Stages:",
            "",
            "- `init`: original contours before affine",
            f"- `affine`: all speakers mapped into `{target['speaker_id']}` space with Step-1 affine only",
            "- `tps`: all speakers mapped into the fixed-reference space with full `Affine + TPS`",
            "",
            "## 4. Variance Metric Used for Stage Evaluation",
            "",
        ]
    )
    if overlap_metric == "dice":
        lines.extend(
            [
                "We use Dice overlap as the primary ROI-alignment metric throughout the analysis. The reference speaker is selected by maximizing mean affine-stage Dice overlap. For stage-wise evaluation, we define Dice distance as one minus Dice overlap and summarize each stage by the mean squared pairwise Dice distance across speakers and anatomical labels. Lower Dice-distance variance indicates better cross-speaker alignment.",
                "",
                "IoU is retained only as an optional compatibility metric: rerun the workflow with `--overlap-metric iou` to produce an IoU-based folder.",
            ]
        )
    else:
        lines.extend(
            [
                "This compatibility run uses IoU distance for stage-wise evaluation. Dice remains the primary metric for the new default workflow.",
            ]
        )
    lines.extend(
        [
            "",
            f"Per-pair {metric_label}:",
            "",
            "```text",
            (
                "Dice(A, B) = 2 |A intersect B| / (|A| + |B|)"
                if overlap_metric == "dice"
                else "IoU(A, B) = |A intersect B| / |A union B|"
            ),
            "```",
            "",
            f"{metric_label} distance:",
            "",
            "```text",
            f"d_{overlap_metric}(A, B) = 1 - {metric_label}(A, B)",
            "```",
            "",
            "Per-label stage variance:",
            "",
            "```text",
            f"V_stage,label = mean_{{i<j}} (1 - {metric_label}(M_i, M_j))^2",
            "```",
            "",
            "Overall stage variance:",
            "",
            "```text",
            "V_stage = mean over labels of V_stage,label",
            "```",
            "",
            "Lower variance is better.",
            "",
            "## 5. Main Results: `all` Cohort",
            "",
            "From `stage_variance.csv`:",
            "",
            f"| stage | overall variance | mean pairwise {metric_label} |",
            "| ----- | ---------------: | ----------------------: |",
        ]
    )
    for stage_name in DEFAULT_STAGE_ORDER:
        row = rows_by_key[("all", stage_name)]
        lines.append(
            f"| `{stage_name}` | `{float(row['overall_variance']):.4f}` | "
            f"`{float(row[columns['overall_mean']]):.4f}` |"
        )

    lines.extend(
        [
            "",
            "Stage reductions:",
            "",
            "```text",
            f"reduction_affine_vs_init = ({all_init:.4f} - {all_affine:.4f}) / {all_init:.4f} = {_reduction_percent(all_init, all_affine):.2f}%",
            f"reduction_tps_vs_affine = ({all_affine:.4f} - {all_tps:.4f}) / {all_affine:.4f} = {_reduction_percent(all_affine, all_tps):.2f}%",
            f"reduction_tps_vs_init = ({all_init:.4f} - {all_tps:.4f}) / {all_init:.4f} = {_reduction_percent(all_init, all_tps):.2f}%",
            "```",
            "",
            "## 6. Additional Cohort Results",
            "",
            "| cohort | init | affine | tps | affine vs init | tps vs init | tps vs affine |",
            "| ------ | ---: | -----: | --: | -------------: | ----------: | ------------: |",
        ]
    )
    for cohort_name in DEFAULT_COHORTS:
        init = float(rows_by_key[(cohort_name, "init")]["overall_variance"])
        affine = float(rows_by_key[(cohort_name, "affine")]["overall_variance"])
        tps = float(rows_by_key[(cohort_name, "tps")]["overall_variance"])
        lines.append(
            f"| `{cohort_name}` | `{init:.4f}` | `{affine:.4f}` | `{tps:.4f}` | "
            f"`{_reduction_percent(init, affine):.2f}%` | `{_reduction_percent(init, tps):.2f}%` | "
            f"`{_reduction_percent(affine, tps):.2f}%` |"
        )

    lines.extend(
        [
            "",
            "## 7. Final Takeaway",
            "",
            f"- `{target['speaker_id']}` is the fixed reference speaker under the affine ROI-Dice selection criterion.",
            f"- Stage effectiveness in this folder is measured with pairwise squared {metric_label}-distance variance.",
            "- `affine` gives the largest single reduction in cross-speaker variance.",
            "- `TPS` gives an additional local-shape improvement after affine.",
            f"- For the `all` cohort, variance drops from `{all_init:.4f}` to `{all_affine:.4f}` after affine, then to `{all_tps:.4f}` after TPS.",
            "",
            "## 8. Reproduction",
            "",
            "Main command:",
            "",
            "```powershell",
            r".\.venv\Scripts\python.exe .\scripts\run\run_p10_stage_overlays_v2.py",
            "```",
            "",
            "Reference-selection evidence:",
            "",
            "- `outputs/average_speaker_roi_affine_c1_c6_incisior_hard_palate/summary.json`",
            "- `outputs/average_speaker_roi_affine_c1_c6_incisior_hard_palate/candidate_scores.csv`",
            "",
            "Stage-evaluation evidence:",
            "",
            f"- `outputs/{stage_dir.name}/stage_variance.csv`",
            f"- `outputs/{stage_dir.name}/stage_label_variance.csv`",
            f"- `outputs/{stage_dir.name}/stage_pairwise_{overlap_metric}_details.csv`",
            f"- `outputs/{stage_dir.name}/stage_overlay_summary.json`",
        ]
    )
    summary_md_path = stage_dir / "WORKSHOP2026_STAGE_OVERLAYS_V2_SUMMARY.md"
    summary_md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary_md_path


def export_fixed_target_stage_overlays(
    speakers: dict[str, LoadedSpeaker],
    output_dir: Path,
    *,
    target_speaker_id: str = DEFAULT_TARGET_SPEAKER_ID,
    nc_template: int = 80,
    overlap_metric: str = DEFAULT_OVERLAP_METRIC,
    stage_dir_name: str | None = None,
) -> dict[str, object]:
    overlap_metric = normalize_overlap_metric(overlap_metric)
    columns = metric_column_names(overlap_metric)
    metric_label = metric_display_name(overlap_metric)
    secondary_metric = "iou" if overlap_metric == "dice" else "dice"
    stage_dir = output_dir / (stage_dir_name or default_stage_dir_name(overlap_metric))
    stage_dir.mkdir(parents=True, exist_ok=True)
    stage_variance_csv_path = stage_dir / "stage_variance.csv"
    stage_label_variance_csv_path = stage_dir / "stage_label_variance.csv"
    stage_speaker_variance_csv_path = stage_dir / "stage_speaker_variance.csv"
    stage_pairwise_csv_path = stage_dir / f"stage_pairwise_{overlap_metric}_details.csv"
    speaker_colors = choose_speaker_colors(speakers)
    target = speakers[target_speaker_id]
    summary_payload: dict[str, object] = {
        "layout_version": DEFAULT_STAGE_LAYOUT_VERSION,
        "target_speaker": {
            "speaker_id": target_speaker_id,
            "basename": target.spec.basename,
            "gender": target.spec.gender,
        },
        "visual_contour_policy": DEFAULT_VISUAL_CONTOUR_POLICY,
        "overlap_metric": overlap_metric,
        "stage_dir_name": stage_dir.name,
        "metric_mask_policy": f"for {metric_label} variance only, each non-pharynx stage contour is straight-closed and rasterized on the reference canvas",
        "variance_excluded_labels": list(DEFAULT_VARIANCE_EXCLUDE_LABELS),
        "variance_definition": {
            "kind": f"pairwise_squared_{overlap_metric}_distance",
            "per_label": f"mean over unordered speaker pairs of (1 - {metric_label}(mask_i, mask_j))^2",
            "overall_stage": f"mean over labels of the per-label {metric_label} variance",
        },
        "secondary_metric_note": (
            "IoU can be exported by rerunning with --overlap-metric iou; this folder uses Dice as the primary variance metric."
            if overlap_metric == "dice"
            else "Dice can be exported by rerunning with --overlap-metric dice; this folder uses IoU as the primary variance metric."
        ),
        "cohorts": {},
    }
    variance_summary_rows: list[dict[str, object]] = []
    variance_label_rows: list[dict[str, object]] = []
    variance_speaker_rows: list[dict[str, object]] = []
    variance_pair_rows: list[dict[str, object]] = []

    for cohort_name in DEFAULT_COHORTS:
        cohort_ids, overlay_ids = resolve_overlay_ids(speakers, cohort_name, target_speaker_id=target_speaker_id)
        common_labels = resolve_common_labels(speakers, overlay_ids)
        variance_labels = resolve_variance_labels(common_labels)
        stage_populations = compute_stage_populations(
            speakers,
            overlay_ids,
            common_labels,
            target_speaker_id,
            nc_template=nc_template,
        )
        variance_details = compute_stage_variance_details(
            speakers,
            variance_labels,
            stage_populations,
            cohort_name=cohort_name,
            target_speaker_id=target_speaker_id,
            overlap_metric=overlap_metric,
        )
        variance_summary_rows.extend(variance_details["summary_rows"])
        variance_label_rows.extend(variance_details["label_rows"])
        variance_speaker_rows.extend(variance_details["speaker_rows"])
        variance_pair_rows.extend(variance_details["pair_rows"])

        cohort_summary = {
            "cohort_speaker_count": len(cohort_ids),
            "overlay_speaker_count": len(overlay_ids),
            "cohort_speakers": [speaker_display_name(speakers, speaker_id) for speaker_id in cohort_ids],
            "overlay_speakers": [speaker_display_name(speakers, speaker_id) for speaker_id in overlay_ids],
            "reference_speaker": speaker_display_name(speakers, target_speaker_id),
            "common_labels": list(common_labels),
            "variance_labels": list(variance_labels),
            "variance_summary": {
                row["stage"]: {
                    "overall_variance": row["overall_variance"],
                    columns["overall_mean"]: row[columns["overall_mean"]],
                    columns["overall_mean_distance"]: row[columns["overall_mean_distance"]],
                    columns["overall_mean_squared"]: row[columns["overall_mean_squared"]],
                    "pair_count_per_label": row["pair_count_per_label"],
                    "average_speaker": row["average_speaker"],
                    "closest_to_mean": row["closest_to_mean"],
                }
                for row in variance_details["summary_rows"]
            },
            "stages": {},
        }

        for stage_name in DEFAULT_STAGE_ORDER:
            contour_path = stage_overlay_output_path(stage_dir, cohort_name, stage_name, "contours")
            grid_path = stage_overlay_output_path(stage_dir, cohort_name, stage_name, "grid")
            save_stage_contour_overlay_figure(
                speakers,
                cohort_ids,
                overlay_ids,
                common_labels,
                stage_populations[stage_name],
                speaker_colors,
                stage_name=stage_name,
                cohort_name=cohort_name,
                target_speaker_id=target_speaker_id,
                output_path=contour_path,
            )
            save_stage_grid_overlay_figure(
                speakers,
                cohort_ids,
                overlay_ids,
                stage_populations[stage_name],
                speaker_colors,
                stage_name=stage_name,
                cohort_name=cohort_name,
                target_speaker_id=target_speaker_id,
                output_path=grid_path,
            )
            cohort_summary["stages"][stage_name] = {
                "stage_label": stage_display_name(stage_name),
                "average_speaker": speaker_display_name(speakers, str(stage_populations[stage_name]["median_speaker"])),
                "closest_to_mean": speaker_display_name(speakers, str(stage_populations[stage_name]["closest_to_mean"])),
                "contour_overlay": str(contour_path),
                "grid_overlay": str(grid_path),
            }

        summary_payload["cohorts"][cohort_name] = cohort_summary

    _write_csv(
        stage_variance_csv_path,
        [
            "cohort",
            "stage",
            "speaker_count",
            "label_count",
            "overall_variance",
            columns["overall_mean"],
            columns["overall_mean_distance"],
            columns["overall_mean_squared"],
            "pair_count_per_label",
            "average_speaker",
            "closest_to_mean",
        ],
        variance_summary_rows,
    )
    _write_csv(
        stage_label_variance_csv_path,
        [
            "cohort",
            "stage",
            "label",
            "label_variance",
            columns["mean"],
            columns["mean_distance"],
            "pair_count",
            "speaker_count",
        ],
        variance_label_rows,
    )
    _write_csv(
        stage_speaker_variance_csv_path,
        [
            "cohort",
            "stage",
            "speaker",
            "is_target",
            columns["mean"],
            columns["mean_distance"],
            columns["mean_squared"],
            "pair_count",
            "label_count",
        ],
        variance_speaker_rows,
    )
    _write_csv(
        stage_pairwise_csv_path,
        [
            "cohort",
            "stage",
            "label",
            "speaker_i",
            "speaker_j",
            columns["metric"],
            columns["distance"],
            columns["squared_distance"],
            secondary_metric,
            "intersection",
            "area_i",
            "area_j",
            "union",
        ],
        variance_pair_rows,
    )

    summary_json_path = stage_dir / "stage_overlay_summary.json"
    summary_txt_path = stage_dir / "stage_overlay_summary.txt"

    lines = [
        "Fixed-Target Stage Overlay Summary",
        "==================================",
        f"layout version: {DEFAULT_STAGE_LAYOUT_VERSION}",
        f"target speaker: {target_speaker_id} / {target.spec.basename}",
        f"visual contour policy: {DEFAULT_VISUAL_CONTOUR_POLICY}",
        "metric ROI mask policy: non-pharynx contours are straight-closed on the reference canvas",
        f"variance excluded labels: {', '.join(DEFAULT_VARIANCE_EXCLUDE_LABELS)}",
        f"overlap metric: {metric_label}",
        f"variance: mean pairwise squared {metric_label} distance",
        "",
    ]
    for cohort_name in DEFAULT_COHORTS:
        cohort_summary = summary_payload["cohorts"].get(cohort_name)
        if not cohort_summary:
            continue
        lines.append(
            f"[{cohort_name}] cohort speakers ({cohort_summary['cohort_speaker_count']}): "
            f"{', '.join(cohort_summary['cohort_speakers'])}"
        )
        lines.append(
            f"overlay speakers ({cohort_summary['overlay_speaker_count']}): "
            f"{', '.join(cohort_summary['overlay_speakers'])}"
        )
        lines.append(f"reference speaker: {cohort_summary['reference_speaker']}")
        lines.append(f"common labels ({len(cohort_summary['common_labels'])}): {', '.join(cohort_summary['common_labels'])}")
        lines.append(
            f"variance labels ({len(cohort_summary['variance_labels'])}): {', '.join(cohort_summary['variance_labels'])}"
        )
        for stage_name in DEFAULT_STAGE_ORDER:
            stage_summary = cohort_summary["stages"][stage_name]
            variance_summary = cohort_summary["variance_summary"][stage_name]
            lines.append(
                f"{stage_name} ({stage_summary['stage_label']}): average={stage_summary['average_speaker']} | "
                f"closest_to_mean={stage_summary['closest_to_mean']}"
            )
            lines.append(
                f"  variance={variance_summary['overall_variance']:.4f} | "
                f"{columns['overall_mean']}={variance_summary[columns['overall_mean']]:.4f} | "
                f"{columns['overall_mean_distance']}={variance_summary[columns['overall_mean_distance']]:.4f}"
            )
            lines.append(f"  contour overlay: {stage_summary['contour_overlay']}")
            lines.append(f"  grid overlay: {stage_summary['grid_overlay']}")
        lines.append("")
    summary_txt_path.write_text("\n".join(lines), encoding="utf-8")
    summary_payload["summary_json"] = str(summary_json_path)
    summary_payload["summary_txt"] = str(summary_txt_path)
    summary_payload["stage_variance_csv"] = str(stage_variance_csv_path)
    summary_payload["stage_label_variance_csv"] = str(stage_label_variance_csv_path)
    summary_payload["stage_speaker_variance_csv"] = str(stage_speaker_variance_csv_path)
    summary_payload[f"stage_pairwise_{overlap_metric}_csv"] = str(stage_pairwise_csv_path)
    workshop_summary_path = write_workshop_stage_overlay_summary(
        stage_dir,
        output_dir,
        summary_payload,
        variance_summary_rows,
        overlap_metric=overlap_metric,
    )
    summary_payload["workshop_summary_md"] = str(workshop_summary_path)
    summary_json_path.write_text(json.dumps(summary_payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary_payload


__all__ = [
    "DEFAULT_TARGET_SPEAKER_ID",
    "DEFAULT_TARGET_BASENAME",
    "DEFAULT_COHORTS",
    "DEFAULT_STAGE_ORDER",
    "DEFAULT_STAGE_LAYOUT_VERSION",
    "DEFAULT_VISUAL_CONTOUR_POLICY",
    "DEFAULT_OVERLAP_METRIC",
    "SUPPORTED_OVERLAP_METRICS",
    "default_stage_dir_name",
    "load_fixed_target_stage_overlay_speakers",
    "resolve_cohort_member_ids",
    "resolve_overlay_ids",
    "resolve_common_labels",
    "resolve_variance_labels",
    "close_visual_contour",
    "compute_stage_populations",
    "compute_stage_variance_details",
    "export_fixed_target_stage_overlays",
]
