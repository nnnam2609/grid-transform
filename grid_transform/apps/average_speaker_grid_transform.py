from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from grid_transform.analysis_shared import (
    CURATED_PLUS_NNUNET_GENDER,
    choose_label_colors,
    read_curated_specs_map,
    speaker_id_sort_key,
)
from grid_transform.config import (
    DEFAULT_OUTPUT_DIR,
    DEFAULT_VTLN_DIR,
    TONGUE_COLOR,
    VT_SEG_CONTOURS_ROOT,
    VT_SEG_DATA_ROOT,
)
from grid_transform.io import load_frame_npy, load_frame_vtln
from grid_transform.transfer import build_two_step_transform, smooth_transformed_contours, transform_contours
from grid_transform.transform_helpers import apply_transform, format_target_frame, resample_polyline
from grid_transform.vt import build_grid


DEFAULT_CASE = "2008-003^01-1791/test"
DEFAULT_SOURCE_FRAME = 143020
DEFAULT_RESAMPLE_POINTS = 80
DEFAULT_TEMPLATE_POINTS = 80
DEFAULT_ALPHA = 0.22
DEFAULT_STAGE_ALPHA = 0.28
DEFAULT_STAGE_GRID_ALPHA = 0.24
DEFAULT_STAGE_LINEWIDTH = 1.15
DEFAULT_STAGE_HIGHLIGHT_LINEWIDTH = 2.8
DEFAULT_COHORTS = ("all", "male", "female")
DEFAULT_STAGE_ORDER = ("init", "affine", "tps")


def stage_display_name(stage_name: str) -> str:
    if stage_name == "init":
        return "Before affine (init)"
    if stage_name == "affine":
        return "Affine only"
    if stage_name == "tps":
        return "After TPS"
    return stage_name


def estimate_similarity_umeyama(src, dst):
    """Estimate a 2D similarity transform dst ≈ c * src @ R^T + t."""
    src = np.asarray(src, dtype=float)
    dst = np.asarray(dst, dtype=float)
    n_points = len(src)
    if n_points != len(dst) or n_points < 2:
        raise ValueError("Need at least two matched 2D points.")

    mu_src = src.mean(axis=0)
    mu_dst = dst.mean(axis=0)
    centered_src = src - mu_src
    centered_dst = dst - mu_dst

    var_src = np.sum(centered_src ** 2) / n_points
    cov = (centered_dst.T @ centered_src) / n_points
    u_mat, singular_vals, v_t = np.linalg.svd(cov)

    det_sign = np.linalg.det(u_mat) * np.linalg.det(v_t)
    diag = np.array([1.0, 1.0 if det_sign > 0 else -1.0])
    rot = u_mat @ np.diag(diag) @ v_t
    scale = float(np.sum(singular_vals * diag) / var_src) if var_src > 1e-12 else 1.0
    shift = mu_dst - scale * (rot @ mu_src)
    return scale, rot, shift


def rigid_align(pts, template):
    """Align a point set to the template using rotation + translation only."""
    pts = np.asarray(pts, dtype=float)
    template = np.asarray(template, dtype=float)
    _, rot, _ = estimate_similarity_umeyama(pts, template)
    mu_pts = pts.mean(axis=0)
    mu_template = template.mean(axis=0)
    shift = mu_template - rot @ mu_pts
    aligned = (pts @ rot.T) + shift
    return aligned, rot, shift


def compute_notebook_style_mean(contours_by_speaker: dict[str, dict[str, np.ndarray]], nc_resample: int, max_iter: int = 100, tol: float = 1e-8):
    """Replicate the notebook-style rigid GPA mean template in pixel scale."""
    speaker_ids = sorted(contours_by_speaker)
    label_sets = [set(contours_by_speaker[s].keys()) for s in speaker_ids]
    common_labels = sorted(set.intersection(*label_sets))
    if not common_labels:
        raise ValueError("No common contour labels found across speakers.")

    contours_resampled = {}
    speaker_shapes = {}
    for speaker_id in speaker_ids:
        contours_resampled[speaker_id] = {}
        parts = []
        for label in common_labels:
            pts = np.asarray(contours_by_speaker[speaker_id][label], dtype=float)
            pts = resample_polyline(pts, nc_resample)
            contours_resampled[speaker_id][label] = pts
            parts.append(pts)
        speaker_shapes[speaker_id] = np.vstack(parts)

    template = np.mean([speaker_shapes[s] for s in speaker_ids], axis=0)
    template -= template.mean(axis=0)

    rigid_transforms = {}
    speaker_shapes_aligned = {}
    for _ in range(max_iter):
        aligned_iter = {}
        for speaker_id in speaker_ids:
            aligned, rot, shift = rigid_align(speaker_shapes[speaker_id], template)
            rigid_transforms[speaker_id] = (rot, shift)
            aligned_iter[speaker_id] = aligned

        new_template = np.mean([aligned_iter[s] for s in speaker_ids], axis=0)
        new_template -= new_template.mean(axis=0)
        delta = float(np.sqrt(np.mean((new_template - template) ** 2)))
        template = new_template
        speaker_shapes_aligned = aligned_iter
        if delta < tol:
            break

    all_contours_aligned = {speaker_id: {} for speaker_id in speaker_ids}
    template_per_label = {}
    median_per_label = {}
    label_variability_to_median = {}
    for li, label in enumerate(common_labels):
        start = li * nc_resample
        stop = (li + 1) * nc_resample
        template_per_label[label] = template[start:stop]
        for speaker_id in speaker_ids:
            all_contours_aligned[speaker_id][label] = speaker_shapes_aligned[speaker_id][start:stop]
        label_stack = np.stack([all_contours_aligned[speaker_id][label] for speaker_id in speaker_ids], axis=0)
        median_per_label[label] = np.median(label_stack, axis=0)
        label_variability_to_median[label] = float(
            np.mean(np.sqrt(np.mean(np.sum((label_stack - median_per_label[label][None, :, :]) ** 2, axis=2), axis=1)))
        )

    rms_to_template = {
        speaker_id: float(np.sqrt(np.mean((speaker_shapes_aligned[speaker_id] - template) ** 2)))
        for speaker_id in speaker_ids
    }
    median_shape = np.vstack([median_per_label[label] for label in common_labels])
    speaker_rms_to_median = {
        speaker_id: float(np.sqrt(np.mean((speaker_shapes_aligned[speaker_id] - median_shape) ** 2)))
        for speaker_id in speaker_ids
    }

    pairwise = np.zeros((len(speaker_ids), len(speaker_ids)), dtype=float)
    for i, speaker_i in enumerate(speaker_ids):
        for j, speaker_j in enumerate(speaker_ids):
            if i == j:
                continue
            pairwise[i, j] = float(np.sqrt(np.mean((speaker_shapes_aligned[speaker_i] - speaker_shapes_aligned[speaker_j]) ** 2)))

    sum_dist = {speaker_id: float(pairwise[i].sum()) for i, speaker_id in enumerate(speaker_ids)}
    geometric_median = min(sum_dist, key=sum_dist.get)
    closest_to_mean = min(rms_to_template, key=rms_to_template.get)

    return {
        "speaker_ids": speaker_ids,
        "common_labels": common_labels,
        "nc_per_label": nc_resample,
        "template_mean": template,
        "template_per_label": template_per_label,
        "median_per_label": median_per_label,
        "speaker_shapes": speaker_shapes,
        "speaker_shapes_aligned": speaker_shapes_aligned,
        "all_contours_aligned": all_contours_aligned,
        "rms_to_template": rms_to_template,
        "speaker_rms_to_median": speaker_rms_to_median,
        "label_variability_to_median": label_variability_to_median,
        "sum_dist": sum_dist,
        "geometric_median": geometric_median,
        "closest_to_mean": closest_to_mean,
        "pairwise": pairwise,
    }


def plot_contour_dict(ax, contour_dict, labels, colors, *, alpha=0.8, lw=2.0, linestyle="-", label_prefix=None):
    """Plot a contour dictionary with one color per label."""
    handles = []
    for label in labels:
        pts = np.asarray(contour_dict[label], dtype=float)
        line, = ax.plot(
            pts[:, 0],
            pts[:, 1],
            linestyle=linestyle if label != "tongue" else linestyle,
            color=colors[label],
            lw=2.8 if label == "tongue" else lw,
            alpha=0.98 if label == "tongue" else alpha,
        )
        if label_prefix is not None:
            handles.append((line, f"{label_prefix}{label}"))
    return handles


def centered_contours(contours: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Center a contour dictionary around its joint centroid."""
    all_pts = np.vstack([np.asarray(pts, dtype=float) for pts in contours.values()])
    center = all_pts.mean(axis=0)
    return {label: np.asarray(pts, dtype=float) - center for label, pts in contours.items()}


def resolve_reference_speaker(requested_reference: str, speakers: dict[str, dict], default_reference: str) -> str:
    """Resolve a reference speaker from a key or a display name."""
    if requested_reference == "auto":
        return default_reference
    if requested_reference in speakers:
        return requested_reference

    requested_norm = requested_reference.strip().upper()
    matches = [
        speaker_id
        for speaker_id, speaker in speakers.items()
        if str(speaker.get("display_name") or speaker_id).upper() == requested_norm
    ]
    if len(matches) == 1:
        return matches[0]
    raise ValueError(f"Reference speaker '{requested_reference}' not found in the loaded population.")


def speaker_display_name(speakers: dict[str, dict], speaker_id: str) -> str:
    return str(speakers[speaker_id].get("display_name") or speaker_id)


def speaker_order_key(speakers: dict[str, dict], speaker_id: str) -> tuple[int, str]:
    return speaker_id_sort_key(speaker_display_name(speakers, speaker_id))


def choose_speaker_colors(speakers: dict[str, dict]) -> dict[str, tuple[float, float, float, float]]:
    """Assign one stable color per speaker across every overlay figure."""
    cmap = plt.get_cmap("tab20")
    ordered_ids = sorted(speakers, key=lambda speaker_id: speaker_order_key(speakers, speaker_id))
    return {
        speaker_id: cmap(index % 20)
        for index, speaker_id in enumerate(ordered_ids)
    }


def format_white_canvas(ax, image, title: str) -> None:
    image_arr = np.asarray(image)
    height, width = image_arr.shape[:2]
    ax.set_facecolor("white")
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xlim(0, width)
    ax.set_ylim(height, 0)
    ax.set_aspect("equal")
    ax.axis("off")


def map_grid_lines(grid, mapping_fn) -> dict[str, list[np.ndarray]]:
    """Map an existing grid line set without rebuilding the grid."""
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
    """Compute stage-level mean/median summaries without changing existing metrics code."""
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
    speakers: dict[str, dict],
    common_labels: list[str],
    reference_id: str,
    nc_template: int,
) -> dict[str, dict[str, object]]:
    """Compute init, affine-only, and affine+TPS populations."""
    reference = speakers[reference_id]
    mapped_payload = {
        "init": {"contours": {}, "grids": {}},
        "affine": {"contours": {}, "grids": {}},
        "tps": {"contours": {}, "grids": {}},
    }

    for speaker_id, speaker in speakers.items():
        identity_map = lambda pts: np.asarray(pts, dtype=float)
        init_contours = {label: np.asarray(speaker["contours"][label], dtype=float) for label in common_labels}
        init_grid = map_grid_lines(speaker["grid"], identity_map)
        if speaker_id == reference_id:
            affine_contours = {label: np.asarray(speaker["contours"][label], dtype=float) for label in common_labels}
            tps_contours = {label: np.asarray(speaker["contours"][label], dtype=float) for label in common_labels}
            affine_grid = map_grid_lines(speaker["grid"], identity_map)
            tps_grid = map_grid_lines(speaker["grid"], identity_map)
        else:
            transform = build_two_step_transform(speaker["grid"], reference["grid"])

            def apply_affine_only(points, *, affine=transform["step1_affine"]):
                return np.asarray(apply_transform(affine, np.asarray(points, dtype=float)), dtype=float)

            affine_contours = transform_contours(speaker["contours"], apply_affine_only, common_labels)
            tps_contours = transform_contours(speaker["contours"], transform["apply_two_step"], common_labels)
            affine_grid = map_grid_lines(speaker["grid"], apply_affine_only)
            tps_grid = map_grid_lines(speaker["grid"], transform["apply_two_step"])

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


def plot_single_speaker_contours(
    ax,
    contour_dict: dict[str, np.ndarray],
    labels: list[str],
    color,
    *,
    alpha: float,
    lw: float,
    zorder: int,
) -> None:
    """Plot every contour for one speaker using one consistent color."""
    for label in labels:
        pts = np.asarray(contour_dict[label], dtype=float)
        ax.plot(
            pts[:, 0],
            pts[:, 1],
            color=color,
            lw=lw * (1.12 if label == "tongue" else 1.0),
            alpha=alpha,
            zorder=zorder,
        )


def plot_single_speaker_grid(
    ax,
    grid_payload: dict[str, list[np.ndarray]],
    color,
    *,
    alpha: float,
    lw: float,
    zorder: int,
) -> None:
    """Plot mapped grid lines for one speaker using one consistent color."""
    horiz_lines = list(grid_payload["horiz_lines"])
    vert_lines = list(grid_payload["vert_lines"])
    for index, line in enumerate(horiz_lines):
        boundary = index in {0, len(horiz_lines) - 1}
        ax.plot(line[:, 0], line[:, 1], color=color, lw=lw * (1.2 if boundary else 1.0), alpha=alpha, zorder=zorder)
    for index, line in enumerate(vert_lines):
        boundary = index in {0, len(vert_lines) - 1}
        ax.plot(line[:, 0], line[:, 1], color=color, lw=lw * (1.15 if boundary else 0.92), alpha=alpha, zorder=zorder)


def add_speaker_legend(
    fig,
    speakers: dict[str, dict],
    ordered_ids: list[str],
    speaker_colors: dict[str, tuple[float, float, float, float]],
    median_speaker: str,
) -> None:
    handles = []
    labels = []
    for speaker_id in ordered_ids:
        is_median = speaker_id == median_speaker
        handles.append(
            plt.Line2D(
                [0],
                [0],
                color=speaker_colors[speaker_id],
                lw=3.2 if is_median else 1.8,
                alpha=0.98 if is_median else 0.82,
            )
        )
        label = speaker_display_name(speakers, speaker_id)
        if is_median:
            label += " (median)"
        labels.append(label)
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=min(6, len(handles)),
        fontsize=9,
        bbox_to_anchor=(0.5, -0.02),
        framealpha=0.96,
    )


def save_stage_contour_overlay_figure(
    speakers: dict[str, dict],
    labels: list[str],
    stage_payload: dict[str, object],
    speaker_colors: dict[str, tuple[float, float, float, float]],
    *,
    stage_name: str,
    cohort_name: str,
    output_path: Path,
) -> None:
    reference = speakers[str(stage_payload["reference_id"])]
    median_speaker = str(stage_payload["median_speaker"])
    ordered_ids = sorted(stage_payload["mapped_contours"], key=lambda speaker_id: speaker_order_key(speakers, speaker_id))
    draw_order = [speaker_id for speaker_id in ordered_ids if speaker_id != median_speaker] + [median_speaker]
    stage_label = stage_display_name(stage_name)
    cohort_label = "all speakers" if cohort_name == "all" else f"{cohort_name} cohort"

    fig, ax = plt.subplots(1, 1, figsize=(10, 10), dpi=180)
    format_white_canvas(ax, reference["image"], f"{stage_label}: contour overlay ({cohort_label})")

    for speaker_id in draw_order:
        is_median = speaker_id == median_speaker
        plot_single_speaker_contours(
            ax,
            stage_payload["mapped_contours"][speaker_id],
            labels,
            speaker_colors[speaker_id],
            alpha=0.98 if is_median else DEFAULT_STAGE_ALPHA,
            lw=DEFAULT_STAGE_HIGHLIGHT_LINEWIDTH if is_median else DEFAULT_STAGE_LINEWIDTH,
            zorder=5 if is_median else 2,
        )

    ax.text(
        0.02,
        0.02,
        "\n".join(
            [
                f"Speakers: {len(ordered_ids)}",
                (
                    f"Later reference: {speaker_display_name(speakers, str(stage_payload['reference_id']))}"
                    if stage_name == "init"
                    else f"Reference: {speaker_display_name(speakers, str(stage_payload['reference_id']))}"
                ),
                f"Median speaker: {speaker_display_name(speakers, median_speaker)}",
            ]
        ),
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=10,
        bbox=dict(boxstyle="round,pad=0.35", fc="white", alpha=0.97),
    )
    add_speaker_legend(fig, speakers, ordered_ids, speaker_colors, median_speaker)
    fig.tight_layout(rect=(0.0, 0.05, 1.0, 1.0))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def save_stage_grid_overlay_figure(
    speakers: dict[str, dict],
    stage_payload: dict[str, object],
    speaker_colors: dict[str, tuple[float, float, float, float]],
    *,
    stage_name: str,
    cohort_name: str,
    output_path: Path,
) -> None:
    reference = speakers[str(stage_payload["reference_id"])]
    median_speaker = str(stage_payload["median_speaker"])
    ordered_ids = sorted(stage_payload["mapped_grids"], key=lambda speaker_id: speaker_order_key(speakers, speaker_id))
    draw_order = [speaker_id for speaker_id in ordered_ids if speaker_id != median_speaker] + [median_speaker]
    stage_label = stage_display_name(stage_name)
    cohort_label = "all speakers" if cohort_name == "all" else f"{cohort_name} cohort"

    fig, ax = plt.subplots(1, 1, figsize=(10, 10), dpi=180)
    format_white_canvas(ax, reference["image"], f"{stage_label}: grid overlay ({cohort_label})")

    for speaker_id in draw_order:
        is_median = speaker_id == median_speaker
        plot_single_speaker_grid(
            ax,
            stage_payload["mapped_grids"][speaker_id],
            speaker_colors[speaker_id],
            alpha=0.98 if is_median else DEFAULT_STAGE_GRID_ALPHA,
            lw=2.2 if is_median else 0.9,
            zorder=5 if is_median else 2,
        )

    ax.text(
        0.02,
        0.02,
        "\n".join(
            [
                f"Speakers: {len(ordered_ids)}",
                (
                    f"Later reference: {speaker_display_name(speakers, str(stage_payload['reference_id']))}"
                    if stage_name == "init"
                    else f"Reference: {speaker_display_name(speakers, str(stage_payload['reference_id']))}"
                ),
                f"Median speaker: {speaker_display_name(speakers, median_speaker)}",
            ]
        ),
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=10,
        bbox=dict(boxstyle="round,pad=0.35", fc="white", alpha=0.97),
    )
    add_speaker_legend(fig, speakers, ordered_ids, speaker_colors, median_speaker)
    fig.tight_layout(rect=(0.0, 0.05, 1.0, 1.0))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def export_stage_overlay_figures(
    speakers: dict[str, dict],
    *,
    requested_reference: str,
    nc_resample: int,
    nc_template: int,
    output_dir: Path,
) -> dict[str, object]:
    """Export white-background contour and grid overlays for all/male/female cohorts."""
    stage_dir = output_dir / "stage_overlays"
    stage_dir.mkdir(parents=True, exist_ok=True)
    speaker_colors = choose_speaker_colors(speakers)
    summary_payload: dict[str, object] = {
        "nnunet_gender_assignment": CURATED_PLUS_NNUNET_GENDER["nnUNet_A"],
        "cohorts": {},
    }

    for cohort_name in DEFAULT_COHORTS:
        if cohort_name == "all":
            cohort_speakers = dict(speakers)
        else:
            cohort_speakers = {
                speaker_id: speaker
                for speaker_id, speaker in speakers.items()
                if speaker.get("gender") == cohort_name
            }
        if len(cohort_speakers) < 2:
            continue

        contours_by_speaker = {
            speaker_id: speaker["contours"]
            for speaker_id, speaker in cohort_speakers.items()
        }
        cohort_ms = compute_notebook_style_mean(contours_by_speaker, nc_resample=nc_resample)
        reference_id = resolve_reference_speaker(requested_reference, cohort_speakers, str(cohort_ms["geometric_median"]))
        stage_populations = compute_stage_populations(
            cohort_speakers,
            cohort_ms["common_labels"],
            reference_id,
            nc_template=nc_template,
        )

        cohort_summary = {
            "speaker_count": len(cohort_speakers),
            "speakers": [speaker_display_name(cohort_speakers, speaker_id) for speaker_id in sorted(cohort_speakers, key=lambda sid: speaker_order_key(cohort_speakers, sid))],
            "reference_speaker": speaker_display_name(cohort_speakers, reference_id),
            "common_labels": list(cohort_ms["common_labels"]),
            "stages": {},
        }

        for stage_name in DEFAULT_STAGE_ORDER:
            contour_path = stage_dir / f"{cohort_name}_{stage_name}_contours_overlay.png"
            grid_path = stage_dir / f"{cohort_name}_{stage_name}_grid_overlay.png"
            save_stage_contour_overlay_figure(
                cohort_speakers,
                cohort_ms["common_labels"],
                stage_populations[stage_name],
                speaker_colors,
                stage_name=stage_name,
                cohort_name=cohort_name,
                output_path=contour_path,
            )
            save_stage_grid_overlay_figure(
                cohort_speakers,
                stage_populations[stage_name],
                speaker_colors,
                stage_name=stage_name,
                cohort_name=cohort_name,
                output_path=grid_path,
            )
            cohort_summary["stages"][stage_name] = {
                "stage_label": stage_display_name(stage_name),
                "median_speaker": speaker_display_name(cohort_speakers, str(stage_populations[stage_name]["median_speaker"])),
                "closest_to_mean": speaker_display_name(cohort_speakers, str(stage_populations[stage_name]["closest_to_mean"])),
                "contour_overlay": str(contour_path),
                "grid_overlay": str(grid_path),
            }

        summary_payload["cohorts"][cohort_name] = cohort_summary

    summary_json_path = stage_dir / "stage_overlay_summary.json"
    summary_txt_path = stage_dir / "stage_overlay_summary.txt"
    summary_json_path.write_text(json.dumps(summary_payload, indent=2, ensure_ascii=False), encoding="utf-8")

    lines = [
        "Average-Speaker Stage Overlay Summary",
        "=====================================",
        "nnUNet_A cohort assignment: female",
        "",
    ]
    for cohort_name in DEFAULT_COHORTS:
        cohort_summary = summary_payload["cohorts"].get(cohort_name)
        if not cohort_summary:
            continue
        lines.append(f"[{cohort_name}] speakers ({cohort_summary['speaker_count']}): {', '.join(cohort_summary['speakers'])}")
        lines.append(f"reference speaker: {cohort_summary['reference_speaker']}")
        for stage_name in DEFAULT_STAGE_ORDER:
            stage_summary = cohort_summary["stages"][stage_name]
            lines.append(
                f"{stage_name} ({stage_summary['stage_label']}): median={stage_summary['median_speaker']} | "
                f"closest_to_mean={stage_summary['closest_to_mean']}"
            )
            lines.append(f"  contour overlay: {stage_summary['contour_overlay']}")
            lines.append(f"  grid overlay: {stage_summary['grid_overlay']}")
        lines.append("")
    summary_txt_path.write_text("\n".join(lines), encoding="utf-8")
    summary_payload["summary_json"] = str(summary_json_path)
    summary_payload["summary_txt"] = str(summary_txt_path)
    return summary_payload


def compute_transformed_population(speakers: dict[str, dict], common_labels: list[str], reference_id: str, nc_template: int):
    """Warp every speaker into the chosen reference grid space and compute mean/median contours."""
    reference = speakers[reference_id]
    mapped_contours = {}
    transform_meta = {}

    for speaker_id, speaker in speakers.items():
        if speaker_id == reference_id:
            mapped = {label: np.asarray(speaker["contours"][label], dtype=float) for label in common_labels}
            transform_meta[speaker_id] = {"step1": [], "step2": []}
        else:
            transform = build_two_step_transform(speaker["grid"], reference["grid"])
            mapped = transform_contours(speaker["contours"], transform["apply_two_step"], common_labels)
            mapped = smooth_transformed_contours(mapped)
            transform_meta[speaker_id] = {
                "step1": transform["step1_labels"],
                "step2": transform["step2_labels"],
            }
        mapped_contours[speaker_id] = mapped

    mean_contours = {}
    median_contours = {}
    per_label_stack = {}
    label_variability_to_median = {}
    for label in common_labels:
        stack = np.stack([resample_polyline(mapped_contours[speaker_id][label], nc_template) for speaker_id in sorted(mapped_contours)], axis=0)
        per_label_stack[label] = stack
        mean_contours[label] = np.mean(stack, axis=0)
        median_contours[label] = np.median(stack, axis=0)
        label_variability_to_median[label] = float(
            np.mean(np.sqrt(np.mean(np.sum((stack - median_contours[label][None, :, :]) ** 2, axis=2), axis=1)))
        )

    mean_shape = np.vstack([mean_contours[label] for label in common_labels])
    median_shape = np.vstack([median_contours[label] for label in common_labels])
    mean_rms = {}
    median_rms = {}
    for speaker_id in sorted(mapped_contours):
        speaker_shape = np.vstack([resample_polyline(mapped_contours[speaker_id][label], nc_template) for label in common_labels])
        mean_rms[speaker_id] = float(np.sqrt(np.mean((speaker_shape - mean_shape) ** 2)))
        median_rms[speaker_id] = float(np.sqrt(np.mean((speaker_shape - median_shape) ** 2)))

    return {
        "reference_id": reference_id,
        "mapped_contours": mapped_contours,
        "mean_contours": mean_contours,
        "median_contours": median_contours,
        "mean_rms": mean_rms,
        "median_rms": median_rms,
        "label_variability_to_median": label_variability_to_median,
        "transform_meta": transform_meta,
        "nc_template": nc_template,
    }


def save_notebook_style_figure(ms: dict, output_path: Path):
    """Save a notebook-style variability summary without drawing the mean contour."""
    label_colors = choose_label_colors(ms["common_labels"])
    fig, axes = plt.subplots(1, 3, figsize=(25, 9))

    ax = axes[0]
    for speaker_id in ms["speaker_ids"]:
        centered = centered_contours({
            label: ms["speaker_shapes"][speaker_id][li * ms["nc_per_label"]:(li + 1) * ms["nc_per_label"]]
            for li, label in enumerate(ms["common_labels"])
        })
        plot_contour_dict(ax, centered, ms["common_labels"], label_colors, alpha=DEFAULT_ALPHA, lw=1.2)
    ax.set_title("Notebook-style mean speaker\nbefore/centred overlay", fontsize=14, fontweight="bold")
    ax.set_aspect("equal")
    ax.invert_yaxis()
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    for speaker_id in ms["speaker_ids"]:
        plot_contour_dict(ax, ms["all_contours_aligned"][speaker_id], ms["common_labels"], label_colors, alpha=DEFAULT_ALPHA, lw=1.1)
    plot_contour_dict(ax, ms["all_contours_aligned"][ms["geometric_median"]], ms["common_labels"], label_colors, alpha=0.98, lw=3.0)
    ax.set_title(
        f"After rigid GPA\nhighlighted median speaker = {ms['geometric_median']}",
        fontsize=14,
        fontweight="bold",
    )
    ax.set_aspect("equal")
    ax.invert_yaxis()
    ax.grid(True, alpha=0.3)

    ax = axes[2]
    items = sorted(ms["label_variability_to_median"].items(), key=lambda item: item[1], reverse=True)
    labels = [item[0] for item in items]
    values = [item[1] for item in items]
    y_pos = np.arange(len(labels))
    bar_colors = [TONGUE_COLOR if label == "tongue" else label_colors[label] for label in labels]
    ax.barh(y_pos, values, color=bar_colors, edgecolor="black", alpha=0.85)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("RMS to pointwise median (px)", fontsize=10)
    ax.set_title("Speaker variability by articulator\n(after rigid GPA)", fontsize=14, fontweight="bold")
    ax.invert_yaxis()
    ax.grid(True, axis="x", alpha=0.3)
    for yi, val in zip(y_pos, values):
        ax.text(val + 0.15, yi, f"{val:.2f}", va="center", fontsize=8.5)

    legend_handles = [plt.Line2D([0], [0], color=label_colors[label], lw=3, label=label) for label in ms["common_labels"]]
    fig.legend(legend_handles, ms["common_labels"], loc="lower center", ncol=min(5, len(ms["common_labels"])), fontsize=8, bbox_to_anchor=(0.5, -0.02))
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_transformed_average_figure(speakers: dict[str, dict], ms: dict, transformed: dict, output_path: Path):
    """Save the transformed population and variability without drawing the mean contour."""
    reference_id = transformed["reference_id"]
    reference = speakers[reference_id]
    label_colors = choose_label_colors(ms["common_labels"])
    speaker_colors = plt.get_cmap("tab10")

    fig, axes = plt.subplots(1, 3, figsize=(28, 10))

    ax = axes[0]
    format_target_frame(ax, reference["image"], f"All speakers mapped to reference: {reference_id}")
    for idx, speaker_id in enumerate(sorted(transformed["mapped_contours"])):
        mapped = transformed["mapped_contours"][speaker_id]
        for label in ms["common_labels"]:
            pts = mapped[label]
            ax.plot(
                pts[:, 0],
                pts[:, 1],
                color=TONGUE_COLOR if label == "tongue" else speaker_colors(idx % 10),
                lw=1.3 if label != "tongue" else 1.8,
                alpha=0.18 if label != "tongue" else 0.26,
                linestyle="-" if speaker_id == reference_id else "--",
            )
    ax.text(
        0.02,
        0.02,
        f"{len(transformed['mapped_contours'])} speakers\nall mapped contours shown",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=10,
        bbox=dict(boxstyle="round,pad=0.35", fc="white", alpha=0.95),
    )

    ax = axes[1]
    format_target_frame(ax, reference["image"], "Reference contour (gray) vs transformed median contour")
    for label in ms["common_labels"]:
        ref_pts = np.asarray(reference["contours"][label], dtype=float)
        ax.plot(ref_pts[:, 0], ref_pts[:, 1], color="0.55", lw=1.0, alpha=0.4)
    for speaker_id in sorted(transformed["mapped_contours"]):
        plot_contour_dict(ax, transformed["mapped_contours"][speaker_id], ms["common_labels"], label_colors, alpha=0.08, lw=1.0, linestyle="--")
    plot_contour_dict(ax, transformed["median_contours"], ms["common_labels"], label_colors, alpha=0.98, lw=3.0, label_prefix="Median: ")
    ax.set_title("Reference contour (gray) vs transformed median contour", fontsize=14, fontweight="bold")
    legend_handles = [
        plt.Line2D([0], [0], color="black", lw=3.0),
        plt.Line2D([0], [0], color="0.55", lw=1.4),
    ]
    ax.legend(legend_handles, ["Median contour", "Reference speaker"], loc="upper right", fontsize=10, framealpha=0.92)
    ax.text(
        0.02,
        0.02,
        f"Reference = {reference_id}\nMedian speaker = {ms['geometric_median']}\nClosest-to-mean = {ms['closest_to_mean']}",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=9.5,
        bbox=dict(boxstyle="round,pad=0.35", fc="white", alpha=0.95),
    )

    ax = axes[2]
    items = sorted(transformed["label_variability_to_median"].items(), key=lambda item: item[1], reverse=True)
    labels = [item[0] for item in items]
    values = [item[1] for item in items]
    y_pos = np.arange(len(labels))
    bar_colors = [TONGUE_COLOR if label == "tongue" else label_colors[label] for label in labels]
    ax.barh(y_pos, values, color=bar_colors, edgecolor="black", alpha=0.85)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("RMS to transformed median (px)", fontsize=10)
    ax.set_title("Speaker variability by articulator\n(after grid transform)", fontsize=14, fontweight="bold")
    ax.invert_yaxis()
    ax.grid(True, axis="x", alpha=0.3)
    for yi, val in zip(y_pos, values):
        ax.text(val + 0.15, yi, f"{val:.2f}", va="center", fontsize=8.5)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def write_summary(path: Path, speakers: dict[str, dict], ms: dict, transformed: dict):
    """Write a short summary file for the averaged speaker experiment."""
    lines = [
        "Average/Median Speaker Summary",
        "================================",
        f"Speakers used ({len(speakers)}): {', '.join(sorted(speakers))}",
        f"Common labels ({len(ms['common_labels'])}): {', '.join(ms['common_labels'])}",
        f"Geometric median speaker: {ms['geometric_median']}",
        f"Closest to rigid-GPA mean: {ms['closest_to_mean']}",
        f"Reference speaker for grid transform: {transformed['reference_id']}",
        "",
        "RMS to transformed median contour (lower is better):",
    ]
    for speaker_id, value in sorted(transformed["median_rms"].items(), key=lambda item: item[1]):
        lines.append(f"  {speaker_id:20s} {value:7.2f} px")
    lines.append("")
    lines.append("Per-label variability after grid transform (RMS to pointwise median):")
    for label, value in sorted(transformed["label_variability_to_median"].items(), key=lambda item: item[1], reverse=True):
        lines.append(f"  {label:20s} {value:7.2f} px")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def load_speakers(case: str, source_frame: int, vtln_dir: Path, include_nnunet: bool):
    """Load the speaker set and record which speakers are missing tongue."""
    speakers = {}
    missing_tongue = []

    if include_nnunet:
        image, contours = load_frame_npy(
            source_frame,
            VT_SEG_DATA_ROOT / case,
            VT_SEG_CONTOURS_ROOT / case,
        )
        if "tongue" not in contours:
            missing_tongue.append("nnUNet_A")
        speakers["nnUNet_A"] = {
            "image": image,
            "contours": contours,
            "source": "nnUNet",
            "display_name": "nnUNet_A",
            "gender": CURATED_PLUS_NNUNET_GENDER["nnUNet_A"],
        }

    curated_specs = read_curated_specs_map(vtln_dir)
    for speaker_code in sorted(curated_specs, key=speaker_id_sort_key):
        spec = curated_specs[speaker_code]
        speaker_id = spec.basename
        image, contours = load_frame_vtln(speaker_id, vtln_dir)
        if "tongue" not in contours:
            missing_tongue.append(speaker_code)
        speakers[speaker_id] = {
            "image": image,
            "contours": contours,
            "source": "VTLN",
            "display_name": speaker_code,
            "gender": spec.gender,
            "basename": spec.basename,
        }

    for speaker_id, speaker in speakers.items():
        speaker["grid"] = build_grid(speaker["image"], speaker["contours"], n_vert=9, n_points=250, frame_number=0 if speaker["source"] == "VTLN" else source_frame)

    return speakers, missing_tongue


def build_parser():
    parser = argparse.ArgumentParser(description="Compute notebook-style and grid-transformed average/median speakers with tongue included.")
    parser.add_argument("--source-frame", type=int, default=DEFAULT_SOURCE_FRAME, help="nnUNet source frame to include as nnUNet_A.")
    parser.add_argument("--case", default=DEFAULT_CASE, help="nnUNet case relative path.")
    parser.add_argument("--vtln-dir", type=Path, default=DEFAULT_VTLN_DIR, help="Folder containing VTLN images and ROI zip files.")
    parser.add_argument("--exclude-nnunet", action="store_true", help="Use only VTLN speakers.")
    parser.add_argument(
        "--reference-speaker",
        default="auto",
        help="Reference speaker for grid transform. Default 'auto' uses the geometric median speaker from the rigid-GPA stage.",
    )
    parser.add_argument("--nc-resample", type=int, default=DEFAULT_RESAMPLE_POINTS, help="Resample count for the notebook-style GPA stage.")
    parser.add_argument("--nc-template", type=int, default=DEFAULT_TEMPLATE_POINTS, help="Resample count for the transformed mean/median contours.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Where to save figures and summary.")
    return parser


def main(argv: list[str] | None = None):
    args = build_parser().parse_args(argv)
    speakers, missing_tongue = load_speakers(
        case=args.case,
        source_frame=args.source_frame,
        vtln_dir=args.vtln_dir,
        include_nnunet=not args.exclude_nnunet,
    )
    if len(speakers) < 2:
        raise ValueError("Need at least two speakers with tongue to compute an average or median speaker.")

    contours_by_speaker = {speaker_id: speaker["contours"] for speaker_id, speaker in speakers.items()}
    ms = compute_notebook_style_mean(contours_by_speaker, nc_resample=args.nc_resample)

    reference_id = resolve_reference_speaker(args.reference_speaker, speakers, str(ms["geometric_median"]))

    transformed = compute_transformed_population(speakers, ms["common_labels"], reference_id, nc_template=args.nc_template)
    stage_overlay_summary = export_stage_overlay_figures(
        speakers,
        requested_reference=args.reference_speaker,
        nc_resample=args.nc_resample,
        nc_template=args.nc_template,
        output_dir=args.output_dir,
    )

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    notebook_fig = output_dir / "average_speaker_notebook_style_tongue.png"
    transformed_fig = output_dir / "average_speaker_after_grid_transform_tongue.png"
    summary_txt = output_dir / "average_speaker_grid_transform_summary.txt"

    save_notebook_style_figure(ms, notebook_fig)
    save_transformed_average_figure(speakers, ms, transformed, transformed_fig)
    write_summary(summary_txt, speakers, ms, transformed)

    print("Speakers missing tongue contour but still included:", ", ".join(missing_tongue) if missing_tongue else "none")
    print(
        "Speakers used:",
        ", ".join(
            speaker_display_name(speakers, speaker_id)
            for speaker_id in sorted(speakers, key=lambda speaker_id: speaker_order_key(speakers, speaker_id))
        ),
    )
    print("Common labels:", ", ".join(ms["common_labels"]))
    print("Geometric median speaker:", speaker_display_name(speakers, str(ms["geometric_median"])))
    print("Closest to notebook-style mean:", speaker_display_name(speakers, str(ms["closest_to_mean"])))
    print("Reference speaker for grid transform:", speaker_display_name(speakers, reference_id))
    print(f"Saved: {notebook_fig}")
    print(f"Saved: {transformed_fig}")
    print(f"Saved: {summary_txt}")
    print(f"Saved: {stage_overlay_summary['summary_json']}")
    print(f"Saved: {stage_overlay_summary['summary_txt']}")


if __name__ == "__main__":
    main()
