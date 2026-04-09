from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from grid_transform.analysis_shared import choose_label_colors
from grid_transform.config import (
    DEFAULT_OUTPUT_DIR,
    DEFAULT_VTLN_DIR,
    TONGUE_COLOR,
    VT_SEG_CONTOURS_ROOT,
    VT_SEG_DATA_ROOT,
)
from grid_transform.io import load_frame_npy, load_frame_vtln
from grid_transform.transfer import build_two_step_transform, smooth_transformed_contours, transform_contours
from grid_transform.transform_helpers import format_target_frame, resample_polyline
from grid_transform.vt import build_grid


DEFAULT_CASE = "2008-003^01-1791/test"
DEFAULT_SOURCE_FRAME = 143020
DEFAULT_RESAMPLE_POINTS = 80
DEFAULT_TEMPLATE_POINTS = 80
DEFAULT_ALPHA = 0.22


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
    """Load the speaker set and keep only speakers that contain tongue."""
    speakers = {}
    excluded_no_tongue = []

    if include_nnunet:
        image, contours = load_frame_npy(
            source_frame,
            VT_SEG_DATA_ROOT / case,
            VT_SEG_CONTOURS_ROOT / case,
        )
        if "tongue" in contours:
            speakers["nnUNet_A"] = {"image": image, "contours": contours, "source": "nnUNet"}
        else:
            excluded_no_tongue.append("nnUNet_A")

    for zip_path in sorted(vtln_dir.glob("*.zip")):
        speaker_id = zip_path.stem
        image, contours = load_frame_vtln(speaker_id, vtln_dir)
        if "tongue" not in contours:
            excluded_no_tongue.append(speaker_id)
            continue
        speakers[speaker_id] = {"image": image, "contours": contours, "source": "VTLN"}

    for speaker_id, speaker in speakers.items():
        speaker["grid"] = build_grid(speaker["image"], speaker["contours"], n_vert=9, n_points=250, frame_number=0 if speaker["source"] == "VTLN" else source_frame)

    return speakers, excluded_no_tongue


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
    speakers, excluded_no_tongue = load_speakers(
        case=args.case,
        source_frame=args.source_frame,
        vtln_dir=args.vtln_dir,
        include_nnunet=not args.exclude_nnunet,
    )
    if len(speakers) < 2:
        raise ValueError("Need at least two speakers with tongue to compute an average or median speaker.")

    contours_by_speaker = {speaker_id: speaker["contours"] for speaker_id, speaker in speakers.items()}
    ms = compute_notebook_style_mean(contours_by_speaker, nc_resample=args.nc_resample)

    if args.reference_speaker == "auto":
        reference_id = ms["geometric_median"]
    else:
        reference_id = args.reference_speaker
        if reference_id not in speakers:
            raise ValueError(f"Reference speaker '{reference_id}' not found in the loaded population.")

    transformed = compute_transformed_population(speakers, ms["common_labels"], reference_id, nc_template=args.nc_template)

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    notebook_fig = output_dir / "average_speaker_notebook_style_tongue.png"
    transformed_fig = output_dir / "average_speaker_after_grid_transform_tongue.png"
    summary_txt = output_dir / "average_speaker_grid_transform_summary.txt"

    save_notebook_style_figure(ms, notebook_fig)
    save_transformed_average_figure(speakers, ms, transformed, transformed_fig)
    write_summary(summary_txt, speakers, ms, transformed)

    print("Excluded because tongue was missing:", ", ".join(excluded_no_tongue) if excluded_no_tongue else "none")
    print("Speakers used:", ", ".join(sorted(speakers)))
    print("Common labels:", ", ".join(ms["common_labels"]))
    print("Geometric median speaker:", ms["geometric_median"])
    print("Closest to notebook-style mean:", ms["closest_to_mean"])
    print("Reference speaker for grid transform:", reference_id)
    print(f"Saved: {notebook_fig}")
    print(f"Saved: {transformed_fig}")
    print(f"Saved: {summary_txt}")


if __name__ == "__main__":
    main()
