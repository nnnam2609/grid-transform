from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from grid_transform.config import TONGUE_COLOR
from grid_transform.figures import format_frame
from grid_transform.transfer import DEFAULT_ARTICULATORS, resolve_common_articulators
from grid_transform.transform_helpers import polyline_rms, resample_polyline


MOVED_SOURCE_COLOR = "#ef476f"
RAW_TARGET_COLOR = "#16a085"
SOURCE_ARTIC_COLOR = "#3a86ff"


def parse_articulators(text: str | None, target_contours: dict, source_contours: dict):
    """Resolve the articulator names to compare."""
    return resolve_common_articulators(
        target_contours,
        source_contours,
        text,
        defaults=DEFAULT_ARTICULATORS,
    )


def compute_articulator_errors(moved_contours: dict, target_contours: dict):
    """Compute per-articulator RMS after resampling onto a common number of points."""
    errors = {}
    for name, moved in moved_contours.items():
        target = np.asarray(target_contours[name], dtype=float)
        n_samples = max(len(moved), len(target), 120)
        moved_rs = resample_polyline(moved, n=n_samples)
        target_rs = resample_polyline(target, n=n_samples)
        errors[name] = polyline_rms(moved_rs, target_rs)
    return errors


def plot_contours(ax, contours: dict, names, color, *, lw=2.0, alpha=0.9, label_prefix=None, linestyle="-"):
    """Draw a selected contour set on an axes."""
    handles = []
    for name in names:
        contour = np.asarray(contours[name], dtype=float)
        line_color = TONGUE_COLOR if name == "tongue" else color
        line_style = "--" if name == "tongue" and "Moved source" in (label_prefix or "") else linestyle
        handle, = ax.plot(contour[:, 0], contour[:, 1], color=line_color, lw=lw, alpha=alpha, linestyle=line_style)
        handles.append((handle, f"{label_prefix}{name}" if label_prefix else name))
    return handles


def save_comparison_figure(
    target_image,
    source_image,
    target_contours,
    source_contours,
    moved_source_contours,
    articulators,
    errors,
    output_path: Path,
):
    """Render a three-panel articulator-transfer comparison figure."""
    fig, axes = plt.subplots(1, 3, figsize=(22, 7))

    ax = axes[0]
    format_frame(ax, target_image, "Target speaker\nraw articulators")
    plot_contours(ax, target_contours, articulators, RAW_TARGET_COLOR, lw=2.2, alpha=0.95)

    ax = axes[1]
    format_frame(ax, source_image, "Source speaker\nraw articulators")
    plot_contours(ax, source_contours, articulators, SOURCE_ARTIC_COLOR, lw=2.2, alpha=0.95)

    ax = axes[2]
    format_frame(ax, target_image, "Moved source vs target\nin target space")
    target_handles = plot_contours(ax, target_contours, articulators, RAW_TARGET_COLOR, lw=2.2, alpha=0.90, label_prefix="Target: ")
    moved_handles = plot_contours(ax, moved_source_contours, articulators, MOVED_SOURCE_COLOR, lw=2.2, alpha=0.90, label_prefix="Moved source: ")

    legend_handles = []
    legend_labels = []
    for handle, label in target_handles[:1] + moved_handles[:1]:
        legend_handles.append(handle)
        legend_labels.append(label.split(":")[0])
    ax.legend(legend_handles, legend_labels, loc="upper right", framealpha=0.92, fontsize=10)

    report_lines = ["Articulator RMS (px)"]
    for name in articulators:
        report_lines.append(f"{name}: {errors[name]:.2f}")
    ax.text(
        0.02,
        0.02,
        "\n".join(report_lines),
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=9.5,
        family="monospace",
        bbox=dict(boxstyle="round,pad=0.35", fc="white", alpha=0.95),
    )

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
