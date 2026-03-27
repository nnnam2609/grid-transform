from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from grid_transform.articulators import (
    compute_articulator_errors,
    parse_articulators,
    save_comparison_figure as save_articulator_transfer_figure,
)
from grid_transform.config import (
    DEFAULT_VTNL_DIR,
    PROJECT_DIR,
    REPORT_FIGURES_DIR,
    REPORT_GENERATED_DIR,
    TONGUE_COLOR,
    VT_SEG_CONTOURS_ROOT,
    VT_SEG_DATA_ROOT,
)
from grid_transform.io import load_frame_npy, load_frame_vtnl
from grid_transform.transfer import build_two_step_transform, smooth_transformed_contours, transform_contours
from grid_transform.vt import build_grid, visualize_grid
from grid_transform.warp import (
    save_comparison_figure as save_warp_figure,
    warp_image_to_target_space,
)
from grid_transform.apps.method4_transform import (
    SOURCE_COLOR,
    TARGET_COLOR,
    apply_tps,
    apply_transform,
    build_step1_anchors,
    build_step2_controls,
    compute_metrics,
    draw_grid_line_labels,
    draw_grid_single_color,
    draw_named_points,
    estimate_affine,
    extract_true_landmarks,
    fit_tps,
    format_target_frame,
    map_landmarks,
    rms_point_error,
    smooth_transformed_grid,
)



DEFAULT_CASE = "2008-003^01-1791/test"
DEFAULT_TARGET_SPEAKER = "1640_s10_0829"
DEFAULT_SOURCE_FRAME = 143020
DEFAULT_FIGURE_DPI = 220
VERT_COLOR = "#118ab2"
MANDIBLE_COLOR = "#f4a261"
MAIN_CONTOUR_STYLE = {
    "incisior-hard-palate": {"color": "#ef476f", "label": "incisior-hard-palate", "frac": 0.18, "offset": (-55, -28)},
    "soft-palate-midline": {"color": "#8338ec", "label": "soft-palate-midline", "frac": 0.25, "offset": (14, -28)},
    "soft-palate": {"color": "#ff7f50", "label": "soft-palate", "frac": 0.55, "offset": (24, 6)},
    "mandible-incisior": {"color": MANDIBLE_COLOR, "label": "mandible-incisior", "frac": 0.20, "offset": (-72, 20)},
    "pharynx": {"color": "#118ab2", "label": "pharynx", "frac": 0.42, "offset": (22, 10)},
    "tongue": {"color": TONGUE_COLOR, "label": "tongue", "frac": 0.50, "offset": (-14, 24)},
}


@dataclass
class ReportArtifacts:
    target_speaker: str
    source_frame: int
    target_image: object
    source_image: object
    target_contours: dict
    source_contours: dict
    target_grid: object
    source_grid: object
    step1_anchor_labels: list[str]
    step2_ctrl_labels: list[str]
    step1_anchor_tgt: np.ndarray
    step1_anchor_hat: np.ndarray
    step2_ctrl_tgt: np.ndarray
    step2_ctrl_src: np.ndarray
    step1_horiz: list[np.ndarray]
    step1_vert: list[np.ndarray]
    final_horiz: list[np.ndarray]
    final_vert: list[np.ndarray]
    apply_two_step: object
    inverse_two_step: object
    final_metrics: dict
    step1_anchor_rms: float
    moved_source_contours: dict
    articulator_errors: dict
    shared_articulators: list[str]
    warped_source_arr: np.ndarray
    warped_mask_arr: np.ndarray


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate dedicated figures and LaTeX values for the grid transformation report.")
    parser.add_argument("--target-speaker", default=DEFAULT_TARGET_SPEAKER, help="VTNL target speaker/image name.")
    parser.add_argument("--source-frame", type=int, default=DEFAULT_SOURCE_FRAME, help="nnUNet source frame number.")
    parser.add_argument("--case", default=DEFAULT_CASE, help="nnUNet case relative path.")
    parser.add_argument("--vtnl-dir", type=Path, default=DEFAULT_VTNL_DIR, help="Folder containing VTNL images and ROI zip files.")
    parser.add_argument("--outdir", type=Path, default=REPORT_FIGURES_DIR, help="Where to save the generated report figures.")
    parser.add_argument("--generated-dir", type=Path, default=REPORT_GENERATED_DIR, help="Where to save generated LaTeX include files.")
    return parser


def sample_polyline_point(points: np.ndarray, fraction: float) -> np.ndarray:
    points = np.asarray(points, dtype=float)
    idx = int(np.clip(round(fraction * (len(points) - 1)), 0, len(points) - 1))
    return points[idx]


def label_callout(ax, point: np.ndarray, text: str, color: str, dx: float, dy: float) -> None:
    point = np.asarray(point, dtype=float)
    label_xy = point + np.array([dx, dy], dtype=float)
    ax.annotate(
        text,
        xy=point,
        xytext=label_xy,
        color=color,
        fontsize=9.5,
        fontweight="bold",
        ha="left" if dx >= 0 else "right",
        va="center",
        arrowprops=dict(arrowstyle="->", lw=1.2, color=color, alpha=0.85),
        bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.92),
        zorder=20,
    )


def tex_escape(text: str) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "_": r"\_",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(ch, ch) for ch in text)


def tex_code_list(items: list[str]) -> str:
    return r", \allowbreak ".join(rf"\texttt{{{tex_escape(item)}}}" for item in items)


def save_figure(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=DEFAULT_FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)


def compute_report_artifacts(args: argparse.Namespace) -> ReportArtifacts:
    target_image, target_contours = load_frame_vtnl(args.target_speaker, args.vtnl_dir)
    source_image, source_contours = load_frame_npy(
        args.source_frame,
        VT_SEG_DATA_ROOT / args.case,
        VT_SEG_CONTOURS_ROOT / args.case,
    )

    target_grid = build_grid(target_image, target_contours, n_vert=9, n_points=250, frame_number=0)
    source_grid = build_grid(source_image, source_contours, n_vert=9, n_points=250, frame_number=args.source_frame)

    lm_tgt = extract_true_landmarks(target_grid)
    lm_src = extract_true_landmarks(source_grid)

    step1_anchor_src, step1_anchor_tgt, step1_anchor_labels = build_step1_anchors(lm_src, lm_tgt)
    step1_affine = estimate_affine(step1_anchor_src, step1_anchor_tgt)
    step1_lm = map_landmarks(lambda pts: apply_transform(step1_affine, pts), lm_src)
    step1_horiz_raw = [apply_transform(step1_affine, h) for h in source_grid.horiz_lines]
    step1_horiz, step1_vert = smooth_transformed_grid(step1_horiz_raw, step1_lm, source_grid.n_vert, top_axis_passes=4)
    step1_anchor_hat = apply_transform(step1_affine, step1_anchor_src)
    step1_anchor_rms = rms_point_error(step1_anchor_hat, step1_anchor_tgt)

    step2_ctrl_src, step2_ctrl_tgt, step2_ctrl_labels = build_step2_controls(step1_lm, lm_tgt)
    step2_tps = fit_tps(step2_ctrl_src, step2_ctrl_tgt, smoothing=0.0)

    def apply_two_step(pts):
        affine_pts = apply_transform(step1_affine, pts)
        return apply_tps(step2_tps, affine_pts)

    mapped_final = map_landmarks(apply_two_step, lm_src)
    final_metrics = compute_metrics(mapped_final, lm_tgt)
    final_horiz_raw = [apply_two_step(h) for h in source_grid.horiz_lines]
    final_horiz, final_vert = smooth_transformed_grid(final_horiz_raw, mapped_final, source_grid.n_vert, top_axis_passes=4)

    shared_articulators = parse_articulators(None, target_contours, source_contours)
    moved_source_contours = transform_contours(source_contours, apply_two_step, shared_articulators)
    moved_source_contours = smooth_transformed_contours(moved_source_contours)
    articulator_errors = compute_articulator_errors(moved_source_contours, target_contours)

    inverse_transform = build_two_step_transform(target_grid, source_grid)
    warped_source_arr, warped_mask_arr = warp_image_to_target_space(
        source_image,
        np.asarray(target_image).shape,
        inverse_transform["apply_two_step"],
    )

    return ReportArtifacts(
        target_speaker=args.target_speaker,
        source_frame=args.source_frame,
        target_image=target_image,
        source_image=source_image,
        target_contours=target_contours,
        source_contours=source_contours,
        target_grid=target_grid,
        source_grid=source_grid,
        step1_anchor_labels=step1_anchor_labels,
        step2_ctrl_labels=step2_ctrl_labels,
        step1_anchor_tgt=step1_anchor_tgt,
        step1_anchor_hat=step1_anchor_hat,
        step2_ctrl_tgt=step2_ctrl_tgt,
        step2_ctrl_src=step2_ctrl_src,
        step1_horiz=step1_horiz,
        step1_vert=step1_vert,
        final_horiz=final_horiz,
        final_vert=final_vert,
        apply_two_step=apply_two_step,
        inverse_two_step=inverse_transform["apply_two_step"],
        final_metrics=final_metrics,
        step1_anchor_rms=step1_anchor_rms,
        moved_source_contours=moved_source_contours,
        articulator_errors=articulator_errors,
        shared_articulators=shared_articulators,
        warped_source_arr=warped_source_arr,
        warped_mask_arr=warped_mask_arr,
    )


def save_raw_speakers_figure(artifacts: ReportArtifacts, output_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 6.5))
    format_target_frame(axes[0], artifacts.target_image, f"Target speaker: {artifacts.target_speaker}")
    format_target_frame(axes[1], artifacts.source_image, f"Source speaker: nnUNet frame {artifacts.source_frame}")
    save_figure(fig, output_path)


def save_landmark_notation_figure(artifacts: ReportArtifacts, output_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(18, 8.5))

    ax = axes[0]
    format_target_frame(ax, artifacts.target_image, "Main contours used to build the grid")
    for name, style in MAIN_CONTOUR_STYLE.items():
        if name not in artifacts.target_contours:
            continue
        contour = np.asarray(artifacts.target_contours[name], dtype=float)
        ax.plot(contour[:, 0], contour[:, 1], color=style["color"], lw=2.4, alpha=0.95)
        point = sample_polyline_point(contour, style["frac"])
        label_callout(ax, point, style["label"], style["color"], *style["offset"])

    c_centers = np.array([artifacts.target_grid.cervical_centers[f"c{i}"] for i in range(1, 7)], dtype=float)
    ax.plot(c_centers[:, 0], c_centers[:, 1], "o-", color=VERT_COLOR, lw=2.0, ms=5, zorder=15)
    label_callout(ax, np.mean(c_centers, axis=0), "c1-c6 vertebrae", VERT_COLOR, 38, -12)

    ax = axes[1]
    style = {
        "horiz_color": TARGET_COLOR,
        "vert_color": "#ffd166",
        "vt_color": "#ef476f",
        "spine_color": VERT_COLOR,
        "guide_color": "#ff006e",
        "i_color": "#ef476f",
        "c_color": VERT_COLOR,
        "p1_color": "#ff006e",
        "m1_color": MANDIBLE_COLOR,
        "l6_color": "#06d6a0",
        "interp_color": TARGET_COLOR,
        "mandible_color": MANDIBLE_COLOR,
        "tongue_color": TONGUE_COLOR,
    }
    visualize_grid(
        artifacts.target_grid,
        ax=ax,
        figsize=(8, 8),
        show_contours=False,
        show_landmarks=True,
        show_labels=True,
        style=style,
    )
    if ax.legend_ is not None:
        ax.legend_.remove()
    ax.set_title("Landmarks and grid notation", fontsize=13, fontweight="bold")
    draw_grid_line_labels(ax, artifacts.target_grid.horiz_lines, artifacts.target_grid.vert_lines, TARGET_COLOR, alpha=0.84)
    save_figure(fig, output_path)


def save_raw_with_grids_figure(artifacts: ReportArtifacts, output_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    ax = axes[0]
    format_target_frame(ax, artifacts.target_image, "Target speaker with anatomical grid")
    draw_grid_single_color(ax, artifacts.target_grid.horiz_lines, artifacts.target_grid.vert_lines, TARGET_COLOR, alpha=0.92, lw_major=3.0, lw_minor=1.1)
    if "tongue" in artifacts.target_contours:
        target_tongue = np.asarray(artifacts.target_contours["tongue"], dtype=float)
        ax.plot(target_tongue[:, 0], target_tongue[:, 1], color=TONGUE_COLOR, lw=2.0, alpha=0.95)

    ax = axes[1]
    format_target_frame(ax, artifacts.source_image, "Source speaker with anatomical grid")
    draw_grid_single_color(ax, artifacts.source_grid.horiz_lines, artifacts.source_grid.vert_lines, SOURCE_COLOR, alpha=0.90, lw_major=3.0, lw_minor=1.1)
    if "tongue" in artifacts.source_contours:
        source_tongue = np.asarray(artifacts.source_contours["tongue"], dtype=float)
        ax.plot(source_tongue[:, 0], source_tongue[:, 1], color=TONGUE_COLOR, lw=2.0, alpha=0.95)

    save_figure(fig, output_path)


def save_step1_affine_figure(artifacts: ReportArtifacts, output_path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(21, 7.5))

    ax = axes[0]
    format_target_frame(ax, artifacts.target_image, "Step 0: source grid before alignment")
    draw_grid_single_color(ax, artifacts.target_grid.horiz_lines, artifacts.target_grid.vert_lines, TARGET_COLOR, alpha=0.95, lw_major=3.0, lw_minor=1.1)
    draw_grid_single_color(ax, artifacts.source_grid.horiz_lines, artifacts.source_grid.vert_lines, SOURCE_COLOR, alpha=0.65, lw_major=2.5, lw_minor=0.8)

    ax = axes[1]
    format_target_frame(ax, artifacts.target_image, "Step 1a: affine fit on axis anchors")
    draw_named_points(ax, artifacts.step1_anchor_tgt, artifacts.step1_anchor_labels, TARGET_COLOR, marker="o")
    draw_named_points(ax, artifacts.step1_anchor_hat, artifacts.step1_anchor_labels, SOURCE_COLOR, marker="s", dx=6, dy=10)
    for src_pt, tgt_pt in zip(artifacts.step1_anchor_hat, artifacts.step1_anchor_tgt):
        ax.annotate("", xy=tgt_pt, xytext=src_pt, arrowprops=dict(arrowstyle="->", color="black", lw=1.0, alpha=0.7))
    ax.text(
        0.02,
        0.02,
        f"{len(artifacts.step1_anchor_labels)} anchors\nAnchor RMS = {artifacts.step1_anchor_rms:.2f} px",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=10.5,
        bbox=dict(boxstyle="round,pad=0.35", fc="white", alpha=0.94),
    )

    ax = axes[2]
    format_target_frame(ax, artifacts.target_image, "Step 1b: source grid after affine")
    draw_grid_single_color(ax, artifacts.target_grid.horiz_lines, artifacts.target_grid.vert_lines, TARGET_COLOR, alpha=0.95, lw_major=3.0, lw_minor=1.1)
    draw_grid_single_color(ax, artifacts.step1_horiz, artifacts.step1_vert, SOURCE_COLOR, alpha=0.88, lw_major=2.6, lw_minor=0.9)
    ax.legend(
        [
            plt.Line2D([0], [0], color=TARGET_COLOR, lw=3),
            plt.Line2D([0], [0], color=SOURCE_COLOR, lw=3),
        ],
        ["Target grid", "Affine-mapped source grid"],
        fontsize=10,
        loc="upper right",
        framealpha=0.92,
    )

    save_figure(fig, output_path)


def save_step2_tps_figure(artifacts: ReportArtifacts, output_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(16, 7.5))

    ax = axes[0]
    format_target_frame(ax, artifacts.target_image, "Step 2a: TPS controls after Step 1")
    draw_named_points(ax, artifacts.step2_ctrl_tgt, artifacts.step2_ctrl_labels, TARGET_COLOR, marker="o")
    draw_named_points(ax, artifacts.step2_ctrl_src, artifacts.step2_ctrl_labels, SOURCE_COLOR, marker="s", dx=6, dy=10)
    for src_pt, tgt_pt in zip(artifacts.step2_ctrl_src, artifacts.step2_ctrl_tgt):
        ax.annotate("", xy=tgt_pt, xytext=src_pt, arrowprops=dict(arrowstyle="->", color="black", lw=1.0, alpha=0.7))
    ax.text(
        0.02,
        0.02,
        "TPS starts from the affine result.\nEach source control point is bent onto its target partner.",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=10,
        bbox=dict(boxstyle="round,pad=0.35", fc="white", alpha=0.94),
    )

    ax = axes[1]
    format_target_frame(ax, artifacts.target_image, "Step 2b: local bending from affine to final")
    draw_grid_single_color(ax, artifacts.target_grid.horiz_lines, artifacts.target_grid.vert_lines, TARGET_COLOR, alpha=0.95, lw_major=3.0, lw_minor=1.1)
    draw_grid_single_color(ax, artifacts.step1_horiz, artifacts.step1_vert, "#8d99ae", alpha=0.65, lw_major=2.4, lw_minor=0.7)
    draw_grid_single_color(ax, artifacts.final_horiz, artifacts.final_vert, SOURCE_COLOR, alpha=0.90, lw_major=2.8, lw_minor=0.9)
    ax.legend(
        [
            plt.Line2D([0], [0], color=TARGET_COLOR, lw=3),
            plt.Line2D([0], [0], color="#8d99ae", lw=3),
            plt.Line2D([0], [0], color=SOURCE_COLOR, lw=3),
        ],
        ["Target grid", "After Step 1 affine", "After Step 2 TPS"],
        fontsize=10,
        loc="upper right",
        framealpha=0.92,
    )

    save_figure(fig, output_path)


def save_final_alignment_figure(artifacts: ReportArtifacts, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(11.5, 11))
    format_target_frame(ax, artifacts.target_image, "Final source-to-target alignment")
    draw_grid_single_color(ax, artifacts.target_grid.horiz_lines, artifacts.target_grid.vert_lines, TARGET_COLOR, alpha=0.95, lw_major=3.0, lw_minor=1.1)
    draw_grid_single_color(ax, artifacts.final_horiz, artifacts.final_vert, SOURCE_COLOR, alpha=0.88, lw_major=2.8, lw_minor=0.9)
    draw_grid_line_labels(ax, artifacts.target_grid.horiz_lines, artifacts.target_grid.vert_lines, TARGET_COLOR, alpha=0.82)

    if "tongue" in artifacts.target_contours:
        target_tongue = np.asarray(artifacts.target_contours["tongue"], dtype=float)
        ax.plot(target_tongue[:, 0], target_tongue[:, 1], color=TONGUE_COLOR, lw=2.4, alpha=0.95, label="Target tongue")
    if "tongue" in artifacts.moved_source_contours:
        moved_tongue = np.asarray(artifacts.moved_source_contours["tongue"], dtype=float)
        ax.plot(moved_tongue[:, 0], moved_tongue[:, 1], "--", color=TONGUE_COLOR, lw=2.0, alpha=0.95, label="Mapped source tongue")

    metrics = artifacts.final_metrics
    text_lines = [
        f"Step 1 anchor RMS: {artifacts.step1_anchor_rms:.2f} px",
        f"Final spine RMS: {metrics['spine_rms']:.2f} px",
        f"Final horiz-axis RMS: {metrics['horiz_axis_rms']:.2f} px" if metrics["horiz_axis_rms"] is not None else "Final horiz-axis RMS: --",
        f"Final vert-axis RMS: {metrics['vert_axis_rms']:.2f} px" if metrics["vert_axis_rms"] is not None else "Final vert-axis RMS: --",
    ]
    if "tongue" in artifacts.articulator_errors:
        text_lines.append(f"Tongue RMS after transfer: {artifacts.articulator_errors['tongue']:.2f} px")
    ax.text(
        0.02,
        0.02,
        "\n".join(text_lines),
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=10,
        bbox=dict(boxstyle="round,pad=0.35", fc="white", alpha=0.94),
    )

    ax.legend(loc="upper right", fontsize=10, framealpha=0.92)
    save_figure(fig, output_path)


def save_tongue_transfer_figure(artifacts: ReportArtifacts, output_path: Path) -> None:
    save_articulator_transfer_figure(
        artifacts.target_image,
        artifacts.source_image,
        artifacts.target_contours,
        artifacts.source_contours,
        artifacts.moved_source_contours,
        artifacts.shared_articulators,
        artifacts.articulator_errors,
        output_path,
    )


def save_full_image_warp_figure(artifacts: ReportArtifacts, output_path: Path) -> None:
    save_warp_figure(
        artifacts.target_image,
        artifacts.source_image,
        artifacts.warped_source_arr,
        artifacts.target_contours,
        artifacts.source_contours,
        artifacts.moved_source_contours,
        artifacts.shared_articulators,
        output_path,
    )


def write_report_values(args: argparse.Namespace, artifacts: ReportArtifacts, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    tongue_rms = artifacts.articulator_errors.get("tongue")
    try:
        vtnl_display = args.vtnl_dir.relative_to(PROJECT_DIR).as_posix()
    except ValueError:
        vtnl_display = args.vtnl_dir.as_posix()
    lines = [
        rf"\newcommand{{\TargetSpeakerName}}{{\texttt{{{tex_escape(args.target_speaker)}}}}}",
        rf"\newcommand{{\SourceFrameNumber}}{{{args.source_frame}}}",
        rf"\newcommand{{\VTNLFolderName}}{{\path{{{vtnl_display}}}}}",
        rf"\newcommand{{\StepOneAnchorCount}}{{{len(artifacts.step1_anchor_labels)}}}",
        rf"\newcommand{{\StepTwoControlCount}}{{{len(artifacts.step2_ctrl_labels)}}}",
        rf"\newcommand{{\StepOneAnchorList}}{{{tex_code_list(artifacts.step1_anchor_labels)}}}",
        rf"\newcommand{{\StepTwoControlList}}{{{tex_code_list(artifacts.step2_ctrl_labels)}}}",
        rf"\newcommand{{\SharedArticulatorList}}{{{tex_code_list(artifacts.shared_articulators)}}}",
        rf"\newcommand{{\StepOneAnchorRMS}}{{{artifacts.step1_anchor_rms:.2f}}}",
        rf"\newcommand{{\FinalSpineRMS}}{{{artifacts.final_metrics['spine_rms']:.2f}}}",
        rf"\newcommand{{\FinalHorizRMS}}{{{artifacts.final_metrics['horiz_axis_rms']:.2f}}}",
        rf"\newcommand{{\FinalVertRMS}}{{{artifacts.final_metrics['vert_axis_rms']:.2f}}}",
        rf"\newcommand{{\TongueRMS}}{{{tongue_rms:.2f}}}" if tongue_rms is not None else r"\newcommand{\TongueRMS}{--}",
    ]
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    artifacts = compute_report_artifacts(args)

    args.outdir.mkdir(parents=True, exist_ok=True)
    args.generated_dir.mkdir(parents=True, exist_ok=True)

    save_raw_speakers_figure(artifacts, args.outdir / "raw_speakers.png")
    save_landmark_notation_figure(artifacts, args.outdir / "landmark_notation.png")
    save_raw_with_grids_figure(artifacts, args.outdir / "raw_with_grids.png")
    save_step1_affine_figure(artifacts, args.outdir / "step1_affine.png")
    save_step2_tps_figure(artifacts, args.outdir / "step2_tps.png")
    save_final_alignment_figure(artifacts, args.outdir / "final_alignment.png")
    save_tongue_transfer_figure(artifacts, args.outdir / "tongue_transfer.png")
    save_full_image_warp_figure(artifacts, args.outdir / "full_image_warp.png")
    write_report_values(args, artifacts, args.generated_dir / "report_values.tex")

    print(f"Generated figures in: {args.outdir}")
    print(f"Generated LaTeX values in: {args.generated_dir / 'report_values.tex'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
