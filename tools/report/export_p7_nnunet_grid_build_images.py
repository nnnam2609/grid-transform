from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import matplotlib.patheffects as pe
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from tools.report.slide_p7_nnunet_config import load_slide_p7_nnunet_config, nested_dict, xy_tuple
from grid_transform.config import DEFAULT_OUTPUT_DIR, DEFAULT_VTLN_DIR
from grid_transform.image_utils import as_grayscale_uint8
from grid_transform.io import load_frame_npy, load_frame_vtln
from grid_transform.vt_grid import build_grid
from tools.report.common import NNUNET_CASE_DIR_DEFAULT, subset_common_contours

from tools.report.export_p7_nnunet_common_label_images import (
    COMMON_CONTOUR_COLORS,
    NNUNET_FRAME_DEFAULT,
    P7_BASENAME,
)

GRID_H_COLOR = "#35c9ff"
GRID_V_COLOR = "#ffd166"
INTERP_COLOR = "#2dd4bf"
POINT_STYLE = {
    "I1": ((-18.0, -8.0), "incisior-hard-palate"),
    "I2": ((-12.0, -10.0), "incisior-hard-palate"),
    "I3": ((0.0, -12.0), "incisior-hard-palate"),
    "I4": ((8.0, -10.0), "incisior-hard-palate"),
    "I5": ((10.0, -2.0), "incisior-hard-palate"),
    "I6": ((10.0, 6.0), "soft-palate-midline"),
    "I7": ((10.0, -6.0), "soft-palate-midline"),
    "P1": ((10.0, 10.0), "pharynx"),
    "M1": ((10.0, -8.0), "mandible-incisior"),
    "L6": ((10.0, 10.0), "mandible-incisior"),
    "L3": ((-12.0, 2.0), None),
    "L4": ((-12.0, 2.0), None),
    "L5": ((-12.0, 2.0), None),
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export step-by-step grid construction visuals for P7 and the nnUNet target."
    )
    parser.add_argument("--vtln-dir", type=Path, default=DEFAULT_VTLN_DIR, help="Folder containing canonical VTLN PNG/ZIP pairs.")
    parser.add_argument("--p7-basename", default=P7_BASENAME, help="Canonical VTLN basename for P7.")
    parser.add_argument(
        "--nnunet-case-dir",
        type=Path,
        default=NNUNET_CASE_DIR_DEFAULT,
        help="nnUNet case folder containing PNG_MR and contours.",
    )
    parser.add_argument("--nnunet-frame", type=int, default=NNUNET_FRAME_DEFAULT, help="nnUNet frame number.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR / "slide" / "images",
        help="Output folder for the generated PNG images.",
    )
    return parser.parse_args(argv)


def draw_base(ax: plt.Axes, image, contours: dict[str, np.ndarray], labels: list[str]) -> None:
    gray = as_grayscale_uint8(image)
    height, width = gray.shape[:2]
    ax.imshow(gray, cmap="gray", vmin=0, vmax=255, extent=(0, width, height, 0))
    for label in labels:
        pts = np.asarray(contours[label], dtype=float)
        ax.plot(
            pts[:, 0],
            pts[:, 1],
            color=COMMON_CONTOUR_COLORS[label],
            lw=3.0,
            solid_capstyle="round",
            zorder=2,
        )
    ax.set_xlim(0, width)
    ax.set_ylim(height, 0)
    ax.axis("off")


def point_text(ax: plt.Axes, x: float, y: float, text: str, color: str) -> None:
    slide_cfg = load_slide_p7_nnunet_config()
    style_cfg = slide_cfg["grid_build_landmarks"]
    artist = ax.text(
        x,
        y,
        text,
        color=color,
        fontsize=float(style_cfg.get("font_size", 9.2)),
        fontweight="bold",
        ha="center",
        va="center",
        zorder=8,
    )
    artist.set_path_effects([pe.withStroke(linewidth=2.6, foreground="black")])


def color_for_point(name: str, fallback: str = "#ffffff") -> str:
    style = POINT_STYLE.get(name)
    if style is None or style[1] is None:
        return fallback
    return COMMON_CONTOUR_COLORS.get(style[1], fallback)


def draw_point(ax: plt.Axes, name: str, point: np.ndarray, *, marker: str = "o", size: float = 62.0, edgecolor: str = "white") -> None:
    slide_cfg = load_slide_p7_nnunet_config()
    style_cfg = slide_cfg["grid_build_landmarks"]
    point_sizes = nested_dict(slide_cfg, "grid_build_landmarks", "point_sizes")
    point_offsets = nested_dict(slide_cfg, "grid_build_landmarks", "point_offsets")
    color = color_for_point(name, INTERP_COLOR)
    configured_size = float(point_sizes.get(name, point_sizes.get("default", size)))
    ax.scatter(
        [point[0]],
        [point[1]],
        s=configured_size,
        marker=marker,
        c=[color],
        edgecolors=edgecolor,
        linewidths=float(style_cfg.get("marker_edge_width", 1.2)),
        zorder=7,
    )
    configured_offset = xy_tuple(
        point_offsets.get(name),
        default=xy_tuple(style_cfg.get("default_offset"), default=(10.0, -10.0)),
    )
    point_text(ax, point[0] + configured_offset[0], point[1] + configured_offset[1], name, color)


def save_landmark_construction_figure(
    image,
    contours: dict[str, np.ndarray],
    common_labels: list[str],
    grid,
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(4.8, 4.8), dpi=100)
    fig.patch.set_facecolor("black")
    ax.set_facecolor("black")
    draw_base(ax, image, contours, common_labels)

    if grid.cervical_centers:
        c_points = np.array([grid.cervical_centers[f"c{i}"] for i in range(1, 7)], dtype=float)
        ax.plot(c_points[:, 0], c_points[:, 1], "--", color="#8ecae6", lw=1.6, alpha=0.85, zorder=3)

    if grid.M1 is not None and grid.L6 is not None:
        ax.plot(
            [grid.M1[0], grid.L6[0]],
            [grid.M1[1], grid.L6[1]],
            "--",
            color=COMMON_CONTOUR_COLORS["mandible-incisior"],
            lw=1.6,
            alpha=0.82,
            zorder=3,
        )

    if grid.I_points:
        palate_sequence = [grid.I_points[name] for name in ("I1", "I2", "I3", "I4", "I5") if name in grid.I_points]
        if len(palate_sequence) >= 2:
            palate_arr = np.asarray(palate_sequence, dtype=float)
            ax.plot(
                palate_arr[:, 0],
                palate_arr[:, 1],
                "--",
                color=COMMON_CONTOUR_COLORS["incisior-hard-palate"],
                lw=1.5,
                alpha=0.8,
                zorder=3,
            )
        soft_sequence = [grid.I_points[name] for name in ("I6", "I7") if name in grid.I_points]
        if len(soft_sequence) >= 2:
            soft_arr = np.asarray(soft_sequence, dtype=float)
            ax.plot(
                soft_arr[:, 0],
                soft_arr[:, 1],
                "--",
                color=COMMON_CONTOUR_COLORS["soft-palate-midline"],
                lw=1.5,
                alpha=0.8,
                zorder=3,
            )
        if "I5" in grid.I_points and "c1" in (grid.cervical_centers or {}):
            guide = [np.asarray(grid.I_points["I5"], dtype=float)]
            if grid.P1_point is not None:
                guide.append(np.asarray(grid.P1_point, dtype=float))
            guide.append(np.asarray(grid.cervical_centers["c1"], dtype=float))
            guide_arr = np.asarray(guide, dtype=float)
            ax.plot(guide_arr[:, 0], guide_arr[:, 1], "--", color=GRID_H_COLOR, lw=1.5, alpha=0.9, zorder=4)

    if grid.I_points:
        for name in ("I1", "I2", "I3", "I4", "I5", "I6", "I7"):
            if name in grid.I_points:
                draw_point(ax, name, np.asarray(grid.I_points[name], dtype=float))

    if grid.cervical_centers:
        for idx in range(1, 7):
            name = f"C{idx}"
            point = np.asarray(grid.cervical_centers[f"c{idx}"], dtype=float)
            draw_point(ax, name, point)

    if grid.P1_point is not None:
        draw_point(ax, "P1", np.asarray(grid.P1_point, dtype=float), marker="X", size=78.0)
    if grid.M1 is not None:
        draw_point(ax, "M1", np.asarray(grid.M1, dtype=float))
    if grid.L6 is not None:
        draw_point(ax, "L6", np.asarray(grid.L6, dtype=float), marker="D", size=72.0)

    if grid.left_pts is not None:
        for idx in range(2, grid.n_horiz - 1):
            name = f"L{idx + 1}"
            point = np.asarray(grid.left_pts[idx], dtype=float)
            draw_point(ax, name, point, marker="s", size=54.0, edgecolor="white")

    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=100, pad_inches=0, facecolor=fig.get_facecolor())
    plt.close(fig)


def save_grid_overlay_figure(
    image,
    contours: dict[str, np.ndarray],
    common_labels: list[str],
    grid,
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(4.8, 4.8), dpi=100)
    fig.patch.set_facecolor("black")
    ax.set_facecolor("black")
    draw_base(ax, image, contours, common_labels)

    for row_idx, line in enumerate(grid.horiz_lines, start=1):
        lw = 2.6 if row_idx in {1, len(grid.horiz_lines)} else 1.5
        ax.plot(line[:, 0], line[:, 1], color=GRID_H_COLOR, lw=lw, alpha=0.95, zorder=4)
        left_pt = np.asarray(line[0], dtype=float)
        point_text(ax, left_pt[0] + 18.0, left_pt[1] - 10.0, f"H{row_idx}", GRID_H_COLOR)

    for col_idx, line in enumerate(grid.vert_lines, start=1):
        lw = 2.2 if col_idx in {1, len(grid.vert_lines)} else 1.15
        ax.plot(line[:, 0], line[:, 1], color=GRID_V_COLOR, lw=lw, alpha=0.92, zorder=4)
        top_pt = np.asarray(line[0], dtype=float)
        label_y = min(max(top_pt[1] + 14.0, 18.0), 36.0)
        point_text(ax, top_pt[0], label_y, f"V{col_idx}", GRID_V_COLOR)

    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=100, pad_inches=0, facecolor=fig.get_facecolor())
    plt.close(fig)


def export_case(
    *,
    image,
    contours: dict[str, np.ndarray],
    common_labels: list[str],
    frame_number: int,
    prefix: str,
    output_dir: Path,
) -> tuple[Path, Path]:
    focused_contours = {label: np.asarray(contours[label], dtype=float).copy() for label in common_labels}
    grid = build_grid(image, focused_contours, frame_number=frame_number)

    landmark_output = output_dir / f"{prefix}_grid_landmarks.png"
    grid_output = output_dir / f"{prefix}_grid_overlay.png"
    save_landmark_construction_figure(image, focused_contours, common_labels, grid, landmark_output)
    save_grid_overlay_figure(image, focused_contours, common_labels, grid, grid_output)
    return landmark_output, grid_output


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    p7_image, p7_contours = load_frame_vtln(args.p7_basename, args.vtln_dir)
    nnunet_image, nnunet_contours = load_frame_npy(args.nnunet_frame, args.nnunet_case_dir, args.nnunet_case_dir / "contours")
    common_labels = subset_common_contours(p7_contours, nnunet_contours)
    if not common_labels:
        raise ValueError("No shared contours available for the requested P7/nnUNet pair.")

    p7_landmarks, p7_grid = export_case(
        image=p7_image,
        contours=p7_contours,
        common_labels=common_labels,
        frame_number=int(args.p7_basename.split("_")[-1][1:]),
        prefix="p7",
        output_dir=args.output_dir,
    )
    nn_landmarks, nn_grid = export_case(
        image=nnunet_image,
        contours=nnunet_contours,
        common_labels=common_labels,
        frame_number=args.nnunet_frame,
        prefix="nnunet",
        output_dir=args.output_dir,
    )

    print(f"Shared contours ({len(common_labels)}): {', '.join(common_labels)}")
    print(f"Saved P7 landmarks: {p7_landmarks}")
    print(f"Saved P7 grid: {p7_grid}")
    print(f"Saved nnUNet landmarks: {nn_landmarks}")
    print(f"Saved nnUNet grid: {nn_grid}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
