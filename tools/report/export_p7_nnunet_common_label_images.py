from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe

from grid_transform.config import DEFAULT_OUTPUT_DIR, DEFAULT_VTLN_DIR
from grid_transform.image_utils import as_grayscale_uint8
from grid_transform.io import load_frame_npy, load_frame_vtln
from tools.report.common import NNUNET_CASE_DIR_DEFAULT, NNUNET_FRAME_DEFAULT, P7_BASENAME


COMMON_CONTOUR_COLORS = {
    "c1": "#457b9d",
    "c2": "#4d7ea8",
    "c3": "#5c96bc",
    "c4": "#7aa6c2",
    "c5": "#90b8d8",
    "c6": "#b8d8f2",
    "incisior-hard-palate": "#ef476f",
    "mandible-incisior": "#f4a261",
    "pharynx": "#118ab2",
    "soft-palate": "#fb8500",
    "soft-palate-midline": "#8338ec",
}
DISPLAY_NAMES = {
    "c1": "C1",
    "c2": "C2",
    "c3": "C3",
    "c4": "C4",
    "c5": "C5",
    "c6": "C6",
    "incisior-hard-palate": "Incisor hard palate",
    "mandible-incisior": "Mandible incisor",
    "pharynx": "Pharynx",
    "soft-palate": "Soft palate",
    "soft-palate-midline": "Soft palate\nmidline",
}
RIGHT_SIDE_LABELS = {"c1", "c2", "c3", "c4", "c5", "c6", "pharynx"}
LEFT_SIDE_LABELS = {"incisior-hard-palate", "mandible-incisior", "soft-palate", "soft-palate-midline"}
MANUAL_TEXT_POSITIONS = {
    "soft-palate-midline": (145.0, 430.0),
    "soft-palate": (90.0, 390.0),
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export two slide-ready labeled images for P7 and the nnUNet target using only shared anatomical elements."
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
        help="Output folder for the annotated PNG images.",
    )
    return parser.parse_args(argv)


def contour_anchor(points: np.ndarray) -> np.ndarray:
    pts = np.asarray(points, dtype=float)
    return np.mean(pts, axis=0)


def split_labels_by_side(labels: list[str], anchors: dict[str, np.ndarray]) -> tuple[list[str], list[str]]:
    left = [label for label in labels if label in LEFT_SIDE_LABELS]
    right = [label for label in labels if label in RIGHT_SIDE_LABELS]
    remaining = [label for label in labels if label not in LEFT_SIDE_LABELS and label not in RIGHT_SIDE_LABELS]
    for label in remaining:
        if anchors[label][0] < 240:
            left.append(label)
        else:
            right.append(label)
    left.sort(key=lambda label: anchors[label][1])
    right.sort(key=lambda label: anchors[label][1])
    return left, right


def side_positions(labels: list[str], y_min: float, y_max: float) -> dict[str, float]:
    if not labels:
        return {}
    if len(labels) == 1:
        return {labels[0]: (y_min + y_max) / 2.0}
    values = np.linspace(y_min, y_max, len(labels))
    return {label: float(y) for label, y in zip(labels, values)}


def draw_labeled_figure(
    image,
    contours: dict[str, np.ndarray],
    labels: list[str],
    *,
    output_path: Path,
) -> None:
    gray = as_grayscale_uint8(image)
    height, width = gray.shape[:2]

    anchors = {label: contour_anchor(contours[label]) for label in labels}
    left_labels, right_labels = split_labels_by_side(labels, anchors)
    y_min = 38
    y_max = height - 38
    left_y = side_positions(left_labels, y_min, y_max)
    right_y = side_positions(right_labels, y_min, y_max)

    fig_w = width / 100.0
    fig_h = height / 100.0
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=100)
    fig.patch.set_facecolor("black")
    ax.set_facecolor("black")

    ax.imshow(gray, cmap="gray", vmin=0, vmax=255, extent=(0, width, height, 0))

    for label in labels:
        pts = np.asarray(contours[label], dtype=float)
        ax.plot(
            pts[:, 0],
            pts[:, 1],
            color=COMMON_CONTOUR_COLORS[label],
            lw=3.0,
            solid_capstyle="round",
            zorder=3,
        )

    for label in left_labels:
        anchor = anchors[label]
        manual_position = MANUAL_TEXT_POSITIONS.get(label)
        text_x = manual_position[0] if manual_position is not None else 14.0
        text_y = manual_position[1] if manual_position is not None else left_y[label]
        annotation = ax.annotate(
            DISPLAY_NAMES[label],
            xy=(anchor[0], anchor[1]),
            xytext=(text_x, text_y),
            ha="left",
            va="center",
            fontsize=10.5,
            color="white",
            arrowprops=dict(
                arrowstyle="->",
                color=COMMON_CONTOUR_COLORS[label],
                lw=1.8,
                shrinkA=6,
                shrinkB=4,
                connectionstyle="arc3,rad=0.08",
            ),
            zorder=5,
        )
        annotation.set_path_effects([pe.withStroke(linewidth=3.0, foreground="black")])

    for label in right_labels:
        anchor = anchors[label]
        text_x = width - 14
        text_y = right_y[label]
        annotation = ax.annotate(
            DISPLAY_NAMES[label],
            xy=(anchor[0], anchor[1]),
            xytext=(text_x, text_y),
            ha="right",
            va="center",
            fontsize=10.5,
            color="white",
            arrowprops=dict(
                arrowstyle="->",
                color=COMMON_CONTOUR_COLORS[label],
                lw=1.8,
                shrinkA=6,
                shrinkB=4,
                connectionstyle="arc3,rad=-0.08",
            ),
            zorder=5,
        )
        annotation.set_path_effects([pe.withStroke(linewidth=3.0, foreground="black")])

    ax.set_xlim(0, width)
    ax.set_ylim(height, 0)
    ax.axis("off")
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=100, pad_inches=0, facecolor=fig.get_facecolor())
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    p7_image, p7_contours = load_frame_vtln(args.p7_basename, args.vtln_dir)
    nnunet_image, nnunet_contours = load_frame_npy(args.nnunet_frame, args.nnunet_case_dir, args.nnunet_case_dir / "contours")
    common_labels = sorted(set(p7_contours) & set(nnunet_contours), key=lambda label: (label not in RIGHT_SIDE_LABELS, label))
    if not common_labels:
        raise ValueError("No common anatomical elements were found between P7 and the nnUNet target.")

    p7_output = args.output_dir / "p7_common_elements_labeled.png"
    nnunet_output = args.output_dir / "nnunet_common_elements_labeled.png"

    draw_labeled_figure(
        p7_image,
        p7_contours,
        common_labels,
        output_path=p7_output,
    )
    draw_labeled_figure(
        nnunet_image,
        nnunet_contours,
        common_labels,
        output_path=nnunet_output,
    )

    print(f"Common labels ({len(common_labels)}): {', '.join(common_labels)}")
    print(f"Saved labeled P7 image: {p7_output}")
    print(f"Saved labeled nnUNet image: {nnunet_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
