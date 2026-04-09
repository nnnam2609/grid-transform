r"""Export a labeled image of ALL ground-truth nnUNet annotations drawn on the speaker image.

Usage:
  cd "C:\Users\nhnguyen\PhD_A2A\grid transform"
  .\.venv\Scripts\python .\scripts\run\run_export_nnunet_all_elements.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use("Agg")

from grid_transform.config import DEFAULT_OUTPUT_DIR, VT_SEG_DATA_ROOT
from grid_transform.image_utils import as_grayscale_uint8
from grid_transform.io import load_frame_npy

NNUNET_CASE_DIR = VT_SEG_DATA_ROOT / "2008-003^01-1791" / "test"
NNUNET_FRAME = 143020
OUTPUT_PATH = DEFAULT_OUTPUT_DIR / "slide" / "images" / "nnunet_all_elements_labeled.png"

# ── colour palette ────────────────────────────────────────────────────────────
# 30 perceptually distinct colours cycling through hue, assigning named ones first
NAMED_COLORS: dict[str, str] = {
    "c1":                   "#457b9d",
    "c2":                   "#4d7ea8",
    "c3":                   "#5c96bc",
    "c4":                   "#7aa6c2",
    "c5":                   "#90b8d8",
    "c6":                   "#b8d8f2",
    "incisior-hard-palate": "#ef476f",
    "mandible-incisior":    "#f4a261",
    "pharynx":              "#118ab2",
    "soft-palate":          "#fb8500",
    "soft-palate-midline":  "#8338ec",
    "tongue":               "#06d6a0",
    "tongue-floor":         "#05c491",
    "epiglottis":           "#ffd166",
    "hyoid":                "#e76f51",
    "thyroid-cartilage":    "#2ec4b6",
    "arytenoid-cartilage":  "#cbf3f0",
    "vocal-folds":          "#ff6b6b",
    "geniohyoid-muscle":    "#a8dadc",
    "soft-palate-midline":  "#c77dff",
    "upper-lip":            "#ffb4a2",
    "lower-lip":            "#e5989b",
    "chin":                 "#b5838d",
    "nose-tip":             "#e63946",
    "nasal-root":           "#f1faee",
    "frontal-sinus":        "#a8c8fa",
    "sphenoid":             "#fae588",
    "occipital-crest":      "#d62839",
    "brain-stem":           "#82b3d9",
    "cerebellum":           "#4cc9f0",
    "vomer":                "#7678ed",
}


def _centroid(pts: np.ndarray) -> np.ndarray:
    return np.asarray(pts, dtype=float).mean(axis=0)


def main() -> int:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    image, contours = load_frame_npy(NNUNET_FRAME, NNUNET_CASE_DIR, NNUNET_CASE_DIR / "contours")
    gray = as_grayscale_uint8(image)
    H, W = gray.shape[:2]

    labels = sorted(contours.keys())
    anchors = {lbl: _centroid(contours[lbl]) for lbl in labels}
    colors  = {lbl: NAMED_COLORS.get(lbl, "#cccccc") for lbl in labels}

    # ── split left / right by centroid x position (midpoint = W/2) ──────────
    mid = W / 2.0
    left_labels  = sorted([l for l in labels if anchors[l][0] <= mid], key=lambda l: anchors[l][1])
    right_labels = sorted([l for l in labels if anchors[l][0]  > mid], key=lambda l: anchors[l][1])

    margin_top    = 30
    margin_bottom = 30
    y_range       = H - margin_top - margin_bottom

    def spread_y(group: list[str]) -> dict[str, float]:
        if not group:
            return {}
        ys = np.linspace(margin_top, H - margin_bottom, len(group))
        return {lbl: float(y) for lbl, y in zip(group, ys)}

    left_y  = spread_y(left_labels)
    right_y = spread_y(right_labels)

    # ── figure: 2× wide to accommodate label columns on both sides ───────────
    side_col  = 2.0          # inches for each label column
    img_w_in  = W / 100.0
    img_h_in  = H / 100.0
    fig_w     = img_w_in + 2 * side_col
    fig_h     = img_h_in

    fig = plt.figure(figsize=(fig_w, fig_h), dpi=100)
    fig.patch.set_facecolor("black")

    # image axes (centred; the side columns are pure white-space for arrows landing)
    ax = fig.add_axes([side_col / fig_w, 0, img_w_in / fig_w, 1.0])
    ax.set_facecolor("black")
    ax.imshow(gray, cmap="gray", vmin=0, vmax=255, extent=(0, W, H, 0))

    # draw contours
    for lbl in labels:
        pts = np.asarray(contours[lbl], dtype=float)
        if pts.ndim == 1:
            pts = pts.reshape(-1, 2)
        ax.plot(pts[:, 0], pts[:, 1], color=colors[lbl], lw=2.0,
                solid_capstyle="round", zorder=3)

    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)
    ax.axis("off")

    # ── annotations ──────────────────────────────────────────────────────────
    def annotate(group: dict[str, float], ha: str, text_x_data: float) -> None:
        for lbl, ty in group.items():
            anchor = anchors[lbl]
            display = lbl.replace("-", " ")
            ann = ax.annotate(
                display,
                xy=(anchor[0], anchor[1]),
                xytext=(text_x_data, ty),
                xycoords="data",
                textcoords="data",
                ha=ha,
                va="center",
                fontsize=8.5,
                color="white",
                arrowprops=dict(
                    arrowstyle="->",
                    color=colors[lbl],
                    lw=1.3,
                    shrinkA=4,
                    shrinkB=3,
                    connectionstyle="arc3,rad=0.0",
                ),
                zorder=5,
            )
            ann.set_path_effects([pe.withStroke(linewidth=2.5, foreground="black")])

    # left labels: place text just outside left edge of image (negative x in data coords)
    text_x_left  = -W * (side_col / img_w_in) * 0.88   # slightly in from the canvas edge
    text_x_right =  W * (1 + side_col / img_w_in * 0.88)

    annotate(left_y,  ha="left",  text_x_data=text_x_left)
    annotate(right_y, ha="right", text_x_data=text_x_right)

    plt.savefig(OUTPUT_PATH, dpi=100, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)

    print(f"Wrote: {OUTPUT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
