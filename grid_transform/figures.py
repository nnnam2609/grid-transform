from __future__ import annotations

import numpy as np


def format_frame(ax, image, title: str) -> None:
    """Display one frame while keeping image-space coordinates intact."""
    img = np.asarray(image)
    h_img, w_img = img.shape[:2]
    ax.imshow(image, cmap="gray")
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.axis("off")
    ax.set_xlim(0, w_img)
    ax.set_ylim(h_img, 0)

