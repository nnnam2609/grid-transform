from __future__ import annotations

import numpy as np


def reference_center_channel(image) -> np.ndarray:
    """Return the center frame channel for RGB triplets, else the image itself."""
    array = np.asarray(image)
    if array.ndim == 3:
        channel_index = 1 if array.shape[2] >= 2 else 0
        array = array[..., channel_index]
    return array


def as_grayscale_uint8(image) -> np.ndarray:
    array = reference_center_channel(image)
    return np.clip(array, 0, 255).astype(np.uint8)


def as_grayscale_float(image) -> np.ndarray:
    return np.asarray(reference_center_channel(image), dtype=float)
