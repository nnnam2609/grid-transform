from __future__ import annotations

from pathlib import Path

import numpy as np

from grid_transform.config import VT_SEG_DATA_ROOT


P7_BASENAME = "1640_P7_S2_F0829"
NNUNET_CASE_DIR_DEFAULT = VT_SEG_DATA_ROOT / "2008-003^01-1791" / "test"
NNUNET_FRAME_DEFAULT = 143020
COMMON_NNUNET_LABELS = (
    "c1",
    "c2",
    "c3",
    "c4",
    "c5",
    "c6",
    "incisior-hard-palate",
    "mandible-incisior",
    "pharynx",
    "soft-palate",
    "soft-palate-midline",
)


def subset_common_contours(
    p7_contours: dict[str, np.ndarray],
    nnunet_contours: dict[str, np.ndarray],
) -> list[str]:
    return [label for label in COMMON_NNUNET_LABELS if label in p7_contours and label in nnunet_contours]
