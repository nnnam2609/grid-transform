from __future__ import annotations

import unittest

from grid_transform.contour_names import normalize_contour_name
from tools.report.common import COMMON_NNUNET_LABELS, subset_common_contours


class ContourNameTests(unittest.TestCase):
    def test_normalize_contour_name_strips_frame_prefix(self) -> None:
        self.assertEqual(
            normalize_contour_name("143020_mandible-incisior", COMMON_NNUNET_LABELS),
            "mandible-incisior",
        )

    def test_normalize_contour_name_leaves_unknown_label(self) -> None:
        self.assertEqual(normalize_contour_name("custom_label", COMMON_NNUNET_LABELS), "custom_label")

    def test_subset_common_contours_keeps_shared_report_order(self) -> None:
        p7_contours = {"c2": object(), "soft-palate": object(), "tongue": object(), "c1": object()}
        nnunet_contours = {"c1": object(), "soft-palate": object(), "c2": object(), "pharynx": object()}
        self.assertEqual(subset_common_contours(p7_contours, nnunet_contours), ["c1", "c2", "soft-palate"])


if __name__ == "__main__":
    unittest.main()
