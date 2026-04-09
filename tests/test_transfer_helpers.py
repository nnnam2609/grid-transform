from __future__ import annotations

import unittest

import numpy as np

from grid_transform.transfer import resolve_common_articulators, smooth_transformed_contours


class TransferHelperTests(unittest.TestCase):
    def test_resolve_common_articulators_respects_requested_order(self) -> None:
        target = {"tongue": None, "soft-palate": None, "pharynx": None}
        source = {"pharynx": None, "tongue": None, "soft-palate": None}
        self.assertEqual(
            resolve_common_articulators(target, source, requested="pharynx,tongue,missing"),
            ["pharynx", "tongue"],
        )

    def test_smooth_transformed_contours_keeps_mandible_unsmoothed(self) -> None:
        contours = {
            "mandible-incisior": np.array(
                [[0.0, 0.0], [1.0, 2.0], [2.0, -1.0], [3.0, 3.0], [4.0, 0.0], [5.0, 1.0]],
                dtype=float,
            ),
            "tongue": np.array(
                [[0.0, 0.0], [1.0, 3.0], [2.0, -2.0], [3.0, 4.0], [4.0, -1.0], [5.0, 1.0]],
                dtype=float,
            ),
        }
        smoothed = smooth_transformed_contours(contours)
        self.assertTrue(np.allclose(smoothed["mandible-incisior"], contours["mandible-incisior"]))
        self.assertFalse(np.allclose(smoothed["tongue"], contours["tongue"]))


if __name__ == "__main__":
    unittest.main()
