from __future__ import annotations

import unittest

import numpy as np

from grid_transform.config import DEFAULT_VTNL_DIR
from grid_transform.io import load_frame_vtnl
from grid_transform.vt import build_grid


class VTGridConstraintTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.image, cls.contours = load_frame_vtnl("1640_s10_0829", DEFAULT_VTNL_DIR)

    def test_empty_constraints_are_noop(self) -> None:
        base = build_grid(self.image, self.contours, n_vert=9, n_points=120, frame_number=0)
        constrained = build_grid(
            self.image,
            self.contours,
            n_vert=9,
            n_points=120,
            frame_number=0,
            grid_constraints={"velum_paths": [], "pharynx_paths": []},
        )

        np.testing.assert_allclose(base.horiz_lines[0], constrained.horiz_lines[0])
        np.testing.assert_allclose(base.vt_curve, constrained.vt_curve)
        if base.P1_point is None:
            self.assertIsNone(constrained.P1_point)
        else:
            np.testing.assert_allclose(base.P1_point, constrained.P1_point)

    def test_invalid_constraint_payload_falls_back_cleanly(self) -> None:
        base = build_grid(self.image, self.contours, n_vert=9, n_points=120, frame_number=0)
        constrained = build_grid(
            self.image,
            self.contours,
            n_vert=9,
            n_points=120,
            frame_number=0,
            grid_constraints={
                "velum_paths": ["not-a-path"],
                "pharynx_paths": [np.array([1.0, 2.0, 3.0])],
            },
        )

        np.testing.assert_allclose(base.horiz_lines[0], constrained.horiz_lines[0])
        np.testing.assert_allclose(base.vt_curve, constrained.vt_curve)


if __name__ == "__main__":
    unittest.main()
