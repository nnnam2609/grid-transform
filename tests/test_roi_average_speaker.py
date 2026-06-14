from __future__ import annotations

from types import SimpleNamespace
import unittest

import numpy as np

from grid_transform.roi_average_speaker import (
    build_affine_only_transform,
    close_contour_polygon,
    compute_mask_overlap_metrics,
    polygon_to_mask,
    resolve_roi_average_labels,
)


def _make_fake_grid(offset: tuple[float, float], *, m1: tuple[float, float], left_pts: np.ndarray):
    dx, dy = offset
    return SimpleNamespace(
        cervical_centers={f"c{i}": np.array([7.0 + dx, float(i - 1) + dy], dtype=float) for i in range(1, 7)},
        left_pts=np.asarray(left_pts, dtype=float),
        M1_point=np.array(m1, dtype=float),
        P1_point=np.array([7.0 + dx, 0.0 + dy], dtype=float),
        I_points={f"I{i}": np.array([float(i - 1) + dx, 0.0 + dy], dtype=float) for i in range(1, 8)},
    )


class RoiAverageSpeakerTests(unittest.TestCase):
    def test_close_contour_polygon_appends_first_point(self) -> None:
        contour = np.array([[1.0, 2.0], [3.0, 2.0], [2.0, 4.0]], dtype=float)
        closed = close_contour_polygon(contour)
        self.assertEqual(closed.shape, (4, 2))
        self.assertTrue(np.allclose(closed[0], contour[0]))
        self.assertTrue(np.allclose(closed[-1], contour[0]))

    def test_polygon_mask_overlap_identity_and_translation(self) -> None:
        square = np.array([[5.0, 5.0], [12.0, 5.0], [12.0, 12.0], [5.0, 12.0]], dtype=float)
        translated = square + np.array([3.0, 0.0], dtype=float)
        mask_a = polygon_to_mask(square, (24, 24))
        mask_b = polygon_to_mask(square, (24, 24))
        mask_c = polygon_to_mask(translated, (24, 24))

        same_metrics = compute_mask_overlap_metrics(mask_a, mask_b)
        shifted_metrics = compute_mask_overlap_metrics(mask_a, mask_c)
        self.assertEqual(same_metrics["intersection"], same_metrics["union"])
        self.assertAlmostEqual(float(same_metrics["dice"]), 1.0)
        self.assertAlmostEqual(float(same_metrics["iou"]), 1.0)
        self.assertLess(float(shifted_metrics["dice"]), 1.0)
        self.assertLess(float(shifted_metrics["iou"]), 1.0)

    def test_resolve_roi_average_labels_reports_exclusions(self) -> None:
        speakers = {
            "P1": SimpleNamespace(
                contours={
                    "c1": np.array([[0.0, 0.0], [1.0, 0.0]], dtype=float),
                    "c2": np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]], dtype=float),
                    "pharynx": np.array([[0.0, 0.0], [2.0, 0.0], [2.0, 1.0]], dtype=float),
                }
            ),
            "P2": SimpleNamespace(
                contours={
                    "c1": np.array([[0.0, 0.0], [1.0, 0.0]], dtype=float),
                    "c2": np.array([[0.0, 0.0], [2.0, 0.0], [2.0, 1.0]], dtype=float),
                }
            ),
        }
        labels, excluded = resolve_roi_average_labels(speakers, exclude_labels=("pharynx",))
        self.assertEqual(labels, ["c2"])
        excluded_map = {row["label"]: row for row in excluded}
        self.assertIn("c1", excluded_map)
        self.assertIn("pharynx", excluded_map)
        self.assertEqual(excluded_map["pharynx"]["reasons"][0]["type"], "explicit_exclude")

    def test_build_affine_only_transform_uses_step1_anchors_only(self) -> None:
        source_grid = _make_fake_grid(
            (0.0, 0.0),
            m1=(100.0, 100.0),
            left_pts=np.array([[0.0, 10.0], [0.0, 20.0]], dtype=float),
        )
        target_grid = _make_fake_grid(
            (10.0, 5.0),
            m1=(999.0, 999.0),
            left_pts=np.array([[42.0, 13.0], [52.0, 31.0]], dtype=float),
        )

        transform = build_affine_only_transform(source_grid, target_grid)
        mapped = transform["apply_affine"](np.array([[3.5, 2.5]], dtype=float))
        self.assertTrue(np.allclose(mapped, np.array([[13.5, 7.5]], dtype=float)))
        self.assertIn("C6", transform["step1_labels"])
        self.assertNotIn("M1", transform["step1_labels"])
        self.assertNotIn("L6", transform["step1_labels"])


if __name__ == "__main__":
    unittest.main()
