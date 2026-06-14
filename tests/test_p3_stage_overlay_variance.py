from __future__ import annotations

from types import SimpleNamespace
import unittest

import numpy as np

from grid_transform.p3_stage_overlay_variance import (
    compute_source_to_target_stage_metrics,
    compute_stage_variance,
    contour_to_mask,
    dice_distance,
    dice_score,
    resolve_closed_roi_labels,
)


def _square(x0: float, y0: float, x1: float, y1: float) -> np.ndarray:
    return np.array(
        [
            [x0, y0],
            [x1, y0],
            [x1, y1],
            [x0, y1],
        ],
        dtype=float,
    )


class P3StageOverlayVarianceTests(unittest.TestCase):
    def test_resolve_closed_roi_labels_keeps_allowlisted_intersection(self) -> None:
        speakers = {
            "P1": SimpleNamespace(
                contours={
                    "c1": _square(0.0, 0.0, 2.0, 2.0),
                    "tongue": _square(0.0, 0.0, 3.0, 1.0),
                    "pharynx": _square(0.0, 0.0, 3.0, 2.0),
                }
            ),
            "P3": SimpleNamespace(
                contours={
                    "c1": _square(1.0, 1.0, 3.0, 3.0),
                    "tongue": _square(0.0, 0.0, 3.0, 1.0),
                }
            ),
        }
        labels, excluded = resolve_closed_roi_labels(speakers)
        self.assertEqual(labels, ["c1"])
        excluded_map = {row["label"]: row for row in excluded}
        self.assertIn("tongue", excluded_map)
        self.assertIn("pharynx", excluded_map)

    def test_dice_helpers_detect_identity_and_shift(self) -> None:
        mask_a = contour_to_mask(_square(4.0, 4.0, 10.0, 10.0), (24, 24))
        mask_b = contour_to_mask(_square(4.0, 4.0, 10.0, 10.0), (24, 24))
        mask_c = contour_to_mask(_square(7.0, 4.0, 13.0, 10.0), (24, 24))
        self.assertAlmostEqual(dice_score(mask_a, mask_b), 1.0)
        self.assertAlmostEqual(dice_distance(mask_a, mask_b), 0.0)
        self.assertLess(dice_score(mask_a, mask_c), 1.0)
        self.assertGreater(dice_distance(mask_a, mask_c), 0.0)

    def test_stage_variance_collapses_after_mapping(self) -> None:
        mask_ref = contour_to_mask(_square(4.0, 4.0, 10.0, 10.0), (24, 24))
        mask_shifted = contour_to_mask(_square(7.0, 4.0, 13.0, 10.0), (24, 24))
        labels = ["c1", "c2"]
        stage_masks = {
            "before_affine": {
                "P1": {"c1": mask_ref, "c2": mask_ref},
                "P3": {"c1": mask_shifted, "c2": mask_ref},
            },
            "after_affine": {
                "P1": {"c1": mask_ref, "c2": mask_ref},
                "P3": {"c1": mask_ref, "c2": mask_ref},
            },
            "after_tps": {
                "P1": {"c1": mask_ref, "c2": mask_ref},
                "P3": {"c1": mask_ref, "c2": mask_ref},
            },
        }

        metrics = compute_stage_variance(stage_masks, labels)
        summaries = metrics["stage_summaries"]
        self.assertGreater(summaries["before_affine"].overall_variance, 0.0)
        self.assertAlmostEqual(summaries["after_affine"].overall_variance, 0.0)
        self.assertAlmostEqual(summaries["after_tps"].overall_variance, 0.0)
        self.assertGreater(summaries["after_affine"].overall_mean_pairwise_dice, summaries["before_affine"].overall_mean_pairwise_dice)

    def test_source_to_target_summary_uses_label_mean(self) -> None:
        mask_ref = contour_to_mask(_square(4.0, 4.0, 10.0, 10.0), (24, 24))
        mask_shifted = contour_to_mask(_square(7.0, 4.0, 13.0, 10.0), (24, 24))
        stage_masks = {
            "before_affine": {
                "P1": {"c1": mask_shifted},
                "P3": {"c1": mask_ref},
            },
            "after_affine": {
                "P1": {"c1": mask_ref},
                "P3": {"c1": mask_ref},
            },
            "after_tps": {
                "P1": {"c1": mask_ref},
                "P3": {"c1": mask_ref},
            },
        }
        rows = compute_source_to_target_stage_metrics(stage_masks, ["c1"])
        row_map = {row["stage"]: row for row in rows}
        self.assertEqual(row_map["before_affine"]["source_speaker"], "P1")
        self.assertEqual(row_map["before_affine"]["target_speaker"], "P3")
        self.assertGreater(row_map["before_affine"]["mean_dice_distance"], 0.0)
        self.assertAlmostEqual(row_map["after_affine"]["mean_dice"], 1.0)
        self.assertAlmostEqual(row_map["after_tps"]["mean_squared_dice_distance"], 0.0)


if __name__ == "__main__":
    unittest.main()
