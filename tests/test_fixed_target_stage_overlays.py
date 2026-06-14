from __future__ import annotations

from types import SimpleNamespace
import unittest

import numpy as np

from grid_transform.fixed_target_stage_overlays import (
    close_visual_contour,
    compute_stage_variance_details,
    default_stage_dir_name,
    resolve_common_labels,
    resolve_overlay_ids,
    resolve_variance_labels,
    stage_title,
)


def _speaker(speaker_id: str, gender: str, labels: list[str]):
    contours = {label: np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]], dtype=float) for label in labels}
    spec = SimpleNamespace(speaker_id=speaker_id, basename=f"{speaker_id}_case", gender=gender)
    return SimpleNamespace(spec=spec, contours=contours)


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


class FixedTargetStageOverlayTests(unittest.TestCase):
    def test_close_visual_contour_appends_first_point(self) -> None:
        contour = np.array([[1.0, 2.0], [3.0, 2.0], [2.0, 4.0]], dtype=float)
        closed = close_visual_contour(contour)
        self.assertEqual(closed.shape, (4, 2))
        self.assertTrue(np.allclose(closed[0], closed[-1]))

    def test_close_visual_contour_keeps_pharynx_open(self) -> None:
        contour = np.array([[1.0, 2.0], [3.0, 2.0], [2.0, 4.0]], dtype=float)
        open_contour = close_visual_contour(contour, "pharynx")
        self.assertEqual(open_contour.shape, (3, 2))
        self.assertFalse(np.allclose(open_contour[0], open_contour[-1]))

    def test_resolve_overlay_ids_adds_target_to_gender_cohort(self) -> None:
        speakers = {
            "P1": _speaker("P1", "male", ["c1"]),
            "P2": _speaker("P2", "male", ["c1"]),
            "P10": _speaker("P10", "female", ["c1"]),
        }
        cohort_ids, overlay_ids = resolve_overlay_ids(speakers, "male", target_speaker_id="P10")
        self.assertEqual(cohort_ids, ["P1", "P2"])
        self.assertEqual(overlay_ids, ["P1", "P2", "P10"])

    def test_resolve_common_labels_uses_target_intersection(self) -> None:
        speakers = {
            "P1": _speaker("P1", "male", ["c1", "tongue"]),
            "P2": _speaker("P2", "male", ["c1", "tongue"]),
            "P10": _speaker("P10", "female", ["c1", "tongue", "soft-palate"]),
        }
        labels = resolve_common_labels(speakers, ["P1", "P2", "P10"])
        self.assertEqual(labels, ["c1", "tongue"])

    def test_resolve_variance_labels_excludes_pharynx(self) -> None:
        labels = resolve_variance_labels(["c1", "pharynx", "soft-palate"])
        self.assertEqual(labels, ["c1", "soft-palate"])

    def test_stage_title_compacts_all_contour_titles(self) -> None:
        title = stage_title("all", "affine", "contours", target_speaker_id="P10", target_injected=False)
        self.assertEqual(title, "Affine contours")

    def test_compute_stage_variance_details_reports_zero_for_identical_contours(self) -> None:
        speakers = {
            "P1": _speaker("P1", "male", ["c1"]),
            "P10": _speaker("P10", "female", ["c1"]),
        }
        identical = _square(0.0, 0.0, 2.0, 2.0)
        stage_populations = {
            stage: {
                "mapped_contours": {
                    "P1": {"c1": identical},
                    "P10": {"c1": identical},
                },
                "median_speaker": "P1",
                "closest_to_mean": "P1",
                "nc_template": 4,
            }
            for stage in ("init", "affine", "tps")
        }
        details = compute_stage_variance_details(
            speakers,
            ["c1"],
            stage_populations,
            cohort_name="all",
            target_speaker_id="P10",
        )
        for row in details["summary_rows"]:
            self.assertAlmostEqual(row["overall_variance"], 0.0)
            self.assertAlmostEqual(row["overall_mean_pairwise_dice"], 1.0)
            self.assertAlmostEqual(row["overall_mean_pairwise_dice_distance"], 0.0)
        for row in details["speaker_rows"]:
            self.assertAlmostEqual(row["mean_pairwise_dice"], 1.0)
            self.assertAlmostEqual(row["mean_squared_dice_distance"], 0.0)
        for row in details["pair_rows"]:
            self.assertAlmostEqual(row["dice"], 1.0)
            self.assertAlmostEqual(row["dice_distance_squared"], 0.0)

    def test_compute_stage_variance_details_can_preserve_iou_metric(self) -> None:
        speakers = {
            "P1": _speaker("P1", "male", ["c1"]),
            "P10": _speaker("P10", "female", ["c1"]),
        }
        identical = _square(0.0, 0.0, 2.0, 2.0)
        stage_populations = {
            stage: {
                "mapped_contours": {
                    "P1": {"c1": identical},
                    "P10": {"c1": identical},
                },
                "median_speaker": "P1",
                "closest_to_mean": "P1",
                "nc_template": 4,
            }
            for stage in ("init", "affine", "tps")
        }
        details = compute_stage_variance_details(
            speakers,
            ["c1"],
            stage_populations,
            cohort_name="all",
            target_speaker_id="P10",
            overlap_metric="iou",
        )
        for row in details["summary_rows"]:
            self.assertAlmostEqual(row["overall_variance"], 0.0)
            self.assertAlmostEqual(row["overall_mean_pairwise_iou"], 1.0)
            self.assertAlmostEqual(row["overall_mean_pairwise_iou_distance"], 0.0)
        for row in details["pair_rows"]:
            self.assertAlmostEqual(row["iou"], 1.0)
            self.assertAlmostEqual(row["iou_distance_squared"], 0.0)

    def test_dice_distance_variance_is_lower_than_iou_for_same_partial_overlap(self) -> None:
        speakers = {
            "P1": _speaker("P1", "male", ["c1"]),
            "P10": _speaker("P10", "female", ["c1"]),
        }
        source = _square(0.0, 0.0, 4.0, 4.0)
        target = _square(2.0, 0.0, 6.0, 4.0)
        stage_populations = {
            stage: {
                "mapped_contours": {
                    "P1": {"c1": source},
                    "P10": {"c1": target},
                },
                "median_speaker": "P1",
                "closest_to_mean": "P1",
                "nc_template": 4,
            }
            for stage in ("init", "affine", "tps")
        }
        dice_details = compute_stage_variance_details(
            speakers,
            ["c1"],
            stage_populations,
            cohort_name="all",
            target_speaker_id="P10",
        )
        iou_details = compute_stage_variance_details(
            speakers,
            ["c1"],
            stage_populations,
            cohort_name="all",
            target_speaker_id="P10",
            overlap_metric="iou",
        )
        self.assertLess(dice_details["summary_rows"][0]["overall_variance"], iou_details["summary_rows"][0]["overall_variance"])
        self.assertLess(dice_details["pair_rows"][0]["dice_distance_squared"], iou_details["pair_rows"][0]["iou_distance_squared"])

    def test_default_stage_dir_name_uses_metric_suffix(self) -> None:
        self.assertEqual(default_stage_dir_name("dice"), "stage_overlays_v2_dice")
        self.assertEqual(default_stage_dir_name("iou"), "stage_overlays_v2_iou")


if __name__ == "__main__":
    unittest.main()
