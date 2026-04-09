from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from grid_transform.analysis_shared import (
    choose_label_colors,
    compute_global_common_labels,
    read_curated_specs_list,
    read_curated_specs_map,
    stack_resampled_contours,
)


class AnalysisSharedTests(unittest.TestCase):
    def test_read_curated_specs_helpers_parse_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir_str:
            tmp_dir = Path(tmp_dir_str)
            manifest_path = tmp_dir / "selection_manifest.csv"
            manifest_path.write_text(
                "speaker,output_basename,raw_subject,selected_source\n"
                "P7,1640_P7_S2_F0829,1640,1640/S2/F0829\n"
                "P4,1637_P4_S3_F0123,1637,1637/S3/F0123\n",
                encoding="utf-8",
            )
            (tmp_dir / "1640_P7_S2_F0829.png").write_bytes(b"png")
            (tmp_dir / "1637_P4_S3_F0123.png").write_bytes(b"png")
            specs_map = read_curated_specs_map(tmp_dir)
            specs_list = read_curated_specs_list(tmp_dir)
            self.assertEqual(sorted(specs_map), ["P4", "P7"])
            self.assertEqual([spec.basename for spec in specs_list], ["1637_P4_S3_F0123", "1640_P7_S2_F0829"])
            self.assertEqual(specs_map["P7"].session, "S2")
            self.assertEqual(specs_map["P7"].frame, 829)

    def test_stack_resampled_contours_and_colors(self) -> None:
        contours = {
            "c1": np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]], dtype=float),
            "tongue": np.array([[0.0, 1.0], [1.0, 2.0], [2.0, 1.0]], dtype=float),
        }
        stacked = stack_resampled_contours(contours, ["c1", "tongue"], 5)
        self.assertEqual(stacked.shape, (10, 2))
        colors = choose_label_colors(["tongue", "c1"])
        self.assertIn("tongue", colors)
        self.assertIn("c1", colors)

    def test_compute_global_common_labels_intersects_all_speakers(self) -> None:
        class DummySpeaker:
            def __init__(self, contours):
                self.contours = contours

        loaded = {
            "P7": DummySpeaker({"c1": np.zeros((2, 2)), "tongue": np.zeros((2, 2)), "pharynx": np.zeros((2, 2))}),
            "P4": DummySpeaker({"c1": np.zeros((2, 2)), "tongue": np.zeros((2, 2))}),
        }
        self.assertEqual(compute_global_common_labels(loaded), ["c1", "tongue"])


if __name__ == "__main__":
    unittest.main()
