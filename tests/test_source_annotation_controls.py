from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from grid_transform.source_annotation import (
    controls_to_grid_constraints,
    load_source_annotation_json,
    normalize_grid_controls,
    save_source_annotation_json,
)


class SourceAnnotationControlsTests(unittest.TestCase):
    def test_legacy_json_loads_with_default_grid_controls(self) -> None:
        payload = {
            "metadata": {
                "artspeech_speaker": "P7",
                "session": "S2",
            },
            "contours": {
                "c1": [[0.0, 0.0], [1.0, 1.0]],
                "pharynx": [[2.0, 0.0], [2.0, 2.0]],
            },
        }
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "legacy_annotation.json"
            path.write_text(json.dumps(payload), encoding="utf-8")

            loaded = load_source_annotation_json(path)

        controls = loaded["metadata"]["grid_controls"]
        self.assertTrue(controls["include_in_grid"]["c1"])
        self.assertTrue(controls["include_in_grid"]["pharynx"])
        self.assertEqual(controls["constraint_group"]["c1"], "none")
        self.assertEqual(controls["constraint_group"]["pharynx"], "none")

    def test_save_roundtrip_persists_normalized_grid_controls(self) -> None:
        contours = {
            "c1": np.array([[0.0, 0.0], [1.0, 1.0]], dtype=float),
            "extra-pharynx": np.array([[2.0, 0.0], [2.0, 2.0]], dtype=float),
        }
        metadata = {
            "grid_controls": {
                "include_in_grid": {
                    "c1": False,
                    "extra-pharynx": True,
                },
                "constraint_group": {
                    "extra-pharynx": "pharynx",
                },
            }
        }
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "annotation.json"
            save_source_annotation_json(path, metadata, contours)
            loaded = load_source_annotation_json(path)

        controls = loaded["metadata"]["grid_controls"]
        self.assertTrue(controls["include_in_grid"]["c1"], "required contours must stay enabled")
        self.assertTrue(controls["include_in_grid"]["extra-pharynx"])
        self.assertEqual(controls["constraint_group"]["extra-pharynx"], "pharynx")

    def test_controls_to_grid_constraints_skips_disabled_or_missing_names(self) -> None:
        contours = {
            "extra-velum": np.array([[0.0, 0.0], [1.0, 0.0]], dtype=float),
            "extra-pharynx": np.array([[2.0, 0.0], [2.0, 1.0]], dtype=float),
        }
        controls = normalize_grid_controls(
            contours,
            {
                "include_in_grid": {
                    "extra-velum": False,
                    "extra-pharynx": True,
                    "missing-name": True,
                },
                "constraint_group": {
                    "extra-velum": "velum",
                    "extra-pharynx": "pharynx",
                    "missing-name": "velum",
                },
            },
        )

        constraints = controls_to_grid_constraints(contours, controls)
        self.assertEqual(len(constraints["velum_paths"]), 0)
        self.assertEqual(len(constraints["pharynx_paths"]), 1)


if __name__ == "__main__":
    unittest.main()
