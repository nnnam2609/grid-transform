from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

import numpy as np

from grid_transform.source_annotation import save_source_annotation_json
from grid_transform.vtln_annotation_sync import (
    build_temp_annotation_sync_spec,
    find_latest_source_annotation,
    resolve_vtln_data_dir,
)


class VtlnAnnotationSyncTests(unittest.TestCase):
    def test_resolve_vtln_data_dir_prefers_nested_data_folder(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir_str:
            tmp_dir = Path(tmp_dir_str)
            vtln_root = tmp_dir / "VTLN"
            vtln_data = vtln_root / "data"
            vtln_data.mkdir(parents=True)
            self.assertEqual(resolve_vtln_data_dir(vtln_root), vtln_data)
            self.assertEqual(resolve_vtln_data_dir(vtln_data), vtln_data)

    def test_find_latest_source_annotation_prefers_newest_matching_payload(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir_str:
            output_root = Path(tmp_dir_str)
            older_path = output_root / "annotation_to_grid_transform" / "ws1" / "source_annotation.latest.json"
            newer_path = output_root / "source_annotation_edits" / "edit1" / "edited_annotation.json"
            ignored_target_path = output_root / "annotation_to_grid_transform" / "_vtln_promotions" / "target" / "p7" / "source_annotation.latest.json"
            contours = {"tongue": np.array([[0.0, 0.0], [1.0, 1.0]], dtype=float)}

            source_metadata = {
                "artspeech_speaker": "P7",
                "session": "S2",
                "source_frame": 829,
                "reference_speaker": "1640_P7_S2_F0829",
                "source_shape": [80, 80],
            }
            save_source_annotation_json(older_path, source_metadata, contours)
            save_source_annotation_json(newer_path, source_metadata, contours)
            save_source_annotation_json(
                ignored_target_path,
                {
                    "artspeech_speaker": "P7",
                    "session": "S2",
                    "source_frame": 829,
                    "reference_speaker": "1640_P7_S2_F0829",
                    "source_shape": [80, 80],
                    "promoted_from_role": "target",
                },
                contours,
            )

            os.utime(older_path, (1000, 1000))
            os.utime(newer_path, (2000, 2000))
            os.utime(ignored_target_path, (3000, 3000))

            payload = find_latest_source_annotation(
                artspeech_speaker="P7",
                session="S2",
                source_frame=829,
                current_source_shape=(80, 80),
                output_root=output_root,
            )

            self.assertIsNotNone(payload)
            self.assertEqual(Path(str(payload["path"])), newer_path)

    def test_build_temp_annotation_sync_spec_supports_vtln_target_payload(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir_str:
            tmp_dir = Path(tmp_dir_str)
            vtln_dir = tmp_dir / "custom_vtln" / "data"
            vtln_dir.mkdir(parents=True)
            target_json = tmp_dir / "outputs" / "annotation_to_grid_transform" / "ws1" / "target_annotation.latest.json"
            contours = {"tongue": np.array([[0.0, 0.0], [1.0, 1.0]], dtype=float)}
            save_source_annotation_json(
                target_json,
                {
                    "role": "target",
                    "target_type": "vtln",
                    "vtln_reference": "1640_P7_S2_F0829",
                    "vtln_dir": str(vtln_dir),
                    "target_shape": [480, 480],
                },
                contours,
            )

            spec = build_temp_annotation_sync_spec(target_json)

            self.assertIsNotNone(spec)
            self.assertEqual(spec.role, "target")
            self.assertEqual(spec.reference_name, "1640_P7_S2_F0829")
            self.assertEqual(spec.vtln_dir, vtln_dir)
            self.assertEqual(spec.source_shape, (480, 480))


if __name__ == "__main__":
    unittest.main()
