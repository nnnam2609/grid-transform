from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from grid_transform.curated_batch import (
    BatchCase,
    build_case_output_dir,
    load_cases,
    parse_manifest_row,
    resolve_reference_bundle,
    resolve_speaker_root,
)


class CuratedBatchTests(unittest.TestCase):
    def test_parse_manifest_row_extracts_case_metadata(self) -> None:
        case = parse_manifest_row(
            {
                "output_basename": "1640_P7_S2_F0829",
                "annotation_status": "ready",
                "annotation_source": "C:/tmp/1640_P7_S2_F0829.zip",
                "reference_bundle_dir": "C:/tmp/bundle",
                "reference_bundle_name": "1640_P7_S2_F0829",
            }
        )
        self.assertEqual(case.speaker, "P7")
        self.assertEqual(case.session, "S2")
        self.assertEqual(case.frame_index_1based, 829)
        self.assertEqual(case.annotation_status, "ready")
        self.assertEqual(case.reference_bundle_name, "1640_P7_S2_F0829")

    def test_load_cases_and_build_case_output_dir(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir_str:
            tmp_dir = Path(tmp_dir_str)
            manifest_path = tmp_dir / "selection_manifest.csv"
            manifest_path.write_text(
                "output_basename,annotation_status,annotation_source,reference_bundle_dir,reference_bundle_name\n"
                "1640_P7_S2_F0829,ready,,,,\n",
                encoding="utf-8",
            )
            cases = load_cases(manifest_path, tmp_dir)
            self.assertEqual(len(cases), 1)
            self.assertEqual(build_case_output_dir(tmp_dir / "out", cases[0]), tmp_dir / "out" / "1640_P7_S2_F0829")

    def test_resolve_reference_bundle_prefers_override_bundle(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir_str:
            bundle_dir = Path(tmp_dir_str)
            (bundle_dir / "1640_P7_S2_F0829.zip").write_bytes(b"zip")
            (bundle_dir / "1640_P7_S2_F0829.png").write_bytes(b"png")
            case = BatchCase(
                output_basename="1640_P7_S2_F0829",
                speaker="P7",
                raw_subject="1640",
                session="S2",
                frame_index_1based=829,
                annotation_status="ready",
                annotation_source_path=None,
                reference_bundle_dir=bundle_dir,
                reference_bundle_name="1640_P7_S2_F0829",
            )
            name, resolved_dir, image_path, zip_path = resolve_reference_bundle(case)
            self.assertEqual(name, "1640_P7_S2_F0829")
            self.assertEqual(resolved_dir, bundle_dir)
            self.assertEqual(image_path, bundle_dir / "1640_P7_S2_F0829.png")
            self.assertEqual(zip_path, bundle_dir / "1640_P7_S2_F0829.zip")

    def test_resolve_speaker_root_handles_direct_and_nested_layouts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir_str:
            root = Path(tmp_dir_str)
            direct = root / "P7"
            (direct / "DCM_2D").mkdir(parents=True)
            (direct / "OTHER").mkdir(parents=True)
            self.assertEqual(resolve_speaker_root(root, "P7"), direct)

        with tempfile.TemporaryDirectory() as tmp_dir_str:
            root = Path(tmp_dir_str)
            nested = root / "P7" / "P7"
            (nested / "DCM_2D").mkdir(parents=True)
            (nested / "OTHER").mkdir(parents=True)
            self.assertEqual(resolve_speaker_root(root, "P7"), nested)


if __name__ == "__main__":
    unittest.main()
