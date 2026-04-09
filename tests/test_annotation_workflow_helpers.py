from __future__ import annotations

import unittest
from pathlib import Path

from grid_transform.workspace_paths import sanitize_workspace_token, step_file_paths


class AnnotationWorkflowHelperTests(unittest.TestCase):
    def test_sanitize_workspace_token_normalizes_unsafe_characters(self) -> None:
        self.assertEqual(sanitize_workspace_token(" P7 / S2 : frame#1 "), "P7_S2_frame_1")

    def test_step_file_paths_uses_latest_workspace_convention(self) -> None:
        workspace_dir = Path("workspace") / "example"
        paths = step_file_paths(workspace_dir)
        self.assertEqual(paths["selection"], workspace_dir / "workspace_selection.latest.json")
        self.assertEqual(paths["transform_spec"], workspace_dir / "transform_spec.latest.json")
        self.assertEqual(paths["overview_preview"], workspace_dir / "step2_overview.latest.png")


if __name__ == "__main__":
    unittest.main()
