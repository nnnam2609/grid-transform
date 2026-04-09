from __future__ import annotations

import re
from pathlib import Path


WORKSPACE_SAFE_RE = re.compile(r"[^A-Za-z0-9._-]+")


def sanitize_workspace_token(value: str) -> str:
    cleaned = WORKSPACE_SAFE_RE.sub("_", value.strip())
    return cleaned.strip("._-") or "workspace"


def step_file_paths(workspace_dir: Path) -> dict[str, Path]:
    return {
        "selection": workspace_dir / "workspace_selection.latest.json",
        "source_annotation": workspace_dir / "source_annotation.latest.json",
        "target_annotation": workspace_dir / "target_annotation.latest.json",
        "landmark_overrides": workspace_dir / "landmark_overrides.latest.json",
        "transform_spec": workspace_dir / "transform_spec.latest.json",
        "transform_review": workspace_dir / "transform_review.latest.json",
        "native_preview": workspace_dir / "step2_native.latest.png",
        "affine_preview": workspace_dir / "step2_affine.latest.png",
        "final_preview": workspace_dir / "step2_final.latest.png",
        "overview_preview": workspace_dir / "step2_overview.latest.png",
    }
