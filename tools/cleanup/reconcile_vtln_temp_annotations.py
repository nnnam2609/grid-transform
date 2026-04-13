from __future__ import annotations

import argparse
import json
from pathlib import Path

from grid_transform.config import DEFAULT_OUTPUT_DIR, DEFAULT_VTLN_DIR
from grid_transform.vtln_annotation_sync import (
    build_temp_annotation_sync_spec,
    discover_saved_source_annotation_paths,
    discover_saved_target_annotation_paths,
    promote_temp_annotation_to_vtln,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Promote temporary source/VTLN-target annotation JSONs into canonical VTLN/data zips "
            "when the temp file is newer, then delete the temp JSON."
        )
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_DIR, help="Root outputs directory to scan.")
    parser.add_argument(
        "--source-vtln-dir",
        type=Path,
        default=DEFAULT_VTLN_DIR,
        help="Canonical VTLN/data directory used for source-side temp annotations.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Report planned actions without writing or deleting files.")
    return parser.parse_args(argv)


def _all_candidate_paths(output_root: Path) -> list[Path]:
    paths = {
        *discover_saved_source_annotation_paths(output_root=output_root),
        *discover_saved_target_annotation_paths(output_root=output_root),
    }
    def _mtime_key(path: Path) -> float:
        try:
            return path.stat().st_mtime
        except OSError:
            return float("-inf")
    return sorted(paths, key=_mtime_key, reverse=True)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    scanned = 0
    promoted = 0
    deleted = 0
    skipped = 0
    rows: list[dict[str, object]] = []

    for path in _all_candidate_paths(args.output_root):
        scanned += 1
        spec = build_temp_annotation_sync_spec(path, default_source_vtln_dir=args.source_vtln_dir)
        if spec is None:
            skipped += 1
            rows.append(
                {
                    "path": str(path),
                    "action": "skip",
                    "reason": "unsupported-or-unparseable",
                }
            )
            continue

        if args.dry_run:
            rows.append(
                {
                    "path": str(path),
                    "role": spec.role,
                    "reference_name": spec.reference_name,
                    "vtln_dir": str(spec.vtln_dir),
                    "action": "would-promote-or-delete",
                }
            )
            continue

        existed_before = spec.path.exists()
        was_promoted = promote_temp_annotation_to_vtln(
            path=spec.path,
            reference_name=spec.reference_name,
            vtln_dir=spec.vtln_dir,
            contours=spec.contours,
            source_shape=spec.source_shape,
            stop_root=args.output_root,
        )
        if was_promoted:
            promoted += 1
        if existed_before and not spec.path.exists():
            deleted += 1
        rows.append(
            {
                "path": str(spec.path),
                "role": spec.role,
                "reference_name": spec.reference_name,
                "vtln_dir": str(spec.vtln_dir),
                "action": "promoted" if was_promoted else "deleted-stale",
            }
        )

    payload = {
        "output_root": str(args.output_root),
        "source_vtln_dir": str(args.source_vtln_dir),
        "dry_run": bool(args.dry_run),
        "scanned": scanned,
        "promoted": promoted,
        "deleted": deleted,
        "skipped": skipped,
        "rows": rows,
    }
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
