from __future__ import annotations

import argparse
import json
from pathlib import Path

from grid_transform.config import DEFAULT_VTLN_DIR, PROJECT_DIR
from grid_transform.vtln_release import build_vtln_release_bundle, publish_vtln_release_bundle


DEFAULT_TARGET = "main"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a versioned shared VTLN/data release bundle with a deterministic zip asset, "
            "manifest, checksum, and optional GitHub release publication."
        )
    )
    parser.add_argument(
        "--version",
        required=True,
        help="Release version for the data bundle, for example 0.1.13.",
    )
    parser.add_argument(
        "--vtln-dir",
        type=Path,
        default=DEFAULT_VTLN_DIR,
        help="Canonical VTLN data directory to package.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Optional artifact output directory. Defaults to outputs/release_assets/vtln_data/<version>/.",
    )
    parser.add_argument(
        "--archive-root",
        default="VTLN/data",
        help="Root path stored inside the zip archive. Default keeps a drop-in VTLN/data layout.",
    )
    parser.add_argument(
        "--tag",
        default=None,
        help="Optional GitHub release tag. Defaults to vtln-data-v<version>.",
    )
    parser.add_argument(
        "--title",
        default=None,
        help="Optional GitHub release title. Defaults to 'VTLN data v<version>'.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite any existing local artifact files for the requested version.",
    )
    parser.add_argument(
        "--publish",
        action="store_true",
        help="Publish the built assets to GitHub Releases with gh.",
    )
    parser.add_argument(
        "--target",
        default=DEFAULT_TARGET,
        help="Remote branch or commit used when creating a new GitHub release tag.",
    )
    parser.add_argument(
        "--draft",
        action="store_true",
        help="Create the GitHub release as a draft when --publish is used.",
    )
    parser.add_argument(
        "--prerelease",
        action="store_true",
        help="Mark the GitHub release as a prerelease when --publish is used.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    manifest = build_vtln_release_bundle(
        version=args.version,
        data_dir=args.vtln_dir,
        output_dir=args.output_dir,
        archive_root=args.archive_root,
        tag=args.tag,
        title=args.title,
        overwrite=args.overwrite,
        repo_root=PROJECT_DIR,
    )
    if args.publish:
        publish_vtln_release_bundle(
            manifest,
            target=args.target,
            draft=args.draft,
            prerelease=args.prerelease,
            cwd=PROJECT_DIR,
        )
        manifest["published"] = True
    else:
        manifest["published"] = False
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
