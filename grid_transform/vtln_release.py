from __future__ import annotations

import hashlib
import json
import re
import subprocess
import zipfile
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

from grid_transform.config import DEFAULT_OUTPUT_DIR, DEFAULT_VTLN_DIR


ZIP_TIMESTAMP = (1980, 1, 1, 0, 0, 0)
ZIP_FILE_MODE = 0o100644 << 16
VERSION_PATTERN = re.compile(r"^[0-9A-Za-z][0-9A-Za-z._-]*$")


@dataclass(frozen=True)
class ReleaseFileEntry:
    relative_path: str
    archive_path: str
    bytes: int
    sha256: str


@dataclass(frozen=True)
class ReleaseBundlePaths:
    version: str
    tag: str
    title: str
    output_dir: Path
    asset_zip: Path
    manifest_json: Path
    sha256_txt: Path
    release_notes: Path


def normalize_release_version(version: str) -> str:
    value = str(version).strip()
    if value.lower().startswith("v"):
        value = value[1:]
    if not value or not VERSION_PATTERN.fullmatch(value):
        raise ValueError(
            f"Invalid release version {version!r}. Use a simple tag-like value such as '0.1.13'."
        )
    return value


def default_release_tag(version: str) -> str:
    return f"vtln-data-v{normalize_release_version(version)}"


def default_release_title(version: str) -> str:
    return f"VTLN data v{normalize_release_version(version)}"


def default_release_output_dir(version: str) -> Path:
    return DEFAULT_OUTPUT_DIR / "release_assets" / "vtln_data" / normalize_release_version(version)


def default_asset_stem(version: str) -> str:
    return f"vtln-data-{normalize_release_version(version)}"


def resolve_release_paths(
    *,
    version: str,
    output_dir: Path | None = None,
    tag: str | None = None,
    title: str | None = None,
) -> ReleaseBundlePaths:
    normalized_version = normalize_release_version(version)
    resolved_output_dir = Path(output_dir) if output_dir is not None else default_release_output_dir(normalized_version)
    stem = default_asset_stem(normalized_version)
    return ReleaseBundlePaths(
        version=normalized_version,
        tag=tag or default_release_tag(normalized_version),
        title=title or default_release_title(normalized_version),
        output_dir=resolved_output_dir,
        asset_zip=resolved_output_dir / f"{stem}.zip",
        manifest_json=resolved_output_dir / f"{stem}.manifest.json",
        sha256_txt=resolved_output_dir / f"{stem}.sha256",
        release_notes=resolved_output_dir / f"{stem}.release.md",
    )


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def collect_release_files(data_dir: Path, *, archive_root: str) -> list[ReleaseFileEntry]:
    resolved_root = Path(data_dir)
    if not resolved_root.is_dir():
        raise FileNotFoundError(f"VTLN data directory not found: {resolved_root}")
    normalized_archive_root = archive_root.strip().strip("/\\")
    if not normalized_archive_root:
        raise ValueError("archive_root must not be empty.")

    files = sorted(path for path in resolved_root.rglob("*") if path.is_file())
    if not files:
        raise FileNotFoundError(f"No files found under VTLN data directory: {resolved_root}")

    entries: list[ReleaseFileEntry] = []
    for path in files:
        relative_path = path.relative_to(resolved_root).as_posix()
        archive_path = f"{normalized_archive_root}/{relative_path}"
        entries.append(
            ReleaseFileEntry(
                relative_path=relative_path,
                archive_path=archive_path,
                bytes=path.stat().st_size,
                sha256=sha256_file(path),
            )
        )
    return entries


def write_deterministic_zip(
    *,
    data_dir: Path,
    entries: list[ReleaseFileEntry],
    asset_zip: Path,
    overwrite: bool,
) -> None:
    if asset_zip.exists() and not overwrite:
        raise FileExistsError(
            f"Release asset already exists: {asset_zip}. Use --overwrite to rebuild it."
        )
    asset_zip.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(asset_zip, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for entry in entries:
            source_path = Path(data_dir) / Path(entry.relative_path)
            payload = source_path.read_bytes()
            zip_info = zipfile.ZipInfo(entry.archive_path, date_time=ZIP_TIMESTAMP)
            zip_info.compress_type = zipfile.ZIP_DEFLATED
            zip_info.create_system = 3
            zip_info.external_attr = ZIP_FILE_MODE
            archive.writestr(zip_info, payload)


def resolve_git_head(cwd: Path | None = None) -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            cwd=cwd,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip() or None


def summarize_entry_groups(entries: list[ReleaseFileEntry]) -> dict[str, int]:
    png_count = 0
    roi_zip_count = 0
    nnunet_file_count = 0
    metadata_file_count = 0
    for entry in entries:
        if entry.relative_path.startswith("nnunet_data_80/"):
            nnunet_file_count += 1
        elif entry.relative_path.endswith(".png"):
            png_count += 1
        elif entry.relative_path.endswith(".zip"):
            roi_zip_count += 1
        else:
            metadata_file_count += 1
    return {
        "png_count": png_count,
        "roi_zip_count": roi_zip_count,
        "nnunet_file_count": nnunet_file_count,
        "metadata_file_count": metadata_file_count,
    }


def build_release_notes(
    *,
    version: str,
    tag: str,
    title: str,
    archive_root: str,
    built_at_utc: str,
    built_from_commit: str | None,
    entries: list[ReleaseFileEntry],
    asset_zip: Path,
    zip_sha256: str,
) -> str:
    counts = summarize_entry_groups(entries)
    commit_line = built_from_commit or "unknown local commit"
    return "\n".join(
        [
            f"# {title}",
            "",
            "Versioned shared release bundle for the canonical `VTLN/data` tree.",
            "",
            "## Contents",
            f"- archive root: `{archive_root}`",
            f"- curated PNG references: `{counts['png_count']}`",
            f"- ROI annotation zips: `{counts['roi_zip_count']}`",
            f"- bundled nnUNet target files: `{counts['nnunet_file_count']}`",
            f"- metadata/support files: `{counts['metadata_file_count']}`",
            f"- total files: `{len(entries)}`",
            "",
            "## Install",
            "1. Download the zip asset for this release.",
            "2. Extract it at the repository root so the bundle lands under `VTLN/data/`.",
            "3. Keep existing commands pointed at `--vtln-dir VTLN/data` or the extracted equivalent.",
            "",
            "## Integrity",
            f"- asset: `{asset_zip.name}`",
            f"- sha256: `{zip_sha256}`",
            f"- manifest: `{asset_zip.with_suffix('.manifest.json').name}`",
            "",
            "## Provenance",
            f"- tag: `{tag}`",
            f"- built at UTC: `{built_at_utc}`",
            f"- built from commit: `{commit_line}`",
        ]
    ) + "\n"


def build_vtln_release_bundle(
    *,
    version: str,
    data_dir: Path = DEFAULT_VTLN_DIR,
    output_dir: Path | None = None,
    archive_root: str = "VTLN/data",
    tag: str | None = None,
    title: str | None = None,
    overwrite: bool = False,
    repo_root: Path | None = None,
) -> dict[str, object]:
    paths = resolve_release_paths(version=version, output_dir=output_dir, tag=tag, title=title)
    entries = collect_release_files(data_dir, archive_root=archive_root)
    write_deterministic_zip(
        data_dir=Path(data_dir),
        entries=entries,
        asset_zip=paths.asset_zip,
        overwrite=overwrite,
    )

    zip_sha256 = sha256_file(paths.asset_zip)
    built_at_utc = datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    built_from_commit = resolve_git_head(repo_root)
    summary_groups = summarize_entry_groups(entries)
    manifest = {
        "bundle_name": "vtln-data",
        "version": paths.version,
        "tag": paths.tag,
        "title": paths.title,
        "archive_root": archive_root.strip().strip("/\\"),
        "source_dir": str(Path(data_dir).resolve()),
        "output_dir": str(paths.output_dir.resolve()),
        "asset_zip": str(paths.asset_zip.resolve()),
        "asset_zip_name": paths.asset_zip.name,
        "asset_zip_sha256": zip_sha256,
        "asset_zip_bytes": paths.asset_zip.stat().st_size,
        "built_at_utc": built_at_utc,
        "built_from_commit": built_from_commit,
        "file_count": len(entries),
        "total_uncompressed_bytes": sum(entry.bytes for entry in entries),
        **summary_groups,
        "files": [asdict(entry) for entry in entries],
    }
    paths.output_dir.mkdir(parents=True, exist_ok=True)
    paths.manifest_json.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    paths.sha256_txt.write_text(f"{zip_sha256} *{paths.asset_zip.name}\n", encoding="utf-8")
    paths.release_notes.write_text(
        build_release_notes(
            version=paths.version,
            tag=paths.tag,
            title=paths.title,
            archive_root=manifest["archive_root"],
            built_at_utc=built_at_utc,
            built_from_commit=built_from_commit,
            entries=entries,
            asset_zip=paths.asset_zip,
            zip_sha256=zip_sha256,
        ),
        encoding="utf-8",
    )
    manifest["manifest_json"] = str(paths.manifest_json.resolve())
    manifest["sha256_txt"] = str(paths.sha256_txt.resolve())
    manifest["release_notes"] = str(paths.release_notes.resolve())
    return manifest


def run_command(command: list[str], *, cwd: Path | None = None) -> None:
    subprocess.run(command, cwd=cwd, check=True)


def github_release_exists(tag: str, *, cwd: Path | None = None) -> bool:
    result = subprocess.run(
        ["gh", "release", "view", tag],
        cwd=cwd,
        capture_output=True,
        text=True,
    )
    return result.returncode == 0


def publish_vtln_release_bundle(
    manifest: dict[str, object],
    *,
    target: str = "main",
    draft: bool = False,
    prerelease: bool = False,
    cwd: Path | None = None,
) -> None:
    asset_zip = Path(str(manifest["asset_zip"]))
    manifest_json = Path(str(manifest["manifest_json"]))
    sha256_txt = Path(str(manifest["sha256_txt"]))
    release_notes = Path(str(manifest["release_notes"]))
    tag = str(manifest["tag"])
    title = str(manifest["title"])

    run_command(["gh", "auth", "status"], cwd=cwd)
    if github_release_exists(tag, cwd=cwd):
        run_command(
            [
                "gh",
                "release",
                "upload",
                tag,
                str(asset_zip),
                str(manifest_json),
                str(sha256_txt),
                "--clobber",
            ],
            cwd=cwd,
        )
        run_command(
            ["gh", "release", "edit", tag, "--title", title, "--notes-file", str(release_notes)],
            cwd=cwd,
        )
        return

    command = [
        "gh",
        "release",
        "create",
        tag,
        str(asset_zip),
        str(manifest_json),
        str(sha256_txt),
        "--title",
        title,
        "--notes-file",
        str(release_notes),
        "--target",
        target,
    ]
    if draft:
        command.append("--draft")
    if prerelease:
        command.append("--prerelease")
    run_command(command, cwd=cwd)
