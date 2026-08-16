from __future__ import annotations

"""Promote the validated v0.1.18 S5 pourri #2 candidate transactionally."""

import argparse
import csv
import hashlib
import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path

from grid_transform.config import DEFAULT_VTLN_DIR, PROJECT_DIR


VERSION = "0.1.18"
SPEAKERS = tuple(f"P{index}" for index in range(1, 11))
AFFINE_CONTROLS = tuple(
    [
        *(f"I{index}" for index in range(1, 8)),
        "P1",
        *(f"C{index}" for index in range(1, 7)),
    ]
)
TPS_CONTROLS = AFFINE_CONTROLS + ("M1", "L6")
MAX_TPS_CONTROL_RESIDUAL_PX = 1e-6
DEFAULT_CANDIDATE_DIR = (
    PROJECT_DIR
    / "outputs"
    / "geometry_reference_candidates"
    / f"v{VERSION}_s5_pourri2"
    / "VTLN"
    / "data"
)
DEFAULT_BACKUP_DIR = (
    PROJECT_DIR
    / "outputs"
    / "geometry_reference_backups"
    / f"v{VERSION}_before_s5_pourri2"
)
STALE_METADATA = {
    "geometry_source_build_summary.v0.1.16.json",
    "lower_incisor_update_manifest.json",
    "selection_manifest_rgb_triplets.source.csv",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-dir", type=Path, default=DEFAULT_CANDIDATE_DIR)
    parser.add_argument("--vtln-dir", type=Path, default=DEFAULT_VTLN_DIR)
    parser.add_argument("--backup-dir", type=Path, default=DEFAULT_BACKUP_DIR)
    parser.add_argument("--overwrite-backup", action="store_true")
    parser.add_argument(
        "--reuse-backup",
        action="store_true",
        help="Reuse and verify the existing pre-v0.1.18 backup when refreshing the candidate.",
    )
    parser.add_argument(
        "--restore",
        action="store_true",
        help="Restore the verified pre-promotion top-level backup and leave subdirectories intact.",
    )
    return parser


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def read_manifest(data_dir: Path) -> list[dict[str, str]]:
    path = Path(data_dir) / "selection_manifest.csv"
    with path.open("r", encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def validate_candidate(candidate_dir: Path) -> tuple[list[dict[str, str]], list[Path]]:
    rows = read_manifest(candidate_dir)
    if len(rows) != 10 or tuple(row["speaker"] for row in rows) != SPEAKERS:
        raise ValueError("Candidate manifest must contain ordered P1-P10 exactly once")
    if any(row["session"] != "S5" for row in rows):
        raise ValueError("Every v0.1.18 candidate reference must use S5")
    basenames = [row["output_basename"] for row in rows]
    expected = {
        *(f"{basename}.png" for basename in basenames),
        *(f"{basename}.zip" for basename in basenames),
        "README.md",
        "selection_manifest.csv",
        "build_summary.json",
    }
    files = sorted(path for path in Path(candidate_dir).iterdir() if path.is_file())
    actual = {path.name for path in files}
    if actual != expected:
        raise ValueError(
            f"Unexpected candidate top-level files: missing={sorted(expected - actual)}, "
            f"extra={sorted(actual - expected)}"
        )
    summary = json.loads((Path(candidate_dir) / "build_summary.json").read_text())
    if summary["status"] != "PASS" or summary["tps_nonpositive_jacobian_speakers"]:
        raise ValueError("Candidate geometry validation is not PASS")
    lower_controls = summary.get("tps_lower_incisor_shape_controls", {})
    if lower_controls.get("enabled") or lower_controls.get("labels"):
        raise ValueError("Candidate must not use M-/M+ lower-incisor controls")
    transforms = summary.get("transforms_to_p10", {})
    if tuple(transforms) != SPEAKERS:
        raise ValueError("Candidate transform audit must contain ordered P1-P10")
    for speaker, transform in transforms.items():
        if tuple(transform.get("affine_controls", ())) != AFFINE_CONTROLS:
            raise ValueError(f"{speaker}: affine controls do not match the 14-control contract")
        if tuple(transform.get("tps_controls", ())) != TPS_CONTROLS:
            raise ValueError(f"{speaker}: TPS controls do not match the 16-control contract")
        residual = float(transform.get("tps_control_residual_max_px", float("inf")))
        if residual > MAX_TPS_CONTROL_RESIDUAL_PX:
            raise ValueError(
                f"{speaker}: TPS controls do not interpolate the P10 targets: {residual}px"
            )
    return rows, files


def atomic_copy(source: Path, destination: Path) -> None:
    temporary = destination.with_name(f".{destination.name}.v018-new")
    shutil.copy2(source, temporary)
    os.replace(temporary, destination)


def promote(args: argparse.Namespace) -> dict[str, object]:
    candidate_dir = Path(args.candidate_dir).resolve()
    vtln_dir = Path(args.vtln_dir).resolve()
    backup_dir = Path(args.backup_dir).resolve()
    rows, candidate_files = validate_candidate(candidate_dir)
    current_files = sorted(path for path in vtln_dir.iterdir() if path.is_file())
    current_hashes = {path.name: sha256_file(path) for path in current_files}

    if args.reuse_backup and not backup_dir.exists():
        raise FileNotFoundError(f"Cannot reuse missing backup: {backup_dir}")
    if backup_dir.exists():
        if args.reuse_backup and args.overwrite_backup:
            raise ValueError("Use only one of --reuse-backup or --overwrite-backup")
        if args.reuse_backup:
            previous_summary_path = backup_dir / "promotion_summary.json"
            if not previous_summary_path.is_file():
                raise FileNotFoundError(previous_summary_path)
            previous_summary = json.loads(
                previous_summary_path.read_text(encoding="utf-8")
            )
            backup_source_hashes = {
                str(name): str(value)
                for name, value in previous_summary["backup_hashes"].items()
            }
            for name, expected_hash in backup_source_hashes.items():
                path = backup_dir / name
                if not path.is_file() or sha256_file(path) != expected_hash:
                    raise RuntimeError(f"Existing backup verification failed: {path}")
        elif not args.overwrite_backup:
            raise FileExistsError(f"Backup already exists: {backup_dir}")
        else:
            shutil.rmtree(backup_dir)
    if not args.reuse_backup:
        backup_dir.mkdir(parents=True)
        for path in current_files:
            shutil.copy2(path, backup_dir / path.name)
        backup_source_hashes = {
            path.name: sha256_file(path)
            for path in backup_dir.iterdir()
            if path.is_file()
        }
        if backup_source_hashes != current_hashes:
            raise RuntimeError("Backup hash verification failed; active data was not changed")

    candidate_hashes = {path.name: sha256_file(path) for path in candidate_files}
    try:
        for path in candidate_files:
            atomic_copy(path, vtln_dir / path.name)

        new_basenames = {row["output_basename"] for row in rows}
        for path in current_files:
            if (
                path.suffix.lower() in {".png", ".zip"}
                and path.stem not in new_basenames
            ) or path.name in STALE_METADATA:
                path.unlink()

        active_hashes = {
            path.name: sha256_file(vtln_dir / path.name) for path in candidate_files
        }
        if active_hashes != candidate_hashes:
            raise RuntimeError("Post-promotion candidate hash verification failed")
        active_rows = read_manifest(vtln_dir)
        if [row["output_basename"] for row in active_rows] != [
            row["output_basename"] for row in rows
        ]:
            raise RuntimeError("Post-promotion manifest verification failed")
    except Exception:
        for path in list(vtln_dir.iterdir()):
            if path.is_file():
                path.unlink()
        for path in backup_dir.iterdir():
            if path.is_file():
                shutil.copy2(path, vtln_dir / path.name)
        raise

    result = {
        "schema_version": "s5-pourri2-geometry-reference-promotion-v1",
        "version": VERSION,
        "status": "PASS",
        "promoted_at_utc": datetime.now(timezone.utc).isoformat(),
        "candidate_dir": str(candidate_dir),
        "vtln_dir": str(vtln_dir),
        "backup_dir": str(backup_dir),
        "backup_file_count": len(backup_source_hashes),
        "active_candidate_file_count": len(candidate_files),
        "canonical_speaker_count": len(rows),
        "preserved_subdirectories": sorted(
            path.name for path in vtln_dir.iterdir() if path.is_dir()
        ),
        "active_hashes": candidate_hashes,
        "backup_hashes": backup_source_hashes,
    }
    (backup_dir / "promotion_summary.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return result


def restore(args: argparse.Namespace) -> dict[str, object]:
    vtln_dir = Path(args.vtln_dir).resolve()
    backup_dir = Path(args.backup_dir).resolve()
    promotion_summary_path = backup_dir / "promotion_summary.json"
    if not promotion_summary_path.is_file():
        raise FileNotFoundError(promotion_summary_path)
    promotion = json.loads(promotion_summary_path.read_text(encoding="utf-8"))
    expected_hashes = {
        str(name): str(value)
        for name, value in promotion["backup_hashes"].items()
    }
    backup_files = {
        path.name: path
        for path in backup_dir.iterdir()
        if path.is_file() and path.name in expected_hashes
    }
    if set(backup_files) != set(expected_hashes):
        raise RuntimeError("Backup file inventory does not match promotion provenance")
    for name, path in backup_files.items():
        if sha256_file(path) != expected_hashes[name]:
            raise RuntimeError(f"Backup hash mismatch: {path}")

    current_files = sorted(path for path in vtln_dir.iterdir() if path.is_file())
    for path in current_files:
        path.unlink()
    for name, path in backup_files.items():
        atomic_copy(path, vtln_dir / name)
    restored_hashes = {
        name: sha256_file(vtln_dir / name) for name in sorted(expected_hashes)
    }
    if restored_hashes != expected_hashes:
        raise RuntimeError("Restored active hash verification failed")
    result = {
        "schema_version": "s5-pourri2-geometry-reference-restore-v1",
        "version": VERSION,
        "status": "PASS",
        "restored_at_utc": datetime.now(timezone.utc).isoformat(),
        "vtln_dir": str(vtln_dir),
        "backup_dir": str(backup_dir),
        "restored_file_count": len(restored_hashes),
        "preserved_subdirectories": sorted(
            path.name for path in vtln_dir.iterdir() if path.is_dir()
        ),
        "restored_hashes": restored_hashes,
    }
    (backup_dir / "restore_summary.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return result


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    summary = restore(args) if args.restore else promote(args)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
