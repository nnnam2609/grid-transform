"""Update only the canonical lower-incisor ROI from corrected inference contours."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import zipfile

import numpy as np
from roifile import ImagejRoi, ROI_TYPE

from grid_transform.config import DEFAULT_VTLN_DIR, PROJECT_DIR


NATIVE_SHAPE = (136, 136)
REFERENCE_SHAPE = (480, 480)
LOWER_MEMBER_TOKEN = "mandible-incisior"
CASES = (
    ("P1", "S16", 952, "1612_P1_S16_F0952"),
    ("P2", "S9", 1478, "1617_P2_S9_F1478"),
    ("P3", "S14", 1556, "1618_P3_S14_F1556"),
    ("P4", "S4", 196, "1628_P4_S4_F0196"),
    ("P5", "S6", 324, "1635_P5_S6_F0324"),
    ("P6", "S8", 138, "1638_P6_S8_F0138"),
    ("P7", "S2", 829, "1640_P7_S2_F0829"),
    ("P8", "S2", 159, "1653_P8_S2_F0159"),
    ("P9", "S5", 196, "1659_P9_S5_F0196"),
    ("P10", "S14", 110, "1662_P10_S14_F0110"),
)
DEFAULT_INFERENCE_ROOT = PROJECT_DIR.parent / "Preprocess" / "inference"
DEFAULT_BACKUP_ROOT = PROJECT_DIR / "outputs" / "geometry_reference_backups"
DEFAULT_PROVENANCE = DEFAULT_VTLN_DIR / "lower_incisor_update_manifest.json"


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def default_run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_source(path: Path) -> np.ndarray:
    points = np.load(path, allow_pickle=False)
    if points.shape != (50, 2):
        raise ValueError(f"Invalid lower-incisor shape in {path}: {points.shape}")
    points = np.asarray(points, dtype=np.float64)
    if not bool(np.isfinite(points).all()):
        raise ValueError(f"Non-finite lower-incisor points in {path}")
    return points


def scaled_reference_points(points: np.ndarray) -> np.ndarray:
    scale = np.array(
        [REFERENCE_SHAPE[1] / NATIVE_SHAPE[1], REFERENCE_SHAPE[0] / NATIVE_SHAPE[0]],
        dtype=np.float64,
    )
    result = np.asarray(points, dtype=np.float64) * scale
    if np.any(result < 0.0) or np.any(result[:, 0] > REFERENCE_SHAPE[1]) or np.any(result[:, 1] > REFERENCE_SHAPE[0]):
        raise ValueError("Scaled lower-incisor lies outside the 480x480 reference canvas")
    return result


def find_lower_member(names: list[str], basename: str) -> str:
    matches = [
        name
        for name in names
        if name.lower().endswith(".roi") and LOWER_MEMBER_TOKEN in Path(name).stem.lower()
    ]
    expected = f"{basename}_{LOWER_MEMBER_TOKEN}.roi"
    if matches != [expected]:
        raise ValueError(f"Expected exactly {expected!r}, found {matches}")
    return expected


def roi_payload(points: np.ndarray, member_name: str) -> bytes:
    roi = ImagejRoi.frompoints(points, name=Path(member_name).stem)
    if roi.roitype != ROI_TYPE.FREEHAND:
        raise RuntimeError(f"Expected closed FREEHAND ROI, got {roi.roitype}")
    return roi.tobytes()


def replace_lower_member(zip_path: Path, points: np.ndarray, *, backup_path: Path, run_id: str) -> dict[str, object]:
    if backup_path.exists():
        raise FileExistsError(f"Backup already exists: {backup_path}")
    with zipfile.ZipFile(zip_path, "r") as source:
        infos = source.infolist()
        names = [info.filename for info in infos]
        lower_name = find_lower_member(names, zip_path.stem)
        payloads = {info.filename: source.read(info.filename) for info in infos}
    before_zip_sha = sha256_file(zip_path)
    before_member_hashes = {name: sha256_bytes(payload) for name, payload in payloads.items()}
    old_roi = ImagejRoi.frombytes(payloads[lower_name])
    new_payload = roi_payload(points, lower_name)
    if new_payload == payloads[lower_name]:
        raise ValueError(f"Refusing unchanged lower-incisor ROI: {zip_path}")

    backup_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(zip_path, backup_path)
    if sha256_file(backup_path) != before_zip_sha:
        raise RuntimeError(f"Backup checksum mismatch: {backup_path}")

    temp = zip_path.with_name(f".{zip_path.name}.{run_id}.tmp")
    try:
        with zipfile.ZipFile(temp, "w") as destination:
            for info in infos:
                payload = new_payload if info.filename == lower_name else payloads[info.filename]
                destination.writestr(info, payload)
        os.replace(temp, zip_path)
    finally:
        temp.unlink(missing_ok=True)

    with zipfile.ZipFile(zip_path, "r") as updated:
        updated_names = updated.namelist()
        if updated_names != names:
            raise RuntimeError(f"ZIP member order/name changed: {zip_path}")
        after_payloads = {name: updated.read(name) for name in updated_names}
    changed = [name for name in names if before_member_hashes[name] != sha256_bytes(after_payloads[name])]
    if changed != [lower_name]:
        shutil.copy2(backup_path, zip_path)
        raise RuntimeError(f"Unexpected changed ZIP members for {zip_path}: {changed}")
    updated_roi = ImagejRoi.frombytes(after_payloads[lower_name])
    updated_points = np.asarray(updated_roi.coordinates(), dtype=np.float64)
    if updated_roi.roitype != ROI_TYPE.FREEHAND or updated_points.shape != (50, 2):
        shutil.copy2(backup_path, zip_path)
        raise RuntimeError(f"Invalid updated lower-incisor ROI in {zip_path}")
    encoding_rms = float(np.sqrt(np.mean(np.sum((updated_points - points) ** 2, axis=1))))
    if encoding_rms > 1e-3:
        shutil.copy2(backup_path, zip_path)
        raise RuntimeError(f"ROI encoding error too large for {zip_path}: {encoding_rms}")
    return {
        "status": "PASS",
        "zip": zip_path.name,
        "member": lower_name,
        "old_roi_type": old_roi.roitype.name,
        "old_point_count": int(len(old_roi.coordinates())),
        "new_roi_type": updated_roi.roitype.name,
        "new_point_count": int(len(updated_points)),
        "encoding_rms_px_480": encoding_rms,
        "zip_sha256_before": before_zip_sha,
        "zip_sha256_after": sha256_file(zip_path),
        "lower_roi_sha256_before": before_member_hashes[lower_name],
        "lower_roi_sha256_after": sha256_bytes(after_payloads[lower_name]),
        "unchanged_member_count": len(names) - 1,
        "backup": str(backup_path),
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inference-root", type=Path, default=DEFAULT_INFERENCE_ROOT)
    parser.add_argument("--vtln-dir", type=Path, default=DEFAULT_VTLN_DIR)
    parser.add_argument("--backup-root", type=Path, default=DEFAULT_BACKUP_ROOT)
    parser.add_argument("--provenance", type=Path, default=DEFAULT_PROVENANCE)
    parser.add_argument("--run-id")
    parser.add_argument("--apply", action="store_true", help="Back up and replace the nine canonical lower-incisor ROIs.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    inference_root = args.inference_root.resolve()
    vtln_dir = args.vtln_dir.resolve()
    run_id = args.run_id or default_run_id()
    backup_dir = args.backup_root.resolve() / run_id
    rows: list[dict[str, object]] = []
    errors: list[dict[str, str]] = []
    p2_sha_before = sha256_file(vtln_dir / "1617_P2_S9_F1478.zip")
    for speaker, session, frame, basename in CASES:
        zip_path = vtln_dir / f"{basename}.zip"
        if speaker == "P2":
            rows.append(
                {
                    "speaker": speaker,
                    "session": session,
                    "frame": frame,
                    "basename": basename,
                    "status": "SKIPPED_NO_PROTOTYPE",
                    "zip_sha256": p2_sha_before,
                }
            )
            continue
        source_path = inference_root / speaker / session / "contours" / f"{frame:04d}_lower-incisor.npy"
        try:
            source_points = load_source(source_path)
            points_480 = scaled_reference_points(source_points)
            row: dict[str, object] = {
                "speaker": speaker,
                "session": session,
                "frame": frame,
                "basename": basename,
                "source_relative_to_inference": source_path.relative_to(inference_root).as_posix(),
                "source_sha256": sha256_file(source_path),
                "source_shape": list(NATIVE_SHAPE),
                "reference_shape": list(REFERENCE_SHAPE),
                "scale_xy": [REFERENCE_SHAPE[1] / NATIVE_SHAPE[1], REFERENCE_SHAPE[0] / NATIVE_SHAPE[0]],
            }
            if args.apply:
                row.update(
                    replace_lower_member(
                        zip_path,
                        points_480,
                        backup_path=backup_dir / zip_path.name,
                        run_id=run_id,
                    )
                )
            else:
                with zipfile.ZipFile(zip_path) as archive:
                    lower_name = find_lower_member(archive.namelist(), basename)
                    old_roi = ImagejRoi.frombytes(archive.read(lower_name))
                row.update(
                    {
                        "status": "DRY_RUN",
                        "member": lower_name,
                        "old_roi_type": old_roi.roitype.name,
                        "old_point_count": int(len(old_roi.coordinates())),
                        "new_roi_type": "FREEHAND",
                        "new_point_count": 50,
                    }
                )
            rows.append(row)
        except Exception as exc:
            errors.append({"speaker": speaker, "basename": basename, "error": str(exc)})

    p2_sha_after = sha256_file(vtln_dir / "1617_P2_S9_F1478.zip")
    if p2_sha_after != p2_sha_before:
        errors.append({"speaker": "P2", "basename": "1617_P2_S9_F1478", "error": "P2 ZIP changed unexpectedly"})
    expected_pass = 9
    passed = sum(row.get("status") == ("PASS" if args.apply else "DRY_RUN") for row in rows)
    summary = {
        "schema_version": 1,
        "created_at_utc": utc_now(),
        "run_id": run_id,
        "status": "PASS" if passed == expected_pass and not errors else "PARTIAL",
        "applied": bool(args.apply),
        "scope": "canonical geometry-reference lower-incisor ROI only",
        "coordinate_mapping": "(x480,y480)=(x136,y136)*(480/136)",
        "topology": "closed ImageJ FREEHAND ROI",
        "p2_policy": "unchanged because no corrected prototype exists",
        "updated_count": passed,
        "rows": rows,
        "errors": errors,
    }
    if args.apply:
        args.provenance.parent.mkdir(parents=True, exist_ok=True)
        args.provenance.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0 if summary["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
