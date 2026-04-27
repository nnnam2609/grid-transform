from __future__ import annotations

import json
import zipfile
from pathlib import Path

from grid_transform.vtln_release import build_vtln_release_bundle


def write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)


def test_build_vtln_release_bundle_creates_deterministic_asset(tmp_path: Path) -> None:
    data_dir = tmp_path / "VTLN" / "data"
    write_bytes(data_dir / "1612_example.png", b"png-data")
    write_bytes(data_dir / "1612_example.zip", b"roi-data")
    write_bytes(data_dir / "selection_manifest.csv", b"manifest-data")
    write_bytes(data_dir / "nnunet_data_80" / "case" / "frame.png", b"nnunet-data")

    manifest_first = build_vtln_release_bundle(
        version="0.1.13",
        data_dir=data_dir,
        output_dir=tmp_path / "out1",
        overwrite=True,
    )
    manifest_second = build_vtln_release_bundle(
        version="0.1.13",
        data_dir=data_dir,
        output_dir=tmp_path / "out2",
        overwrite=True,
    )

    assert manifest_first["file_count"] == 4
    assert manifest_first["png_count"] == 1
    assert manifest_first["roi_zip_count"] == 1
    assert manifest_first["nnunet_file_count"] == 1
    assert manifest_first["asset_zip_sha256"] == manifest_second["asset_zip_sha256"]

    manifest_path = Path(str(manifest_first["manifest_json"]))
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert payload["archive_root"] == "VTLN/data"
    assert payload["file_count"] == 4

    with zipfile.ZipFile(Path(str(manifest_first["asset_zip"])), "r") as archive:
        names = sorted(archive.namelist())
    assert names == [
        "VTLN/data/1612_example.png",
        "VTLN/data/1612_example.zip",
        "VTLN/data/nnunet_data_80/case/frame.png",
        "VTLN/data/selection_manifest.csv",
    ]
