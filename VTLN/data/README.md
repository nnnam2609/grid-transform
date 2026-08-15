# VTLN/data

Canonical bundle for the current pipeline.

- `*.png`: 480x480 RGB triplets with channel order `R=t-1, G=t, B=t+1`.
- `*.zip`: ROI contours scaled into the same 480x480 coordinate space.
- `nnunet_data_80/`: bundled target MRI image case and groundtruth contours used by the apps.
- versioned shared release assets can be generated from this folder with `scripts/run/run_build_vtln_release_bundle.py --version <x.y.z>`.
- `lower_incisor_update_manifest.json`: provenance and hashes for the 2026-08-16 speaker-specific lower-incisor geometry update. P2 remains unchanged because no corrected prototype exists.
- `selection_manifest.csv` and `build_summary.json` are build-time provenance metadata. They may contain absolute local source paths for traceability; use the local files in this folder as the portable bundle contract.

All grayscale computations should use the center channel `G=t`.
The legacy folder/tag prefix `VTLN`/`vtln-data-*` is retained for compatibility; the release display name is `Grid Transform Geometry Reference Data`.
