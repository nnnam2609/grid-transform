# S5 pourri #2 contour review workspace

This is an additive manual-review workspace. It is **not** a replacement for
the ten canonical VTLN reference cases or `selection_manifest.csv`.

## Data contract

- cases: P1 and P3-P10; P2 remains excluded
- phone/frame: sole `/u/` midpoint in `pourri` repetition 2
- image: display MRI crop in native `136 x 136`
- contours: eleven open, ordered, finite `(50, 2)` arrays in native `(x,y)`
- P10 alignment provenance: `old_textgrid_fallback`

Each folder under `cases/` contains:

- `mri_native_136.png`: ImageJ/Napari background
- `preview_overlay.png`: read-only visual check
- `imagej_rois.zip`: eleven open ImageJ polyline ROIs
- `contours_npy/*.npy`: exact source arrays for Napari/reproducibility
- `metadata.json`: source hashes and selection provenance

## ImageJ

1. Open `mri_native_136.png` without resizing it.
2. Open ROI Manager, then load `imagej_rois.zip`.
3. Select one ROI, drag existing vertices, then click `Update` in ROI Manager
   before selecting another ROI. Keep every path open and keep its 50 vertices.
4. Save the complete edited ROI set as `imagej_rois_edited.zip` in the same
   case folder; keep `imagej_rois.zip` as the recoverable baseline.

ImageJ edits do not automatically update `contours_npy/`. Keep the edited zip
for a validated import step before rebuilding a later release.

## Napari

Install Napari in a separate environment if needed:

```powershell
python -m pip install "napari[all]" numpy pillow roifile
```

From this folder:

```powershell
python .\napari_edit.py --case P1_S5_F0970
```

All contours are open Path layers. Napari uses `(row,column)`, while the stored
files use `(x,y)`; the script converts both directions. Press `Ctrl+Shift+S` to
save. Saving requires exactly one finite 50-point path per contour, creates a
timestamped backup, atomically updates NPY files, rebuilds `imagej_rois.zip`,
and appends `edits.jsonl`.

After manual edits, rebuild the release so manifests and hashes describe the
edited snapshot. Do not compare prior affine/TPS/P2CP results without rerunning
them from the edited contours.
