# Method Contract

This note records the current method contract used by the active package code. It is meant to be the short canonical reference when README prose, older notebooks, or generated reports use looser wording.

## Coordinate Space

- The curated VTLN bundle in `VTLN/data/` uses a shared `480x480` image space.
- RGB triplets are stored as `R=t-1, G=t, B=t+1`; grayscale analysis should use the center channel `G=t`.
- ROI zip contours are expected to already be in the same `480x480` coordinate space as the triplet image.
- External ArtSpeech session frames are separate data and are projected or resized for review workflows; they are not silently treated as canonical VTLN coordinates.

## Grid And Landmark Source

The grid is built from the current annotation contours by `grid_transform/vt_grid.py`. The transfer pipeline then extracts landmarks from that grid with `extract_true_landmarks(...)`.

Current landmark families used by the active transform helpers:

- `I1..I7`: top-axis / palate-side landmarks when available.
- `P1`: intermediate top-axis point when available.
- `C1..C6`: cervical / vertical-axis centers.
- `M1`: optional midline-related point.
- `L1` and `L6`: left-side endpoints used for reporting; `L6` is also a TPS control when available.

Do not change landmark definitions, point order, or grid construction assumptions without treating the result as a method change.

## Two-Step Transform

The active speaker-to-speaker mapping is an axis-first two-step transform implemented by `grid_transform/transfer.py` and `grid_transform/transform_helpers.py`.

### Step 1: Full Affine

Step 1 fits a full 2D affine transform by least squares:

```text
x' = A x + t
```

The affine anchor order is:

```text
I1, I2, I3, I4, I5, I6, I7, P1, C1, C2, C3, C4, C5, C6
```

Only landmarks present in both source and target are used. `C1` conceptually belongs to both the horizontal and vertical axes, but appears only once in the fit.

This stage captures global translation, rotation, scale, shear, and other full-affine effects. It should be interpreted as global alignment, not local anatomical deformation.

### Step 2: TPS Refinement

After Step 1, source landmarks are first mapped by the affine transform. A thin-plate spline is then fitted from those affine-mapped source landmarks to the target landmarks with zero smoothing:

```text
x'' = TPS(A x + t)
```

The TPS control order is:

```text
I1, I2, I3, I4, I5, I6, I7, P1, C1, C2, C3, C4, C5, C6, M1, L6
```

Only controls present after Step 1 and in the target are used. This stage captures local residual deformation after the global affine alignment. Because TPS is non-rigid, large improvements in overlap should still be checked against anatomical plausibility and grid diagnostics.

## Smoothing And Visualization

- Contour smoothing in transfer outputs is visualization/post-processing around mapped open contours; it does not redefine the source annotations.
- Stage-overlay workflows may visually close most contours by drawing a last-to-first segment for readability.
- In `stage_overlays_v2`, `pharynx` is intentionally kept open in overlays and excluded from ROI-overlap variance masks.
- ROI-overlap reports may straight-close selected contours only for binary mask construction. This is an evaluation-time mask convention, not a change to canonical contour geometry.

## Metrics Contract

Core transform metrics remain point- and grid-based:

- `spine_rms`: RMS over the mapped `C` landmark sequence.
- `L1_err`, `L6_err`, `M1_err`, `P1_err`: point errors for named landmarks when available.
- `L_2pt_rms`: RMS over `L1` and `L6`.
- `horiz_axis_rms`: RMS over available `I1..I7`, `P1`, `C1`.
- `vert_axis_rms`: RMS over available `C1..C5`.

Stage-overlay V2 uses Dice-distance variance by default:

```text
dice = 2 |A intersect B| / (|A| + |B|)
dice_distance = 1 - dice
label_variance = mean(dice_distance^2 over unordered speaker pairs)
overall_variance = mean(label_variance over labels)
```

IoU remains available only when explicitly requested with `--overlap-metric iou`.

## Comparability Rules

For comparable experiments:

- Keep the same landmark definitions and anchor order.
- Keep `init -> affine -> tps` stage meanings unchanged.
- Keep direct stage-mapped grids separate from grids rebuilt after transformation.
- Keep visualization-only closure separate from ROI mask closure and from canonical contour geometry.
- Report when a label is excluded because it is missing or treated as open geometry.
