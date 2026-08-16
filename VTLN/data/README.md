# GTGRD v0.1.18

GTGRD means Grid Transform Geometry Reference Data. `VTLN/data` remains the
runtime compatibility path.

- Ten top-level references P1-P10 use the midpoint frame of S5 `pourri #2 /u/`.
- RGB triplets are `R=t-1, G=t, B=t+1`, cropped from the matching review AVI and resized to 480x480.
- Dynamic contours are the latest available native 136x136 inference annotations, scaled directly to 480x480.
- For P1 and P3-P10, `lower-incisor` is the exact manually reviewed ImageJ prototype from the S5 pourri #2 review workspace. This deliberately takes precedence over the later all-frame propagated copy, which preserves placement but not the exact reviewed 50-point prototype.
- P2 has no manually reviewed lower-incisor prototype, so its observed inference contour is retained.
- `upper-incisor` and `lower-incisor` are stored under the historical canonical labels `incisior-hard-palate` and `mandible-incisior`.
- C1-C6 remain the fixed speaker-specific auxiliary contours from the preceding canonical geometry release.
- P2 has ten observed dynamic contours; `vocal-folds` is absent at its selected interval and is not imputed.
- All dynamic ROIs are stored as open ImageJ polylines; C1-C6 retain closed FREEHAND topology.
