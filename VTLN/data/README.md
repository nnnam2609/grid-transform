# GTGRD v0.1.20

GTGRD means Grid Transform Geometry Reference Data. `VTLN/data` remains the
runtime compatibility path.

- The authoritative final ImageJ `RoiSet.zip` edits are imported for ASD2, P7,
  P8, P9, and P10. P1-P6 remain unchanged from v0.1.19.
- P10/S5/F0859 is stored in the accepted stage-1 registered coordinate system
  by default. Its RGB MRI triplet, all reviewed contours, `vocal-folds`, and
  C1-C6 share that coordinate system. Downstream code must not apply the rigid
  registration a second time.
- ASD2 is `1791/S29/F2180`, the exact `/u/` in `cou`. Its registered MRI
  triplet is F2179-F2181. The 16 edited ROIs come from the per-slice
  `RoiSet.zip`; `vocal-folds` is retained from the matching F2180 full17 set.
- P7-P10 use their final per-slice ImageJ review coordinates. Editing packages
  omit `vocal-folds`, so the matching full17 target value is preserved.
- Dynamic contours are open polylines; C1-C6 are stored as closed FREEHAND
  contours. Historical canonical names `incisior-hard-palate` and
  `mandible-incisior` remain unchanged.
- No smoothing, reordering, landmark redefinition, metric change, affine/TPS
  protocol change, or imputation is introduced. P2 still lacks `vocal-folds`.
