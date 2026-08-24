# GTGRD v0.1.19

GTGRD means Grid Transform Geometry Reference Data. `VTLN/data` remains the
runtime compatibility path.

- The ten ASD1 references P1-P10 are byte-identical to GTGRD v0.1.18 and use the midpoint frame of S5 `pourri #2 /u/`.
- ASD2 adds `1791/S29/F2812`, the user-selected exact `/u/` shape reference with the final ImageJ-reviewed `RoiSet_update.zip` geometry.
- RGB triplets use `R=t-1, G=t, B=t+1` and are 480x480. ASD1 triplets retain their original review-AVI conversion. ASD2 uses registered uint16 MRI frames F2811-F2813, one shared linear min/max conversion to uint8, and cubic resize from 136x136 to 480x480.
- ASD2 contains 11 dynamic contours plus C1-C6. The current server-`bf` incisors and final manually updated cervical contours are retained exactly before the common 136-to-480 resize.
- Dynamic contours are stored as open ImageJ polylines; C1-C6 retain FREEHAND topology.
- Historical labels `incisior-hard-palate` and `mandible-incisior` remain the canonical stored names for upper and lower incisors.
- P2 still has ten observed dynamic contours; `vocal-folds` is absent and is not imputed.
- GTGRD v0.1.19 changes data only. It does not change affine/TPS controls, transform ordering, landmark definitions, or P2CP metrics.
