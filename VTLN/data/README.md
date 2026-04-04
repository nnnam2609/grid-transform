# VTLN/data

Canonical bundle for the current pipeline.

- `*.png`: 480x480 RGB triplets with channel order `R=t-1, G=t, B=t+1`.
- `*.zip`: ROI contours scaled into the same 480x480 coordinate space.
- `nnunet_data_80/`: bundled target MRI image case and groundtruth contours used by the apps.

All grayscale computations should use the center channel `G=t`.
