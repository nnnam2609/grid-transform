# ImageJ nine-case stack

Run `Open_S5_Pourri2_Batch.ijm` from ImageJ with `Plugins > Macros > Run...`.
The macro opens one nine-slice MRI stack and 99 open polyline ROIs. Each ROI is
associated with exactly one slice, so moving the stack slider changes speaker
and displays only the matching eleven contours.

Before selecting another ROI, click `Update` in ROI Manager to retain the
current vertex edits. Do not add/delete vertices, close paths, resize images,
or use Flatten. Each contour must remain an open 50-point polyline.

After all edits, run `Save_S5_Pourri2_Edited.ijm`. It writes
`s5_pourri2_99_positioned_rois_edited.zip` without replacing the baseline ROI
zip. The edited zip still requires validated import before NPY geometry and
downstream affine/TPS/P2CP results are updated.
