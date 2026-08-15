requires("1.54a");
base = File.getDirectory(getInfo("macro.filepath"));
open(base + "s5_pourri2_9cases_stack.tif");
run("ROI Manager...");
roiManager("reset");
roiManager("open", base + "s5_pourri2_99_positioned_rois.zip");
RoiManager.associateROIsWithSlices(true);
if (nSlices != 9)
    exit("Expected 9 stack slices, got " + nSlices);
if (RoiManager.size != 99)
    exit("Expected 99 ROIs, got " + RoiManager.size);
for (z = 1; z <= 9; z++) {
    setSlice(z);
    RoiManager.selectPosition(0, z, 0);
    if (RoiManager.selected != 11)
        exit("Slice " + z + " has " + RoiManager.selected + " ROIs");
}
print("IMAGEJ_BATCH_PASS slices=9 rois=99 rois_per_slice=11");
