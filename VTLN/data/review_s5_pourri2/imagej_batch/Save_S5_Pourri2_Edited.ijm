requires("1.54a");
base = File.getDirectory(getInfo("macro.filepath"));
if (RoiManager.size != 99)
    exit("Expected 99 ROIs in ROI Manager.");
roiManager("select all");
roiManager("save", base + "s5_pourri2_99_positioned_rois_edited.zip");
roiManager("deselect");
showMessage("Saved",
    "Saved s5_pourri2_99_positioned_rois_edited.zip");
