requires("1.54a");
base = File.getDirectory(getInfo("macro.filepath"));
open(base + "s5_pourri2_9cases_stack.tif");
run("ROI Manager...");
roiManager("reset");
roiManager("open", base + "s5_pourri2_99_positioned_rois.zip");
RoiManager.associateROIsWithSlices(true);
RoiManager.useNamesAsLabels(true);
roiManager("show all with labels");
setSlice(1);
showMessage("S5 pourri #2 ready",
    "Use the stack slider or mouse wheel to change speaker.\n" +
    "Click Update after editing each ROI.");
