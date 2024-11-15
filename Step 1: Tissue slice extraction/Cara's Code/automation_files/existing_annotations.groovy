import qupath.lib.geom.Point2

//EXPORTS ONLY EXISTING ANNOTATIONS THAT ARE OF POSITIVE CLASS
// GOOD TO REDUCE MEMORY CAPACITY FOR SOX10, MELANA

// Make output dir
def outputDir = buildFilePath(PROJECT_BASE_DIR, 'processed_images')
mkdirs(outputDir)  // Create the output directory if it doesn't exist
def imageData = getCurrentImageData()
def baseImageName = getProjectEntry().getImageName()
def server = imageData.getServer()

//UPDATED for manual extraction

// Only run if annotations do not already exist (will cause bugs otherwise)
// Then create annotations based on the pixel classifier
def existingAnnotations = getAnnotationObjects().findAll { it.getPathClass() == getPathClass("Positive") }

// STEP 3: Export all annotation Regions of interest (ROIs)
def l = 0
double downsample = 1.0
for (annotation in existingAnnotations) {
    def roi = annotation.getROI()
    //change downsample parameter here to get rid of black boxes for the corrupted images
    def requestROI = RegionRequest.createInstance(server.getPath(), downsample, roi)
    l = l + 1
    def currentImagePath = buildFilePath(outputDir, baseImageName.replaceAll(".tif - Series 0","") + '_ROI_' + l + ".tif")
    writeImageRegion(server, requestROI, currentImagePath)
}
if (l == 0) {
    println("No ROIs exported. Did you set annotation class to Positive?")
} else {
    println(l + " ROIs identified and exported")
}