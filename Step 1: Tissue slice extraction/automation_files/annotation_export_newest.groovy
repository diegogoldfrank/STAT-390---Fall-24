import qupath.lib.geom.Point2

// Make output dir
def outputDir = buildFilePath(PROJECT_BASE_DIR, 'processed_images')
mkdirs(outputDir)  // Create the output directory if it doesn't exist
def imageData = getCurrentImageData()
def baseImageName = getProjectEntry().getImageName()
def server = imageData.getServer()

// STEP 0: Make sure this is not a mask
if (baseImageName.toLowerCase().contains("mask")) {
    print("Skipping image: " + baseImageName + " (contains 'mask')")
    return
}

// STEP 1: image processing
// Stain deconvolution
setImageType('BRIGHTFIELD_H_E');
setColorDeconvolutionStains('{"Name" : "H&E default", "Stain 1" : "Hematoxylin", "Values 1" : "0.65111 0.70119 0.29049", "Stain 2" : "Eosin", "Values 2" : "0.2159 0.8012 0.5581", "Background" : " 255 255 255"}');
print("Stain Deconvoluted")

// Only run if annotations do not already exist (will cause bugs otherwise)
// Then create annotations based on the pixel classifier
def existingAnnotations = getAnnotationObjects().findAll { it.getPathClass() == getPathClass("Positive") }
if (existingAnnotations.isEmpty()) {
    createAnnotationsFromPixelClassifier("tissues_2", 50000.0, 3000.0, "", "SPLIT", "DELETE_EXISTING", "SELECT_NEW")
} else {
    print("Using existing annotations")
}

// STEP 2: Merge tissue samples that are close together into one annotation object
// Get all annotations
def annotations = getAnnotationObjects().findAll{it.getPathClass() == getPathClass("Positive")}

// Define the threshold for merging annotations
double distanceThreshold = 5000.0  // if tissue annotations are 5000 micrometers togther

// Calculate the Euclidean distance between centroids (middle points) of two annotations
def calculateDistance(roi1, roi2) {
    def centroid1 = new Point2(roi1.getCentroidX(), roi1.getCentroidY())
    def centroid2 = new Point2(roi2.getCentroidX(), roi2.getCentroidY())
    return centroid1.distance(centroid2)
}

// Merge annotations if within distance threshold (keep tissue samples together)
for (int i = 0; i < annotations.size(); i++) {
    def annotation1 = annotations[i]
    for (int j = i + 1; j < annotations.size(); j++) {
        def annotation2 = annotations[j]
        def distance = calculateDistance(annotation1.getROI(), annotation2.getROI())
        print("Distance between annotation " + i + " and annotation " + j + ": " + distance)

        if (distance < distanceThreshold) {
            print("Merging annotations: Distance = " + distance)
            
            // Select the two annotations for merging
            selectObjects([annotation1, annotation2])
            
            // Perform the merge
            mergeSelectedAnnotations()
            
            // Update annotations list after merge
            annotations = getAnnotationObjects().findAll{it.getPathClass() == getPathClass("Positive")}
        }
    }
}

// Remove duplicate annotations after merging
// Define a threshold for how close centroids can be before considering them duplicates
double duplicateThreshold = 50.0

// List to store duplicates for removal
def toRemove = []

// Check for duplicates
for (int i = 0; i < annotations.size(); i++) {
    def annotation1 = annotations[i]
    for (int j = i + 1; j < annotations.size(); j++) {
        def annotation2 = annotations[j]
        def distance = calculateDistance(annotation1.getROI(), annotation2.getROI())
        
        // If the centroids are close enough, mark one for removal
        if (distance < duplicateThreshold) {
            print("Duplicate found, removing one of the annotations")
            toRemove << annotation2  // Add the duplicate to the list for removal
        }
    }
}

// Remove duplicate merged annotations from the image
removeObjects(toRemove, true)

print("Duplicate annotations removed")

// STEP 3: Export all annotation Regions of interest (ROIs)
def l = 0
double downsample = 1.0
for (annotation in annotations) {
    def roi = annotation.getROI()
    //change downsample parameter here to get rid of black boxes for the corrupted images
    def requestROI = RegionRequest.createInstance(server.getPath(), downsample, roi)
    l = l + 1
    def currentImagePath = buildFilePath(outputDir, baseImageName.replaceAll(".tif - Series 0","") + '_ROI_' + l + ".tif")
    writeImageRegion(server, requestROI, currentImagePath)
}
print(l + ' ROIs identified and exported')  
