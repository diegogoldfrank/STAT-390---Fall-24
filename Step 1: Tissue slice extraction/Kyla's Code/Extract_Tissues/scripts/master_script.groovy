/// run on the project to run on every image loaded into the project. 

// Create annotations from a pixel classifier
//def imageName = imageData.getServer().getMetadata().getName()
def imageName = getCurrentImageData().getServer().getMetadata().getName()

// Define paths to classifiers
def heClassifier = "Users/kylabruno/Downloads/Stat_390/Extract_Tissues/classifiers/pixel_classifiers/find_tissue_h&e.json"
def melanClassifier = "Users/kylabruno/Downloads/Stat_390/Extract_Tissues/classifiers/pixel_classifiers/find_tissue_melan.json"
def sox10Classifier = "Users/kylabruno/Downloads/Stat_390/Extract_Tissues/classifiers/pixel_classifiers/find_tissue_sox10.json"

// Define area parameters for annotations
double minArea = 10000  // at least 1000 um to be kept as annotation
//double maxArea = 1000000000000 // Adjust the maximum area as needed
double minHoleArea = 1.0

// from CARA
if (imageName.toLowerCase().contains("mask")) {
    print("Skipping image: " + baseImageName + " (contains 'mask')")
    return
}

// Identify the correct pixel classifier using standardized name
if (imageName.toLowerCase().contains("h&e")) {
        print("Applying H&E classifier to " + imageName)
        
         // Clear any existing objects (so dont have to reload when testing)
        clearAllObjects()
        
        // Create Objects
        createAnnotationsFromPixelClassifier(heClassifier, minArea, minHoleArea) //"SPLIT"
 
    } else if (imageName.toLowerCase().contains("melan")) {
        print("Applying Melan classifier to " + imageName)
        
        clearAllObjects()
        createAnnotationsFromPixelClassifier(melanClassifier, minArea, minHoleArea)
        
       
    } else if (imageName.toLowerCase().contains("sox10")) {
        print("Applying Sox10 classifier to " + imageName)
        
        clearAllObjects()
        createAnnotationsFromPixelClassifier(sox10Classifier, minArea, minHoleArea)
 
    }
      
      
import qupath.lib.objects.PathObjects
import qupath.lib.roi.RoiTools

// Get all annotation objects
def annotations = getAnnotationObjects()

// Loop through each annotation and split based on criteria (e.g., convexity)
annotations.each { annotation ->
    def roi = annotation.getROI()
    
    // Split the annotation by identifying the individual parts
    def splitROIs = RoiTools.splitROI(roi)
    
    // Create separate annotation objects for each split region
    splitROIs.each { subROI ->
        def subAnnotation = PathObjects.createAnnotationObject(subROI)
        addObject(subAnnotation)
    }
    
    // Remove the original merged annotation if needed
    removeObject(annotation, true)
}


// Define the desired class (create it if it doesn't already exist)
def tissueClass = getPathClass("Tissue")

// Get all annotations in the image
def annotations_2 = getAnnotationObjects()

// Set each annotation's class to "Tissue"
annotations_2.each { annotation ->
    annotation.setPathClass(tissueClass)
}

// Update the hierarchy to reflect changes
fireHierarchyUpdate()

print "All annotations have been classified as 'Tissue'."


import qupath.lib.roi.GeometryTools
import qupath.lib.roi.interfaces.ROI
import org.locationtech.jts.geom.Geometry

// Parameters
double distanceThreshold = 7000.0  // Distance threshold in micrometers

// Retrieve all annotations of the "Tissue" class
def tissueAnnotations = getAnnotationObjects().findAll { it.getPathClass() != null && it.getPathClass().getName() == "Tissue" }
print "Number of tissue annotations found: ${tissueAnnotations.size()}"

// Function to calculate Euclidean distance between annotation centroids
double calculateDistance(def annotation1, def annotation2) {
    double x1 = annotation1.getROI().getCentroidX()
    double y1 = annotation1.getROI().getCentroidY()
    double x2 = annotation2.getROI().getCentroidX()
    double y2 = annotation2.getROI().getCentroidY()
    return Math.sqrt(Math.pow(x2 - x1, 2) + Math.pow(y2 - y1, 2))
}

// Function to merge two annotations
def mergeAnnotations(def annotation1, def annotation2) {
    def roi1 = annotation1.getROI()
    def roi2 = annotation2.getROI()
    
    // Convert ROIs to JTS Geometries for merging
    Geometry geom1 = GeometryTools.roiToGeometry(roi1)
    Geometry geom2 = GeometryTools.roiToGeometry(roi2)
    
    // Perform the union operation on the geometries
    Geometry mergedGeometry = geom1.union(geom2)
    
    // Convert back to an ROI
    def mergedROI = GeometryTools.geometryToROI(mergedGeometry, roi1.getImagePlane())
    
    // Create the merged annotation
    def mergedAnnotation = PathObjects.createAnnotationObject(mergedROI, annotation1.getPathClass())
    return mergedAnnotation
}

// Perform multiple rounds of merging
boolean merged
do {
    merged = false
    def toRemove = []
    def toAdd = []

    for (int i = 0; i < tissueAnnotations.size(); i++) {
        def annotation1 = tissueAnnotations[i]

        for (int j = i + 1; j < tissueAnnotations.size(); j++) {
            def annotation2 = tissueAnnotations[j]

            double distance = calculateDistance(annotation1, annotation2)
            if (distance < distanceThreshold) {
                print "Merging annotation ${i} and annotation ${j} with distance: ${distance} micrometers"
                
                // Merge the two annotations
                def mergedAnnotation = mergeAnnotations(annotation1, annotation2)

                // Schedule annotations to be removed and added
                if (mergedAnnotation != null) {
                    toRemove.add(annotation1)
                    toRemove.add(annotation2)
                    toAdd.add(mergedAnnotation)
                    merged = true
                    break
                }
            }
        }
        if (merged) break
    }

    // Apply changes outside the loop to avoid modifying the list during iteration
    tissueAnnotations.removeAll(toRemove)
    tissueAnnotations.addAll(toAdd)
    toRemove.each { removeObject(it, true) }
    toAdd.each { addObject(it) }
} while (merged)

print "Merging complete!"

        