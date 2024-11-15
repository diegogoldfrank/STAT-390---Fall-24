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
