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
