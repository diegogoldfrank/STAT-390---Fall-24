// RUN FOR PROJECT

// load libraries
import qupath.lib.objects.PathObjects
import qupath.lib.roi.RoiTools
import qupath.lib.roi.GeometryTools
import qupath.lib.roi.interfaces.ROI
import org.locationtech.jts.geom.Geometry


// LOAD IMAGE ================================
// Get the current open project
def project = getProject()
if (project == null) {
    println "No project is currently open."
    return
}

// Store project's base directory
def projectBaseDir = project.getBaseDirectory()
// println "The project base directory is: ${projectBaseDir}"

// Store image name in project
def imageName = getCurrentImageData().getServer().getMetadata().getName()

// Clean image name
if (imageName.contains(" - Series 0")) {
    imageName = imageName.replace(" - Series 0", "")
    //println "Updated image name to: ${imageName}"
}

// Don't use image masks
if (imageName.toLowerCase().contains(" - mask")) {
    return
}


// CREATE ANNOTATIONS ================================
// Define paths to classifiers 
def heClassifier = new File(projectBaseDir, "classifiers/pixel_classifiers/find_tissue_h&e.json").getAbsolutePath()
def melanClassifier = new File(projectBaseDir, "classifiers/pixel_classifiers/find_tissue_melan.json").getAbsolutePath()
def sox10Classifier = new File(projectBaseDir, "classifiers/pixel_classifiers/find_tissue_sox10.json").getAbsolutePath()

//println "H&E Classifier Path: ${heClassifier}"
//println "Melan Classifier Path: ${melanClassifier}"
//println "Sox10 Classifier Path: ${sox10Classifier}"

// Annotation Params
double minArea = 20000
double minHoleArea = 1.0

// Identify the correct pixel classifier using standardized name
if (imageName.toLowerCase().contains("h&e")) {
        print("Applying H&E classifier to " + imageName)
        
        // Clear any existing objects (so dont have to reload)
        clearAllObjects()
        
        // Create Objects
        createAnnotationsFromPixelClassifier(heClassifier, minArea, minHoleArea)
 
    } else if (imageName.toLowerCase().contains("melan")) {
        print("Applying Melan classifier to " + imageName)
        
        clearAllObjects()
        createAnnotationsFromPixelClassifier(melanClassifier, minArea, minHoleArea)
       
    } else if (imageName.toLowerCase().contains("sox10")) {
        print("Applying Sox10 classifier to " + imageName)
        
        clearAllObjects()
        createAnnotationsFromPixelClassifier(sox10Classifier, minArea, minHoleArea)
    }
      

// SPLIT ANNOTATIONS INTO SEPARATE TISSUES =====================================
// Get all annotations made in the current image
def annotations = getAnnotationObjects()

// usually creates one big annotation
annotations.each { annotation ->
    def roi = annotation.getROI()
    
    // Split the annotation by identifying the individual parts
    def splitROIs = RoiTools.splitROI(roi)
    
    // Create separate annotation objects for each  region
    splitROIs.each { subROI ->
        def subAnnotation = PathObjects.createAnnotationObject(subROI)
        addObject(subAnnotation)
    }
    
    // Remove the original annotation if needed
    removeObject(annotation, true)
}


// REDEFINE AS TISSUE CLASS ==================================
// Define the desired class (create it if it doesn't already exist)
def tissueClass = getPathClass("Tissue")

// Get split annotations
def annotations_split = getAnnotationObjects()

// Set class to "Tissue"
annotations_split.each { annotation ->
    annotation.setPathClass(tissueClass)
}

// Update the hierarchy (just in case)
fireHierarchyUpdate()


// REMOVE DEBRIS (TOO LARGE TO BE TISSUE) ========================
// Get updated annotations
def annotations_tissue = getAnnotationObjects()

// Remove annotations with an area greater than threshold
def largeAnnotations = annotations_tissue.findAll { it.getROI()?.getArea() > 10000000} // 8 digits 
    
largeAnnotations.each { annotation ->
    removeObject(annotation, true) 
}


// MERGE BROKEN TISSUE SEGMENTS =============================
double distanceThreshold = 7000.0

// get all tissue annotations with debris removed
def tissueAnnotations = getAnnotationObjects().findAll { it.getPathClass() != null && it.getPathClass().getName() == "Tissue" }

// help function to get Euclidean distance between centroids
double calculateDistance(def annotation1, def annotation2) {
    double x1 = annotation1.getROI().getCentroidX()
    double y1 = annotation1.getROI().getCentroidY()
    double x2 = annotation2.getROI().getCentroidX()
    double y2 = annotation2.getROI().getCentroidY()
    return Math.sqrt(Math.pow(x2 - x1, 2) + Math.pow(y2 - y1, 2))
}

// helper function to merge two annotations
def mergeAnnotations(def annotation1, def annotation2) {
    def roi1 = annotation1.getROI()
    def roi2 = annotation2.getROI()
    
    // Convert ROIs to JTS Geometries for merging
    Geometry geom1 = GeometryTools.roiToGeometry(roi1)
    Geometry geom2 = GeometryTools.roiToGeometry(roi2)
    
    // Union operation
    Geometry mergedGeometry = geom1.union(geom2)
    
    // Convert back to ROI
    def mergedROI = GeometryTools.geometryToROI(mergedGeometry, roi1.getImagePlane())
    
    // Create merged annotation
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
                //print "Merging annotation ${i} and annotation ${j} with distance: ${distance} micrometers"
                
                // Merge
                def mergedAnnotation = mergeAnnotations(annotation1, annotation2)

                // Store to keep or remove
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

    // Apply changes outside loop to not change list while iterating
    tissueAnnotations.removeAll(toRemove)
    tissueAnnotations.addAll(toAdd)
    toRemove.each { removeObject(it, true) }
    toAdd.each { addObject(it) }
    
} while (merged)

print "Number of tissue annotations found: ${tissueAnnotations.size()}"

        