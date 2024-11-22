// JUST RUN

// load libraries
import qupath.lib.projects.ProjectImageEntry
import qupath.lib.objects.PathObject
import qupath.lib.regions.RegionRequest
import qupath.lib.gui.scripting.QPEx
import qupath.lib.images.ImageData

// get all images in project
def imageList = project.getImageList()

// for every image in the project
for (entry in imageList) { 
    try {
        def imageName = entry.getImageName()
        
        // Skip over mask images
        if (imageName.toLowerCase().contains("- mask")) {
            continue
        }

        // Open current image
        def imageData
        try {
            imageData = entry.readImageData()
            if (imageData == null) {
                println "No image data available for: ${imageName}"
                continue
            }
        } catch (Exception e) {
            println "ERROR: Unable to load image data for ${imageName}. Exception: ${e.message}"
            continue
        }

        QPEx.setBatchProjectAndImage(project, imageData)

        def server = imageData.getServer()
        def path = server.getPath()
        def downsample = 1.0 //change this ?????? depends on complexity

        // Make directories for outcomes
        def name = GeneralTools.stripExtension(server.getMetadata().getName())
        def pathOutput = buildFilePath(PROJECT_BASE_DIR, "Tissues", name)

        try {
            mkdirs(pathOutput)
        } catch (Exception e) {
            println "ERROR: Unable to create output directory for ${name}. Exception: ${e.message}"
            continue
        }

        // Get all annotations in curr images
        def hierarchy = imageData.getHierarchy()
        def annotations
        try {
            annotations = hierarchy.getAnnotationObjects()
        } catch (Exception e) {
            println "ERROR: Unable to retrieve annotations for ${name}. Exception: ${e.message}"
            continue
        }
        
        // Go through each annotation
        def i = 1 // for naming
        def j = 1 // for looping

        if (annotations.isEmpty()) {
            println "No tissues to export from image: ${name}"
        } else {
            annotations.each { annotation ->
                try {
                    def roi = annotation.getROI()
                    if (roi == null) {
                        println "Can't export annotation with null ROI for ${name}"
                        return
                    }
                    
                    // when annotation does exist as expected
                    def request = RegionRequest.createInstance(path, downsample, roi)
                    
                    if (annotation.getPathClass()?.toString()?.contains("Tissue")) {
                        def outputFilePath = pathOutput + "/t_" + i + ".tif"
                        
                        writeImageRegion(server, request, outputFilePath)
                        println "Exported t_" + i + " from ${name}"
                        i++
                        
                    } else {
                        j++
                    }
                    
                } catch (Exception e) {
                    println "ERROR: Failed to process annotation in ${name}. Exception: ${e.message}"
                }
            }
        }

        // Save back to the current project
        try {
            entry.saveImageData(imageData)
            
        } catch (Exception e) {
            println "ERROR: Unable to save image data for ${name}. Exception: ${e.message}"
        }
        
        // confirm completion
        println "Finished processing image: ${name}\n"
        
    } catch (Exception e) {
        println "Unexpected error processing image: ${entry.getImageName()}. Exception: ${e.message}"
    }
}
