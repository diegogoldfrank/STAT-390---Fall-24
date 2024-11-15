import qupath.lib.projects.ProjectImageEntry
import qupath.lib.objects.PathObject
import qupath.lib.regions.RegionRequest
import qupath.lib.gui.scripting.QPEx
import qupath.lib.images.ImageData

def imageList = project.getImageList() // get all images

for (entry in imageList) { // loop through each image
    // open image
    def imageData = entry.readImageData()
    QPEx.setBatchProjectAndImage(project, imageData)
    
    def server = imageData.getServer()
    def path = server.getPath()
    def downsample = 1.0
    
    // make directories
    def name = GeneralTools.getNameWithoutExtension(server.getMetadata().getName())
    //def name = GeneralTools.stripExtension(server.getMetadata().getName())
    def pathOutput = buildFilePath(PROJECT_BASE_DIR, "Tissues", name)

    mkdirs(pathOutput)

    // get annotations
    def hierarchy = imageData.getHierarchy()
    def annotations = hierarchy.getAnnotationObjects()
    
    def i = 1
    def j = 1

    // go through each annotation
    if (annotations.isEmpty()) {
        print("No annotations found for image: " + name)
    } else {
        for (annotation in annotations) {
            def roi = annotation.getROI()
            def request = RegionRequest.createInstance(path, downsample, roi)
    
            if (annotation.getPathClass().toString().contains("Tissue")) {
                writeImageRegion(server, request, pathOutput + "/t_" + i + ".tif")
                i++
            } else {
                j++
            }
        }
    }

    // save back to the project
    entry.saveImageData(imageData)

    print("Finished processing image: " + name)
}
