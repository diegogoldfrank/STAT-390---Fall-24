/// run on the project to run on every image loaded into the project. 

// Create annotations from a pixel classifier
//def imageName = imageData.getServer().getMetadata().getName()
def imageName = getCurrentImageData().getServer().getMetadata().getName()

// Define paths to classifiers
def heClassifier = "Users/kylabruno/Downloads/Stat_390/Week6_Presentation5/classifiers/pixel_classifiers/find_tissue_h&e.json"
def melanClassifier = "Users/kylabruno/Downloads/Stat_390/Week6_Presentation5/classifiers/pixel_classifiers/find_tissue_melan.json"
def sox10Classifier = "Users/kylabruno/Downloads/Stat_390/Week6_Presentation5/classifiers/pixel_classifiers/find_tissue_sox10.json"

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
        
        