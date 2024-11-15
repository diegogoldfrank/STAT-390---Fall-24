// Define the desired class (create it if it doesn't already exist)
def tissueClass = getPathClass("Tissue")

// Get all annotations in the image
def annotations = getAnnotationObjects()

// Set each annotation's class to "Tissue"
annotations.each { annotation ->
    annotation.setPathClass(tissueClass)
}

// Update the hierarchy to reflect changes
fireHierarchyUpdate()

print "All annotations have been classified as 'Tissue'."
