def inputDir = new File("Users/kylabruno/Downloads/Stat_390/needs_manual_intervention") // Replace with path to test folder
//def inputDir = new File("Volumes/DATA_STORE/raw_data_liverpool")

// Standardize filenames in the input directory
inputDir.eachFile { file ->
    if (file.name.endsWith(".tif")) {
        def stainType = getStainType(file.name)
        // Remove stain type keywords before extracting the sample ID, and prepend 'h'
        def sampleID = "h" + file.name.replace(".tif", "")
                               .replace("sox10", "")
                               .replace("melan", "")
                               .replace("mela", "")
                               .replace("H&E", "")
                               .replaceAll("[^\\d]", "") // Extract numeric ID only

        
        def standardizedName = "${stainType}_${sampleID}.tif"
        
        def standardizedFile = new File(inputDir, standardizedName)
        if (!file.name.equals(standardizedName)) {
            file.renameTo(standardizedFile)
            println "Renamed ${file.name} to ${standardizedName}"
        }
    }
}

// Helper function to determine stain type based on filename pattern
def getStainType(filename) {
    filename = filename.toLowerCase()
    if (filename.contains("sox10")) return "sox10"
    if (filename.contains("melan") || filename.contains("mela")) return "melan"
    if (filename.contains("h&e")) return "h&e"
    return "unknown"
}
