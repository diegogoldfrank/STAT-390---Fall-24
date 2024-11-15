# Presentation 6

#### Objective: 
- ***precisely*** answering padding related questions - Atharva and Parth 
- avoid manual intervention in tissue slice extraction
	- Team 3 requests steps 1-3 have the code refined to be easily executable by the stakeholders
- ensure that tissue slice extraction can handle the Sheffeild Sox10 cases. 

## Instructions to Extract Tissue Slices

1. Download raw image data to your local directory if not already there.
2. Run `1_rename_to_standard` to standardize the naming convention for all images
	- make sure to set directory to where you have put the raw image data
3. Open the QuPath project. Add the raw images
	- Delete mask images (will automate this next)
4. Automate --> Project Scripts --> master_script.groovy --> **Run for project** 
	- from here a user can review what will be exported (might be useful for stakeholder to review but if not, I will automate exporting as well)
	- when making GUI I'm thinking there can be a pause here where the user is prompted "Export Extracted Tissue Samples? Yes, No"
5. To export: Automate --> Project Scripts --> 6_export_annotations.groovy --> **Run**
	- 3 min 28 seconds to export 6 .tif files

>Note the difference between "Run for project" and "Run". The master script is meant to be run for the entire project while the export annotations script is built to simply "Run"

#### TLDR
- rename to standard
- load images to QuPath project. Delete masks
- master_script.groovy
- 6_export_annotations.groovy

## Next Steps
Automate skipping over or removing the mask images
- check for QuPath setting to not create them in the first place
- Cara has a way to do this but I got an error and need to look closer at it

Make decisions for deletion process
- detect images with abnormalities, then display to user and allow them to decide if sample should be used or not. User hits Yes/No to keep or not keep. Deletion/exclusion is then automated

Implement into GUI (make user friendly)
- adjust code to take in input values from the user (eg. local directory where raw data is, yes/no selections, etc.) so the stakeholders don't have to directly work with code
- might be useful to have scripts to organize local directories in a way that the user specifies (eg. post tissue extraction, make it really easy to group samples by folder, by patient, etc.)

## The Most Detailed Instructions

On your computer, you need
- Qupath - can download version 0.5.1 [here](https://qupath.github.io/).
  	- note that this code was developed with the methods and functions available in this specific version of QuPath, which is the most recent public version as of November 2024
- raw image data - can download from our OneDrive (or use new data)
- the Extract_Tissues folder in this directory

Then complete the following steps in order
1. Open QuPath
2. Click "Open Project" --> `/Extract_Tissues` --> `project.qpproj`
3. Remove any images already in there (shouldn't be any but I may have accidentally uploaded a version of the project with some preloaded)
4. Click "Add Images"
5. Drag and drop your raw image data folder. No need to change default settings. No need to open any of the images in the Viewer. Click "Import"
6. *Step to be Eliminated*: Select all mask images, right click, "Remove images"
7. In toolbar, click "Automate" --> "Project scripts" --> `master script` --> click three dots in bottom right of box --> Click "Run for project"
8. Can review annotated results here if you'd like and delete annotations for unusable samples. Can also delete them later after exportation
9. "Automate" --> "Project scripts" --> `6 export annotations` --> Click "Run"
10. Navigate to the Extract Tissues folder in your directory. There should now be an additional folder called "Tissues" that contains subfolders corresponding to each raw image. Within each subfolder are the individual tissues extracted from the raw image (anywhere from 1 tissue to 6)
11. if you are seeing no tissue samples in a bunch of folders, it is just because the .tif files take a while to load into your directory
