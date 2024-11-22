# Instructions to Extract Tissue Slices

#### Preliminary Steps
- Download QuPath version 0.5.1 [here](https://qupath.github.io/).
    - *note that this code was developed with the methods and functions available in this specific version of QuPath, which is the most recent public version as of November 2024*
- Download raw image data 
	- from our [OneDrive](https://nuwildcat-my.sharepoint.com/personal/akl0407_ads_northwestern_edu/_layouts/15/onedrive.aspx?id=%2Fpersonal%2Fakl0407%5Fads%5Fnorthwestern%5Fedu%2FDocuments%2FSTAT390%2FData) or use new data
- Download `Extract_Tissues` folder

#### Complete the following steps in order
##### Open the Project
1. Open QuPath
2. Click "Open Project" --> `/Extract_Tissues` --> `project.qpproj`

##### Upload Images
3. Click "Add Images"
4. Drag and drop your raw image data folder. Or upload individual image files. Click "Import"
	- No need to change any settings

##### Run Master Script
5. In toolbar, click "Automate" --> "Project scripts" --> `1 master script` 
6. Click three dots in bottom right of box --> Click "Run for project" --> hit ">>" in center to select all images --> click "OK"
	- wait for batch processing to complete
7. Close the script editor
	- *Optional: review annotations and delete any you dont want

>Note the difference between "Run for project" and "Run". The master script is meant to be run for the entire project while the export annotations script is built to simply "Run"

##### Export Results
8. In toolbar, click "Automate" --> "Project scripts" --> `2 export annotations` --> Click "Run"
> its possible to get an "Out of memory" interruption. Make sure you have storage on your machine. Perhaps later we can implement an option to export to a flashdrive or some external disk in the GUI. The default is to load the extracted slices to the directory of the project. 

##### View Extracted Slices
9. Navigate to `/Extract_Tissues/Tissues` in your directory. 
	- Each subfolder corresponds to a single raw image. 
	- Within each subfolder are the individual tissues extracted from the raw image (anywhere from 1 tissue to 6)
> Note: if you are unexpectedly seeing no tissue samples in some folders, wait a couple minutes as the `.tif` files take a while to load into your directory
