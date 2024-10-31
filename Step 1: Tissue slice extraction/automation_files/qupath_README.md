# QuPath scripting files to automate tissue extraction
By: Cara Chang, updated 10/25/24

Images in a QuPath Project --> processed_data/tissue_image.tif
Contains tissues_1.json, tissues_2.json and automate_export_newest.groovy
New for other stain types to expedite process: existing_annotations.groovy


## Setup in QuPath:
1. Open a project that has all your images
2. Put tissue_2.json into base_proj_dir/classifiers/pixel_classifiers (make if does not exist)
Note, use tissues_2.json for most recent results (not tissues_1 but you can still try this too. tissues_2 contains broader parameters for a more sensitive model, works on more stains and images)
3. Put automate_export_newest.groovy into base_project_dir/scripts (make if does not exist)
4. Make sure you have an image open in QuPath interface
5. In QuPath, top bar --> Automate --> Project scripts --> automate_export_newest.groovy
6. Script Editor has three vertical dots at the bottom --> Run for project
7. Data will save in processed_data dir in your base project dir

## To deal with more difficult stain types if you decide to manually annotate:
## Runs like automate_export_newest.groovy but only if you already have annotations
1. Need to set annotation class to "Positive" in QuPath (Annotations --> Positive --> Set selected and for future annotations to be auto "Positive," press "Auto set"")
2. To export existing annotations only, run existing_annotations.groovy
3. existing_annotations.groovy --> base_project_dir/scripts
4. In QuPath, top bar --> Automate --> Project scripts --> existing_annotations.groovy
5. Script Editor has three vertical dots at the bottom --> Run for project
6. Data will save in processed_data dir in your base project dir

## To create a new pixel classifier or modify mine (optional):
1. QuPath Interface top bar --> Classify --> Pixel Classification --> Create thresholder
2. See tissues_1.json and tissues_2.json for my parameters, and you can work from there
3. Save this and then replace "tissues_2" in .groovy script.