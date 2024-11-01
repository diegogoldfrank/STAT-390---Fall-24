# STAT-390---Fall-24

# STEP 1 - TISSUE SLICE EXTRACTION
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

# Cara's Automation
All information / documentation of how to use Cara's code can be found on the google doc below:
https://docs.google.com/document/d/1u3e0Bw7LGokr4gZgAOPgqrvykZYsyqcx-LGDGNzFQV0/edit?tab=t.0

# STEP 2 - MATCHING SLICES ACROSS STAINS

# Tissue Matching Pipeline

This repository provides a streamlined pipeline for processing and matching tissue images across different stains (H&E, Melanin, Sox10) for patient samples. The pipeline includes image preprocessing, contour extraction, and similarity scoring to automate tissue slice matching.

## Project Overview

The project contains two main files:
- **`NoahsMatching.ipynb`**: An interactive Jupyter Notebook for processing and visualizing tissue matches.
- **`match_pipeline.py`**: A Python script designed for batch processing of images.

---

## Table of Contents
- [Setup](#setup)
- [Step-by-Step Manual](#step-by-step-manual)
- [Pipeline Workflow](#pipeline-workflow)
  - [Image Preprocessing](#image-preprocessing)
  - [Matching Process](#matching-process)
- [Functions](#functions)
  - [Notebook Functions](#notebook-functions)
  - [Script Functions](#script-functions)
- [Results](#results)
- [Known Limitations](#known-limitations)
- [Extraction and Quality Issues](#extraction-and-quality-issues)
- [Troubleshooting](#troubleshooting)

---

## Setup

Ensure you have the following dependencies installed:

- **Python** 3.8+
- **OpenCV** 4.5.4
- **Numpy** 1.21.0
- **Matplotlib** 3.4.3
- **Tkinter** (for GUI functionality, often pre-installed with Python)

Place `.tif` image files in a designated directory, such as `processed_images`.

---

## Step-by-Step Manual

Follow these steps to use the pipeline to generate matching slices for all patients or a specific subset:

1. **Prepare Image Data**: Place `.tif` image files in the designated folder (e.g., `processed_images`). Ensure that each patient has images for all three stains (H&E, Melanin, Sox10).

2. **Choose Full or Subset Processing**:
   - **For All Patients**: Place all images in the same `processed_images` directory.
   - **For a Subset of Patients**: Separate patient image files into their own folders within `processed_images` (e.g., `processed_images/patient1`, `processed_images/patient2`).

3. **Run the Pipeline**:
   - For an interactive run, open **`NoahsMatching.ipynb`** in Jupyter Notebook and execute cells in sequence.
   - For batch processing, run `match_pipeline.py`. A GUI will prompt you to select the directory containing the images.

4. **Process and Review Matches**:
   - Upon execution, the pipeline will preprocess, extract contours, and match images, saving results in the `matches` folder. Images will be organized by patient, with matched slices grouped in subfolders.

---

## Pipeline Workflow

The pipeline is divided into two main stages: image preprocessing and the matching process.

### Image Preprocessing
1. **Standardize File Names**: Filenames are reformatted to `[PatientID]_[StrainType].tif`.
2. **Remove Incomplete Patient Data**: Any patient missing one or more strains is excluded.
3. **Filter by Resolution**: For patients with multiple copies of the same strain, only the highest-resolution image is kept.
4. **Slice Extraction**: Combines Noah's cropping algorithm with Cara's automation to generate slices from each image.
5. **Quality Control on Slices**: Removes low-quality slices (smallest file size) and allows manual review for further filtering.
6. **Organize by Patient and Strain**: Files are moved to folders named `[PatientID]_[StrainType]`, with slice numbers appended to each filename.
7. **Match Set Creation**: Groups images into subfolders of matched slices, deleting any remaining unmatched images.

### Matching Process
1. **Contour Extraction**: Extracts main contours from each image to represent shape.
2. **Similarity Scoring**: Uses `cv2.matchShapes` to calculate contour similarity and builds a similarity matrix.
3. **Match Selection**: Identifies the closest matches and saves them in the `matches` directory.

---

## Functions

### Notebook Functions (`NoahsMatching.ipynb`)

- **`extract_main_contour(image)`**: Finds the largest contour in an image.
- **`shape_similarity(contour1, contour2)`**: Calculates similarity between two contours.
- **`similarity_matrix(set1, set2, names)`**: Generates a similarity matrix for two image sets.
- **`match(sim_matrix, row, col)`**: Matches images based on similarity scores.
- **`show_matches(matches)`**: Visualizes matched image pairs.

### Script Functions (`match_pipeline.py`)

- **`preprocess_files()`**: Handles renaming, filtering, and organization of images.
- **`extract_images(folder_path, patient)`**: Reads images for a specified patient.
- **`calculate_shape_similarity(image1, image2, image3)`**: Calculates similarity among three images.
- **`calculate_all_similarities(image_dict)`**: Finds minimum similarity scores for image groups.
- **`process_all_patients(folder_path)`**: Processes all patients in the directory and saves matches.

---

## Results

Processed images and matched sets are saved in the `matches` directory, organized by patient. Only matched images are retained, with unmatched or low-quality images removed.

---

## Known Limitations

The pipeline has the following limitations due to certain generalizations and non-generalized aspects:

- **Contour Matching with Holes in Tissue**: The algorithm currently handles a single example with a hole in the tissue, as there were limited examples available. It averages the similarity between the three contours when a hole is present, relying on the similarity of the largest contour, not just edge similarity.
- **Similarity Score Threshold**: A fixed threshold has been set for similarity scores, which may work well for most cases but could be insufficient for specific cases with varying image quality or abnormal structures. 
- **Adaptive Thresholding by Stain Type**: Custom adaptive thresholding is applied for each stain, tuned to reduce noise in Melan-a and Sox10 images. However, this adaptation is not entirely generalized and may need tuning for new or varied data sources.
- **Pixel-Based Detection**: The pipeline uses pixel values corresponding to stain-specific colors (e.g., purple for H&E). This color dependency may limit generalizability to other types of stains or differently colored tissue samples.
- **Contour Selection Based on Size**: The algorithm selects the two largest contours to account for tissue holes or irregular shapes. If there are multiple areas of interest, this method may not capture smaller but relevant contours, which could affect accuracy in more complex samples.
- **Limited Testing for Hole Detection**: The approach for detecting tissue holes has only been tested on a few examples, limiting our understanding of its robustness across diverse sample types.

---

## Extraction and Quality Issues

The following issues were identified during the extraction and matching process:

### Extraction Issues (3/38 Patients)
1. **Patient h1845484**: Unable to extract H&E slices with either Noah or Cara’s code.
2. **Patient h1942528**: Unable to extract slices for all three strains.
3. **Patient h2114167**: No slices were extracted.

### Missing Slices (13/38 Patients)
- **Patient h1857578**: Missing 1 H&E slice (likely due to low file size).
- **Patient h2114162**: Missing all slices.
- **Patient h2114163**: Missing Melan slice.
- **Patient h2114164**: Missing 2 H&E, all Melan, and 1 Sox10 slice.
- **Patient h2114165**: Missing 2 H&E, all Melan, and all Sox10 slices.
- **Patient h2114166**: Missing Melan and Sox10 slices.
- **Patient h2114169**: Missing Melan slice.
- **Patient h2114170**: Missing Melan and Sox10 slices.
- **Patient h2114171**: Missing Sox10 slices.
- **Patient h2114172**: Missing Sox10 slices.
- **Patient h2114179**: Missing Sox10 slices.
- **Patient h2114180**: Missing Sox10 slices.
- **Patient h2114188**: Missing 1 H&E, 2 Melan, and all Sox10 slices.

### Image Quality Issues (3/38 Patients)
- **Patient h2114161**: Sox10 image shows bars across it.
- **Patient h2114166**: One H&E slice has gray bars.
- **Patient h2114181**: Some split H&E slices; one slice has a black line.

### Uncertainty (1/38 Patients)
- **Patient h2114168**: Uncertain if extracted slices are correct or only parts of a whole image.

---

## Troubleshooting

- **File Errors**: Verify that `.tif` images are correctly stored in the specified directory.
- **Low Match Quality**: Adjust contour extraction parameters or use higher-resolution images.

# STEP 3 - EPITHELIUM DETECTION
# Epithelium Detection and Extraction

This method is designed to analyze tissue images, focusing on identifying and quantifying purple-stained regions. The analysis begins by generating an epithelium mask to isolate tissue regions using contour detection. The input image is preprocessed through grayscale conversion, blurring, and thresholding to create a binary mask. Morphological operations refine this mask by smoothing edges and removing noise, while contours outlining the epithelium are extracted and drawn on the original image, creating a clear visual distinction of the tissue region.

Following this, superpixel segmentation uses color and texture similarity to divide the image into smaller segments. Each segment is analyzed for purple concentration based on defined HSV color thresholds, which allows for consistent analysis across varying lighting conditions. Regions with high staining concentration retain their original color, moderate concentrations are marked in white, and low concentrations in black, emphasizing tissue areas by staining intensity.

---

This requires Kevin's algorithm, which is in this folder (Epithelium Extraction.ipynb). 

## Table of Contents
- [Instruction Manual](#instruction-manual)
- [Limitations](#limitations)

---

## Instruction Manual 
Ensure image input is in .tif format and all functions match the image input

Ensure you have the following dependencies installed:

- **Python** 3.8+
- **OpenCV** 4.5.4
- **Numpy** 1.21.0
- **Matplotlib** 3.4.3
- **scikit-image** 0.18.1

Install the necessary packages if you haven’t already:
```bash
pip install opencv-python-headless numpy scikit-image matplotlib
```

Adjust parameters like `num_segments` for superpixel count and 
`lower_purple`/`upper_purple` to refine HSV color thresholds as needed. 


## Limitations

While this method effectively highlights stained regions, it has some limitations. 
It relies on specific threshold and morphological settings, which may need adjustment for different images.
Variations in lighting or stain intensity can also affect results, requiring color threshold adjustments. 
Additionally, the code is not fully developed for handling outside regions and struggles with overlapping
tissue areas where superpixels may blend inside and outside regions. Expanding these capabilities would 
improve its effectiveness across a wider range of tissue samples.

# STEP 4 - PATCHING

README file explaning Eli's method for creating and exporting patches:

Eli's algorithm takes in the processed images from Cara’s segment (in one drive and linked in "Step 1" folder on github"). Eli's algorithm identifies the stroma and the epithelium, applying a sliding kernel across the image of either a) the local epithelium (W) and 100 pixels (H) or b) If more than 50% of the "window" is epithelium, records a new segment. 

Description of code: 
Loads images and converts them to grayscale. Applies a Guassian blur to smooth the image. Use thresholding to segment the epithelium. Apply morphological operations to clean up the mask. Find contours to identify the epithelium region. Create a mask for the epithelium. Calculate the width of the epithelium at multiple segments for each row. Define the sliding window function and slide it accross the image. If epithelium occupies more than 50%, adjust box width. Ensure the box does not exceed image boundaries. Ensure box does not overlap with already covered areas. Apply merging logic and merge windows vertically within the same horizontal region. Draw the merged windows. Save the final merged windows image. Calculate the average area of all merged windows. Calculate the side length of a square with the same average area. Draw the square windows.
--> Loop through all images in the input folder, and process the images and save the output, mask, intermediate, final, and squared images

Room for improvement: The epithelium extraction can still be refined and there are still open questions about how to best sample the image (pending the research team).

Hardcoding: Only hardcoded value relates to the color thresholding used in the epithelium extraction. Other group members / team 6 have been working on ways to generalize this segment and can likely be combined. 

Next steps: Merge windows vertically across regions
