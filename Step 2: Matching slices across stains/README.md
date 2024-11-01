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
 - Run `match_pipeline.py`. A GUI will prompt you to select the directory containing the images.
 - Open **`NoahsMatching.ipynb`** in Jupyter Notebook and execute cells in sequence.


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
