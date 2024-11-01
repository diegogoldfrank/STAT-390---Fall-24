# Epithelium Extraction

This method is designed to analyze tissue images, specifically focusing on identifying and quantifying purple-stained regions. 
The pipeline first generates an epithelium mask to isolate tissue regions using contour detection. 
After preprocessing the input image with grayscale conversion, blurring, and thresholding, a binary mask is created.
Morphological operations then refine this mask by smoothing edges and removing noise. 
Contours outlining the epithelium are extracted and drawn on the original image, creating a clear visual distinction of the tissue region.
Next, superpixel segmentation uses color and texture similarity to divide the image into smaller segments. 
Each segment is analyzed for purple concentration based on predefined HSV color thresholds, allowing a more consistent analysis across lighting variations. Regions with high staining concentration retain their original color, moderate concentrations are marked in white, and low concentrations in black, emphasizing tissue areas by staining intensity.

---

## Table of Contents
- [Setup](#setup)
- [Instruction Manual for Use](#instruction-manual-for-use)
- [Pipeline Workflow](#pipeline-workflow)
  - [Image Preprocessing](#image-preprocessing)
  - [Segmentation Process](#segmentation-process)
- [Known Limitations](#known-limitations)
- [Troubleshooting](#troubleshooting)

---

## Setup

Ensure you have the following dependencies installed:

- **Python** 3.8+
- **OpenCV** 4.5.4
- **Numpy** 1.21.0
- **Matplotlib** 3.4.3
- **Scikit-image** 0.18.1

Install the necessary packages if you haven't already:
```bash
pip install opencv-python-headless numpy scikit-image matplotlib

Run code after and interpret results

Adjust parameters like `num_segments` for superpixel count and 
`lower_purple`/`upper_purple` to refine HSV color thresholds as needed. 


## Limitations 

While this method effectively highlights stained regions, it has some limitations. 
It relies on specific threshold and morphological settings, which may need adjustment for different images.
Variations in lighting or stain intensity can also affect results, requiring color threshold adjustments. 
Additionally, the code is not fully developed for handling outside regions and struggles with overlapping
tissue areas where superpixels may blend inside and outside regions. Expanding these capabilities would 
improve its effectiveness across a wider range of tissue samples.

