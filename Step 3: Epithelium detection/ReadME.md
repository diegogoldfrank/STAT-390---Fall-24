# Epithelium Detection and Extraction

This method is designed to analyze tissue images, focusing on identifying and quantifying purple-stained regions. The analysis begins by generating an epithelium mask to isolate tissue regions using contour detection. The input image is preprocessed through grayscale conversion, blurring, and thresholding to create a binary mask. Morphological operations refine this mask by smoothing edges and removing noise, while contours outlining the epithelium are extracted and drawn on the original image, creating a clear visual distinction of the tissue region.

Following this, superpixel segmentation uses color and texture similarity to divide the image into smaller segments. Each segment is analyzed for purple concentration based on defined HSV color thresholds, which allows for consistent analysis across varying lighting conditions. Regions with high staining concentration retain their original color, moderate concentrations are marked in white, and low concentrations in black, emphasizing tissue areas by staining intensity.

---

## Table of Contents
- [Instruction Manual](#instruction-manual)
- [Limitations](#limitations)

---

## Instruction Manual 

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

