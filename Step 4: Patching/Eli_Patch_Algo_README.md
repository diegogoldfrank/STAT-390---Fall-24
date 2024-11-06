## README file explaning Eli's method for creating and exporting patches:

## Updates from previous week: 
The new code calculates width / height for patches based on Krish’s specifications and then applies horizontal / vertical patching across the image. The new version of the code introduces the calculate_optimal_patch_dimensions function, which calculates the optimal patch height and width based on the epithelium mask. The patch dimensions are based on the maximum continuous width of white pixels (i.e., the epithelium area) and the overall epithelium area in the mask. The optimal patch height is the maximum width of continuous white pixels (in each row), with a minimum height of 100. The patch width is calculated by dividing the epithelium area by 100 and the optimal patch height.

## About Eli's Algorithm:
Eli's algorithm takes in the processed images from Cara’s segment (in one drive and linked in "Step 1" folder on github"). Eli's algorithm identifies the stroma and the epithelium, applying a sliding kernel across the image of either a) the local epithelium (W) and 100 pixels (H) or b) If more than 50% of the "window" is epithelium, records a new segment. 

## Description of code: 

#### Directories:
Defines input_folder (where processed images are stored) and output_folder (where the results will be saved).

#### calculate_optimal_patch_dimensions(mask): 
Analyzes the epithelium mask to determine the optimal patch size for image processing. Finds the maximum continuous width of white pixels (epithelium) in each row to determine the optimal patch height. Calculates the patch width based on the total epithelium area, ensuring it's large enough to cover meaningful portions of the image.

#### calculate_overlap(patch_coords, placed_patches): 
Checks if a new patch overlaps with any already placed patches. Returns True if the overlap area exceeds 10% of the patch's total area, indicating that the patch should not be placed.

#### calculate_coverage(mask, patches): 
Calculates the coverage percentage of epithelium in the provided mask by summing the areas covered by all patches.Returns the percentage of epithelium covered by patches.

#### apply_patches(epithelium_mask, patch_height, patch_width, stride, orientation): 
Applies patches to the epithelium mask, either vertically or horizontally. Iterates over the image using a sliding window approach, placing patches where the epithelium ratio is 50% or more. Checks for overlaps using calculate_overlap and ensures patches are placed in non-overlapping areas.

#### process_image(image_path, output_image_path, output_mask_path, region_outline_path): 
Reads and Preprocesses Image: Loads the image, converts it to grayscale, applies Gaussian blur, and thresholds to create a binary mask.

#### Mask Cleaning: 
Applies morphological operations (close and open) to clean the mask.

#### Epithelium Detection: 
Finds contours, identifies the largest contour, and creates an epithelium mask.

#### Patch Calculation: 
Uses calculate_optimal_patch_dimensions to determine the optimal patch size and stride for patch placement.

#### Apply Patches: 
Uses apply_patches for both vertical and horizontal patch orientations and calculates the coverage for each.

#### Region Outline: 
Draws rectangles around regions in the original image and saves an image showing the regions where patches will be applied.

#### Draw Patches: 
Based on the coverage, selects the best orientation (vertical or horizontal) and draws patches on the image.

#### Save Output: 
Saves the final processed image with patches drawn, as well as the region outline image and the epithelium mask.

#### Image Processing Loop: 
Loops through all image files in the input_folder with .tif, .jpg, or .png extensions. For each image, it processes the image and saves: the epithelium mask, the final image with mixed patches applied, the region outline image showing the patches' locations.

Prints a message indicating that the processing is complete for all images.

### Summary of the Main Steps:
The code preprocesses each image by converting it to grayscale and applying Gaussian blur and thresholding.
It detects the largest epithelium region using contour detection.
It calculates optimal patch dimensions based on the epithelium area.
Patches are applied using a sliding window approach, ensuring minimal overlap and maximizing coverage.
The final processed images, with patches applied and region outlines, are saved to the specified output folder.

## Room for improvement: 
Still open questions about how to best sample the image (pending the research team).

## Hardcoding: 
Only hardcoded value relates to the color thresholding used in the epithelium extraction. Other group members / team 6 have been working on ways to generalize this segment and can likely be combined. 

## Next steps: 
Merge windows vertically across regions







