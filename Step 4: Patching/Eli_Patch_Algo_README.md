## README file explaning Eli's method for creating and exporting patches:

### Updates from previous week: 
The new code calculates width / height for patches based on Krish’s specifications and then applies horizontal / vertical patching across the image.

Eli's algorithm takes in the processed images from Cara’s segment (in one drive and linked in "Step 1" folder on github"). Eli's algorithm identifies the stroma and the epithelium, applying a sliding kernel across the image of either a) the local epithelium (W) and 100 pixels (H) or b) If more than 50% of the "window" is epithelium, records a new segment. 

### Description of code: 
Loads images and converts them to grayscale. Applies a Guassian blur to smooth the image. Use thresholding to segment the epithelium. Apply morphological operations to clean up the mask. Find contours to identify the epithelium region. Create a mask for the epithelium. Calculate the width of the epithelium at multiple segments for each row. Define the sliding window function and slide it accross the image. If epithelium occupies more than 50%, adjust box width. Ensure the box does not exceed image boundaries. Ensure box does not overlap with already covered areas. Apply merging logic and merge windows vertically within the same horizontal region. Draw the merged windows. Save the final merged windows image. Calculate the average area of all merged windows. Calculate the side length of a square with the same average area. Draw the square windows.
--> Loop through all images in the input folder, and process the images and save the output, mask, intermediate, final, and squared images

### Room for improvement: 
The epithelium extraction can still be refined and there are still open questions about how to best sample the image (pending the research team).

### Hardcoding: 
Only hardcoded value relates to the color thresholding used in the epithelium extraction. Other group members / team 6 have been working on ways to generalize this segment and can likely be combined. 

### Next steps: 
Merge windows vertically across regions







