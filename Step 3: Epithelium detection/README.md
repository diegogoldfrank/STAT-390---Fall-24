# cmil

Extracting Conjunctival melanocytic intraepithelial lesions (C-MIL) tissue from Eye tissue and making a Deep Learning Classification Model for Cancer Detection.  


[Slides](https://docs.google.com/presentation/d/1eMNb1Jq0VQQtGIA3xF9gm0oATRtgcBYlBgqJnGiM9Pw/edit?usp=sharing)  


## Usage: python3 segmentation_algo.py

## User input:

show_images = 1                             # to display images in real-time  
save_intermediate_images = 0  
input_dir = "data/sheffield_h&e"            # all files in here will be read, expected filenames: <filename>.tif  
output_dir = "extracted/sheffield_h&e"      # epithelia and stroma will be saved here  
run_over_all_images = 1                     # to run over all images in 'input_dir'  
overwrite_output = 1                        # to overwrite previous output  
image_name = "h2114186 h&e_ROI_3"	        # specific image to run with 'run_over_all_images = 0'  
save_epithelia_and_stroma = 1		        # to save epithelia and stroma output  
  
 visualizations  
output_visualization_dir = "extracted/sheffield_h&e/visualization"      # output for visual comparison b/w input_img-segmented_stroma-segmented_epithelia  
save_bins_representation = 0                # to save Lumma and Red Chroma Bins for visualization  


## Folder structure
- data: put all tif images here  
- extracted: tissues extracted from images in data will be saved here  
- workingdir: R&D directory  


## Dependencies
- opencv 4.10.0-dev: https://docs.opencv.org/4.10.0/d7/d9f/tutorial_linux_install.html  
- python3: Python 3.8.10  
- python3 pip: should already be there with python3  
- tiffFile: https://pypi.org/project/tifffile/  
- scikit-image: https://scikit-image.org/  


## Notes
- view tiff image using: tifffile <filename>
- view tiff image series 0 using: tifffile <filename> -s 0
