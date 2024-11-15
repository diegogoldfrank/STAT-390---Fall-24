
##########################################

##       SEGMENTATION ALGORITHM         ##

show_images = 0                                 # to display images in real-time
save_intermediate_images = 0                    # save R-G-B and Y-Cb-Cr Channels
#input_dir = "data/sheffield_h&e"               # all files in here will be read, expected filenames: <filename>.tif
#input_dir = "data/liverpool_h&e"
#input_dir = "data"
#output_dir = "extracted/sheffield_h&e"         # epithelia and stroma will be saved here
#output_dir = "extracted/liverpool_h&e"
#output_dir = "workingdir/segmented"
run_over_all_images = 1                         # to run over all images in 'input_dir'
overwrite_output = 0                            # to overwrite previous output
image_name = "h1810898A  h&e_ROI_1"             # specific image to run with 'run_over_all_images = 0'
save_epithelia_and_stroma = 1                   # to save epithelia and stroma output

# visualizations

output_visualization_dir = "extracted/sheffield_h&e/visualization"         # output for visual comparison b/w input_img-segmented_stroma-segmented_epithelia
#output_visualization_dir = "extracted/liverpool_h&e/visualization"
#output_visualization_dir = "workingdir/segmented"
save_bins_representation = 0                    # to save Lumma and Red Chroma Bins for visualization

##########################################


import argparse
import os

# Step 1: Argument parsing
def parse_args():
    """Parses command line arguments."""
    parser = argparse.ArgumentParser(description='Segmentation Algorithm')
    parser.add_argument('--input_dir', type=str, required=True, help='Path to the input directory')
    parser.add_argument('--output_dir', type=str, required=True, help='Path to the output directory')
    
    return parser.parse_args()

# Step 2: Parse the arguments
args = parse_args()

# Step 3: Assign arguments to variables
input_dir = args.input_dir
output_dir = args.output_dir

# Step 4: Validate the input directory
if not os.path.isdir(input_dir):
    print(f"Error: The input directory '{input_dir}' does not exist.")
    exit(1)

# Step 5: Ensure output directory exists (optional, can be added later in the script if needed)
os.makedirs(output_dir, exist_ok=True)

# Step 6: Debug prints for validation (optional)
print(f"DEBUG: Input Directory: {input_dir}")
print(f"DEBUG: Output Directory: {output_dir}")

## Import OpenCV

import sys
sys.path.append('/usr/local/lib/python3.8/site-packages')
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
#from skimage import data
#from skimage import color
from skimage import morphology
#from skimage import segmentation
#from skimage import filters
#import scipy.ndimage as ndimage
import gc

# Plot the image
def imshow(img, title):
    global show_images
    if show_images == 0:
        return
    
    plt.imshow(img)
    plt.tight_layout()
    plt.axis('off')
    plt.title(title)
    plt.show(block=True)
    plt.close('all')

#def pixels_within_distance(mask, distance):
#    """Finds pixels within a specified distance from a 2D mask."""
#    # Create a distance map from the mask
#    distance_map = ndimage.distance_transform_edt(np.logical_not(mask))
#    # Find pixels within the specified distance
#    pixels_within = distance_map <= distance
#    return pixels_within


if not os.path.isdir(input_dir):
    print("input_dir: '" + input_dir + "' directory does not exist! Exiting...")
    exit()

if not os.path.isdir(output_dir):
    os.mkdir(output_dir)
    print("output_dir: '" + output_dir + "' directory created.")

if not os.path.isdir(output_visualization_dir):
    os.mkdir(output_visualization_dir)
    print("output_visualization_dir: '" + output_visualization_dir + "' directory created.")
    

# Files and Folders in Input Dir
files = os.listdir(input_dir)
# Filtering only the files.
files = [f for f in files if os.path.isfile(input_dir+'/'+f)]

for f in files:
    if run_over_all_images:
        print("\n*********** CURRENT FILE: " + f)
        x = f.split(".", 1)
        if x[1] != "tif":
            print(x)
            print("Unexpected Filename - more dots than anticipated. Exiting...")
            exit()
        
        image_name = x[0]
        del x
    
    input_filepath = os.path.join(input_dir, image_name)
    print(input_filepath + ".tif")
    
    if overwrite_output == 0 and run_over_all_images:
        if os.path.exists(os.path.join(output_visualization_dir, image_name + ".png")):
            continue
    
    
    # Load Image
    img_rgb = cv2.imread(input_filepath + ".tif")
    img_rgb = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2RGB)
    imshow(img_rgb, 'img_rgb')
    if save_bins_representation or show_images:
        fig = plt.figure(num=1, clear=True, figsize=(20, 12))
        ax_arr = fig.subplots(1, 4, sharex=True, sharey=True)
        fig.suptitle('RGB - R - G - B', fontsize = 25)
        ax1, ax2, ax3, ax4 = ax_arr.ravel()
        ax1.imshow(img_rgb)
        ax1.set_title('RGB888', fontsize = 20)
        ax1.set_axis_off()
        ax2.imshow(img_rgb[:,:,0])
        ax2.set_title('Red Channel', fontsize = 20)
        ax2.set_axis_off()
        ax3.imshow(img_rgb[:,:,1])
        ax3.set_title('Green Channel', fontsize = 20)
        ax3.set_axis_off()
        ax4.imshow(img_rgb[:,:,2])
        ax4.set_title('Blue Channel', fontsize = 20)
        ax4.set_axis_off()
        plt.tight_layout()
        if save_bins_representation:
            filename = os.path.join(output_visualization_dir, image_name + " RGB Channels.png")
            plt.savefig(filename)
            print(filename + " saved")
        
        if show_images:
            plt.show()
        
    
    img_ycrcb = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2YCrCb)
    #imshow(img_ycrcb, 'img_ycrcb')
    if save_bins_representation or show_images:
        fig = plt.figure(num=1, clear=True, figsize=(20, 12))
        ax_arr = fig.subplots(1, 5, sharex=True, sharey=True)
        fig.suptitle('RGB888 - Y - Cb - Cr - Cg', fontsize = 25)
        ax1, ax2, ax3, ax4, ax5 = ax_arr.ravel()
        ax1.imshow(img_rgb)
        ax1.set_title('RGB888', fontsize = 20)
        ax1.set_axis_off()
        ax2.imshow(img_ycrcb[:,:,0])
        ax2.set_title('Y Channel', fontsize = 20)
        ax2.set_axis_off()
        ax3.imshow(img_ycrcb[:,:,1])
        ax3.set_title('Cb Channel', fontsize = 20)
        ax3.set_axis_off()
        ax4.imshow(img_ycrcb[:,:,2])
        ax4.set_title('Cr Channel', fontsize = 20)
        ax4.set_axis_off()
        # Green Difference Chroma
        # img_Cg = Y - 0.299 * Y - 0.587 * Cr - 0.114 * Cb
        img_Cg = img_ycrcb[:,:,0] - 0.299 * img_ycrcb[:,:,0] - 0.587 * img_ycrcb[:,:,2] - 0.114 * img_ycrcb[:,:,1]
        #filename = os.path.join(output_visualization_dir, image_name + " Green Chroma.png")
        #cv2.imwrite(filename, img_Cg, [cv2.IMWRITE_PNG_COMPRESSION , 0])
        #imshow(img_Cg, "img_Cg")
        #print(filename + " saved")
        ax5.imshow(img_Cg)
        ax5.set_title('Cg Channel', fontsize = 20)
        ax5.set_axis_off()
        plt.tight_layout()
        if save_bins_representation:
            filename = os.path.join(output_visualization_dir, image_name + " YCbCr Channels.png")
            plt.savefig(filename)
            print(filename + " saved")
        
        if show_images:
            plt.show()
        
    
    #img_lumma = img_ycrcb[:,:,0]
    imshow(img_ycrcb[:,:,0], "img_lumma")
    
    #img_gray= cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    #imshow(img_gray, "img_gray")
    
    #fig = plt.figure(num=1, clear=True, figsize=(20, 12))
    #ax_arr = fig.subplots(1, 3, sharex=True, sharey=True)
    #fig.suptitle('Grayscale vs YCbCr Lumma - Stroma-Epithelia better differentiated', fontsize = 25)
    #ax1, ax2, ax3 = ax_arr.ravel()
    #ax1.imshow(img_rgb)
    #ax1.set_title('RGB888', fontsize = 20)
    #ax1.set_axis_off()
    #ax2.imshow(img_gray)
    #ax2.set_title('RGB888_to_Grayscale', fontsize = 20)
    #ax2.set_axis_off()
    #ax3.imshow(img_ycrcb[:,:,0])
    #ax3.set_title('YCbCr_Y(Luminance)', fontsize = 20)
    #ax3.set_axis_off()
    #plt.tight_layout()
    #filename = os.path.join(output_visualization_dir, image_name + " RGB Grayscale vs YUV Lumma.png")
    #plt.savefig(filename)
    #print(filename + " saved")
    ##plt.show()
    
    # we will get background from lumma channel
    
    # define binning in lumma image
    lumma_bins_n = 20
    divisor = (np.floor(255 / lumma_bins_n).astype(np.uint8))
    
    # decimate lumma image into lumma_bins_n
    lumma_binned = (np.floor(img_ycrcb[:,:,0]/divisor)).astype(np.uint8)
    #filename = os.path.join(output_visualization_dir, image_name + " Lumma " + str(lumma_bins_n) + " Bins.png")
    #cv2.imwrite(filename, lumma_binned * divisor, [cv2.IMWRITE_PNG_COMPRESSION , 0])
    #imshow(lumma_binned, "lumma_binned " + str(lumma_bins_n))
    #print(filename + " saved")
    
    ## figure to show different lumma bins
    if save_bins_representation:
        fig = plt.figure(num=1, clear=True, figsize=(20, 12))
        ax_arr = fig.subplots(2, int(lumma_bins_n/2) + 1, sharex=True, sharey=True)
        fig.suptitle(image_name + " Lumma " + str(lumma_bins_n) + " Bins Representation", fontsize = 25)
        row=0
        col=0
        for bin_i in range(0,lumma_bins_n + 2):
            ax_arr[row,col].set_title("bin " + str(bin_i), fontsize = 20)
            ax_arr[row,col].set_axis_off()
            ax_arr[row,col].imshow(lumma_binned == bin_i)
            col=col+1
            if col == int(lumma_bins_n/2) + 1:
                row = 1
                col = 0
        
        plt.tight_layout()
        filename = os.path.join(output_visualization_dir, image_name + " Lumma " + str(lumma_bins_n) + " Bins Representation.png")
        plt.savefig(filename)
        print(filename + " saved")
        if show_images:
            plt.show()
        
    
    # find bin with most number of pixels
    
    most_pixels_bin = -1;
    most_pixels = 0
    for bin_i in range(0,lumma_bins_n+1):
        n_pixels = np.count_nonzero(lumma_binned == bin_i)
        #print("bin = " + str(bin_i) + "   n_pixels = " + str(n_pixels))
        if n_pixels > most_pixels:
            most_pixels = n_pixels
            most_pixels_bin = bin_i
    
    print("\nlumma_binned: most_pixels_bin = " + str(most_pixels_bin) + "   most_pixels = " + str(most_pixels))
    
    # find background
    
    background_bin = most_pixels_bin
    background = lumma_binned == background_bin
    imshow(background, "bg1 bin=" + str(most_pixels_bin))
    background = morphology.remove_small_objects(background, 5000)
    imshow(background, "bg2")
    background = morphology.remove_small_holes(background, 10000)
    print('background found')
    imshow(background, "background")
    if save_intermediate_images:
        filename = os.path.join(output_dir, image_name + " Lumma Background Mask.png")
        cv2.imwrite(filename, background.astype(np.uint8)*255, [cv2.IMWRITE_PNG_COMPRESSION , 0])
        print(filename + " saved")
    
    
    ## Apply the background mask to image
    #background_img = cv2.bitwise_and(img_rgb, img_rgb, mask = (background.astype(np.uint8) * 255))
    #imshow(background_img, "background_img")
    #filename = os.path.join(output_dir, image_name + " Lumma Background Image.png")
    #cv2.imwrite(filename, background_img, [cv2.IMWRITE_PNG_COMPRESSION , 0])
    #del background_img
    
    #background_grown = pixels_within_distance(background, 20)
    
    # get Red Chroma channel

    imshow(img_ycrcb[:,:,2], "img_Cr")
    
    # we will get epithelia from Red Chroma channel
    
    # define binning in Red Chroma image
    Cr_bins_n = 50
    divisor = (np.floor(255 / Cr_bins_n).astype(np.uint8))
    
    # decimate Red Chroma image into Cr_bins_n
    Cr_binned = (np.floor(img_ycrcb[:,:,2]/divisor)).astype(np.uint8)
    #filename = os.path.join(output_visualization_dir, image_name + " Red Chroma " + str(Cr_bins_n) + " Bins.png")
    #cv2.imwrite(filename, Cr_binned * divisor, [cv2.IMWRITE_PNG_COMPRESSION , 0])
    #imshow(Cr_binned, "Cr_binned " + str(Cr_bins_n))
    #print(filename + " saved")
    
    ## figure to show different Red Chroma bins
    if save_bins_representation:
        fig = plt.figure(num=1, clear=True, figsize=(20, 12))
        ax_arr = fig.subplots(2, 6, sharex=True, sharey=True)
        fig.suptitle(image_name + " Red Chroma " + str(Cr_bins_n) + " Bins Representation", fontsize = 25)
        row=0
        col=0
        for bin_i in range(0,12):
            ax_arr[row,col].set_title("bin " + str(bin_i + 21), fontsize = 20)
            ax_arr[row,col].set_axis_off()
            ax_arr[row,col].imshow(Cr_binned == (bin_i + 21))
            col=col+1
            if col == 6:
                row = 1
                col = 0
        
        plt.tight_layout()
        filename = os.path.join(output_visualization_dir, image_name + " Red Chroma " + str(Cr_bins_n) + " Bins Representation.png")
        plt.savefig(filename)
        print(filename + " saved")
        if show_images:
            plt.show()
        
    
    # find bin with most number of pixels
    
    most_pixels_bin = -1;
    most_pixels = 0
    for bin_i in range(0,Cr_bins_n+1):
        n_pixels = np.count_nonzero(Cr_binned == bin_i)
        #print("bin = " + str(bin_i) + "   n_pixels = " + str(n_pixels))
        if n_pixels > most_pixels:
            most_pixels = n_pixels
            most_pixels_bin = bin_i
    
    print("\nCr_binned: most_pixels_bin = " + str(most_pixels_bin) + "   most_pixels = " + str(most_pixels))
    
    # Find Definite Stroma
    
    # Stroma is three bins from  max pixels Cr bin - 2 to max pixels Cr bin
    stroma_bin = most_pixels_bin
    stroma = Cr_binned == stroma_bin
    stroma = stroma + (Cr_binned == stroma_bin - 1)
    stroma = stroma + (Cr_binned == stroma_bin - 2)
    imshow(stroma, "stroma1")
    # remove background pixels from stroma
    stroma = stroma * np.invert(background)
    imshow(stroma, "stroma2")
    # dilation
    stroma = morphology.dilation(stroma, morphology.square(3))
    imshow(stroma, "stroma3")
    # remove small objects
    stroma = morphology.remove_small_objects(stroma, 1000)
    imshow(stroma, "definite stroma")
    print('definite stroma found')
    
    
    # Mark regions with blue ink drop for removal from epithelia
    
    #img_Cb = img_ycrcb[:,:,1]
    #imshow(img_Cb, "img_Cb")
    #blue_ink = img_ycrcb[:,:,1] < 120
    #imshow(blue_ink, "blue_ink")
    
    ## define binning in Blue Chroma image
    #Cb_bins_n = 10
    #divisor = (np.floor(255 / Cb_bins_n).astype(np.uint8))
    #
    ## decimate Blue Chroma image into Cb_bins_n
    #Cb_binned = (np.floor(img_ycrcb[:,:,1]/divisor)).astype(np.uint8)
    ##filename = os.path.join(output_visualization_dir, image_name + " Blue Chroma " + str(Cb_bins_n) + " Bins.png")
    ##cv2.imwrite(filename, Cb_binned * divisor, [cv2.IMWRITE_PNG_COMPRESSION , 0])
    ##imshow(Cb_binned, "Cb_binned " + str(Cb_bins_n))
    ##print(filename + " saved")
    #
    ### figure to show different Blue Chroma bins
    #if save_bins_representation:
    #    fig = plt.figure(num=1, clear=True, figsize=(20, 12))
    #    ax_arr = fig = plt.subplots(2, 6, sharex=True, sharey=True)
    #    fig.suptitle(image_name + " Blue Chroma " + str(Cb_bins_n) + " Bins Representation", fontsize = 25)
    #    row=0
    #    col=0
    #    for bin_i in range(0,12):
    #        ax_arr[row,col].set_title("bin " + str(bin_i + 3), fontsize = 20)
    #        ax_arr[row,col].set_axis_off()
    #        ax_arr[row,col].imshow(Cb_binned == (bin_i + 3))
    #        col=col+1
    #        if col == 6:
    #            row = 1
    #            col = 0
    #    
    #    plt.tight_layout()
    #    filename = os.path.join(output_visualization_dir, image_name + " Blue Chroma " + str(Cb_bins_n) + " Bins Representation.png")
    #    plt.savefig(filename)
    #    print(filename + " saved")
    #    if show_images:
    #        plt.show()
    #    
    
    
    ## define binning in Green Channel
    #green_bins_n = 20
    #divisor = (np.floor(255 / green_bins_n).astype(np.uint8))
    #
    ## decimate Green Channel into green_bins_n
    ##green_binned = (np.floor(img_rgb[:,:,1]/divisor)).astype(np.uint8)
    #Cg_binned = (np.floor(img_Cg/divisor)).astype(np.uint8)
    #filename = os.path.join(output_visualization_dir, image_name + " Green Chroma " + str(green_bins_n) + " Bins.png")
    #cv2.imwrite(filename, Cg_binned * divisor, [cv2.IMWRITE_PNG_COMPRESSION , 0])
    #imshow(Cg_binned, "Cg_binned " + str(green_bins_n))
    #print(filename + " saved")
    #
    ### figure to show different Green Channel bins
    #if save_bins_representation:
    #    fig = plt.figure(num=1, clear=True, figsize=(20, 12))
    #    ax_arr = fig.subplots(2, int(green_bins_n/2) + 1, sharex=True, sharey=True)
    #    fig.suptitle(image_name + " Green Chroma " + str(green_bins_n) + " Bins Representation", fontsize = 25)
    #    row=0
    #    col=0
    #    for bin_i in range(0,green_bins_n + 2):
    #        ax_arr[row,col].set_title("bin " + str(bin_i), fontsize = 20)
    #        ax_arr[row,col].set_axis_off()
    #        ax_arr[row,col].imshow(Cg_binned == bin_i)
    #        col=col+1
    #        if col == int(green_bins_n/2) + 1:
    #            row = 1
    #            col = 0
    #    
    #    plt.tight_layout()
    #    filename = os.path.join(output_visualization_dir, image_name + " Green Chroma " + str(green_bins_n) + " Bins Representation.png")
    #    plt.savefig(filename)
    #    print(filename + " saved")
    #    if show_images:
    #        plt.show()
    #    
    #
    ## find bin with most number of pixels
    #
    #most_pixels_bin = -1;
    #most_pixels = 0
    #for bin_i in range(0,green_bins_n+1):
    #    n_pixels = np.count_nonzero(Cg_binned == bin_i)
    #    #print("bin = " + str(bin_i) + "   n_pixels = " + str(n_pixels))
    #    if n_pixels > most_pixels:
    #        most_pixels = n_pixels
    #        most_pixels_bin = bin_i
    #
    #print("\ngreen_binned: most_pixels_bin = " + str(most_pixels_bin) + "   most_pixels = " + str(most_pixels))
    
    
    # Find Epithelia
    
    # 1-line easy approximation: :)
    # imshow((img_ycrcb[:,:,2] >= 137) * (img_ycrcb[:,:,2] <= 145), "epithelia")
    
    # Epithelia is the first five bins that are 1 bin ahead of stroma in Red Chroma
    epithelia_bin = stroma_bin + 2
    epithelia = Cr_binned == epithelia_bin
    epithelia = epithelia + (Cr_binned == epithelia_bin + 1)
    epithelia = epithelia + (Cr_binned == epithelia_bin + 2)
    epithelia = epithelia + (Cr_binned == epithelia_bin + 3)
    epithelia = epithelia + (Cr_binned == epithelia_bin + 4)
    imshow(epithelia, "epithelia1")
    # remove background pixels from epithelia
    epithelia = epithelia * np.invert(background)
    imshow(epithelia, "epithelia2")
    # remove stroma pixels from epithelia
    epithelia = epithelia * np.invert(stroma)
    imshow(epithelia, "epithelia3")
    ## add Bins from Lumma - background bin * 0.6 and two below
    #epithelia = epithelia + (lumma_binned == int(background_bin * 0.6))
    #epithelia = epithelia + (lumma_binned == int(background_bin * 0.6 - 1))
    #epithelia = epithelia + (lumma_binned == int(background_bin * 0.6 - 2))
    #imshow(epithelia, "epithelia3lu")
    # remove blue ink drop region
    #blue_ink = img_ycrcb[:,:,1] < 120
    #imshow(blue_ink, "blue_ink")
    epithelia = epithelia * np.invert(img_ycrcb[:,:,1] < 120)
    imshow(epithelia, "epithelia4")
    # dilation
    #epithelia = morphology.dilation(epithelia, morphology.square(3))
    epithelia = morphology.dilation(epithelia, morphology.square(2))
    imshow(epithelia, "epithelia5")
    # remove very small objects
    epithelia = morphology.remove_small_objects(epithelia, 500)
    imshow(epithelia, "epithelia6")
    # remove small holes
    epithelia = morphology.remove_small_holes(epithelia, 10000)
    imshow(epithelia, "epithelia7")
    # find out epithelia information at current state
    n_epithelia_pixels = np.count_nonzero(epithelia)
    percent_epithelia_pixels = n_epithelia_pixels / (epithelia.shape[0] * epithelia.shape[1])
    print("percent_epithelia_pixels = " + str(percent_epithelia_pixels))
    small_obj_size_to_remove = 10000
    if percent_epithelia_pixels < 0.01:
        small_obj_size_to_remove = 1000
    elif percent_epithelia_pixels < 0.03:
        small_obj_size_to_remove = 2000
    
    print("small_obj_size_to_remove = " + str(small_obj_size_to_remove))
    # remove small objects
    epithelia = morphology.remove_small_objects(epithelia, small_obj_size_to_remove)
    imshow(epithelia, "epithelia8")
    # dilation
    epithelia = morphology.dilation(epithelia, morphology.square(10))
    imshow(epithelia, "epithelia9")
    # remove small holes
    epithelia = morphology.remove_small_holes(epithelia, 20000)
    print('epithelia found')
    imshow(epithelia, "epithelia")
    if save_intermediate_images:
        filename = os.path.join(output_dir, image_name + " Epithelia Mask.png")
        cv2.imwrite(filename, epithelia.astype(np.uint8)*255, [cv2.IMWRITE_PNG_COMPRESSION , 0])
        print(filename + " saved")
    
    
    # Apply the epithelia mask to image
    epithelia_img = cv2.bitwise_and(img_rgb,img_rgb,mask = (epithelia.astype(np.uint8) * 255))
    imshow(epithelia_img, "epithelia_img")
    if save_epithelia_and_stroma:
        filename = os.path.join(output_dir, image_name + " Epithelia.png")
        cv2.imwrite(filename, cv2.cvtColor(epithelia_img, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_PNG_COMPRESSION , 0])
        print(filename + " saved")
    
    
    # Expanded Stroma
    
    # Expanded Stroma is four bins from  max pixels Cr bin - 2 to max pixels Cr bin + 1
    stroma = Cr_binned == stroma_bin + 1
    stroma = stroma + (Cr_binned == stroma_bin)
    stroma = stroma + (Cr_binned == stroma_bin - 1)
    stroma = stroma + (Cr_binned == stroma_bin - 2)
    imshow(stroma, "stroma1")
    # remove background pixels from stroma
    stroma = stroma * np.invert(background)
    imshow(stroma, "stroma2")
    # remove small holes
    stroma = morphology.remove_small_holes(stroma, 10000)
    imshow(stroma, "stroma3")
    # dilation
    stroma = morphology.dilation(stroma, morphology.square(10))
    imshow(stroma, "stroma4")
    # remove small holes
    stroma = morphology.remove_small_holes(stroma, 20000)
    imshow(stroma, "stroma5")
    # remove background pixels from expanded stroma
    stroma = stroma * np.invert(background)
    # remove stroma pixels from expanded stroma
    stroma = stroma * np.invert(epithelia)
    print('stroma expanded')
    imshow(stroma, "stroma")
    if save_intermediate_images:
        filename = os.path.join(output_dir, image_name + " Stroma Mask.png")
        cv2.imwrite(filename, stroma.astype(np.uint8)*255, [cv2.IMWRITE_PNG_COMPRESSION , 0])
        print(filename + " saved")
    
    # Apply the stroma mask to image
    stroma_img = cv2.bitwise_and(img_rgb,img_rgb,mask = (stroma.astype(np.uint8) * 255))
    imshow(stroma_img, "stroma_img")
    if save_epithelia_and_stroma:
        filename = os.path.join(output_dir, image_name + " Stroma.png")
        cv2.imwrite(filename, cv2.cvtColor(stroma_img, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_PNG_COMPRESSION , 0])
        print(filename + " saved")
    
    
    
    ## find stroma
    #
    ## anything not background and not epithelia is stroma
    #stroma1 = np.full(lumma_binned.shape, True)
    ## remove background pixels from stroma
    #stroma2 = stroma1 * np.invert(background)
    ## remove stroma pixels from stroma
    #stroma = stroma2 * np.invert(epithelia)
    #print('stroma found')
    #imshow(stroma, "stroma")
    #if save_intermediate_images:
    #    filename = os.path.join(output_dir, image_name + " Stroma Mask.png")
    #    cv2.imwrite(filename, stroma.astype(np.uint8)*255, [cv2.IMWRITE_PNG_COMPRESSION , 0])
    #    print(filename + " saved")
    #
    ## Apply the stroma mask to image
    #stroma6 = stroma.astype(np.uint8) * 255
    #stroma_img = cv2.bitwise_and(img_rgb,img_rgb,mask = stroma6)
    #imshow(stroma_img, "stroma_img")
    #filename = os.path.join(output_dir, image_name + " Stroma.png")
    #cv2.imwrite(filename, cv2.cvtColor(stroma_img, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_PNG_COMPRESSION , 0])
    #print(filename + " saved")
    
    
    # Save a segmentation visualization image
    fig = plt.figure(num=1, clear=True, figsize=(20, 12))
    ax_arr = fig.subplots(1, 3, sharex=True, sharey=True)
    fig.suptitle('Input - Stroma - Epithelia Segmentation Visualization', fontsize = 25)
    ax1, ax2, ax3 = ax_arr.ravel()
    ax1.imshow(img_rgb)
    ax1.set_title('img_rgb', fontsize = 20)
    ax1.set_axis_off()
    ax2.imshow(stroma_img)
    ax2.set_title('stroma_img', fontsize = 20)
    ax2.set_axis_off()
    ax3.imshow(epithelia_img)
    ax3.set_title('epithelia_img', fontsize = 20)
    ax3.set_axis_off()
    plt.tight_layout()
    filename = os.path.join(output_visualization_dir, image_name + ".png")
    plt.savefig(filename)
    print(filename + " saved")
    if show_images:
        plt.show()
    
    del img_rgb, img_ycrcb, stroma_img, epithelia_img, background, epithelia, stroma, Cr_binned, lumma_binned, filename
    # Force a garbage collection
    gc.collect()
    
    if run_over_all_images == 0:
        break

print("done!")
exit()

