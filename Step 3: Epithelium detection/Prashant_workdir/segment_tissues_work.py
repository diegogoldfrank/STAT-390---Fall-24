
##########################################

##       ENTER PATIENT IMAGE ID         ##

patient_id = "h2114154 h&e"

##########################################


## Import OpenCV

import sys
sys.path.append('/usr/local/lib/python3.8/site-packages')
import cv2
import numpy as np
import matplotlib.pyplot as plt

from skimage import data
from skimage import color
from skimage import morphology
from skimage import segmentation
from skimage import filters

import os

# Plot the image
def imshow(img, title):
    plt.imshow(img)
    plt.tight_layout()
    plt.axis('off')
    plt.title(title)
    plt.show(block=True)

extracted_dir = "extracted"

segmented_images_dir = "segmented"


sub_img_num =1
filename = os.path.join(extracted_dir, patient_id, patient_id + " series[0] img#" + str(sub_img_num))
print(filename + " Lumma.png")

#Image loading
img_color = cv2.imread(filename + ".png")
#imshow(img_color, 'img_color')

lumma_channel = cv2.imread(filename + " Lumma.png", cv2.IMREAD_GRAYSCALE)
#imshow(lumma_channel, "lumma_channel")

img_gray= cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)
#imshow(img_gray, "img_gray")

fig, ax_arr = plt.subplots(1, 3, sharex=True, sharey=True, figsize=(20, 12))
fig.suptitle('RGB Grayscale vs YUV Lumma - Stroma-Epithelia better differentiated', fontsize = 25)
ax1, ax2, ax3 = ax_arr.ravel()
ax1.imshow(img_color)
ax1.set_title('RGB888', fontsize = 20)
ax1.set_axis_off()
ax2.imshow(img_gray)
ax2.set_title('RGB888_to_Grayscale', fontsize = 20)
ax2.set_axis_off()
ax3.imshow(lumma_channel)
ax3.set_title('YCbCr_Y(Luminance)_channel', fontsize = 20)
ax3.set_axis_off()
plt.tight_layout()
filename = os.path.join(segmented_images_dir, patient_id + " img#" + str(sub_img_num) + " RGB Grayscale vs YUV Lumma.png")
plt.savefig(filename)
#plt.show()
plt.close()
del fig


red = img_color[:,:,0]
fig, ax_arr = plt.subplots(1, 2, sharex=True, sharey=True, figsize=(20, 12))
fig.suptitle("Lumma vs Red Channel", fontsize = 25)
ax_arr[0].set_title("Lumma Channel", fontsize = 20)
ax_arr[0].set_axis_off()
ax_arr[0].imshow(lumma_channel)
ax_arr[1].set_title("Red Channel", fontsize = 20)
ax_arr[1].set_axis_off()
ax_arr[1].imshow(red)
plt.tight_layout()
filename = os.path.join(segmented_images_dir, patient_id + " img#" + str(sub_img_num) + " Lumma vs Red Channel.png")
plt.savefig(filename)
plt.show()
plt.close()
del fig



# define binning in image
bins = 20
divisor = (np.floor(255 / bins).astype(np.uint8))


# Histogram
fig = plt.hist(red, bins=bins) 
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.title("Red Histogram " + str(bins) + " bins")
filename = os.path.join(segmented_images_dir, patient_id + " img#" + str(sub_img_num) + " Red Histogram " + str(bins) + " bins.png")
plt.savefig(filename)
plt.show()
del fig

# decimate image into bins bands
red_bands = (np.floor(red/divisor)).astype(np.uint8)
filename = os.path.join(segmented_images_dir, patient_id + " img#" + str(sub_img_num) + " Red Bands " + str(bins) + ".png")
cv2.imwrite(filename, red_bands * divisor, [cv2.IMWRITE_PNG_COMPRESSION , 0])
imshow(red_bands, "red_bands " + str(bins))

# figure to show different decimated values
fig, ax_arr = plt.subplots(2, int(bins/2), sharex=True, sharey=True, figsize=(20, 12))
fig.suptitle("Red Bands " + str(bins) + " Bins Representation", fontsize = 25)
row=0
col=0
for band_i in range(0,bins):
    print("band_i = " + str(band_i))
    ax_arr[row,col].set_title("value == " + str(band_i + 1), fontsize = 20)
    ax_arr[row,col].set_axis_off()
    ax_arr[row,col].imshow(red_bands == band_i + 1)
    col=col+1
    if col == int(bins/2):
        row = 1
        col = 0

plt.tight_layout()
filename = os.path.join(segmented_images_dir, patient_id + " img#" + str(sub_img_num) + " Red Bands " + str(bins) + " Representation.png")
plt.savefig(filename)
plt.show()
plt.close()
del fig





# Histogram
fig = plt.hist(lumma_channel, bins=bins) 
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.title("lumma_channel Histogram " + str(bins) + " bins" )
filename = os.path.join(segmented_images_dir, patient_id + " img#" + str(sub_img_num) + " Lumma Histogram " + str(bins) + " bins.png")
plt.savefig(filename)
plt.show()
del fig

# decimate image into bins bands
lumma_bands = (np.floor(lumma_channel/divisor)).astype(np.uint8)
filename = os.path.join(segmented_images_dir, patient_id + " img#" + str(sub_img_num) + " Lumma Bands " + str(bins) + ".png")
cv2.imwrite(filename, lumma_bands * divisor, [cv2.IMWRITE_PNG_COMPRESSION , 0])
imshow(lumma_bands, "lumma_bands " + str(bins))

# figure to show different decimated values
fig, ax_arr = plt.subplots(2, int(bins/2), sharex=True, sharey=True, figsize=(20, 12))
fig.suptitle("Lumma Bands " + str(bins) + " Bins Representation", fontsize = 25)
row=0
col=0
for band_i in range(0,bins):
    print("band_i = " + str(band_i))
    ax_arr[row,col].set_title("value == " + str(band_i + 1), fontsize = 20)
    ax_arr[row,col].set_axis_off()
    ax_arr[row,col].imshow(lumma_bands == band_i + 1)
    col=col+1
    if col == int(bins/2):
        row = 1
        col = 0

plt.tight_layout()
filename = os.path.join(segmented_images_dir, patient_id + " img#" + str(sub_img_num) + " Lumma Bands " + str(bins) + " Representation.png")
plt.savefig(filename)
plt.show()
plt.close()
del fig


# find band with most number of pixels

most_pixels_band = -1;
most_pixels = 0
for band_i in range(0,bins+1):
    n_pixels = np.count_nonzero(lumma_bands == band_i)
    print("band = " + str(band_i) + "   n_pixels = " + str(n_pixels))
    if n_pixels > most_pixels:
        most_pixels = n_pixels
        most_pixels_band = band_i

print("\nmost_pixels_band = " + str(most_pixels_band) + "   most_pixels = " + str(most_pixels))

# find background

bg1 = lumma_bands == most_pixels_band
imshow(bg1, "bg1")
bg2 = morphology.remove_small_objects(bg1, 5000)
imshow(bg2, "bg2")
background = morphology.remove_small_holes(bg2, 10000)
imshow(background, "background")
filename = os.path.join(segmented_images_dir, patient_id + " img#" + str(sub_img_num) + " Lumma Background Mask.png")
cv2.imwrite(filename, background.astype(np.uint8)*255, [cv2.IMWRITE_PNG_COMPRESSION , 0])


# Apply the background mask to image
bg3 = background.astype(np.uint8) * 255
background_img = cv2.bitwise_and(img_color,img_color,mask = bg3)
imshow(background_img, "background_img")
filename = os.path.join(segmented_images_dir, patient_id + " img#" + str(sub_img_num) + " Lumma Background Image.png")
cv2.imwrite(filename, background_img, [cv2.IMWRITE_PNG_COMPRESSION , 0])



import scipy.ndimage as ndimage

def pixels_within_distance(mask, distance):
    """Finds pixels within a specified distance from a 2D mask."""
    
    # Create a distance map from the mask
    distance_map = ndimage.distance_transform_edt(np.logical_not(mask))
    
    # Find pixels within the specified distance
    pixels_within = distance_map <= distance
    
    return pixels_within

result = pixels_within_distance(background, 20)



# find stroma

# 2 band prior to background band
stroma_band = most_pixels_band-1
stroma1 = lumma_bands == stroma_band
# add one more band behind
stroma1 = stroma1 + lumma_bands == stroma_band-1
imshow(stroma1, "stroma1")
# remove background pixels from stroma
stroma2 = stroma1 * np.invert(background)
imshow(stroma2, "stroma2")
# remove pixels nearby background pixels from stroma
stroma3 = stroma2 * np.invert(result)
imshow(stroma3, "stroma3")
# dilation
stroma4 = morphology.dilation(stroma3, morphology.square(10))
imshow(stroma4, "stroma4")
# remove small objects
stroma5 = morphology.remove_small_objects(stroma4, 10000)
imshow(stroma5, "stroma5")
# remove small holes
stroma6 = morphology.remove_small_holes(stroma5, 20000)
# dilation
stroma7 = morphology.dilation(stroma6, morphology.square(10))
# remove small holes
stroma = morphology.remove_small_holes(stroma7, 20000)
imshow(stroma, "stroma")
filename = os.path.join(segmented_images_dir, patient_id + " img#" + str(sub_img_num) + " Lumma Stroma Mask.png")
cv2.imwrite(filename, stroma.astype(np.uint8)*255, [cv2.IMWRITE_PNG_COMPRESSION , 0])

# Apply the stroma mask to image
stroma6 = stroma.astype(np.uint8) * 255
stroma_img = cv2.bitwise_and(img_color,img_color,mask = stroma6)
imshow(stroma_img, "stroma_img")
filename = os.path.join(segmented_images_dir, patient_id + " img#" + str(sub_img_num) + " Lumma Stroma Image.png")
cv2.imwrite(filename, stroma_img, [cv2.IMWRITE_PNG_COMPRESSION , 0])


# Find Epithelia

# Anything not Background and not Stroma is Epithelia
# band prior to Stroma band
epithelia_band = stroma_band-2
epithelia1 = lumma_bands <= epithelia_band
imshow(epithelia1, "epithelia1")
epithelia11 = lumma_bands > most_pixels_band
epithelia12 = epithelia11 + epithelia1
imshow(epithelia12, "epithelia1")
# remove background pixels from stroma
epithelia2 = epithelia12 * np.invert(background)
# remove stroma pixels from stroma
epithelia = epithelia2 * np.invert(stroma)
imshow(epithelia, "epithelia")
filename = os.path.join(segmented_images_dir, patient_id + " img#" + str(sub_img_num) + " Lumma Epithelia Mask.png")
cv2.imwrite(filename, epithelia.astype(np.uint8)*255, [cv2.IMWRITE_PNG_COMPRESSION , 0])


# Apply the epithelia mask to image
epithelia_m = epithelia.astype(np.uint8) * 255
epithelia_img = cv2.bitwise_and(img_color,img_color,mask = epithelia_m)
imshow(epithelia_img, "epithelia_img")
filename = os.path.join(segmented_images_dir, patient_id + " img#" + str(sub_img_num) + " Lumma Epithelia Image.png")
cv2.imwrite(filename, epithelia_img, [cv2.IMWRITE_PNG_COMPRESSION , 0])










# decimate image into 25 bands
lumma_bands = (np.floor(lumma_channel/10)).astype(np.uint8)
filename = os.path.join(segmented_images_dir, patient_id + " img#" + str(sub_img_num) + " Lumma Bands 25.png")
cv2.imwrite(filename, lumma_bands * 10, [cv2.IMWRITE_PNG_COMPRESSION , 0])
imshow(lumma_bands, 'lumma_bands')



# fill in values 0-3 from nearby regions




lumma_f = lumma_band.astype(np.float64)
lumma_f = lumma_f / 255
#imshow(lumma_f, 'lumma_f')
test = (lumma_f < 0.8) & (lumma_f > 0.7)
imshow(test, 'test')
lumma_fg = filters.gaussian(lumma_f, sigma=10)#, preserve_range=True)
imshow(lumma_fg, 'lumma_fg')
imshow(lumma_f < 0.3, 'black_holes')

# gaussian filter
#gaussian_smooth = filters.gaussian(lumma_channel, sigma=10)#, preserve_range=True)
#imshow(gaussian_smooth, 'gaussian_smooth')

# Compute a mask
mskg = gaussian_smooth < 0.7
imshow(mskg, 'mskg')
msk1 = morphology.remove_small_objects(mskg, 500)
#imshow(msk1, 'msk1')
msk2 = morphology.remove_small_holes(msk1, 4000)
#imshow(msk2, 'msk2')

# dilation
dilated_lumma_mask = morphology.dilation(msk2, morphology.square(25))
#imshow(dilated_lumma_mask, 'dilated_mask')
remove_small_holes = morphology.remove_small_holes(dilated_lumma_mask, 5000)
mask = morphology.remove_small_objects(remove_small_holes, 5000)

#mask = morphology.opening(dilated_mask, morphology.disk(10))
#imshow(mask, 'mask')

# show mask
fig, ax_arr = plt.subplots(1, 4, sharex=True, sharey=True, figsize=(20, 12))
ax1, ax2, ax3, ax4 = ax_arr.ravel()
ax1.imshow(img_color)
ax1.set_title('img_color', fontsize = 25)
ax1.set_axis_off()
ax2.imshow(lumma_channel)
ax2.set_title('lumma_channel', fontsize = 25)
ax2.set_axis_off()
ax3.imshow(mask)
ax3.set_title('mask', fontsize = 25)
ax3.set_axis_off()
ax4.imshow(segmentation.mark_boundaries(img_color, mask))
ax4.contour(mask, colors='red', linewidths=3)
ax4.set_title('img tumor', fontsize = 25)
ax4.set_axis_off()
plt.tight_layout()
#plt.show()
filename = os.path.join(segmented_images_dir, patient_id + " img#" + str(sub_img_num) + " Tumor.png")
plt.savefig(filename)
print("'" + filename + "' saved!")
plt.close()

