

##########################################

##       ENTER PATIENT IMAGE ID         ##

patient_id = "h2114153 h&e"

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
    plt.axis('off')
    plt.title(title)
    plt.show(block=True)

extracted_dir = "extracted"

# count number of sub images
no_of_images = 1

while True:
    filename = patient_id + " series[0] img#" + str(no_of_images)
    if os.path.exists(os.path.join(extracted_dir, patient_id, filename + ".png")):
        no_of_images=no_of_images+1
    else:
        no_of_images=no_of_images-1
        print("no_of_images = " + str(no_of_images))
        break

if no_of_images == 0:
    print("no extracted tissue images for patient image id: " + patient_id)
    exit()

segmented_images_dir = os.path.join(extracted_dir, patient_id, "segmented")

if not os.path.isdir(segmented_images_dir):
    os.mkdir(segmented_images_dir)
    print("'" + segmented_images_dir + "' directory created")
else:
    print("'" + segmented_images_dir + "' directory exists")
    userinp = input("Overwrite data (y/n)?")
    if userinp == "y":
        import shutil
        for root, dirs, files in os.walk(segmented_images_dir):
            for f in files:
                os.unlink(os.path.join(root, f))
            for d in dirs:
                shutil.rmtree(os.path.join(root, d))
        print("'" + os.path.join(extracted_dir, patient_id) + "' made empty.")
        del userinp, root, dirs, files
    else:
        exit()


for sub_img_num in range(1,no_of_images+1):
    filename = os.path.join(extracted_dir, patient_id, patient_id + " series[0] img#" + str(sub_img_num))
    print(filename + " Lumma.png")
    
    #Image loading
    img_lumma = cv2.imread(filename + " Lumma.png")
    # Show image
    #imshow(img_lumma, 'img_lumma')
    img_color = cv2.imread(filename + ".png")
    # Show image
    #imshow(img_color, 'img_color')
    
    
    #image grayscale conversion
    lumma_channel = cv2.cvtColor(img_lumma, cv2.COLOR_BGR2GRAY)
    #imshow(lumma_channel, "lumma_channel")
    
    # gaussian filter
    gaussian_smooth = filters.gaussian(lumma_channel, sigma=10)#, preserve_range=True)
    #imshow(gaussian_smooth, 'gaussian_smooth')
    
    # Compute a mask
    msk1 = morphology.remove_small_objects(gaussian_smooth < 0.7, 500)
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
    ax2.imshow(lumma_channel)
    ax2.set_title('lumma_channel', fontsize = 25)
    ax3.imshow(mask)
    ax3.set_title('mask', fontsize = 25)
    ax4.imshow(segmentation.mark_boundaries(img_color, mask))
    ax4.contour(mask, colors='red', linewidths=3)
    ax4.set_title('img tumor', fontsize = 25)
    for ax in ax_arr.ravel():
        ax.set_axis_off()
    plt.tight_layout()
    #plt.show()
    filename = os.path.join(segmented_images_dir, patient_id + " img#" + str(sub_img_num) + " Tumor.png")
    plt.savefig(filename)
    print("'" + filename + "' saved!")
    plt.close()

    
    
    
    
    # Find similar
    
    
    #image grayscale conversion
    gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)
    #imshow(lumma_channel, "lumma_channel")
    
    # gaussian filter
    gaussian_smooth_gray = filters.gaussian(gray, sigma=10)#, preserve_range=True)
    #imshow(gaussian_smooth, 'gaussian_smooth')
    
    # Compute a mask
    g1 = morphology.remove_small_objects(gaussian_smooth_gray < 0.9, 500)
    g2 = morphology.remove_small_holes(g1, 500)
    #imshow(g2, 'g2')
    
    # dilation
    dilated_g2 = morphology.dilation(g2, morphology.square(25))
    #imshow(dilated_g2, 'dilated_g2')
    g3 = morphology.remove_small_holes(dilated_g2, 5000)
    g4 = morphology.remove_small_objects(g3, 10000)
    #imshow(g4, 'color image 2D mask')
    fig = plt.subplots(1, 1, sharex=True, sharey=True, figsize=(8, 12))
    plt.imshow(g4)
    plt.axis('off')
    #plt.title('color image 2D mask', fontsize = 25)
    plt.tight_layout()
    filename = os.path.join(segmented_images_dir, patient_id + " img#" + str(sub_img_num) + " 2D mask.png")
    plt.savefig(filename)
    print("'" + filename + "' saved!")
    plt.close()
    
    
    # show mask
    fig, ax_arr = plt.subplots(1, 3, sharex=True, sharey=True, figsize=(20, 12))
    ax1, ax2, ax3 = ax_arr.ravel()
    ax1.imshow(img_color)
    ax1.set_title('img_color', fontsize = 25)
    ax2.imshow(g4)
    ax2.set_title('2D mask', fontsize = 25)
    ax3.imshow(segmentation.mark_boundaries(img_color, g4))
    ax3.contour(g4, colors='red', linewidths=3)
    ax3.set_title('img 2D mask', fontsize = 25)
    for ax in ax_arr.ravel():
        ax.set_axis_off()
    plt.tight_layout()
    #plt.show()
    filename = os.path.join(segmented_images_dir, patient_id + " img#" + str(sub_img_num) + " 2D mask figure.png")
    plt.savefig(filename)
    print("'" + filename + "' saved!")
    plt.close()
    
    
    
    
    
    # do ICP transform / fitting and find best fit image
    
    
    
    
    
    
    
    
    
print("done!")
exit()








# Display result
fig, ax_arr = plt.subplots(2, 2, sharex=True, sharey=True, figsize=(10, 10))
ax1, ax2, ax3, ax4 = ax_arr.ravel()

ax1.imshow(img)
ax1.set_title('Original image')

ax2.imshow(mask, cmap='gray')
ax2.set_title('Mask')

ax3.imshow(segmentation.mark_boundaries(img, slic))
ax3.contour(mask, colors='red', linewidths=1)
ax3.set_title('SLIC')

ax4.imshow(segmentation.mark_boundaries(img, m_slic))
ax4.contour(mask, colors='red', linewidths=1)
ax4.set_title('maskSLIC')

for ax in ax_arr.ravel():
    ax.set_axis_off()

plt.tight_layout()
plt.show()




















#image grayscale conversion
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
print("gray")
imshow(gray)

#Threshold Processing
ret, bin_img = cv2.threshold(gray,
                             0, 255, 
                             cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
print("Thresholding Otsu’s binarization")
imshow(bin_img)

# noise removal
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10))
bin_img = cv2.morphologyEx(bin_img, 
                           cv2.MORPH_OPEN,
                           kernel,
                           iterations=2)
dilation = cv2.dilate(bin_img,kernel,iterations = 10)
print("noise removal")
imshow(dilation)








# Detecting the black background and foreground of the image

# Create subplots with 1 row and 2 columns
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(8, 8))
# sure background area
sure_bg = cv2.dilate(dilation, kernel, iterations=3)
imshow(sure_bg, axes[0,0])
axes[0, 0].set_title('Sure Background')
 
# Distance transform
dist = cv2.distanceTransform(bin_img, cv2.DIST_L2, 5)
imshow(dist, axes[0,1])
axes[0, 1].set_title('Distance Transform')
 
#foreground area
ret, sure_fg = cv2.threshold(dist, 0.1 * dist.max(), 255, cv2.THRESH_BINARY)
sure_fg = sure_fg.astype(np.uint8)  
imshow(sure_fg, axes[1,0])
axes[1, 0].set_title('Sure Foreground')
 
# unknown area
unknown = cv2.subtract(sure_bg, sure_fg)
imshow(unknown, axes[1,1])
axes[1, 1].set_title('Unknown')
 
plt.show()



# Marker labelling
# sure foreground 
ret, markers = cv2.connectedComponents(sure_fg)
 
# Add one to all labels so that background is not 0, but 1
markers += 1
# mark the region of unknown with zero
markers[unknown == 255] = 0
 
fig, ax = plt.subplots(figsize=(6, 6))
ax.imshow(markers, cmap="tab20b")
ax.axis('off')
plt.show()



# watershed Algorithm
markers = cv2.watershed(img, markers)
 
fig, ax = plt.subplots(figsize=(5, 5))
ax.imshow(markers, cmap="tab20b")
ax.axis('off')
plt.show()
 
 
labels = np.unique(markers)
 
coins = []
for label in labels[2:]:  
 
# Create a binary image in which only the area of the label is in the foreground 
#and the rest of the image is in the background   
    target = np.where(markers == label, 255, 0).astype(np.uint8)
   
  # Perform contour extraction on the created binary image
    contours, hierarchy = cv2.findContours(
        target, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    coins.append(contours[0])
 
# Draw the outline
img = cv2.drawContours(img, coins, -1, color=(255, 0, 0), thickness=6)
imshow(img)








#input("Press Enter to exit..")
exit()

## Import Tiff image using TiffFile

import tifffile as tf

# image = cv2.imread('h2114153 h&e.tif', cv2.IMREAD_COLOR_RGB)
# image = cv2.imread('h2114153 h&e.tif', cv2.IMREAD_ANYDEPTH | cv2.IMREAD_COLOR)
# image = tf.imread('h2114153 h&e.tif', key=2)
tif = tf.TiffFile('h2114153 h&e.tif')
#image = tif.pages[2].asarray()
#print("Image Dimensions (Height, Width, Channels):", image.shape)

# cv2.imwrite('test.png', image, [cv2.IMWRITE_PNG_COMPRESSION , 5])
# print("Succes")
# exit()

image = tif.series[0].asarray()

img = image[88000:95000, 8500:11500, 0]

cv2.imwrite("h2114153 h&e series[0] Lumma img#3.png", img, [cv2.IMWRITE_PNG_COMPRESSION , 0])
print("Succes")

exit()

# Check if the image is loaded successfully
if image is None:
    print("Error: Could not read the image.")
else:
    # Print the image information
    print("Image Dimensions (Height, Width, Channels):", image.shape)
    print("Image Data Type:", image.dtype)

## Read smaler image size

for i in [0,1,2]:
    print("i = " + str(i))

    img = image[88000:95000, 8500:11500, i]

    ## Display Image

    import matplotlib.pyplot as plt

    plt.imshow(img)
    plt.title("h2114153 h&e.tif " + str(i+1))

    # Block execution until the figure is closed
    plt.show(block=True)

