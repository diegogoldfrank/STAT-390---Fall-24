
## Import OpenCV

import sys
sys.path.append('/usr/local/lib/python3.8/site-packages')
import cv2
import numpy as np
from IPython.display import Image, display
from matplotlib import pyplot as plt


# Plot the image
def imshow(img, ax=None):
    if ax is None:
        #ret, encoded = cv2.imencode(".jpg", img)
        #display(Image(encoded))
        plt.imshow(img)
        plt.axis('off')
        plt.show(block=True)
    else:
        ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        ax.axis('off')
        ax.show(block=True)

 
#Image loading
img = cv2.imread("h2114153 h&e series[0] Lumma img#3.png")
# Show image
print("img")
imshow(img)

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

