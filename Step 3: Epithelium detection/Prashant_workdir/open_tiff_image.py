
## Import OpenCV

import sys
sys.path.append('/usr/local/lib/python3.8/site-packages')
import cv2
print("OpenCV Version = " + cv2.__version__)


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

