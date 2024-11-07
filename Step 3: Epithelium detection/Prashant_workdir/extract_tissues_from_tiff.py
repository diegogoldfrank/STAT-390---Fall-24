

##########################################

##       ENTER PATIENT IMAGE ID         ##

patient_id = "h2114154 h&e"

##########################################

## Import OpenCV

import sys
sys.path.append('/usr/local/lib/python3.8/site-packages')
import cv2
print("OpenCV Version = " + cv2.__version__)
import matplotlib.pyplot as plt
from skimage import morphology
from skimage import filters
import os
import shutil

# Plot the image
def imshow(img, title):
    plt.imshow(img)
    plt.axis('off')
    plt.title(title)
    plt.show(block=True)

## Import Tiff image using TiffFile

import tifffile as tf

data_dir = "data"
extracted_dir = "extracted"

if not os.path.isdir(data_dir):
    os.mkdir(data_dir)
    print("'" + data_dir + "' directory created. Please put patient image data.")
    exit()

if not os.path.isdir(extracted_dir):
    os.mkdir(extracted_dir)
    print("extracted_dir: '" + extracted_dir + "' directory created.")
    exit()

if os.path.isdir(os.path.join(extracted_dir, patient_id)):
    # delete current contents
    for root, dirs, files in os.walk(os.path.join(extracted_dir, patient_id)):
        for f in files:
            print("Output dir: '" + os.path.join(extracted_dir, patient_id) + "' made empty.")
        
    del root, dirs, files
else:
    # create dir
    os.mkdir(os.path.join(extracted_dir, patient_id))
    print("Output dir: '" + os.path.join(extracted_dir, patient_id) + "' created.")

if not os.path.exists(os.path.join(data_dir, patient_id + ".tif")):
    print("Input file: '" + os.path.join(data_dir, patient_id + ".tif") + "' does not exist.")
    exit()

tif = tf.TiffFile(os.path.join(data_dir, patient_id + ".tif"))

series_num = 0
image = tif.series[series_num].asarray()

# sub image finding and cutting operation

# Convert to grayscale
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Get the original image dimensions
height, width = gray_image.shape[:2]

# Define the new dimensions (downscaling by half)
downscale_factor = 10
new_width = int(width / downscale_factor)
new_height = int(height / downscale_factor)

# Downscale the image using INTER_AREA interpolation
downscaled_image = cv2.resize(gray_image, (new_width, new_height), interpolation=cv2.INTER_AREA)
#imshow(downscaled_image, 'downscaled_image')


# gaussian filter
gaussian_smooth_gray = filters.gaussian(downscaled_image, sigma=10)#, preserve_range=True)
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
filename = os.path.join(extracted_dir, patient_id, patient_id + " 2D mask.png")
plt.savefig(filename)
print("'" + filename + "' saved!")
plt.close()

# sub image finding and cutting operation
from skimage.measure import label, regionprops, regionprops_table
label_img = label(g4)
regions = regionprops(label_img)

for props in regions:
    min_row, min_col, max_row, max_col = props.bbox
    print(f"Label {props.label}:")
    print(f"  Min coordinates: ({min_row}, {min_col})")
    print(f"  Max coordinates: ({max_row}, {max_col})")
    sub_image_downscaled = downscaled_image[min_row:max_row, min_col:max_col]
    #imshow(sub_image_downscaled, f"Label {props.label}:")
    sub_image = image[min_row*10:min(max_row*10,height), min_col*10:min(max_col*10,width)]
    filename = os.path.join(extracted_dir, patient_id, patient_id + " series[" + str(series_num) + "] img#" + str(props.label) + ".png")
    cv2.imwrite(filename, cv2.cvtColor(sub_image, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_PNG_COMPRESSION , 0])
    print("'" + filename + "' saved!")
    sub_img_lumma = sub_image[:, :, 0]
    filename = os.path.join(extracted_dir, patient_id, patient_id + " series[" + str(series_num) + "] img#" + str(props.label) + " Lumma.png")
    cv2.imwrite(filename, sub_img_lumma, [cv2.IMWRITE_PNG_COMPRESSION , 0])
    print("'" + filename + "' saved!")

print("done!")
