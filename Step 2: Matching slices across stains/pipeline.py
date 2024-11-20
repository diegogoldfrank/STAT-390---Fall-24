from glob import glob
import os

import tkinter as tk
from tkinter import filedialog

import cv2
import numpy as np
from itertools import product, chain
from collections import defaultdict

from skimage import morphology
from PIL import Image


### GUI CODE ###

# close the gui once both selections are submitted
def submit():
    if selected['dir']:
        root.destroy()
    else:
        print('Select a directory.')


# open directory selecter
def select_dir():
    dir = filedialog.askdirectory(title = 'Select patient directory')
    if dir:
        selected['dir'] = dir
        dir_label.config(text = f'Selected directory: {dir}')


# dictionary to store user selection
selected = {'dir': None}

# initialize gui window
root = tk.Tk()
root.title('Patching Pipeline')
root.geometry('400x300')

# button to select patient directory
dir_button = tk.Button(root, text = 'Select patient directory', 
                       command = select_dir, bg = 'navy', fg = 'white')
dir_button.pack(pady = 5)

# display selected directory
dir_label = tk.Label(root, text = 'No directory selected')
dir_label.pack(pady = 0)

# add button to submit selections
submit_button = tk.Button(root, text = 'Submit', command = submit)
submit_button.pack(pady = 0)

# run the gui
root.mainloop()

if selected['dir']:
    patient_dir = selected['dir']
else:
    raise Exception('Please select a directory')

### END OF GUI CODE ###


### IMAGE FILE PREPROCESSING ###

# first thing we need to do is create a standardized patient ID for each patient
# this will be patient_id + strain_type
# we will rename the images to this standardized patient ID
def rename_files():
    # Get list of all .tif files in the patient directory
    image_files = glob(f'{patient_dir}/*.tif')

    for image_file in image_files:
        # looking at the current data, a patient ID is: patient_name + possible extra info + strain_type
        file_name_parts = image_file.split()

        # Get the patient ID from the filename and clean it up
        patient_id = file_name_parts[0]
        if ',' in patient_id:
            patient_id = patient_id.replace(',', '')
        if '.' in patient_id:
            patient_id = patient_id.replace('.', '')
        if len(file_name_parts) > 2 and 'mela' not in file_name_parts[1] and 'h&e' not in file_name_parts[1]:
            patient_id += file_name_parts[1][0]
        patient_id.replace('H', 'h')

        # Get the strain type from the filename and clean it up
        strain_type = None
        for part in file_name_parts:
            if 'mela' in part.lower():
                strain_type = 'melan'
            elif 'h&e' in part.lower():
                strain_type = 'h&e'
            elif 'sox10' in part.lower():
                strain_type = 'sox10'

        # Construct the new filename
        new_filename = f'{patient_id}_{strain_type}.tif'

        # Rename the file
        if not os.path.exists(new_filename):
            os.rename(image_file, new_filename)
        else:
            print(f"File {new_filename} already exists. Skipping.")


def delete_patients():
    # Get list of all .tif files in the patient directory
    image_files = glob(f'{patient_dir}/*.tif')

    # Get all patient IDs
    patient_ids = []
    for image_file in image_files:
        index = image_file.find("\\")
        result = image_file[index + 1:]
        patient_ids.append(result.split('_')[0])

    # Get all patient IDs with all 3 strain types
    patient_ids_with_all_strain_types = set()
    for patient_id in patient_ids:
        strain_types = []
        for image_file in image_files:
            index = image_file.find("\\")
            result = image_file[index + 1:]
            if patient_id in result:
                strain_type = result.split('_')[1]
                if strain_type not in strain_types:
                    strain_types.append(strain_type)
        if len(strain_types) >= 3:
            patient_ids_with_all_strain_types.add(patient_id)

    # Delete all patients without all 3 strain types
    for patient_id in patient_ids:
        if patient_id not in patient_ids_with_all_strain_types:
            for image_file in image_files:
                index = image_file.find("\\")
                result = image_file[index + 1:]
                if patient_id in result:
                    os.remove(image_file)


def keep_highest_res_image(files):
    # Get the resolution of each image
    resolutions = []
    for file in files:
        resolutions.append(os.path.getsize(file))
    
    # Get the index of the image with the highest resolution
    max_res_index = resolutions.index(max(resolutions))

    # Delete all images except the one with the highest resolution
    for i in range(len(files)):
        if i != max_res_index:
            os.remove(files[i])

def keep_highest_res():
    # Get list of all .tif files in the patient directory
    image_files = glob(f'{patient_dir}/*.tif')

    # Get all patient IDs
    patient_ids = []
    for image_file in image_files:
        index = image_file.find("\\")
        result = image_file[index + 1:]
        patient_ids.append(result.split('_')[0])

    # Get all patient IDs with all 3 strain types
    patient_ids_with_all_strain_types_multiple_of_one = set()
    patient_to_strains = dict()
    for patient_id in patient_ids:
        strain_types = []
        for image_file in image_files:
            index = image_file.find("\\")
            result = image_file[index + 1:]
            if patient_id in result:
                strain_type = result.split('_')[1]
                if strain_type not in strain_types:
                    strain_types.append(strain_type)
        if len(strain_types) > 3:
            patient_ids_with_all_strain_types_multiple_of_one.add(patient_id)
        patient_to_strains[patient_id] = strain_types
        

    # For all strains with mutliples, keep the one with the highest resolution
    for key in patient_to_strains.keys():
        h_and_e_count, melan_count, sox10_count = 0, 0, 0
        h_and_e_files, melan_files, sox10_files = [], [], []
        for item in patient_to_strains[key]:
            if 'h&e' in item:
                h_and_e_count += 1
                h_and_e_files.append(f'{patient_dir}\\' + key + "_" + item)
            if 'melan' in item:
                melan_count += 1
                melan_files.append(f'{patient_dir}\\' + key + "_" + item)
            if 'sox10' in item:
                sox10_count += 1
                sox10_files.append(f'{patient_dir}\\' + key + "_" + item)
        if h_and_e_count > 1:
            keep_highest_res_image(h_and_e_files)
        if melan_count > 1:
            keep_highest_res_image(melan_files)
        if sox10_count > 1:
            keep_highest_res_image(sox10_files)

def remove_last_char():
    image_files = glob(f'{patient_dir}/*.tif')
    for image_file in image_files:
        new_image_file = image_file[:-4]
        if new_image_file[-1] == '2':
            new_image_file = new_image_file[:-1]
        new_image_file += '.tif'
        os.rename(image_file, new_image_file)


def remove_small_files():
    # Get list of all .tif files in the patient directory
    image_files = glob(f'{patient_dir}/*.tif')

    # Remove all files less than 15 MB and greater than 1GB
    for image_file in image_files:
        if os.path.getsize(image_file) < 15000000 or os.path.getsize(image_file) > 1000000000:
            os.remove(image_file)

def rename_files():
    # Get list of all .tif files in the patient directory
    image_files = glob(f'{patient_dir}/*.tif')

    # Replace ROI with slice
    for image_file in image_files:
        new_filename = image_file.replace('ROI', 'slice')
        os.rename(image_file, new_filename)

# split up files by patient
def split_files_by_patient():
    # Get list of all .tif files in the patient directory
    image_files = glob(f'{patient_dir}/*.tif')

    patient_ids = set()
    for image in image_files:
        index = image.find("\\")
        result = image[index + 1:]
        patient_ids.add(result.split('_')[0])

    # Create a folder for each patient
    for patient_id in patient_ids:
        os.makedirs(os.path.join(patient_dir, patient_id), exist_ok = True)


    # Move each image to the correct patient folder
    for image_file in image_files:
        index = image_file.find("\\")
        result = image_file[index + 1:]
        patient_id = result.split('_')[0]
        os.rename(image_file, f'{patient_dir}/{patient_id}/{result}')

def preprocess_files():
    rename_files()

    delete_patients()

    keep_highest_res()

    remove_last_char()

    remove_small_files()

    rename_files()

    split_files_by_patient()

### END OF IMAGE FILE PREPROCESSING ###



### MATCHING ALGORITHM ###

def read_unmatched(folder):
    image_dict = defaultdict(list)

    imgs = [cv2.imread(os.path.join(folder, img)) for img in os.listdir(folder)]

    for img, name in zip(imgs, os.listdir(folder)):
        stain = 'h&e' if 'h&e' in name.lower() else 'melan' if 'melan' in name.lower() else 'sox10'

        image_dict[stain].append(img)

    # reorder image sets such that h&e is first
    return dict(sorted(image_dict.items()))


def extract_main_contour(image):
    # convert image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # apply Gaussian Blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # adaptive threshold to handle variations in color intensity
    adaptive_thresh = cv2.adaptiveThreshold(blurred, 255,
                                            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                            cv2.THRESH_BINARY_INV, 11, 2)

    # apply morphological operations to clean up image
    kernel = np.ones((10, 10), np.uint8)
    morph = cv2.morphologyEx(adaptive_thresh, cv2.MORPH_CLOSE, kernel)

    # find contours
    contours, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # return the largest contour
    return max(contours, key = cv2.contourArea)


def distance(image1, image2):
    # extract main contours
    contour1 = extract_main_contour(image1)
    contour2 = extract_main_contour(image2)

    # calculate the area of both contours
    area1 = cv2.contourArea(contour1)
    area2 = cv2.contourArea(contour2)
    
    # calculate the area of the images
    image1_area, image2_area = image1.shape[0] * image1.shape[1], image2.shape[0] * image2.shape[1]

    # total area of contour
    contour1_percentage, contour2_percentage = area1 / image1_area, area2 / image2_area

    # if the contour area area aren't within 30% of each other, return maximum distance
    if abs(contour1_percentage - contour2_percentage) > 0.3:
        return np.inf
    
    # return distance score between the two contours
    return cv2.matchShapes(contour1, contour2, cv2.CONTOURS_MATCH_I1, 0.0)


def distance_matrix(image_dict):
    # set of images for each stain
    set1, set2, set3 = image_dict.values()

    # empty distance matrix to fill in
    matrix = np.zeros((len(set1), len(set2), len(set3)))

    # iterate through all possible 3-way matches
    for (i, img1), (j, img2), (k, img3) in product(enumerate(set1), enumerate(set2), enumerate(set3)):
        # compute distance score and fill in matrix
        matrix[i, j, k] = distance(img1, img2) + distance(img2, img3) + distance(img1, img3)

    return matrix


def match(images, matches = None):
    if not matches:
        matches = []

    # check if all stains have at least one image present
    if any(len(imgs) == 0 for imgs in images.values()):
        return matches

    distances = distance_matrix(images)

    min_idx = np.unravel_index(distances.argmin(), distances.shape)

    matched_images = []

    for stain, idx in zip(images.keys(), min_idx):
        matched_images.append(images[stain][idx])

        del images[stain][idx]

    matches.append(matched_images)

    # recursively call the function on the remaining images
    return match(images, matches)


def write_tif(img, path):
    cv2.imwrite(path,
                img,
                [cv2.IMWRITE_TIFF_COMPRESSION,
                 cv2.IMWRITE_TIFF_COMPRESSION_NONE])
    

def write_matched(matches, unmatched, patient):
    base_dir = os.path.join('matches', patient)

    os.makedirs(base_dir, exist_ok = True)

    # iterate through 3-way matches
    for i, match in enumerate(matches, 1):
        match_dir = os.path.join(base_dir, f'match{i}')

        os.makedirs(match_dir, exist_ok = True)

        for j, slice in enumerate(match, 1):
            write_tif(slice, os.path.join(match_dir, f'slice{j}.tif'))

    # also save unmatched images
    slices = list(chain.from_iterable(unmatched.values()))

    if len(slices) > 0:
        unmatched_dir = os.path.join(base_dir, 'unmatched')

        os.makedirs(unmatched_dir, exist_ok = True)

        for i, slice in enumerate(slices, 1):
            write_tif(slice, os.path.join(unmatched_dir, f'slice{i}.tif'))
                

def match_pipeline(folder):
    for patient in os.listdir(folder):
        try:
            image_dict = read_unmatched(os.path.join(folder, patient))

            matches = match(image_dict)

            write_matched(matches, image_dict, patient)
        except:
            continue

### END OF MATCHING ALGORITHM ###


### ROTATION ALGORITHM ###

def crop(img):
    contour = extract_main_contour(img)

    x, y, w, h = cv2.boundingRect(contour)

    mask = np.zeros_like(img)
    
    cv2.drawContours(mask, [contour], 0, (255, 255, 255), thickness = cv2.FILLED)
    
    intersected = cv2.bitwise_and(mask, img)

    return intersected[y:y + h, x:x + w]


def resize(img1, img2):
    contour_areas = list(map(lambda x: cv2.contourArea(extract_main_contour(x)), [img1, img2]))

    if contour_areas[0] < contour_areas[1]:
        return cv2.resize(img1, img2.shape[1::-1]), img2
    
    return img1, cv2.resize(img2, img1.shape[1::-1])


def rotate_image(image, angle):
    h, w = image.shape[:2]
    center = (w // 2, h // 2)

    # Calculate new dimensions after rotation
    radians = np.deg2rad(angle)
    new_w = int(abs(w * np.cos(radians)) + abs(h * np.sin(radians)))
    new_h = int(abs(h * np.cos(radians)) + abs(w * np.sin(radians)))

    # Update the rotation matrix for the new center and dimensions
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotation_matrix[0, 2] += (new_w - w) / 2
    rotation_matrix[1, 2] += (new_h - h) / 2

    return cv2.warpAffine(image, rotation_matrix, (new_w, new_h))


def maximize_overlap(img1, img2, stepsize, cropped = False, angles = np.arange(0, 360, 90)):
    # crop the images to their ROIs if not already done
    if not cropped:
        img1, img2 = crop(img1), crop(img2)

    # calculate overlap score for each angle
    scores = np.array([overlap_score(img1, img2, angle) for angle in angles])

    # find angle that maximizes overlap score
    best_angle = angles[np.argmax(scores)]

    curr_stepsize = angles[1] - angles[0]

    # base case
    if curr_stepsize <= stepsize:
        return best_angle

    # zoom in on the best angle from the previous function call
    new_angles = np.arange(best_angle - curr_stepsize, best_angle + curr_stepsize + 1, np.ceil(curr_stepsize/2))

    # recursively call the function on the updated array of angles
    return maximize_overlap(img1, img2, stepsize, True, new_angles)


def overlap_score(img1, img2, angle):
    # rotate and crop image 1
    img1 = crop(rotate_image(img1, angle))

    # get image dimensions
    heights = [img1.shape[0], img2.shape[0]]
    widths = [img1.shape[1], img2.shape[1]]

    # ensure that both dimensions of the rotated and reference images are within 50% of one another
    if max(max(heights)/min(heights), max(widths)/min(widths)) > 1.5:
        return 0
    
    # resize the images 
    img1, img2 = resize(img1, img2)

    intersection = cv2.bitwise_and(img1, img2)
    union = cv2.bitwise_or(img1, img2)
    
    return np.sum(intersection) / np.sum(union)


def align_images(imgs, stepsize = 1):
    img1, img2, img3 = imgs

    # align the images with respect to the h&e image (img1)
    best_angle1 = maximize_overlap(img2, img1, stepsize)
    best_angle2 = maximize_overlap(img3, img1, stepsize)

    img2 = crop(rotate_image(crop(img2), best_angle1))
    img3 = crop(rotate_image(crop(img3), best_angle2))

    img1 = crop(img1)
    
    return img1, img2, img3


def read(dir):
    return [cv2.imread(os.path.join(dir, slice)) for slice in os.listdir(dir)]

### END OF ROTATION ALGORITHM ###


### EPITHELIUM EXTRACTION ###

def extract_epithelium(img, output_dir):
    # Convert the image to RGB and YCrCb
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_ycrcb = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2YCrCb)
    
    # Binning the luminance (Y) channel
    lumma_bins_n = 20
    divisor = np.floor(255 / lumma_bins_n).astype(np.uint8)
    lumma_binned = np.floor(img_ycrcb[:, :, 0] / divisor).astype(np.uint8)
    
    # Find the most common luminance bin
    most_pixels_bin = -1
    most_pixels = 0
    for bin_i in range(0, lumma_bins_n + 1):
        n_pixels = np.count_nonzero(lumma_binned == bin_i)
        if n_pixels > most_pixels:
            most_pixels = n_pixels
            most_pixels_bin = bin_i

    background_bin = most_pixels_bin
    background = lumma_binned == background_bin
    background = morphology.remove_small_objects(background, 5000)
    background = morphology.remove_small_holes(background, 10000)

    # Binning the Cr (Red Chroma) channel
    Cr_bins_n = 50
    divisor = np.floor(255 / Cr_bins_n).astype(np.uint8)
    Cr_binned = np.floor(img_ycrcb[:, :, 2] / divisor).astype(np.uint8)
    
    # Find the most common Cr bin
    most_pixels_bin = -1
    most_pixels = 0
    for bin_i in range(0, Cr_bins_n + 1):
        n_pixels = np.count_nonzero(Cr_binned == bin_i)
        if n_pixels > most_pixels:
            most_pixels = n_pixels
            most_pixels_bin = bin_i

    # Stroma mask generation
    stroma_bin = most_pixels_bin
    stroma = Cr_binned == stroma_bin
    stroma = stroma + (Cr_binned == stroma_bin - 1)
    stroma = stroma + (Cr_binned == stroma_bin - 2)
    stroma = stroma * np.invert(background)
    stroma = morphology.dilation(stroma, morphology.square(3))
    stroma = morphology.remove_small_objects(stroma, 1000)

    # Epithelia mask generation
    epithelia_bin = stroma_bin + 2
    epithelia = Cr_binned == epithelia_bin
    epithelia = epithelia + (Cr_binned == epithelia_bin + 1)
    epithelia = epithelia + (Cr_binned == epithelia_bin + 2)
    epithelia = epithelia + (Cr_binned == epithelia_bin + 3)
    epithelia = epithelia + (Cr_binned == epithelia_bin + 4)
    epithelia = epithelia * np.invert(background)
    epithelia = epithelia * np.invert(stroma)
    epithelia = epithelia * np.invert(img_ycrcb[:, :, 1] < 120)
    epithelia = morphology.dilation(epithelia, morphology.square(2))
    epithelia = morphology.remove_small_objects(epithelia, 500)
    epithelia = morphology.remove_small_holes(epithelia, 10000)

    # Adjust stroma to only show areas not covered by epithelia
    stroma_only = stroma * np.invert(epithelia)  

    # Apply the masks to the original image 
    epithelia_img = cv2.bitwise_and(img_rgb, img_rgb, mask=(epithelia.astype(np.uint8) * 255))
    stroma_img = cv2.bitwise_and(img_rgb, img_rgb, mask=(stroma_only.astype(np.uint8) * 255))

    # Combine both epithelia and stroma images
    combined_img = cv2.add(epithelia_img, stroma_img)

    #################################
    if output_dir == 'output_dir':
        return epithelia_img, stroma_img, combined_img
    #################################

    # Save the combined image
    combined_filename = os.path.join(output_dir, "epithelia_and_stroma_combined.tif")
    cv2.imwrite(combined_filename, cv2.cvtColor(combined_img, cv2.COLOR_RGB2BGR))

    # Save the individual epithelium and stroma images if desired
    epithelia_filename = os.path.join(output_dir, "epithelia.tif")
    cv2.imwrite(epithelia_filename, cv2.cvtColor(epithelia_img, cv2.COLOR_RGB2BGR))

    stroma_filename = os.path.join(output_dir, "stroma.tif")
    cv2.imwrite(stroma_filename, cv2.cvtColor(stroma_img, cv2.COLOR_RGB2BGR))


def write(imgs, patient, match_name):
    output_dir = os.path.join('matches', patient, match_name)
    os.makedirs(output_dir, exist_ok = True)

    for i, img in enumerate(imgs, 1):
        # Save the aligned images
        cv2.imwrite(os.path.join(output_dir, f'slice{i}.tif'), img)

        # Only apply extract_epithelium on slice1 
        if i == 1:
            extract_epithelium(img, output_dir)

### END OF EPITHELIUM EXTRACTION ###

### PATCHING ALGORITHM ###

def pad_images(image1, image2, image3):
    # convert images to PIL format
    image1 = Image.fromarray(cv2.cvtColor(image1, cv2.COLOR_BGR2RGB))
    image2 = Image.fromarray(cv2.cvtColor(image2, cv2.COLOR_BGR2RGB))
    image3 = Image.fromarray(cv2.cvtColor(image3, cv2.COLOR_BGR2RGB))

    # get the maximum width and height - this is what we will pad all images to
    max_width = max(image1.width, image2.width, image3.width)
    max_height = max(image1.height, image2.height, image3.height)

    # function to pad the image with black pixels
    def pad_image(image, target_size):
        padded_image = Image.new("RGB", target_size, (0, 0, 0))
        left = (target_size[0] - image.width) // 2
        top = (target_size[1] - image.height) // 2
        padded_image.paste(image, (left, top))
        return padded_image

    # pad all images to the maximum width and height
    padded_image1 = pad_image(image1, (max_width, max_height))
    padded_image2 = pad_image(image2, (max_width, max_height))
    padded_image3 = pad_image(image3, (max_width, max_height))

    # convert images back to cv2 format
    padded_image1 = cv2.cvtColor(np.array(padded_image1), cv2.COLOR_RGB2BGR)
    padded_image2 = cv2.cvtColor(np.array(padded_image2), cv2.COLOR_RGB2BGR)
    padded_image3 = cv2.cvtColor(np.array(padded_image3), cv2.COLOR_RGB2BGR)

    return padded_image1, padded_image2, padded_image3


def extract_patches(he_image, slice2, slice3, patch_width, patch_height):
    output_dir = 'output_dir' #not needed, but used in extract_epithelium function, so I modified it a little
    # first pad the images
    image1, image2, image3 = pad_images(he_image, slice2, slice3)
    # then extract the epithelium and stroma images
    epithelia_img, stroma_img, combined_img = extract_epithelium(image1, output_dir)
        
    # Convert to grayscale and binarize the combined image
    gray = cv2.cvtColor(combined_img, cv2.COLOR_BGR2GRAY)
    binary = np.array(gray) > 0  # Non-black areas become True
    image_pil = Image.fromarray(cv2.cvtColor(image1, cv2.COLOR_BGR2RGB))
        
    # Skeletonize the shape to get the centerline
    skeleton = morphology.skeletonize(binary)
        
    # Find the coordinates of skeleton pixels
    y, x = np.where(skeleton)
    skeleton_coords = list(zip(x, y))[::100]
        
    # Calculate gradients at each skeleton point to determine the tangent direction
    patches = []
    coords = []
    for i, (x, y) in enumerate(skeleton_coords[:-1]):
        # Compute approximate direction to next skeleton point (dx, dy)
        dx = skeleton_coords[i + 1][0] - x
        dy = skeleton_coords[i + 1][1] - y
        length = np.sqrt(dx**2 + dy**2)
        if length == 0:
            continue  # Skip if there's no movement
            
        # Normalize direction to get unit vector
        dx /= length
        dy /= length
            
        # Compute perpendicular vector for the patch orientation
        perp_dx = -dy
        perp_dy = dx
            
        # Calculate corner points of the patch rectangle
        half_width = patch_width // 2
        half_height = patch_height // 2
        corners = [
            (x + perp_dx * half_width, y + perp_dy * half_height),
            (x - perp_dx * half_width, y - perp_dy * half_height),
            (x + perp_dx * half_width + dx * patch_height, y + perp_dy * half_height + dy * patch_height),
            (x - perp_dx * half_width + dx * patch_height, y - perp_dy * half_height + dy * patch_height),
        ]
            
        # Crop the patch from the image using the bounding box of the rotated rectangle
        min_x = int(min(c[0] for c in corners))
        max_x = int(max(c[0] for c in corners))
        min_y = int(min(c[1] for c in corners))
        max_y = int(max(c[1] for c in corners))
        patch = image_pil.crop((min_x, min_y, max_x, max_y))
            
        # Check if the patch contains parts from the stroma and epithelium image
        patch_stroma = stroma_img[min_y:max_y, min_x:max_x]
        patch_epithelia = epithelia_img[min_y:max_y, min_x:max_x]
        
        # Check if patch contains both stroma and epithelia
        patch_np = np.array(patch)
        if np.any(patch_stroma > 0) and np.any(patch_epithelia > 0):
            # Check if the patch has between 10-70% black pixels - this is approximately the range we want
            black_pixels = np.sum(np.all(patch_np == [0, 0, 0], axis=-1))
            total_pixels = patch_np.shape[0] * patch_np.shape[1]
            if black_pixels / total_pixels > 0.1 and black_pixels / total_pixels < 0.7:
                patches.append(patch)
                coords.append((min_x, min_y, max_x, max_y))

    # Check for overlapping patches and remove them
    non_overlapping_patches = []
    non_overlapping_coords = []
    for i, coord1 in enumerate(coords):
        overlap = False
        for j, coord2 in enumerate(non_overlapping_coords):
            if not (coord1[2] < coord2[0] or coord1[0] > coord2[2] or coord1[3] < coord2[1] or coord1[1] > coord2[3]):
                overlap = True
                break
        if not overlap:
            non_overlapping_patches.append(patches[i])
            non_overlapping_coords.append(coord1)
    patches = non_overlapping_patches
    coords = non_overlapping_coords

    return patches, coords, image1, image2, image3


def get_patches_across_stains(patches, coords, image1, image2, image3):
    # Check the same region across all stains
    patches_across_stains = []
    for i, patch in enumerate(patches):
        min_x, min_y, max_x, max_y = coords[i]
        patch_across_stains = []
        patch_across_stains.append(image1[min_y:max_y, min_x:max_x])
        patch_across_stains.append(image2[min_y:max_y, min_x:max_x])
        patch_across_stains.append(image3[min_y:max_y, min_x:max_x])
        # Check if the patch contains parts from all three images
        if all(np.any(patch > 0) for patch in patch_across_stains):
            # Check if the patch has between 5-80% black pixels - this is approximately the range we want
            black_pixels = [np.sum(np.all(patch == [0, 0, 0], axis=-1)) for patch in patch_across_stains]
            total_pixels = patch_across_stains[0].shape[0] * patch_across_stains[0].shape[1]
            black_pixel_percentages = [black_pixel / total_pixels for black_pixel in black_pixels]
            if all(0.05 <= percentage <= 0.8 for percentage in black_pixel_percentages):
                patches_across_stains.append(patch_across_stains)

    return patches_across_stains


def rotate_extract_patch(match_dir):
    for patient in os.listdir(match_dir):
        for match in os.listdir(os.path.join(match_dir, patient)):
            if match.startswith('match'):
                imgs = read(os.path.join(match_dir, patient, match))

                aligned_images = align_images(imgs)

                write(aligned_images, patient, match)

                a1, a2, a3 = aligned_images

                # Extract patches from the aligned images
                patches, coords, image1, image2, image3 = extract_patches(a1, a2, a3, patch_width = 800, patch_height = 400)

                # Get patches across stains
                patches_across_stains = get_patches_across_stains(patches, coords, image1, image2, image3)

                # Save the patches
                output_dir = os.path.join('patches', patient, match)
                os.makedirs(output_dir, exist_ok = True)
                for i, patch_set in enumerate(patches_across_stains, 1):
                    patch_dir = os.path.join(output_dir, f'patch{i}')

                    os.makedirs(patch_dir, exist_ok = True)

                    for j, patch in enumerate(patch_set, 1):
                        cv2.imwrite(os.path.join(patch_dir, f'stain{j}.tif'), patch)

### END OF PATCHING ALGORITHM ###


### PIPELINE ###
preprocess_files()

match_pipeline(patient_dir)

rotate_extract_patch('matches')