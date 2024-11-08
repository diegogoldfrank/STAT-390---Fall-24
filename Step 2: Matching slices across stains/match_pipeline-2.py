from glob import glob
import os

import tkinter as tk
from tkinter import filedialog

import cv2
import numpy as np
from itertools import product, chain
from collections import defaultdict


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
root.title('Tissue Match Pipeline')
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

def read(folder):
    image_dict = defaultdict(list)

    imgs = [cv2.imread(os.path.join(folder, img)) for img in os.listdir(folder)]

    for img, name in zip(imgs, os.listdir(folder)):
        stain = 'h&e' if 'h&e' in name.lower() else 'melan' if 'melan' in name.lower() else 'sox10'

        image_dict[stain].append(img)

    return image_dict


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
    

def write(matches, unmatched, patient):
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
                

def pipeline(folder):
    for patient in os.listdir(folder):
        try:
            image_dict = read(os.path.join(folder, patient))

            matches = match(image_dict)

            write(matches, image_dict, patient)
        except:
            continue

### END OF MATCHING ALGORITHM ###


### PIPELINE ###
preprocess_files()
print('Finished preprocessing files')

pipeline(patient_dir)