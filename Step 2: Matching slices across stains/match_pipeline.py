from glob import glob
import os

import tkinter as tk
from tkinter import filedialog

import cv2
import numpy as np
import itertools


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
    # Get list of all .tif files in the processed_images folder
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
    # Get list of all .tif files in the processed_images folder
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
    # Get list of all .tif files in the processed_images folder
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
    # Get list of all .tif files in the processed_images folder
    image_files = glob('processed_images/*.tif')

    # Remove all files less than 15 MB and greater than 1GB
    for image_file in image_files:
        if os.path.getsize(image_file) < 15000000 or os.path.getsize(image_file) > 1000000000:
            os.remove(image_file)

def rename_files():
    # Get list of all .tif files in the processed_images folder
    image_files = glob('processed_images/*.tif')

    # Replace ROI with slice
    for image_file in image_files:
        new_filename = image_file.replace('ROI', 'slice')
        os.rename(image_file, new_filename)

# split up files by patient
def split_files_by_patient():
    # Get list of all .tif files in the processed_images folder
    image_files = glob('processed_images/*.tif')

    patient_ids = set()
    for image in image_files:
        index = image.find("\\")
        result = image[index + 1:]
        patient_ids.add(result.split('_')[0])

    # Create a folder for each patient
    for patient_id in patient_ids:
        if not os.path.exists(f'processed_images/{patient_id}'):
            os.makedirs(f'processed_images/{patient_id}')

    # Move each image to the correct patient folder
    for image_file in image_files:
        index = image_file.find("\\")
        result = image_file[index + 1:]
        patient_id = result.split('_')[0]
        os.rename(image_file, f'processed_images/{patient_id}/{result}')

def preprocess_files():
    rename_files()

    delete_patients()

    keep_highest_res()

    remove_last_char()

    remove_small_files()

    rename_files()

    split_files_by_patient()

### END OF IMAGE FILE PREPROCESSING ###



### FAITH MATCHING ALGORITHM ###

def extract_images(folder_path, patient):
    '''
    folder_path: a path to a folder of all patients
    patient: the patient id
    '''
    raw_images = {}
    masked_images = {}

    path = os.path.join(folder_path, patient)
    images = os.listdir(path)

    all_files = []
    all_labels = []

    for i in images:
        all_files.append(os.path.join(path, i))
        all_labels.append(f"{i}")

    print(f"Total images: {len(all_files)}")

    names_concat = ''.join(images)

    if len(all_files) < 3 or not all(['h&e' in names_concat, 'melan' in names_concat, 'sox10' in names_concat]):
        return (None, None)

    for idx, image_path in enumerate(all_files):
        image = cv2.imread(image_path)

        if image is None:
            print(f"Error loading image: {image_path}")
            continue
        
        # Extracting image name
        image_name = os.path.basename(all_labels[idx])

        raw_images[image_name] = image

        # Converting to grayscale
        image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Enhance contrast to improve contour accuracy
        image_gray = cv2.equalizeHist(image_gray)

        # Blurring the image to reduce noise
        blurred = cv2.GaussianBlur(image_gray, (5, 5), 0)
        
        if 'melan' in image_name:
            adaptive_thresholding = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY, 11, 1)
        elif 'sox10' in image_name:
            adaptive_thresholding = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY, 13, 1)
        else: 
            # Adaptive thresholding used to seperate foreground objects from the background
            adaptive_thresholding = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                           cv2.THRESH_BINARY, 13, 1.8)

        # Find initial contours and create binary mask
        contours, _ = cv2.findContours(adaptive_thresholding, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        mask = np.zeros_like(adaptive_thresholding)
        cv2.drawContours(mask, contours, -1, (255), thickness=cv2.FILLED)

        # Blurring to reduce noise
        blurred_thresh = cv2.boxFilter(adaptive_thresholding, -1, (111, 111))
        
        # Create a mask for non-white areas directly from the binary image
        lower_white = 200 
        mask_binary = blurred_thresh > lower_white
        
        # Convert the boolean mask to a binary mask so it saves in the correct form
        mask_binary_image = (mask_binary.astype(np.uint8) * 255)

        # Save the new masked image to the dictionary
        masked_images[image_name] = mask_binary_image

    return raw_images, masked_images


def calculate_shape_similarity(image1, image2, image3):
    # Invert images
    image1 = cv2.bitwise_not(image1)
    image2 = cv2.bitwise_not(image2)
    image3 = cv2.bitwise_not(image3)

    # Find contours
    contours1, _ = cv2.findContours(image1, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours2, _ = cv2.findContours(image2, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours3, _ = cv2.findContours(image3, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)


    if contours1 and contours2 and contours3:
        # Get the two largest contours for each set
        sorted_contours1 = sorted(contours1, key=cv2.contourArea, reverse=True)[:2]
        sorted_contours2 = sorted(contours2, key=cv2.contourArea, reverse=True)[:2]
        sorted_contours3 = sorted(contours3, key=cv2.contourArea, reverse=True)[:2]

        similarity_second1 = similarity_second2 = similarity_second3 = None
        similarity_largest1 = similarity_largest2 = similarity_largest3 = 5
        # Ensure there is at least one contour to compare
        if len(sorted_contours1) > 0 and len(sorted_contours2) > 0:
            similarity_largest1 = cv2.matchShapes(sorted_contours1[0], sorted_contours2[0], cv2.CONTOURS_MATCH_I1, 0.0)
            if len(sorted_contours1) == 2 and len(sorted_contours2) == 2:
                similarity_second1 = cv2.matchShapes(sorted_contours1[1], sorted_contours2[1], cv2.CONTOURS_MATCH_I1, 0.0)

        if len(sorted_contours1) > 0 and len(sorted_contours3) > 0:
            similarity_largest2 = cv2.matchShapes(sorted_contours1[0], sorted_contours3[0], cv2.CONTOURS_MATCH_I1, 0.0)
            if len(sorted_contours1) == 2 and len(sorted_contours3) == 2:
                similarity_second2 = cv2.matchShapes(sorted_contours1[1], sorted_contours3[1], cv2.CONTOURS_MATCH_I1, 0.0)

        if len(sorted_contours2) > 0 and len(sorted_contours3) > 0:
            similarity_largest3 = cv2.matchShapes(sorted_contours2[0], sorted_contours3[0], cv2.CONTOURS_MATCH_I1, 0.0)
            if len(sorted_contours2) == 2 and len(sorted_contours3) == 2:
                similarity_second3 = cv2.matchShapes(sorted_contours2[1], sorted_contours3[1], cv2.CONTOURS_MATCH_I1, 0.0)

        # Compute the final similarity
        if all([similarity_second1, similarity_second2, similarity_second3]):
            if (similarity_second1 + similarity_second2 + similarity_second3) <= 0.5:
                return (((similarity_largest1 + similarity_largest2 + similarity_largest3) / 3) - 0.1)
            
        return ((similarity_largest1 + similarity_largest2 + similarity_largest3) / 3)



def calculate_all_similarities(raw_images, image_dict, matches = None):
    if not matches:
        matches = []
        
    similarity_scores = {}

    # Generate all combinations of three images
    for (key1, value1), (key2, value2), (key3, value3) in itertools.combinations(image_dict.items(), 3):
        key_string = key1 + key2 + key3
        # Checks if group contains all three stains
        if 'h&e' in key_string and 'melan' in key_string and 'sox10' in key_string:
            similarity = calculate_shape_similarity(value1, value2, value3)

            if similarity is not None and similarity < 0.6 * 0.9 ** len(matches):
                similarity_scores[(key1, key2, key3)] = similarity

    if similarity_scores:
        min_similarity = min(similarity_scores.values())
        
        # Find the corresponding image group
        min_group = min(similarity_scores, key = similarity_scores.get)
        
        print(f"Minimum similarity score: {min_similarity:.4f} between images: {min_group}")
        
        matches.append({name:raw_images[name] for name in min_group})

        for img in min_group:
            del image_dict[img]
            del raw_images[img]

        return calculate_all_similarities(raw_images, image_dict, matches)
        
    if len(matches) > 0:
        return matches
    
    print("No similar groups found - using alternative algorithm")
    return None


### END OF FAITH MATCHING ALGORITHM ###



### NOAH MATCHING ALGORITHM ###

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
    kernel = np.ones((15, 15), np.uint8)
    morph = cv2.morphologyEx(adaptive_thresh, cv2.MORPH_CLOSE, kernel)
    
    # find contours
    contours, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # return the largest contour
    return max(contours, key = cv2.contourArea)


# compute shape similarity between two contours
def shape_similarity(image1, image2):
    # extract main contours
    contour1 = extract_main_contour(image1)
    contour2 = extract_main_contour(image2)
    # matchShapes returns distance between shapes, take the inverse for similarity
    return 1/cv2.matchShapes(contour1, contour2, cv2.CONTOURS_MATCH_I1, 0.0) if cv2.matchShapes(contour1, contour2, cv2.CONTOURS_MATCH_I1, 0.0) > 0 else np.inf

def read(folder_path):
    images = {}

    for stain in os.listdir(folder_path):
        images[stain] = cv2.imread(os.path.join(folder_path, stain))

    return images


def find_groups(folder):
    images = read(folder)
    he_images, melan_images, sox10_images = {}, {}, {}
    for i in images.items():
        if 'h&e' in i[0]:
            he_images[i[0]] = i[1]
        elif 'melan' in i[0]:
            melan_images[i[0]] = i[1]
        elif 'sox10' in i[0]:
            sox10_images[i[0]] = i[1]

    return he_images, melan_images, sox10_images

# Find all 3 way intersectionS from different sets
def find_best_groups(he_images, melan_images, sox10_images):
    best_groups = []
    for i in he_images.items():
        for j in melan_images.items():
            for k in sox10_images.items():
                similarity_score = shape_similarity(i[1], j[1]) + shape_similarity(i[1], k[1]) + shape_similarity(j[1], k[1])
                best_groups.append((i[0], j[0], k[0], similarity_score))
    best_groups.sort(key = lambda x: x[3], reverse = True)
    return best_groups

# Now remove all but the best groups
# The way I will do this is to look at the 1st group, and remove all other groups that have the same image
def remove_duplicates(groups):
    new_groups = []
    used_images = set()
    for i in range(len(groups)):
        if groups[i][0] in used_images or groups[i][1] in used_images or groups[i][2] in used_images:
            continue
        used_images.add(groups[i][0])
        used_images.add(groups[i][1])
        used_images.add(groups[i][2])
        new_groups.append(groups[i])

    return new_groups

def make_new_image_groups(folder):
    he_images, melan_images, sox10_images = find_groups(folder)
    best_groups = find_best_groups(he_images, melan_images, sox10_images)
    best_groups = remove_duplicates(best_groups)
    
    return best_groups

### END OF NOAH MATCHING ALGORITHM ###



### MATCHING ALGORITH - FAITH + NOAH ###

def process_all_patients(folder_path):
    patient_folders = os.listdir(folder_path)

    for patient in patient_folders:
        patient_path = os.path.join(folder_path, patient)
        if os.path.isdir(patient_path):
            raw_images, image_dict = extract_images(folder_path, patient)

            if image_dict:
                similarity_results = calculate_all_similarities(raw_images, image_dict)

                output_dir = os.path.join('matches', patient)

                if similarity_results:

                    for i, match in enumerate(similarity_results, 1):
                        match_dir = os.path.join(output_dir, f'match{i}')

                        os.makedirs(match_dir, exist_ok = True)

                        for i, (name, img) in enumerate(match.items()):

                            cv2.imwrite(os.path.join(match_dir, name.replace('tif', 'jpg')),
                                        img,
                                        [int(cv2.IMWRITE_JPEG_QUALITY), 50])

                else:
                    g = make_new_image_groups(patient_path) # noah's algo
                    
                    if len(g) > 0:
                        for i, match in enumerate(g, 1):
                            score = match[-1]
                            names = match[:-1]

                            # similarity threshold
                            if score > 0.1:
                                match_dir = os.path.join(output_dir, f'match{i}')

                                os.makedirs(match_dir, exist_ok = True)

                                for i, name in enumerate(names):

                                    img = raw_images[name]

                                    cv2.imwrite(os.path.join(match_dir, name.replace('tif', 'jpg')),
                                                img,
                                                [int(cv2.IMWRITE_JPEG_QUALITY), 50])




            print(f"Processed {patient}")
            print("------------------------------------------------")




### PIPELINE
preprocess_files()
print('Finished preprocessing files')

process_all_patients('processed_images')