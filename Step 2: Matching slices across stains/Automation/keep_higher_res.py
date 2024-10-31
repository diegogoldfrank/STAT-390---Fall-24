# Sometimes, we have multiple images of the same strain of the same patient
# We want to keep the one with the highest resolution

from glob import glob
import os

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
    image_files = glob('all_patient_images/*.tif')

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
                h_and_e_files.append("all_patient_images\\" + key + "_" + item)
            if 'melan' in item:
                melan_count += 1
                melan_files.append("all_patient_images\\" + key + "_" + item)
            if 'sox10' in item:
                sox10_count += 1
                sox10_files.append("all_patient_images\\" + key + "_" + item)
        if h_and_e_count > 1:
            keep_highest_res_image(h_and_e_files)
        if melan_count > 1:
            keep_highest_res_image(melan_files)
        if sox10_count > 1:
            keep_highest_res_image(sox10_files)

def remove_last_char():
    image_files = glob('all_patient_images/*.tif')
    for image_file in image_files:
        new_image_file = image_file[:-4]
        if new_image_file[-1] == '2':
            new_image_file = new_image_file[:-1]
        new_image_file += '.tif'
        os.rename(image_file, new_image_file)
