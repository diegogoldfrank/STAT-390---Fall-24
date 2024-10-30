# We only use data from patients with all 3 strain types, so we can delete the rest

import os
from glob import glob

def delete_patients():
    # Get list of all .tif files in the processed_images folder
    image_files = glob('all_patient_images/*.tif')

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