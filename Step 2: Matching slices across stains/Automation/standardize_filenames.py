import os
from glob import glob

# first thing we need to do is create a standardized patient ID for each patient
# this will be patient_id + strain_type
# we will rename the images to this standardized patient ID
def rename_files():
    # Get list of all .tif files in the processed_images folder
    image_files = glob('all_patient_images/*.tif')

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

## This mostly works - I had to do some manual tweaking to get the filenames to match the patient IDs
