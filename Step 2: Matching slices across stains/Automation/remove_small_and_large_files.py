# In the processed images folder, we have slices of tissues from different strains
# Sometimes, the slices are incorrect (blurry/small) and we want to remove them

from glob import glob
import os

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
