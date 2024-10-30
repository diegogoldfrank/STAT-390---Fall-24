from standardize_filenames import rename_files
from delete_patients import delete_patients
from keep_higher_res import keep_highest_res, keep_highest_res_image, remove_last_char
from remove_small_and_large_files import remove_small_files, rename_files, split_files_by_patient
from find_groups import make_new_image_groups

from glob import glob
import sys
import os

if __name__ == '__main__':
    # do this first
    rename_files()
    # then manually rename the files that aren't formatted properly
    # then run the rest of the functions
    delete_patients()
    keep_highest_res()
    remove_last_char()
    remove_small_files()
    rename_files()
    split_files_by_patient()
    for folder in glob('processed_images/*'):
        make_new_image_groups(folder)
    
    sys.exit(0)