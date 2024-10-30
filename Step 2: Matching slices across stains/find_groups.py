# Create groups of 3 for each strain per patient and drop the rest
from glob import glob
from similarity_score import read, shape_similarity
import os
import shutil

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
    
    # make a new folder within the folder with 3 images as specified
    for i in range(len(best_groups)):
        new_folder = folder + f'/group_{i}'
        if not os.path.exists(new_folder):
            os.mkdir(new_folder)
        # add images to the new folder
        for i in best_groups[i][:3]:
            src = folder + "/" + i
            dst = new_folder
            shutil.copy2(src, dst)

            # rename the file in the new folder
            if 'h&e' in i:
                strain = 'h&e'
            elif 'melan' in i:
                strain = 'melan'
            elif 'sox10' in i:
                strain = 'sox10'
            os.rename(dst + "/" + i, dst + f"/{strain}.tif")

    # delete the rest of the images - that aren't folders
    for i in glob(folder + "/*"):
        if os.path.isfile(i):
            os.remove(i)