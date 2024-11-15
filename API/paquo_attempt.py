from paquo import QuPathImage, QuPathImageType
from shapely.geometry import Point
import os
import json

# Load the JSON configuration
with open('config.json') as f:
    config = json.load(f)

# Constants and configurations
output_dir = os.path.join(os.getcwd(), 'processed_images')
os.makedirs(output_dir, exist_ok=True)
distance_threshold = 5000.0  # Distance threshold for merging annotations (micrometers)
duplicate_threshold = 50.0    # Distance threshold for identifying duplicates (micrometers)
downsample = 1.0              # Downsampling factor for ROI exports
pixel_classifier_name = "tissues_2"  # Classifier name used in the Groovy code

# Load .tif image
tif_file_path = "path_to_your_image.tif"  # Replace with your .tif file path
base_image_name = os.path.basename(tif_file_path).replace(".tif", "")

# Open image with Paquo
image = QuPathImage.open(tif_file_path)

# STEP 0: Check if the image is a mask and skip if necessary
if "mask" in base_image_name.lower():
    print(f"Skipping image: {base_image_name} (contains 'mask')")
else:
    # STEP 1: Set image type and perform stain deconvolution
    image.image_type = QuPathImageType.BRIGHTFIELD_H_E
    image.set_color_deconvolution_stains({
        "Name": "H&E default",
        "Stain 1": "Hematoxylin",
        "Values 1": "0.65111 0.70119 0.29049",
        "Stain 2": "Eosin",
        "Values 2": "0.2159 0.8012 0.5581",
        "Background": "255 255 255"
    })
    print("Stain deconvolution applied")

    # STEP 2: Create annotations if none exist
    annotations = [a for a in image.get_annotation_objects() if a.path_class == "Positive"]
    if not annotations:
        print(f"Creating annotations with pixel classifier '{pixel_classifier_name}'")
        # Assuming a function to apply pixel classifier settings from JSON
        image.create_annotations_from_pixel_classifier(
            classifier_name=pixel_classifier_name,
            min_area=50000.0,
            max_area=3000.0,
            split=True,
            delete_existing=True,
            select_new=True
        )
    else:
        print("Using existing annotations")

    # Define helper function to calculate Euclidean distance between annotation centroids
    def calculate_distance(annotation1, annotation2):
        p1 = Point(annotation1.centroid)
        p2 = Point(annotation2.centroid)
        return p1.distance(p2)

    # STEP 3: Merge nearby annotations based on distance threshold
    for i, annotation1 in enumerate(annotations):
        for j, annotation2 in enumerate(annotations[i + 1:], start=i + 1):
            distance = calculate_distance(annotation1.roi, annotation2.roi)
            print(f"Distance between annotation {i} and annotation {j}: {distance}")
            if distance < distance_threshold:
                print(f"Merging annotations: Distance = {distance}")
                image.select_objects([annotation1, annotation2])
                image.merge_selected_annotations()
                annotations = [a for a in image.get_annotation_objects() if a.path_class == "Positive"]

    # STEP 4: Remove duplicate annotations within a specified proximity
    to_remove = []
    for i, annotation1 in enumerate(annotations):
        for annotation2 in annotations[i + 1:]:
            distance = calculate_distance(annotation1.roi, annotation2.roi)
            if distance < duplicate_threshold:
                print("Duplicate found, removing one of the annotations")
                to_remove.append(annotation2)

    # Remove duplicates
    image.remove_objects(to_remove)

    # STEP 5: Export each annotation's ROI
    for idx, annotation in enumerate(annotations, start=1):
        roi = annotation.roi
        output_image_path = os.path.join(output_dir, f"{base_image_name}_ROI_{idx}.tif")

        # Export the region with specified downsample factor
        image.write_image_region(roi, output_image_path, downsample=downsample)

    print(f"{len(annotations)} ROIs identified and exported.")
