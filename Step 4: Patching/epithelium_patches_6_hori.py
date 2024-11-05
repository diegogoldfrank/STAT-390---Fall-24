import cv2
import numpy as np
import os

# Directories
input_folder = "processed_images_sub"
output_folder = "filtered_images"

# Ensure output folder exists
os.makedirs(output_folder, exist_ok=True)

def calculate_optimal_patch_dimensions(mask):
    height, width = mask.shape
    max_continuous_width = 0
    for y in range(height):
        row = mask[y, :]
        white_pixel_indices = np.where(row == 255)[0]

        if len(white_pixel_indices) > 0:
            white_segments = np.split(white_pixel_indices, np.where(np.diff(white_pixel_indices) != 1)[0] + 1)
            row_max_width = max((seg[-1] - seg[0] + 1) for seg in white_segments if len(seg) > 0)
            max_continuous_width = max(max_continuous_width, row_max_width)

    optimal_patch_height = max(max_continuous_width, 100)
    epithelium_area = np.count_nonzero(mask)
    optimal_patch_width = max(int((epithelium_area / 100) / optimal_patch_height), 5)
    return optimal_patch_height, optimal_patch_width

def calculate_overlap(patch_coords, placed_patches):
    x1, y1, x2, y2 = patch_coords
    for px1, py1, px2, py2 in placed_patches:
        overlap_x1 = max(x1, px1)
        overlap_y1 = max(y1, py1)
        overlap_x2 = min(x2, px2)
        overlap_y2 = min(y2, py2)
        if overlap_x1 < overlap_x2 and overlap_y1 < overlap_y2:
            overlap_area = (overlap_x2 - overlap_x1) * (overlap_y2 - overlap_y1)
            patch_area = (x2 - x1) * (y2 - y1)
            if overlap_area / patch_area > 0.10:
                return True
    return False

def calculate_coverage(mask, patches):
    total_epithelium_area = np.count_nonzero(mask)
    patch_area = 0
    for x1, y1, x2, y2 in patches:
        patch_mask = mask[y1:y2, x1:x2]
        patch_area += np.count_nonzero(patch_mask)
    coverage = (patch_area / total_epithelium_area) * 100 if total_epithelium_area > 0 else 0
    return coverage

def apply_patches(epithelium_mask, patch_height, patch_width, stride, orientation):
    height, width = epithelium_mask.shape
    placed_patches = []

    for y in range(0, height - patch_height + 1, stride):
        for x in range(0, width - patch_width + 1, stride):
            patch_mask = epithelium_mask[y:y + patch_height, x:x + patch_width]
            epithelium_ratio = np.count_nonzero(patch_mask) / patch_mask.size

            if epithelium_ratio >= 0.5:
                patch_coords = (x, y, x + patch_width, y + patch_height)
                if not calculate_overlap(patch_coords, placed_patches):
                    placed_patches.append(patch_coords)
    return placed_patches

def process_image(image_path, output_image_path, output_mask_path, region_outline_path):
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image file not found at path: {image_path}")

    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred_image = cv2.GaussianBlur(gray_image, (25, 25), 0)
    _, thresholded_mask = cv2.threshold(blurred_image, 150, 255, cv2.THRESH_BINARY_INV)

    kernel_small = np.ones((20, 20), np.uint8)
    cleaned_mask = cv2.morphologyEx(thresholded_mask, cv2.MORPH_CLOSE, kernel_small)
    cleaned_mask = cv2.morphologyEx(cleaned_mask, cv2.MORPH_OPEN, kernel_small)

    contours, _ = cv2.findContours(cleaned_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        epithelium_mask = np.zeros_like(cleaned_mask)
        cv2.drawContours(epithelium_mask, [largest_contour], -1, (255), thickness=cv2.FILLED)
        
        cv2.imwrite(output_mask_path, epithelium_mask)

        optimal_patch_height, optimal_patch_width = calculate_optimal_patch_dimensions(epithelium_mask)
        stride = max(int(min(optimal_patch_height, optimal_patch_width) * 0.95), 1)

        print(f"Processing mixed patches with optimal patch size: ({optimal_patch_width}, {optimal_patch_height}), stride: {stride}")

        final_image = image.copy()
        combined_patches = []
        region_size = optimal_patch_height

        # Draw the region outline on the tissue image itself
        region_outline_image = image.copy()

        for y in range(0, epithelium_mask.shape[0], region_size):
            for x in range(0, epithelium_mask.shape[1], region_size):
                region_mask = epithelium_mask[y:y + region_size, x:x + region_size]

                # Draw region boundaries directly on the tissue image
                cv2.rectangle(region_outline_image, (x, y), (x + region_size, y + region_size), (0, 0, 255), thickness=4)

                vertical_patches = apply_patches(region_mask, optimal_patch_height, optimal_patch_width, stride, orientation="vertical")
                horizontal_patches = apply_patches(region_mask, optimal_patch_width, optimal_patch_height, stride, orientation="horizontal")

                vertical_coverage = calculate_coverage(region_mask, vertical_patches)
                horizontal_coverage = calculate_coverage(region_mask, horizontal_patches)

                if vertical_coverage >= horizontal_coverage:
                    combined_patches.extend([(x1 + x, y1 + y, x2 + x, y2 + y) for x1, y1, x2, y2 in vertical_patches])
                else:
                    combined_patches.extend([(x1 + x, y1 + y, x2 + x, y2 + y) for x1, y1, x2, y2 in horizontal_patches])

        # Draw combined patches on the final image
        for (x1, y1, x2, y2) in combined_patches:
            cv2.rectangle(final_image, (x1, y1), (x2, y2), (0, 0, 255), thickness=4)

        total_coverage = calculate_coverage(epithelium_mask, combined_patches)
        print(f"Mixed Coverage: {total_coverage:.2f}% with {len(combined_patches)} patches")
        
        # Save the final patch image and region outline image
        cv2.imwrite(output_image_path, final_image)
        cv2.imwrite(region_outline_path, region_outline_image)
    else:
        print(f"No epithelium contours were detected in {image_path}. Try adjusting the parameters.")

for filename in os.listdir(input_folder):
    if filename.endswith(".tif") or filename.endswith(".jpg") or filename.endswith(".png"):
        input_image_path = os.path.join(input_folder, filename)
        output_mask_path = os.path.join(output_folder, f"mask_{filename}")
        output_image_path_mixed = os.path.join(output_folder, f"mixed_patches_{filename}")
        region_outline_path = os.path.join(output_folder, f"regions_outline_{filename}")

        print(f"Processing image: {filename}")
        process_image(input_image_path, output_image_path_mixed, output_mask_path, region_outline_path)

print("Processing complete for all images.")
