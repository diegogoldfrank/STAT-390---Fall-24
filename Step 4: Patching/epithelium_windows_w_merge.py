import cv2
import numpy as np
import os

# Directories
input_folder = "processed_images"
output_folder = "filtered_images"

# Check if output folder exists, if not create it
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Function to process each image
def process_image(image_path, output_image_path, output_mask_path, intermediate_image_path, final_image_path, squared_image_path):
    # Load the image
    image = cv2.imread(image_path)

    # Check if the image was loaded correctly
    if image is None:
        raise FileNotFoundError(f"Image file not found at path: {image_path}")

    # Convert to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply a Gaussian blur to smooth the image (reduce blur kernel size)
    blurred_image = cv2.GaussianBlur(gray_image, (25, 25), 0)

    # Thresholding to segment the epithelium
    _, thresholded_mask = cv2.threshold(blurred_image, 150, 255, cv2.THRESH_BINARY_INV)

    # Morphological operations to clean up the mask
    kernel_small = np.ones((20, 20), np.uint8)
    cleaned_mask = cv2.morphologyEx(thresholded_mask, cv2.MORPH_CLOSE, kernel_small)
    cleaned_mask = cv2.morphologyEx(cleaned_mask, cv2.MORPH_OPEN, kernel_small)

    # Find contours to identify the epithelium region
    contours, _ = cv2.findContours(cleaned_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)

        # Create a blank mask to draw the epithelium contour
        epithelium_mask = np.zeros_like(cleaned_mask)
        cv2.drawContours(epithelium_mask, [largest_contour], -1, (255), thickness=cv2.FILLED)

        # Save the mask image
        cv2.imwrite(output_mask_path, epithelium_mask)

        # Create a mask to track which areas are covered by windows
        covered_mask = np.zeros_like(epithelium_mask)

        # Calculate the width of the epithelium at multiple segments for each row
        def calculate_epithelium_width(mask, y_start, y_end, center_x):
            widths = []
            for y in range(y_start, y_end):
                row = mask[y, :]  # Get the row
                white_pixel_indices = np.where(row == 255)[0]  # Get the indices of white pixels (epithelium)

                if len(white_pixel_indices) > 0:
                    # Filter only the segments that contain the center point
                    segments = np.split(white_pixel_indices, np.where(np.diff(white_pixel_indices) != 1)[0] + 1)
                    for segment in segments:
                        if len(segment) > 0 and segment[0] <= center_x <= segment[-1]:
                            width = segment[-1] - segment[0]
                            widths.append(width)
            return max(widths) if len(widths) > 0 else 0

        # Define the sliding window function
        def sliding_window(image, mask, step_size, window_size):
            # Slide a window across the image
            for y in range(0, image.shape[0] - window_size, step_size):
                for x in range(0, image.shape[1] - window_size, step_size):
                    yield (x, y, image[y:y + window_size, x:x + window_size], mask[y:y + window_size, x:x + window_size])

        # Now applying the sliding window logic based on the mask
        stride = 100  # No overlap for now
        window_size = 100  # Initialize window size

        final_windows_image = image.copy()

        windows = []  # Store windows for merging

        # Sliding window process
        for (x, y, window, window_mask) in sliding_window(image, epithelium_mask, stride, window_size):
            # Calculate the ratio of epithelium (white) area in the window
            total_area = window_mask.size
            epithelium_area = np.count_nonzero(window_mask)
            epithelium_ratio = epithelium_area / total_area

            # If epithelium occupies more than 50%, adjust box width
            if epithelium_ratio > 0.50:
                x_start, y_start = x, y
                center_x = x_start + window_size // 2

                # Find the local epithelium width at the bottom of the window
                local_epithelium_width = calculate_epithelium_width(epithelium_mask, y_start, y_start + window_size, center_x)

                # Ensure the box does not exceed image boundaries
                x_end = min(x_start + local_epithelium_width, image.shape[1])

                # Ensure box does not overlap with already covered areas
                window_area_to_check = covered_mask[y_start:y_start + window_size, x_start:x_end]
                if np.count_nonzero(window_area_to_check) == 0:
                    # Mark the area as covered
                    covered_mask[y_start:y_start + window_size, x_start:x_end] = 255

                    # Draw the rectangle around the adjusted window size
                    cv2.rectangle(final_windows_image, (x_start, y_start), (x_end, y_start + window_size), (0, 0, 255), 8)

                    # Save the windows for merging
                    windows.append((x_start, y_start, x_end, y_start + window_size))

        # Save the intermediate windows image before merging
        cv2.imwrite(intermediate_image_path, final_windows_image)

        # Now, apply merging logic
        final_image = image.copy()

        # Sort windows by x (horizontal) then y (vertical)
        windows.sort(key=lambda w: (w[0], w[1]))

        merged_windows = []
        used_windows = set()

        # Merge windows vertically within the same horizontal region
        for i, (x_start, y_start, x_end, y_end) in enumerate(windows):
            if i in used_windows:
                continue

            current_windows = [(x_start, y_start, x_end, y_end)]
            current_x_start = x_start
            current_x_end = x_end

            # Try to merge vertically for up to 4 windows
            for j in range(i + 1, len(windows)):
                if len(current_windows) >= 4:
                    break
                x2_start, y2_start, x2_end, y2_end = windows[j]
                # Check if vertically aligned and close
                if current_x_start <= x2_start <= current_x_end and abs(y2_start - y_end) <= stride:
                    current_windows.append((x2_start, y2_start, x2_end, y2_end))
                    y_end = y2_end  # Expand the vertical range
                    current_x_end = max(current_x_end, x2_end)  # Expand the width if needed
                    used_windows.add(j)

            # If valid merge, add the merged window
            if len(current_windows) >= 2:
                merged_windows.append((x_start, current_windows[0][1], current_x_end, y_end))

        # Draw the merged windows
        for (x_start, y_start, x_end, y_end) in merged_windows:
            cv2.rectangle(final_image, (x_start, y_start), (x_end, y_end), (0, 0, 255), 8)

        # Save the final merged windows image
        cv2.imwrite(final_image_path, final_image)

        # Calculate the average area of all merged windows
        total_area = sum((x_end - x_start) * (y_end - y_start) for (x_start, y_start, x_end, y_end) in merged_windows)
        avg_area = total_area / len(merged_windows) if merged_windows else 0

        # Calculate the side length of a square with the same average area
        square_size = int(np.sqrt(avg_area))

        # Draw the square windows
        squared_image = image.copy()
        for (x_start, y_start, x_end, y_end) in merged_windows:
            # Adjust the position to draw the square
            square_x_end = x_start + square_size
            square_y_end = y_start + square_size

            # Ensure the square fits within image boundaries
            square_x_end = min(square_x_end, image.shape[1])
            square_y_end = min(square_y_end, image.shape[0])

            # Draw the square
            cv2.rectangle(squared_image, (x_start, y_start), (square_x_end, square_y_end), (0, 0, 255), 8)

        # Save the squared windows image
        cv2.imwrite(squared_image_path, squared_image)

        print(f"Processed and saved {output_image_path}, mask saved at {output_mask_path}, intermediate, final, and squared images saved.")

    else:
        print(f"No epithelium contours were detected in {image_path}. Try adjusting the parameters.")

# Loop through all images in the input folder
for filename in os.listdir(input_folder):
    if filename.endswith(".tif") or filename.endswith(".jpg") or filename.endswith(".png"):  # Adjust based on your file types
        input_image_path = os.path.join(input_folder, filename)
        output_image_path = os.path.join(output_folder, f"filtered_{filename}")
        output_mask_path = os.path.join(output_folder, f"mask_{filename}")
        intermediate_image_path = os.path.join(output_folder, f"intermediate_{filename}")
        final_image_path = os.path.join(output_folder, f"final_{filename}")
        squared_image_path = os.path.join(output_folder, f"window_squared_{filename}")

        # Process the image and save the output, mask, intermediate, final, and squared images
        process_image(input_image_path, output_image_path, output_mask_path, intermediate_image_path, final_image_path, squared_image_path)

print("Processing complete for all images.")
