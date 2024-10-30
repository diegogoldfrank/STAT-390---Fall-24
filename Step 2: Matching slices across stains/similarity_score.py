## Credit to Noah

import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from itertools import product

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