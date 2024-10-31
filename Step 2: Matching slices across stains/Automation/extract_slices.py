## Credit to Noah

import tifffile as tiff
from itertools import combinations
import cv2
import numpy as np
import math
import os

def read_image(image_path):
    strain_type = image_path.split('_')[-1][:-4]
    with tiff.TiffFile(image_path) as tif:
        image_highres = tif.pages[3].asarray()
        image_lowres = tif.pages[-3].asarray()
    return image_highres, image_lowres, strain_type

def create_contours(image, strain_type):
    contours = None
    # create masks based on strain type

    # h&e is pink/purple
    if strain_type == 'h&e':
        hsv_img = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        lower_bound = np.array([120, 50, 50])
        upper_bound = np.array([170, 255, 255])

        mask = cv2.inRange(hsv_img, lower_bound, upper_bound)
        kernel = np.ones((60, 60), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # melan is brown and blue
    elif strain_type == 'melan':
        hsv_img = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        lower_bound = np.array([10, 10, 10])
        upper_bound = np.array([200, 200, 200])

        mask = cv2.inRange(hsv_img, lower_bound, upper_bound)
        kernel = np.ones((15, 15), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # sox10 is blue-ish
    elif strain_type == 'sox10':
        hsv_img = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        lower_bound = np.array([10, 10, 10])
        upper_bound = np.array([200, 200, 200])

        mask = cv2.inRange(hsv_img, lower_bound, upper_bound)
        kernel = np.ones((100, 100), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours

def remove_overlap(contours):
    if len(contours) > 1:
        pairs = combinations(range(len(contours)), 2)
        for i, j in pairs:
            c1, c2 = contours[i], contours[j]
            x1, y1, w1, h1 = cv2.boundingRect(c1)
            x2, y2, w2, h2 = cv2.boundingRect(c2)

            if all([x1 < x2 + w2, x1 + w1 > x2, y1 < y2 + h2, y1 + h1 > y2]):
                if cv2.contourArea(c1) > cv2.contourArea(c2):
                    contours.pop(j)
                else:
                    contours.pop(i)

                return remove_overlap(contours)
    return contours

def filter_slices(contours, image_highres, image_lowres):
    slices = []

    if contours is None:
        return
    contours = sorted(filter(lambda x: cv2.contourArea(x) >= 10, contours), key=cv2.contourArea, reverse=True)[:8]
    contours = remove_overlap(contours)

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)

        ymin_norm, ymax_norm = y/image_lowres.shape[0], (y+h)/image_lowres.shape[0]
        xmin_norm, xmax_norm = x/image_lowres.shape[1], (x+w)/image_lowres.shape[1]

        slices.append(image_highres[math.floor(ymin_norm*image_highres.shape[0]-200):math.ceil(ymax_norm*image_highres.shape[0]+200),
                                    math.floor(xmin_norm*image_highres.shape[1]-200):math.ceil(xmax_norm*image_highres.shape[1]+200)])
        
    return slices

def extract_slices(slices, image_path):
    if slices is None:
        return
    output_dir = os.path.join(os.path.dirname(image_path), 'slices')
    os.makedirs(output_dir, exist_ok=True)
    for idx, slice_img in enumerate(slices):
        output_path = os.path.join(output_dir, f"{os.path.basename(image_path)[:-4]}_slice_{idx}.tif")
        tiff.imwrite(output_path, slice_img)
