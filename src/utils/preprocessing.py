"""
Image preprocessing functions
Created by Ethan
"""

import cv2 as cv
import numpy as np
import src.utils.config as config

# Convert a BGR image to grayscale.
def bgr_to_gray(image):
    return cv.cvtColor(image, cv.COLOR_BGR2GRAY)

# Convert a BGR image to HSV.
def bgr_to_hsv(image):
    return cv.cvtColor(image, cv.COLOR_BGR2HSV)

# Convert a grayscale image to BGR.
def gray_to_bgr(image):
    return cv.cvtColor(image, cv.COLOR_GRAY2BGR)

# Convert a HSV image to BGR.
def hsv_to_bgr(image):
    return cv.cvtColor(image, cv.COLOR_HSV2BGR)

# Convert a RGB image to grayscale.
def rgb_to_gray(image):
    return cv.cvtColor(image, cv.COLOR_RGB2GRAY)

# Convert a RGB image to HSV.
def rgb_to_hsv(image):
    return cv.cvtColor(image, cv.COLOR_RGB2HSV)

# Convert a grayscale image to RGB.
def gray_to_rgb(image):
    return cv.cvtColor(image, cv.COLOR_GRAY2RGB)

# Convert a HSV image to RGB.
def hsv_to_rgb(image):
    return cv.cvtColor(image, cv.COLOR_HSV2RGB)

# Sharpen an image.
def sharpen_image(
        image, 
        sharp_kernel=np.array([
            [0, -1, 0],
            [-1, 5, -1],
            [0, -1, 0],
        ])):
    return cv.filter2D(image, -1, sharp_kernel)

# Apply Gaussian blur to an image.
def gaussian_blur(image, kernel_size=(3,3), sigma_x=0.0, sigma_y=0.0):
    return cv.GaussianBlur(image, kernel_size, sigma_x, sigma_y)

# applies dilation to an image.
def dilate(image, kernel=None, iterations=1):
    return cv.dilate(image, kernel, iterations=iterations)

# applies erosion to an image.
def erode(image, kernel=None, iterations=1):
    return cv.erode(image, kernel, iterations=iterations)

# applies dilation and then erosion to an image.
def open_image(image, kernel=None, iterations=1):
    return cv.morphologyEx(image, cv.MORPH_OPEN, kernel, iterations=iterations)

# applies erosion and then dilation to an image.
def close_image(image, kernel=None, iterations=1):
    return cv.morphologyEx(image, cv.MORPH_CLOSE, kernel, iterations=iterations)

# applies canny edge detection to image.
def canny_edge_detection(image, threshold1=100, threshold2=200):
    return cv.Canny(image, threshold1, threshold2)

# Deskew an image such that what we presume to be the object of interest is rotated upright.
def deskew_image(image):
    # first, we need to determine if the image needs deskewing.
    # we start by converting to grayscale and applying edge detection.
    gray = bgr_to_gray(image)
    edges = canny_edge_detection(gray)
    edges = dilate(edges, iterations=1) # enhance edges

    # now we want to find the contours in the edged image.
    contours, _ = cv.findContours(edges, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)

    # if we found any contours, we want to look for the largest one; this is likely to surround the object of interest.
    # sometimes, in real situations, there may be something obscuring the edges around the object.
    # that's why we use convex hulls to close gaps in the contours before finding the largest one.
    if contours:
        # before we find the largest contour we want to convex hull all contours to close gaps.
        hulls = [cv.convexHull(c) for c in contours]

        # we assume the largest contour by area is around the object of interest.
        largest_contour = max(hulls, key=cv.contourArea)
        
        # get the minimum area rectangle that bounds the largest contour
        rect = cv.minAreaRect(largest_contour)

        angle = rect[-1] % 90  # get the angle of the rectangle

        # adjust angle to be within [-45, 45] for easier rotation
        if angle > 45:
            angle -= 90  
        elif angle < -45:
            angle += 90

        if abs(angle) > 1:  # only deskew if the angle is significant
            # now we can deskew the image using the angle.
            (height, width) = image.shape[:2] # shape returns (height, width, channels), we don't want channels
            center = (width // 2, height // 2)
            rotation_matrix = cv.getRotationMatrix2D(center, angle, 1.0)
            
            image = cv.warpAffine(image, rotation_matrix, (width, height))
    
    return image