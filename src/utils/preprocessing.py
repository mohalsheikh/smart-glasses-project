"""
Image preprocessing functions
Created by Ethan
"""

import cv2 as cv

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