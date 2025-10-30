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

# Sharpen an image using a predefined kernel.
def sharpen_image(image):
    return cv.filter2D(image, -1, config.SHARP)

# Apply Gaussian blur to an image using predefined settings.
def gaussian_blur(image):
    return cv.GaussianBlur(image, config.GAUSSIAN_BLUR_KERNEL_SIZE, config.GAUSSIAN_BLUR_SIGMA_X, config.GAUSSIAN_BLUR_SIGMA_Y)

# Deskew an image such that what we presume to be the largest block of 