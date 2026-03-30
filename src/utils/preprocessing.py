"""
Image preprocessing functions
Created by Ethan
"""

import cv2 as cv
import numpy as np

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

# Convert a BGR image to RGB.
def bgr_to_rgb(image):
    return cv.cvtColor(image, cv.COLOR_BGR2RGB)

# Convert a RGB image to BGR.
def rgb_to_bgr(image):
    return cv.cvtColor(image, cv.COLOR_RGB2BGR)

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
# Function generator that returns multiple (4) deskewed versions of the image in order of likelihood of being the correct orientation for OCR.  
def deskew_image(image):
    # first, we need to determine if the image needs deskewing.
    # we start by converting to grayscale and applying edge detection.
    gray = bgr_to_gray(image)
    edges = canny_edge_detection(gray)
    edges = dilate(edges, iterations=1) # enhance edges

    # now we want to find the contours in the edged image.
    contours, _ = cv.findContours(edges, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)

    # if we found any contours, we want to look for the largest one; this is likely to surround the object of interest.
    if contours:
        # before we find the largest contour we want to convex hull all contours to close gaps.
        hulls = [cv.convexHull(c) for c in contours]

        # we assume the largest contour by area is around the object of interest.
        largest_hull = max(hulls, key=cv.contourArea)
        
        # get the minimum area rectangle that bounds the largest contour
        rect = cv.minAreaRect(largest_hull)

        angle = rect[-1] % 90  # get the angle of the rectangle

        # adjust angle to be within [-45, 45]
        if angle > 45:
            angle -= 90  
        elif angle < -45:
            angle += 90

        # now we can deskew the image using the angle.

        (height, width) = image.shape[:2] # shape returns (height, width, channels), we don't want channels
        center = (width // 2, height // 2)

        rotation_matrix = cv.getRotationMatrix2D(center, angle, 1.0)
        image1 = cv.warpAffine(image, rotation_matrix, (width, height))

        yield image1

        # attempt "second best" rotation if we enter into this function again
        angle2 = angle + 90 if angle <= 0 else angle - 90
        rotation_matrix = cv.getRotationMatrix2D(center, angle2, 1.0)
        image2 = cv.warpAffine(image, rotation_matrix, (width, height))

        yield image2

        # "third best" rotation is best rotation + 180 degrees.
        rotation_matrix = cv.getRotationMatrix2D(center, angle + 180, 1.0)
        image3 = cv.warpAffine(image, rotation_matrix, (width, height))

        yield image3

        # "fourth best" rotation is second best + 180 degrees.
        rotation_matrix = cv.getRotationMatrix2D(center, angle2 + 180, 1.0)
        image4 = cv.warpAffine(image, rotation_matrix, (width, height))

        yield image4
    else:
        # yield original image if no contours found  
        for _ in range(4):
            yield image


# TODO decide if we want to use this. Eric needs to test after changing ocr confidence to not reflect the count of boxes.
# Deskew an image such that what we presume to be the object of interest is rotated upright.
# Function generator that returns multiple (4) deskewed versions of the image in order of likelihood of being the correct orientation for OCR.  
def deskew_image(image, original_img_avg_conf):
    # first, we need to determine if the image needs deskewing.
    # we start by converting to grayscale and applying edge detection.
    gray = bgr_to_gray(image)
    edges = canny_edge_detection(gray)
    edges = dilate(edges, iterations=1) # enhance edges

    # now we want to find the contours in the edged image.
    contours, _ = cv.findContours(edges, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)

    # if we found any contours, we want to look for the largest one; this is likely to surround the object of interest.
    if contours:
        # before we find the largest contour we want to convex hull all contours to close gaps.
        hulls = [cv.convexHull(c) for c in contours]

        # we assume the largest contour by area is around the object of interest.
        largest_hull = max(hulls, key=cv.contourArea)
        
        # get the minimum area rectangle that bounds the largest contour
        rect = cv.minAreaRect(largest_hull)

        angle = rect[-1] % 90  # get the angle of the rectangle

        # adjust angle to be within [-45, 45]
        if angle > 45:
            angle -= 90  
        elif angle < -45:
            angle += 90
        
        (height, width) = image.shape[:2] # shape returns (height, width, channels), we don't want channels
        center = (width // 2, height // 2)

        # now we can deskew the image using the angle.
        # deskewing with the angle variable will make the smallest possible rotation that would cause our min area rectangle's angle to become 0.
        # alt_angle is the second smallest possible rotation that would do this.
        alt_angle = angle + 90 if angle <= 0 else angle - 90

        # we choose which to rotate with first depending on the original_img_avg_conf.
        # if our average ocr confidence in the original image is below a threshold then we make the alt_angle rotation first, 
        # on the principle that it is more likely that a generally alt rotation is required for ocr to read text if it didn't pick up on anything at all the first time.
        if (original_img_avg_conf >= 0.35):
            angle1 = angle
            angle2 = alt_angle
        else:
            angle1 = alt_angle
            angle2 = angle

        # attempting "best" rotation for ocr reading.
        rotation_matrix = cv.getRotationMatrix2D(center, angle1, 1.0)
        image1 = cv.warpAffine(image, rotation_matrix, (width, height))

        yield image1

        # attempt "second best" rotation if we enter into this function again
        rotation_matrix = cv.getRotationMatrix2D(center, angle2, 1.0)
        image2 = cv.warpAffine(image, rotation_matrix, (width, height))

        yield image2

        # "third best" rotation is best rotation + 180 degrees.
        rotation_matrix = cv.getRotationMatrix2D(center, angle + 180, 1.0)
        image3 = cv.warpAffine(image, rotation_matrix, (width, height))

        yield image3

        # "fourth best" rotation is second best + 180 degrees.
        rotation_matrix = cv.getRotationMatrix2D(center, angle2 + 180, 1.0)
        image4 = cv.warpAffine(image, rotation_matrix, (width, height))

        yield image4
    else:
        # yield original image if no contours found  
        for _ in range(4):
            yield image