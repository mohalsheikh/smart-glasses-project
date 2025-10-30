"""
Text-detection using a basic EasyOCR implementation
Created by Ethan Walden
"""

import cv2 as cv
import easyocr
import os
from src.camera_handler import CameraHandler
from src.utils.config import DEFAULT_OCR_CONFIDENCE_THRESHOLD
from src.utils.preprocessing import bgr_to_gray

cam = CameraHandler()
easyocr_reader = easyocr.Reader(['en'], gpu=False)

# read image with error handling
image_path = "Test_img2.jpg"  # replace with your image path

# Check if file exists
if not os.path.exists(image_path):
    raise FileNotFoundError(f"Error: Image file not found at '{image_path}'")

# Load image
image = cv.imread(image_path)
image = cv.resize(image, (640, 480))

image = image[80:393, 30:620]

# zoom in for better visualization

# Verify image loaded successfully
if image is None:
    raise ValueError(f"Error: Failed to load image from '{image_path}'")

# first, we need to determine if the image needs deskewing.
# we start by converting to grayscale and applying edge detection.
gray = bgr_to_gray(image)
edges = cv.Canny(gray, 50, 100)

# now we want to find the contours in the edged image.
contours, _ = cv.findContours(edges, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)

# if we found any contours, we want to look for the largest one; this is likely to surround the object of interest.
# sometimes, in real situations, there may be something obscuring the edges around the object, like the hand and clip in test image 2.
# that's why we use convex hulls to close gaps in the contours before finding the largest one.
if contours:
    # before we find the largest contour we want to convex hull all contours to close gaps.
    hulls = [cv.convexHull(c) for c in contours]

    # we assume the largest contour by area is around the poster.
    largest_contour = max(hulls, key=cv.contourArea)

    # get the minimum area rectangle that bounds the largest contour
    rect = cv.minAreaRect(largest_contour)

    # draw the rectangle on a copy of the original image for visualization
    cv.polylines(image, [cv.boxPoints(rect).astype(int)], isClosed=True, color=(0, 255, 0), thickness=2)
    cv.drawContours(image, hulls, -1, (255, 0, 0), 2)
    cv.drawContours(image, [largest_contour], -1, (0, 0, 255), 2)

while cv.waitKey(1) & 0xFF != ord("q"):
    cam.show_image(image)