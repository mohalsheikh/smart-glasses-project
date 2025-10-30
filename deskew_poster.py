"""
Text-detection using a basic EasyOCR implementation
Created by Eric Leon
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

# Verify image loaded successfully
if image is None:
    raise ValueError(f"Error: Failed to load image from '{image_path}'")

# first, we need to determine if the image needs deskewing.
# we start by converting to grayscale and applying edge detection.
gray = bgr_to_gray(image)
edges = cv.Canny(gray, 50, 100)

# now we want to find the contours in the edged image.
contours, _ = cv.findContours(edges, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)

# if we found any contours, we can assume the image is skewed and needs deskewing.
if contours:
    # we assume the largest contour by area is around the poster. 
    largest_contour = max(contours, key=cv.contourArea)

while cv.waitKey(1) & 0xFF != ord("q"):
    cam.show_image(edges)