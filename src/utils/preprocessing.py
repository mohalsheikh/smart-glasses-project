"""
Image preprocessing functions
"""

import cv2

def to_gray(frame):
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
