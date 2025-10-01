"""
Common image preprocessing utilities.
"""

import cv2
import numpy as np


def to_gray(frame_bgr):
    return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)


def clahe_gray(gray, clip_limit: float = 2.0, tile_grid_size=(8, 8)):
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    return clahe.apply(gray)


def resize_keep_ar(img, width=None, height=None, inter=cv2.INTER_AREA):
    if width is None and height is None:
        return img
    (h, w) = img.shape[:2]
    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))
    return cv2.resize(img, dim, interpolation=inter)


def denoise_bilateral(gray, d=7, sigma_color=55, sigma_space=55):
    return cv2.bilateralFilter(gray, d=d, sigmaColor=sigma_color, sigmaSpace=sigma_space)
