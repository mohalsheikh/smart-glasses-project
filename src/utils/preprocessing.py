"""
Common image preprocessing utilities tuned for skinny objects (pins).
"""

import cv2
import numpy as np


def to_gray(frame_bgr):
    return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)


def clahe_gray(gray, clip_limit: float = 2.0, tile_grid_size=(8, 8)):
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    return clahe.apply(gray)


def resize_keep_ar(img, width=None, height=None, inter=cv2.INTER_CUBIC):
    """
    Use CUBIC when upscaling to keep edges; AREA when downscaling.
    """
    if width is None and height is None:
        return img
    (h, w) = img.shape[:2]
    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))
    inter_auto = cv2.INTER_AREA if (r < 1.0) else inter
    return cv2.resize(img, dim, interpolation=inter_auto)


def denoise_bilateral(gray, d=5, sigma_color=40, sigma_space=40):
    """
    Slightly milder defaults to avoid killing thin edges.
    """
    return cv2.bilateralFilter(gray, d=d, sigmaColor=sigma_color, sigmaSpace=sigma_space)


def unsharp_mask(img_bgr, ksize=(0, 0), sigma=1.2, amount=1.0, thresh=0):
    """
    Simple unsharp mask to crispen edges before detection.
    """
    blurred = cv2.GaussianBlur(img_bgr, ksize, sigma)
    sharp = cv2.addWeighted(img_bgr, 1 + amount, blurred, -amount, 0)
    if thresh <= 0:
        return sharp
    low_contrast_mask = (cv2.absdiff(img_bgr, blurred) < thresh)
    return np.where(low_contrast_mask, img_bgr, sharp)
