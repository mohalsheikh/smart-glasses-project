"""
Configuration file for constants
Created by Mohammed
Edited by Ethan
"""

import numpy as np

# camera settings
CAMERA_INDEX: int = 0
FRAME_WIDTH: int = 640
FRAME_HEIGHT: int = 480

# YOLO settings
DEFAULT_MODEL_NAME: str = "yolov8n.pt"
DEFAULT_YOLO_CONFIDENCE_THRESHOLD: float = 0.35
DEFAULT_IOU_THRESHOLD: float = 0.45
DEFAULT_TRACKER: str = "bytetrack.yaml"
DEFAULT_MAX_DETECTIONS: int = 100

# OCR settings
DEFAULT_OCR_CONFIDENCE_THRESHOLD: float = 0.25

# Preprocessing settings
GAUSSIAN_BLUR_KERNEL_SIZE: tuple = (5, 5)
GAUSSIAN_BLUR_SIGMA_X: float = 0
GAUSSIAN_BLUR_SIGMA_Y: float = 0

SHARP = np.array([[0, -1, 0], # Image processing kernel for sharpening
                  [-1, 5,-1],  
                  [0, -1, 0]])
