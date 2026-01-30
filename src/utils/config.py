"""
Configuration file for constants
Created by Mohammed
Optimized for SPEED and real-time performance
"""

import numpy as np

# ---------------------------------------------------------------------------
# Camera settings
# ---------------------------------------------------------------------------

DEFAULT_CAMERA_INDEX: int = 0

# Lower resolution for MUCH better FPS (you can increase if GPU is good)
DEFAULT_FRAME_WIDTH: int = 640
DEFAULT_FRAME_HEIGHT: int = 480

# ---------------------------------------------------------------------------
# YOLO object detection settings - OPTIMIZED FOR SPEED
# ---------------------------------------------------------------------------

DEFAULT_MODEL_NAME: str = "yolov8n-oiv7.pt"  

# Lower confidence to catch more objects
DEFAULT_YOLO_CONFIDENCE_THRESHOLD: float = 0.20  # Balanced for speed/accuracy

DEFAULT_IOU_THRESHOLD: float = 0.45

DEFAULT_TRACKER: str = "bytetrack.yaml"

# Reduced for speed
DEFAULT_MAX_DETECTIONS: int = 100

# ---------------------------------------------------------------------------
# AUTO CONTROLLER PERFORMANCE SETTINGS - KEY FOR SPEED (these are irrelevant to manual controller)
# ---------------------------------------------------------------------------

# Process every Nth frame (HUGE speed boost)
PROCESS_EVERY_N_FRAMES: int = 3  # Only process every 3rd frame

# Use smaller image size for YOLO inference (MAJOR speed boost)
YOLO_INFERENCE_SIZE: int = 480  # Much smaller = much faster

# Disable expensive features
USE_MULTISCALE_DETECTION: bool = False  # This was killing your FPS!
USE_ADAPTIVE_PREPROCESSING: bool = False  # Disable preprocessing
USE_CLAHE: bool = False
AUTO_ADJUST_BRIGHTNESS: bool = False
USE_SCENE_CHANGE_DETECTION: bool = False  # Adds overhead

# GPU settings
USE_GPU: bool = True
USE_HALF_PRECISION: bool = True  # FP16 for speed on GPU

# Tracking - disable for more speed
ENABLE_TRACKING: bool = False  # Tracking adds overhead

# Agnostic NMS
AGNOSTIC_NMS: bool = False

# ---------------------------------------------------------------------------
# Smart Detection Settings (simplified)
# ---------------------------------------------------------------------------

# Small objects that need lower confidence
SMALL_OBJECTS: set = {
    "Pen", "Pencil", "Toothbrush", "Spoon", "Fork", "Knife", 
    "Remote control", "Computer mouse", "Glasses", "Watch"
}

CONFIDENCE_BY_CATEGORY: dict = {
    "small_objects": 0.15,
    "priority_objects": 0.20,
    "general_objects": 0.25,
}

# ---------------------------------------------------------------------------
# OCR settings
# ---------------------------------------------------------------------------

DEFAULT_OCR_CONFIDENCE_THRESHOLD: float = 0.25

# Minimal OCR whitelist
OCR_WHITELIST = (
    "book", "laptop", "cell phone", "sign", "bottle", "can"
)

# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------

GAUSSIAN_BLUR_KERNEL_SIZE: tuple = (3, 3)
GAUSSIAN_BLUR_SIGMA_X: float = 0.0
GAUSSIAN_BLUR_SIGMA_Y: float = 0.0

SHARP = np.array([
    [0, -1, 0],
    [-1, 5, -1],
    [0, -1, 0],
])

DILATION_ITERATIONS: int = 1

# ---------------------------------------------------------------------------
# Speech Settings
# ---------------------------------------------------------------------------

# Speak less often for better performance
SPEAK_EVERY_N_FRAMES: int = 60  # Every 60 frames = ~2 seconds at 30fps
DEBUG_PRINT_EVERY_N_FRAMES: int = 90

# Fewer items for faster speech
MAX_SPEECH_ITEMS: int = 5

# Disable distance estimation (adds processing)
ENABLE_DISTANCE_ESTIMATION: bool = False

REFERENCE_SIZES: dict = {
    "person": 400,
    "door": 500,
    "chair": 250,
}

# ---------------------------------------------------------------------------
# Debug Settings
# ---------------------------------------------------------------------------

SHOW_DEBUG_WINDOW: bool = True
SAVE_DEBUG_FRAMES: bool = False
DEBUG_FRAME_PATH: str = "./debug_frames/"
LOG_LEVEL: str = "INFO"
SHOW_FPS: bool = True
SHOW_DETECTION_COUNT: bool = True