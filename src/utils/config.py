"""
Configuration file for constants
"""

CAMERA_INDEX: int = 0
FRAME_WIDTH: int = 640
FRAME_HEIGHT: int = 480

MODEL: str = "yolov8n.pt"
CONFIDENCE_THRESHOLD: float = 0.35  # Confidence threshold
IOU_THRESHOLD: float = 0.45  # IoU threshold
IMGSZ: int = FRAME_WIDTH if FRAME_WIDTH < FRAME_HEIGHT else FRAME_HEIGHT  # Image size for model input
TRACKER: str = "bytetrack.yaml"  # the tracker we're using
MAX_DETECTIONS: int = 100  # maximum number of objects to detect in a frame