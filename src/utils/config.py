import os
import torch

#
# Global project config for paths and runtime behavior.
#

# Root of the project (this file is in smart-glasses-project/src/utils/)
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

# Path to the trained YOLO weights you want to run on device
YOLO_WEIGHTS_PATH = os.path.join(
    PROJECT_ROOT,
    "models",
    "yolo_glasses_v2.pt",  # <-- this is your fine-tuned 22-class model
)

# Runtime device selection
# Try Apple MPS first (Metal / M2), else CPU fallback.
if torch.backends.mps.is_available():
    DEVICE = "mps"
else:
    DEVICE = "cpu"

# Minimum confidence for reporting a detection (you can tweak this)
DEFAULT_CONF_THRESHOLD = 0.3

# Max number of detections per frame to speak/announce/etc
MAX_DETECTIONS_PER_FRAME = 10

# Optional: list of classes we care about MOST for spoken feedback/navigation
PRIORITY_CLASSES = [
    "person",
    "car",
    "stop sign",
    "traffic light",
    "stairs",
    "bottle",
    "mobile phone",
    "chair",
    "table",
    "toothbrush",
    "pen",
]
