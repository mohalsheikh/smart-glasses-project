"""
Configuration file for constants
Created by Mohammed
Optimized for SPEED and real-time performance
"""

import numpy as np
from enum import Enum

# ---------------------------------------------------------------------------
# Camera settings
# ---------------------------------------------------------------------------

# Lower resolution for MUCH better FPS (you can increase if GPU is good)
DEFAULT_FRAME_WIDTH: int = 640
DEFAULT_FRAME_HEIGHT: int = 480

class Direction(Enum):
    LEFT = 1
    FRONT = 2
    RIGHT = 3

STRING_TO_DIRECTION = {
    "left": Direction.LEFT,
    "front": Direction.FRONT,
    "right": Direction.RIGHT
}