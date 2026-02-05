"""
Configuration file for constants
Created by Mohammed
Optimized for SPEED and real-time performance
"""

import numpy as np

# ---------------------------------------------------------------------------
# Camera settings
# ---------------------------------------------------------------------------

# Lower resolution for MUCH better FPS (you can increase if GPU is good)
DEFAULT_FRAME_WIDTH: int = 640
DEFAULT_FRAME_HEIGHT: int = 480

# ---------------------------------------------------------------------------
# Voice Input Settings (Offline Speech-to-Text via Vosk)
# ---------------------------------------------------------------------------

# This is used by manual_controller.py so you can press "r" to take a frame,
# then speak a command like: "hey what is in front of me".
#
# Requirements:
#   pip install vosk sounddevice
#   (and download a Vosk model like "vosk-model-en-us-0.22")

VOICE_INPUT_ENABLED: bool = True

# Absolute path to your Vosk model folder (must contain am/, conf/, graph/)
# NOTE: change this to where YOU unzipped the model.
VOSK_MODEL_PATH: str = r"C:\\Repos\\vosk-model-en-us-0.22\\vosk-model-en-us-0.22"

# Mic device index. Set to None to use your system default microphone.
VOICE_INPUT_DEVICE_INDEX: int | None = None

# These are safe defaults for most mics.
VOICE_INPUT_TARGET_RATE: int = 16000
VOICE_INPUT_BLOCK_SIZE: int = 8000
VOICE_INPUT_TIMEOUT_SECONDS: float = 8.0

# Phrases the system should treat as "describe what's in front of me".
# Matching is case-insensitive and whitespace-normalized.
VOICE_DESCRIBE_COMMANDS: tuple[str, ...] = (
    "hey what is in front of me",
    "what is in front of me",
)
