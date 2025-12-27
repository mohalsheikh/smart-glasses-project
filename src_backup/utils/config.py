"""
Configuration file for constants
Created by Mohammed
Optimized for SPEED, reliability, and real-time performance
"""

from __future__ import annotations

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths (absolute, stable no matter where python is run from)
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]  # .../smart-glasses-project-personal
RUNTIME_DIR = PROJECT_ROOT / "runtime"
RUNTIME_DIR.mkdir(parents=True, exist_ok=True)

LOCATION_JSON_PATH: str = str(RUNTIME_DIR / "location.json")

# ---------------------------------------------------------------------------
# Global debug flag (export DEBUG=1 to enable)
# ---------------------------------------------------------------------------

DEBUG: bool = os.getenv("DEBUG", "0") == "1"

# ---------------------------------------------------------------------------
# Environment / OpenAI Settings
# ---------------------------------------------------------------------------

OPENAI_API_KEY_PRESENT: bool = bool(os.getenv("OPENAI_API_KEY"))

OPENAI_VISION_MODEL: str = os.getenv("OPENAI_VISION_MODEL", "gpt-4o")
OPENAI_CHAT_MODEL: str = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")
OPENAI_TRANSCRIBE_MODEL: str = os.getenv("OPENAI_TRANSCRIBE_MODEL", "whisper-1")

# Some parts of your code expect this name:
OPENAI_TIMEOUT_SECONDS: float = float(os.getenv("OPENAI_TIMEOUT_SECONDS", "20.0"))

# ---------------------------------------------------------------------------
# Camera settings
# ---------------------------------------------------------------------------

DEFAULT_CAMERA_INDEX: int = int(os.getenv("CAMERA_INDEX", "0"))
DEFAULT_FRAME_WIDTH: int = int(os.getenv("FRAME_WIDTH", "640"))
DEFAULT_FRAME_HEIGHT: int = int(os.getenv("FRAME_HEIGHT", "480"))
CAMERA_LOW_LATENCY: bool = os.getenv("CAMERA_LOW_LATENCY", "1") == "1"

# ---------------------------------------------------------------------------
# YOLO object detection settings
# ---------------------------------------------------------------------------

DEFAULT_MODEL_NAME: str = os.getenv("YOLO_MODEL", "yolov8s-oiv7.pt")

DEFAULT_YOLO_CONFIDENCE_THRESHOLD: float = float(os.getenv("YOLO_CONF", "0.25"))
DEFAULT_IOU_THRESHOLD: float = float(os.getenv("YOLO_IOU", "0.45"))

DEFAULT_TRACKER: str = os.getenv("YOLO_TRACKER", "bytetrack.yaml")
DEFAULT_MAX_DETECTIONS: int = int(os.getenv("YOLO_MAX_DETECTIONS", "100"))

# ---------------------------------------------------------------------------
# PERFORMANCE SETTINGS
# ---------------------------------------------------------------------------

PROCESS_EVERY_N_FRAMES: int = int(os.getenv("PROCESS_EVERY_N_FRAMES", "3"))
YOLO_INFERENCE_SIZE: int = int(os.getenv("YOLO_INFERENCE_SIZE", "480"))

ENABLE_TRACKING: bool = os.getenv("ENABLE_TRACKING", "0") == "1"

USE_GPU: bool = os.getenv("USE_GPU", "1") == "1"
USE_HALF_PRECISION: bool = os.getenv("USE_HALF_PRECISION", "1") == "1"
AGNOSTIC_NMS: bool = os.getenv("AGNOSTIC_NMS", "0") == "1"

# ---------------------------------------------------------------------------
# Detection logic
# ---------------------------------------------------------------------------

SMALL_OBJECTS: set[str] = {
    "Pen",
    "Pencil",
    "Toothbrush",
    "Spoon",
    "Fork",
    "Knife",
    "Remote control",
    "Computer mouse",
    "Glasses",
    "Watch",
}

CONFIDENCE_BY_CATEGORY: dict[str, float] = {
    "small_objects": float(os.getenv("CONF_SMALL", "0.15")),
    "priority_objects": float(os.getenv("CONF_PRIORITY", "0.20")),
    "general_objects": float(os.getenv("CONF_GENERAL", "0.25")),
}

MAX_SPEECH_ITEMS: int = int(os.getenv("MAX_SPEECH_ITEMS", "5"))

# ---------------------------------------------------------------------------
# OCR settings (legacy)
# ---------------------------------------------------------------------------

DEFAULT_OCR_CONFIDENCE_THRESHOLD: float = float(os.getenv("OCR_CONF", "0.25"))

OCR_WHITELIST = (
    "book",
    "laptop",
    "cell phone",
    "sign",
    "bottle",
    "can",
)

# ---------------------------------------------------------------------------
# Speech / narration behavior
# ---------------------------------------------------------------------------

SPEAK_EVERY_N_FRAMES: int = int(os.getenv("SPEAK_EVERY_N_FRAMES", "0"))
MIN_SPEECH_INTERVAL_SECONDS: float = float(os.getenv("MIN_SPEECH_INTERVAL_SECONDS", "3.0"))
MANUAL_DESCRIBE_COOLDOWN_SECONDS: float = float(os.getenv("MANUAL_DESCRIBE_COOLDOWN_SECONDS", "0.25"))

# ---------------------------------------------------------------------------
# Debug UI / logging
# ---------------------------------------------------------------------------

SHOW_DEBUG_WINDOW: bool = os.getenv("SHOW_DEBUG_WINDOW", "1") == "1"
SAVE_DEBUG_FRAMES: bool = os.getenv("SAVE_DEBUG_FRAMES", "0") == "1"
DEBUG_FRAME_PATH: str = os.getenv("DEBUG_FRAME_PATH", "./debug_frames/")
LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")

SHOW_FPS: bool = os.getenv("SHOW_FPS", "1") == "1"
SHOW_DETECTION_COUNT: bool = os.getenv("SHOW_DETECTION_COUNT", "1") == "1"
DEBUG_PRINT_EVERY_N_FRAMES: int = int(os.getenv("DEBUG_PRINT_EVERY_N_FRAMES", "90"))

# ---------------------------------------------------------------------------
# Scene AI (vision API) performance tuning
# ---------------------------------------------------------------------------

SCENE_AI_MAX_WIDTH: int = int(os.getenv("SCENE_AI_MAX_WIDTH", "768"))
SCENE_AI_JPEG_QUALITY: int = int(os.getenv("SCENE_AI_JPEG_QUALITY", "75"))
SCENE_AI_MAX_TOKENS: int = int(os.getenv("SCENE_AI_MAX_TOKENS", "260"))
SCENE_AI_TEMPERATURE: float = float(os.getenv("SCENE_AI_TEMPERATURE", "0.2"))
SCENE_AI_FORCE_RGB: bool = os.getenv("SCENE_AI_FORCE_RGB", "0") == "1"

SCENE_AI_RETRIES: int = int(os.getenv("SCENE_AI_RETRIES", "2"))
SCENE_AI_RETRY_BASE_DELAY_S: float = float(os.getenv("SCENE_AI_RETRY_BASE_DELAY_S", "0.4"))
SCENE_AI_TIMEOUT_S: float = float(os.getenv("SCENE_AI_TIMEOUT_S", "12.0"))
SCENE_AI_DETECTIONS_MAX_ITEMS: int = int(os.getenv("SCENE_AI_DETECTIONS_MAX_ITEMS", "12"))

# ---------------------------------------------------------------------------
# GPS / Navigation
# ---------------------------------------------------------------------------

NAV_MIN_STEP_DISTANCE_M: float = float(os.getenv("NAV_MIN_STEP_DISTANCE_M", "18"))
NAV_MAX_STORED_STEPS: int = int(os.getenv("NAV_MAX_STORED_STEPS", "200"))

NAV_INITIAL_STEPS_SPOKEN: int = int(os.getenv("NAV_INITIAL_STEPS_SPOKEN", "4"))
NAV_CONTINUE_STEPS_SPOKEN: int = int(os.getenv("NAV_CONTINUE_STEPS_SPOKEN", "5"))
NAV_MAX_STEPS_PER_RESPONSE: int = int(os.getenv("NAV_MAX_STEPS_PER_RESPONSE", "10"))
NAV_MAX_TTS_CHARS: int = int(os.getenv("NAV_MAX_TTS_CHARS", "900"))

GPS_LOCATION_FILE: str = os.getenv("GPS_LOCATION_FILE", LOCATION_JSON_PATH)
GPS_STALE_SECONDS: float = float(os.getenv("GPS_STALE_SECONDS", "45"))

NAV_LOCATION_FILE: str = os.getenv("NAV_LOCATION_FILE", GPS_LOCATION_FILE)

ORS_API_KEY: str = os.getenv("ORS_API_KEY", "").strip()
OPENROUTESERVICE_API_KEY: str = os.getenv("OPENROUTESERVICE_API_KEY", ORS_API_KEY).strip()
if not ORS_API_KEY and OPENROUTESERVICE_API_KEY:
    ORS_API_KEY = OPENROUTESERVICE_API_KEY

ORS_BASE_URL: str = os.getenv("ORS_BASE_URL", "https://api.openrouteservice.org").strip()
ORS_TIMEOUT_S: float = float(os.getenv("ORS_TIMEOUT_S", "12.0"))

NAV_PROFILE: str = os.getenv("NAV_PROFILE", "foot-walking")
NAV_LANGUAGE: str = os.getenv("NAV_LANGUAGE", "en")
NAV_HTTP_TIMEOUT_S: float = float(os.getenv("NAV_HTTP_TIMEOUT_S", "12.0"))

NAV_ORIGIN_LAT: str = os.getenv("NAV_ORIGIN_LAT", "").strip()
NAV_ORIGIN_LON: str = os.getenv("NAV_ORIGIN_LON", "").strip()

NAV_GEO_SIZE: int = int(os.getenv("NAV_GEO_SIZE", "8"))
NAV_GEO_RADIUS_KM: float = float(os.getenv("NAV_GEO_RADIUS_KM", "25"))
NAV_GEO_COUNTRY: str = os.getenv("NAV_GEO_COUNTRY", "US").strip()
NAV_MAX_REASONABLE_KM: float = float(os.getenv("NAV_MAX_REASONABLE_KM", "150"))

# ---------------------------------------------------------------------------
# Obstacle detection / safety layer
# ---------------------------------------------------------------------------

OBSTACLE_ENABLED: bool = os.getenv("OBSTACLE_ENABLED", "1") == "1"
OBSTACLE_EVERY_N_FRAMES: int = int(os.getenv("OBSTACLE_EVERY_N_FRAMES", "3"))

OBSTACLE_MODE: str = os.getenv("OBSTACLE_MODE", "bbox").strip().lower()
OBSTACLE_DEPTH_MODEL: str = os.getenv("OBSTACLE_DEPTH_MODEL", "midas_small").strip().lower()

OBSTACLE_ALERT_COOLDOWN_S: float = float(os.getenv("OBSTACLE_ALERT_COOLDOWN_S", "2.0"))
OBSTACLE_REPEAT_AFTER_S: float = float(os.getenv("OBSTACLE_REPEAT_AFTER_S", "7.0"))

OBSTACLE_VERY_CLOSE_M: float = float(os.getenv("OBSTACLE_VERY_CLOSE_M", "0.9"))
OBSTACLE_CLOSE_M: float = float(os.getenv("OBSTACLE_CLOSE_M", "1.8"))

OBSTACLE_TOF_ENABLED: bool = os.getenv("OBSTACLE_TOF_ENABLED", "0") == "1"
OBSTACLE_TOF_I2C_BUS: int = int(os.getenv("OBSTACLE_TOF_I2C_BUS", "1"))
OBSTACLE_TOF_TIMING_BUDGET_MS: int = int(os.getenv("OBSTACLE_TOF_TIMING_BUDGET_MS", "50"))

OBSTACLE_PRIORITY_KEYWORDS: tuple[str, ...] = (
    "stairs", "stair", "step", "steps", "curb", "kerb",
    "pole", "post", "bollard",
    "chair", "table",
    "bicycle", "bike",
    "dog", "cat", "pet",
    "skateboard", "scooter",
)

# ---------------------------------------------------------------------------
# Depth Estimation (MiDaS) - Optional
# ---------------------------------------------------------------------------

DEPTH_ENABLED: bool = os.getenv("DEPTH_ENABLED", "0") == "1"
DEPTH_MODEL: str = os.getenv("DEPTH_MODEL", "MiDaS_small").strip()
DEPTH_EVERY_N_FRAMES: int = int(os.getenv("DEPTH_EVERY_N_FRAMES", "10"))
DEPTH_INPUT_MAX_WIDTH: int = int(os.getenv("DEPTH_INPUT_MAX_WIDTH", "256"))

DEPTH_STEP_DOWN_ENABLED: bool = os.getenv("DEPTH_STEP_DOWN_ENABLED", "1") == "1"
DEPTH_STEP_DOWN_DROP_RATIO: float = float(os.getenv("DEPTH_STEP_DOWN_DROP_RATIO", "0.22"))
DEPTH_STEP_DOWN_MIN_UPPER: float = float(os.getenv("DEPTH_STEP_DOWN_MIN_UPPER", "0.45"))

DEPTH_CLOSE_THRESH: float = float(os.getenv("DEPTH_CLOSE_THRESH", "0.55"))
DEPTH_VERY_CLOSE_THRESH: float = float(os.getenv("DEPTH_VERY_CLOSE_THRESH", "0.70"))

# ---------------------------------------------------------------------------
# Guidance mode (continuous assistance)
# ---------------------------------------------------------------------------

GUIDANCE_ENABLED: bool = os.getenv("GUIDANCE_ENABLED", "0") == "1"
GUIDANCE_PROFILE: str = os.getenv("GUIDANCE_PROFILE", "indoor").strip().lower()
GUIDANCE_EVERY_N_FRAMES: int = int(os.getenv("GUIDANCE_EVERY_N_FRAMES", "6"))

GUIDANCE_COOLDOWN_S: float = float(os.getenv("GUIDANCE_COOLDOWN_S", "2.2"))
GUIDANCE_REPEAT_AFTER_S: float = float(os.getenv("GUIDANCE_REPEAT_AFTER_S", "9.0"))
GUIDANCE_ESCALATE_MIN_GAP_S: float = float(os.getenv("GUIDANCE_ESCALATE_MIN_GAP_S", "1.2"))
GUIDANCE_INFO_MIN_GAP_S: float = float(os.getenv("GUIDANCE_INFO_MIN_GAP_S", "18.0"))

GUIDANCE_SUPPRESS_DURING_VOICE: bool = os.getenv("GUIDANCE_SUPPRESS_DURING_VOICE", "1") == "1"
GUIDANCE_SUPPRESS_DURING_OBSTACLE: bool = os.getenv("GUIDANCE_SUPPRESS_DURING_OBSTACLE", "1") == "1"

GUIDANCE_MIN_CONF: float = float(os.getenv("GUIDANCE_MIN_CONF", "0.20"))

GUIDANCE_BBOX_NEAR_AREA_RATIO: float = float(os.getenv("GUIDANCE_BBOX_NEAR_AREA_RATIO", "0.10"))
GUIDANCE_BBOX_CLOSE_AREA_RATIO: float = float(os.getenv("GUIDANCE_BBOX_CLOSE_AREA_RATIO", "0.18"))
GUIDANCE_BBOX_DANGER_AREA_RATIO: float = float(os.getenv("GUIDANCE_BBOX_DANGER_AREA_RATIO", "0.28"))

# ---------------------------------------------------------------------------
# Hybrid OCR (local first, optional escalate to SceneAI)
# ---------------------------------------------------------------------------

OCR_ENGINE: str = os.getenv("OCR_ENGINE", "auto").strip().lower()  # auto | easyocr | tesseract
OCR_LANGS: str = os.getenv("OCR_LANGS", "en")  # comma-separated, e.g. "en" or "en,es"
OCR_LOCAL_MIN_CONF: float = float(os.getenv("OCR_LOCAL_MIN_CONF", "0.55"))  # below => escalate to SceneAI
OCR_MIN_CHARS: int = int(os.getenv("OCR_MIN_CHARS", "12"))  # too little text => escalate

OCR_USE_REGION_DETECTION: bool = os.getenv("OCR_USE_REGION_DETECTION", "1") == "1"
OCR_MAX_REGIONS: int = int(os.getenv("OCR_MAX_REGIONS", "8"))
OCR_REGION_PADDING_PX: int = int(os.getenv("OCR_REGION_PADDING_PX", "8"))

OCR_PREPROCESS_SCALE: float = float(os.getenv("OCR_PREPROCESS_SCALE", "1.6"))  # enlarge for OCR
OCR_BINARIZE: bool = os.getenv("OCR_BINARIZE", "1") == "1"

# Document mode cache
DOC_CACHE_TTL_S: float = float(os.getenv("DOC_CACHE_TTL_S", "60.0"))

# Scene AI OCR
SCENE_OCR_MAX_TOKENS: int = int(os.getenv("SCENE_OCR_MAX_TOKENS", "700"))
SCENE_OCR_TEMPERATURE: float = float(os.getenv("SCENE_OCR_TEMPERATURE", "0.0"))

# Avoid spamming Scene OCR when called repeatedly
OCR_SCENE_COOLDOWN_S: float = float(os.getenv("OCR_SCENE_COOLDOWN_S", "1.2"))

# ✅ Default reading mode used by controller / document reader
# Allowed: "offline" | "hybrid" | "ai"
READING_MODE_DEFAULT: str = os.getenv("READING_MODE_DEFAULT", "hybrid").strip().lower()

# Backward-compatible alias (some older code might look for OCR_MODE)
# Allowed: "hybrid" | "local_only" | "scene_only"
OCR_MODE: str = os.getenv("OCR_MODE", "").strip().lower()

# ---------------------------------------------------------------------------
# Speech priority / interruption control
# ---------------------------------------------------------------------------

SPEECH_MUTE_SAFETY_DURING_READING: bool = os.getenv("SPEECH_MUTE_SAFETY_DURING_READING", "1") == "1"
SAFETY_SPEECH_ENABLED: bool = os.getenv("SAFETY_SPEECH_ENABLED", "1") == "1"

TELEMETRY_ENABLED = True
TELEM_MAX_DETS_LOG = 20
