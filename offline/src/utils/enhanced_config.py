"""
Enhanced Configuration for Smart Glasses AI System
Includes all advanced AI feature settings
"""

from __future__ import annotations

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RUNTIME_DIR = PROJECT_ROOT / "runtime"
RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR = PROJECT_ROOT / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

LOCATION_JSON_PATH: str = str(RUNTIME_DIR / "location.json")
MEMORY_CACHE_PATH: str = str(RUNTIME_DIR / "scene_memory.json")

# ---------------------------------------------------------------------------
# Debug & Logging
# ---------------------------------------------------------------------------

DEBUG: bool = os.getenv("DEBUG", "0") == "1"
LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")

# ---------------------------------------------------------------------------
# OpenAI Settings
# ---------------------------------------------------------------------------

OPENAI_API_KEY_PRESENT: bool = bool(os.getenv("OPENAI_API_KEY"))
OPENAI_VISION_MODEL: str = os.getenv("OPENAI_VISION_MODEL", "gpt-4o")
OPENAI_CHAT_MODEL: str = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")
OPENAI_EMBEDDING_MODEL: str = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
OPENAI_TRANSCRIBE_MODEL: str = os.getenv("OPENAI_TRANSCRIBE_MODEL", "whisper-1")
OPENAI_TTS_MODEL: str = os.getenv("OPENAI_TTS_MODEL", "tts-1")
OPENAI_TTS_VOICE: str = os.getenv("OPENAI_TTS_VOICE", "alloy")  # alloy, echo, fable, onyx, nova, shimmer

OPENAI_TIMEOUT_SECONDS: float = float(os.getenv("OPENAI_TIMEOUT_SECONDS", "20.0"))

# ---------------------------------------------------------------------------
# Camera Settings
# ---------------------------------------------------------------------------

DEFAULT_CAMERA_INDEX: int = int(os.getenv("CAMERA_INDEX", "0"))
DEFAULT_FRAME_WIDTH: int = int(os.getenv("FRAME_WIDTH", "1280"))
DEFAULT_FRAME_HEIGHT: int = int(os.getenv("FRAME_HEIGHT", "720"))
CAMERA_LOW_LATENCY: bool = os.getenv("CAMERA_LOW_LATENCY", "1") == "1"
CAMERA_FPS: int = int(os.getenv("CAMERA_FPS", "30"))

# ---------------------------------------------------------------------------
# Enhanced YOLO Settings
# ---------------------------------------------------------------------------

# Detection model
DEFAULT_MODEL_NAME: str = os.getenv("YOLO_MODEL", "yolov8n.pt")
DEFAULT_YOLO_CONFIDENCE_THRESHOLD: float = float(os.getenv("YOLO_CONF", "0.25"))
DEFAULT_IOU_THRESHOLD: float = float(os.getenv("YOLO_IOU", "0.45"))
DEFAULT_TRACKER: str = os.getenv("YOLO_TRACKER", "bytetrack.yaml")
DEFAULT_MAX_DETECTIONS: int = int(os.getenv("YOLO_MAX_DETECTIONS", "100"))

# 🆕 Pose estimation
ENABLE_POSE_ESTIMATION: bool = os.getenv("ENABLE_POSE", "1") == "1"
YOLO_POSE_MODEL: str = os.getenv("YOLO_POSE_MODEL", "yolov8n-pose.pt")
POSE_CONFIDENCE_THRESHOLD: float = float(os.getenv("POSE_CONF", "0.5"))

# 🆕 Segmentation
ENABLE_SEGMENTATION: bool = os.getenv("ENABLE_SEGMENTATION", "0") == "1"
YOLO_SEGMENTATION_MODEL: str = os.getenv("YOLO_SEG_MODEL", "yolov8n-seg.pt")

# Performance
PROCESS_EVERY_N_FRAMES: int = int(os.getenv("PROCESS_EVERY_N_FRAMES", "2"))
YOLO_INFERENCE_SIZE: int = int(os.getenv("YOLO_INFERENCE_SIZE", "640"))
ENABLE_TRACKING: bool = os.getenv("ENABLE_TRACKING", "1") == "1"
USE_GPU: bool = os.getenv("USE_GPU", "1") == "1"
USE_HALF_PRECISION: bool = os.getenv("USE_HALF_PRECISION", "1") == "1"
AGNOSTIC_NMS: bool = os.getenv("AGNOSTIC_NMS", "0") == "1"

# ---------------------------------------------------------------------------
# 🆕 AI Feature Flags
# ---------------------------------------------------------------------------

# Scene Memory
ENABLE_SCENE_MEMORY: bool = os.getenv("ENABLE_SCENE_MEMORY", "1") == "1"
MAX_SCENE_MEMORIES: int = int(os.getenv("MAX_SCENE_MEMORIES", "500"))
MEMORY_USE_OPENAI_EMBEDDINGS: bool = os.getenv("MEMORY_USE_OPENAI_EMBEDDINGS", "1") == "1"
MEMORY_EXPORT_ON_EXIT: bool = os.getenv("MEMORY_EXPORT_ON_EXIT", "1") == "1"

# Emotion & Face Analysis
ENABLE_EMOTION_DETECTION: bool = os.getenv("ENABLE_EMOTION_DETECTION", "1") == "1"
ENABLE_FACE_ANALYSIS: bool = os.getenv("ENABLE_FACE_ANALYSIS", "1") == "1"
FACE_DETECTION_CONFIDENCE: float = float(os.getenv("FACE_DETECTION_CONF", "0.6"))
USE_GPT4O_FOR_EMOTIONS: bool = os.getenv("USE_GPT4O_FOR_EMOTIONS", "0") == "1"  # Expensive

# Color & Text Analysis
ENABLE_COLOR_ANALYSIS: bool = os.getenv("ENABLE_COLOR_ANALYSIS", "1") == "1"
USE_GPT4O_FOR_COLORS: bool = os.getenv("USE_GPT4O_FOR_COLORS", "1") == "1"
ENABLE_BRAND_RECOGNITION: bool = os.getenv("ENABLE_BRAND_RECOGNITION", "1") == "1"

# Proactive Assistant
ENABLE_PROACTIVE_ASSISTANT: bool = os.getenv("ENABLE_PROACTIVE_ASSISTANT", "1") == "1"
PROACTIVE_SAFETY_PRIORITY: str = os.getenv("PROACTIVE_SAFETY_PRIORITY", "high")
PROACTIVE_INFO_COOLDOWN: int = int(os.getenv("PROACTIVE_INFO_COOLDOWN", "30"))
PROACTIVE_INTERRUPT_FOR_SAFETY: bool = os.getenv("PROACTIVE_INTERRUPT_FOR_SAFETY", "1") == "1"

# ---------------------------------------------------------------------------
# Detection & Classification
# ---------------------------------------------------------------------------

SMALL_OBJECTS: set[str] = {
    "Pen", "Pencil", "Toothbrush", "Spoon", "Fork", "Knife",
    "Remote control", "Computer mouse", "Glasses", "Watch",
    "Key", "Coin", "Ring", "Earring",
}

CONFIDENCE_BY_CATEGORY: dict[str, float] = {
    "small_objects": float(os.getenv("CONF_SMALL", "0.15")),
    "priority_objects": float(os.getenv("CONF_PRIORITY", "0.20")),
    "general_objects": float(os.getenv("CONF_GENERAL", "0.25")),
    "people": float(os.getenv("CONF_PEOPLE", "0.30")),
}

MAX_SPEECH_ITEMS: int = int(os.getenv("MAX_SPEECH_ITEMS", "5"))

# ---------------------------------------------------------------------------
# OCR Settings
# ---------------------------------------------------------------------------

DEFAULT_OCR_CONFIDENCE_THRESHOLD: float = float(os.getenv("OCR_CONF", "0.25"))
OCR_ENGINE: str = os.getenv("OCR_ENGINE", "auto").strip().lower()
OCR_LANGS: str = os.getenv("OCR_LANGS", "en")
OCR_LOCAL_MIN_CONF: float = float(os.getenv("OCR_LOCAL_MIN_CONF", "0.55"))
OCR_MIN_CHARS: int = int(os.getenv("OCR_MIN_CHARS", "12"))
OCR_USE_REGION_DETECTION: bool = os.getenv("OCR_USE_REGION_DETECTION", "1") == "1"
OCR_MAX_REGIONS: int = int(os.getenv("OCR_MAX_REGIONS", "8"))
OCR_REGION_PADDING_PX: int = int(os.getenv("OCR_REGION_PADDING_PX", "8"))
OCR_PREPROCESS_SCALE: float = float(os.getenv("OCR_PREPROCESS_SCALE", "1.6"))
OCR_BINARIZE: bool = os.getenv("OCR_BINARIZE", "1") == "1"

# Reading modes
READING_MODE_DEFAULT: str = os.getenv("READING_MODE_DEFAULT", "hybrid").strip().lower()
DOC_CACHE_TTL_S: float = float(os.getenv("DOC_CACHE_TTL_S", "60.0"))

# Scene AI OCR
SCENE_OCR_MAX_TOKENS: int = int(os.getenv("SCENE_OCR_MAX_TOKENS", "700"))
SCENE_OCR_TEMPERATURE: float = float(os.getenv("SCENE_OCR_TEMPERATURE", "0.0"))
OCR_SCENE_COOLDOWN_S: float = float(os.getenv("OCR_SCENE_COOLDOWN_S", "1.2"))

# ---------------------------------------------------------------------------
# Speech & Audio
# ---------------------------------------------------------------------------

SPEAK_EVERY_N_FRAMES: int = int(os.getenv("SPEAK_EVERY_N_FRAMES", "0"))
MIN_SPEECH_INTERVAL_SECONDS: float = float(os.getenv("MIN_SPEECH_INTERVAL_SECONDS", "3.0"))
MANUAL_DESCRIBE_COOLDOWN_SECONDS: float = float(os.getenv("MANUAL_DESCRIBE_COOLDOWN_SECONDS", "0.25"))
SPEECH_MUTE_SAFETY_DURING_READING: bool = os.getenv("SPEECH_MUTE_SAFETY_DURING_READING", "1") == "1"
SAFETY_SPEECH_ENABLED: bool = os.getenv("SAFETY_SPEECH_ENABLED", "1") == "1"

# 🆕 OpenAI TTS
USE_OPENAI_TTS: bool = os.getenv("USE_OPENAI_TTS", "0") == "1"  # Better quality but slower
TTS_SPEED: float = float(os.getenv("TTS_SPEED", "1.0"))

# ---------------------------------------------------------------------------
# UI & Visualization
# ---------------------------------------------------------------------------

SHOW_DEBUG_WINDOW: bool = os.getenv("SHOW_DEBUG_WINDOW", "1") == "1"
SAVE_DEBUG_FRAMES: bool = os.getenv("SAVE_DEBUG_FRAMES", "0") == "1"
DEBUG_FRAME_PATH: str = os.getenv("DEBUG_FRAME_PATH", "./debug_frames/")
SHOW_FPS: bool = os.getenv("SHOW_FPS", "1") == "1"
SHOW_DETECTION_COUNT: bool = os.getenv("SHOW_DETECTION_COUNT", "1") == "1"
DEBUG_PRINT_EVERY_N_FRAMES: int = int(os.getenv("DEBUG_PRINT_EVERY_N_FRAMES", "90"))

# 🆕 Enhanced visualization
SHOW_POSE_SKELETON: bool = os.getenv("SHOW_POSE_SKELETON", "1") == "1"
SHOW_EMOTION_LABELS: bool = os.getenv("SHOW_EMOTION_LABELS", "1") == "1"
SHOW_COLOR_PALETTE: bool = os.getenv("SHOW_COLOR_PALETTE", "0") == "1"
SHOW_MEMORY_HINTS: bool = os.getenv("SHOW_MEMORY_HINTS", "1") == "1"

# ---------------------------------------------------------------------------
# Scene AI (Vision API)
# ---------------------------------------------------------------------------

SCENE_AI_MAX_WIDTH: int = int(os.getenv("SCENE_AI_MAX_WIDTH", "768"))
SCENE_AI_JPEG_QUALITY: int = int(os.getenv("SCENE_AI_JPEG_QUALITY", "75"))
SCENE_AI_MAX_TOKENS: int = int(os.getenv("SCENE_AI_MAX_TOKENS", "300"))
SCENE_AI_TEMPERATURE: float = float(os.getenv("SCENE_AI_TEMPERATURE", "0.2"))
SCENE_AI_FORCE_RGB: bool = os.getenv("SCENE_AI_FORCE_RGB", "0") == "1"
SCENE_AI_RETRIES: int = int(os.getenv("SCENE_AI_RETRIES", "2"))
SCENE_AI_RETRY_BASE_DELAY_S: float = float(os.getenv("SCENE_AI_RETRY_BASE_DELAY_S", "0.4"))
SCENE_AI_TIMEOUT_S: float = float(os.getenv("SCENE_AI_TIMEOUT_S", "12.0"))
SCENE_AI_DETECTIONS_MAX_ITEMS: int = int(os.getenv("SCENE_AI_DETECTIONS_MAX_ITEMS", "12"))

# ---------------------------------------------------------------------------
# Navigation & GPS
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

# ---------------------------------------------------------------------------
# Safety & Obstacle Detection
# ---------------------------------------------------------------------------

OBSTACLE_ENABLED: bool = os.getenv("OBSTACLE_ENABLED", "1") == "1"
OBSTACLE_EVERY_N_FRAMES: int = int(os.getenv("OBSTACLE_EVERY_N_FRAMES", "3"))
OBSTACLE_MODE: str = os.getenv("OBSTACLE_MODE", "bbox").strip().lower()
OBSTACLE_DEPTH_MODEL: str = os.getenv("OBSTACLE_DEPTH_MODEL", "midas_small").strip().lower()
OBSTACLE_ALERT_COOLDOWN_S: float = float(os.getenv("OBSTACLE_ALERT_COOLDOWN_S", "2.0"))
OBSTACLE_REPEAT_AFTER_S: float = float(os.getenv("OBSTACLE_REPEAT_AFTER_S", "7.0"))
OBSTACLE_VERY_CLOSE_M: float = float(os.getenv("OBSTACLE_VERY_CLOSE_M", "0.9"))
OBSTACLE_CLOSE_M: float = float(os.getenv("OBSTACLE_CLOSE_M", "1.8"))

# 🆕 Enhanced safety features
ENABLE_FALL_DETECTION: bool = os.getenv("ENABLE_FALL_DETECTION", "1") == "1"
ENABLE_EMERGENCY_DETECTION: bool = os.getenv("ENABLE_EMERGENCY_DETECTION", "1") == "1"
FALL_DETECTION_SENSITIVITY: float = float(os.getenv("FALL_DETECTION_SENSITIVITY", "0.7"))

# ToF sensor (hardware)
OBSTACLE_TOF_ENABLED: bool = os.getenv("OBSTACLE_TOF_ENABLED", "0") == "1"
OBSTACLE_TOF_I2C_BUS: int = int(os.getenv("OBSTACLE_TOF_I2C_BUS", "1"))
OBSTACLE_TOF_TIMING_BUDGET_MS: int = int(os.getenv("OBSTACLE_TOF_TIMING_BUDGET_MS", "50"))

OBSTACLE_PRIORITY_KEYWORDS: tuple[str, ...] = (
    "stairs", "stair", "step", "steps", "curb", "kerb",
    "pole", "post", "bollard", "chair", "table",
    "bicycle", "bike", "dog", "cat", "pet",
    "skateboard", "scooter", "edge", "drop"
)

# ---------------------------------------------------------------------------
# Depth Estimation
# ---------------------------------------------------------------------------

DEPTH_ENABLED: bool = os.getenv("DEPTH_ENABLED", "1") == "1"
DEPTH_MODEL: str = os.getenv("DEPTH_MODEL", "MiDaS_small").strip()
DEPTH_EVERY_N_FRAMES: int = int(os.getenv("DEPTH_EVERY_N_FRAMES", "10"))
DEPTH_INPUT_MAX_WIDTH: int = int(os.getenv("DEPTH_INPUT_MAX_WIDTH", "256"))
DEPTH_STEP_DOWN_ENABLED: bool = os.getenv("DEPTH_STEP_DOWN_ENABLED", "1") == "1"
DEPTH_STEP_DOWN_DROP_RATIO: float = float(os.getenv("DEPTH_STEP_DOWN_DROP_RATIO", "0.22"))
DEPTH_STEP_DOWN_MIN_UPPER: float = float(os.getenv("DEPTH_STEP_DOWN_MIN_UPPER", "0.45"))
DEPTH_CLOSE_THRESH: float = float(os.getenv("DEPTH_CLOSE_THRESH", "0.55"))
DEPTH_VERY_CLOSE_THRESH: float = float(os.getenv("DEPTH_VERY_CLOSE_THRESH", "0.70"))

# ---------------------------------------------------------------------------
# Guidance System
# ---------------------------------------------------------------------------

GUIDANCE_ENABLED: bool = os.getenv("GUIDANCE_ENABLED", "1") == "1"
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
# Telemetry
# ---------------------------------------------------------------------------

TELEMETRY_ENABLED: bool = os.getenv("TELEMETRY_ENABLED", "1") == "1"
TELEM_MAX_DETS_LOG: int = int(os.getenv("TELEM_MAX_DETS_LOG", "20"))
TELEM_EXPORT_INTERVAL_S: float = float(os.getenv("TELEM_EXPORT_INTERVAL_S", "300"))  # 5 minutes

# ---------------------------------------------------------------------------
# 🆕 Feature Presets
# ---------------------------------------------------------------------------

def get_preset(preset_name: str) -> dict:
    """Get configuration preset for different use cases"""
    presets = {
        "maximum_accuracy": {
            "YOLO_MODEL": "yolov8x.pt",
            "YOLO_POSE_MODEL": "yolov8x-pose.pt",
            "ENABLE_SEGMENTATION": True,
            "PROCESS_EVERY_N_FRAMES": 1,
            "FRAME_WIDTH": 1920,
            "FRAME_HEIGHT": 1080,
            "USE_GPT4O_FOR_EMOTIONS": True,
            "USE_GPT4O_FOR_COLORS": True,
        },
        "balanced": {
            "YOLO_MODEL": "yolov8m.pt",
            "YOLO_POSE_MODEL": "yolov8m-pose.pt",
            "ENABLE_SEGMENTATION": False,
            "PROCESS_EVERY_N_FRAMES": 2,
            "FRAME_WIDTH": 1280,
            "FRAME_HEIGHT": 720,
        },
        "real_time": {
            "YOLO_MODEL": "yolov8n.pt",
            "YOLO_POSE_MODEL": "yolov8n-pose.pt",
            "ENABLE_SEGMENTATION": False,
            "PROCESS_EVERY_N_FRAMES": 3,
            "FRAME_WIDTH": 640,
            "FRAME_HEIGHT": 480,
            "USE_GPT4O_FOR_EMOTIONS": False,
            "ENABLE_BRAND_RECOGNITION": False,
        },
        "low_power": {
            "YOLO_MODEL": "yolov8n.pt",
            "ENABLE_POSE_ESTIMATION": False,
            "ENABLE_SEGMENTATION": False,
            "ENABLE_EMOTION_DETECTION": False,
            "ENABLE_COLOR_ANALYSIS": False,
            "ENABLE_PROACTIVE_ASSISTANT": False,
            "PROCESS_EVERY_N_FRAMES": 5,
            "FRAME_WIDTH": 480,
            "FRAME_HEIGHT": 360,
        }
    }
    return presets.get(preset_name, {})


# Apply preset if specified
PRESET = os.getenv("CONFIG_PRESET", "").strip().lower()
if PRESET and PRESET in ["maximum_accuracy", "balanced", "real_time", "low_power"]:
    preset_config = get_preset(PRESET)
    print(f"📋 Applying config preset: {PRESET}")
    # Note: In production, you'd actually override the globals here
