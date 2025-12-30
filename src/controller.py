# src/controller.py

from __future__ import annotations

# --- allow running as: python src/controller.py (without "No module named src") ---
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from collections import deque, OrderedDict
from typing import List, Dict, Any, Optional, Tuple
import time
import threading
import datetime

import cv2 as cv
import numpy as np

from src.camera_handler import CameraHandler
from src.object_detector import ObjectDetector
from src.currency_recognizer import CurrencyRecognizer
from src.ocr_engine import OCREngine
from src.speech_engine import SpeechEngine
from src.scene_ai_client import SceneAIClient
from src.voice_listener import VoiceListener
from src.ai_features.scene_memory import SceneMemoryEngine
from src.ai_features.human_analyzer import HumanAnalyzer, create_human_analyzer

# Sign Language Interpreter import
from src.ai_features.sign_language_interpreter import (
    SignLanguageInterpreter,
    InterpreterMode,
    create_sign_interpreter,
)

from src.weather_client import WeatherClient
from src.navigation_client import NavigationClient
from src.assistant_brain import AssistantBrain
import src.utils.config as config

from src.safety.obstacle_layer import ObstacleLayer
from src.safety.depth_estimator import DepthEstimator
from src.safety.guidance_engine import GuidanceEngine

from src.document_reader import DocumentReader

# Enhanced telemetry imports
from src.utils.telemetry import (
    TelemetryLogger,
    set_global_logger,
    log_voice,
    log_safety,
    log_ai,
    log_speech,
    log_error,
    log_system_health,
)


# ---------------------------------------------------------------------------
# Label handling
# ---------------------------------------------------------------------------

IGNORE_LABELS = {
    "Clothing",
    "Human arm",
    "Human hair",
    "Human leg",
    "Human body",
    "Human head",
    "Human ear",
    "Human eye",
    "Human mouth",
    "Human nose",
    "Human hand",
    "Human foot",
    "Human face",
    "Fashion accessory",
}

MERGE_LABELS = {
    "Human face": "person",
    "Man": "person",
    "Woman": "person",
    "Boy": "person",
    "Girl": "person",
    "Person": "person",
    "Laptop computer": "laptop",
    "Computer keyboard": "keyboard",
    "Computer mouse": "mouse",
    "Mobile phone": "phone",
    "Cellular telephone": "phone",
    "Telephone": "phone",
    "Television": "TV",
    "Drink": "beverage",
}

PRIORITY_LABELS = {
    "person",
    "Door",
    "Door handle",
    "Stairs",
    "Chair",
    "Table",
    "Car",
    "Bus",
    "Truck",
    "Bicycle",
    "Motorcycle",
    "Traffic light",
    "Traffic sign",
    "Stop sign",
    "Laptop",
    "laptop",
    "phone",
    "Mug",
    "Bottle",
    "Toilet",
    "Sink",
    "Bed",
    "Couch",
}


def normalize_label(label: str) -> Optional[str]:
    if label in IGNORE_LABELS:
        return None
    if label in MERGE_LABELS:
        return MERGE_LABELS[label]
    return label


def direction_from_center(center: Optional[Tuple[float, float]], frame_width: int) -> Optional[str]:
    if center is None or frame_width <= 0:
        return None

    x = center[0]
    left_thresh = frame_width / 3
    right_thresh = 2 * frame_width / 3

    if x < left_thresh:
        return "on your left"
    if x > right_thresh:
        return "on your right"
    return "in front of you"


def add_indefinite_article(label: str) -> str:
    if not label:
        return label
    first_letter = label[0].lower()
    return f"an {label}" if first_letter in "aeiou" else f"a {label}"


def pluralize(label: str) -> str:
    if not label:
        return label
    if label.endswith("s"):
        return label
    return label + "s"


def get_confidence_threshold(label: str) -> float:
    if label in getattr(config, "SMALL_OBJECTS", set()):
        return float(getattr(config, "CONFIDENCE_BY_CATEGORY", {}).get("small_objects", 0.15))
    if label in PRIORITY_LABELS:
        return float(getattr(config, "CONFIDENCE_BY_CATEGORY", {}).get("priority_objects", 0.20))
    return float(getattr(config, "CONFIDENCE_BY_CATEGORY", {}).get("general_objects", 0.25))


def summarize_detections(detections: List[Dict[str, Any]], frame_width: int, max_items: Optional[int] = None) -> str:
    if max_items is None:
        max_items = int(getattr(config, "MAX_SPEECH_ITEMS", 5) or 5)

    filtered = []
    for d in detections or []:
        raw_label = d.get("label", "object")
        cleaned = normalize_label(raw_label)
        if cleaned is None:
            continue

        conf = float(d.get("confidence", 0.0))
        required_conf = get_confidence_threshold(cleaned)
        if conf >= required_conf:
            filtered.append({"label": cleaned, "confidence": conf, "center": d.get("center")})

    if not filtered:
        return "I don't see any objects clearly."

    filtered.sort(key=lambda x: x["confidence"], reverse=True)

    priority = [d for d in filtered if d["label"] in PRIORITY_LABELS]
    non_priority = [d for d in filtered if d["label"] not in PRIORITY_LABELS]
    combined = (priority + non_priority)[: max_items * 2]

    groups: "OrderedDict[Tuple[str, Optional[str]], int]" = OrderedDict()
    for d in combined:
        label = d["label"]
        direction = direction_from_center(d["center"], frame_width)
        key = (label, direction)
        groups[key] = groups.get(key, 0) + 1

    phrases = []
    for idx, ((label, direction), count) in enumerate(groups.items()):
        if idx >= max_items:
            break

        if count == 1:
            base = add_indefinite_article(label)
        elif count == 2:
            base = f"two {pluralize(label)}"
        elif count == 3:
            base = f"three {pluralize(label)}"
        else:
            base = f"{count} {pluralize(label)}"

        phrases.append(f"{base} {direction}".strip() if direction else base)

    if len(phrases) == 1:
        return f"I see {phrases[0]}."
    if len(phrases) == 2:
        return f"I see {phrases[0]} and {phrases[1]}."
    return f"I see {', '.join(phrases[:-1])}, and {phrases[-1]}."


def _simple_guidance_type(msg: str) -> str:
    t = (msg or "").lower()
    if "left" in t:
        return "left"
    if "right" in t:
        return "right"
    if "stop" in t or "wait" in t:
        return "stop"
    if "slow" in t:
        return "slow"
    if "forward" in t or "ahead" in t or "go" in t:
        return "forward"
    return "other"


def _extract_direction_from_msg(msg: str) -> Optional[str]:
    """Extract direction from a guidance/obstacle message."""
    t = (msg or "").lower()
    if "left" in t:
        return "left"
    if "right" in t:
        return "right"
    if "ahead" in t or "front" in t:
        return "ahead"
    return None


def _extract_distance_from_msg(msg: str) -> Optional[str]:
    """Extract distance from a guidance/obstacle message."""
    t = (msg or "").lower()
    if "very close" in t:
        return "very_close"
    if "close" in t:
        return "close"
    if "near" in t:
        return "near"
    return None


class MainController:
    def __init__(self):
        self.camera = CameraHandler()
        self.detector = ObjectDetector()
        self.currency = CurrencyRecognizer()
        self.ocr = OCREngine()
        self.speech = SpeechEngine()
        self.scene_ai = SceneAIClient()

        self.voice_listener = VoiceListener()
        self.weather_client = WeatherClient()
        self.navigation_client = NavigationClient()

        self.assistant = AssistantBrain(
            scene_ai=self.scene_ai,
            weather_client=self.weather_client,
            navigation_client=self.navigation_client,
        )
        
        # Scene Memory System
        self.scene_memory = SceneMemoryEngine(max_memories=500)

        # Safety & guidance
        self.obstacle_layer = ObstacleLayer()
        self.depth_estimator = DepthEstimator()
        self.guidance = GuidanceEngine()

        # Document mode
        self.doc_reader = DocumentReader(self.ocr)

        # FPS tracking (use perf_counter for accuracy)
        self.fps_queue = deque(maxlen=30)
        self.last_frame_time = time.perf_counter()

        self.last_detections: List[Dict[str, Any]] = []
        self.last_annotated = None

        self.last_manual_describe_time: float = 0.0
        self.last_auto_speak_time: float = 0.0

        # Locks (avoid overlaps)
        self._voice_lock = threading.Lock()
        self._describe_lock = threading.Lock()
        self._obstacle_lock = threading.Lock()

        # Single "TTS lane" so nothing overlaps audio
        self._tts_lock = threading.Lock()

        self._voice_busy = False
        self._describe_busy = False
        self._obstacle_busy = False

        # Reading state (used to suppress warnings during reading)
        self._reading_lock = threading.Lock()
        self._is_reading = False

        # Runtime toggle: safety speech ON/OFF (default from config)
        self._safety_speech_enabled = bool(getattr(config, "SAFETY_SPEECH_ENABLED", True))
        self._mute_safety_during_reading = bool(getattr(config, "SPEECH_MUTE_SAFETY_DURING_READING", True))

        # Reading mode controls (offline vs AI)
        self._read_mode_lock = threading.Lock()
        self._read_mode = (
            self.ocr.get_mode()
            if hasattr(self.ocr, "get_mode")
            else (getattr(config, "OCR_MODE", "hybrid") or "hybrid")
        )

        # Backward-compatible alias (older code referenced _reading_mode)
        self._reading_mode = self._read_mode

        # Request IDs to link voice -> TTS -> events
        self._req_lock = threading.Lock()
        self._req_counter = 0

        print("⚡ FAST Smart Glasses System Initialized (AI narration + voice assistant)")
        print(f"📊 Model: {getattr(config, 'DEFAULT_MODEL_NAME', 'yolo')}")
        print(f"🎯 Processing every {getattr(config, 'PROCESS_EVERY_N_FRAMES', 3)} frames")
        print(f"🧠 Scene AI enabled: {getattr(self.scene_ai, 'enabled', True)}")
        print("🎛 Controls: 'q' quit | 'd' describe | 'v' voice | 'r' read | 's' toggle warnings")
        print("📖 Reading mode keys: '1' offline | '2' hybrid | '3' AI-only | 'm' cycle mode")
        print("🤟 Sign language: 'g' toggle sign mode")

        if getattr(config, "OBSTACLE_ENABLED", False):
            print(f"🧯 Safety layer: enabled (mode={getattr(config, 'OBSTACLE_MODE', 'bbox')})")
        if getattr(config, "GUIDANCE_ENABLED", False):
            print(f"🧭 Guidance: enabled (profile={getattr(config, 'GUIDANCE_PROFILE', 'indoor')})")

        print(f"🔔 Warnings: {'ON' if self._safety_speech_enabled else 'OFF'}")
        if self._mute_safety_during_reading:
            print("📖 Reading protection: warnings muted during reading")

        print(f"📚 Current reading mode: {self._get_read_mode()}")

        # ----------------------------
        # Human Analyzer (sci-fi visualization)
        # ----------------------------
        self.human_analyzer = None
        self._human_viz_enabled = True  # Toggle with 'h' key
        
        try:
            self.human_analyzer = create_human_analyzer(
                enable_pose=True,
                enable_animations=True,
                show_skeleton=True,
                show_labels=True,
                confidence_threshold=0.5,
            )
            print("🔬 Human Analyzer enabled (press 'h' to toggle)")
        except Exception as e:
            print(f"⚠️ Human Analyzer not available: {e}")

        # ----------------------------
        # Sign Language Interpreter
        # ----------------------------
        self.sign_interpreter = None
        self._sign_mode_enabled = False
        
        try:
            self.sign_interpreter = create_sign_interpreter(
                mode="continuous",
                speech_callback=self._speak_sign_callback,
                min_detection_confidence=0.7,
                min_tracking_confidence=0.5,
                speak_letters=True,
                speak_words=True,
                enable_visual_feedback=True,
            )
            print("🤟 Sign Language Interpreter enabled (press 'g' to toggle)")
            print(f"   └─ Interpreter object: {type(self.sign_interpreter).__name__}")
        except Exception as e:
            print(f"⚠️ Sign Language Interpreter not available: {e}")
            import traceback
            traceback.print_exc()

        # ----------------------------
        # Telemetry (JSONL) for graphs
        # ----------------------------
        self.telemetry: Optional[TelemetryLogger] = None
        if bool(getattr(config, "TELEMETRY_ENABLED", True)):
            run_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

            # make sure folder exists (avoid missing-dir issues)
            Path("telemetry/runs").mkdir(parents=True, exist_ok=True)

            out_path = Path("telemetry/runs") / f"run_{run_id}.jsonl"
            self.telemetry = TelemetryLogger(out_path)

            # optional: expose as global for other modules later
            set_global_logger(self.telemetry)

            self.telemetry.log_meta(
                {
                    "app": "smart_glasses",
                    "run_id": run_id,
                    "model": getattr(config, "DEFAULT_MODEL_NAME", "unknown"),
                    "process_every_n": int(getattr(config, "PROCESS_EVERY_N_FRAMES", 3) or 3),
                    "obstacle_enabled": bool(getattr(config, "OBSTACLE_ENABLED", False)),
                    "guidance_enabled": bool(getattr(config, "GUIDANCE_ENABLED", False)),
                    "ocr_default_mode": getattr(config, "OCR_MODE", "hybrid"),
                    "sign_language_available": self.sign_interpreter is not None,
                }
            )

            print(f"🧾 Telemetry ON → {out_path}")

    # ----------------------------
    # Sign Language Speech Callback
    # ----------------------------

    def _speak_sign_callback(self, text: str) -> None:
        """Callback for sign language interpreter to speak recognized signs."""
        if not text:
            return
        
        # Use non-blocking speech to avoid blocking the main loop
        def speak_thread():
            try:
                self._speak_blocking(text, meta={"source": "sign_interpreter"})
            except Exception as e:
                print(f"⚠️ Sign speech error: {e}")
        
        threading.Thread(target=speak_thread, daemon=True).start()

    # ----------------------------
    # Sign Language Voice Commands
    # ----------------------------

    def _is_sign_language_command(self, text: str) -> bool:
        """Check if the voice command is related to sign language mode."""
        t = (text or "").strip().lower()
        if not t:
            return False
        keywords = [
            "sign language",
            "sign mode",
            "signing mode",
            "enable sign",
            "disable sign",
            "turn on sign",
            "turn off sign",
            "start sign",
            "stop sign",
            "what did they sign",
            "clear sign buffer",
            "fingerspelling",
            "word signs",
        ]
        return any(k in t for k in keywords)

    def _run_sign_language_command(self, text: str) -> Optional[str]:
        """Execute a sign language related voice command."""
        t = (text or "").strip().lower()
        if not t:
            return None

        if self.sign_interpreter is None:
            return "Sign language interpreter is not available."

        # Enable/disable commands
        if any(k in t for k in ["enable sign", "turn on sign", "start sign"]):
            self._sign_mode_enabled = True
            return "Sign language interpreter enabled."
        
        if any(k in t for k in ["disable sign", "turn off sign", "stop sign"]):
            self._sign_mode_enabled = False
            return "Sign language interpreter disabled."

        # Query commands
        if "what did they sign" in t:
            word = self.sign_interpreter.get_current_word()
            if word:
                return f"The current word being spelled is: {word}"
            return "No signs detected recently."

        if "clear sign buffer" in t or "clear buffer" in t:
            self.sign_interpreter.clear_buffer()
            return "Sign buffer cleared."

        # Mode switching
        if "fingerspelling" in t:
            self.sign_interpreter.set_mode(InterpreterMode.FINGERSPELLING)
            return "Switched to fingerspelling mode."
        
        if "word signs" in t:
            self.sign_interpreter.set_mode(InterpreterMode.WORD_SIGNS)
            return "Switched to word signs mode."

        if "continuous" in t:
            self.sign_interpreter.set_mode(InterpreterMode.CONTINUOUS)
            return "Switched to continuous mode."

        return None

    # ----------------------------
    # Request IDs
    # ----------------------------

    def _next_req_id(self, prefix: str) -> str:
        with self._req_lock:
            self._req_counter += 1
            n = self._req_counter
        return f"{prefix}_{n}"

    # ----------------------------
    # Timing / FPS
    # ----------------------------

    def calculate_fps(self) -> float:
        current_time = time.perf_counter()
        fps = 1.0 / (current_time - self.last_frame_time) if self.last_frame_time else 0.0
        self.last_frame_time = current_time
        self.fps_queue.append(fps)
        return sum(self.fps_queue) / len(self.fps_queue) if self.fps_queue else 0.0

    # ----------------------------
    # Reading state helpers
    # ----------------------------

    def _set_reading(self, val: bool) -> None:
        with self._reading_lock:
            self._is_reading = val

    def _get_reading(self) -> bool:
        with self._reading_lock:
            return self._is_reading

    # ----------------------------
    # Reading mode helpers
    # ----------------------------

    def _set_read_mode(self, mode: str) -> str:
        m = (mode or "").strip().lower()
        if m not in {"local_only", "hybrid", "scene_only"}:
            m = "hybrid"

        with self._read_mode_lock:
            self._read_mode = m
            self._reading_mode = m

        if hasattr(self.ocr, "set_mode"):
            try:
                self.ocr.set_mode(m)
            except Exception:
                pass

        if self.telemetry is not None:
            self.telemetry.log_event("read_mode_set", {"mode": m})

        if m == "local_only":
            return "Reading mode set to offline."
        if m == "scene_only":
            return "Reading mode set to AI."
        return "Reading mode set to hybrid."

    def _get_read_mode(self) -> str:
        with self._read_mode_lock:
            return self._read_mode

    def _cycle_read_mode(self) -> str:
        cur = self._get_read_mode()
        nxt = "local_only" if cur == "scene_only" else ("hybrid" if cur == "local_only" else "scene_only")
        return self._set_read_mode(nxt)

    def _is_read_mode_command(self, text: str) -> bool:
        t = (text or "").strip().lower()
        if not t:
            return False
        keys = [
            "reading mode",
            "read mode",
            "ocr mode",
            "offline reading",
            "read offline",
            "use offline",
            "ai reading",
            "read with ai",
            "use ai",
            "hybrid reading",
            "read hybrid",
            "use hybrid",
        ]
        return any(k in t for k in keys)

    def _run_read_mode_command(self, text: str) -> Optional[str]:
        t = (text or "").strip().lower()
        if not t:
            return None

        if "offline" in t or "local" in t:
            return self._set_read_mode("local_only")

        if "ai" in t or "scene" in t:
            return self._set_read_mode("scene_only")

        if "hybrid" in t:
            return self._set_read_mode("hybrid")

        if "toggle" in t or "cycle" in t:
            return self._cycle_read_mode()

        return None

    # ----------------------------
    # Speech helpers (+ telemetry)
    # ----------------------------

    def _speak_blocking(self, sentence: str, meta: Optional[Dict[str, Any]] = None) -> None:
        sentence = (sentence or "").strip()
        if not sentence:
            return

        meta = dict(meta or {})
        meta.setdefault("text_len", len(sentence))
        meta.setdefault("words", len(sentence.split()))
        source = meta.get("source", "unknown")

        try:
            with self._tts_lock:
                if self.telemetry is not None:
                    rid = meta.get("req_id", "")
                    self.telemetry.log_event("tts_start", {"req_id": rid, **meta})

                t0 = time.perf_counter()
                print(f"🔊 Speaking: {sentence}")
                self.speech.speak(sentence)
                dur_ms = (time.perf_counter() - t0) * 1000.0

                if self.telemetry is not None:
                    rid = meta.get("req_id", "")
                    self.telemetry.log_event("tts_end", {"req_id": rid, "tts_ms": float(dur_ms), **meta})
                
                # Log structured speech telemetry
                log_speech(
                    text_len=len(sentence),
                    duration_ms=dur_ms,
                    source=source,
                    queued=False,
                )

        except Exception as e:
            print(f"⚠️ Speech error: {e!r}")
            log_error(e, context="speak_blocking")
            if self.telemetry is not None:
                rid = meta.get("req_id", "")
                self.telemetry.log_event("tts_error", {"req_id": rid, "err": repr(e), **meta})

    def _speak_if_free(self, sentence: str, meta: Optional[Dict[str, Any]] = None) -> bool:
        sentence = (sentence or "").strip()
        if not sentence:
            return False

        meta = dict(meta or {})
        meta.setdefault("text_len", len(sentence))
        meta.setdefault("words", len(sentence.split()))
        source = meta.get("source", "unknown")

        acquired = self._tts_lock.acquire(blocking=False)
        if not acquired:
            if self.telemetry is not None:
                rid = meta.get("req_id", "")
                self.telemetry.log_event("tts_attempt", {"req_id": rid, "acquired": False, "reason": "tts_lock_busy", **meta})
            return False

        try:
            if self.telemetry is not None:
                rid = meta.get("req_id", "")
                self.telemetry.log_event("tts_attempt", {"req_id": rid, "acquired": True, **meta})
                self.telemetry.log_event("tts_start", {"req_id": rid, **meta})

            t0 = time.perf_counter()
            print(f"🔊 Speaking: {sentence}")
            self.speech.speak(sentence)
            dur_ms = (time.perf_counter() - t0) * 1000.0

            if self.telemetry is not None:
                rid = meta.get("req_id", "")
                self.telemetry.log_event("tts_end", {"req_id": rid, "tts_ms": float(dur_ms), **meta})
            
            # Log structured speech telemetry
            log_speech(
                text_len=len(sentence),
                duration_ms=dur_ms,
                source=source,
                queued=True,
            )

            return True

        except Exception as e:
            print(f"⚠️ Speech error: {e!r}")
            log_error(e, context="speak_if_free")
            if self.telemetry is not None:
                rid = meta.get("req_id", "")
                self.telemetry.log_event("tts_error", {"req_id": rid, "err": repr(e), **meta})
            return False

        finally:
            try:
                self._tts_lock.release()
            except Exception:
                pass

    # ----------------------------
    # Busy locks
    # ----------------------------

    def _try_start_voice(self) -> bool:
        with self._voice_lock:
            if self._voice_busy:
                return False
            self._voice_busy = True
            return True

    def _end_voice(self) -> None:
        with self._voice_lock:
            self._voice_busy = False

    def _try_start_describe(self) -> bool:
        with self._describe_lock:
            if self._describe_busy:
                return False
            self._describe_busy = True
            return True

    def _end_describe(self) -> None:
        with self._describe_lock:
            self._describe_busy = False

    def _try_start_obstacle(self) -> bool:
        with self._obstacle_lock:
            if self._obstacle_busy:
                return False
            self._obstacle_busy = True
            return True

    def _end_obstacle(self) -> None:
        with self._obstacle_lock:
            self._obstacle_busy = False

    # ----------------------------
    # Runtime toggles (voice + key)
    # ----------------------------

    def _toggle_warnings(self, enable: Optional[bool] = None) -> str:
        if enable is None:
            self._safety_speech_enabled = not self._safety_speech_enabled
        else:
            self._safety_speech_enabled = bool(enable)

        if self.telemetry is not None:
            self.telemetry.log_event("warnings_toggle", {"enabled": bool(self._safety_speech_enabled)})

        return f"Warnings are {'on' if self._safety_speech_enabled else 'off'}."

    def _is_toggle_command(self, text: str) -> bool:
        t = (text or "").strip().lower()
        if not t:
            return False
        keys = [
            "warnings off",
            "warning off",
            "mute warnings",
            "mute safety",
            "safety off",
            "warnings on",
            "warning on",
            "unmute warnings",
            "unmute safety",
            "safety on",
            "toggle warnings",
            "toggle safety",
        ]
        return any(k in t for k in keys)

    def _run_toggle_command(self, text: str) -> Optional[str]:
        t = (text or "").strip().lower()
        if not t:
            return None
        if any(k in t for k in ["warnings off", "warning off", "mute warnings", "mute safety", "safety off"]):
            return self._toggle_warnings(enable=False)
        if any(k in t for k in ["warnings on", "warning on", "unmute warnings", "unmute safety", "safety on"]):
            return self._toggle_warnings(enable=True)
        if "toggle" in t:
            return self._toggle_warnings(enable=None)
        return None

    # ----------------------------
    # Manual describe
    # ----------------------------

    def _handle_manual_describe(self, frame_width: int, frame) -> None:
        now = time.time()
        if now - self.last_manual_describe_time < float(
            getattr(config, "MANUAL_DESCRIBE_COOLDOWN_SECONDS", 0.25) or 0.25
        ):
            return

        if not self._try_start_describe():
            return

        req_id = self._next_req_id("describe")
        detections_snapshot = list(self.last_detections) if self.last_detections else []
        frame_snapshot = frame.copy() if frame is not None else None

        def worker():
            t0 = time.perf_counter()
            try:
                if self.telemetry is not None:
                    self.telemetry.log_event("describe_start", {"req_id": req_id, "n_detections": len(detections_snapshot)})

                if frame_snapshot is None:
                    self._speak_blocking("I don't have a clear view right now.", meta={"req_id": req_id, "source": "describe"})
                    return

                local_sentence = summarize_detections(detections_snapshot, frame_width=frame_width)

                try:
                    ai_t0 = time.perf_counter()
                    answer = self.assistant.handle_query(
                        "describe the environment",
                        frame=frame_snapshot,
                        detections=detections_snapshot,
                    )
                    ai_ms = (time.perf_counter() - ai_t0) * 1000.0
                    
                    # Log AI operation
                    log_ai(
                        operation="scene_describe",
                        latency_ms=ai_ms,
                        success=True,
                        result_len=len(answer or ""),
                    )
                except Exception as e:
                    print(f"❌ AssistantBrain describe error: {e!r}")
                    log_error(e, context="describe_scene")
                    answer = local_sentence

                self._speak_blocking(answer, meta={"req_id": req_id, "source": "describe"})

            finally:
                if self.telemetry is not None:
                    total_ms = (time.perf_counter() - t0) * 1000.0
                    self.telemetry.log_event("describe_end", {"req_id": req_id, "total_ms": float(total_ms)})
                self._end_describe()

        threading.Thread(target=worker, daemon=True).start()
        self.last_manual_describe_time = now

    # ----------------------------
    # Document read (keypress)
    # ----------------------------

    def _handle_read_page(self, frame) -> None:
        with self._voice_lock:
            if self._voice_busy:
                return

        frame_snapshot = frame.copy() if frame is not None else None
        if frame_snapshot is None:
            self._speak_blocking("I don't have a clear view right now.", meta={"source": "ocr"})
            return

        mode = self._get_read_mode()
        req_id = self._next_req_id("ocr")

        def worker():
            self._set_reading(True)
            t0 = time.perf_counter()
            try:
                if hasattr(self.ocr, "set_mode"):
                    try:
                        self.ocr.set_mode(mode)
                    except Exception:
                        pass

                msg = self.doc_reader.start(frame_snapshot, mode=mode)

                if getattr(config, "OCR_DEBUG_PRINT", False):
                    msg = f"[{mode}] {msg}"

                dur_ms = (time.perf_counter() - t0) * 1000.0
                words = len((msg or "").split())
                chars = len(msg or "")
                success = bool(chars > 0 and "don't have a clear view" not in (msg or "").lower())

                if self.telemetry is not None:
                    self.telemetry.log_event(
                        "ocr_read",
                        {
                            "req_id": req_id,
                            "mode": mode,
                            "latency_ms": float(dur_ms),
                            "words": int(words),
                            "chars": int(chars),
                            "success": success,
                        },
                    )
                
                # Log structured AI telemetry for OCR
                log_ai(
                    operation="ocr_read",
                    latency_ms=dur_ms,
                    success=success,
                    mode=mode,
                    result_len=chars,
                )

                self._speak_blocking(msg, meta={"req_id": req_id, "source": "ocr", "mode": mode})

            except Exception as e:
                log_error(e, context="handle_read_page")
            finally:
                self._set_reading(False)

        threading.Thread(target=worker, daemon=True).start()

    # ----------------------------
    # Voice interaction
    # ----------------------------

    def _is_doc_command(self, text: str) -> bool:
        t = (text or "").strip().lower()
        if not t:
            return False
        keywords = [
            "read this",
            "read this page",
            "read the page",
            "read document",
            "document mode",
            "next",
            "next paragraph",
            "continue",
            "keep going",
            "repeat",
            "say that again",
            "summarize",
            "summary",
            "summarize this page",
            "stop reading",
            "exit document",
        ]
        return any(k in t for k in keywords) or t in {"next", "repeat", "summarize"}

    def _run_doc_command(self, text: str, frame_snapshot) -> Optional[str]:
        t = (text or "").strip().lower()
        if not t:
            return None

        if "stop reading" in t or "exit document" in t:
            self.doc_reader.state = None
            return "Okay. I stopped reading."

        if "read" in t or "document mode" in t:
            return self.doc_reader.start(frame_snapshot, mode=self._get_read_mode())

        if t in {"next", "next paragraph", "continue", "keep going"}:
            return self.doc_reader.next()

        if t in {"repeat", "say that again"}:
            return self.doc_reader.repeat()

        if "summarize" in t or "summary" in t:
            if self.doc_reader.state is None:
                _ = self.doc_reader.start(frame_snapshot, mode=self._get_read_mode())
            return self.doc_reader.summarize()

        return None

    def save_scene_memory(self, frame: np.ndarray) -> None:
        """Save current scene to memory - simplified version"""
        try:
            if frame is None:
                self._speak_blocking("No scene to save.")
                return
            
            # Build description from detected objects
            # Detection format: {'label': 'Human face', 'conf': 0.86, 'bbox': [...]}
            objects = []
            detections = []
            
            if self.last_detections:
                for det in self.last_detections:
                    # Handle both 'label' and 'name' keys
                    obj_name = det.get('label') or det.get('name', 'unknown')
                    objects.append(obj_name)
                    detections.append({
                        'name': obj_name,
                        'conf': det.get('conf', 0)
                    })
            
            if not objects:
                description = "Empty scene with no detected objects"
            else:
                description = f"Scene with: {', '.join(objects[:5])}"
            
            # Save to memory
            self.scene_memory.store_scene(
                description=description,
                detections=detections,
                location="current_location",
                importance=1.0
            )
            
            self._speak_blocking(f"Scene saved. I see {len(objects)} objects.")
            print(f"💾 Saved scene to memory")
            print(f"   Objects: {', '.join(objects[:5])}")
            print(f"   Total memories: {len(self.scene_memory.memories)}")
            
        except Exception as e:
            print(f"❌ Error saving scene: {e}")
            log_error(e, context="save_scene_memory")
            import traceback
            traceback.print_exc()
            self._speak_blocking("Sorry, couldn't save the scene.")

    
    def _handle_voice_interaction(self, frame) -> None:
        if not self._try_start_voice():
            return

        req_id = self._next_req_id("voice")
        detections_snapshot = list(self.last_detections) if self.last_detections else []
        frame_snapshot = frame.copy() if frame is not None else None

        def worker():
            t_all0 = time.perf_counter()
            command_type = "unknown"
            try:
                if self.telemetry is not None:
                    self.telemetry.log_event("voice_start", {"req_id": req_id})
                
                # Log voice start
                log_voice(req_id=req_id, phase="start")

                # listen/transcribe timing
                t0 = time.perf_counter()
                text = self.voice_listener.listen_and_transcribe()
                listen_ms = (time.perf_counter() - t0) * 1000.0

                if self.telemetry is not None:
                    self.telemetry.log_event(
                        "voice_transcribed",
                        {"req_id": req_id, "listen_ms": float(listen_ms), "text_len": len(text or "")},
                    )
                
                # Log voice transcription
                log_voice(
                    req_id=req_id,
                    phase="transcribed",
                    listen_ms=listen_ms,
                    text_len=len(text or ""),
                    success=bool(text),
                )

                if not text:
                    self._speak_blocking("I didn't catch that. Try again.", meta={"req_id": req_id, "source": "voice"})
                    return

                # 1) Sign language commands
                if self._is_sign_language_command(text):
                    msg = self._run_sign_language_command(text)
                    if msg:
                        command_type = "sign_language"
                        if self.telemetry is not None:
                            self.telemetry.log_event("voice_command", {"req_id": req_id, "cmd": "sign_language"})
                        self._speak_blocking(msg, meta={"req_id": req_id, "source": "voice"})
                        return

                # 2) Toggle warnings
                if self._is_toggle_command(text):
                    msg = self._run_toggle_command(text)
                    if msg:
                        command_type = "toggle_warnings"
                        if self.telemetry is not None:
                            self.telemetry.log_event("voice_command", {"req_id": req_id, "cmd": "toggle_warnings"})
                        self._speak_blocking(msg, meta={"req_id": req_id, "source": "voice"})
                        return

                # 3) Reading mode command
                if self._is_read_mode_command(text):
                    msg = self._run_read_mode_command(text)
                    if msg:
                        command_type = "read_mode"
                        if self.telemetry is not None:
                            self.telemetry.log_event("voice_command", {"req_id": req_id, "cmd": "read_mode"})
                        self._speak_blocking(msg, meta={"req_id": req_id, "source": "voice"})
                        return

                # 4) Document commands (protected reading mode)
                if frame_snapshot is not None and self._is_doc_command(text):
                    command_type = "document"
                    self._set_reading(True)
                    try:
                        if hasattr(self.ocr, "set_mode"):
                            try:
                                self.ocr.set_mode(self._get_read_mode())
                            except Exception:
                                pass

                        tdoc0 = time.perf_counter()
                        msg = self._run_doc_command(text, frame_snapshot)
                        doc_ms = (time.perf_counter() - tdoc0) * 1000.0

                        if msg:
                            words = len((msg or "").split())
                            chars = len(msg or "")

                            if self.telemetry is not None:
                                self.telemetry.log_event(
                                    "ocr_doc_cmd",
                                    {
                                        "req_id": req_id,
                                        "mode": self._get_read_mode(),
                                        "latency_ms": float(doc_ms),
                                        "words": int(words),
                                        "chars": int(chars),
                                        "cmd_text": (text or "")[:120],
                                    },
                                )
                            
                            # Log AI operation for OCR
                            log_ai(
                                operation="ocr_doc_cmd",
                                latency_ms=doc_ms,
                                success=True,
                                mode=self._get_read_mode(),
                                result_len=chars,
                            )

                            if getattr(config, "OCR_DEBUG_PRINT", False):
                                msg = f"[{self._get_read_mode()}] {msg}"

                            self._speak_blocking(msg, meta={"req_id": req_id, "source": "ocr", "mode": self._get_read_mode()})
                            return
                    finally:
                        self._set_reading(False)

                # 5) Otherwise route to assistant
                command_type = "brain"
                tbrain0 = time.perf_counter()
                try:
                    answer = self.assistant.handle_query(
                        text,
                        frame=frame_snapshot,
                        detections=detections_snapshot,
                    )
                    brain_ms = (time.perf_counter() - tbrain0) * 1000.0
                    
                    # Log AI operation
                    log_ai(
                        operation="assistant_query",
                        latency_ms=brain_ms,
                        success=True,
                        result_len=len(answer or ""),
                    )
                except Exception as e:
                    print(f"❌ AssistantBrain.handle_query error: {e!r}")
                    log_error(e, context="assistant_handle_query")
                    answer = "Something went wrong while answering."
                    brain_ms = (time.perf_counter() - tbrain0) * 1000.0

                if self.telemetry is not None:
                    self.telemetry.log_event(
                        "voice_brain",
                        {
                            "req_id": req_id,
                            "brain_ms": float(brain_ms),
                            "query_preview": (text or "")[:120],
                        },
                    )

                self._speak_blocking(answer, meta={"req_id": req_id, "source": "voice"})

            except Exception as e:
                log_error(e, context="voice_interaction")
            finally:
                total_ms = (time.perf_counter() - t_all0) * 1000.0
                
                if self.telemetry is not None:
                    self.telemetry.log_event("voice_end", {"req_id": req_id, "total_ms": float(total_ms)})
                
                # Log voice completion
                log_voice(
                    req_id=req_id,
                    phase="complete",
                    total_ms=total_ms,
                    command_type=command_type,
                    success=True,
                )
                
                self._end_voice()

        threading.Thread(target=worker, daemon=True).start()

    # ----------------------------
    # Optional Auto Speak
    # ----------------------------

    def _maybe_auto_speak(self, frame_width: int) -> None:
        n = int(getattr(config, "SPEAK_EVERY_N_FRAMES", 0) or 0)
        if n <= 0:
            return

        now = time.time()
        if now - self.last_auto_speak_time < float(getattr(config, "MIN_SPEECH_INTERVAL_SECONDS", 3.0) or 3.0):
            return

        with self._voice_lock:
            if self._voice_busy:
                return

        if self._get_reading():
            return

        dets = list(self.last_detections) if self.last_detections else []
        if not dets:
            return

        sentence = summarize_detections(dets, frame_width=frame_width)
        if sentence:
            ok = self._speak_if_free(sentence, meta={"source": "auto_speak"})
            if ok:
                self.last_auto_speak_time = now
            else:
                if self.telemetry is not None:
                    self.telemetry.log_event("auto_speak_skipped", {"reason": "tts_busy"})

    # ----------------------------
    # Obstacle alerts
    # ----------------------------

    def _maybe_obstacle_alert(self, frame_idx: int, frame, detections: List[Dict[str, Any]]) -> None:
        if not self._safety_speech_enabled:
            return

        if self._mute_safety_during_reading and self._get_reading():
            return

        if not getattr(config, "OBSTACLE_ENABLED", False):
            return

        every = int(getattr(config, "OBSTACLE_EVERY_N_FRAMES", 3) or 3)
        if every <= 0:
            every = 3
        if frame_idx % every != 0:
            return

        with self._voice_lock:
            if self._voice_busy:
                return

        if self._tts_lock.locked():
            return

        if not self._try_start_obstacle():
            return

        req_id = self._next_req_id("obstacle")

        frame_snapshot = frame.copy() if frame is not None else None
        dets_snapshot = list(detections) if detections else []
        mode = (getattr(config, "OBSTACLE_MODE", "bbox") or "bbox").strip().lower()

        def worker():
            t0 = time.perf_counter()
            try:
                if frame_snapshot is None:
                    return

                depth_map = None
                depth_quality = 0.0
                if mode == "depth" and getattr(config, "DEPTH_ENABLED", False):
                    try:
                        depth_map = self.depth_estimator.estimate(frame_snapshot, frame_idx=frame_idx)
                        depth_quality = 0.8  # Assume good quality if successful
                    except Exception as e:
                        print(f"⚠️ DepthEstimator error: {e!r}")
                        log_error(e, context="depth_estimator")
                        depth_map = None

                msg = self.obstacle_layer.update(
                    frame_snapshot,
                    dets_snapshot,
                    depth_map=depth_map,
                    now=time.time(),
                )
                if not msg:
                    return

                dur_ms = (time.perf_counter() - t0) * 1000.0

                if self.telemetry is not None:
                    self.telemetry.log_event(
                        "obstacle_alert",
                        {
                            "req_id": req_id,
                            "latency_ms": float(dur_ms),
                            "mode": mode,
                            "msg_preview": (msg or "")[:120],
                        },
                    )
                
                # Log structured safety telemetry
                log_safety(
                    event_type="obstacle",
                    severity=2 if "very close" in msg.lower() else (1 if "close" in msg.lower() else 0),
                    direction=_extract_direction_from_msg(msg),
                    distance=_extract_distance_from_msg(msg),
                    message=msg,
                    depth_quality=depth_quality if mode == "depth" else None,
                )

                with self._voice_lock:
                    if self._voice_busy:
                        return
                if self._mute_safety_during_reading and self._get_reading():
                    return

                self._speak_if_free(msg, meta={"req_id": req_id, "source": "obstacle", "mode": mode})

            except Exception as e:
                log_error(e, context="obstacle_alert")
            finally:
                self._end_obstacle()

        threading.Thread(target=worker, daemon=True).start()

    # ----------------------------
    # Guidance mode
    # ----------------------------

    def _maybe_guidance(self, frame_idx: int, detections: List[Dict[str, Any]], frame_w: int, frame_h: int) -> None:
        if not self._safety_speech_enabled:
            return
        if self._mute_safety_during_reading and self._get_reading():
            return

        if not getattr(config, "GUIDANCE_ENABLED", False):
            return

        every = int(getattr(config, "GUIDANCE_EVERY_N_FRAMES", 6) or 6)
        if every <= 0:
            every = 6
        if frame_idx % every != 0:
            return

        if getattr(config, "GUIDANCE_SUPPRESS_DURING_VOICE", True):
            with self._voice_lock:
                if self._voice_busy:
                    return

        if self._tts_lock.locked():
            return

        t0 = time.perf_counter()
        msg = self.guidance.update(
            detections=detections or [],
            frame_w=frame_w,
            frame_h=frame_h,
            profile=getattr(config, "GUIDANCE_PROFILE", "indoor"),
            now=time.time(),
        )
        dur_ms = (time.perf_counter() - t0) * 1000.0

        if msg:
            gtype = _simple_guidance_type(msg)

            if self.telemetry is not None:
                self.telemetry.log_event(
                    "guidance_msg",
                    {
                        "latency_ms": float(dur_ms),
                        "type": gtype,
                        "msg_preview": (msg or "")[:120],
                    },
                )
            
            # Log structured safety telemetry for guidance
            log_safety(
                event_type="guidance",
                severity=2 if "very close" in msg.lower() else (1 if "close" in msg.lower() else 0),
                direction=_extract_direction_from_msg(msg),
                distance=_extract_distance_from_msg(msg),
                message=msg,
            )

            if self._mute_safety_during_reading and self._get_reading():
                return

            self._speak_if_free(msg, meta={"source": "guidance", "type": gtype})

    # ----------------------------
    # Main loop
    # ----------------------------

    def run(self):
        print("🚀 Smart Glasses System Starting...")
        print("Press 'q' quit | 'd' describe | 'v' voice | 'r' read | 's' toggle warnings")
        print("Reading mode keys: '1' offline | '2' hybrid | '3' AI-only | 'm' cycle mode")
        print("Sign language: 'g' toggle | Human viz: 'h' toggle | 'f' toggle fullscreen")

        frame_idx = 0
        
        # Window setup - create named window with fullscreen support
        window_name = "Smart Glasses - AI Assistant"
        cv.namedWindow(window_name, cv.WINDOW_NORMAL)
        
        # Check config for fullscreen preference, default to True
        self._fullscreen = getattr(config, "FULLSCREEN_WINDOW", True)
        if self._fullscreen:
            cv.setWindowProperty(window_name, cv.WND_PROP_FULLSCREEN, cv.WINDOW_FULLSCREEN)
        else:
            # Set a larger default window size
            cv.resizeWindow(window_name, 1280, 720)
        
        # Mirror mode - flip horizontally so it feels like a mirror
        self._mirror_mode = getattr(config, "MIRROR_MODE", False)

        # how many detections to store per frame in telemetry
        det_log_max = int(getattr(config, "TELEM_MAX_DETS_LOG", 20) or 20)
        
        # System health logging interval
        system_health_interval = int(getattr(config, "TELEM_SYSTEM_HEALTH_INTERVAL", 100) or 100)

        try:
            while True:
                loop_t0 = time.perf_counter()

                # Capture timing (accurate)
                cap_t0 = time.perf_counter()
                frame = self.camera.capture_frame()
                capture_ms = (time.perf_counter() - cap_t0) * 1000.0

                if frame is None:
                    print("⚠️ No frame from camera, exiting.")
                    if self.telemetry is not None:
                        self.telemetry.log_event("capture_fail", {"frame_idx": frame_idx})
                    break

                frame_height, frame_width = frame.shape[:2]
                fps = self.calculate_fps()

                did_detect = False
                detect_ms = 0.0
                scene_ctx_ms = 0.0

                if frame_idx % int(getattr(config, "PROCESS_EVERY_N_FRAMES", 3) or 3) == 0:
                    did_detect = True

                    det_t0 = time.perf_counter()
                    detections, annotated_frame = self.detector.detect(frame, annotate=True)
                    detect_ms = (time.perf_counter() - det_t0) * 1000.0

                    # Apply human analyzer overlay (sci-fi visualization)
                    if (self.human_analyzer is not None and 
                        self._human_viz_enabled and 
                        detections and
                        not self._sign_mode_enabled):  # Don't overlap with sign mode
                        try:
                            # Check if any humans detected
                            human_labels = {'person', 'man', 'woman', 'human face', 'boy', 'girl', 'human body'}
                            has_humans = any(
                                (d.get('label', '') or '').lower() in human_labels
                                for d in detections
                            )
                            if has_humans:
                                _, annotated_frame = self.human_analyzer.analyze_humans(
                                    annotated_frame, detections
                                )
                        except Exception as e:
                            pass  # Don't crash on visualization errors

                    # Process sign language if enabled
                    if self._sign_mode_enabled and self.sign_interpreter is not None:
                        try:
                            signs, annotated_frame = self.sign_interpreter.process_frame(
                                annotated_frame, 
                                detections
                            )
                            if signs:
                                # Only process high-confidence signs
                                best_sign = max(signs, key=lambda s: s.confidence)
                                if best_sign.confidence >= 0.70:
                                    # Print to console for debugging
                                    print(f"🤟 Sign detected: {best_sign.sign} ({best_sign.confidence:.0%})")
                        except Exception as e:
                            # Only print unique errors (not every frame)
                            pass

                    self.last_detections = detections
                    self.last_annotated = annotated_frame

                    try:
                        ctx_t0 = time.perf_counter()
                        self.assistant.update_scene_context(frame=frame, detections=detections)
                        scene_ctx_ms = (time.perf_counter() - ctx_t0) * 1000.0
                    except Exception as e:
                        print(f"⚠️ Error updating scene context: {e!r}")
                        log_error(e, context="update_scene_context")
                else:
                    detections = self.last_detections
                    annotated_frame = self.last_annotated if self.last_annotated is not None else frame

                self._maybe_obstacle_alert(frame_idx, frame, detections)
                self._maybe_guidance(frame_idx, detections, frame_width, frame_height)
                self._maybe_auto_speak(frame_width)

                if getattr(config, "SHOW_FPS", True):
                    cv.putText(
                        annotated_frame,
                        f"FPS: {fps:.1f}",
                        (10, 30),
                        cv.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 255, 0),
                        2,
                    )

                if getattr(config, "SHOW_DETECTION_COUNT", True):
                    cv.putText(
                        annotated_frame,
                        f"Objects: {len(detections)}",
                        (10, 60),
                        cv.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 255, 0),
                        2,
                    )

                mode = self._get_read_mode()
                cv.putText(
                    annotated_frame,
                    f"ReadMode: {mode}",
                    (10, 90),
                    cv.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2,
                )

                # Show sign language mode status
                if self._sign_mode_enabled:
                    cv.putText(
                        annotated_frame,
                        "SIGN MODE: ON",
                        (10, 120),
                        cv.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 255, 255),
                        2,
                    )

                if getattr(config, "SHOW_DEBUG_WINDOW", True):
                    # Apply mirror if enabled
                    if self._mirror_mode:
                        display_frame = cv.flip(annotated_frame, 1)
                    else:
                        display_frame = annotated_frame
                    cv.imshow(window_name, display_frame)

                key = cv.waitKey(1) & 0xFF
                if key == ord("q"):
                    print("👋 Exiting.")
                    break
                elif key == ord("f"):
                    # Toggle fullscreen
                    self._fullscreen = not self._fullscreen
                    if self._fullscreen:
                        cv.setWindowProperty(window_name, cv.WND_PROP_FULLSCREEN, cv.WINDOW_FULLSCREEN)
                        print("🖥️ Fullscreen: ON")
                    else:
                        cv.setWindowProperty(window_name, cv.WND_PROP_FULLSCREEN, cv.WINDOW_NORMAL)
                        cv.resizeWindow(window_name, 1280, 720)
                        print("🖥️ Fullscreen: OFF")
                elif key == ord("p"):
                    # Toggle mirror mode
                    self._mirror_mode = not self._mirror_mode
                    state = "ON" if self._mirror_mode else "OFF"
                    print(f"🪞 Mirror mode: {state}")
                elif key == ord("d"):
                    self._handle_manual_describe(frame_width, frame)
                elif key == ord("v"):
                    self._handle_voice_interaction(frame)
                elif key == ord("r"):
                    self._handle_read_page(frame)
                elif key == ord("s"):
                    msg = self._toggle_warnings(enable=None)
                    print(f"🔔 {msg}")
                    self._speak_blocking(msg, meta={"source": "system"})

                # Reading mode hotkeys
                elif key == ord("1"):
                    msg = self._set_read_mode("local_only")
                    print(f"📚 {msg}")
                    self._speak_blocking(msg, meta={"source": "system"})
                elif key == ord("2"):
                    msg = self._set_read_mode("hybrid")
                    print(f"📚 {msg}")
                    self._speak_blocking(msg, meta={"source": "system"})
                elif key == ord("3"):
                    msg = self._set_read_mode("scene_only")
                    print(f"📚 {msg}")
                    self._speak_blocking(msg, meta={"source": "system"})
                elif key == ord("m"):
                    msg = self._cycle_read_mode()
                    print(f"📚 {msg}")
                    self._speak_blocking(msg, meta={"source": "system"})
                elif key == ord("x"):
                    # Save scene to memory
                    print("💾 Saving scene to memory...")
                    self.save_scene_memory(frame)
                elif key == ord("z"):
                    # Check memory stats
                    total = len(self.scene_memory.memories)
                    print(f"\n📊 Memory Stats: {total} memories stored")
                    if total > 0:
                        print("   Recent memories:")
                        for mem in list(self.scene_memory.memories)[-3:]:
                            print(f"   - {mem.description}")
                    else:
                        print("   No memories yet - press 'x' to save scenes")
                elif key == ord("h"):
                    # Toggle human analyzer visualization
                    self._human_viz_enabled = not self._human_viz_enabled
                    state = "ON" if self._human_viz_enabled else "OFF"
                    print(f"🔬 Human visualization: {state}")
                    self._speak_blocking(f"Human analysis visualization {state.lower()}.", meta={"source": "system"})
                elif key == ord("g"):
                    # Toggle sign language interpreter
                    if self.sign_interpreter is not None:
                        self._sign_mode_enabled = not self._sign_mode_enabled
                        state = "ON" if self._sign_mode_enabled else "OFF"
                        print(f"🤟 Sign language mode: {state}")
                        self._speak_blocking(f"Sign language interpreter {state.lower()}.", meta={"source": "system"})
                        
                        # Log the toggle
                        if self.telemetry is not None:
                            self.telemetry.log_event("sign_mode_toggle", {"enabled": self._sign_mode_enabled})
                    else:
                        print("⚠️ Sign language interpreter not available")
                        self._speak_blocking("Sign language interpreter not available.", meta={"source": "system"})
                elif key == ord("c"):
                    # Clear sign language buffer (when in sign mode)
                    if self._sign_mode_enabled and self.sign_interpreter is not None:
                        self.sign_interpreter.clear_buffer()
                        print("📝 Sign buffer cleared")
                        self._speak_blocking("Sign buffer cleared.", meta={"source": "system"})

                # --- Frame telemetry ---
                loop_total_ms = (time.perf_counter() - loop_t0) * 1000.0

                # compact list of detections for label/conf charts
                dets_compact: List[Dict[str, Any]] = []
                if detections:
                    # sort by confidence descending and log up to det_log_max
                    det_sorted = sorted(detections, key=lambda d: float(d.get("confidence", 0.0)), reverse=True)
                    for d in det_sorted[:det_log_max]:
                        lbl = str(d.get("label", ""))
                        conf = float(d.get("confidence", 0.0))
                        dets_compact.append({"label": lbl, "confidence": conf})

                top_label = ""
                top_conf = 0.0
                if dets_compact:
                    top_label = str(dets_compact[0].get("label", ""))
                    top_conf = float(dets_compact[0].get("confidence", 0.0))

                if self.telemetry is not None:
                    self.telemetry.log_frame(
                        {
                            "frame_idx": frame_idx,
                            "fps": float(fps),
                            "capture_ms": float(capture_ms),
                            "detect_ms": float(detect_ms),
                            "scene_ctx_ms": float(scene_ctx_ms),
                            "loop_total_ms": float(loop_total_ms),
                            "did_detect": bool(did_detect),
                            "n_detections": int(len(detections) if detections else 0),
                            "top_label": top_label,
                            "top_conf": float(top_conf),
                            "read_mode": self._get_read_mode(),
                            "is_reading": bool(self._get_reading()),
                            "safety_enabled": bool(self._safety_speech_enabled),
                            "tts_locked": bool(self._tts_lock.locked()),
                            "sign_mode_enabled": bool(self._sign_mode_enabled),
                            "dets_compact": dets_compact,
                        }
                    )

                # Log system health periodically
                if system_health_interval > 0 and frame_idx % system_health_interval == 0:
                    log_system_health()

                frame_idx += 1

                debug_every = int(getattr(config, "DEBUG_PRINT_EVERY_N_FRAMES", 90) or 90)
                if debug_every > 0 and frame_idx % debug_every == 0 and detections:
                    print(f"\n--- Frame {frame_idx} | FPS: {fps:.1f} | Objects: {len(detections)} ---")
                    for d in (detections or [])[:8]:
                        print(f"  {str(d.get('label', 'object')):20s} conf={float(d.get('confidence', 0)):.2f}")

        except KeyboardInterrupt:
            print("🛑 Interrupted.")
        except Exception as e:
            log_error(e, context="main_loop")
            raise
        finally:
            print("\n🧹 Cleaning up...")
            
            # Step 1: Release camera FIRST (stops new frames)
            try:
                if hasattr(self, 'camera') and self.camera is not None:
                    if hasattr(self.camera, 'cap') and self.camera.cap is not None:
                        self.camera.cap.release()
                        print("   Camera released.")
            except Exception as e:
                print(f"   Camera release error: {e}")
            
            # Step 2: Process pending window events and destroy windows
            # On macOS, we need multiple waitKey calls to flush the event queue
            try:
                for _ in range(5):
                    cv.waitKey(1)
                cv.destroyAllWindows()
                for _ in range(5):
                    cv.waitKey(1)
                print("   Windows closed.")
            except Exception as e:
                print(f"   Window cleanup error: {e}")

            # Step 3: Close telemetry safely
            try:
                if self.telemetry is not None:
                    self.telemetry.close()
                    print("   Telemetry closed.")
            except Exception:
                pass

            # Step 4: Clean up MediaPipe resources in human analyzer
            try:
                if hasattr(self, 'human_analyzer') and self.human_analyzer is not None:
                    del self.human_analyzer
                    self.human_analyzer = None
            except Exception:
                pass
            
            # Step 5: Clean up sign interpreter
            try:
                if hasattr(self, 'sign_interpreter') and self.sign_interpreter is not None:
                    del self.sign_interpreter
                    self.sign_interpreter = None
            except Exception:
                pass

            # Step 6: Remove global logger
            try:
                set_global_logger(None)
            except Exception:
                pass

            avg_fps = sum(self.fps_queue) / len(self.fps_queue) if self.fps_queue else 0.0
            print("\n📊 Final Stats:")
            print(f"   Average FPS: {avg_fps:.1f}")
            print(f"   Total frames: {frame_idx}")
            print("✅ Cleanup complete.")


if __name__ == "__main__":
    controller = MainController()
    controller.run()