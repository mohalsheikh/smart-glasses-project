#!/usr/bin/env python3
"""
VisionAssist Competition Demo Mode
====================================
Streamlined demo launcher for the CBU Business Competition.

Input Methods:
  1. ESP32 Button: tap=voice, double-tap=describe, hold=read
  2. Keyboard: d/v/r/c/p/f/t/w/q (for debugging)
"""

from __future__ import annotations

import sys
import os
import argparse
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

os.environ.setdefault("SHOW_DEBUG_WINDOW", "1")
os.environ.setdefault("PROCESS_EVERY_N_FRAMES", "2")
os.environ.setdefault("OBSTACLE_ENABLED", "0")
os.environ.setdefault("GUIDANCE_ENABLED", "0")
os.environ.setdefault("TELEMETRY_ENABLED", "0")

import cv2 as cv
import numpy as np

from src.camera_handler import CameraHandler
from src.object_detector import ObjectDetector
from src.speech_engine import SpeechEngine
from src.scene_ai_client import SceneAIClient
from src.currency_recognizer import CurrencyRecognizer
from src.ocr_engine import OCREngine
from src.document_reader import DocumentReader
from src.weather_client import WeatherClient
from src.navigation_client import NavigationClient
from src.assistant_brain import AssistantBrain

try:
    from src.voice.advanced_voice_listener import VoiceListener
except Exception:
    VoiceListener = None

try:
    from esp32_listener import ESP32ButtonListener
except ImportError:
    ESP32ButtonListener = None

import src.utils.config as config
from collections import deque
from typing import List, Dict, Any, Optional
import threading


# ============================================================================
# DEMO OVERLAY
# ============================================================================
class DemoOverlay:
    """Renders a professional HUD overlay on the demo feed."""

    def __init__(self):
        self.show_help = False
        self.last_status = "Ready"
        self.last_status_color = (0, 255, 0)
        self.auto_narrate = False
        self.fps_queue = deque(maxlen=30)
        self.last_frame_time = time.perf_counter()

    def update_fps(self) -> float:
        now = time.perf_counter()
        fps = 1.0 / (now - self.last_frame_time) if self.last_frame_time else 0.0
        self.last_frame_time = now
        self.fps_queue.append(fps)
        return sum(self.fps_queue) / len(self.fps_queue) if self.fps_queue else 0.0

    def set_status(self, text: str, color=(0, 255, 0)):
        self.last_status = text
        self.last_status_color = color

    def render(self, frame: np.ndarray, detections: List[Dict], fps: float) -> np.ndarray:
        h, w = frame.shape[:2]

        overlay = frame.copy()
        cv.rectangle(overlay, (0, 0), (w, 50), (20, 20, 20), -1)
        frame = cv.addWeighted(overlay, 0.7, frame, 0.3, 0)

        cv.putText(frame, "VisionAssist", (10, 35),
                    cv.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        cv.putText(frame, "AI Smart Glasses", (220, 35),
                    cv.FONT_HERSHEY_SIMPLEX, 0.6, (150, 150, 150), 1)
        cv.putText(frame, f"FPS: {fps:.1f}", (w - 130, 35),
                    cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)

        n_objects = len(detections) if detections else 0
        cv.putText(frame, f"Objects: {n_objects}", (w - 280, 35),
                    cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)

        overlay2 = frame.copy()
        cv.rectangle(overlay2, (0, h - 40), (w, h), (20, 20, 20), -1)
        frame = cv.addWeighted(overlay2, 0.7, frame, 0.3, 0)

        cv.putText(frame, f"Status: {self.last_status}", (10, h - 12),
                    cv.FONT_HERSHEY_SIMPLEX, 0.55, self.last_status_color, 1)

        if self.auto_narrate:
            cv.putText(frame, "AUTO", (w - 70, h - 12),
                        cv.FONT_HERSHEY_SIMPLEX, 0.55, (0, 200, 255), 2)

        if self.show_help:
            frame = self._render_help(frame)

        return frame

    def _render_help(self, frame: np.ndarray) -> np.ndarray:
        h, w = frame.shape[:2]
        overlay = frame.copy()
        cv.rectangle(overlay, (w // 4, 60), (3 * w // 4, h - 50), (0, 0, 0), -1)
        frame = cv.addWeighted(overlay, 0.85, frame, 0.15, 0)

        x0 = w // 4 + 20
        y0 = 95
        dy = 30

        commands = [
            ("VisionAssist Controls", (255, 255, 255)),
            ("", (0, 0, 0)),
            ("Button: tap=voice  2x=describe  hold=read", (255, 200, 100)),
            ("", (0, 0, 0)),
            ("D  - Describe scene", (0, 255, 0)),
            ("V  - Voice command", (100, 255, 255)),
            ("R  - Read text", (255, 100, 255)),
            ("C  - Currency check", (0, 200, 255)),
            ("P  - Describe people", (255, 200, 0)),
            ("F  - Find object", (255, 255, 100)),
            ("T  - Current time", (200, 200, 200)),
            ("W  - Weather info", (200, 200, 200)),
            ("", (0, 0, 0)),
            ("SPACE - Toggle auto-narrate", (0, 200, 255)),
            ("H  - Toggle this help", (150, 150, 150)),
            ("Q  - Quit", (100, 100, 255)),
        ]

        for i, (text, color) in enumerate(commands):
            if text:
                scale = 0.7 if i == 0 else 0.55
                thick = 2 if i == 0 else 1
                cv.putText(frame, text, (x0, y0 + i * dy),
                            cv.FONT_HERSHEY_SIMPLEX, scale, color, thick)

        return frame


# ============================================================================
# DEMO CONTROLLER
# ============================================================================
class DemoController:
    """Streamlined controller for competition demos."""

    def __init__(self, webcam_index: int = 0, enable_voice: bool = True):
        print("=" * 60)
        print("  VisionAssist - Competition Demo Mode")
        print("  AI-Powered Smart Glasses for the Visually Impaired")
        print("=" * 60)
        print()

        self.camera = CameraHandler(camera_index=webcam_index)
        self.detector = ObjectDetector()
        self.speech = SpeechEngine()
        self.scene_ai = SceneAIClient()
        self.currency = CurrencyRecognizer()
        self.ocr = OCREngine()
        self.doc_reader = DocumentReader(self.ocr)
        self.weather_client = WeatherClient()
        self.navigation_client = NavigationClient()
        self.assistant = AssistantBrain(
            scene_ai=self.scene_ai,
            weather_client=self.weather_client,
            navigation_client=self.navigation_client,
        )

        self.voice_listener = None
        if enable_voice and VoiceListener:
            try:
                self.voice_listener = VoiceListener()
                print("🎤 Voice commands: ENABLED")
            except Exception as e:
                print(f"⚠️ Voice not available: {e}")

        self.overlay = DemoOverlay()
        self.last_detections: List[Dict[str, Any]] = []
        self.last_annotated = None
        self.last_frame = None
        self._tts_lock = threading.Lock()
        self._busy = False
        self.last_auto_narrate_time = 0.0

        # ESP32 button controller
        self.esp32 = None
        if ESP32ButtonListener:
            try:
                self.esp32 = ESP32ButtonListener(
                    on_voice=lambda: self._do_voice(self.last_frame) if self.last_frame is not None else None,
                    on_describe=lambda: self._do_describe(self.last_frame) if self.last_frame is not None else None,
                    on_read=lambda: self._do_read(self.last_frame) if self.last_frame is not None else None,
                )
                self.esp32.start()
                print("🎮 ESP32 button controller: CONNECTED")
            except Exception as e:
                print(f"⚠️ ESP32 listener not available: {e}")

        print()
        print("🚀 Demo ready! Press the button to get started.")
        print()

    def _speak(self, text: str):
        """Thread-safe speech."""
        text = (text or "").strip()
        if not text:
            return
        print(f"🔊 {text}")
        self.speech.speak(text)

    def _async_task(self, fn, status_text: str, status_color=(255, 200, 0)):
        """Run a function in a background thread with status updates."""
        if self._busy:
            return
        self._busy = True
        self.overlay.set_status(status_text, status_color)

        def worker():
            try:
                fn()
            except Exception as e:
                print(f"❌ Error: {e}")
                self._speak("Sorry, something went wrong.")
            finally:
                self._busy = False
                self.overlay.set_status("Ready", (0, 255, 0))

        threading.Thread(target=worker, daemon=True).start()

    # ------------------------------------------------------------------
    # Demo Actions
    # ------------------------------------------------------------------
    def _do_describe(self, frame):
        """AI scene description."""
        def task():
            answer = self.assistant.handle_query(
                "describe the environment",
                frame=frame,
                detections=self.last_detections,
            )
            self._speak(answer)
        self._async_task(task, "Describing scene...", (0, 255, 0))

    def _do_currency(self, frame):
        """Currency recognition."""
        def task():
            result = self.currency.recognize(frame)
            self._speak(result)
        self._async_task(task, "Checking currency...", (0, 200, 255))

    def _do_read(self, frame):
        """OCR text reading."""
        def task():
            msg = self.doc_reader.start(frame, mode="hybrid")
            self._speak(msg)
        self._async_task(task, "Reading text...", (255, 100, 255))

    def _do_people(self, frame):
        """Describe people in scene."""
        def task():
            answer = self.assistant.handle_query(
                "describe the people I see",
                frame=frame,
                detections=self.last_detections,
            )
            self._speak(answer)
        self._async_task(task, "Analyzing people...", (255, 200, 0))

    def _do_find_object(self, frame):
        """Find a specific object."""
        def task():
            answer = self.assistant.handle_query(
                "what objects do you see around me?",
                frame=frame,
                detections=self.last_detections,
            )
            self._speak(answer)
        self._async_task(task, "Finding objects...", (255, 255, 100))

    def _do_voice(self, frame):
        """Voice command interaction (ESP32 button / keyboard)."""
        if not self.voice_listener:
            self._speak("Voice commands not available in this mode.")
            return

        def task():
            # Stop any current speech FIRST
            if hasattr(self.speech, 'interrupt'):
                self.speech.interrupt()

            time.sleep(0.3)
            self.overlay.set_status("Listening...", (100, 255, 255))
            print("🎤 Listening for voice command...")

            text = self.voice_listener.listen_and_transcribe()

            if not text:
                self._speak("I didn't catch that. Try again.")
                return

            print(f"🎤 Heard: {text!r}")
            self.overlay.set_status(f"Heard: {text[:40]}...", (100, 255, 255))

            t_lower = text.strip().lower()
            currency_keywords = ["how much money", "identify money", "check money",
                                 "what money", "count money", "what bills", "how much cash",
                                 "scan money", "identify currency"]
            if any(k in t_lower for k in currency_keywords) and frame is not None:
                result = self.currency.recognize(frame)
                self._speak(result)
                return

            answer = self.assistant.handle_query(
                text,
                frame=frame,
                detections=self.last_detections,
            )
            self._speak(answer)

        self._async_task(task, "Listening...", (100, 255, 255))

    def _do_time(self):
        """Tell the time."""
        def task():
            import datetime
            now = datetime.datetime.now()
            time_str = now.strftime("%-I:%M %p")
            self._speak(f"It's {time_str}.")
        self._async_task(task, "Checking time...")

    def _do_weather(self):
        """Weather info."""
        def task():
            answer = self.assistant.handle_query(
                "what's the weather like?",
                frame=None,
                detections=[],
            )
            self._speak(answer)
        self._async_task(task, "Getting weather...", (200, 200, 200))

    # ------------------------------------------------------------------
    # Main Loop
    # ------------------------------------------------------------------
    def run(self):
        """Main demo loop."""
        frame_idx = 0
        process_every = int(getattr(config, "PROCESS_EVERY_N_FRAMES", 2))

        self._speak("Hey there! VisionAssist is ready. Press the button to talk to me, or double tap and I'll describe what's around you.")

        try:
            while True:
                frame = self.camera.capture_frame()
                if frame is None:
                    print("⚠️ No frame from camera.")
                    break

                self.last_frame = frame.copy()

                fps = self.overlay.update_fps()

                if frame_idx % process_every == 0:
                    detections, annotated = self.detector.detect(frame, annotate=True)
                    self.last_detections = detections
                    self.last_annotated = annotated

                    try:
                        self.assistant.update_scene_context(frame=frame, detections=detections)
                    except Exception:
                        pass
                else:
                    detections = self.last_detections
                    annotated = self.last_annotated if self.last_annotated is not None else frame

                if self.overlay.auto_narrate and not self._busy:
                    now = time.time()
                    if now - self.last_auto_narrate_time > 8.0:
                        self._do_describe(frame.copy())
                        self.last_auto_narrate_time = now

                display = self.overlay.render(annotated, detections, fps)
                cv.imshow("VisionAssist Demo", display)

                key = cv.waitKey(1) & 0xFF
                frame_copy = frame.copy()

                if key == ord("q"):
                    print("👋 Demo ended.")
                    break
                elif key == ord("d"):
                    self._do_describe(frame_copy)
                elif key == ord("v"):
                    self._do_voice(frame_copy)
                elif key == ord("r"):
                    self._do_read(frame_copy)
                elif key == ord("c"):
                    self._do_currency(frame_copy)
                elif key == ord("p"):
                    self._do_people(frame_copy)
                elif key == ord("f"):
                    self._do_find_object(frame_copy)
                elif key == ord("t"):
                    self._do_time()
                elif key == ord("w"):
                    self._do_weather()
                elif key == ord("h"):
                    self.overlay.show_help = not self.overlay.show_help
                elif key == ord(" "):
                    self.overlay.auto_narrate = not self.overlay.auto_narrate
                    state = "ON" if self.overlay.auto_narrate else "OFF"
                    self._speak(f"Auto narration {state}.")

                frame_idx += 1

        except KeyboardInterrupt:
            print("🛑 Interrupted.")
        finally:
            if self.esp32:
                try:
                    self.esp32.stop()
                except Exception:
                    pass
            try:
                self.camera.release()
            except Exception:
                pass
            cv.destroyAllWindows()
            print("✅ Demo cleanup complete.")


# ============================================================================
# Entry Point
# ============================================================================
def main():
    parser = argparse.ArgumentParser(description="VisionAssist Competition Demo")
    parser.add_argument("--webcam", type=int, default=0, help="Camera index (default: 0)")
    parser.add_argument("--no-voice", action="store_true", help="Disable voice commands")
    args = parser.parse_args()

    demo = DemoController(
        webcam_index=args.webcam,
        enable_voice=not args.no_voice,
    )
    demo.run()


if __name__ == "__main__":
    main()
