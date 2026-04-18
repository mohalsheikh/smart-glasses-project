#!/usr/bin/env python3
"""
VisionAssist Competition Demo Mode
====================================
Input Methods:
  1. ESP32 Button: tap=voice, double-tap=describe, hold=sign language
  2. Keyboard: d/v/r/s/c/p/f/t/w/q
  3. Voice: "start sign language", "what are they signing", etc.

Headless Mode:
  python demo_mode.py --headless
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

try:
    from asl_interpreter import ASLInterpreter
except ImportError:
    ASLInterpreter = None

try:
    from body_tracker import BodyTracker
except ImportError:
    BodyTracker = None

import src.utils.config as config
from collections import deque
from typing import List, Dict, Any, Optional
import threading


# ============================================================================
# DEMO OVERLAY
# ============================================================================
class DemoOverlay:

    def __init__(self):
        self.show_help = False
        self.last_status = "Ready"
        self.last_status_color = (0, 255, 0)
        self.auto_narrate = False
        self.asl_active = False
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

        if self.asl_active:
            cv.putText(frame, "ASL", (w - 140, h - 12),
                        cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)

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
            ("Button: tap=voice  2x=describe  hold=ASL", (255, 200, 100)),
            ("", (0, 0, 0)),
            ("D  - Describe scene", (0, 255, 0)),
            ("V  - Voice command", (100, 255, 255)),
            ("R  - Read text", (255, 100, 255)),
            ("S  - Toggle sign language mode", (0, 165, 255)),
            ("C  - Currency check", (0, 200, 255)),
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

    def __init__(self, webcam_index: int = 0, enable_voice: bool = True, headless: bool = False):
        print("=" * 60)
        print("  VisionAssist - Competition Demo Mode")
        print("  AI-Powered Smart Glasses for the Visually Impaired")
        if headless:
            print("  [HEADLESS MODE — no screen required]")
        print("=" * 60)
        print()

        self.headless = headless
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

        # ASL Sign Language Interpreter
        self.asl = None
        if ASLInterpreter:
            try:
                self.asl = ASLInterpreter(speech_engine=self.speech)
            except Exception as e:
                print(f"⚠️ ASL Interpreter not available: {e}")

        # Body tracking overlay (face mesh + hand landmarks + pose)
        self.tracker = None
        if not headless and BodyTracker:
            try:
                self.tracker = BodyTracker()
            except Exception as e:
                print(f"⚠️ Body tracker not available: {e}")

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

        # ESP32 button controller — long press toggles ASL mode
        self.esp32 = None
        if ESP32ButtonListener:
            try:
                self.esp32 = ESP32ButtonListener(
                    on_voice=lambda: self._do_voice(self.last_frame) if self.last_frame is not None else None,
                    on_describe=lambda: self._do_describe(self.last_frame) if self.last_frame is not None else None,
                    on_read=lambda: self._toggle_asl_mode(),
                )
                self.esp32.start()
                print("🎮 ESP32 button controller: CONNECTED")
                print("   tap=voice | double-tap=describe | hold=sign language")
            except Exception as e:
                print(f"⚠️ ESP32 listener not available: {e}")

        print()
        print("🚀 Demo ready! Press the button to get started.")
        print()

    def _speak(self, text: str):
        text = (text or "").strip()
        if not text:
            return
        print(f"🔊 {text}")
        self.speech.speak(text)

    def _async_task(self, fn, status_text: str, status_color=(255, 200, 0)):
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
    # ASL MODE TOGGLE
    # ------------------------------------------------------------------
    def _toggle_asl_mode(self):
        """Toggle continuous ASL interpretation on/off."""
        if not self.asl or not self.asl.available:
            self._speak("Sign language interpreter not available.")
            return

        if self.asl.is_active:
            sentence = self.asl.stop()
            self.overlay.asl_active = False
            self.overlay.set_status("Ready", (0, 255, 0))
            if sentence and sentence != "No signs were detected.":
                self._speak(f"Sign language mode off. Full message was: {sentence}")
            else:
                self._speak("Sign language mode off.")
        else:
            self._speak("Sign language mode on. I'll translate what they're signing.")
            time.sleep(3.0)
            self.asl.start(
                get_frame_fn=lambda: self.last_frame.copy() if self.last_frame is not None else None,
                speech_engine=self.speech,
            )
            self.overlay.asl_active = True
            self.overlay.set_status("ASL Mode Active", (0, 165, 255))

    # ------------------------------------------------------------------
    # Demo Actions
    # ------------------------------------------------------------------
    def _do_describe(self, frame):
        def task():
            answer = self.assistant.handle_query(
                "describe the environment",
                frame=frame,
                detections=self.last_detections,
            )
            self._speak(answer)
        self._async_task(task, "Describing scene...", (0, 255, 0))

    def _do_currency(self, frame):
        def task():
            result = self.currency.recognize(frame)
            self._speak(result)
        self._async_task(task, "Checking currency...", (0, 200, 255))

    def _do_read(self, frame):
        def task():
            msg = self.doc_reader.start(frame, mode="hybrid")
            self._speak(msg)
        self._async_task(task, "Reading text...", (255, 100, 255))

    def _do_sign_language_once(self, frame):
        """Single-shot ASL interpretation."""
        if not self.asl or not self.asl.available:
            self._speak("Sign language interpreter not available.")
            return

        def task():
            print("🤟 Interpreting sign language...")
            result = self.asl.interpret_once(frame)
            self._speak(result)
        self._async_task(task, "Reading sign language...", (0, 165, 255))

    def _do_people(self, frame):
        def task():
            answer = self.assistant.handle_query(
                "describe the people I see",
                frame=frame,
                detections=self.last_detections,
            )
            self._speak(answer)
        self._async_task(task, "Analyzing people...", (255, 200, 0))

    def _do_find_object(self, frame):
        def task():
            answer = self.assistant.handle_query(
                "what objects do you see around me?",
                frame=frame,
                detections=self.last_detections,
            )
            self._speak(answer)
        self._async_task(task, "Finding objects...", (255, 255, 100))

    def _do_voice(self, frame):
        if not self.voice_listener:
            self._speak("Voice commands not available in this mode.")
            return

        def task():
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

            # Route sign language commands
            sign_start = ["start sign language", "sign language mode", "translate sign",
                          "start asl", "asl mode", "read sign language", "turn on sign"]
            sign_stop = ["stop sign language", "stop asl", "turn off sign",
                         "exit sign language", "end sign language"]
            sign_once = ["what are they signing", "interpret sign", "what is the sign",
                         "read the sign", "what sign"]

            if any(k in t_lower for k in sign_stop):
                if self.asl and self.asl.is_active:
                    self._toggle_asl_mode()
                else:
                    self._speak("Sign language mode is not active.")
                return

            if any(k in t_lower for k in sign_start):
                if self.asl and not self.asl.is_active:
                    self._toggle_asl_mode()
                else:
                    self._speak("Sign language mode is already active.")
                return

            if any(k in t_lower for k in sign_once):
                if self.asl and self.asl.available and frame is not None:
                    result = self.asl.interpret_once(frame)
                    self._speak(result)
                    return

            # Route currency commands
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
        def task():
            import datetime
            now = datetime.datetime.now()
            time_str = now.strftime("%-I:%M %p")
            self._speak(f"It's {time_str}.")
        self._async_task(task, "Checking time...")

    def _do_weather(self):
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
        frame_idx = 0
        process_every = int(getattr(config, "PROCESS_EVERY_N_FRAMES", 2))

        self._speak("Hey there! VisionAssist is ready. Press the button to talk to me, or double tap and I'll describe what's around you.")

        try:
            while True:
                frame = self.camera.capture_frame()
                if frame is None:
                    if self.headless:
                        time.sleep(0.5)
                        continue
                    print("⚠️ No frame from camera.")
                    break

                self.last_frame = frame.copy()

                # Feed frame to ASL rolling buffer (non-blocking)
                if self.asl and self.asl.is_active:
                    self.asl.feed_frame(frame)

                # Draw body tracking landmarks on frame (hands, face, pose)
                if self.tracker and self.tracker.available:
                    frame = self.tracker.draw(frame)

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

                if not self.headless:
                    display = self.overlay.render(annotated, detections, fps)
                    cv.imshow("VisionAssist Demo", display)

                if self.headless:
                    time.sleep(0.1)
                    key = 0
                else:
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
                elif key == ord("s"):
                    self._toggle_asl_mode()
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
            if self.asl and self.asl.is_active:
                self.asl.stop()
            if self.tracker:
                try:
                    self.tracker.close()
                except Exception:
                    pass
            if self.esp32:
                try:
                    self.esp32.stop()
                except Exception:
                    pass
            try:
                self.camera.release()
            except Exception:
                pass
            if not self.headless:
                cv.destroyAllWindows()
            print("✅ Demo cleanup complete.")


# ============================================================================
# Entry Point
# ============================================================================
def main():
    parser = argparse.ArgumentParser(description="VisionAssist Competition Demo")
    parser.add_argument("--webcam", type=int, default=0, help="Camera index (default: 0)")
    parser.add_argument("--no-voice", action="store_true", help="Disable voice commands")
    parser.add_argument("--headless", action="store_true", help="Run without display (no screen needed)")
    args = parser.parse_args()

    demo = DemoController(
        webcam_index=args.webcam,
        enable_voice=not args.no_voice,
        headless=args.headless,
    )
    demo.run()


if __name__ == "__main__":
    main()