"""
FAST Controller - optimized for real-time performance
Created by Mohammed
Speed-optimized: removed all slow features

This module is responsible for:
- Initializing hardware and core services
- Running the main processing loop
- Delegating description/summarization to helper utilities
"""

from collections import deque
import time

import cv2 as cv

from src.camera_handler import CameraHandler
from src.object_detector import ObjectDetector
from src.currency_recognizer import CurrencyRecognizer
from src.ocr_engine import OCREngine
from src.speech_engine import SpeechEngine
import src.utils.config as config
from src.utils.object_description import summarize_detections


class MainController:
    def __init__(self) -> None:
        # Core components
        self.camera = CameraHandler()
        self.detector = ObjectDetector()
        self.currency = CurrencyRecognizer()
        self.ocr = OCREngine()
        self.speech = SpeechEngine()

        # FPS tracking
        self.fps_queue: deque[float] = deque(maxlen=30)
        self.last_frame_time: float = time.time()

        # Last spoken sentence to avoid repetition
        self.last_spoken_sentence: str | None = None

        # Cache last detections for skipped frames
        self.last_detections = []
        self.last_annotated = None

        print("⚡ FAST Smart Glasses System Initialized")
        print(f"📊 Model: {config.DEFAULT_MODEL_NAME}")
        print(f"🎯 Processing every {config.PROCESS_EVERY_N_FRAMES} frames")

    def calculate_fps(self) -> float:
        """Calculate smoothed FPS using a moving average."""
        current_time = time.time()
        fps = 1.0 / (current_time - self.last_frame_time) if self.last_frame_time else 0.0
        self.last_frame_time = current_time
        self.fps_queue.append(fps)
        return sum(self.fps_queue) / len(self.fps_queue) if self.fps_queue else 0.0

    def run(self) -> None:
        """Main loop for the smart glasses system."""
        print("🚀 Smart Glasses System Starting...")
        print("Press 'q' to quit.")

        frame_idx = 0

        try:
            while True:
                frame = self.camera.capture_frame()
                if frame is None:
                    print("⚠️ No frame from camera, exiting.")
                    break

                frame_height, frame_width = frame.shape[:2]

                # Calculate FPS
                fps = self.calculate_fps()

                # Only run detection every Nth frame for speed
                if frame_idx % config.PROCESS_EVERY_N_FRAMES == 0:
                    detections, annotated_frame = self.detector.detect(frame, annotate=True)
                    self.last_detections = detections
                    self.last_annotated = annotated_frame
                else:
                    # Use cached detections/frame
                    detections = self.last_detections
                    annotated_frame = self.last_annotated if self.last_annotated is not None else frame

                # Overlays
                if config.SHOW_FPS:
                    cv.putText(
                        annotated_frame,
                        f"FPS: {fps:.1f}",
                        (10, 30),
                        cv.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 255, 0),
                        2,
                    )

                if config.SHOW_DETECTION_COUNT:
                    cv.putText(
                        annotated_frame,
                        f"Objects: {len(detections)}",
                        (10, 60),
                        cv.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 255, 0),
                        2,
                    )

                # Debug / visualization window
                if config.SHOW_DEBUG_WINDOW:
                    self.camera.show_image(
                        annotated_frame,
                        window_name="Smart Glasses - Fast Mode",
                    )

                # Quit on 'q'
                if cv.waitKey(1) & 0xFF == ord("q"):
                    print("👋 Exiting.")
                    break

                frame_idx += 1

                # Periodic debug logging
                if frame_idx % config.DEBUG_PRINT_EVERY_N_FRAMES == 0 and detections:
                    print(
                        f"\n--- Frame {frame_idx} | FPS: {fps:.1f} "
                        f"| Objects: {len(detections)} ---"
                    )
                    for d in detections[:8]:
                        label = str(d.get("label", "object"))
                        conf = float(d.get("confidence", 0.0))
                        print(f"  {label:20s} conf={conf:.2f}")

                # Periodic speech output
                if frame_idx % config.SPEAK_EVERY_N_FRAMES == 0 and detections:
                    objects_sentence = summarize_detections(
                        detections,
                        frame_width=frame_width,
                    )

                    if objects_sentence != self.last_spoken_sentence:
                        message = objects_sentence  # keep it simple for speed

                        try:
                            self.speech.speak(message)
                        except Exception as e:  # noqa: BLE001
                            print(f"⚠️ Speech error: {e}")

                        self.last_spoken_sentence = objects_sentence

        except KeyboardInterrupt:
            print("🛑 Interrupted by user.")

        finally:
            try:
                cv.destroyAllWindows()
            except Exception:  # noqa: BLE001
                pass

            avg_fps = sum(self.fps_queue) / len(self.fps_queue) if self.fps_queue else 0.0
            print("\n📊 Final Stats:")
            print(f"   Average FPS: {avg_fps:.1f}")
            print(f"   Total frames: {frame_idx}")
