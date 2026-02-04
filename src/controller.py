"""
Controller module - orchestrates all components
Created by Mohammed
Updated to:
  - run in a loop
  - show a live camera window with detections
  - speak a short summary instead of raw arrays
"""

import time
from collections import Counter
from typing import Any, Dict, List, Optional

import cv2 as cv

from src.camera_handler import CameraHandler
from src.object_detector import ObjectDetector
from src.currency_recognizer import CurrencyRecognizer
from src.ocr_engine import OCREngine
from src.speech_engine import SpeechEngine


class MainController:
    def __init__(self):
        self.camera = CameraHandler()
        self.detector = ObjectDetector()
        self.currency = CurrencyRecognizer()
        self.ocr = OCREngine()
        self.speech = SpeechEngine()

        # Simple speech rate limiting
        self._last_message: Optional[str] = None
        self._last_message_time: float = 0.0

    # ───────────────────── Helper methods ─────────────────────

    def _maybe_speak(self, message: Optional[str]) -> None:
        """Speak only if message is non-empty and not spammy."""
        if not message:
            return

        now = time.time()
        min_gap = 2.0  # seconds between messages

        # Avoid repeating the exact same message too frequently
        if (
            self._last_message is not None
            and message == self._last_message
            and (now - self._last_message_time) < 6.0
        ):
            return

        if (now - self._last_message_time) < min_gap:
            return

        print(f"🔊 Speaking: {message}")
        self.speech.speak(message)
        self._last_message = message
        self._last_message_time = now

    @staticmethod
    def _summarize_detections(detections: List[Dict[str, Any]]) -> Optional[str]:
        """
        Turn detections into a simple sentence like:
          'one person and one couch detected ahead.'
        """
        if not detections:
            return None

        labels = [d.get("label") for d in detections if d.get("label")]
        if not labels:
            return None

        counts = Counter(labels)
        parts = []
        for label, count in counts.most_common(3):  # max 3 types mentioned
            if count == 1:
                parts.append(f"one {label}")
            else:
                # basic pluralization
                if label.endswith("s"):
                    parts.append(f"{count} {label}")
                else:
                    parts.append(f"{count} {label}s")

        if not parts:
            return None

        if len(parts) == 1:
            phrase = parts[0]
        elif len(parts) == 2:
            phrase = " and ".join(parts)
        else:
            phrase = ", ".join(parts[:-1]) + f", and {parts[-1]}"

        return f"{phrase} detected ahead."

    # ───────────────────── Main loop ─────────────────────

    def run(self):
        print("🚀 Smart Glasses System Starting...")
        print("Press 'q' in the camera window to quit.")

        try:
            while True:
                frame = self.camera.capture_frame()
                if frame is None:
                    print("[WARN] No frame captured from camera.")
                    continue

                # detector.detect returns (detections_list, frame_out)
                detections, annotated_frame = self.detector.detect(
                    frame,
                    annotate=True
                )

                # Show what the system sees (with boxes)
                self.camera.show_image(annotated_frame, window_name="Smart Glasses View")

                # Press 'q' to quit
                if cv.waitKey(1) & 0xFF == ord("q"):
                    break

                # Placeholder OCR and currency
                text = self.ocr.read_text(frame)
                currency = self.currency.recognize(frame)

                # Build a short spoken message ONLY from detections for now
                message = self._summarize_detections(detections)

                # Debug print in terminal
                print("\n[DEBUG] Detections:")
                for d in detections:
                    print(
                        f"  - {d['label']} "
                        f"(conf={float(d['confidence']):.2f}, center={d['center']})"
                    )
                print(f"[DEBUG] OCR text: {text}")
                print(f"[DEBUG] Currency: {currency}")
                print(f"[INFO] Message: {message}")

                # Speak the message (if any)
                self._maybe_speak(message)

        finally:
            cv.destroyAllWindows()
            print("🛑 Smart Glasses System Stopped.")
