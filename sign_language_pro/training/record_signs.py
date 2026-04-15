#!/usr/bin/env python3
"""
ASL Sign Data Recorder
========================

Record your own ASL sign data using a webcam.
Creates training data compatible with prepare_data.py → train.py pipeline.

Usage:
  # Record static signs (alphabet, numbers, etc.)
  python record_signs.py --mode static --output ./my_recordings

  # Record dynamic signs (words, phrases)
  python record_signs.py --mode dynamic --output ./my_recordings

  # Record with guided prompts (walks you through signs)
  python record_signs.py --mode static --guided --output ./my_recordings

Controls:
  SPACE  — Start/stop recording
  s      — Save current recording
  n      — Next sign (in guided mode)
  r      — Redo current sign
  q      — Quit
"""

from __future__ import annotations

import os
import sys
import cv2
import time
import json
import argparse
import numpy as np
import mediapipe as mp
from pathlib import Path
from datetime import datetime
from collections import deque


# =============================================================================
# ASL SIGNS TO RECORD
# =============================================================================

STATIC_SIGNS = {
    "alphabet": list("ABCDEFGHIJKLMNOPQRSTUVWXYZ"),
    "numbers": list("0123456789"),
    "extras": ["SPACE", "DELETE", "NOTHING"],
}

DYNAMIC_SIGNS = {
    "greetings": ["hello", "goodbye", "please", "thank_you", "sorry", "welcome"],
    "questions": ["what", "where", "who", "when", "why", "how", "how_much"],
    "responses": ["yes", "no", "maybe", "ok", "dont_know", "understand", "dont_understand"],
    "common": [
        "help", "stop", "more", "want", "need", "like", "love", "eat", "drink",
        "water", "food", "bathroom", "home", "work", "school", "go", "come",
        "wait", "again", "slow", "fast", "good", "bad", "big", "small",
        "hot", "cold", "happy", "sad", "tired", "sick", "name", "friend",
        "family", "mother", "father", "brother", "sister", "baby",
    ],
    "emergency": ["emergency", "danger", "hurt", "pain", "call", "doctor", "hospital"],
    "time": ["today", "tomorrow", "yesterday", "morning", "afternoon", "night", "now", "later"],
    "descriptions": [
        "beautiful", "ugly", "new", "old", "same", "different",
        "right", "wrong", "easy", "hard", "important",
    ],
}

# Minimum recordings per sign for good training
MIN_STATIC_SAMPLES = 50   # images per sign
MIN_DYNAMIC_SAMPLES = 15  # video clips per sign


class SignRecorder:
    """Interactive sign language data recorder."""

    def __init__(
        self,
        mode: str = "static",
        output_dir: str = "./my_recordings",
        camera_id: int = 0,
        resolution: tuple = (1280, 720),
        fps: int = 30,
        video_duration: float = 3.0,  # seconds per dynamic recording
    ):
        self.mode = mode
        self.output_dir = Path(output_dir) / mode
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.fps = fps
        self.video_duration = video_duration
        self.resolution = resolution

        # Camera
        self.cap = cv2.VideoCapture(camera_id)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, resolution[0])
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, resolution[1])
        self.cap.set(cv2.CAP_PROP_FPS, fps)

        # MediaPipe for live hand detection feedback
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.mp_draw = mp.solutions.drawing_utils

        # Recording state
        self.is_recording = False
        self.current_sign = ""
        self.recording_frames = []
        self.recording_start = 0.0

        # Stats
        self.sign_counts = self._load_existing_counts()

    def _load_existing_counts(self) -> dict:
        """Count existing recordings."""
        counts = {}
        if not self.output_dir.exists():
            return counts

        for sign_dir in self.output_dir.iterdir():
            if sign_dir.is_dir():
                if self.mode == "static":
                    count = len(list(sign_dir.glob("*.jpg")))
                else:
                    count = len(list(sign_dir.glob("*.mp4")))
                counts[sign_dir.name] = count

        return counts

    def record_static_guided(self):
        """Guided mode: walk through each static sign."""
        all_signs = []
        for group in STATIC_SIGNS.values():
            all_signs.extend(group)

        sign_idx = 0
        auto_capture = False
        auto_timer = 0
        frames_captured = 0

        print("\n" + "=" * 60)
        print("📸 STATIC SIGN RECORDER (Guided Mode)")
        print("=" * 60)
        print("Controls:")
        print("  SPACE — Capture one frame")
        print("  a     — Toggle auto-capture (1 per second)")
        print("  n     — Next sign")
        print("  p     — Previous sign")
        print("  r     — Redo (delete last capture)")
        print("  q     — Quit")
        print("=" * 60)

        while sign_idx < len(all_signs):
            sign = all_signs[sign_idx]
            sign_dir = self.output_dir / sign
            sign_dir.mkdir(parents=True, exist_ok=True)

            existing = len(list(sign_dir.glob("*.jpg")))
            min_needed = MIN_STATIC_SAMPLES

            ret, frame = self.cap.read()
            if not ret:
                break

            # Detect hands for visual feedback
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(frame_rgb)
            hand_detected = False

            if results.multi_hand_landmarks:
                hand_detected = True
                for hand_lm in results.multi_hand_landmarks:
                    self.mp_draw.draw_landmarks(
                        frame, hand_lm, self.mp_hands.HAND_CONNECTIONS
                    )

            # Draw UI
            h, w = frame.shape[:2]

            # Top banner
            cv2.rectangle(frame, (0, 0), (w, 80), (30, 30, 30), -1)
            cv2.putText(frame, f"Sign: {sign}", (20, 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 128), 2)
            cv2.putText(frame, f"Captured: {existing + frames_captured}/{min_needed}",
                        (20, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1)

            # Progress bar
            progress = min(1.0, (existing + frames_captured) / min_needed)
            bar_w = int((w - 40) * progress)
            color = (0, 255, 0) if progress >= 1.0 else (0, 200, 255)
            cv2.rectangle(frame, (20, 72), (20 + bar_w, 78), color, -1)

            # Sign index
            cv2.putText(frame, f"[{sign_idx + 1}/{len(all_signs)}]", (w - 120, 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150, 150, 150), 1)

            # Auto-capture indicator
            if auto_capture:
                cv2.putText(frame, "AUTO", (w - 120, 65),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            # Hand status
            status_color = (0, 255, 0) if hand_detected else (0, 0, 255)
            status_text = "Hand Detected" if hand_detected else "No Hand"
            cv2.putText(frame, status_text, (w - 200, h - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 1)

            # Instructions at bottom
            cv2.rectangle(frame, (0, h - 50), (w, h), (30, 30, 30), -1)
            cv2.putText(frame, "SPACE=capture  a=auto  n=next  p=prev  q=quit",
                        (20, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)

            cv2.imshow("Sign Recorder", frame)

            # Auto-capture
            if auto_capture and hand_detected and time.time() - auto_timer > 0.5:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                cv2.imwrite(str(sign_dir / f"{sign}_{timestamp}.jpg"), frame)
                frames_captured += 1
                auto_timer = time.time()

            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                break
            elif key == ord(' ') and hand_detected:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                cv2.imwrite(str(sign_dir / f"{sign}_{timestamp}.jpg"), frame)
                frames_captured += 1
                print(f"  📸 Captured {sign} ({existing + frames_captured})")
            elif key == ord('a'):
                auto_capture = not auto_capture
                auto_timer = time.time()
                print(f"  Auto-capture: {'ON' if auto_capture else 'OFF'}")
            elif key == ord('n'):
                print(f"  ✅ {sign}: {existing + frames_captured} samples")
                sign_idx += 1
                frames_captured = 0
                auto_capture = False
            elif key == ord('p') and sign_idx > 0:
                sign_idx -= 1
                frames_captured = 0
                auto_capture = False
            elif key == ord('r'):
                # Delete last captured
                files = sorted(sign_dir.glob(f"{sign}_*.jpg"))
                if files:
                    files[-1].unlink()
                    frames_captured = max(0, frames_captured - 1)
                    print(f"  ↩️ Deleted last capture")

        self._cleanup()

    def record_dynamic_guided(self):
        """Guided mode for dynamic (word) signs."""
        all_signs = []
        for group in DYNAMIC_SIGNS.values():
            all_signs.extend(group)

        sign_idx = 0

        print("\n" + "=" * 60)
        print("🎬 DYNAMIC SIGN RECORDER (Guided Mode)")
        print("=" * 60)
        print(f"Each recording is {self.video_duration}s")
        print("Controls:")
        print("  SPACE — Start/stop recording")
        print("  n     — Next sign")
        print("  p     — Previous sign")
        print("  q     — Quit")
        print("=" * 60)

        while sign_idx < len(all_signs):
            sign = all_signs[sign_idx]
            sign_dir = self.output_dir / sign
            sign_dir.mkdir(parents=True, exist_ok=True)

            existing = len(list(sign_dir.glob("*.mp4")))

            ret, frame = self.cap.read()
            if not ret:
                break

            h, w = frame.shape[:2]

            # Detect hands
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(frame_rgb)

            if results.multi_hand_landmarks:
                for hand_lm in results.multi_hand_landmarks:
                    self.mp_draw.draw_landmarks(
                        frame, hand_lm, self.mp_hands.HAND_CONNECTIONS
                    )

            if self.is_recording:
                self.recording_frames.append(frame.copy())
                elapsed = time.time() - self.recording_start

                # Recording indicator
                cv2.circle(frame, (w - 30, 30), 12, (0, 0, 255), -1)
                cv2.putText(frame, f"REC {elapsed:.1f}s / {self.video_duration}s",
                            (w - 200, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

                # Auto-stop
                if elapsed >= self.video_duration:
                    self._save_dynamic_recording(sign, sign_dir)
                    existing += 1
            else:
                # UI
                cv2.rectangle(frame, (0, 0), (w, 80), (30, 30, 30), -1)
                cv2.putText(frame, f"Sign: {sign.replace('_', ' ').upper()}", (20, 35),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 128), 2)
                cv2.putText(frame, f"Recordings: {existing}/{MIN_DYNAMIC_SAMPLES}",
                            (20, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1)

                cv2.putText(frame, f"[{sign_idx + 1}/{len(all_signs)}]", (w - 120, 35),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150, 150, 150), 1)

            cv2.rectangle(frame, (0, h - 50), (w, h), (30, 30, 30), -1)
            cv2.putText(frame, "SPACE=record  n=next  p=prev  q=quit",
                        (20, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)

            cv2.imshow("Sign Recorder", frame)

            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                break
            elif key == ord(' '):
                if not self.is_recording:
                    self.is_recording = True
                    self.recording_frames = []
                    self.recording_start = time.time()
                    print(f"  🔴 Recording '{sign}'...")
                else:
                    self._save_dynamic_recording(sign, sign_dir)
                    existing += 1
            elif key == ord('n') and not self.is_recording:
                sign_idx += 1
            elif key == ord('p') and not self.is_recording and sign_idx > 0:
                sign_idx -= 1

        self._cleanup()

    def _save_dynamic_recording(self, sign: str, sign_dir: Path):
        """Save recorded frames as video."""
        self.is_recording = False

        if not self.recording_frames:
            return

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = sign_dir / f"{sign}_{timestamp}.mp4"

        h, w = self.recording_frames[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(str(filename), fourcc, self.fps, (w, h))

        for f in self.recording_frames:
            writer.write(f)
        writer.release()

        self.recording_frames = []
        print(f"  ✅ Saved {filename.name} ({len(self.recording_frames)} frames)")

    def record_free(self):
        """Free-form recording mode."""
        print("\n" + "=" * 60)
        print("🎬 FREE RECORDING MODE")
        print("=" * 60)
        print("Type a sign name, then record it.")
        print("Commands: 'list', 'stats', 'quit'")
        print("=" * 60)

        while True:
            sign = input("\nSign name (or command): ").strip().lower()

            if sign == 'quit':
                break
            elif sign == 'list':
                if self.mode == 'static':
                    for g, signs in STATIC_SIGNS.items():
                        print(f"  {g}: {', '.join(signs)}")
                else:
                    for g, signs in DYNAMIC_SIGNS.items():
                        print(f"  {g}: {', '.join(signs)}")
                continue
            elif sign == 'stats':
                counts = self._load_existing_counts()
                for s, c in sorted(counts.items()):
                    print(f"  {s}: {c}")
                continue
            elif not sign:
                continue

            sign_dir = self.output_dir / sign
            sign_dir.mkdir(parents=True, exist_ok=True)

            if self.mode == "static":
                n = int(input(f"  How many captures for '{sign}'? [20]: ") or "20")
                self._capture_static(sign, sign_dir, n)
            else:
                n = int(input(f"  How many recordings for '{sign}'? [5]: ") or "5")
                self._capture_dynamic(sign, sign_dir, n)

    def _capture_static(self, sign: str, sign_dir: Path, count: int):
        """Capture N static images."""
        captured = 0
        print(f"  Press SPACE to capture, 'a' for auto, 'q' when done")

        auto = False
        auto_timer = 0

        while captured < count:
            ret, frame = self.cap.read()
            if not ret:
                break

            h, w = frame.shape[:2]
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(frame_rgb)
            hand_ok = bool(results.multi_hand_landmarks)

            if results.multi_hand_landmarks:
                for hand_lm in results.multi_hand_landmarks:
                    self.mp_draw.draw_landmarks(frame, hand_lm, self.mp_hands.HAND_CONNECTIONS)

            cv2.putText(frame, f"{sign.upper()} [{captured}/{count}]", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 128), 2)

            if auto and hand_ok and time.time() - auto_timer > 0.4:
                ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                cv2.imwrite(str(sign_dir / f"{sign}_{ts}.jpg"), frame)
                captured += 1
                auto_timer = time.time()

            cv2.imshow("Sign Recorder", frame)
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                break
            elif key == ord(' ') and hand_ok:
                ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                cv2.imwrite(str(sign_dir / f"{sign}_{ts}.jpg"), frame)
                captured += 1
            elif key == ord('a'):
                auto = not auto
                auto_timer = time.time()

        print(f"  ✅ Captured {captured} images for '{sign}'")

    def _capture_dynamic(self, sign: str, sign_dir: Path, count: int):
        """Capture N video clips."""
        recorded = 0

        while recorded < count:
            print(f"  [{recorded + 1}/{count}] Press SPACE to start recording...")

            # Wait for space
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    return

                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = self.hands.process(frame_rgb)
                if results.multi_hand_landmarks:
                    for h in results.multi_hand_landmarks:
                        self.mp_draw.draw_landmarks(frame, h, self.mp_hands.HAND_CONNECTIONS)

                cv2.putText(frame, f"Ready: {sign.upper()} [{recorded + 1}/{count}]",
                            (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                cv2.putText(frame, "Press SPACE to record", (20, 70),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
                cv2.imshow("Sign Recorder", frame)

                key = cv2.waitKey(1) & 0xFF
                if key == ord(' '):
                    break
                elif key == ord('q'):
                    return

            # Record
            frames = []
            start = time.time()

            while time.time() - start < self.video_duration:
                ret, frame = self.cap.read()
                if not ret:
                    break
                frames.append(frame.copy())

                elapsed = time.time() - start
                cv2.circle(frame, (frame.shape[1] - 30, 30), 12, (0, 0, 255), -1)
                cv2.putText(frame, f"REC {elapsed:.1f}s", (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                cv2.imshow("Sign Recorder", frame)
                cv2.waitKey(1)

            # Save
            if frames:
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = sign_dir / f"{sign}_{ts}.mp4"
                h, w = frames[0].shape[:2]
                writer = cv2.VideoWriter(str(filename), cv2.VideoWriter_fourcc(*'mp4v'), self.fps, (w, h))
                for f in frames:
                    writer.write(f)
                writer.release()
                recorded += 1
                print(f"    ✅ Saved ({len(frames)} frames)")

    def _cleanup(self):
        self.cap.release()
        self.hands.close()
        cv2.destroyAllWindows()
        print("\n✅ Recorder closed. Run prepare_data.py to extract landmarks.")


def main():
    parser = argparse.ArgumentParser(description="ASL Sign Data Recorder")
    parser.add_argument("--mode", choices=["static", "dynamic"], default="static")
    parser.add_argument("--output", default="./my_recordings")
    parser.add_argument("--guided", action="store_true", help="Guided mode")
    parser.add_argument("--free", action="store_true", help="Free-form mode")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--duration", type=float, default=3.0, help="Video clip duration (seconds)")

    args = parser.parse_args()

    recorder = SignRecorder(
        mode=args.mode,
        output_dir=args.output,
        camera_id=args.camera,
        video_duration=args.duration,
    )

    if args.free:
        recorder.record_free()
    elif args.guided:
        if args.mode == "static":
            recorder.record_static_guided()
        else:
            recorder.record_dynamic_guided()
    else:
        # Default: guided
        if args.mode == "static":
            recorder.record_static_guided()
        else:
            recorder.record_dynamic_guided()


if __name__ == "__main__":
    main()
