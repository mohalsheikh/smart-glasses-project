#!/usr/bin/env python3
"""
ASL Data Collection Tool
=========================

Record ASL signs via webcam to build training datasets.
Extracts hand landmarks automatically using MediaPipe.

USAGE:
  # Collect alphabet data (press key for each letter)
  python collect_data.py --mode alphabet --output data/alphabet

  # Collect word data (record sign sequences)
  python collect_data.py --mode words --output data/words --vocab vocab.txt

  # Collect custom signs
  python collect_data.py --mode custom --output data/custom

CONTROLS:
  Alphabet mode:
    Press a-z or 0-9 to start recording that sign
    Hold for 2 seconds, then move to next
    SPACE = skip, Q = quit

  Word mode:
    SPACE = start/stop recording
    N = next word
    R = redo last recording
    Q = quit
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
from typing import List, Optional


class DataCollector:
    """Interactive data collection for ASL signs."""

    def __init__(
        self,
        output_dir: str,
        mode: str = "alphabet",
        camera_id: int = 0,
        resolution: tuple = (1280, 720),
        fps: int = 30,
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.mode = mode
        self.fps = fps

        # Camera
        self.cap = cv2.VideoCapture(camera_id)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, resolution[0])
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, resolution[1])

        # MediaPipe
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.5,
        )
        self.mp_draw = mp.solutions.drawing_utils

        # Stats
        self.total_collected = 0

    def collect_alphabet(self, samples_per_sign: int = 30):
        """Collect static sign data for alphabet and numbers."""
        labels = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ") + [str(i) for i in range(10)]

        print("\n🤟 ALPHABET COLLECTION MODE")
        print("=" * 50)
        print(f"Signs to collect: {len(labels)}")
        print(f"Samples per sign: {samples_per_sign}")
        print(f"Output: {self.output_dir}")
        print("\nControls:")
        print("  Press the letter/number key to record that sign")
        print("  Hold steady for each capture")
        print("  Q = quit\n")

        current_label = None
        recording = False
        record_start = 0
        samples_collected = {}
        capture_interval = 0.1  # Capture every 100ms

        for label in labels:
            samples_collected[label] = 0
            (self.output_dir / "landmarks" / label).mkdir(parents=True, exist_ok=True)

        last_capture = 0

        while True:
            ret, frame = self.cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)  # Mirror
            h, w = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(rgb)

            # Draw hands
            if results.multi_hand_landmarks:
                for hand_lm in results.multi_hand_landmarks:
                    self.mp_draw.draw_landmarks(frame, hand_lm, self.mp_hands.HAND_CONNECTIONS)

            # Status display
            self._draw_status(frame, current_label, recording, samples_collected)

            # Recording logic
            if recording and results.multi_hand_landmarks:
                now = time.time()
                if now - last_capture >= capture_interval:
                    hand_lm = results.multi_hand_landmarks[0]
                    landmarks = np.array(
                        [[l.x, l.y, l.z] for l in hand_lm.landmark],
                        dtype=np.float32
                    )

                    count = samples_collected[current_label]
                    save_path = self.output_dir / "landmarks" / current_label / f"sample_{count:05d}.npy"
                    np.save(str(save_path), landmarks)

                    samples_collected[current_label] += 1
                    self.total_collected += 1
                    last_capture = now

                    # Check if done
                    if samples_collected[current_label] >= samples_per_sign:
                        recording = False
                        print(f"   ✅ {current_label}: {samples_per_sign} samples collected!")

            cv2.imshow("ASL Data Collection", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key != 255:
                char = chr(key).upper()
                if char in samples_collected:
                    current_label = char
                    recording = True
                    record_start = time.time()
                    last_capture = 0
                    print(f"   🔴 Recording: {char} ({samples_collected[char]}/{samples_per_sign})")

        self._save_stats(samples_collected)

    def collect_words(self, vocab_file: Optional[str] = None, samples_per_word: int = 20):
        """Collect dynamic sign sequences for words."""

        # Load vocabulary
        if vocab_file and Path(vocab_file).exists():
            with open(vocab_file) as f:
                words = [line.strip() for line in f if line.strip()]
        else:
            words = [
                "hello", "goodbye", "please", "thank_you", "sorry",
                "yes", "no", "help", "stop", "more",
                "want", "need", "like", "understand", "again",
                "good", "bad", "big", "small", "wait",
                "what", "where", "who", "when", "why", "how",
            ]

        print("\n🤟 WORD COLLECTION MODE")
        print("=" * 50)
        print(f"Words to collect: {len(words)}")
        print(f"Samples per word: {samples_per_word}")
        print("\nControls:")
        print("  SPACE = start/stop recording")
        print("  N = next word")
        print("  P = previous word")
        print("  R = redo last recording")
        print("  Q = quit\n")

        word_idx = 0
        recording = False
        current_sequence_left = []
        current_sequence_right = []
        samples_collected = {w: 0 for w in words}

        for word in words:
            (self.output_dir / "sequences" / word).mkdir(parents=True, exist_ok=True)

        while word_idx < len(words):
            ret, frame = self.cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            h, w_frame = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(rgb)

            # Draw hands
            if results.multi_hand_landmarks:
                for hand_lm in results.multi_hand_landmarks:
                    self.mp_draw.draw_landmarks(frame, hand_lm, self.mp_hands.HAND_CONNECTIONS)

            word = words[word_idx]

            # Status
            self._draw_word_status(frame, word, word_idx, len(words),
                                    recording, len(current_sequence_left),
                                    samples_collected[word], samples_per_word)

            # Record frames
            if recording:
                left_lm = np.zeros((21, 3), dtype=np.float32)
                right_lm = np.zeros((21, 3), dtype=np.float32)

                if results.multi_hand_landmarks and results.multi_handedness:
                    for hl, hi in zip(results.multi_hand_landmarks, results.multi_handedness):
                        lm = np.array([[l.x, l.y, l.z] for l in hl.landmark], dtype=np.float32)
                        if hi.classification[0].label.lower() == "left":
                            left_lm = lm
                        else:
                            right_lm = lm

                current_sequence_left.append(left_lm)
                current_sequence_right.append(right_lm)

            cv2.imshow("ASL Data Collection", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord(' '):
                if not recording:
                    recording = True
                    current_sequence_left = []
                    current_sequence_right = []
                    print(f"   🔴 Recording: '{word}'...")
                else:
                    recording = False
                    if len(current_sequence_left) >= 3:
                        count = samples_collected[word]
                        save_path = self.output_dir / "sequences" / word / f"video_{count:04d}.npz"
                        np.savez(
                            str(save_path),
                            left_hand=np.array(current_sequence_left, dtype=np.float32),
                            right_hand=np.array(current_sequence_right, dtype=np.float32),
                        )
                        samples_collected[word] += 1
                        self.total_collected += 1
                        print(f"   ✅ Saved ({len(current_sequence_left)} frames)")
                    else:
                        print(f"   ⚠️  Too short, discarded")
                    current_sequence_left = []
                    current_sequence_right = []
            elif key == ord('n'):
                word_idx += 1
                recording = False
                current_sequence_left = []
                current_sequence_right = []
                if word_idx < len(words):
                    print(f"\n📝 Next word: '{words[word_idx]}'")
            elif key == ord('p') and word_idx > 0:
                word_idx -= 1
                recording = False
            elif key == ord('r'):
                recording = False
                current_sequence_left = []
                current_sequence_right = []
                print(f"   🔄 Ready to redo")

        # Save vocab
        with open(str(self.output_dir / "vocab.txt"), "w") as f:
            for w in words:
                f.write(f"{w}\n")

        self._save_stats(samples_collected)

    def _draw_status(self, frame, label, recording, counts):
        """Draw collection status overlay."""
        h, w = frame.shape[:2]
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 80), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

        if recording:
            cv2.circle(frame, (30, 40), 12, (0, 0, 255), -1)
            text = f"Recording: {label}"
            count = counts.get(label, 0)
            cv2.putText(frame, f"{text} ({count} samples)", (55, 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        else:
            cv2.putText(frame, "Press letter/number to record | Q=quit",
                       (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)

        cv2.putText(frame, f"Total: {self.total_collected}", (w - 200, 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    def _draw_word_status(self, frame, word, idx, total, recording, frames, collected, target):
        """Draw word collection status."""
        h, w = frame.shape[:2]
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 100), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

        # Word display
        cv2.putText(frame, f"Sign: '{word}'", (20, 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        cv2.putText(frame, f"Word {idx+1}/{total} | Samples: {collected}/{target}",
                   (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

        if recording:
            cv2.circle(frame, (w - 50, 40), 15, (0, 0, 255), -1)
            cv2.putText(frame, f"{frames}f", (w - 100, 80),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        # Controls
        cv2.putText(frame, "SPACE=record | N=next | P=prev | R=redo | Q=quit",
                   (20, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)

    def _save_stats(self, counts):
        """Save collection statistics."""
        stats = {
            "total_collected": self.total_collected,
            "per_class": {k: v for k, v in counts.items() if v > 0},
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        with open(str(self.output_dir / "collection_stats.json"), "w") as f:
            json.dump(stats, f, indent=2)
        print(f"\n📊 Stats saved. Total collected: {self.total_collected}")

    def cleanup(self):
        self.cap.release()
        self.hands.close()
        cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description="ASL Data Collection Tool")
    parser.add_argument("--mode", choices=["alphabet", "words", "custom"], default="alphabet")
    parser.add_argument("--output", type=str, default="data/alphabet")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--samples", type=int, default=30, help="Samples per sign")
    parser.add_argument("--vocab", type=str, default=None, help="Vocab file for word mode")
    parser.add_argument("--resolution", type=str, default="1280x720")

    args = parser.parse_args()
    w, h = map(int, args.resolution.split("x"))

    collector = DataCollector(
        output_dir=args.output,
        mode=args.mode,
        camera_id=args.camera,
        resolution=(w, h),
    )

    try:
        if args.mode == "alphabet":
            collector.collect_alphabet(samples_per_sign=args.samples)
        elif args.mode in ("words", "custom"):
            collector.collect_words(vocab_file=args.vocab, samples_per_word=args.samples)
    except KeyboardInterrupt:
        print("\n👋 Interrupted")
    finally:
        collector.cleanup()


if __name__ == "__main__":
    main()
