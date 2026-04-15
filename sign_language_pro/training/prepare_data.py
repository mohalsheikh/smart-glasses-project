#!/usr/bin/env python3
"""
ASL Dataset Preparation Pipeline
==================================

Downloads ASL datasets and extracts MediaPipe hand landmarks for training.

Supports:
  1. Kaggle ASL Alphabet (static fingerspelling A-Z + space/delete/nothing)
  2. WLASL (Word-Level ASL) video dataset for dynamic signs
  3. Custom recorded data from record_signs.py

Usage:
  python prepare_data.py --dataset kaggle_alphabet --output ./data
  python prepare_data.py --dataset wlasl --output ./data --wlasl_json ./WLASL_v0.3.json
  python prepare_data.py --dataset custom --input ./my_recordings --output ./data

Requirements:
  pip install mediapipe opencv-python numpy tqdm kaggle
"""

from __future__ import annotations

import os
import sys
import json
import shutil
import argparse
import zipfile
import subprocess
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from collections import defaultdict

import cv2
import numpy as np
import mediapipe as mp
from tqdm import tqdm


# =============================================================================
# CONSTANTS
# =============================================================================

# MediaPipe hand landmark count
NUM_LANDMARKS = 21
NUM_COORDS = 3  # x, y, z
FEATURE_DIM = NUM_LANDMARKS * NUM_COORDS  # 63

# For dynamic signs, we extract sequences of this many frames
SEQUENCE_LENGTH = 30

# Minimum hand detection confidence
MIN_DETECTION_CONF = 0.5


# =============================================================================
# LANDMARK EXTRACTION
# =============================================================================

class LandmarkExtractor:
    """Extract and normalize MediaPipe hand landmarks from images/video."""

    def __init__(
        self,
        static_mode: bool = True,
        max_hands: int = 2,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
    ):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=static_mode,
            max_num_hands=max_hands,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

    def extract_from_image(self, image: np.ndarray) -> Optional[np.ndarray]:
        """
        Extract normalized landmarks from a single image.

        Returns:
            np.ndarray of shape (63,) — 21 landmarks × 3 coords, normalized.
            None if no hand detected.
        """
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb)

        if not results.multi_hand_landmarks:
            return None

        # Take the first detected hand
        hand = results.multi_hand_landmarks[0]
        landmarks = np.array([[lm.x, lm.y, lm.z] for lm in hand.landmark])

        return self._normalize(landmarks)

    def extract_from_image_both_hands(self, image: np.ndarray) -> Optional[np.ndarray]:
        """
        Extract landmarks for both hands (for two-handed signs).

        Returns:
            np.ndarray of shape (126,) — 2 hands × 21 landmarks × 3 coords.
            Missing hand is zero-padded.
        """
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb)

        left = np.zeros((NUM_LANDMARKS, NUM_COORDS), dtype=np.float32)
        right = np.zeros((NUM_LANDMARKS, NUM_COORDS), dtype=np.float32)

        if results.multi_hand_landmarks and results.multi_handedness:
            for hand_lm, hand_info in zip(
                results.multi_hand_landmarks, results.multi_handedness
            ):
                lm = np.array([[l.x, l.y, l.z] for l in hand_lm.landmark])
                label = hand_info.classification[0].label.lower()
                if label == "left":
                    left = lm
                else:
                    right = lm

        left_norm = self._normalize(left)
        right_norm = self._normalize(right)

        return np.concatenate([left_norm, right_norm])

    def extract_sequence_from_video(
        self, video_path: str, target_frames: int = SEQUENCE_LENGTH
    ) -> Optional[np.ndarray]:
        """
        Extract a sequence of landmark frames from a video.

        Returns:
            np.ndarray of shape (target_frames, 126) for two hands.
            None if insufficient hand detections.
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return None

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            cap.release()
            return None

        # Sample frames uniformly
        if total_frames >= target_frames:
            indices = np.linspace(0, total_frames - 1, target_frames, dtype=int)
        else:
            indices = list(range(total_frames))

        frames_data = []
        frame_idx = 0
        sample_idx = 0

        while cap.isOpened() and sample_idx < len(indices):
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx == indices[sample_idx]:
                lm = self.extract_from_image_both_hands(frame)
                if lm is not None:
                    frames_data.append(lm)
                else:
                    frames_data.append(np.zeros(NUM_LANDMARKS * NUM_COORDS * 2, dtype=np.float32))
                sample_idx += 1

            frame_idx += 1

        cap.release()

        if len(frames_data) < target_frames // 2:
            return None  # Too few detections

        # Pad or truncate to target length
        sequence = np.array(frames_data, dtype=np.float32)

        if len(sequence) < target_frames:
            pad = np.zeros((target_frames - len(sequence), sequence.shape[1]), dtype=np.float32)
            sequence = np.vstack([sequence, pad])
        elif len(sequence) > target_frames:
            indices = np.linspace(0, len(sequence) - 1, target_frames, dtype=int)
            sequence = sequence[indices]

        return sequence

    def _normalize(self, landmarks: np.ndarray) -> np.ndarray:
        """
        Normalize landmarks:
        1. Center on wrist (translation invariance)
        2. Scale by palm size (scale invariance)
        3. Flatten to 1D
        """
        if landmarks.shape != (NUM_LANDMARKS, NUM_COORDS):
            return np.zeros(FEATURE_DIM, dtype=np.float32)

        # Center on wrist
        wrist = landmarks[0].copy()
        centered = landmarks - wrist

        # Scale by distance from wrist to middle finger MCP (landmark 9)
        palm_size = np.linalg.norm(centered[9])
        if palm_size > 1e-6:
            centered = centered / palm_size

        return centered.flatten().astype(np.float32)

    def close(self):
        self.hands.close()


# =============================================================================
# DATASET: KAGGLE ASL ALPHABET
# =============================================================================

def prepare_kaggle_alphabet(
    dataset_dir: str,
    output_dir: str,
    max_per_class: int = 3000,
    val_split: float = 0.15,
    test_split: float = 0.10,
):
    """
    Prepare the Kaggle ASL Alphabet dataset.

    Expected structure:
      dataset_dir/
        asl_alphabet_train/
          A/ B/ C/ ... Z/ del/ nothing/ space/

    Download from: https://www.kaggle.com/datasets/grassknoted/asl-alphabet
    """
    train_dir = Path(dataset_dir) / "asl_alphabet_train"
    if not train_dir.exists():
        # Try alternate structure
        train_dir = Path(dataset_dir) / "asl_alphabet_train" / "asl_alphabet_train"
    if not train_dir.exists():
        print(f"❌ Could not find training images in {dataset_dir}")
        print("   Expected: asl_alphabet_train/A/, asl_alphabet_train/B/, ...")
        print("   Download from: https://www.kaggle.com/datasets/grassknoted/asl-alphabet")
        return

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    extractor = LandmarkExtractor(static_mode=True, max_hands=1)

    # Map class names
    class_dirs = sorted([d for d in train_dir.iterdir() if d.is_dir()])
    class_names = []
    for d in class_dirs:
        name = d.name.upper()
        if name == "DEL":
            name = "DELETE"
        elif name == "NOTHING":
            name = "NOTHING"
        elif name == "SPACE":
            name = "SPACE"
        class_names.append(name)

    print(f"Found {len(class_names)} classes: {class_names}")

    all_features = []
    all_labels = []

    for class_idx, (class_dir, class_name) in enumerate(zip(class_dirs, class_names)):
        images = list(class_dir.glob("*.jpg")) + list(class_dir.glob("*.png"))
        images = images[:max_per_class]

        class_features = []
        print(f"\n[{class_idx + 1}/{len(class_names)}] Processing '{class_name}' ({len(images)} images)...")

        for img_path in tqdm(images, desc=class_name, leave=False):
            img = cv2.imread(str(img_path))
            if img is None:
                continue

            lm = extractor.extract_from_image(img)
            if lm is not None:
                class_features.append(lm)

        if class_features:
            features = np.array(class_features, dtype=np.float32)
            labels = np.full(len(features), class_idx, dtype=np.int64)
            all_features.append(features)
            all_labels.append(labels)
            print(f"   ✅ {class_name}: {len(features)} landmarks extracted")
        else:
            print(f"   ❌ {class_name}: No landmarks extracted!")

    extractor.close()

    if not all_features:
        print("❌ No data extracted!")
        return

    # Combine
    X = np.vstack(all_features)
    y = np.concatenate(all_labels)

    print(f"\nTotal samples: {len(X)}")

    # Shuffle
    perm = np.random.permutation(len(X))
    X = X[perm]
    y = y[perm]

    # Split
    n = len(X)
    n_test = int(n * test_split)
    n_val = int(n * val_split)
    n_train = n - n_test - n_val

    splits = {
        "train": (X[:n_train], y[:n_train]),
        "val": (X[n_train : n_train + n_val], y[n_train : n_train + n_val]),
        "test": (X[n_train + n_val :], y[n_train + n_val :]),
    }

    # Save
    for split_name, (features, labels) in splits.items():
        np.save(output_path / f"static_{split_name}_X.npy", features)
        np.save(output_path / f"static_{split_name}_y.npy", labels)
        print(f"  {split_name}: {len(features)} samples saved")

    # Save class mapping
    class_map = {name: idx for idx, name in enumerate(class_names)}
    with open(output_path / "static_classes.json", "w") as f:
        json.dump(class_map, f, indent=2)

    print(f"\n✅ Static dataset saved to {output_path}")
    print(f"   Classes: {len(class_names)}")
    print(f"   Feature dim: {X.shape[1]}")


# =============================================================================
# DATASET: WLASL (Word-Level ASL)
# =============================================================================

def prepare_wlasl(
    wlasl_json: str,
    videos_dir: str,
    output_dir: str,
    max_classes: int = 200,
    min_samples_per_class: int = 5,
    sequence_length: int = SEQUENCE_LENGTH,
    val_split: float = 0.15,
    test_split: float = 0.10,
):
    """
    Prepare the WLASL dataset for dynamic sign training.

    Download:
      1. WLASL JSON: https://github.com/dxli94/WLASL
      2. Videos: Use the download script from the WLASL repo

    Expected:
      wlasl_json: Path to WLASL_v0.3.json
      videos_dir: Directory containing downloaded .mp4 files
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Load WLASL annotations
    with open(wlasl_json, "r") as f:
        wlasl_data = json.load(f)

    print(f"WLASL contains {len(wlasl_data)} glosses (words)")

    # Count available videos per class
    videos_path = Path(videos_dir)
    class_counts = {}

    for entry in wlasl_data[:max_classes * 2]:  # Check more to find enough
        gloss = entry["gloss"]
        instances = entry.get("instances", [])
        available = 0

        for inst in instances:
            video_id = inst.get("video_id", "")
            video_file = videos_path / f"{video_id}.mp4"
            if video_file.exists():
                available += 1

        if available >= min_samples_per_class:
            class_counts[gloss] = available

    # Take top N classes by sample count
    sorted_classes = sorted(class_counts.items(), key=lambda x: x[1], reverse=True)
    selected_classes = sorted_classes[:max_classes]

    print(f"Selected {len(selected_classes)} classes with >= {min_samples_per_class} videos")

    class_names = [c[0] for c in selected_classes]
    class_map = {name: idx for idx, name in enumerate(class_names)}

    # Extract landmarks
    extractor = LandmarkExtractor(
        static_mode=False,
        max_hands=2,
        min_detection_confidence=0.4,
        min_tracking_confidence=0.4,
    )

    all_sequences = []
    all_labels = []

    for entry in tqdm(wlasl_data, desc="Processing WLASL"):
        gloss = entry["gloss"]
        if gloss not in class_map:
            continue

        label = class_map[gloss]

        for inst in entry.get("instances", []):
            video_id = inst.get("video_id", "")
            video_file = videos_path / f"{video_id}.mp4"

            if not video_file.exists():
                continue

            sequence = extractor.extract_sequence_from_video(
                str(video_file), target_frames=sequence_length
            )

            if sequence is not None:
                all_sequences.append(sequence)
                all_labels.append(label)

    extractor.close()

    if not all_sequences:
        print("❌ No sequences extracted!")
        return

    X = np.array(all_sequences, dtype=np.float32)
    y = np.array(all_labels, dtype=np.int64)

    print(f"\nTotal sequences: {len(X)}")
    print(f"Shape: {X.shape}")

    # Shuffle and split
    perm = np.random.permutation(len(X))
    X = X[perm]
    y = y[perm]

    n = len(X)
    n_test = int(n * test_split)
    n_val = int(n * val_split)
    n_train = n - n_test - n_val

    splits = {
        "train": (X[:n_train], y[:n_train]),
        "val": (X[n_train : n_train + n_val], y[n_train : n_train + n_val]),
        "test": (X[n_train + n_val :], y[n_train + n_val :]),
    }

    for split_name, (features, labels) in splits.items():
        np.save(output_path / f"dynamic_{split_name}_X.npy", features)
        np.save(output_path / f"dynamic_{split_name}_y.npy", labels)
        print(f"  {split_name}: {len(features)} samples saved")

    with open(output_path / "dynamic_classes.json", "w") as f:
        json.dump(class_map, f, indent=2)

    print(f"\n✅ Dynamic dataset saved to {output_path}")


# =============================================================================
# DATASET: CUSTOM RECORDINGS (from record_signs.py)
# =============================================================================

def prepare_custom(
    recordings_dir: str,
    output_dir: str,
    data_type: str = "auto",
    sequence_length: int = SEQUENCE_LENGTH,
    val_split: float = 0.15,
    test_split: float = 0.10,
):
    """
    Prepare custom recorded data.

    Expected structure (from record_signs.py):
      recordings_dir/
        static/
          A/ B/ C/ ... (directories of .jpg images per sign)
        dynamic/
          hello/ thank_you/ ... (directories of .mp4 videos per sign)
    """
    rec_path = Path(recordings_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Process static recordings
    static_dir = rec_path / "static"
    if static_dir.exists() and data_type in ("auto", "static"):
        print("Processing static recordings...")
        _process_custom_static(static_dir, output_path, val_split, test_split)

    # Process dynamic recordings
    dynamic_dir = rec_path / "dynamic"
    if dynamic_dir.exists() and data_type in ("auto", "dynamic"):
        print("Processing dynamic recordings...")
        _process_custom_dynamic(dynamic_dir, output_path, sequence_length, val_split, test_split)


def _process_custom_static(static_dir: Path, output_path: Path, val_split: float, test_split: float):
    extractor = LandmarkExtractor(static_mode=True, max_hands=1)

    class_dirs = sorted([d for d in static_dir.iterdir() if d.is_dir()])
    class_names = [d.name.upper() for d in class_dirs]

    all_features = []
    all_labels = []

    for idx, (cdir, cname) in enumerate(zip(class_dirs, class_names)):
        images = list(cdir.glob("*.jpg")) + list(cdir.glob("*.png"))
        features = []

        for img_path in tqdm(images, desc=cname, leave=False):
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            lm = extractor.extract_from_image(img)
            if lm is not None:
                features.append(lm)

        if features:
            all_features.append(np.array(features))
            all_labels.append(np.full(len(features), idx, dtype=np.int64))
            print(f"  ✅ {cname}: {len(features)} samples")

    extractor.close()

    if not all_features:
        return

    X = np.vstack(all_features)
    y = np.concatenate(all_labels)

    perm = np.random.permutation(len(X))
    X, y = X[perm], y[perm]

    n = len(X)
    n_test = int(n * test_split)
    n_val = int(n * val_split)

    # Save — prefix with 'custom_static'
    np.save(output_path / "custom_static_train_X.npy", X[: n - n_val - n_test])
    np.save(output_path / "custom_static_train_y.npy", y[: n - n_val - n_test])
    np.save(output_path / "custom_static_val_X.npy", X[n - n_val - n_test : n - n_test])
    np.save(output_path / "custom_static_val_y.npy", y[n - n_val - n_test : n - n_test])
    np.save(output_path / "custom_static_test_X.npy", X[n - n_test :])
    np.save(output_path / "custom_static_test_y.npy", y[n - n_test :])

    with open(output_path / "custom_static_classes.json", "w") as f:
        json.dump({name: idx for idx, name in enumerate(class_names)}, f, indent=2)

    print(f"✅ Custom static data: {len(X)} samples, {len(class_names)} classes")


def _process_custom_dynamic(
    dynamic_dir: Path, output_path: Path, seq_len: int, val_split: float, test_split: float
):
    extractor = LandmarkExtractor(static_mode=False, max_hands=2)

    class_dirs = sorted([d for d in dynamic_dir.iterdir() if d.is_dir()])
    class_names = [d.name.lower() for d in class_dirs]

    all_sequences = []
    all_labels = []

    for idx, (cdir, cname) in enumerate(zip(class_dirs, class_names)):
        videos = list(cdir.glob("*.mp4")) + list(cdir.glob("*.avi"))

        for vid_path in tqdm(videos, desc=cname, leave=False):
            seq = extractor.extract_sequence_from_video(str(vid_path), target_frames=seq_len)
            if seq is not None:
                all_sequences.append(seq)
                all_labels.append(idx)

        count = sum(1 for l in all_labels if l == idx)
        print(f"  ✅ {cname}: {count} sequences")

    extractor.close()

    if not all_sequences:
        return

    X = np.array(all_sequences, dtype=np.float32)
    y = np.array(all_labels, dtype=np.int64)

    perm = np.random.permutation(len(X))
    X, y = X[perm], y[perm]

    n = len(X)
    n_test = int(n * test_split)
    n_val = int(n * val_split)

    np.save(output_path / "custom_dynamic_train_X.npy", X[: n - n_val - n_test])
    np.save(output_path / "custom_dynamic_train_y.npy", y[: n - n_val - n_test])
    np.save(output_path / "custom_dynamic_val_X.npy", X[n - n_val - n_test : n - n_test])
    np.save(output_path / "custom_dynamic_val_y.npy", y[n - n_val - n_test : n - n_test])
    np.save(output_path / "custom_dynamic_test_X.npy", X[n - n_test :])
    np.save(output_path / "custom_dynamic_test_y.npy", y[n - n_test :])

    with open(output_path / "custom_dynamic_classes.json", "w") as f:
        json.dump({name: idx for idx, name in enumerate(class_names)}, f, indent=2)

    print(f"✅ Custom dynamic data: {len(X)} sequences, {len(class_names)} classes")


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="ASL Dataset Preparation")
    parser.add_argument(
        "--dataset",
        choices=["kaggle_alphabet", "wlasl", "custom"],
        required=True,
        help="Which dataset to prepare",
    )
    parser.add_argument("--output", default="./data", help="Output directory for processed data")
    parser.add_argument("--input", default=None, help="Input directory (for kaggle/custom)")
    parser.add_argument("--wlasl_json", default=None, help="Path to WLASL JSON file")
    parser.add_argument("--max_classes", type=int, default=200, help="Max classes for WLASL")
    parser.add_argument("--max_per_class", type=int, default=3000, help="Max images per class (Kaggle)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()
    np.random.seed(args.seed)

    if args.dataset == "kaggle_alphabet":
        if not args.input:
            print("❌ --input required: path to extracted Kaggle ASL Alphabet dataset")
            print("   Download: https://www.kaggle.com/datasets/grassknoted/asl-alphabet")
            sys.exit(1)
        prepare_kaggle_alphabet(args.input, args.output, max_per_class=args.max_per_class)

    elif args.dataset == "wlasl":
        if not args.wlasl_json or not args.input:
            print("❌ --wlasl_json and --input (videos dir) required")
            print("   Clone: https://github.com/dxli94/WLASL")
            sys.exit(1)
        prepare_wlasl(args.wlasl_json, args.input, args.output, max_classes=args.max_classes)

    elif args.dataset == "custom":
        if not args.input:
            print("❌ --input required: path to recordings from record_signs.py")
            sys.exit(1)
        prepare_custom(args.input, args.output)


if __name__ == "__main__":
    main()
