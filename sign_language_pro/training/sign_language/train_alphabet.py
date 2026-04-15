#!/usr/bin/env python3
"""
Train ASL Alphabet + Numbers Classifier
=========================================

Trains StaticSignNet on hand landmark data for recognizing A-Z + 0-9.

DATA SOURCES (choose one or combine):
  1. Collect your own data with collect_data.py (recommended for best accuracy)
  2. Use Kaggle ASL Alphabet dataset → extract landmarks with MediaPipe
  3. Generate synthetic landmarks from the rule-based patterns

USAGE:
  # Step 1: Collect or prepare data
  python collect_data.py --mode alphabet --output data/alphabet

  # Step 2: Train
  python train_alphabet.py --data data/alphabet --epochs 100

  # Step 3: Export model
  # (automatically saves to models/alphabet_model.pt)

Optimized for RTX 4090: mixed precision, large batches, fast convergence.
"""

from __future__ import annotations

import os
import sys
import json
import time
import random
import argparse
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.cuda.amp import GradScaler, autocast

# Import model
from models import (
    StaticSignNet,
    normalize_landmarks,
    normalize_landmarks_batch,
    ALPHABET_LABELS,
    ALPHABET_TO_IDX,
    IDX_TO_ALPHABET,
)


# =============================================================================
# DATASET
# =============================================================================

class ASLAlphabetDataset(Dataset):
    """
    Dataset of hand landmarks labeled with sign classes.

    Expected data format (in data_dir):
      landmarks/
        A/
          sample_0001.npy   # (21, 3) array
          sample_0002.npy
          ...
        B/
          ...
        0/
          ...

    OR a single file:
      alphabet_data.npz with keys 'landmarks' (N, 21, 3) and 'labels' (N,)
    """

    def __init__(
        self,
        data_dir: str,
        augment: bool = True,
        max_samples_per_class: Optional[int] = None,
    ):
        self.augment = augment
        self.samples: List[Tuple[np.ndarray, int]] = []

        data_path = Path(data_dir)

        # Check for .npz file first
        npz_file = data_path / "alphabet_data.npz"
        if npz_file.exists():
            print(f"📂 Loading from {npz_file}")
            data = np.load(str(npz_file))
            landmarks = data["landmarks"]  # (N, 21, 3)
            labels = data["labels"]        # (N,) string labels
            for lm, label in zip(landmarks, labels):
                label_str = str(label).upper()
                if label_str in ALPHABET_TO_IDX:
                    self.samples.append((lm, ALPHABET_TO_IDX[label_str]))
        else:
            # Load from directory structure
            landmarks_dir = data_path / "landmarks"
            if not landmarks_dir.exists():
                landmarks_dir = data_path  # Try data_dir directly

            for label_str in ALPHABET_LABELS:
                class_dir = landmarks_dir / label_str
                if not class_dir.exists():
                    continue

                files = sorted(class_dir.glob("*.npy"))
                if max_samples_per_class:
                    files = files[:max_samples_per_class]

                for f in files:
                    lm = np.load(str(f))
                    if lm.shape == (21, 3):
                        self.samples.append((lm, ALPHABET_TO_IDX[label_str]))

        # Class distribution
        label_counts: Dict[int, int] = {}
        for _, label in self.samples:
            label_counts[label] = label_counts.get(label, 0) + 1

        print(f"📊 Loaded {len(self.samples)} samples across {len(label_counts)} classes")
        for idx in sorted(label_counts):
            print(f"   {IDX_TO_ALPHABET[idx]}: {label_counts[idx]} samples")

        self.label_counts = label_counts

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        landmarks, label = self.samples[idx]

        if self.augment:
            landmarks = self._augment(landmarks.copy())

        # Normalize
        normalized = normalize_landmarks(landmarks)
        return torch.tensor(normalized, dtype=torch.float32), label

    def _augment(self, lm: np.ndarray) -> np.ndarray:
        """Apply data augmentation to landmarks."""

        # 1. Random rotation in x-y plane (±15 degrees)
        if random.random() < 0.5:
            angle = np.radians(random.uniform(-15, 15))
            cos_a, sin_a = np.cos(angle), np.sin(angle)
            rot = np.array([[cos_a, -sin_a, 0], [sin_a, cos_a, 0], [0, 0, 1]])
            center = lm[0].copy()
            lm = (lm - center) @ rot.T + center

        # 2. Random scale (0.85 - 1.15)
        if random.random() < 0.5:
            scale = random.uniform(0.85, 1.15)
            center = lm[0].copy()
            lm = (lm - center) * scale + center

        # 3. Random translation (small shift)
        if random.random() < 0.5:
            shift = np.random.uniform(-0.03, 0.03, size=3).astype(np.float32)
            lm += shift

        # 4. Random noise
        if random.random() < 0.3:
            noise = np.random.randn(*lm.shape).astype(np.float32) * 0.005
            lm += noise

        # 5. Mirror (simulate left/right hand) — flip x axis
        if random.random() < 0.3:
            lm[:, 0] = 1.0 - lm[:, 0]

        return lm

    def get_class_weights(self) -> torch.Tensor:
        """Get inverse frequency weights for balanced sampling."""
        total = len(self.samples)
        weights = []
        for _, label in self.samples:
            w = total / (len(self.label_counts) * self.label_counts[label])
            weights.append(w)
        return torch.tensor(weights, dtype=torch.float64)


# =============================================================================
# SYNTHETIC DATA GENERATION (for bootstrap training)
# =============================================================================

def generate_synthetic_data(output_dir: str, samples_per_class: int = 500):
    """
    Generate synthetic training data from known ASL hand configurations.
    This gives you a starting model to fine-tune with real data.
    """
    import mediapipe as mp

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"🔧 Generating {samples_per_class} synthetic samples per class...")

    # Define approximate landmark positions for each sign
    # These are rough estimates — real data will be much better
    base_hand = np.array([
        [0.5, 0.8, 0.0],    # 0: Wrist
        [0.45, 0.7, -0.02],  # 1: Thumb CMC
        [0.38, 0.6, -0.03],  # 2: Thumb MCP
        [0.33, 0.52, -0.02], # 3: Thumb IP
        [0.30, 0.45, -0.01], # 4: Thumb Tip
        [0.45, 0.48, 0.0],   # 5: Index MCP
        [0.44, 0.35, 0.0],   # 6: Index PIP
        [0.43, 0.27, 0.0],   # 7: Index DIP
        [0.42, 0.20, 0.0],   # 8: Index Tip
        [0.50, 0.46, 0.0],   # 9: Middle MCP
        [0.50, 0.32, 0.0],   # 10: Middle PIP
        [0.50, 0.24, 0.0],   # 11: Middle DIP
        [0.50, 0.18, 0.0],   # 12: Middle Tip
        [0.55, 0.48, 0.0],   # 13: Ring MCP
        [0.55, 0.35, 0.0],   # 14: Ring PIP
        [0.56, 0.27, 0.0],   # 15: Ring DIP
        [0.56, 0.22, 0.0],   # 16: Ring Tip
        [0.60, 0.52, 0.0],   # 17: Pinky MCP
        [0.61, 0.40, 0.0],   # 18: Pinky PIP
        [0.62, 0.33, 0.0],   # 19: Pinky DIP
        [0.63, 0.28, 0.0],   # 20: Pinky Tip
    ], dtype=np.float32)

    def make_fist(hand):
        """Curl all fingers into fist."""
        h = hand.copy()
        # Curl fingers: move tips close to MCP
        for tip, pip, mcp in [(8,6,5), (12,10,9), (16,14,13), (20,18,17)]:
            h[tip] = h[mcp] + (h[tip] - h[mcp]) * 0.15
            h[pip] = h[mcp] + (h[pip] - h[mcp]) * 0.3
        return h

    def extend_fingers(hand, fingers):
        """Keep specified fingers extended, curl others."""
        h = make_fist(hand)
        base = hand.copy()
        finger_map = {
            'thumb': (4, 3, 2),
            'index': (8, 7, 6, 5),
            'middle': (12, 11, 10, 9),
            'ring': (16, 15, 14, 13),
            'pinky': (20, 19, 18, 17),
        }
        for f in fingers:
            if f in finger_map:
                for idx in finger_map[f]:
                    h[idx] = base[idx]
        return h

    # Define patterns for each letter/number
    sign_generators = {
        'A': lambda h: make_fist(h),  # Fist, thumb beside
        'B': lambda h: extend_fingers(h, ['index', 'middle', 'ring', 'pinky']),
        'C': lambda h: h * np.array([1, 1, 0.7])[None, :] + 0.02,  # Curved hand
        'D': lambda h: extend_fingers(h, ['index']),
        'E': lambda h: make_fist(h) * 0.95,
        'F': lambda h: extend_fingers(h, ['middle', 'ring', 'pinky']),
        'G': lambda h: extend_fingers(h, ['thumb', 'index']),
        'H': lambda h: extend_fingers(h, ['index', 'middle']),
        'I': lambda h: extend_fingers(h, ['pinky']),
        'J': lambda h: extend_fingers(h, ['pinky']),
        'K': lambda h: extend_fingers(h, ['thumb', 'index', 'middle']),
        'L': lambda h: extend_fingers(h, ['thumb', 'index']),
        'M': lambda h: make_fist(h),
        'N': lambda h: make_fist(h),
        'O': lambda h: h * 0.9,  # All curved to thumb
        'P': lambda h: extend_fingers(h, ['thumb', 'index', 'middle']),
        'Q': lambda h: extend_fingers(h, ['thumb', 'index']),
        'R': lambda h: extend_fingers(h, ['index', 'middle']),
        'S': lambda h: make_fist(h),
        'T': lambda h: make_fist(h),
        'U': lambda h: extend_fingers(h, ['index', 'middle']),
        'V': lambda h: extend_fingers(h, ['index', 'middle']),
        'W': lambda h: extend_fingers(h, ['index', 'middle', 'ring']),
        'X': lambda h: extend_fingers(h, []),  # Hooked index
        'Y': lambda h: extend_fingers(h, ['thumb', 'pinky']),
        'Z': lambda h: extend_fingers(h, ['index']),
        '0': lambda h: h * 0.9,
        '1': lambda h: extend_fingers(h, ['index']),
        '2': lambda h: extend_fingers(h, ['index', 'middle']),
        '3': lambda h: extend_fingers(h, ['thumb', 'index', 'middle']),
        '4': lambda h: extend_fingers(h, ['index', 'middle', 'ring', 'pinky']),
        '5': lambda h: extend_fingers(h, ['thumb', 'index', 'middle', 'ring', 'pinky']),
        '6': lambda h: extend_fingers(h, ['thumb', 'pinky']),
        '7': lambda h: extend_fingers(h, ['thumb', 'ring']),
        '8': lambda h: extend_fingers(h, ['thumb', 'middle']),
        '9': lambda h: extend_fingers(h, ['thumb', 'index']),
    }

    all_landmarks = []
    all_labels = []

    for label, gen_fn in sign_generators.items():
        class_dir = output_path / "landmarks" / label
        class_dir.mkdir(parents=True, exist_ok=True)

        for i in range(samples_per_class):
            # Generate with random variation
            hand = base_hand.copy()

            # Random hand position
            hand += np.random.uniform(-0.05, 0.05, size=(21, 3)).astype(np.float32)

            # Apply sign pattern
            lm = gen_fn(hand)

            # Add noise
            lm += np.random.randn(21, 3).astype(np.float32) * 0.01

            # Random scale
            scale = random.uniform(0.8, 1.2)
            center = lm[0].copy()
            lm = (lm - center) * scale + center

            # Random rotation
            angle = np.radians(random.uniform(-20, 20))
            cos_a, sin_a = np.cos(angle), np.sin(angle)
            rot = np.array([[cos_a, -sin_a, 0], [sin_a, cos_a, 0], [0, 0, 1]])
            lm = (lm - lm[0]) @ rot.T + lm[0]

            # Save
            np.save(str(class_dir / f"sample_{i:04d}.npy"), lm.astype(np.float32))
            all_landmarks.append(lm)
            all_labels.append(label)

    # Also save as .npz for easy loading
    np.savez(
        str(output_path / "alphabet_data.npz"),
        landmarks=np.array(all_landmarks),
        labels=np.array(all_labels),
    )

    print(f"✅ Generated {len(all_landmarks)} synthetic samples")
    print(f"   Saved to: {output_path}")


# =============================================================================
# EXTRACT LANDMARKS FROM IMAGES (Kaggle ASL dataset)
# =============================================================================

def extract_from_images(image_dir: str, output_dir: str):
    """
    Extract landmarks from ASL alphabet image dataset using MediaPipe.

    Expected structure:
      image_dir/
        A/   (images of sign 'A')
        B/   (images of sign 'B')
        ...

    Popular dataset: https://www.kaggle.com/datasets/grassknoted/asl-alphabet
    """
    import cv2
    import mediapipe as mp

    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=1,
        min_detection_confidence=0.5,
    )

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    image_path = Path(image_dir)
    total = 0
    failed = 0

    for label_dir in sorted(image_path.iterdir()):
        if not label_dir.is_dir():
            continue

        label = label_dir.name.upper()
        if label not in ALPHABET_TO_IDX:
            continue

        out_class = output_path / "landmarks" / label
        out_class.mkdir(parents=True, exist_ok=True)

        images = list(label_dir.glob("*.jpg")) + list(label_dir.glob("*.png"))
        print(f"   Processing {label}: {len(images)} images...", end="", flush=True)

        count = 0
        for img_file in images:
            image = cv2.imread(str(img_file))
            if image is None:
                continue

            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)

            if results.multi_hand_landmarks:
                lm = results.multi_hand_landmarks[0]
                landmarks = np.array([[l.x, l.y, l.z] for l in lm.landmark], dtype=np.float32)
                np.save(str(out_class / f"sample_{count:05d}.npy"), landmarks)
                count += 1
                total += 1
            else:
                failed += 1

        print(f" → {count} landmarks extracted")

    hands.close()
    print(f"\n✅ Extracted {total} landmarks ({failed} images had no hand detected)")


# =============================================================================
# TRAINING LOOP
# =============================================================================

def train(
    data_dir: str,
    output_dir: str = "models",
    epochs: int = 100,
    batch_size: int = 256,       # Large batch for RTX 4090
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    patience: int = 15,
    use_amp: bool = True,        # Mixed precision
    num_workers: int = 4,
    val_split: float = 0.15,
    seed: int = 42,
):
    """Train the alphabet classifier."""

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️  Device: {device}")
    if device.type == "cuda":
        print(f"   GPU: {torch.cuda.get_device_name()}")
        print(f"   VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")

    # Load dataset
    full_dataset = ASLAlphabetDataset(data_dir, augment=True)

    if len(full_dataset) == 0:
        print("❌ No data found! Run collect_data.py or generate_synthetic_data first.")
        print("   Quick start: python train_alphabet.py --generate-synthetic --data data/alphabet")
        return

    # Split train/val
    n_val = int(len(full_dataset) * val_split)
    n_train = len(full_dataset) - n_val
    train_set, val_set = torch.utils.data.random_split(full_dataset, [n_train, n_val])

    # Disable augmentation for validation
    val_set.dataset = ASLAlphabetDataset(data_dir, augment=False)

    # Balanced sampling
    train_weights = full_dataset.get_class_weights()
    train_indices = train_set.indices
    sampler_weights = train_weights[train_indices]
    sampler = WeightedRandomSampler(sampler_weights, len(train_indices), replacement=True)

    train_loader = DataLoader(
        train_set, batch_size=batch_size, sampler=sampler,
        num_workers=num_workers, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_set, batch_size=batch_size * 2, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )

    # Model
    num_classes = len(full_dataset.label_counts)
    model = StaticSignNet(num_classes=num_classes).to(device)
    print(f"📐 Model: StaticSignNet ({sum(p.numel() for p in model.parameters()):,} params)")
    print(f"   Classes: {num_classes}")

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    scaler = GradScaler(enabled=use_amp)

    # Training
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    best_val_acc = 0.0
    patience_counter = 0
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    print(f"\n🏋️ Training for {epochs} epochs...")
    print(f"   Batch size: {batch_size}, LR: {learning_rate}")
    print(f"   Train: {n_train}, Val: {n_val}")
    print("-" * 60)

    for epoch in range(1, epochs + 1):
        # --- Train ---
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0

        for landmarks, labels in train_loader:
            landmarks = landmarks.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with autocast(enabled=use_amp, device_type=device.type):
                logits = model(landmarks)
                loss = criterion(logits, labels)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item() * labels.size(0)
            train_correct += (logits.argmax(1) == labels).sum().item()
            train_total += labels.size(0)

        scheduler.step()

        avg_train_loss = train_loss / train_total
        train_acc = train_correct / train_total

        # --- Validate ---
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for landmarks, labels in val_loader:
                landmarks = landmarks.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)

                with autocast(enabled=use_amp, device_type=device.type):
                    logits = model(landmarks)
                    loss = criterion(logits, labels)

                val_loss += loss.item() * labels.size(0)
                val_correct += (logits.argmax(1) == labels).sum().item()
                val_total += labels.size(0)

        avg_val_loss = val_loss / max(val_total, 1)
        val_acc = val_correct / max(val_total, 1)

        # Log
        history["train_loss"].append(avg_train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(avg_val_loss)
        history["val_acc"].append(val_acc)

        lr = optimizer.param_groups[0]["lr"]
        print(
            f"  Epoch {epoch:3d}/{epochs} | "
            f"Train {train_acc:.1%} (loss {avg_train_loss:.4f}) | "
            f"Val {val_acc:.1%} (loss {avg_val_loss:.4f}) | "
            f"LR {lr:.2e}"
        )

        # Save best
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            save_path = output_path / "alphabet_model.pt"
            torch.save({
                "model_state_dict": model.state_dict(),
                "num_classes": num_classes,
                "val_acc": val_acc,
                "epoch": epoch,
                "label_map": IDX_TO_ALPHABET,
            }, str(save_path))
            print(f"  💾 New best! Saved → {save_path} ({val_acc:.1%})")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\n⏹️ Early stopping at epoch {epoch} (no improvement for {patience} epochs)")
                break

    print(f"\n✅ Training complete! Best validation accuracy: {best_val_acc:.1%}")
    print(f"   Model saved to: {output_path / 'alphabet_model.pt'}")

    # Save training history
    with open(str(output_path / "alphabet_history.json"), "w") as f:
        json.dump(history, f, indent=2)


# =============================================================================
# EVALUATION
# =============================================================================

def evaluate(model_path: str, data_dir: str):
    """Evaluate a trained model and print confusion matrix."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    checkpoint = torch.load(model_path, map_location=device, weights_only=True)
    num_classes = checkpoint.get("num_classes", 36)
    label_map = checkpoint.get("label_map", IDX_TO_ALPHABET)

    model = StaticSignNet(num_classes=num_classes).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # Load data
    dataset = ASLAlphabetDataset(data_dir, augment=False)
    loader = DataLoader(dataset, batch_size=512, shuffle=False, num_workers=4)

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for landmarks, labels in loader:
            landmarks = landmarks.to(device)
            logits = model(landmarks)
            preds = logits.argmax(1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    accuracy = (all_preds == all_labels).mean()
    print(f"\n📊 Overall Accuracy: {accuracy:.1%}")

    # Per-class accuracy
    print(f"\n{'Class':>6} {'Correct':>8} {'Total':>6} {'Acc':>6}")
    print("-" * 30)
    for idx in sorted(set(all_labels)):
        mask = all_labels == idx
        class_acc = (all_preds[mask] == idx).mean()
        total = mask.sum()
        correct = (all_preds[mask] == idx).sum()
        label = label_map.get(idx, str(idx))
        print(f"{label:>6} {correct:>8} {total:>6} {class_acc:>6.1%}")

    # Most confused pairs
    print(f"\n🔀 Most Confused Pairs:")
    from collections import Counter
    confusions = Counter()
    for pred, true in zip(all_preds, all_labels):
        if pred != true:
            pair = (label_map.get(int(true), str(true)), label_map.get(int(pred), str(pred)))
            confusions[pair] += 1

    for (true_l, pred_l), count in confusions.most_common(10):
        print(f"   {true_l} → {pred_l}: {count} times")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Train ASL Alphabet Classifier")
    parser.add_argument("--data", type=str, default="data/alphabet", help="Data directory")
    parser.add_argument("--output", type=str, default="models", help="Output directory for model")
    parser.add_argument("--epochs", type=int, default=100, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=256, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--patience", type=int, default=15, help="Early stopping patience")
    parser.add_argument("--generate-synthetic", action="store_true", help="Generate synthetic data first")
    parser.add_argument("--extract-images", type=str, default=None, help="Path to ASL image dataset")
    parser.add_argument("--evaluate", type=str, default=None, help="Path to model to evaluate")
    parser.add_argument("--samples-per-class", type=int, default=500, help="Synthetic samples per class")

    args = parser.parse_args()

    if args.generate_synthetic:
        generate_synthetic_data(args.data, args.samples_per_class)

    if args.extract_images:
        extract_from_images(args.extract_images, args.data)

    if args.evaluate:
        evaluate(args.evaluate, args.data)
        return

    train(
        data_dir=args.data,
        output_dir=args.output,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        patience=args.patience,
    )


if __name__ == "__main__":
    main()
