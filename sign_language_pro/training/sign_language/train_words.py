#!/usr/bin/env python3
"""
Train Word-Level ASL Classifier (WLASL)
=========================================

Trains DynamicSignNet on temporal landmark sequences for recognizing
ASL words and phrases.

DATA SOURCE:
  WLASL dataset: https://dxli94.github.io/WLASL/
  - 2000 words, 21,000+ videos
  - Download videos and extract landmarks with this script

USAGE:
  # Step 1: Download WLASL
  python train_words.py --download-wlasl --wlasl-dir data/wlasl

  # Step 2: Extract landmarks from videos
  python train_words.py --extract --wlasl-dir data/wlasl --output data/words

  # Step 3: Train
  python train_words.py --data data/words --epochs 200 --num-classes 100

  # For full 2000-word vocabulary:
  python train_words.py --data data/words --epochs 300 --num-classes 2000

Optimized for RTX 4090 with mixed precision and large batches.
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
from collections import Counter

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.cuda.amp import GradScaler, autocast

from models import (
    DynamicSignNet,
    TwoHandFusion,
    normalize_landmarks,
)


# =============================================================================
# DATASET
# =============================================================================

class WLASLDataset(Dataset):
    """
    Dataset of temporal landmark sequences for word-level ASL.

    Expected format:
      data_dir/
        sequences/
          hello/
            video_0001.npz  # keys: 'left_hand', 'right_hand', 'pose' (each T×N×3)
            ...
          goodbye/
            ...
        vocab.txt           # one word per line

    OR single file:
      word_data.npz with keys:
        'sequences' (N, max_T, input_dim)
        'lengths'   (N,) actual lengths
        'labels'    (N,) integer labels
        'vocab'     list of word strings
    """

    def __init__(
        self,
        data_dir: str,
        max_seq_len: int = 64,
        augment: bool = True,
        num_classes: Optional[int] = None,  # Limit to top N classes
        min_samples: int = 5,               # Minimum samples per class
    ):
        self.max_seq_len = max_seq_len
        self.augment = augment
        self.samples: List[Tuple[np.ndarray, int, int]] = []  # (sequence, label, length)

        data_path = Path(data_dir)

        # Try loading from .npz
        npz_file = data_path / "word_data.npz"
        if npz_file.exists():
            print(f"📂 Loading from {npz_file}")
            data = np.load(str(npz_file), allow_pickle=True)
            sequences = data["sequences"]
            lengths = data["lengths"]
            labels = data["labels"]
            self.vocab = list(data.get("vocab", []))

            for seq, length, label in zip(sequences, lengths, labels):
                self.samples.append((seq, int(label), int(length)))
        else:
            # Load from directory structure
            seq_dir = data_path / "sequences"
            vocab_file = data_path / "vocab.txt"

            if vocab_file.exists():
                with open(str(vocab_file)) as f:
                    self.vocab = [line.strip() for line in f if line.strip()]
            else:
                self.vocab = sorted([d.name for d in seq_dir.iterdir() if d.is_dir()])

            word_to_idx = {w: i for i, w in enumerate(self.vocab)}

            for word in self.vocab:
                word_dir = seq_dir / word
                if not word_dir.exists():
                    continue

                idx = word_to_idx[word]
                for f in sorted(word_dir.glob("*.npz")):
                    data = np.load(str(f))

                    # Combine hands into single feature vector
                    left = data.get("left_hand", np.zeros((1, 21, 3)))
                    right = data.get("right_hand", np.zeros((1, 21, 3)))

                    T = max(left.shape[0], right.shape[0])

                    # Pad shorter hand sequence
                    if left.shape[0] < T:
                        left = np.pad(left, ((0, T - left.shape[0]), (0, 0), (0, 0)))
                    if right.shape[0] < T:
                        right = np.pad(right, ((0, T - right.shape[0]), (0, 0), (0, 0)))

                    # Flatten and concatenate: (T, 126) = both hands × 21 landmarks × 3 coords
                    left_flat = left.reshape(T, -1)    # (T, 63)
                    right_flat = right.reshape(T, -1)  # (T, 63)
                    combined = np.concatenate([left_flat, right_flat], axis=1)  # (T, 126)

                    self.samples.append((combined, idx, T))

        # Filter by minimum samples
        label_counts = Counter(label for _, label, _ in self.samples)
        valid_labels = {l for l, c in label_counts.items() if c >= min_samples}

        if num_classes and len(valid_labels) > num_classes:
            # Keep top N classes by sample count
            top_labels = set(
                l for l, _ in label_counts.most_common(num_classes) if l in valid_labels
            )
            valid_labels = top_labels

        # Re-map labels to be contiguous
        old_to_new = {}
        new_vocab = []
        for old_label in sorted(valid_labels):
            new_label = len(new_vocab)
            old_to_new[old_label] = new_label
            if old_label < len(self.vocab):
                new_vocab.append(self.vocab[old_label])

        self.samples = [
            (seq, old_to_new[label], length)
            for seq, label, length in self.samples
            if label in old_to_new
        ]
        self.vocab = new_vocab
        self.num_classes = len(new_vocab)

        # Recalculate
        self.label_counts = Counter(label for _, label, _ in self.samples)

        print(f"📊 Loaded {len(self.samples)} sequences, {self.num_classes} classes")
        print(f"   Avg sequence length: {np.mean([l for _, _, l in self.samples]):.1f} frames")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, int]:
        sequence, label, length = self.samples[idx]

        if self.augment:
            sequence = self._augment(sequence.copy())

        # Pad or truncate to max_seq_len
        T = sequence.shape[0]
        if T > self.max_seq_len:
            # Uniform subsample
            indices = np.linspace(0, T - 1, self.max_seq_len, dtype=int)
            sequence = sequence[indices]
            length = self.max_seq_len
        elif T < self.max_seq_len:
            pad = np.zeros((self.max_seq_len - T, sequence.shape[1]), dtype=np.float32)
            sequence = np.concatenate([sequence, pad], axis=0)

        return (
            torch.tensor(sequence, dtype=torch.float32),
            label,
            min(length, self.max_seq_len),
        )

    def _augment(self, seq: np.ndarray) -> np.ndarray:
        """Augment a landmark sequence."""
        T, D = seq.shape

        # 1. Temporal scaling (speed up/slow down)
        if random.random() < 0.5:
            scale = random.uniform(0.8, 1.2)
            new_T = max(3, int(T * scale))
            indices = np.linspace(0, T - 1, new_T).astype(int)
            seq = seq[indices]

        # 2. Random noise
        if random.random() < 0.5:
            noise = np.random.randn(*seq.shape).astype(np.float32) * 0.005
            seq += noise

        # 3. Random frame drop (simulate occlusion)
        if random.random() < 0.3 and len(seq) > 5:
            n_drop = random.randint(1, max(1, len(seq) // 10))
            drop_indices = random.sample(range(len(seq)), n_drop)
            mask = np.ones(len(seq), dtype=bool)
            mask[drop_indices] = False
            seq = seq[mask]

        # 4. Spatial jitter
        if random.random() < 0.5:
            shift = np.random.uniform(-0.02, 0.02, size=(1, D)).astype(np.float32)
            seq += shift

        return seq

    def get_class_weights(self) -> torch.Tensor:
        """Inverse frequency weights for balanced training."""
        total = len(self.samples)
        weights = []
        for _, label, _ in self.samples:
            w = total / (self.num_classes * self.label_counts[label])
            weights.append(w)
        return torch.tensor(weights, dtype=torch.float64)


# =============================================================================
# WLASL DOWNLOAD & EXTRACTION
# =============================================================================

def download_wlasl(output_dir: str):
    """
    Download WLASL dataset metadata.
    Videos need to be downloaded separately (many are YouTube links).
    """
    import urllib.request

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # WLASL JSON with video info
    url = "https://raw.githubusercontent.com/dxli94/WLASL/master/start_kit/WLASL_v0.3.json"
    json_path = output_path / "WLASL_v0.3.json"

    if not json_path.exists():
        print(f"📥 Downloading WLASL metadata...")
        urllib.request.urlretrieve(url, str(json_path))
        print(f"   Saved to {json_path}")
    else:
        print(f"   WLASL metadata already exists at {json_path}")

    with open(str(json_path)) as f:
        wlasl_data = json.load(f)

    print(f"   Found {len(wlasl_data)} glosses (words)")

    # Save vocab
    vocab = [entry["gloss"] for entry in wlasl_data]
    with open(str(output_path / "vocab.txt"), "w") as f:
        for w in vocab:
            f.write(f"{w}\n")

    print(f"   Vocab saved to {output_path / 'vocab.txt'}")

    # Print download instructions
    print(f"""
╔══════════════════════════════════════════════════════════════╗
║  WLASL VIDEO DOWNLOAD INSTRUCTIONS                          ║
╠══════════════════════════════════════════════════════════════╣
║                                                              ║
║  WLASL videos are hosted on YouTube and other platforms.     ║
║  You have two options:                                       ║
║                                                              ║
║  OPTION A: Use the WLASL download script                     ║
║    git clone https://github.com/dxli94/WLASL.git             ║
║    cd WLASL/start_kit                                        ║
║    python video_downloader.py                                ║
║                                                              ║
║  OPTION B: Use yt-dlp for YouTube videos                     ║
║    pip install yt-dlp                                        ║
║    (then use extract_from_videos to process them)            ║
║                                                              ║
║  OPTION C: Use your own sign language videos                 ║
║    Record signs with collect_data.py instead                 ║
║                                                              ║
║  After downloading, run:                                     ║
║    python train_words.py --extract \\                         ║
║      --video-dir {output_dir}/videos \\                       ║
║      --output data/words                                     ║
╚══════════════════════════════════════════════════════════════╝
""")


def extract_from_videos(
    video_dir: str,
    output_dir: str,
    wlasl_json: Optional[str] = None,
    max_videos_per_word: int = 50,
):
    """
    Extract hand landmark sequences from sign language videos.

    Expected structure:
      video_dir/
        hello/
          video1.mp4
          ...
        goodbye/
          ...

    OR with WLASL JSON mapping.
    """
    import cv2
    import mediapipe as mp

    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.4,
    )

    output_path = Path(output_dir) / "sequences"
    output_path.mkdir(parents=True, exist_ok=True)

    video_path = Path(video_dir)
    total_extracted = 0
    total_failed = 0

    # Get word directories
    if wlasl_json:
        with open(wlasl_json) as f:
            wlasl_data = json.load(f)
        words = [(entry["gloss"], video_path / entry["gloss"]) for entry in wlasl_data]
    else:
        words = [(d.name, d) for d in sorted(video_path.iterdir()) if d.is_dir()]

    vocab = []

    for word, word_dir in words:
        if not word_dir.exists():
            continue

        videos = list(word_dir.glob("*.mp4")) + list(word_dir.glob("*.avi")) + list(word_dir.glob("*.webm"))
        if not videos:
            continue

        videos = videos[:max_videos_per_word]
        out_word = output_path / word
        out_word.mkdir(parents=True, exist_ok=True)

        count = 0
        for vid_file in videos:
            cap = cv2.VideoCapture(str(vid_file))
            if not cap.isOpened():
                total_failed += 1
                continue

            left_seq = []
            right_seq = []

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = hands.process(rgb)

                left_lm = np.zeros((21, 3), dtype=np.float32)
                right_lm = np.zeros((21, 3), dtype=np.float32)

                if results.multi_hand_landmarks and results.multi_handedness:
                    for hl, hi in zip(results.multi_hand_landmarks, results.multi_handedness):
                        lm = np.array([[l.x, l.y, l.z] for l in hl.landmark], dtype=np.float32)
                        if hi.classification[0].label.lower() == "left":
                            left_lm = lm
                        else:
                            right_lm = lm

                left_seq.append(left_lm)
                right_seq.append(right_lm)

            cap.release()

            if len(left_seq) >= 3:  # Minimum 3 frames
                np.savez(
                    str(out_word / f"video_{count:04d}.npz"),
                    left_hand=np.array(left_seq, dtype=np.float32),
                    right_hand=np.array(right_seq, dtype=np.float32),
                )
                count += 1
                total_extracted += 1

        if count > 0:
            vocab.append(word)
            print(f"   {word}: {count}/{len(videos)} videos extracted")

    hands.close()

    # Save vocab
    with open(str(Path(output_dir) / "vocab.txt"), "w") as f:
        for w in vocab:
            f.write(f"{w}\n")

    print(f"\n✅ Extracted {total_extracted} sequences for {len(vocab)} words")
    print(f"   Failed: {total_failed} videos")


# =============================================================================
# GENERATE SYNTHETIC WORD DATA (for bootstrapping)
# =============================================================================

def generate_synthetic_words(output_dir: str, words: Optional[List[str]] = None, samples_per_word: int = 50):
    """Generate synthetic word-level sequences for bootstrapping."""

    if words is None:
        words = [
            "hello", "goodbye", "please", "thank_you", "sorry",
            "yes", "no", "help", "stop", "more",
            "want", "need", "like", "understand", "again",
            "good", "bad", "big", "small", "wait",
            "what", "where", "who", "when", "why", "how",
            "sign", "speak", "listen", "repeat",
            "emergency", "danger", "fast", "slow",
        ]

    output_path = Path(output_dir)
    seq_dir = output_path / "sequences"

    all_sequences = []
    all_labels = []
    all_lengths = []

    for word_idx, word in enumerate(words):
        word_dir = seq_dir / word
        word_dir.mkdir(parents=True, exist_ok=True)

        for i in range(samples_per_word):
            # Generate random sequence length (10-50 frames)
            T = random.randint(10, 50)

            # Create base hand landmarks with some structure per word
            np.random.seed(word_idx * 10000 + i)  # Reproducible per word
            base_left = np.random.randn(21, 3).astype(np.float32) * 0.1
            base_right = np.random.randn(21, 3).astype(np.float32) * 0.1

            # Create temporal trajectory
            left_seq = []
            right_seq = []
            for t in range(T):
                phase = t / T
                # Smooth trajectory with word-specific pattern
                motion = np.sin(phase * np.pi * (word_idx % 5 + 1)) * 0.05
                noise = np.random.randn(21, 3).astype(np.float32) * 0.01

                left_frame = base_left + motion + noise
                right_frame = base_right - motion + noise
                left_seq.append(left_frame)
                right_seq.append(right_frame)

            left_arr = np.array(left_seq, dtype=np.float32)
            right_arr = np.array(right_seq, dtype=np.float32)

            np.savez(
                str(word_dir / f"video_{i:04d}.npz"),
                left_hand=left_arr,
                right_hand=right_arr,
            )

            # Combined for .npz format
            combined = np.concatenate([
                left_arr.reshape(T, -1),
                right_arr.reshape(T, -1),
            ], axis=1)  # (T, 126)

            all_sequences.append(combined)
            all_labels.append(word_idx)
            all_lengths.append(T)

    # Save vocab
    with open(str(output_path / "vocab.txt"), "w") as f:
        for w in words:
            f.write(f"{w}\n")

    # Pad all to same length
    max_len = max(all_lengths)
    padded = np.zeros((len(all_sequences), max_len, 126), dtype=np.float32)
    for i, seq in enumerate(all_sequences):
        padded[i, :seq.shape[0]] = seq

    np.savez(
        str(output_path / "word_data.npz"),
        sequences=padded,
        labels=np.array(all_labels),
        lengths=np.array(all_lengths),
        vocab=np.array(words),
    )

    print(f"✅ Generated {len(all_sequences)} synthetic word sequences for {len(words)} words")


# =============================================================================
# TRAINING LOOP
# =============================================================================

def train(
    data_dir: str,
    output_dir: str = "models",
    epochs: int = 200,
    batch_size: int = 64,          # Smaller than alphabet (sequences are larger)
    learning_rate: float = 5e-4,
    weight_decay: float = 1e-4,
    patience: int = 25,
    use_amp: bool = True,
    num_workers: int = 4,
    val_split: float = 0.15,
    num_classes: Optional[int] = None,
    max_seq_len: int = 64,
    hidden_dim: int = 256,
    seed: int = 42,
):
    """Train the word-level classifier."""

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️  Device: {device}")
    if device.type == "cuda":
        print(f"   GPU: {torch.cuda.get_device_name()}")
        print(f"   VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")

    # Load dataset
    full_dataset = WLASLDataset(
        data_dir,
        max_seq_len=max_seq_len,
        augment=True,
        num_classes=num_classes,
    )

    if len(full_dataset) == 0:
        print("❌ No data found!")
        print("   Options:")
        print("   1. python train_words.py --generate-synthetic --data data/words")
        print("   2. python collect_data.py --mode words --output data/words")
        print("   3. python train_words.py --download-wlasl --wlasl-dir data/wlasl")
        return

    actual_classes = full_dataset.num_classes
    vocab = full_dataset.vocab

    # Split
    n_val = int(len(full_dataset) * val_split)
    n_train = len(full_dataset) - n_val
    train_set, val_set = torch.utils.data.random_split(full_dataset, [n_train, n_val])

    # Collate with lengths
    def collate_fn(batch):
        seqs, labels, lengths = zip(*batch)
        return torch.stack(seqs), torch.tensor(labels), torch.tensor(lengths)

    # Balanced sampling
    train_weights = full_dataset.get_class_weights()
    train_indices = train_set.indices
    sampler_weights = train_weights[train_indices]
    sampler = WeightedRandomSampler(sampler_weights, len(train_indices), replacement=True)

    train_loader = DataLoader(
        train_set, batch_size=batch_size, sampler=sampler,
        num_workers=num_workers, pin_memory=True, collate_fn=collate_fn, drop_last=True,
    )
    val_loader = DataLoader(
        val_set, batch_size=batch_size * 2, shuffle=False,
        num_workers=num_workers, pin_memory=True, collate_fn=collate_fn,
    )

    # Model
    model = DynamicSignNet(
        num_classes=actual_classes,
        input_dim=126,  # Both hands
        hidden_dim=hidden_dim,
        max_seq_len=max_seq_len,
    ).to(device)

    print(f"📐 Model: DynamicSignNet ({sum(p.numel() for p in model.parameters()):,} params)")
    print(f"   Classes: {actual_classes}, Hidden: {hidden_dim}, Max seq: {max_seq_len}")

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=20, T_mult=2)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    scaler = GradScaler(enabled=use_amp)

    # Output
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    best_val_acc = 0.0
    patience_counter = 0

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

        for seqs, labels, lengths in train_loader:
            seqs = seqs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            lengths = lengths.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with autocast(enabled=use_amp, device_type=device.type):
                logits = model(seqs, lengths)
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
        top5_correct = 0

        with torch.no_grad():
            for seqs, labels, lengths in val_loader:
                seqs = seqs.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                lengths = lengths.to(device, non_blocking=True)

                with autocast(enabled=use_amp, device_type=device.type):
                    logits = model(seqs, lengths)
                    loss = criterion(logits, labels)

                val_loss += loss.item() * labels.size(0)
                val_correct += (logits.argmax(1) == labels).sum().item()
                val_total += labels.size(0)

                # Top-5 accuracy
                _, top5 = logits.topk(min(5, actual_classes), dim=1)
                top5_correct += sum(labels[i] in top5[i] for i in range(labels.size(0)))

        avg_val_loss = val_loss / max(val_total, 1)
        val_acc = val_correct / max(val_total, 1)
        top5_acc = top5_correct / max(val_total, 1)

        lr = optimizer.param_groups[0]["lr"]
        print(
            f"  Epoch {epoch:3d}/{epochs} | "
            f"Train {train_acc:.1%} | "
            f"Val {val_acc:.1%} (top5: {top5_acc:.1%}) | "
            f"Loss {avg_val_loss:.4f} | LR {lr:.2e}"
        )

        # Save best
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            save_path = output_path / "word_model.pt"
            torch.save({
                "model_state_dict": model.state_dict(),
                "num_classes": actual_classes,
                "hidden_dim": hidden_dim,
                "max_seq_len": max_seq_len,
                "val_acc": val_acc,
                "top5_acc": top5_acc,
                "epoch": epoch,
                "vocab": vocab,
            }, str(save_path))
            print(f"  💾 New best! Saved → {save_path} ({val_acc:.1%}, top5: {top5_acc:.1%})")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\n⏹️ Early stopping at epoch {epoch}")
                break

    print(f"\n✅ Training complete! Best val accuracy: {best_val_acc:.1%}")
    print(f"   Model saved to: {output_path / 'word_model.pt'}")

    # Save vocab separately
    with open(str(output_path / "word_vocab.txt"), "w") as f:
        for w in vocab:
            f.write(f"{w}\n")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Train Word-Level ASL Classifier")
    parser.add_argument("--data", type=str, default="data/words", help="Data directory")
    parser.add_argument("--output", type=str, default="models", help="Output model directory")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--patience", type=int, default=25)
    parser.add_argument("--num-classes", type=int, default=None, help="Limit to top N classes")
    parser.add_argument("--max-seq-len", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=256)

    parser.add_argument("--download-wlasl", action="store_true")
    parser.add_argument("--wlasl-dir", type=str, default="data/wlasl")
    parser.add_argument("--extract", action="store_true")
    parser.add_argument("--video-dir", type=str, default=None)
    parser.add_argument("--generate-synthetic", action="store_true")

    args = parser.parse_args()

    if args.download_wlasl:
        download_wlasl(args.wlasl_dir)
        return

    if args.extract and args.video_dir:
        extract_from_videos(args.video_dir, args.data)
        return

    if args.generate_synthetic:
        generate_synthetic_words(args.data)

    train(
        data_dir=args.data,
        output_dir=args.output,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        patience=args.patience,
        num_classes=args.num_classes,
        max_seq_len=args.max_seq_len,
        hidden_dim=args.hidden_dim,
    )


if __name__ == "__main__":
    main()
