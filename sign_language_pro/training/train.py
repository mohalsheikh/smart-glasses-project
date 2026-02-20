#!/usr/bin/env python3
"""
ASL Sign Language Model Training
==================================

Train static (alphabet) and dynamic (word) sign recognition models.

Usage:
  # Train static model (alphabet A-Z + numbers):
  python train.py --mode static --data ./data --epochs 100 --batch_size 128

  # Train dynamic model (word-level signs):
  python train.py --mode dynamic --data ./data --epochs 80 --batch_size 64

  # Train both:
  python train.py --mode both --data ./data

  # Train on custom data:
  python train.py --mode static --data ./data --data_prefix custom_static

  # Export to ONNX after training:
  python train.py --mode static --data ./data --export_onnx

Hardware:
  Best on RTX 4090 or similar. CPU training works but is slower.
"""

from __future__ import annotations

import os
import sys
import json
import time
import argparse
from pathlib import Path
from typing import Optional, Dict

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, OneCycleLR

# Local imports
from models import (
    StaticSignNet, DynamicSignNet,
    StaticSignNetLite, DynamicSignNetLite,
    export_static_onnx, export_dynamic_onnx,
    print_model_info,
)
from augmentation import LandmarkAugmentor


# =============================================================================
# DATASETS
# =============================================================================

class StaticSignDataset(Dataset):
    """Dataset for static sign landmarks."""

    def __init__(self, X: np.ndarray, y: np.ndarray, augment: bool = False):
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y)
        self.augment = augment
        self.augmentor = LandmarkAugmentor() if augment else None

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = self.X[idx].numpy()
        y = self.y[idx]

        if self.augment and self.augmentor:
            x = self.augmentor.augment_static(x)

        return torch.FloatTensor(x), y


class DynamicSignDataset(Dataset):
    """Dataset for dynamic sign sequences."""

    def __init__(self, X: np.ndarray, y: np.ndarray, augment: bool = False):
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y)
        self.augment = augment
        self.augmentor = LandmarkAugmentor() if augment else None

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = self.X[idx].numpy()
        y = self.y[idx]

        if self.augment and self.augmentor:
            x = self.augmentor.augment_sequence(x)

        return torch.FloatTensor(x), y


# =============================================================================
# TRAINING FUNCTIONS
# =============================================================================

class Trainer:
    """Unified trainer for both static and dynamic models."""

    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        output_dir: str,
        model_name: str = "model",
    ):
        self.model = model.to(device)
        self.device = device
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.model_name = model_name
        self.best_val_acc = 0.0
        self.history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int = 100,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        label_smoothing: float = 0.1,
        patience: int = 15,
    ):
        optimizer = optim.AdamW(
            self.model.parameters(), lr=lr, weight_decay=weight_decay
        )

        scheduler = OneCycleLR(
            optimizer,
            max_lr=lr,
            epochs=epochs,
            steps_per_epoch=len(train_loader),
            pct_start=0.1,
            anneal_strategy='cos',
        )

        criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

        no_improve = 0
        start_time = time.time()

        print(f"\n{'='*60}")
        print(f"Training {self.model_name}")
        print(f"  Device: {self.device}")
        print(f"  Epochs: {epochs}")
        print(f"  LR: {lr}")
        print(f"  Patience: {patience}")
        print(f"{'='*60}\n")

        for epoch in range(1, epochs + 1):
            # Train
            train_loss, train_acc = self._train_epoch(
                train_loader, optimizer, criterion, scheduler
            )

            # Validate
            val_loss, val_acc = self._eval_epoch(val_loader, criterion)

            # Log
            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["train_acc"].append(train_acc)
            self.history["val_acc"].append(val_acc)

            elapsed = time.time() - start_time
            lr_now = optimizer.param_groups[0]['lr']

            print(
                f"Epoch {epoch:3d}/{epochs} | "
                f"Train: {train_loss:.4f} ({train_acc:.1f}%) | "
                f"Val: {val_loss:.4f} ({val_acc:.1f}%) | "
                f"LR: {lr_now:.6f} | "
                f"Time: {elapsed:.0f}s"
            )

            # Save best
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self._save_checkpoint(epoch, val_acc, is_best=True)
                no_improve = 0
                print(f"  ✅ New best: {val_acc:.2f}%")
            else:
                no_improve += 1

            # Early stopping
            if no_improve >= patience:
                print(f"\n⚠️ Early stopping at epoch {epoch} (no improvement for {patience} epochs)")
                break

            # Periodic save
            if epoch % 10 == 0:
                self._save_checkpoint(epoch, val_acc)

        print(f"\n{'='*60}")
        print(f"Training complete! Best val accuracy: {self.best_val_acc:.2f}%")
        print(f"{'='*60}")

        # Save training history
        with open(self.output_dir / f"{self.model_name}_history.json", "w") as f:
            json.dump(self.history, f, indent=2)

    def _train_epoch(self, loader, optimizer, criterion, scheduler):
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0

        for x, y in loader:
            x, y = x.to(self.device), y.to(self.device)

            optimizer.zero_grad()
            logits = self.model(x)
            loss = criterion(logits, y)
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            optimizer.step()
            scheduler.step()

            total_loss += loss.item() * len(x)
            preds = logits.argmax(dim=-1)
            correct += (preds == y).sum().item()
            total += len(x)

        return total_loss / total, 100.0 * correct / total

    @torch.no_grad()
    def _eval_epoch(self, loader, criterion):
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0

        for x, y in loader:
            x, y = x.to(self.device), y.to(self.device)
            logits = self.model(x)
            loss = criterion(logits, y)

            total_loss += loss.item() * len(x)
            preds = logits.argmax(dim=-1)
            correct += (preds == y).sum().item()
            total += len(x)

        return total_loss / total, 100.0 * correct / total

    def test(self, test_loader: DataLoader, class_names: Optional[Dict] = None):
        """Run full evaluation on test set."""
        self.model.eval()
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for x, y in test_loader:
                x = x.to(self.device)
                logits = self.model(x)
                preds = logits.argmax(dim=-1).cpu()
                all_preds.append(preds)
                all_labels.append(y)

        preds = torch.cat(all_preds).numpy()
        labels = torch.cat(all_labels).numpy()
        acc = 100.0 * (preds == labels).mean()

        print(f"\n📊 Test Results:")
        print(f"   Accuracy: {acc:.2f}%")
        print(f"   Samples: {len(labels)}")

        # Per-class accuracy
        if class_names:
            idx_to_name = {v: k for k, v in class_names.items()}
            unique_labels = np.unique(labels)

            print(f"\n   Per-class accuracy:")
            worst = []
            for label in unique_labels:
                mask = labels == label
                class_acc = 100.0 * (preds[mask] == labels[mask]).mean()
                name = idx_to_name.get(label, str(label))
                worst.append((name, class_acc))
                if class_acc < 90:
                    print(f"     {name}: {class_acc:.1f}% ⚠️")

            worst.sort(key=lambda x: x[1])
            if worst:
                print(f"\n   Worst 5 classes:")
                for name, class_acc in worst[:5]:
                    print(f"     {name}: {class_acc:.1f}%")

        return acc

    def _save_checkpoint(self, epoch: int, val_acc: float, is_best: bool = False):
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "val_acc": val_acc,
        }

        if is_best:
            path = self.output_dir / f"{self.model_name}_best.pth"
        else:
            path = self.output_dir / f"{self.model_name}_epoch{epoch}.pth"

        torch.save(checkpoint, path)

    def load_best(self):
        path = self.output_dir / f"{self.model_name}_best.pth"
        if path.exists():
            checkpoint = torch.load(path, map_location=self.device)
            self.model.load_state_dict(checkpoint["model_state_dict"])
            print(f"✅ Loaded best model (epoch {checkpoint['epoch']}, acc {checkpoint['val_acc']:.2f}%)")


# =============================================================================
# MAIN TRAINING PIPELINES
# =============================================================================

def train_static(args):
    """Train static sign classifier."""
    data_dir = Path(args.data)
    prefix = args.data_prefix or "static"

    # Load data
    print("Loading static sign data...")
    X_train = np.load(data_dir / f"{prefix}_train_X.npy")
    y_train = np.load(data_dir / f"{prefix}_train_y.npy")
    X_val = np.load(data_dir / f"{prefix}_val_X.npy")
    y_val = np.load(data_dir / f"{prefix}_val_y.npy")

    # Load class names
    classes_file = data_dir / f"{prefix}_classes.json"
    class_names = json.load(open(classes_file)) if classes_file.exists() else None
    num_classes = len(class_names) if class_names else int(y_train.max()) + 1

    print(f"  Train: {len(X_train)} | Val: {len(X_val)}")
    print(f"  Classes: {num_classes}")
    print(f"  Feature dim: {X_train.shape[1]}")

    # Create datasets
    train_ds = StaticSignDataset(X_train, y_train, augment=True)
    val_ds = StaticSignDataset(X_val, y_val, augment=False)

    # Weighted sampling for class balance
    class_counts = np.bincount(y_train, minlength=num_classes).astype(float)
    class_weights = 1.0 / (class_counts + 1)
    sample_weights = class_weights[y_train]
    sampler = WeightedRandomSampler(
        weights=sample_weights, num_samples=len(train_ds), replacement=True
    )

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, sampler=sampler,
        num_workers=args.workers, pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size * 2, shuffle=False,
        num_workers=args.workers, pin_memory=True,
    )

    # Create model
    if args.lite:
        model = StaticSignNetLite(input_dim=X_train.shape[1], num_classes=num_classes)
        model_name = "static_lite"
    else:
        model = StaticSignNet(input_dim=X_train.shape[1], num_classes=num_classes)
        model_name = "static"

    print_model_info(model, model_name)

    # Train
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trainer = Trainer(model, device, args.output, model_name)

    trainer.train(
        train_loader, val_loader,
        epochs=args.epochs,
        lr=args.lr,
        patience=args.patience,
        label_smoothing=0.1,
    )

    # Test
    test_X_path = data_dir / f"{prefix}_test_X.npy"
    if test_X_path.exists():
        X_test = np.load(test_X_path)
        y_test = np.load(data_dir / f"{prefix}_test_y.npy")
        test_ds = StaticSignDataset(X_test, y_test, augment=False)
        test_loader = DataLoader(test_ds, batch_size=args.batch_size * 2, shuffle=False)
        trainer.load_best()
        trainer.test(test_loader, class_names)

    # Export ONNX
    if args.export_onnx:
        trainer.load_best()
        onnx_path = str(Path(args.output) / f"{model_name}.onnx")
        export_static_onnx(model, onnx_path, input_dim=X_train.shape[1])

    # Save class names alongside model
    if class_names:
        with open(Path(args.output) / f"{model_name}_classes.json", "w") as f:
            json.dump(class_names, f, indent=2)


def train_dynamic(args):
    """Train dynamic sign classifier."""
    data_dir = Path(args.data)
    prefix = args.data_prefix or "dynamic"

    print("Loading dynamic sign data...")
    X_train = np.load(data_dir / f"{prefix}_train_X.npy")
    y_train = np.load(data_dir / f"{prefix}_train_y.npy")
    X_val = np.load(data_dir / f"{prefix}_val_X.npy")
    y_val = np.load(data_dir / f"{prefix}_val_y.npy")

    classes_file = data_dir / f"{prefix}_classes.json"
    class_names = json.load(open(classes_file)) if classes_file.exists() else None
    num_classes = len(class_names) if class_names else int(y_train.max()) + 1

    print(f"  Train: {len(X_train)} | Val: {len(X_val)}")
    print(f"  Classes: {num_classes}")
    print(f"  Sequence shape: {X_train.shape[1:]}")

    train_ds = DynamicSignDataset(X_train, y_train, augment=True)
    val_ds = DynamicSignDataset(X_val, y_val, augment=False)

    # Weighted sampling
    class_counts = np.bincount(y_train, minlength=num_classes).astype(float)
    class_weights = 1.0 / (class_counts + 1)
    sample_weights = class_weights[y_train]
    sampler = WeightedRandomSampler(
        weights=sample_weights, num_samples=len(train_ds), replacement=True
    )

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, sampler=sampler,
        num_workers=args.workers, pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size * 2, shuffle=False,
        num_workers=args.workers, pin_memory=True,
    )

    input_dim = X_train.shape[2]

    if args.lite:
        model = DynamicSignNetLite(
            input_dim=input_dim, num_classes=num_classes, hidden=128
        )
        model_name = "dynamic_lite"
    else:
        model = DynamicSignNet(
            input_dim=input_dim, num_classes=num_classes,
            hidden_dim=256, num_layers=2, num_heads=4,
        )
        model_name = "dynamic"

    print_model_info(model, model_name)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trainer = Trainer(model, device, args.output, model_name)

    trainer.train(
        train_loader, val_loader,
        epochs=args.epochs,
        lr=args.lr,
        patience=args.patience,
        label_smoothing=0.1,
    )

    # Test
    test_X_path = data_dir / f"{prefix}_test_X.npy"
    if test_X_path.exists():
        X_test = np.load(test_X_path)
        y_test = np.load(data_dir / f"{prefix}_test_y.npy")
        test_ds = DynamicSignDataset(X_test, y_test, augment=False)
        test_loader = DataLoader(test_ds, batch_size=args.batch_size * 2, shuffle=False)
        trainer.load_best()
        trainer.test(test_loader, class_names)

    if args.export_onnx:
        trainer.load_best()
        onnx_path = str(Path(args.output) / f"{model_name}.onnx")
        export_dynamic_onnx(model, onnx_path, input_dim=input_dim)

    if class_names:
        with open(Path(args.output) / f"{model_name}_classes.json", "w") as f:
            json.dump(class_names, f, indent=2)


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Train ASL Sign Recognition Models")

    parser.add_argument("--mode", choices=["static", "dynamic", "both"], default="both")
    parser.add_argument("--data", default="./data", help="Path to prepared data")
    parser.add_argument("--output", default="./trained_models", help="Output directory")
    parser.add_argument("--data_prefix", default=None, help="Data file prefix (e.g. 'custom_static')")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--lite", action="store_true", help="Use lightweight models for Pi")
    parser.add_argument("--export_onnx", action="store_true", help="Export to ONNX after training")
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if torch.cuda.is_available():
        print(f"🚀 GPU: {torch.cuda.get_device_name(0)}")
        print(f"   Memory: {torch.cuda.get_device_properties(0).total_mem / 1024**3:.1f} GB")
    else:
        print("⚠️ No GPU detected — training on CPU (will be slower)")

    if args.mode in ("static", "both"):
        train_static(args)

    if args.mode in ("dynamic", "both"):
        train_dynamic(args)


if __name__ == "__main__":
    main()
