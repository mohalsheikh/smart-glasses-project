#!/usr/bin/env python3
"""
Hand Landmark Data Augmentation
=================================

Augmentations that work on normalized MediaPipe landmarks.
These preserve the semantic meaning of signs while increasing data diversity.
"""

from __future__ import annotations

import numpy as np
from typing import List, Tuple, Optional


class LandmarkAugmentor:
    """
    Apply augmentations to normalized hand landmarks.
    Works on both static (63-dim) and sequences (T, 126-dim).
    """

    def __init__(
        self,
        noise_std: float = 0.02,
        scale_range: Tuple[float, float] = (0.85, 1.15),
        rotation_range: float = 15.0,        # degrees
        time_warp_range: float = 0.2,         # for sequences
        finger_jitter_std: float = 0.01,
        mirror_prob: float = 0.5,
        dropout_prob: float = 0.05,            # landmark dropout
    ):
        self.noise_std = noise_std
        self.scale_range = scale_range
        self.rotation_range = rotation_range
        self.time_warp_range = time_warp_range
        self.finger_jitter_std = finger_jitter_std
        self.mirror_prob = mirror_prob
        self.dropout_prob = dropout_prob

    def augment_static(self, landmarks: np.ndarray) -> np.ndarray:
        """
        Augment a single static landmark vector.

        Args:
            landmarks: shape (63,) — 21 landmarks × 3 coords

        Returns:
            Augmented landmarks (63,)
        """
        lm = landmarks.copy().reshape(21, 3)

        # Random rotation around z-axis (palm plane)
        if np.random.random() < 0.5:
            lm = self._rotate_2d(lm, np.random.uniform(-self.rotation_range, self.rotation_range))

        # Random scale
        if np.random.random() < 0.5:
            scale = np.random.uniform(*self.scale_range)
            lm *= scale

        # Gaussian noise
        if np.random.random() < 0.7:
            noise = np.random.randn(*lm.shape).astype(np.float32) * self.noise_std
            lm += noise

        # Finger-specific jitter (different amounts per finger)
        if np.random.random() < 0.3:
            for finger_tips in [[4], [8], [12], [16], [20]]:
                jitter = np.random.randn(len(finger_tips), 3).astype(np.float32) * self.finger_jitter_std
                lm[finger_tips] += jitter

        # Mirror (x-axis flip — simulates other hand)
        if np.random.random() < self.mirror_prob:
            lm[:, 0] = -lm[:, 0]

        # Landmark dropout (zero out random landmarks)
        if np.random.random() < 0.2:
            mask = np.random.random(21) > self.dropout_prob
            lm *= mask[:, np.newaxis]

        return lm.flatten().astype(np.float32)

    def augment_sequence(self, sequence: np.ndarray) -> np.ndarray:
        """
        Augment a sequence of landmarks for dynamic signs.

        Args:
            sequence: shape (T, 126) — T frames × (2 hands × 63)

        Returns:
            Augmented sequence (T, 126)
        """
        seq = sequence.copy()
        T, D = seq.shape

        # Time warp — slightly stretch/compress time
        if np.random.random() < 0.5:
            seq = self._time_warp(seq)

        # Spatial augmentation (apply same transform to all frames)
        if np.random.random() < 0.5:
            angle = np.random.uniform(-self.rotation_range, self.rotation_range)
            scale = np.random.uniform(*self.scale_range)

            for t in range(T):
                # Left hand (0:63)
                lh = seq[t, :63].reshape(21, 3)
                lh = self._rotate_2d(lh, angle) * scale
                seq[t, :63] = lh.flatten()

                # Right hand (63:126)
                rh = seq[t, 63:126].reshape(21, 3)
                rh = self._rotate_2d(rh, angle) * scale
                seq[t, 63:126] = rh.flatten()

        # Temporal noise (different noise per frame)
        if np.random.random() < 0.7:
            noise = np.random.randn(*seq.shape).astype(np.float32) * self.noise_std
            seq += noise

        # Frame dropout (zero out random frames)
        if np.random.random() < 0.3:
            n_drop = max(1, int(T * 0.1))
            drop_idx = np.random.choice(T, n_drop, replace=False)
            seq[drop_idx] = 0.0

        # Speed perturbation (shift frames)
        if np.random.random() < 0.3:
            shift = np.random.randint(-2, 3)
            seq = np.roll(seq, shift, axis=0)

        return seq.astype(np.float32)

    def _rotate_2d(self, landmarks: np.ndarray, angle_degrees: float) -> np.ndarray:
        """Rotate landmarks around z-axis (in xy plane)."""
        angle = np.radians(angle_degrees)
        cos_a = np.cos(angle)
        sin_a = np.sin(angle)

        rotated = landmarks.copy()
        x = landmarks[:, 0]
        y = landmarks[:, 1]
        rotated[:, 0] = x * cos_a - y * sin_a
        rotated[:, 1] = x * sin_a + y * cos_a

        return rotated

    def _time_warp(self, sequence: np.ndarray) -> np.ndarray:
        """Apply random temporal warping."""
        T = len(sequence)
        if T < 4:
            return sequence

        # Create warped time indices
        warp = np.random.uniform(1 - self.time_warp_range, 1 + self.time_warp_range, T)
        warp = np.cumsum(warp)
        warp = warp / warp[-1] * (T - 1)

        # Interpolate
        original_indices = np.arange(T)
        warped = np.zeros_like(sequence)

        for dim in range(sequence.shape[1]):
            warped[:, dim] = np.interp(warp, original_indices, sequence[:, dim])

        return warped


class MixupAugmentor:
    """Mixup augmentation for training."""

    def __init__(self, alpha: float = 0.2):
        self.alpha = alpha

    def mixup(
        self, x1: np.ndarray, y1: int, x2: np.ndarray, y2: int, num_classes: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply mixup to two samples.

        Returns:
            (mixed_x, one_hot_y) where one_hot_y is soft labels
        """
        lam = np.random.beta(self.alpha, self.alpha)
        mixed_x = lam * x1 + (1 - lam) * x2

        y_onehot = np.zeros(num_classes, dtype=np.float32)
        y_onehot[y1] = lam
        y_onehot[y2] = 1 - lam

        return mixed_x, y_onehot
