"""
Sign Language Recognition — Model Architectures
=================================================

Shared between training scripts and the runtime interpreter.
Copy this file to both:
  - training/sign_language/models.py   (for training)
  - src/ai_features/sign_language_models.py (for inference)

Models:
  1. StaticSignNet   — classifies single-frame hand landmarks (alphabet + numbers)
  2. DynamicSignNet  — classifies temporal landmark sequences (word-level signs)
  3. TwoHandFusion   — fuses left/right hand features for two-handed signs

Optimized for RTX 4090 training with mixed precision support.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple, Dict, List


# =============================================================================
# 1. STATIC SIGN CLASSIFIER (Alphabet + Numbers)
# =============================================================================

class StaticSignNet(nn.Module):
    """
    Classifies static hand signs from a single frame of landmarks.

    Input:  (batch, 63)  — 21 landmarks × 3 coords (x, y, z), normalized
    Output: (batch, num_classes) — logits for each sign class

    Architecture:
        Landmarks → Feature Extraction → Residual Blocks → Classifier
        With batch norm, dropout, and skip connections for stability.
    """

    def __init__(
        self,
        num_classes: int = 36,        # 26 letters + 10 digits
        input_dim: int = 63,          # 21 landmarks × 3
        hidden_dims: Tuple[int, ...] = (256, 512, 256, 128),
        dropout: float = 0.3,
        use_finger_features: bool = True,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.use_finger_features = use_finger_features

        # Extra engineered features: 5 finger angles + 5 finger extensions
        # + 10 pairwise fingertip distances + palm orientation (3)
        extra_dim = 23 if use_finger_features else 0
        full_input = input_dim + extra_dim

        # Feature extraction
        self.input_bn = nn.BatchNorm1d(full_input)

        layers = []
        in_dim = full_input
        for h in hidden_dims:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.BatchNorm1d(h))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(dropout))
            in_dim = h

        self.feature_net = nn.Sequential(*layers)

        # Residual refinement
        self.res_block = nn.Sequential(
            nn.Linear(hidden_dims[-1], hidden_dims[-1]),
            nn.BatchNorm1d(hidden_dims[-1]),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(hidden_dims[-1], hidden_dims[-1]),
            nn.BatchNorm1d(hidden_dims[-1]),
        )

        # Classifier head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dims[-1], 64),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(64, num_classes),
        )

    def extract_finger_features(self, landmarks: torch.Tensor) -> torch.Tensor:
        """
        Extract hand-crafted features from landmarks.
        landmarks: (batch, 63) → reshaped to (batch, 21, 3)
        """
        B = landmarks.shape[0]
        lm = landmarks.view(B, 21, 3)

        features = []

        # 1. Finger extension: tip.y < pip.y (5 features)
        tips = [4, 8, 12, 16, 20]
        pips = [3, 6, 10, 14, 18]
        for tip, pip in zip(tips, pips):
            ext = (lm[:, pip, 1] - lm[:, tip, 1]).unsqueeze(1)  # positive = extended
            features.append(ext)

        # 2. Finger curl angles at PIP joint (5 features)
        mcps = [2, 5, 9, 13, 17]
        for mcp, pip, tip in zip(mcps, pips, tips):
            v1 = lm[:, mcp] - lm[:, pip]
            v2 = lm[:, tip] - lm[:, pip]
            cos_a = F.cosine_similarity(v1, v2, dim=1).unsqueeze(1)
            features.append(cos_a)

        # 3. Pairwise fingertip distances (10 features: C(5,2))
        tip_coords = lm[:, tips]  # (B, 5, 3)
        for i in range(5):
            for j in range(i + 1, 5):
                dist = torch.norm(tip_coords[:, i] - tip_coords[:, j], dim=1).unsqueeze(1)
                features.append(dist)

        # 4. Palm normal direction (3 features)
        wrist = lm[:, 0]
        index_mcp = lm[:, 5]
        pinky_mcp = lm[:, 17]
        v1 = index_mcp - wrist
        v2 = pinky_mcp - wrist
        normal = torch.cross(v1, v2, dim=1)
        normal = F.normalize(normal, dim=1)
        features.append(normal)

        return torch.cat(features, dim=1)  # (B, 23)

    def forward(self, landmarks: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        landmarks: (batch, 63) normalized hand landmarks
        """
        if self.use_finger_features:
            extra = self.extract_finger_features(landmarks)
            x = torch.cat([landmarks, extra], dim=1)
        else:
            x = landmarks

        x = self.input_bn(x)
        x = self.feature_net(x)

        # Residual
        residual = x
        x = self.res_block(x)
        x = F.gelu(x + residual)

        return self.classifier(x)


# =============================================================================
# 2. DYNAMIC SIGN CLASSIFIER (Word-Level Signs)
# =============================================================================

class TemporalAttention(nn.Module):
    """Multi-head self-attention over temporal dimension."""

    def __init__(self, hidden_dim: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """x: (batch, seq_len, hidden_dim)"""
        attn_out, _ = self.attn(x, x, x, key_padding_mask=mask)
        return self.norm(x + attn_out)


class DynamicSignNet(nn.Module):
    """
    Classifies dynamic (temporal) sign sequences.

    Input:  (batch, seq_len, input_dim)  — sequence of landmark frames
            input_dim = 126 (both hands: 42 landmarks × 3) or 63 (one hand)
    Output: (batch, num_classes) — logits for each word/phrase

    Architecture:
        Frame Encoder → Bidirectional LSTM → Temporal Attention → Classifier

    Supports variable-length sequences with padding masks.
    """

    def __init__(
        self,
        num_classes: int = 2000,       # WLASL-2000
        input_dim: int = 126,          # 42 landmarks × 3 (both hands)
        hidden_dim: int = 256,
        num_lstm_layers: int = 2,
        num_attn_heads: int = 4,
        dropout: float = 0.3,
        max_seq_len: int = 64,         # Max frames per sequence
    ):
        super().__init__()
        self.num_classes = num_classes
        self.max_seq_len = max_seq_len
        self.hidden_dim = hidden_dim

        # Frame-level feature extractor
        self.frame_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
        )

        # Positional encoding
        self.pos_embedding = nn.Embedding(max_seq_len, hidden_dim)

        # Temporal modeling
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_lstm_layers > 1 else 0,
        )

        # Project bidirectional output
        self.lstm_proj = nn.Linear(hidden_dim * 2, hidden_dim)

        # Temporal attention
        self.temporal_attn = TemporalAttention(hidden_dim, num_attn_heads, dropout)

        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes),
        )

    def forward(
        self,
        sequences: torch.Tensor,
        lengths: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        sequences: (batch, seq_len, input_dim)
        lengths:   (batch,) — actual lengths before padding
        """
        B, T, _ = sequences.shape

        # Frame encoding
        x = self.frame_encoder(sequences)  # (B, T, hidden)

        # Add positional encoding
        positions = torch.arange(T, device=sequences.device).unsqueeze(0).expand(B, -1)
        x = x + self.pos_embedding(positions)

        # LSTM
        if lengths is not None:
            packed = nn.utils.rnn.pack_padded_sequence(
                x, lengths.cpu().clamp(min=1), batch_first=True, enforce_sorted=False
            )
            lstm_out, _ = self.lstm(packed)
            lstm_out, _ = nn.utils.rnn.pad_packed_sequence(lstm_out, batch_first=True)
        else:
            lstm_out, _ = self.lstm(x)

        x = self.lstm_proj(lstm_out)  # (B, T, hidden)

        # Attention with padding mask
        mask = None
        if lengths is not None:
            mask = torch.arange(T, device=sequences.device).unsqueeze(0) >= lengths.unsqueeze(1)

        x = self.temporal_attn(x, mask=mask)  # (B, T, hidden)

        # Pool: use attention-weighted mean
        if mask is not None:
            weights = (~mask).float().unsqueeze(-1)  # (B, T, 1)
            pooled = (x * weights).sum(dim=1) / weights.sum(dim=1).clamp(min=1)
        else:
            pooled = x.mean(dim=1)

        return self.classifier(pooled)


# =============================================================================
# 3. TWO-HAND FUSION (for two-handed signs)
# =============================================================================

class TwoHandFusion(nn.Module):
    """
    Fuses left and right hand features for two-handed sign recognition.
    Can be used as a wrapper around StaticSignNet or independently.
    """

    def __init__(
        self,
        single_hand_dim: int = 63,
        hidden_dim: int = 128,
        num_classes: int = 200,  # Two-handed signs vocabulary
        dropout: float = 0.3,
    ):
        super().__init__()

        # Per-hand encoders
        self.left_encoder = nn.Sequential(
            nn.Linear(single_hand_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
        )
        self.right_encoder = nn.Sequential(
            nn.Linear(single_hand_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
        )

        # Interaction features
        self.interaction = nn.Sequential(
            nn.Linear(hidden_dim * 2 + 15, hidden_dim),  # +15 for spatial features
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )

        self.classifier = nn.Linear(hidden_dim, num_classes)

    def compute_spatial_features(
        self, left_lm: torch.Tensor, right_lm: torch.Tensor
    ) -> torch.Tensor:
        """Compute spatial relationship features between hands."""
        B = left_lm.shape[0]
        left = left_lm.view(B, 21, 3)
        right = right_lm.view(B, 21, 3)

        features = []

        # Distance between wrists
        wrist_dist = torch.norm(left[:, 0] - right[:, 0], dim=1, keepdim=True)
        features.append(wrist_dist)

        # Distance between palm centers
        left_palm = left[:, [0, 5, 9, 13, 17]].mean(dim=1)
        right_palm = right[:, [0, 5, 9, 13, 17]].mean(dim=1)
        palm_dist = torch.norm(left_palm - right_palm, dim=1, keepdim=True)
        features.append(palm_dist)

        # Fingertip-to-fingertip distances (5 features)
        tips = [4, 8, 12, 16, 20]
        for t in tips:
            d = torch.norm(left[:, t] - right[:, t], dim=1, keepdim=True)
            features.append(d)

        # Relative position (3 features)
        rel_pos = right_palm - left_palm
        features.append(rel_pos)

        # Relative orientation via palm normals (3 features)
        left_n = torch.cross(left[:, 5] - left[:, 0], left[:, 17] - left[:, 0], dim=1)
        right_n = torch.cross(right[:, 5] - right[:, 0], right[:, 17] - right[:, 0], dim=1)
        left_n = F.normalize(left_n, dim=1)
        right_n = F.normalize(right_n, dim=1)
        normal_diff = right_n - left_n
        features.append(normal_diff)

        return torch.cat(features, dim=1)  # (B, 15)

    def forward(
        self, left_landmarks: torch.Tensor, right_landmarks: torch.Tensor
    ) -> torch.Tensor:
        """
        left_landmarks:  (batch, 63)
        right_landmarks: (batch, 63)
        """
        left_feat = self.left_encoder(left_landmarks)
        right_feat = self.right_encoder(right_landmarks)
        spatial = self.compute_spatial_features(left_landmarks, right_landmarks)

        fused = torch.cat([left_feat, right_feat, spatial], dim=1)
        x = self.interaction(fused)
        return self.classifier(x)


# =============================================================================
# UTILITY: Landmark Normalization
# =============================================================================

def normalize_landmarks(landmarks: np.ndarray) -> np.ndarray:
    """
    Normalize 21×3 hand landmarks for model input.

    Steps:
        1. Center on wrist (landmark 0)
        2. Scale by palm size (wrist→middle_mcp distance)
        3. Flatten to (63,)

    Args:
        landmarks: (21, 3) array of (x, y, z) coordinates

    Returns:
        (63,) normalized flat array
    """
    lm = landmarks.copy().astype(np.float32)

    # Center on wrist
    wrist = lm[0].copy()
    lm -= wrist

    # Scale by palm size
    palm_size = np.linalg.norm(lm[9])  # middle finger MCP
    if palm_size > 1e-6:
        lm /= palm_size

    return lm.flatten()


def normalize_landmarks_batch(landmarks_batch: np.ndarray) -> np.ndarray:
    """Normalize a batch of landmarks. Input: (N, 21, 3) → Output: (N, 63)"""
    return np.array([normalize_landmarks(lm) for lm in landmarks_batch])


# =============================================================================
# UTILITY: Label Mappings
# =============================================================================

ALPHABET_LABELS = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ") + [str(i) for i in range(10)]
ALPHABET_TO_IDX = {label: idx for idx, label in enumerate(ALPHABET_LABELS)}
IDX_TO_ALPHABET = {idx: label for label, idx in ALPHABET_TO_IDX.items()}


def get_label_maps(vocab_file: Optional[str] = None) -> Tuple[Dict[str, int], Dict[int, str]]:
    """
    Get word-level label maps.
    If vocab_file provided, load from file. Otherwise return alphabet labels.
    """
    if vocab_file:
        with open(vocab_file, 'r') as f:
            words = [line.strip() for line in f if line.strip()]
        word_to_idx = {w: i for i, w in enumerate(words)}
        idx_to_word = {i: w for w, i in word_to_idx.items()}
        return word_to_idx, idx_to_word
    return ALPHABET_TO_IDX, IDX_TO_ALPHABET


# =============================================================================
# UTILITY: Model Loading
# =============================================================================

def load_static_model(
    checkpoint_path: str,
    num_classes: int = 36,
    device: str = "cpu",
) -> StaticSignNet:
    """Load a trained StaticSignNet from checkpoint."""
    model = StaticSignNet(num_classes=num_classes)
    state = torch.load(checkpoint_path, map_location=device, weights_only=True)
    if "model_state_dict" in state:
        model.load_state_dict(state["model_state_dict"])
    else:
        model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


def load_dynamic_model(
    checkpoint_path: str,
    num_classes: int = 2000,
    input_dim: int = 126,
    device: str = "cpu",
) -> DynamicSignNet:
    """Load a trained DynamicSignNet from checkpoint."""
    model = DynamicSignNet(num_classes=num_classes, input_dim=input_dim)
    state = torch.load(checkpoint_path, map_location=device, weights_only=True)
    if "model_state_dict" in state:
        model.load_state_dict(state["model_state_dict"])
    else:
        model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


# =============================================================================
# Quick test
# =============================================================================

if __name__ == "__main__":
    print("Testing model architectures...")

    # Static model
    model_s = StaticSignNet(num_classes=36)
    x = torch.randn(4, 63)
    out = model_s(x)
    print(f"StaticSignNet:  input {x.shape} → output {out.shape}")
    print(f"  Parameters: {sum(p.numel() for p in model_s.parameters()):,}")

    # Dynamic model
    model_d = DynamicSignNet(num_classes=100, input_dim=126)
    seq = torch.randn(4, 32, 126)
    lengths = torch.tensor([32, 20, 15, 28])
    out_d = model_d(seq, lengths)
    print(f"DynamicSignNet: input {seq.shape} → output {out_d.shape}")
    print(f"  Parameters: {sum(p.numel() for p in model_d.parameters()):,}")

    # Two-hand fusion
    model_f = TwoHandFusion(num_classes=50)
    left = torch.randn(4, 63)
    right = torch.randn(4, 63)
    out_f = model_f(left, right)
    print(f"TwoHandFusion:  input 2×{left.shape} → output {out_f.shape}")
    print(f"  Parameters: {sum(p.numel() for p in model_f.parameters()):,}")

    # Normalization
    lm = np.random.randn(21, 3).astype(np.float32)
    norm = normalize_landmarks(lm)
    print(f"\nNormalization: {lm.shape} → {norm.shape}")

    print("\n✅ All models OK!")
