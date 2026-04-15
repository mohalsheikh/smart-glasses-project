#!/usr/bin/env python3
"""
ASL Recognition Model Architectures
=====================================

Two models:
  1. StaticSignNet — MLP for static fingerspelling (A-Z, 0-9)
  2. DynamicSignNet — Bi-LSTM + Attention for word-level dynamic signs

Both models accept MediaPipe hand landmarks as input.
Both can be exported to ONNX for fast inference on Raspberry Pi.
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


# =============================================================================
# STATIC SIGN CLASSIFIER (MLP)
# =============================================================================

class StaticSignNet(nn.Module):
    """
    MLP classifier for static ASL signs (alphabet + numbers).

    Input:  (batch, 63) — 21 landmarks × 3 coordinates, normalized
    Output: (batch, num_classes) — logits
    """

    def __init__(
        self,
        input_dim: int = 63,
        num_classes: int = 29,  # A-Z + space + delete + nothing
        hidden_dims: tuple = (256, 512, 256, 128),
        dropout: float = 0.3,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.num_classes = num_classes

        layers = []
        prev_dim = input_dim

        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            ])
            prev_dim = h_dim

        self.feature_extractor = nn.Sequential(*layers)

        # Final classifier with residual
        self.classifier = nn.Sequential(
            nn.Linear(prev_dim, 64),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(64, num_classes),
        )

        # Temperature scaling for calibrated confidence
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.feature_extractor(x)
        logits = self.classifier(features)
        return logits

    def predict_with_confidence(self, x: torch.Tensor) -> tuple:
        """Returns (predicted_class, confidence, top_k_classes, top_k_probs)"""
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            # Temperature scaling for better calibration
            scaled_logits = logits / self.temperature.clamp(min=0.1)
            probs = F.softmax(scaled_logits, dim=-1)

            top_k = min(5, probs.shape[-1])
            top_probs, top_classes = probs.topk(top_k, dim=-1)

        return top_classes[:, 0], top_probs[:, 0], top_classes, top_probs


# =============================================================================
# ATTENTION MODULE
# =============================================================================

class TemporalAttention(nn.Module):
    """
    Multi-head self-attention for temporal sequences.
    Learns which frames in a sign sequence are most important.
    """

    def __init__(self, hidden_dim: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"

        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.head_dim)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        x: (batch, seq_len, hidden_dim)
        mask: (batch, seq_len) — True for valid positions
        Returns: (batch, hidden_dim) — pooled representation
        """
        B, T, D = x.shape

        Q = self.q_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        # (B, heads, T, T)
        attn = torch.matmul(Q, K.transpose(-2, -1)) / self.scale

        if mask is not None:
            # Expand mask: (B, 1, 1, T)
            mask_expanded = mask.unsqueeze(1).unsqueeze(2)
            attn = attn.masked_fill(~mask_expanded, float('-inf'))

        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, V)  # (B, heads, T, head_dim)
        out = out.transpose(1, 2).contiguous().view(B, T, D)
        out = self.out_proj(out)

        # Weighted mean pooling using attention weights
        # Average attention weights across heads for pooling
        attn_weights = attn.mean(dim=1).mean(dim=1)  # (B, T)
        if mask is not None:
            attn_weights = attn_weights * mask.float()
            attn_weights = attn_weights / (attn_weights.sum(dim=1, keepdim=True) + 1e-8)

        pooled = torch.bmm(attn_weights.unsqueeze(1), out).squeeze(1)  # (B, D)

        return pooled


# =============================================================================
# DYNAMIC SIGN CLASSIFIER (Bi-LSTM + Attention)
# =============================================================================

class DynamicSignNet(nn.Module):
    """
    Sequence model for dynamic ASL word signs.

    Input:  (batch, seq_len, 126) — 30 frames × (2 hands × 21 landmarks × 3 coords)
    Output: (batch, num_classes) — logits

    Architecture:
      Input → LayerNorm → Bi-LSTM × 2 → Temporal Attention → MLP → Output
    """

    def __init__(
        self,
        input_dim: int = 126,          # 2 hands × 63 features
        num_classes: int = 200,
        hidden_dim: int = 256,
        num_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.3,
        bidirectional: bool = True,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim

        # Input normalization
        self.input_norm = nn.LayerNorm(input_dim)

        # Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
        )

        # Positional encoding (learnable)
        self.pos_encoding = nn.Parameter(torch.randn(1, 60, hidden_dim) * 0.02)

        # Bi-LSTM
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
        )

        lstm_output_dim = hidden_dim * (2 if bidirectional else 1)

        # Project LSTM output back to hidden_dim for attention
        self.lstm_proj = nn.Sequential(
            nn.Linear(lstm_output_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )

        # Temporal attention
        self.attention = TemporalAttention(hidden_dim, num_heads=num_heads, dropout=dropout)

        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(hidden_dim // 2, num_classes),
        )

        # Temperature
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)

        self._init_weights()

    def _init_weights(self):
        for name, param in self.lstm.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0)
                # Set forget gate bias to 1
                n = param.size(0)
                param.data[n // 4 : n // 2].fill_(1.0)

        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        x: (batch, seq_len, input_dim)
        mask: (batch, seq_len) — True for valid frames
        """
        B, T, D = x.shape

        # Normalize input
        x = self.input_norm(x)

        # Project
        x = self.input_proj(x)

        # Add positional encoding
        x = x + self.pos_encoding[:, :T, :]

        # LSTM
        if mask is not None:
            lengths = mask.sum(dim=1).cpu()
            packed = nn.utils.rnn.pack_padded_sequence(
                x, lengths, batch_first=True, enforce_sorted=False
            )
            lstm_out, _ = self.lstm(packed)
            lstm_out, _ = nn.utils.rnn.pad_packed_sequence(lstm_out, batch_first=True)
        else:
            lstm_out, _ = self.lstm(x)

        # Project LSTM output
        lstm_out = self.lstm_proj(lstm_out)

        # Attention pooling
        pooled = self.attention(lstm_out, mask)

        # Classify
        logits = self.classifier(pooled)

        return logits

    def predict_with_confidence(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> tuple:
        self.eval()
        with torch.no_grad():
            logits = self.forward(x, mask)
            scaled = logits / self.temperature.clamp(min=0.1)
            probs = F.softmax(scaled, dim=-1)

            top_k = min(5, probs.shape[-1])
            top_probs, top_classes = probs.topk(top_k, dim=-1)

        return top_classes[:, 0], top_probs[:, 0], top_classes, top_probs


# =============================================================================
# LIGHTWEIGHT VARIANT (for Raspberry Pi)
# =============================================================================

class StaticSignNetLite(nn.Module):
    """Lighter model for Raspberry Pi inference."""

    def __init__(self, input_dim: int = 63, num_classes: int = 29):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        return self.net(x)


class DynamicSignNetLite(nn.Module):
    """Lighter model for Raspberry Pi inference."""

    def __init__(self, input_dim: int = 126, num_classes: int = 100, hidden: int = 128):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden)
        self.lstm = nn.LSTM(hidden, hidden, num_layers=1, batch_first=True, bidirectional=True)
        self.classifier = nn.Sequential(
            nn.Linear(hidden * 2, hidden),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden, num_classes),
        )

    def forward(self, x, mask=None):
        x = F.relu(self.input_proj(x))
        out, _ = self.lstm(x)
        # Mean pool over time
        if mask is not None:
            mask_f = mask.unsqueeze(-1).float()
            pooled = (out * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp(min=1)
        else:
            pooled = out.mean(dim=1)
        return self.classifier(pooled)


# =============================================================================
# ONNX EXPORT
# =============================================================================

def export_static_onnx(model: nn.Module, path: str, input_dim: int = 63):
    """Export static model to ONNX for fast inference."""
    model.eval()
    dummy = torch.randn(1, input_dim)
    torch.onnx.export(
        model,
        dummy,
        path,
        input_names=["landmarks"],
        output_names=["logits"],
        dynamic_axes={"landmarks": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=13,
    )
    print(f"✅ Static model exported to {path}")


def export_dynamic_onnx(model: nn.Module, path: str, input_dim: int = 126, seq_len: int = 30):
    """Export dynamic model to ONNX."""
    model.eval()
    dummy = torch.randn(1, seq_len, input_dim)
    torch.onnx.export(
        model,
        dummy,
        path,
        input_names=["sequence"],
        output_names=["logits"],
        dynamic_axes={
            "sequence": {0: "batch", 1: "seq_len"},
            "logits": {0: "batch"},
        },
        opset_version=13,
    )
    print(f"✅ Dynamic model exported to {path}")


# =============================================================================
# MODEL INFO
# =============================================================================

def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def print_model_info(model: nn.Module, name: str = "Model"):
    params = count_parameters(model)
    print(f"\n{'='*50}")
    print(f"  {name}")
    print(f"  Parameters: {params:,}")
    print(f"  Size (est.): {params * 4 / 1024 / 1024:.1f} MB (float32)")
    print(f"{'='*50}")


if __name__ == "__main__":
    # Print model info
    static = StaticSignNet(63, 29)
    print_model_info(static, "StaticSignNet (A-Z + extras)")

    dynamic = DynamicSignNet(126, 200)
    print_model_info(dynamic, "DynamicSignNet (200 words)")

    static_lite = StaticSignNetLite(63, 29)
    print_model_info(static_lite, "StaticSignNetLite (Pi)")

    dynamic_lite = DynamicSignNetLite(126, 100)
    print_model_info(dynamic_lite, "DynamicSignNetLite (Pi)")

    # Test forward pass
    x_static = torch.randn(4, 63)
    print(f"\nStatic forward: {static(x_static).shape}")

    x_dynamic = torch.randn(4, 30, 126)
    print(f"Dynamic forward: {dynamic(x_dynamic).shape}")
