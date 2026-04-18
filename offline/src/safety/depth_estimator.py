# src/safety/depth_estimator.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
import time

import numpy as np
import cv2 as cv

import src.utils.config as config


@dataclass
class DepthResult:
    depth: np.ndarray          # HxW float32 in [0,1], higher=closer
    quality: float             # 0..1 (higher = more reliable)
    ts: float
    frame_idx: int


@dataclass
class _DepthCache:
    result: DepthResult


class DepthEstimator:
    """
    MiDaS depth estimator (fast path).

    Output:
      - DepthResult(depth_norm, quality)
      - depth_norm: HxW float32 in [0,1]
      - Higher = closer (normalized inverse depth-ish)

    Performance:
      - Runs every DEPTH_EVERY_N_FRAMES frames
      - Downscales input to DEPTH_INPUT_MAX_WIDTH
      - Caches last output
    """

    def __init__(self):
        self._loaded = False
        self._midas = None
        self._transform = None
        self._device = None
        self._cache: Optional[_DepthCache] = None

        print(f"🧠 DepthEstimator initialized (enabled={getattr(config, 'DEPTH_ENABLED', False)})")

    def _lazy_load(self) -> None:
        if self._loaded:
            return

        try:
            import torch  # noqa
        except Exception as e:
            raise RuntimeError(
                "PyTorch is required for depth mode. Install with: pip install torch torchvision"
            ) from e

        import torch

        # device preference: MPS (Mac) > CUDA > CPU
        if torch.backends.mps.is_available():
            device = torch.device("mps")
        elif torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")

        model_name = getattr(config, "DEPTH_MODEL", "MiDaS_small") or "MiDaS_small"

        # TorchHub will download weights the first time (needs internet once).
        midas = torch.hub.load("intel-isl/MiDaS", model_name)
        midas.to(device)
        midas.eval()

        transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
        if "small" in model_name.lower():
            transform = transforms.small_transform
        else:
            transform = transforms.dpt_transform

        self._midas = midas
        self._transform = transform
        self._device = device
        self._loaded = True

        print(f"✅ MiDaS loaded: {model_name} on {device}")

    def _resize_for_speed(self, frame_bgr: np.ndarray) -> np.ndarray:
        max_w = int(getattr(config, "DEPTH_INPUT_MAX_WIDTH", 256) or 256)
        h, w = frame_bgr.shape[:2]
        if w <= max_w:
            return frame_bgr
        scale = max_w / float(w)
        new_w = max_w
        new_h = max(1, int(h * scale))
        return cv.resize(frame_bgr, (new_w, new_h), interpolation=cv.INTER_AREA)

    def _compute_quality(self, depth_norm_small: np.ndarray, lo: float, hi: float) -> float:
        """
        Heuristic reliability score 0..1.

        Good depth frames tend to have:
          - meaningful dynamic range (hi-lo not tiny)
          - non-trivial variance in normalized map
        """
        dyn = float(hi - lo)
        if dyn < 1e-6:
            return 0.0

        # variance of normalized depth (0..1)
        std = float(np.std(depth_norm_small))

        # Map std ~0.00..0.18 into 0..1 (tunable)
        q_std = max(0.0, min(1.0, std / 0.18))

        # Relative dynamic range (prevents freak frames that normalize badly)
        denom = float(abs(hi) + 1e-6)
        q_dyn = max(0.0, min(1.0, dyn / denom))

        # Weighted blend
        q = 0.65 * q_std + 0.35 * q_dyn
        return max(0.0, min(1.0, float(q)))

    def estimate(self, frame_bgr: np.ndarray, frame_idx: int) -> Optional[DepthResult]:
        """
        Returns DepthResult or cached result if not time to run.
        """
        if not getattr(config, "DEPTH_ENABLED", False):
            return None

        every = int(getattr(config, "DEPTH_EVERY_N_FRAMES", 10) or 10)
        if every <= 0:
            every = 10

        # Use cache if we shouldn't run this frame
        if self._cache and (frame_idx % every != 0):
            return self._cache.result

        self._lazy_load()

        import torch

        # Downscale for speed + convert to RGB for MiDaS transforms
        small_bgr = self._resize_for_speed(frame_bgr)
        small_rgb = cv.cvtColor(small_bgr, cv.COLOR_BGR2RGB)

        input_batch = self._transform(small_rgb).to(self._device)

        with torch.no_grad():
            prediction = self._midas(input_batch)
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=small_rgb.shape[:2],
                mode="bicubic",
                align_corners=False,
            ).squeeze()

        depth_raw = prediction.detach().float().cpu().numpy()

        # Robust normalize (avoid flicker from outliers)
        lo = float(np.percentile(depth_raw, 5))
        hi = float(np.percentile(depth_raw, 95))

        if (hi - lo) < 1e-6:
            depth_norm_small = np.zeros_like(depth_raw, dtype=np.float32)
            quality = 0.0
        else:
            depth_norm_small = ((depth_raw - lo) / (hi - lo)).clip(0.0, 1.0).astype(np.float32)
            quality = self._compute_quality(depth_norm_small, lo, hi)

        # Upscale back to original frame size
        H, W = frame_bgr.shape[:2]
        depth_norm = cv.resize(depth_norm_small, (W, H), interpolation=cv.INTER_LINEAR)

        res = DepthResult(depth=depth_norm, quality=quality, ts=time.time(), frame_idx=frame_idx)
        self._cache = _DepthCache(result=res)
        return res
