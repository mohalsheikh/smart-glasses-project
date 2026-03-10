# src/currency_recognizer.py
"""
Currency Recognition System
============================
Uses GPT-4o Vision to identify and read currency denominations.
Falls back to local color/shape heuristics when API is unavailable.

Supports: USD, EUR, GBP, and other major currencies.
"""

from __future__ import annotations

import base64
import time
from typing import Optional, Dict, Any, List, Tuple

import cv2 as cv
import numpy as np

import src.utils.config as config

try:
    from src.brain.openai_client import client
except Exception:
    client = None


# ---------------------------------------------------------------------------
# Local heuristic helpers (no API needed)
# ---------------------------------------------------------------------------

# Dominant color ranges for US bills (in HSV)
_USD_GREEN_LOWER = np.array([35, 30, 80])
_USD_GREEN_UPPER = np.array([85, 180, 220])

# Approximate aspect ratio of US bills (6.14" x 2.61")
_BILL_ASPECT_RATIO = 2.35
_BILL_ASPECT_TOLERANCE = 0.6


class CurrencyRecognizer:
    """
    Multi-strategy currency recognition:
      1. GPT-4o Vision (primary) - most accurate, reads denominations
      2. Local heuristic (fallback) - color/shape analysis
    """

    def __init__(self):
        self.enabled = client is not None and bool(config.OPENAI_API_KEY_PRESENT)
        self._last_call_time: float = 0.0
        self._cooldown_s: float = 1.5
        self._last_result: Optional[str] = None

        if self.enabled:
            print("💵 CurrencyRecognizer initialized (GPT-4o Vision + local fallback)")
        else:
            print("💵 CurrencyRecognizer initialized (local heuristic only)")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def recognize(self, frame: np.ndarray) -> str:
        """
        Identify currency in the frame.
        Returns a natural-language string.
        """
        if frame is None:
            return "I need a clearer view to check for currency."

        now = time.time()

        # Cooldown check
        if now - self._last_call_time < self._cooldown_s and self._last_result:
            return self._last_result

        self._last_call_time = now

        # Strategy 1: Local heuristic pre-check (fast reject)
        has_currency_region, local_hint = self._local_precheck(frame)

        # Strategy 2: GPT-4o Vision (if available)
        if self.enabled:
            result = self._vision_api_recognize(frame, local_hint)
            if result:
                self._last_result = result
                return result

        # Strategy 3: Local-only result
        if has_currency_region and local_hint:
            self._last_result = local_hint
            return local_hint

        self._last_result = "No currency detected."
        return "No currency detected."

    def recognize_from_detections(
        self, frame: np.ndarray, detections: List[Dict[str, Any]]
    ) -> Optional[str]:
        """
        Check if any YOLO detections look like money and identify them.
        """
        if frame is None or not detections:
            return None

        money_labels = {"banknote", "coin", "money", "cash", "bill", "currency", "wallet"}
        money_dets = [
            d for d in detections
            if (d.get("label", "") or "").lower() in money_labels
        ]

        if not money_dets:
            return None

        # Crop the region around the best money detection
        best = max(money_dets, key=lambda d: float(d.get("confidence", 0)))
        bbox = best.get("bbox")
        if bbox:
            x1, y1, x2, y2 = bbox
            h, w = frame.shape[:2]
            pad = 20
            x1 = max(0, x1 - pad)
            y1 = max(0, y1 - pad)
            x2 = min(w, x2 + pad)
            y2 = min(h, y2 + pad)
            cropped = frame[y1:y2, x1:x2]
            if cropped.size > 0:
                return self.recognize(cropped)

        return self.recognize(frame)

    # ------------------------------------------------------------------
    # GPT-4o Vision recognition
    # ------------------------------------------------------------------

    def _vision_api_recognize(self, frame: np.ndarray, local_hint: Optional[str] = None) -> Optional[str]:
        """Use GPT-4o Vision to identify currency with detail about positions and totals."""
        if client is None:
            return None

        try:
            image_url = self._frame_to_data_url(frame)

            hint_text = ""
            if local_hint:
                hint_text = f"\nLocal analysis hint: {local_hint}"

            messages = [
                {
                    "role": "system",
                    "content": (
                        "You are a currency identification assistant for a visually impaired user "
                        "who cannot see the bills or coins they are holding. Be very specific and helpful.\n\n"
                        "Rules:\n"
                        "- Identify EVERY bill and coin visible.\n"
                        "- Describe their positions (top, bottom, left, right, front, behind).\n"
                        "- State the denomination of each bill/coin.\n"
                        "- If multiple bills are stacked, describe the order (top to bottom).\n"
                        "- Calculate and state the TOTAL amount.\n"
                        "- Mention the currency (US dollars, euros, etc.).\n"
                        "- If a bill is partially hidden, mention that.\n"
                        "- Speak directly to the user in a natural, clear way.\n"
                        "- Keep it concise but complete — 2-4 sentences.\n"
                        "- If no currency is visible, respond with exactly: NO_CURRENCY"
                    ),
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                "Identify all the money in this image. "
                                "Tell me each bill/coin, where it is positioned, "
                                "and the total amount."
                                f"{hint_text}"
                            ),
                        },
                        {"type": "image_url", "image_url": {"url": image_url}},
                    ],
                },
            ]

            resp = client.chat.completions.create(
                model=config.OPENAI_VISION_MODEL,
                messages=messages,
                max_tokens=200,
                temperature=0.1,
            )

            answer = (resp.choices[0].message.content or "").strip()

            if "NO_CURRENCY" in answer.upper():
                return None

            return answer if answer else None

        except Exception as e:
            if config.DEBUG:
                print(f"⚠️ CurrencyRecognizer Vision API error: {e!r}")
            return None

    # ------------------------------------------------------------------
    # Local heuristic pre-check
    # ------------------------------------------------------------------

    def _local_precheck(self, frame: np.ndarray) -> Tuple[bool, Optional[str]]:
        """
        Fast local analysis: check for bill-shaped green regions (USD)
        and rectangle-like objects with currency aspect ratios.
        """
        try:
            h, w = frame.shape[:2]
            if h < 50 or w < 50:
                return False, None

            # Resize for speed
            scale = 320.0 / max(h, w)
            if scale < 1.0:
                small = cv.resize(frame, None, fx=scale, fy=scale, interpolation=cv.INTER_AREA)
            else:
                small = frame

            hsv = cv.cvtColor(small, cv.COLOR_BGR2HSV)

            # Check for USD green-ish regions
            mask = cv.inRange(hsv, _USD_GREEN_LOWER, _USD_GREEN_UPPER)
            green_ratio = np.sum(mask > 0) / mask.size

            if green_ratio < 0.05:
                return False, None

            # Find contours in green mask
            contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

            for cnt in contours:
                area = cv.contourArea(cnt)
                if area < 500:
                    continue

                rect = cv.minAreaRect(cnt)
                rw, rh = rect[1]
                if rw == 0 or rh == 0:
                    continue

                aspect = max(rw, rh) / min(rw, rh)
                if abs(aspect - _BILL_ASPECT_RATIO) < _BILL_ASPECT_TOLERANCE:
                    return True, "I may see what looks like a US bill. Let me take a closer look."

            if green_ratio > 0.15:
                return True, None

            return False, None

        except Exception:
            return False, None

    # ------------------------------------------------------------------
    # Image encoding
    # ------------------------------------------------------------------

    @staticmethod
    def _frame_to_data_url(frame: np.ndarray) -> str:
        """Encode frame as base64 JPEG data URL."""
        h, w = frame.shape[:2]
        max_w = 512
        if w > max_w:
            scale = max_w / float(w)
            new_w = max(1, int(w * scale))
            new_h = max(1, int(h * scale))
            frame = cv.resize(frame, (new_w, new_h), interpolation=cv.INTER_AREA)

        success, encoded = cv.imencode(".jpg", frame, [int(cv.IMWRITE_JPEG_QUALITY), 80])
        if not success:
            raise RuntimeError("Failed to encode frame as JPEG.")

        b64 = base64.b64encode(encoded.tobytes()).decode("utf-8")
        return f"data:image/jpeg;base64,{b64}"
