"""
Currency recognition baseline (no custom training):
- Finds banknote-like rectangles via contours/aspect ratio
- OCR with EasyOCR on candidate ROIs
- Extracts denomination and infers currency via keywords/symbols
- Returns summary, hits, and annotated overlay
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple, Optional
import re
import cv2
import numpy as np

try:
    import easyocr
except Exception as e:
    easyocr = None
    print(f"⚠️ EasyOCR import failed: {e}. Install with `pip install easyocr`")

ASPECT_MIN = 1.9
ASPECT_MAX = 3.2
MIN_AREA = 12_000
CANNY1, CANNY2 = 60, 160
DILATE_ITER = 1
ERODE_ITER = 1

DENOM_REGEX = re.compile(r"\b(1|2|5|10|20|25|50|100|200|500|1000)\b")
USD_WORDS = ["united states", "america", "federal reserve", "dollar", "dollars", "usd", "in god we trust"]
EUR_WORDS = ["euro", "euros", "europa", "european central bank", "ecb", "eur"]
GBP_WORDS = ["bank of england", "pounds", "pound", "sterling", "gbp"]
CURRENCY_HINTS = [("USD", USD_WORDS), ("EUR", EUR_WORDS), ("GBP", GBP_WORDS)]

COLOR_NOTE = (30, 220, 255)
COLOR_TEXT = (240, 240, 240)
COLOR_BG = (0, 0, 0)

_reader: Optional["easyocr.Reader"] = None


@dataclass
class CurrencyHit:
    currency: str
    denomination: Optional[int]
    confidence: float
    bbox: Tuple[int, int, int, int]
    ocr_text: str


class CurrencyRecognizer:
    def __init__(self, lang_list: Optional[List[str]] = None):
        if easyocr is None:
            raise RuntimeError("EasyOCR is not installed. Run `pip install easyocr`.")
        # Lazy create shared reader (first time only)
        global _reader
        if _reader is None:
            _reader = easyocr.Reader(lang_list or ["en"], gpu=False)
        self.reader = _reader

    def recognize(self, frame_bgr):
        boxes = self._find_note_candidates(frame_bgr)
        hits: List[CurrencyHit] = []

        for (x1, y1, x2, y2) in boxes:
            roi = frame_bgr[y1:y2, x1:x2]
            text = self._ocr_text(roi)
            denom = self._extract_denom(text)
            curr = self._infer_currency(text) or "UNKNOWN"
            conf = self._score(text, denom, curr)
            hits.append(CurrencyHit(curr, denom, conf, (x1, y1, x2, y2), text))

        if hits:
            parts = [f"{h.currency} {h.denomination if h.denomination is not None else '?'}" for h in hits]
            summary = "Currency detected: " + ", ".join(parts)
        else:
            summary = "No currency detected"

        annotated = self._draw(frame_bgr.copy(), hits)
        return summary, hits, annotated

    # ---- internals ----
    def _find_note_candidates(self, img_bgr):
        h, w = img_bgr.shape[:2]
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(gray, CANNY1, CANNY2)
        edges = cv2.dilate(edges, None, iterations=DILATE_ITER)
        edges = cv2.erode(edges, None, iterations=ERODE_ITER)

        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        boxes = []

        for cnt in contours:
            if cv2.contourArea(cnt) < MIN_AREA:
                continue
            epsilon = 0.02 * cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, epsilon, True)
            if len(approx) < 4:
                continue
            x, y, bw, bh = cv2.boundingRect(approx)
            if bw <= 0 or bh <= 0:
                continue
            aspect = bw / float(bh)
            if ASPECT_MIN <= aspect <= ASPECT_MAX:
                pad_x = int(0.02 * bw)
                pad_y = int(0.02 * bh)
                x1 = max(0, x - pad_x)
                y1 = max(0, y - pad_y)
                x2 = min(w, x + bw + pad_x)
                y2 = min(h, y + bh + pad_y)
                boxes.append((x1, y1, x2, y2))

        return self._merge_overlaps(boxes)

    def _merge_overlaps(self, boxes, iou_thresh: float = 0.3):
        if not boxes:
            return boxes
        b = np.array(boxes, dtype=np.float32)
        x1, y1, x2, y2 = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        order = np.argsort(areas)[::-1]
        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            w = np.maximum(0.0, xx2 - xx1 + 1)
            h = np.maximum(0.0, yy2 - yy1 + 1)
            inter = w * h
            iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)
            inds = np.where(iou <= iou_thresh)[0]
            order = order[inds + 1]
        return [tuple(map(int, b[i])) for i in keep]

    def _ocr_text(self, roi_bgr) -> str:
        rgb = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2RGB)
        result = self.reader.readtext(rgb, detail=0, paragraph=True)
        txt = " ".join(result).lower()
        txt = re.sub(r"\s+", " ", txt).strip()
        return txt

    def _extract_denom(self, text: str) -> Optional[int]:
        nums = DENOM_REGEX.findall(text)
        if not nums:
            return None
        counts = {}
        for n in nums:
            counts[n] = counts.get(n, 0) + 1
        best = sorted(counts.items(), key=lambda kv: (kv[1], int(kv[0])), reverse=True)[0][0]
        return int(best)

    def _infer_currency(self, text: str) -> Optional[str]:
        for code, words in CURRENCY_HINTS:
            if any(w in text for w in words):
                return code
        if "$" in text or "usd" in text:
            return "USD"
        if "€" in text or "eur" in text:
            return "EUR"
        if "£" in text or "gbp" in text:
            return "GBP"
        return None

    def _score(self, text: str, denom: Optional[int], curr: Optional[str]) -> float:
        score = 0.0
        if denom is not None:
            score += 0.5
        if curr is not None and curr != "UNKNOWN":
            score += 0.4
        bonus = 0.0
        for _, words in CURRENCY_HINTS:
            bonus = max(bonus, min(0.1, sum(1 for w in words if w in text) * 0.03))
        return min(1.0, score + bonus)

    def _draw(self, frame, hits: List[CurrencyHit]):
        for h in hits:
            x1, y1, x2, y2 = h.bbox
            cv2.rectangle(frame, (x1, y1), (x2, y2), COLOR_NOTE, 2)
            label = f"{h.currency} {h.denomination if h.denomination is not None else '?'} ({h.confidence:.2f})"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
            cv2.rectangle(frame, (x1, max(0, y1 - th - 10)), (x1 + tw + 10, y1), COLOR_BG, -1)
            cv2.putText(frame, label, (x1 + 5, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.55, COLOR_TEXT, 1, cv2.LINE_AA)
        return frame
