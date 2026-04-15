# src/ocr_engine.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import base64
import time

import cv2 as cv
import numpy as np

import src.utils.config as config

try:
    from src.brain.openai_client import client  # your OpenAI client wrapper
except Exception:
    client = None


# ----------------------------
# Data models
# ----------------------------

BBox = Tuple[int, int, int, int]  # x1,y1,x2,y2


@dataclass
class OCRLine:
    text: str
    conf: float  # 0..1
    bbox: BBox


@dataclass
class OCRResult:
    text: str
    conf: float  # 0..1 overall
    lines: List[OCRLine]
    engine: str  # "easyocr" | "tesseract" | "scene_ai" | "none"
    used_regions: bool


# ----------------------------
# Helpers
# ----------------------------

def _now() -> float:
    return time.time()


def _clamp_bbox(x1: int, y1: int, x2: int, y2: int, w: int, h: int) -> BBox:
    x1 = max(0, min(w - 1, x1))
    y1 = max(0, min(h - 1, y1))
    x2 = max(0, min(w, x2))
    y2 = max(0, min(h, y2))
    if x2 <= x1:
        x2 = min(w, x1 + 1)
    if y2 <= y1:
        y2 = min(h, y1 + 1)
    return (x1, y1, x2, y2)


def _pad_bbox(b: BBox, pad: int, w: int, h: int) -> BBox:
    x1, y1, x2, y2 = b
    return _clamp_bbox(x1 - pad, y1 - pad, x2 + pad, y2 + pad, w, h)


def _frame_to_data_url(frame_bgr: np.ndarray, quality: int = 75) -> str:
    encode_params = [int(cv.IMWRITE_JPEG_QUALITY), int(quality)]
    ok, buf = cv.imencode(".jpg", frame_bgr, encode_params)
    if not ok:
        raise RuntimeError("Failed to encode image for Scene AI OCR.")
    b64 = base64.b64encode(buf.tobytes()).decode("utf-8")
    return f"data:image/jpeg;base64,{b64}"


def _clean_text(s: str) -> str:
    s = (s or "").strip()
    s = " ".join(s.split())
    return s


def _median_line_height(lines: List[OCRLine]) -> float:
    hs = []
    for ln in lines:
        x1, y1, x2, y2 = ln.bbox
        hs.append(max(1, (y2 - y1)))
    if not hs:
        return 14.0
    return float(np.median(hs))


def _sort_reading_order(lines: List[OCRLine]) -> List[OCRLine]:
    """
    Sort lines top-to-bottom, left-to-right, with simple row clustering.
    """
    if not lines:
        return []

    lines_sorted = sorted(lines, key=lambda l: (l.bbox[1], l.bbox[0]))

    med_h = _median_line_height(lines_sorted)
    row_tol = max(10.0, med_h * 0.6)

    rows: List[List[OCRLine]] = []
    current: List[OCRLine] = []
    current_y: Optional[float] = None

    for ln in lines_sorted:
        y = ln.bbox[1]
        if current_y is None:
            current_y = float(y)
            current = [ln]
            continue

        if abs(float(y) - current_y) <= row_tol:
            current.append(ln)
            current_y = (current_y * 0.8) + (float(y) * 0.2)
        else:
            current.sort(key=lambda l: l.bbox[0])
            rows.append(current)
            current = [ln]
            current_y = float(y)

    if current:
        current.sort(key=lambda l: l.bbox[0])
        rows.append(current)

    merged: List[OCRLine] = []
    for r in rows:
        merged.extend(r)
    return merged


def _lines_to_paragraphs(lines: List[OCRLine]) -> List[str]:
    """
    Group ordered lines into paragraphs based on vertical gaps.
    """
    if not lines:
        return []

    ordered = _sort_reading_order(lines)
    med_h = _median_line_height(ordered)
    gap_thresh = max(18.0, med_h * 1.6)

    paras: List[List[str]] = []
    cur: List[str] = []

    prev_y2: Optional[int] = None
    for ln in ordered:
        txt = _clean_text(ln.text)
        if not txt:
            continue

        x1, y1, x2, y2 = ln.bbox
        if prev_y2 is not None and (y1 - prev_y2) > gap_thresh:
            if cur:
                paras.append(cur)
                cur = []
        cur.append(txt)
        prev_y2 = y2

    if cur:
        paras.append(cur)

    out: List[str] = []
    for p in paras:
        joined = " ".join(p).strip()
        if joined:
            out.append(joined)
    return out


# ----------------------------
# Region detection (fast OpenCV)
# ----------------------------

def _detect_text_regions(frame_bgr: np.ndarray) -> List[BBox]:
    """
    Heuristic text region detection using morphology + contours.
    Returns list of bboxes in original frame coords.
    """
    if frame_bgr is None:
        return []

    H, W = frame_bgr.shape[:2]
    if H < 40 or W < 40:
        return []

    gray = cv.cvtColor(frame_bgr, cv.COLOR_BGR2GRAY)
    gray = cv.bilateralFilter(gray, 7, 50, 50)

    grad = cv.morphologyEx(gray, cv.MORPH_GRADIENT, np.ones((3, 3), np.uint8))

    bw = cv.adaptiveThreshold(
        grad, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 21, 5
    )

    kernel = cv.getStructuringElement(cv.MORPH_RECT, (19, 5))
    connected = cv.morphologyEx(bw, cv.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv.findContours(connected, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    boxes: List[BBox] = []
    for c in contours:
        x, y, w, h = cv.boundingRect(c)
        area = w * h
        if area < 800:
            continue

        aspect = w / float(h + 1e-6)
        if aspect < 1.1:
            continue

        if h < 18:
            continue

        area_ratio = area / float(W * H)
        if area_ratio > 0.85:
            continue

        boxes.append((x, y, x + w, y + h))

    if not boxes:
        return []

    boxes = sorted(boxes, key=lambda b: (b[1], b[0]))

    merged: List[BBox] = []
    for b in boxes:
        if not merged:
            merged.append(b)
            continue
        x1, y1, x2, y2 = b
        mx1, my1, mx2, my2 = merged[-1]

        inter_x1 = max(x1, mx1)
        inter_y1 = max(y1, my1)
        inter_x2 = min(x2, mx2)
        inter_y2 = min(y2, my2)
        overlap = (inter_x2 > inter_x1) and (inter_y2 > inter_y1)

        near = abs(y1 - my2) < 18 and abs(x1 - mx1) < 60

        if overlap or near:
            merged[-1] = (min(x1, mx1), min(y1, my1), max(x2, mx2), max(y2, my2))
        else:
            merged.append(b)

    merged = sorted(merged, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]), reverse=True)
    merged = merged[: int(getattr(config, "OCR_MAX_REGIONS", 8) or 8)]

    pad = int(getattr(config, "OCR_REGION_PADDING_PX", 8) or 8)
    out: List[BBox] = []
    for b in merged:
        out.append(_pad_bbox(b, pad, W, H))
    return out


# ----------------------------
# Local OCR engines
# ----------------------------

class _EasyOCRBackend:
    def __init__(self, langs: List[str]):
        self._langs = langs
        self._reader = None

    def _lazy(self):
        if self._reader is not None:
            return
        import easyocr  # noqa
        self._reader = easyocr.Reader(self._langs, gpu=False)

    def run(self, img_bgr: np.ndarray) -> List[OCRLine]:
        self._lazy()
        rgb = cv.cvtColor(img_bgr, cv.COLOR_BGR2RGB)
        results = self._reader.readtext(rgb, detail=1, paragraph=False)

        lines: List[OCRLine] = []
        for item in results:
            pts, text, conf = item
            text = _clean_text(text)
            if not text:
                continue
            conf = float(conf) if conf is not None else 0.0
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            x1, y1, x2, y2 = int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))
            lines.append(OCRLine(text=text, conf=max(0.0, min(1.0, conf)), bbox=(x1, y1, x2, y2)))
        return lines


class _TesseractBackend:
    def __init__(self):
        self._ok = False
        try:
            import pytesseract  # noqa
            self._ok = True
        except Exception:
            self._ok = False

    def run(self, img_bgr: np.ndarray) -> List[OCRLine]:
        if not self._ok:
            return []
        import pytesseract

        gray = cv.cvtColor(img_bgr, cv.COLOR_BGR2GRAY)

        data = pytesseract.image_to_data(
            gray,
            output_type=pytesseract.Output.DICT,
            config="--oem 1 --psm 6",
        )
        n = len(data.get("text", []))

        grouped: Dict[Tuple[int, int, int], List[int]] = {}
        for i in range(n):
            txt = (data["text"][i] or "").strip()
            if not txt:
                continue
            conf = float(data.get("conf", [0] * n)[i])
            if conf < 0:
                continue
            key = (int(data["block_num"][i]), int(data["par_num"][i]), int(data["line_num"][i]))
            grouped.setdefault(key, []).append(i)

        lines: List[OCRLine] = []
        for _, idxs in grouped.items():
            words: List[str] = []
            confs: List[float] = []
            xs: List[int] = []
            ys: List[int] = []
            x2s: List[int] = []
            y2s: List[int] = []

            for i in idxs:
                txt = (data["text"][i] or "").strip()
                if not txt:
                    continue
                words.append(txt)
                c = float(data.get("conf", [0] * n)[i])
                confs.append(max(0.0, min(100.0, c)))

                x = int(data["left"][i])
                y = int(data["top"][i])
                w = int(data["width"][i])
                h = int(data["height"][i])

                xs.append(x)
                ys.append(y)
                x2s.append(x + w)
                y2s.append(y + h)

            line_text = _clean_text(" ".join(words))
            if not line_text:
                continue

            conf01 = (float(np.median(confs)) / 100.0) if confs else 0.0
            bbox = (int(min(xs)), int(min(ys)), int(max(x2s)), int(max(y2s)))
            lines.append(OCRLine(text=line_text, conf=conf01, bbox=bbox))

        return lines


def _preprocess_for_ocr(img_bgr: np.ndarray) -> np.ndarray:
    if img_bgr is None:
        return img_bgr

    scale = float(getattr(config, "OCR_PREPROCESS_SCALE", 1.6) or 1.6)
    if scale > 1.01:
        h, w = img_bgr.shape[:2]
        img_bgr = cv.resize(img_bgr, (int(w * scale), int(h * scale)), interpolation=cv.INTER_CUBIC)

    gray = cv.cvtColor(img_bgr, cv.COLOR_BGR2GRAY)
    gray = cv.GaussianBlur(gray, (3, 3), 0)

    if getattr(config, "OCR_BINARIZE", True):
        bw = cv.adaptiveThreshold(
            gray, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 31, 9
        )
        img_bgr = cv.cvtColor(bw, cv.COLOR_GRAY2BGR)
    else:
        img_bgr = cv.cvtColor(gray, cv.COLOR_GRAY2BGR)

    return img_bgr


# ----------------------------
# Hybrid OCR engine
# ----------------------------

class OCREngine:
    """
    OCR modes:
      - offline / local_only: local OCR only (fast, works without internet)
      - hybrid: local OCR, escalate to Scene AI if low confidence/too short
      - ai / scene_only: always use Scene AI OCR

    Notes:
      - We include a cooldown for Scene OCR to avoid spamming.
      - For explicit "read this" commands, use force=True to bypass cooldown.
    """

    def __init__(self):
        langs = [s.strip() for s in (getattr(config, "OCR_LANGS", "en") or "en").split(",") if s.strip()]
        self._easy = _EasyOCRBackend(langs)
        self._tess = _TesseractBackend()

        self._last_scene_ts: float = 0.0
        self._scene_cooldown_s: float = float(getattr(config, "OCR_SCENE_COOLDOWN_S", 1.2) or 1.2)

    def _run_local(self, img_bgr: np.ndarray, engine: str) -> List[OCRLine]:
        img = _preprocess_for_ocr(img_bgr)

        if engine == "easyocr":
            try:
                return self._easy.run(img)
            except Exception as e:
                print(f"⚠️ EasyOCR failed: {e!r}")
                return []

        if engine == "tesseract":
            try:
                return self._tess.run(img)
            except Exception as e:
                print(f"⚠️ Tesseract failed: {e!r}")
                return []

        # auto: try easyocr first, then tesseract
        try:
            lines = self._easy.run(img)
        except Exception:
            lines = []
        if lines:
            return lines
        return self._tess.run(img)

    def _score_conf(self, lines: List[OCRLine]) -> float:
        if not lines:
            return 0.0
        weights: List[float] = []
        vals: List[float] = []
        for ln in lines:
            t = _clean_text(ln.text)
            if not t:
                continue
            w = max(1.0, float(len(t)))
            weights.append(w)
            vals.append(float(ln.conf))
        if not vals:
            return 0.0
        return float(np.average(vals, weights=weights))

    def _scene_ai_ocr(self, frame_bgr: np.ndarray, force: bool = False) -> OCRResult:
        if client is None:
            return OCRResult(text="", conf=0.0, lines=[], engine="scene_ai", used_regions=False)

        now = _now()
        if (not force) and (now - self._last_scene_ts) < self._scene_cooldown_s:
            return OCRResult(text="", conf=0.0, lines=[], engine="scene_ai", used_regions=False)

        self._last_scene_ts = now

        try:
            data_url = _frame_to_data_url(
                frame_bgr,
                quality=int(getattr(config, "SCENE_AI_JPEG_QUALITY", 75) or 75),
            )
            prompt = (
                "Read ALL visible text in this image.\n"
                "Return plain text only.\n"
                "Preserve reading order (top-to-bottom, left-to-right).\n"
                "If multiple sections, separate paragraphs with blank lines.\n"
                "Do NOT add commentary."
            )

            resp = client.chat.completions.create(
                model=getattr(config, "OPENAI_VISION_MODEL", "gpt-4o"),
                messages=[
                    {"role": "system", "content": "You are an OCR engine. Output only the text you read."},
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {"type": "image_url", "image_url": {"url": data_url}},
                        ],
                    },
                ],
                max_tokens=int(getattr(config, "SCENE_OCR_MAX_TOKENS", 700) or 700),
                temperature=float(getattr(config, "SCENE_OCR_TEMPERATURE", 0.0) or 0.0),
            )

            text = _clean_text((resp.choices[0].message.content or ""))
            return OCRResult(text=text, conf=0.95 if text else 0.0, lines=[], engine="scene_ai", used_regions=False)

        except Exception as e:
            print(f"⚠️ Scene AI OCR failed: {e!r}")
            return OCRResult(text="", conf=0.0, lines=[], engine="scene_ai", used_regions=False)

    def _normalize_mode(self, mode: Optional[str]) -> str:
        """
        Normalizes many aliases into one of:
          - "hybrid"
          - "local_only"
          - "scene_only"
        """
        raw = (mode or "").strip().lower()

        # If caller didn't pass mode, try config defaults
        if not raw:
            # prefer READING_MODE_DEFAULT ("offline"|"hybrid"|"ai")
            raw = (getattr(config, "READING_MODE_DEFAULT", "") or "").strip().lower()

        if not raw:
            # backward compat: OCR_MODE might be "hybrid"|"local_only"|"scene_only"
            raw = (getattr(config, "OCR_MODE", "hybrid") or "hybrid").strip().lower()

        # Map human modes
        if raw in {"offline", "local", "localonly", "local_only"}:
            return "local_only"
        if raw in {"ai", "scene", "openai", "sceneonly", "scene_only"}:
            return "scene_only"

        # Keep your original internal naming too
        if raw in {"hybrid", "local_only", "scene_only"}:
            return raw

        # Your old docstring modes
        if raw in {"local-only", "localonly"}:
            return "local_only"
        if raw in {"scene-only", "sceneonly"}:
            return "scene_only"

        return "hybrid"

    def read(self, frame_bgr: np.ndarray, mode: Optional[str] = None, *, force_ai: bool = False) -> OCRResult:
        """
        One-shot OCR read.

        mode (accepted aliases):
          - offline / local / local_only / local_only
          - hybrid
          - ai / scene / scene_only / openai

        force_ai:
          - if True, bypass scene cooldown for this call when using Scene OCR
          - use this for explicit "read this" interactions
        """
        if frame_bgr is None:
            return OCRResult(text="", conf=0.0, lines=[], engine="none", used_regions=False)

        mode_final = self._normalize_mode(mode)

        # 0) AI-only mode
        if mode_final == "scene_only":
            scene = self._scene_ai_ocr(frame_bgr, force=True if force_ai else True)
            if scene.text:
                return scene
            # If AI unavailable, fall back to local so user gets *something*
            mode_final = "local_only"

        engine = (getattr(config, "OCR_ENGINE", "auto") or "auto").strip().lower()
        use_regions = bool(getattr(config, "OCR_USE_REGION_DETECTION", True))

        H, W = frame_bgr.shape[:2]

        regions: List[BBox] = []
        if use_regions:
            try:
                regions = _detect_text_regions(frame_bgr)
            except Exception as e:
                print(f"⚠️ Text region detection failed: {e!r}")
                regions = []

        all_lines: List[OCRLine] = []

        if regions:
            for b in regions:
                x1, y1, x2, y2 = _clamp_bbox(*b, W, H)
                crop = frame_bgr[y1:y2, x1:x2]
                lines = self._run_local(crop, engine=engine)
                for ln in lines:
                    bx1, by1, bx2, by2 = ln.bbox
                    all_lines.append(
                        OCRLine(
                            text=ln.text,
                            conf=ln.conf,
                            bbox=(bx1 + x1, by1 + y1, bx2 + x1, by2 + y1),
                        )
                    )
        else:
            all_lines = self._run_local(frame_bgr, engine=engine)

        ordered = _sort_reading_order(all_lines)
        paragraphs = _lines_to_paragraphs(ordered)
        text = "\n\n".join(paragraphs).strip()

        conf = self._score_conf(ordered)
        min_conf = float(getattr(config, "OCR_LOCAL_MIN_CONF", 0.55) or 0.55)
        min_chars = int(getattr(config, "OCR_MIN_CHARS", 12) or 12)

        used_regions_flag = bool(regions)

        # 1) Hybrid escalation only
        if mode_final == "hybrid":
            if (not text) or (len(text) < min_chars) or (conf < min_conf):
                scene = self._scene_ai_ocr(frame_bgr, force=True if force_ai else True)
                if scene.text:
                    return scene

        # Determine actual engine used (best-effort)
        used_engine = engine
        if engine == "auto":
            used_engine = "easyocr" if ordered else "tesseract"

        return OCRResult(text=text, conf=conf, lines=ordered, engine=used_engine, used_regions=used_regions_flag)

    def read_paragraphs(self, frame_bgr: np.ndarray, mode: Optional[str] = None, *, force_ai: bool = False) -> Tuple[List[str], float, str]:
        """
        Returns (paragraphs, confidence, engine_used)
        """
        r = self.read(frame_bgr, mode=mode, force_ai=force_ai)

        if r.engine == "scene_ai":
            paras = [p.strip() for p in (r.text.split("\n\n")) if p.strip()]
            return paras, r.conf, r.engine

        paras = _lines_to_paragraphs(r.lines)
        return paras, r.conf, r.engine
