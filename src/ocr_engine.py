"""
General OCR engine with EasyOCR.
- Lazily initializes the reader on first use to avoid startup failures (e.g., SSL/cert issues).
"""

from typing import Optional, List
import re
import cv2

try:
    import easyocr
except Exception as e:
    easyocr = None
    print(f"⚠️ EasyOCR import failed: {e}. Install with `pip install easyocr`")


class OCREngine:
    def __init__(self, langs: Optional[List[str]] = None):
        self.langs = langs or ["en"]
        self.reader = None  # lazy

    def _ensure_reader(self):
        if self.reader is None:
            if easyocr is None:
                raise RuntimeError("EasyOCR is not installed. Run `pip install easyocr`.")
            self.reader = easyocr.Reader(self.langs, gpu=False)

    def read_text(self, frame_bgr) -> str:
        try:
            self._ensure_reader()
        except Exception as e:
            # If model download fails, keep the app running
            print(f"⚠️ OCR init failed: {e}")
            return ""

        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        result = self.reader.readtext(rgb, detail=0, paragraph=True)
        text = " ".join(result)
        text = re.sub(r"\s+", " ", text).strip()
        return text
