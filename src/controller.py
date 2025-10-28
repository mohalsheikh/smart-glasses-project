"""
Controller (fast path):
- MPS auto-select on Apple Silicon (if available)
- Frame skipping with reuse (huge FPS win)
- Lighter HUD drawing
- Key controls: Q quit, S speak, V voice, O OCR, C currency, T toggle tiny-boost
"""

import time
from collections import deque, Counter
import cv2

from src.camera_handler import CameraHandler
from src.object_detector import ObjectDetector
from src.ocr_engine import OCREngine
from src.speech_engine import SpeechEngine
from src.utils.preprocessing import to_gray

# Try to use MPS if present (Apple Silicon)
try:
    import torch
    USE_MPS = bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available())
except Exception:
    USE_MPS = False


class EMA:
    def __init__(self, alpha: float = 0.22):
        self.alpha = alpha
        self.value = None

    def update(self, x: float) -> float:
        self.value = x if self.value is None else (self.alpha * x + (1 - self.alpha) * self.value)
        return self.value or 0.0


class Cooldown:
    def __init__(self, seconds: float):
        self.seconds = seconds
        self.t_last = {}

    def ready(self, key: str) -> bool:
        t = time.time()
        if key not in self.t_last or (t - self.t_last[key]) >= self.seconds:
            self.t_last[key] = t
            return True
        return False

    def mark(self, key: str):
        self.t_last[key] = time.time()


class MainController:
    def __init__(self):
        print("🔧 Initializing system components...")
        self.camera = CameraHandler()

        self.detector = ObjectDetector(
            model_name="yolov8n.pt",
            conf=0.50,
            iou=0.50,
            imgsz=640,
            device=("mps" if USE_MPS else None),
            track=False,                 # tracking kills FPS on laptops—off
            max_det=150,
            tiny_boost_every=6           # every 6th frame run 2x2 boost; press 'T' to toggle
        )

        self.speech = SpeechEngine()
        self.ocr = None
        self.currency = None
        print(f"✅ Core components initialized. Device: {'MPS' if USE_MPS else 'CPU'}")

        # Toggles
        self.voice_on = True
        self.do_ocr = False
        self.do_currency = False
        self.tiny_boost_enabled = True

        # Perf
        self.downscale = 0.66     # render smaller for display; detection runs at imgsz inside detector
        self.frame_skip = 2       # run detector every N frames; reuse last dets in between
        self._skip_ctr = 0

        # State
        self.fps_ema = EMA()
        self.last_labels = Counter()
        self.cool = Cooldown(3.5)
        self.hud_font = cv2.FONT_HERSHEY_SIMPLEX

        # cache last outputs to reuse on skipped frames
        self._last_dets = []
        self._last_annot = None

    # ---- helpers ----
    def _prep(self, frame):
        if self.downscale and self.downscale != 1.0:
            h, w = frame.shape[:2]
            frame = cv2.resize(frame, (int(w * self.downscale), int(h * self.downscale)))
        return frame

    def _labels_summary(self, dets) -> Counter:
        return Counter([d["label"] for d in dets])

    def _draw_hud(self, frame, fps, status: str):
        # Very light HUD to reduce draw cost
        y = 22
        cv2.putText(frame, f"{fps:.1f} FPS | Voice:{'ON' if self.voice_on else 'OFF'} | "
                           f"OCR:{'ON' if self.do_ocr else 'OFF'} | Curr:{'ON' if self.do_currency else 'OFF'}",
                    (10, y), self.hud_font, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        y += 20
        if status:
            cv2.putText(frame, status[:64], (10, y), self.hud_font, 0.5, (255, 220, 180), 1, cv2.LINE_AA)
        return frame

    def _speak(self, text: str):
        if self.voice_on and text:
            self.speech.speak(text)

    # ---- main ----
    def run(self):
        print("🚀 Smart Glasses System Starting…")
        print("🎥 Controls: [Q] quit  [S] speak  [V] voice  [O] OCR  [C] currency  [T] tiny-boost on/off")

        while True:
            t0 = time.time()
            frame = self.camera.capture_frame()
            if frame is None:
                print("⚠️ No frame captured. Check camera permissions.")
                break

            frame_small = self._prep(frame)

            # Decide whether to run detector this frame
            self._skip_ctr = (self._skip_ctr + 1) % max(1, self.frame_skip)
            run_now = (self._skip_ctr == 0)

            # Detection
            if run_now:
                try:
                    # flip tiny-boost in detector based on toggle
                    self.detector.tiny_boost_every = 6 if self.tiny_boost_enabled else 0
                    dets, annotated = self.detector.detect(frame_small, visualize=True)
                    self._last_dets, self._last_annot = dets, annotated
                except Exception as e:
                    print(f"❌ Detection error: {e}")
                    dets, annotated = self._last_dets, (self._last_annot if self._last_annot is not None else frame_small)
            else:
                # reuse last results (cheap)
                dets, annotated = self._last_dets, (self._last_annot if self._last_annot is not None else frame_small)

            # Optional OCR
            text_result = ""
            if run_now and self.do_ocr:
                if self.ocr is None:
                    try:
                        self.ocr = OCREngine(["en"])
                    except Exception as e:
                        self.do_ocr = False
                        print(f"OCR init failed: {e}")
                if self.ocr:
                    try:
                        gray = to_gray(frame_small)
                        text_result = self.ocr.read_text(gray)
                    except Exception as e:
                        print(f"⚠️ OCR error: {e}")

            # Optional currency
            curr_result = ""
            if run_now and self.do_currency:
                if self.currency is None:
                    try:
                        from src.currency_recognizer import CurrencyRecognizer
                        self.currency = CurrencyRecognizer()
                    except Exception as e:
                        self.do_currency = False
                        print(f"Currency init failed: {e}")
                if self.currency:
                    try:
                        c_sum, c_hits, c_img = self.currency.recognize(frame_small)
                        annotated = cv2.addWeighted(annotated, 1.0, c_img, 0.55, 0)
                        curr_result = c_sum
                    except Exception as e:
                        print(f"⚠️ Currency error: {e}")

            # Smart speech: announce when objects change (cooldown)
            cur_labels = self._labels_summary(dets)
            status_line = ""
            if cur_labels != self.last_labels and self.cool.ready("objects"):
                if cur_labels:
                    parts = [f"{k} x{v}" for k, v in cur_labels.most_common()]
                    phrase = "I see " + ", ".join(parts)
                else:
                    phrase = "I don't see any known objects"
                self._speak(phrase)
                status_line = phrase
                self.last_labels = cur_labels.copy()

            # Speak OCR/currency occasionally
            if text_result and self.cool.ready("ocr"):
                self._speak(f"Text: {text_result[:80]}")
                status_line = f"Text: {text_result[:48]}"
            if curr_result and self.cool.ready("currency"):
                self._speak(curr_result)
                status_line = curr_result

            # HUD & display
            dt = max(1e-6, time.time() - t0)
            fps = self.fps_ema.update(1.0 / dt)
            annotated = self._draw_hud(annotated, fps, status_line)
            cv2.imshow("Smart Glasses Feed", annotated)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("s"):
                summary = ", ".join([f"{k} x{v}" for k, v in cur_labels.items()]) or "No objects"
                summary += (f". Text: {text_result[:60]}" if text_result else "")
                summary += (f". {curr_result}" if curr_result else "")
                self._speak(summary.strip())
            elif key == ord("v"):
                self.voice_on = not self.voice_on
            elif key == ord("o"):
                self.do_ocr = not self.do_ocr
                if self.do_ocr and self.ocr is None:
                    try:
                        self.ocr = OCREngine(["en"])
                    except Exception as e:
                        self.do_ocr = False
                        print(f"OCR init failed: {e}")
            elif key == ord("c"):
                self.do_currency = not self.do_currency
                if self.do_currency and self.currency is None:
                    try:
                        from src.currency_recognizer import CurrencyRecognizer
                        self.currency = CurrencyRecognizer()
                    except Exception as e:
                        self.do_currency = False
                        print(f"Currency init failed: {e}")
            elif key == ord("t"):
                self.tiny_boost_enabled = not self.tiny_boost_enabled

        self.camera.release()
        cv2.destroyAllWindows()
        print("👋 Camera feed stopped.")
