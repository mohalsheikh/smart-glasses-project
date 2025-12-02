# test_speech.py
# Live YOLO + TTS: announces detections (conf >= 0.50) repeatedly while visible,
# with debug logs and a robust TTS fallback for systems where pyttsx3 stalls.

import sys
import time
import threading
from queue import Queue, Empty

import pyttsx3
import cv2 as cv

from src.camera_handler import CameraHandler
from src.object_detector import ObjectDetector


# TTS worker
class TTSAnnouncer:
    """
    Queue-based TTS. If reinit_each_say=True, we re-create the engine for each
    utterance (slower but very robust on some Windows installs).
    """
    def __init__(self, rate=180, volume=1.0, voice_index=0, reinit_each_say=False):
        self.rate = rate
        self.volume = volume
        self.voice_index = voice_index
        self.reinit_each_say = reinit_each_say

        # Pick a driver explicitly (Windows/macOS/Linux)
        self.driver = {"win32": "sapi5", "darwin": "nsss"}.get(sys.platform, "espeak")

        # Persistent engine (used only if reinit_each_say=False)
        self.engine = None
        if not self.reinit_each_say:
            self.engine = pyttsx3.init(driverName=self.driver)
            self._configure_engine(self.engine)

        self._q: Queue[str] = Queue()
        self._stop = threading.Event()
        self._thr = threading.Thread(target=self._loop, daemon=True)
        self._thr.start()

    def _configure_engine(self, engine):
        engine.setProperty("rate", self.rate)
        engine.setProperty("volume", self.volume)
        voices = engine.getProperty("voices")
        if voices and 0 <= self.voice_index < len(voices):
            engine.setProperty("voice", voices[self.voice_index].id)

    def _loop(self):
        while not self._stop.is_set():
            try:
                text = self._q.get(timeout=0.2)
            except Empty:
                continue
            try:
                if self.reinit_each_say:
                    # Robust mode: make a fresh engine each time
                    eng = pyttsx3.init(driverName=self.driver)
                    self._configure_engine(eng)
                    eng.say(text)
                    eng.runAndWait()
                    try:
                        eng.stop()
                    except Exception:
                        pass
                else:
                    # Normal mode: reuse one engine
                    if self.engine is None:
                        self.engine = pyttsx3.init(driverName=self.driver)
                        self._configure_engine(self.engine)
                    self.engine.say(text)
                    self.engine.runAndWait()
            except Exception:
                # Keep going even if TTS hiccups
                pass

    def say(self, text: str):
        if text:
            self._q.put(text)

    def toggle_reinit(self):
        """Switch between persistent engine and per-utterance engines."""
        self.reinit_each_say = not self.reinit_each_say
        if self.reinit_each_say and self.engine is not None:
            # Drop the persistent engine to avoid conflicts
            try:
                self.engine.stop()
            except Exception:
                pass
            self.engine = None
        return self.reinit_each_say

    def stop(self):
        # Drain quickly and stop
        try:
            with self._q.mutex:
                self._q.queue.clear()
        except Exception:
            pass
        self._stop.set()
        try:
            if self.engine is not None:
                self.engine.stop()
        except Exception:
            pass
        self._thr.join(timeout=1.5)


# ---------- helpers ----------
def side_from_center(cx: float, frame_w: int) -> str | None:
    if frame_w <= 0:
        return None
    x = cx / frame_w
    if x < 0.40:
        return "left"
    if x > 0.60:
        return "right"
    return "center"


# ---------- main ----------
def main():
    cam = CameraHandler()

    # Keep model's own threshold modest; the SPEAK threshold below is the gate.
    try:
        det = ObjectDetector(conf=0.25)
    except TypeError:
        det = ObjectDetector()

    # Start TTS (begin in normal mode; you can toggle robust mode with 'E')
    tts = TTSAnnouncer(rate=180, volume=1.0, reinit_each_say=False)

    # Tunables
    SPEAK_INTERVAL = 2.0      # seconds between repeats for the same visible thing
    MIN_CONF = 0.50           # announce only if confidence >= 0.50
    MAX_PER_FRAME = 2         # cap how many phrases we enqueue per frame
    DEBUG = True              # print what we detect and when we speak

    # last time we spoke for a given key (track-based if available; else label+zone)
    last_spoken = {}
    muted = False

    try:
        while True:
            frame = cam.capture_frame()
            if frame is None:
                if DEBUG:
                    print("[debug] capture_frame returned None")
                continue

            detections, annotated = det.detect(frame, annotate=True)
            h, w = annotated.shape[:2]
            now = time.monotonic()

            if DEBUG:
                # Print a brief summary of detections w/ confidences
                summary = ", ".join(
                    f"{d.get('label','?')}:{float(d.get('confidence',0.0)):.2f}"
                    for d in detections
                )
                print(f"[debug] detections: {summary or 'none'}")

            spoken_this_frame = 0

            for d in detections:
                if spoken_this_frame >= MAX_PER_FRAME:
                    break

                label = d.get("label") or ""
                conf = float(d.get("confidence", 0.0))
                track_id = d.get("track_id")
                center = d.get("center")

                if not label or conf < MIN_CONF:
                    continue

                zone = None
                if isinstance(center, (list, tuple)) and len(center) >= 2:
                    zone = side_from_center(center[0], w)

                # Prefer stable track id; else label+zone
                if track_id is not None:
                    key = f"id:{track_id}"
                else:
                    key = f"{label}:{zone}"

                last = last_spoken.get(key, 0.0)
                if now - last < SPEAK_INTERVAL:
                    continue  # not time yet to repeat

                # Phrase (use comma for clearer speech)
                phrase = f"{label}, {zone}" if zone else label

                if not muted:
                    if DEBUG:
                        print(f"[speak] {phrase} (conf={conf:.2f}) key={key}")
                    tts.say(phrase)
                    spoken_this_frame += 1
                    last_spoken[key] = now

            cam.show_image(annotated, window_name="Camera (Q=quit, M=mute, E=robust TTS, T=test)")

            key = cv.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("m"):
                muted = not muted
                tts.say("Muted" if muted else "Unmuted")
            elif key == ord("e"):
                mode = tts.toggle_reinit()
                tts.say("Robust mode on" if mode else "Robust mode off")
                if DEBUG:
                    print(f"[debug] robust TTS mode: {mode}")
            elif key == ord("t"):
                # Manual test to confirm TTS is healthy
                tts.say("Test phrase from smart glasses")

    finally:
        tts.stop()
        # CameraHandler.__del__ releases camera and closes windows


if __name__ == "__main__":
    main()
