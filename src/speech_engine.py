# src/speech_engine.py

from __future__ import annotations

import platform
import subprocess
import time
import threading
from queue import Queue, Empty
from typing import Optional


class SpeechEngine:
    """
    Text-to-speech wrapper with a single-worker queue.

    Upgrade:
      ✅ All speech is serialized through one background worker thread
         so outputs NEVER overlap (very important for polished UX).

    Extra polish:
      ✅ Optional dedupe: avoids repeating the exact same phrase back-to-back
      ✅ clear_queue() utility
    """

    def __init__(self, debug: bool = True, dedupe_window_seconds: float = 0.75):
        self.debug = debug
        self._platform = platform.system().lower()
        backend = "macOS 'say' backend" if "darwin" in self._platform else "console print backend"

        self._queue: "Queue[str]" = Queue()
        self._stop_event = threading.Event()
        self._worker = threading.Thread(target=self._run_worker, daemon=True)
        self._worker.start()

        self._last_spoken_text: str = ""
        self._last_spoken_time: float = 0.0
        self._dedupe_window = max(0.0, float(dedupe_window_seconds))

        print(f"🗣️  SpeechEngine initialized ({backend}, queued worker, debug={debug})")

    def speak(self, text: str) -> None:
        """
        Enqueue speech to be spoken sequentially.
        Non-blocking.
        """
        text = (text or "").strip()
        if not text:
            return

        # Dedupe: avoid rapid repeats (useful for auto-speak)
        now = time.time()
        if self._dedupe_window > 0 and text == self._last_spoken_text and (now - self._last_spoken_time) < self._dedupe_window:
            if self.debug:
                print("🗣️  [SpeechEngine] dedupe: skipping repeated text")
            return

        self._queue.put(text)

    def clear_queue(self) -> None:
        """Best-effort queue clear (keeps worker alive)."""
        try:
            while True:
                self._queue.get_nowait()
        except Exception:
            pass

    def stop(self) -> None:
        """Stop the worker thread."""
        self._stop_event.set()
        self._queue.put("")  # unblock
        try:
            self._worker.join(timeout=1.0)
        except Exception:
            pass

    # ------------------------------------------------------------------

    def _run_worker(self) -> None:
        while not self._stop_event.is_set():
            try:
                text = self._queue.get(timeout=0.25)
            except Empty:
                continue

            if self._stop_event.is_set():
                break

            text = (text or "").strip()
            if not text:
                continue

            thread_id = threading.get_ident()
            if self.debug:
                print(f"🗣️  [SpeechEngine] speaking on worker thread {thread_id} ({len(text)} chars)")

            start = time.time()

            # record last spoken for dedupe
            self._last_spoken_text = text
            self._last_spoken_time = time.time()

            if "darwin" in self._platform:
                try:
                    subprocess.run(["say", text], check=False)
                except Exception as e:
                    print(f"⚠️ SpeechEngine 'say' failed: {e!r}")
                    print(text)
            else:
                print(f"[TTS] {text}")

            if self.debug:
                print(f"✅ [SpeechEngine] finished in {time.time() - start:.2f}s")
