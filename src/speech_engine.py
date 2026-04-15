# src/speech_engine.py
"""
Speech Engine with Interrupt Support
=====================================

Features:
- Queued speech (never overlaps)
- INTERRUPT capability - stops current speech immediately
- Clear queue functionality
- Dedupe to avoid rapid repeats

Use interrupt() when user wants to speak - stops TTS immediately.
"""

from __future__ import annotations

import platform
import subprocess
import time
import threading
import os
import signal
from queue import Queue, Empty
from typing import Optional


class SpeechEngine:
    """
    Text-to-speech wrapper with interrupt support.
    
    Key feature: interrupt() method stops ANY current speech immediately,
    useful when user presses a key to speak.
    """

    def __init__(self, debug: bool = True, dedupe_window_seconds: float = 0.75):
        self.debug = debug
        self._platform = platform.system().lower()
        backend = "macOS 'say' backend" if "darwin" in self._platform else "console print backend"

        self._queue: "Queue[str]" = Queue()
        self._stop_event = threading.Event()
        self._interrupt_event = threading.Event()
        self._worker = threading.Thread(target=self._run_worker, daemon=True)
        self._worker.start()

        # Track current subprocess for interruption
        self._current_process: Optional[subprocess.Popen] = None
        self._process_lock = threading.Lock()

        self._last_spoken_text: str = ""
        self._last_spoken_time: float = 0.0
        self._dedupe_window = max(0.0, float(dedupe_window_seconds))
        
        # Track if we're currently speaking
        self._is_speaking = False

        print(f"🗣️  SpeechEngine initialized ({backend}, queued worker, interrupt support, debug={debug})")

    def speak(self, text: str) -> None:
        """
        Enqueue speech to be spoken sequentially.
        Non-blocking.
        """
        text = (text or "").strip()
        if not text:
            return

        # Dedupe: avoid rapid repeats
        now = time.time()
        if self._dedupe_window > 0 and text == self._last_spoken_text and (now - self._last_spoken_time) < self._dedupe_window:
            if self.debug:
                print("🗣️  [SpeechEngine] dedupe: skipping repeated text")
            return

        self._queue.put(text)

    def clear_queue(self) -> None:
        """Clear all pending speech (doesn't stop current speech)."""
        try:
            while True:
                self._queue.get_nowait()
        except Exception:
            pass

    def interrupt(self) -> None:
        """
        IMMEDIATELY stop any current speech and clear the queue.
        Call this when user wants to speak.
        """
        # Clear pending queue first
        self.clear_queue()
        
        # Set interrupt flag
        self._interrupt_event.set()
        
        # Kill current speech process
        with self._process_lock:
            if self._current_process is not None:
                try:
                    if self.debug:
                        print("🔇 [SpeechEngine] Interrupting speech...")
                    
                    # On macOS, kill the 'say' process
                    if "darwin" in self._platform:
                        # Kill the specific process
                        self._current_process.terminate()
                        try:
                            self._current_process.wait(timeout=0.5)
                        except subprocess.TimeoutExpired:
                            self._current_process.kill()
                        
                        # Also kill any other 'say' processes (belt and suspenders)
                        try:
                            subprocess.run(["pkill", "-9", "say"], 
                                         capture_output=True, timeout=0.5)
                        except Exception:
                            pass
                    else:
                        self._current_process.terminate()
                    
                    self._current_process = None
                except Exception as e:
                    if self.debug:
                        print(f"⚠️ [SpeechEngine] Interrupt error: {e}")
        
        # Clear the interrupt flag after a moment
        time.sleep(0.1)
        self._interrupt_event.clear()
        self._is_speaking = False

    def is_speaking(self) -> bool:
        """Check if currently speaking."""
        return self._is_speaking

    def stop(self) -> None:
        """Stop the worker thread completely."""
        self._stop_event.set()
        self.interrupt()
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

            # Check for interrupt before starting
            if self._interrupt_event.is_set():
                continue

            thread_id = threading.get_ident()
            if self.debug:
                print(f"🗣️  [SpeechEngine] speaking on worker thread {thread_id} ({len(text)} chars)")

            start = time.time()

            # Record last spoken for dedupe
            self._last_spoken_text = text
            self._last_spoken_time = time.time()
            self._is_speaking = True

            if "darwin" in self._platform:
                try:
                    # Use Popen so we can interrupt
                    with self._process_lock:
                        self._current_process = subprocess.Popen(
                            ["say", text],
                            stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL
                        )
                    
                    # Wait for completion or interrupt
                    while self._current_process.poll() is None:
                        if self._interrupt_event.is_set():
                            break
                        time.sleep(0.05)
                    
                    with self._process_lock:
                        self._current_process = None
                        
                except Exception as e:
                    print(f"⚠️ SpeechEngine 'say' failed: {e!r}")
                    print(text)
            else:
                print(f"[TTS] {text}")

            self._is_speaking = False
            
            if self.debug and not self._interrupt_event.is_set():
                print(f"✅ [SpeechEngine] finished in {time.time() - start:.2f}s")