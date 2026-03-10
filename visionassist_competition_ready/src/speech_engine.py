# src/speech_engine.py
"""
Speech Engine with Interrupt Support
=====================================

Cross-platform TTS:
- macOS: 'say' command (high quality)
- Linux/Pi: espeak-ng -> espeak -> piper -> festival (auto-detect)
- Windows: pyttsx3
- Fallback: console print

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
import shutil
import time
import threading
import os
import signal
from queue import Queue, Empty
from typing import Optional


def _find_linux_tts() -> Optional[str]:
    """Detect available TTS engine on Linux."""
    for cmd in ["espeak-ng", "espeak", "piper", "festival"]:
        if shutil.which(cmd):
            return cmd
    return None


class SpeechEngine:
    """
    Text-to-speech wrapper with interrupt support.
    
    Key feature: interrupt() method stops ANY current speech immediately,
    useful when user presses a key to speak.
    """

    def __init__(self, debug: bool = True, dedupe_window_seconds: float = 0.75):
        self.debug = debug
        self._platform = platform.system().lower()
        self._linux_tts: Optional[str] = None

        # Detect TTS backend
        if "darwin" in self._platform:
            backend = "macOS 'say'"
        elif "linux" in self._platform:
            self._linux_tts = _find_linux_tts()
            if self._linux_tts:
                backend = f"Linux '{self._linux_tts}'"
            else:
                backend = "console print (install espeak-ng for voice: sudo apt install espeak-ng)"
        elif "windows" in self._platform:
            backend = "Windows (pyttsx3 if available)"
        else:
            backend = "console print"

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
                    
                    # Terminate the process
                    self._current_process.terminate()
                    try:
                        self._current_process.wait(timeout=0.5)
                    except subprocess.TimeoutExpired:
                        self._current_process.kill()

                    # On macOS, also kill any lingering 'say' processes
                    if "darwin" in self._platform:
                        try:
                            subprocess.run(["pkill", "-9", "say"], 
                                         capture_output=True, timeout=0.5)
                        except Exception:
                            pass
                    
                    # On Linux, kill any lingering TTS processes
                    if "linux" in self._platform and self._linux_tts:
                        try:
                            subprocess.run(["pkill", "-9", self._linux_tts],
                                         capture_output=True, timeout=0.5)
                        except Exception:
                            pass
                    
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

    def play_listening_cue(self) -> None:
        """Play a short audio cue to indicate the system is listening."""
        if "darwin" in self._platform:
            try:
                # Use macOS 'say' with a very short word at high speed
                subprocess.Popen(
                    ["say", "-r", "300", "hmm"],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                ).wait(timeout=1.0)
            except Exception:
                pass
        elif "linux" in self._platform and self._linux_tts:
            try:
                subprocess.Popen(
                    [self._linux_tts, "-s", "300", "hmm"],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                ).wait(timeout=1.0)
            except Exception:
                pass

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

    def _build_tts_command(self, text: str) -> list:
        """Build the platform-specific TTS command."""
        if "darwin" in self._platform:
            return ["say", text]

        if "linux" in self._platform and self._linux_tts:
            if self._linux_tts in ("espeak-ng", "espeak"):
                # -s 160 = speed, -p 50 = pitch (natural sounding)
                return [self._linux_tts, "-s", "160", "-p", "50", text]
            elif self._linux_tts == "piper":
                # piper reads from stdin
                return ["piper", "--output-raw"]
            elif self._linux_tts == "festival":
                return ["festival", "--tts"]

        return []

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

            cmd = self._build_tts_command(text)

            if cmd:
                try:
                    if self._linux_tts == "festival":
                        # festival reads from stdin
                        with self._process_lock:
                            self._current_process = subprocess.Popen(
                                cmd,
                                stdin=subprocess.PIPE,
                                stdout=subprocess.DEVNULL,
                                stderr=subprocess.DEVNULL,
                            )
                        self._current_process.communicate(input=text.encode(), timeout=30)
                    elif self._linux_tts == "piper":
                        # piper reads from stdin, outputs raw audio
                        with self._process_lock:
                            self._current_process = subprocess.Popen(
                                cmd,
                                stdin=subprocess.PIPE,
                                stdout=subprocess.DEVNULL,
                                stderr=subprocess.DEVNULL,
                            )
                        self._current_process.communicate(input=text.encode(), timeout=30)
                    else:
                        # macOS 'say' or espeak: text is part of the command
                        with self._process_lock:
                            self._current_process = subprocess.Popen(
                                cmd,
                                stdout=subprocess.DEVNULL,
                                stderr=subprocess.DEVNULL,
                            )
                        
                        # Wait for completion or interrupt
                        while self._current_process.poll() is None:
                            if self._interrupt_event.is_set():
                                break
                            time.sleep(0.05)

                    with self._process_lock:
                        self._current_process = None

                except subprocess.TimeoutExpired:
                    with self._process_lock:
                        if self._current_process:
                            self._current_process.kill()
                            self._current_process = None
                except Exception as e:
                    print(f"⚠️ SpeechEngine TTS failed: {e!r}")
                    print(f"[TTS] {text}")
            else:
                # No TTS available — print to console
                print(f"[TTS] {text}")

            self._is_speaking = False
            
            if self.debug and not self._interrupt_event.is_set():
                print(f"✅ [SpeechEngine] finished in {time.time() - start:.2f}s")
