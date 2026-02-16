"""
Voice listener module (Vosk)
Always listening for wake phrase ("hey vision").

Debug upgrade:
- Prints PARTIAL + FINAL transcripts to terminal
- Wake detection uses PARTIAL for fast response
- Command mode uses PHRASE grammar (small set) for reliability
"""

from __future__ import annotations

import json
import queue
import re
import time
from pathlib import Path
from typing import Optional

import sounddevice as sd
from vosk import Model, KaldiRecognizer

import src.utils.config as config


class VoiceCommandListener:
    def __init__(self):
        self._events: "queue.Queue[dict]" = queue.Queue()
        self._stop_flag = False

        self._wake_phrase = config.WAKE_PHRASE.strip().lower()
        self._wake_regex = re.compile(r"\bhey\b.*\bvision\b")

        self._state = "WAIT_WAKE"
        self._wake_time = 0.0

        self._last_partial = ""
        self._last_partial_print_t = 0.0

        self._model = self._load_model(config.VOSK_MODEL_PATH)

        self._sr = self._pick_sample_rate(
            device=config.MIC_DEVICE_INDEX,
            target_rate=config.VOICE_SAMPLE_RATE
        )

        # Wake recognizer: FREEFORM is more forgiving for wake-word detection
        self._rec = KaldiRecognizer(self._model, self._sr)

    # ---------------------------
    # Public API
    # ---------------------------

    def start_background(self):
        """Starts the microphone stream in a background callback."""
        self._stop_flag = False
        self._audio_q: "queue.Queue[bytes]" = queue.Queue()

        def callback(indata, frames, time_info, status):
            if status:
                # Useful if audio glitches
                # print("Audio status:", status)
                pass
            self._audio_q.put(bytes(indata))

        self._stream = sd.RawInputStream(
            device=config.MIC_DEVICE_INDEX,
            samplerate=self._sr,
            blocksize=config.VOICE_BLOCK_SIZE,
            dtype="int16",
            channels=1,
            callback=callback,
        )
        self._stream.start()

    def stop(self):
        self._stop_flag = True
        try:
            self._stream.stop()
        except Exception:
            pass
        try:
            self._stream.close()
        except Exception:
            pass

    def poll_event(self) -> Optional[dict]:
        """Non-blocking event poll: {"type":"wake"} or {"type":"command","text":"..."}"""
        try:
            return self._events.get_nowait()
        except queue.Empty:
            return None

    def tick(self):
        """Call repeatedly from your main loop to drain audio + run Vosk."""
        if self._stop_flag:
            return

        # Command timeout -> return to wake mode
        if self._state == "WAIT_COMMAND":
            if (time.time() - self._wake_time) > float(config.VOICE_COMMAND_TIMEOUT_SEC):
                if config.VOICE_DEBUG_PRINT_FINAL:
                    print("[VOICE] command timeout -> back to wake")
                self._state = "WAIT_WAKE"
                self._rec = KaldiRecognizer(self._model, self._sr)

        drained = 0
        while drained < 40:  # drain more per tick so it feels responsive
            try:
                data = self._audio_q.get_nowait()
            except queue.Empty:
                break
            drained += 1
            self._feed_recognizer(data)

    # ---------------------------
    # Internal helpers
    # ---------------------------

    def _load_model(self, model_path: str) -> Model:
        p = Path(model_path)
        if not (p.exists() and (p / "am").exists() and (p / "conf").exists() and (p / "graph").exists()):
            raise RuntimeError(
                "VOSK_MODEL_PATH must point to the folder containing am/, conf/, graph/.\n"
                f"Current VOSK_MODEL_PATH = {p}\n"
                "Fix src/utils/config.py -> VOSK_MODEL_PATH"
            )
        print("🎙️ Voice input ready (Vosk)")
        return Model(str(p))

    def _pick_sample_rate(self, device: Optional[int], target_rate: int) -> int:
        try:
            sd.check_input_settings(device=device, samplerate=target_rate, channels=1, dtype="int16")
            return int(target_rate)
        except Exception:
            info = sd.query_devices(device if device is not None else sd.default.device[0])
            fallback = int(info["default_samplerate"])
            sd.check_input_settings(device=device, samplerate=fallback, channels=1, dtype="int16")
            return fallback

    def _make_command_recognizer(self) -> KaldiRecognizer:
        # Phrase grammar for command mode (small set)
        grammar = json.dumps(getattr(config, "VOICE_COMMAND_PHRASES", ["[unk]"]))
        return KaldiRecognizer(self._model, self._sr, grammar)

    def _feed_recognizer(self, data: bytes):
        if self._rec.AcceptWaveform(data):
            result = json.loads(self._rec.Result())
            text = (result.get("text") or "").strip().lower()
            if config.VOICE_DEBUG_PRINT_FINAL and text:
                print(f"[VOICE final/{self._state}] {text}")
            if text:
                self._handle_final_text(text)
        else:
            partial = json.loads(self._rec.PartialResult()).get("partial", "").strip().lower()
            if partial:
                self._handle_partial_text(partial)

    def _handle_partial_text(self, partial: str):
        # Print partials (throttled so it doesn't spam too hard)
        if getattr(config, "VOICE_DEBUG_PRINT_PARTIAL", False):
            now = time.time()
            if partial != self._last_partial and (now - self._last_partial_print_t) > 0.10:
                print(f"[VOICE partial/{self._state}] {partial}")
                self._last_partial = partial
                self._last_partial_print_t = now

        # Wake detection from partial for quick response
        if self._state == "WAIT_WAKE":
            if (self._wake_phrase in partial) or self._wake_regex.search(partial):
                self._trigger_wake()

    def _handle_final_text(self, text: str):
        if self._state == "WAIT_WAKE":
            # Wake detection from final (backup)
            if (self._wake_phrase in text) or self._wake_regex.search(text):
                self._trigger_wake()
            return

        # WAIT_COMMAND state:
        # Ignore immediate audio right after wake (avoid hearing our own TTS)
        if (time.time() - self._wake_time) < float(config.VOICE_IGNORE_AFTER_WAKE_SEC):
            return

        # Emit command
        self._events.put({"type": "command", "text": text})

        # Back to wake mode
        self._state = "WAIT_WAKE"
        self._rec = KaldiRecognizer(self._model, self._sr)

    def _trigger_wake(self):
        if self._state != "WAIT_WAKE":
            return
        self._state = "WAIT_COMMAND"
        self._wake_time = time.time()

        # Switch recognizer to COMMAND grammar now
        self._rec = self._make_command_recognizer()

        if config.VOICE_DEBUG_PRINT_FINAL:
            print("[VOICE] WAKE DETECTED -> command mode")

        self._events.put({"type": "wake"})
