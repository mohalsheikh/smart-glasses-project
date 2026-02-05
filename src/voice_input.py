"""
Offline voice input module using Vosk.

This is intentionally small + simple so it can plug into our existing pipeline
without changing the rest of the code.
"""

from __future__ import annotations

import json
import queue
import time
from pathlib import Path
from typing import Optional


class VoiceInput:
    def __init__(
        self,
        model_path: str,
        device_index: int | None = None,
        target_rate: int = 16000,
        block_size: int = 8000,
    ) -> None:
        # We keep imports inside so the project can still run even if
        # someone hasn't installed the voice dependencies yet.
        try:
            import sounddevice as sd  # type: ignore
            from vosk import KaldiRecognizer, Model  # type: ignore
        except Exception as e:  # noqa: BLE001
            raise RuntimeError(
                "VoiceInput needs 'vosk' and 'sounddevice'. "
                "Install them with: pip install vosk sounddevice"
            ) from e

        self._sd = sd
        self._Model = Model
        self._KaldiRecognizer = KaldiRecognizer

        self.device_index = device_index
        self.target_rate = int(target_rate)
        self.block_size = int(block_size)

        # ---- Verify model folder ----
        p = Path(model_path)
        if not (p.exists() and p.is_dir()):
            raise RuntimeError(
                f"Vosk model path does not exist: {p}\n"
                "Set config.VOSK_MODEL_PATH to the folder that contains am/, conf/, graph/."
            )
        if not ((p / "am").exists() and (p / "conf").exists() and (p / "graph").exists()):
            raise RuntimeError(
                f"Invalid Vosk model folder: {p}\n"
                "It must contain: am/, conf/, graph/."
            )

        self.model_path = str(p)
        # Model load is the expensive part, so we do it once.
        self.model = self._Model(self.model_path)

        # This gets decided per-device (some mics won't open at 16kHz).
        self.sample_rate = self._pick_sample_rate()

        self._q: "queue.Queue[bytes]" = queue.Queue()

    def print_input_devices(self) -> None:
        """Quick helper for debugging mic device indexes."""
        sd = self._sd
        try:
            hostapis = sd.query_hostapis()
        except Exception:  # noqa: BLE001
            hostapis = []

        print("\nInput devices (pick one that looks like your microphone):")
        for i, d in enumerate(sd.query_devices()):
            if d.get("max_input_channels", 0) > 0:
                api_name = ""
                try:
                    api_name = hostapis[d["hostapi"]]["name"] if hostapis else ""
                except Exception:  # noqa: BLE001
                    api_name = ""
                print(
                    f"  {i}: {d.get('name')}  (inputs={d.get('max_input_channels')}, "
                    f"default_sr={d.get('default_samplerate')}, api={api_name})"
                )

    def _pick_sample_rate(self) -> int:
        sd = self._sd

        # If device_index isn't set, use the system default input device.
        device = self.device_index
        if device is None:
            device = sd.default.device[0]

        info = sd.query_devices(device)
        device_default = int(info.get("default_samplerate", self.target_rate))

        # Try 16kHz first (ideal for most Vosk models), then fallback.
        sr = self.target_rate
        try:
            sd.check_input_settings(device=device, samplerate=sr, channels=1, dtype="int16")
            return sr
        except Exception:  # noqa: BLE001
            sd.check_input_settings(device=device, samplerate=device_default, channels=1, dtype="int16")
            return device_default

    @staticmethod
    def _normalize(text: str) -> str:
        return " ".join(text.lower().strip().split())

    def matches_any(self, transcript: str, phrases: tuple[str, ...]) -> bool:
        heard = self._normalize(transcript)
        for p in phrases:
            if self._normalize(p) in heard:
                return True
        return False

    def listen_once(self, timeout_seconds: float = 8.0) -> str:
        """Listen for one short command and return the transcript (lowercase-ish)."""

        sd = self._sd
        KaldiRecognizer = self._KaldiRecognizer

        device = self.device_index
        if device is None:
            device = sd.default.device[0]

        # Callback pushes raw audio bytes into the queue.
        def _callback(indata, frames, time_info, status):  # noqa: ANN001
            if status:
                # Don't spam - just one line.
                print("Audio status:", status)
            self._q.put(bytes(indata))

        rec = KaldiRecognizer(self.model, self.sample_rate)

        last_partial: Optional[str] = None
        end_time = time.time() + float(timeout_seconds)

        try:
            with sd.RawInputStream(
                device=device,
                samplerate=self.sample_rate,
                blocksize=self.block_size,
                dtype="int16",
                channels=1,
                callback=_callback,
            ):
                while time.time() < end_time:
                    try:
                        data = self._q.get(timeout=0.2)
                    except queue.Empty:
                        continue

                    if rec.AcceptWaveform(data):
                        result = json.loads(rec.Result())
                        text = str(result.get("text", "")).strip()
                        if text:
                            return text
                    else:
                        partial = json.loads(rec.PartialResult()).get("partial", "")
                        partial = str(partial).strip()
                        if partial:
                            last_partial = partial

        except Exception as e:  # noqa: BLE001
            print(f"[VoiceInput ERROR] {type(e).__name__}: {e}")
            return ""

        return last_partial or ""
