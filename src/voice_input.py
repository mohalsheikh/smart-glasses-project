"""
Offline voice input module using Vosk.

This is intentionally small + simple so it can plug into our existing pipeline
without changing the rest of the code.
"""

from __future__ import annotations
import os
import json
import queue
import time
from pathlib import Path
from typing import Optional
import sounddevice as sd
from vosk import KaldiRecognizer, Model

VOICE_INPUT_ENABLED: bool = True

# Absolute path to your Vosk model folder (must contain am/, conf/, graph/)
# NOTE: change this to where YOU unzipped the model.
VOSK_MODEL_PATH: str = os.environ.get("VOSK_MODEL_PATH", os.path.expanduser("~/smart-glasses-project/vosk-model-small-en-us-0.15"))

VOICE_INPUT_TIMEOUT_SECONDS: float = 8.0

class VoiceInput:
    def __init__(
        self,
        model_path = VOSK_MODEL_PATH,
        device_index: int | None = None,
        target_rate: int = 16000,
        block_size: int = 8000,
        wake_word_grammar: str = '["vision", "[unk]"]',
        command_grammar: str = '["detect", "read", "sleep", "end", "nevermind", "thanks", "[unk]"]',
        model_class_names: dict[int, str] | None = None 
    ) -> None:
        
        self.model_path = model_path
        self.device_index = device_index if device_index is not None else sd.default.device[0]
        self.target_rate = target_rate
        self.block_size = block_size
        self.wake_word_grammar = wake_word_grammar
        
        if model_class_names is not None: # if model class names are provided, we want to add them to the command grammar so that Vosk can recognize them in commands.
            cmd_grammar_json = json.loads(command_grammar)
            self.commands = cmd_grammar_json[:-1] # get the command grammar as a list, excluding the last element which is [unk]

            # Insert class names into command grammar for recognition
            for _, class_name in model_class_names.items():
                cmd_grammar_json.insert(-1, class_name)  # Insert before ["unk"]

            self.command_grammar = json.dumps(cmd_grammar_json)
            print(f"Constructed command grammar with model class names: {self.command_grammar}")
        else:
            self.command_grammar = command_grammar


        # ---- Verify model folder ----
        p = Path(model_path)
        if not (p.exists() and p.is_dir()):
            raise RuntimeError(
                f"Vosk model path does not exist: {p}\n"
                "Set VOSK_MODEL_PATH to the folder that contains am/, conf/, graph/."
            )
        if not ((p / "am").exists() and (p / "conf").exists() and (p / "graph").exists()):
            raise RuntimeError(
                f"Invalid Vosk model folder: {p}\n"
                "It must contain: am/, conf/, graph/."
            )
        
        # This gets decided per-device (some mics won't open at 16kHz).
        self.sample_rate = self._pick_sample_rate()
        
        self.model = Model(self.model_path)
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
        # If device_index isn't set, use the system default input device.

        info = sd.query_devices(self.device_index)
        device_default = int(info.get("default_samplerate", self.target_rate))

        # Try 16kHz first (ideal for most Vosk models), then fallback.
        sr = self.target_rate
        try:
            sd.check_input_settings(device=self.device_index, samplerate=sr, channels=1, dtype="int16")
            return sr
        except Exception:  # noqa: BLE001
            sd.check_input_settings(device=self.device_index, samplerate=device_default, channels=1, dtype="int16")
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
    
    def listen_wake_word(self, timeout_seconds: float = 8.0) -> str:
        """Set grammar to WAKE_WORD_GRAMMAR and then listen for the wake word."""
        self.recognizer = KaldiRecognizer(self.model, self.sample_rate, self.wake_word_grammar)
        return self.listen(timeout_seconds)

    def listen_command(self, timeout_seconds: float = 8.0) -> str:
        """Set grammar to COMMAND_GRAMMAR and then listen for a command."""
        self.recognizer = KaldiRecognizer(self.model, self.sample_rate, self.command_grammar)
        return self.listen(timeout_seconds)

    def listen(self, timeout_seconds: float = 8.0) -> str:
        """Listen for user voice input and return the transcript, based on the set grammar."""

        # Callback pushes raw audio bytes into the queue.
        def _callback(indata, frames, time_info, status):  # noqa: ANN001
            if status:
                # Don't spam - just one line.
                print("Audio status:", status)
            self._q.put(bytes(indata))

        last_partial: Optional[str] = None
        end_time = time.time() + float(timeout_seconds)

        # start listening until timeout
        try:
            with sd.RawInputStream(
                device=self.device_index,
                samplerate=self.sample_rate,
                blocksize=self.block_size,
                dtype="int16",
                channels=1,
                callback=_callback,
            ):
                while time.time() < end_time: # while we haven't hit the timeout...
                    # Check the queue for new audio data. If we get some, feed it to Vosk, otherwise just loop and check again
                    try:
                        data = self._q.get(timeout=0.2)
                    except queue.Empty:
                        continue

                    if self.recognizer.AcceptWaveform(data): # if the user has finished speaking...
                        result = json.loads(self.recognizer.Result())
                        text = str(result.get("text", "")).strip()
                        if text:
                            return text
                    else:
                        partial = json.loads(self.recognizer.PartialResult()).get("partial", "")
                        partial = str(partial).strip()
                        if partial:
                            last_partial = partial

        except Exception as e:  # noqa: BLE001
            print(f"[VoiceInput ERROR] {type(e).__name__}: {e}")
            return ""

        return last_partial or "" # if we got a partial result but no final, return the partial. Otherwise return empty string.
