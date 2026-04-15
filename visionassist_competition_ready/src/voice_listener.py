# src/voice_listener.py
"""
Voice Listener with LOCAL Transcription (No APIs)
===================================================

Features:
- faster-whisper: Runs Whisper model LOCALLY (no API key needed)
- Vosk: Lightweight offline fallback
- macOS-compatible audio (records at native mic rate, resamples to 16kHz)
- Automatic silence detection (VAD)
- Zero API dependencies for voice input

Setup:
  pip install faster-whisper sounddevice soundfile numpy scipy
  Optional: pip install vosk
"""

import os
import json
import tempfile
import time
from typing import Optional
import numpy as np

import sounddevice as sd
import soundfile as sf

from src.utils import config


# =============================================================================
# macOS-COMPATIBLE AUDIO RECORDER
# =============================================================================

class AudioDeviceManager:
    """
    Records at the mic's native sample rate (e.g. 44100/48000 Hz on macOS)
    and resamples to 16kHz for transcription.
    Fixes PortAudio error -10851 on macOS.
    """

    def __init__(self, target_sample_rate: int = 16000):
        self.target_rate = target_sample_rate
        self.native_rate = target_sample_rate
        self.device_index = None
        self.needs_resample = False
        self._detect_device()

    def _detect_device(self):
        try:
            device_info = sd.query_devices(kind='input')
            self.device_index = sd.default.device[0]
            self.native_rate = int(device_info['default_samplerate'])

            print(f"  Audio device: {device_info['name']}")
            print(f"   Native rate: {self.native_rate} Hz, Target: {self.target_rate} Hz")

            if self.native_rate != self.target_rate:
                self.needs_resample = True
                print(f"   Resampling enabled: {self.native_rate} -> {self.target_rate} Hz")

            # Quick mic test
            try:
                test = sd.rec(
                    int(0.05 * self.native_rate),
                    samplerate=self.native_rate, channels=1,
                    dtype="float32", device=self.device_index,
                )
                sd.wait()
                peak = float(np.max(np.abs(test)))
                print(f"   Mic test: OK (peak={peak:.4f})")
            except Exception as e:
                print(f"   WARNING: Mic test failed: {e}")
                print(f"   Check: System Settings > Privacy > Microphone > Terminal")

        except Exception as e:
            print(f"  WARNING: Audio device detection failed: {e}")
            self.native_rate = self.target_rate

    def record_chunk(self, num_target_frames: int) -> np.ndarray:
        """Record and return audio at target sample rate."""
        if self.needs_resample:
            ratio = self.native_rate / self.target_rate
            native_frames = int(num_target_frames * ratio)
            chunk = sd.rec(
                native_frames, samplerate=self.native_rate,
                channels=1, dtype="float32", device=self.device_index,
            )
            sd.wait()
            chunk = self._resample(chunk.flatten(), self.native_rate, self.target_rate)
            if len(chunk) > num_target_frames:
                chunk = chunk[:num_target_frames]
            elif len(chunk) < num_target_frames:
                chunk = np.pad(chunk, (0, num_target_frames - len(chunk)))
            return chunk
        else:
            chunk = sd.rec(
                num_target_frames, samplerate=self.target_rate,
                channels=1, dtype="float32", device=self.device_index,
            )
            sd.wait()
            return chunk.flatten()

    @staticmethod
    def _resample(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
        if orig_sr == target_sr:
            return audio
        try:
            from scipy.signal import resample_poly
            from math import gcd
            g = gcd(orig_sr, target_sr)
            return resample_poly(audio, target_sr // g, orig_sr // g).astype(np.float32)
        except ImportError:
            pass
        duration = len(audio) / orig_sr
        target_len = int(duration * target_sr)
        indices = np.linspace(0, len(audio) - 1, target_len)
        return np.interp(indices, np.arange(len(audio)), audio).astype(np.float32)


# =============================================================================
# LOCAL TRANSCRIPTION ENGINES
# =============================================================================

class LocalWhisperTranscriber:
    """Runs Whisper LOCALLY via faster-whisper. No API key needed."""

    def __init__(self):
        self.model = None
        self.available = False
        self.model_size = os.getenv("WHISPER_MODEL_SIZE", "base")
        self.device = os.getenv("WHISPER_DEVICE", "auto")
        self.compute_type = os.getenv("WHISPER_COMPUTE_TYPE", "auto")

        try:
            from faster_whisper import WhisperModel
            if self.device == "auto":
                try:
                    import torch
                    self.device = "cuda" if torch.cuda.is_available() else "cpu"
                except ImportError:
                    self.device = "cpu"
            if self.compute_type == "auto":
                self.compute_type = "float16" if self.device == "cuda" else "int8"

            print(f"  Loading Whisper '{self.model_size}' locally ({self.device}/{self.compute_type})...")
            self.model = WhisperModel(self.model_size, device=self.device, compute_type=self.compute_type)
            self.available = True
            print(f"  Local Whisper '{self.model_size}' ready (device={self.device})")
        except ImportError:
            print("  faster-whisper not installed. Run: pip install faster-whisper")
        except Exception as e:
            print(f"  Whisper local init failed: {e}")

    def transcribe(self, audio_path: str) -> Optional[str]:
        if not self.available or not self.model:
            return None
        try:
            segments, info = self.model.transcribe(
                audio_path, language="en", beam_size=5, best_of=3,
                temperature=0.0, condition_on_previous_text=False,
                vad_filter=True,
                vad_parameters=dict(min_silence_duration_ms=500, speech_pad_ms=200),
                initial_prompt=(
                    "Voice commands for AI smart glasses. "
                    "Commands: give me directions to Target, "
                    "navigate to Starbucks, next step, continue, "
                    "stop navigation, what do you see, describe the scene, "
                    "read this, how far, status, weather."
                ),
            )
            text_parts = [seg.text.strip() for seg in segments]
            full_text = " ".join(text_parts).strip()
            return full_text if full_text else None
        except Exception as e:
            print(f"  Whisper local error: {e}")
            return None


class VoskTranscriber:
    """Vosk offline fallback."""

    def __init__(self):
        self.model = None
        self.available = False
        model_path = os.getenv("VOSK_MODEL_PATH", "")
        if not model_path:
            for p in ["./models/vosk", "./vosk-model", os.path.expanduser("~/.cache/vosk/model"),
                       "vosk-model-small-en-us-0.15", "vosk-model-en-us-0.22"]:
                if os.path.isdir(p):
                    model_path = p
                    break
        if not model_path or not os.path.isdir(model_path):
            return
        try:
            from vosk import Model, KaldiRecognizer, SetLogLevel
            SetLogLevel(-1)
            self.model = Model(model_path)
            self.available = True
            self._KaldiRecognizer = KaldiRecognizer
            print(f"  Vosk model loaded from {model_path}")
        except ImportError:
            pass
        except Exception as e:
            print(f"  Vosk init failed: {e}")

    def transcribe(self, audio_path: str, sample_rate: int = 16000) -> Optional[str]:
        if not self.available or not self.model:
            return None
        try:
            audio_data, sr = sf.read(audio_path, dtype="int16")
            if len(audio_data.shape) > 1:
                audio_data = audio_data[:, 0]
            rec = self._KaldiRecognizer(self.model, sample_rate)
            rec.SetWords(True)
            for i in range(0, len(audio_data), 4000):
                rec.AcceptWaveform(audio_data[i:i + 4000].tobytes())
            result = json.loads(rec.FinalResult())
            text = result.get("text", "").strip()
            return text if text else None
        except Exception as e:
            print(f"  Vosk error: {e}")
            return None


# =============================================================================
# MAIN VOICE LISTENER
# =============================================================================

class VoiceListener:
    """
    Voice Listener with LOCAL transcription and macOS audio compatibility.
    No API keys needed.
    """

    def __init__(
        self, duration_seconds: float = 8.0, sample_rate: int = 16000,
        silence_threshold: float = 0.012, silence_duration: float = 1.3,
        min_speech_duration: float = 0.15,
    ):
        self.max_duration = duration_seconds
        self.sample_rate = sample_rate
        self.silence_threshold = silence_threshold
        self.silence_duration = silence_duration
        self.min_speech_duration = min_speech_duration
        self.chunk_duration = 0.15
        self.chunk_frames = int(self.chunk_duration * sample_rate)

        # macOS-compatible audio
        self.audio = AudioDeviceManager(target_sample_rate=sample_rate)

        print("  Initializing local transcription engines...")
        self.local_whisper = LocalWhisperTranscriber()
        self.vosk = VoskTranscriber()

        if not self.local_whisper.available and not self.vosk.available:
            print("  WARNING: No transcription engine! pip install faster-whisper")

        engine = "Local Whisper" if self.local_whisper.available else ("Vosk" if self.vosk.available else "NONE")
        print(
            f"  VoiceListener initialized\n"
            f"   Engine: {engine} (fully local, no API)\n"
            f"   Mic native rate: {self.audio.native_rate} Hz\n"
            f"   Max duration: {duration_seconds}s\n"
            f"   Silence to stop: {silence_duration}s"
        )

    def listen_and_transcribe(self) -> Optional[str]:
        audio_path = self._record_with_vad()
        if not audio_path:
            return None
        try:
            if self.local_whisper.available:
                print("  Transcribing with local Whisper...")
                text = self.local_whisper.transcribe(audio_path)
                if text:
                    print(f"  Transcription: {text!r}")
                    return text

            if self.vosk.available:
                print("  Transcribing with Vosk...")
                text = self.vosk.transcribe(audio_path, self.sample_rate)
                if text:
                    print(f"  Transcription: {text!r}")
                    return text

            print("  Transcription failed")
            return None
        finally:
            try: os.remove(audio_path)
            except Exception: pass

    def _calculate_energy(self, audio_chunk: np.ndarray) -> float:
        if len(audio_chunk) == 0:
            return 0.0
        return float(np.sqrt(np.mean(audio_chunk ** 2)))

    def _record_with_vad(self) -> Optional[str]:
        print("  Listening... (speak now)")

        all_audio = []
        speech_started = False
        speech_frames = 0
        silence_frames = 0
        total_time = 0.0

        threshold = self.silence_threshold
        silence_frames_needed = int(self.silence_duration / self.chunk_duration)
        start_time = time.time()

        try:
            while total_time < self.max_duration:
                # Use AudioDeviceManager instead of raw sd.rec
                chunk = self.audio.record_chunk(self.chunk_frames)
                all_audio.append(chunk)
                total_time += self.chunk_duration

                energy = self._calculate_energy(chunk)
                is_speech = energy > threshold

                if not speech_started:
                    if is_speech:
                        speech_started = True
                        speech_frames = 1
                        silence_frames = 0
                        print("  Speech detected...")
                else:
                    if is_speech:
                        speech_frames += 1
                        silence_frames = 0
                    else:
                        silence_frames += 1
                        if silence_frames >= silence_frames_needed:
                            if speech_frames > 0:
                                elapsed = time.time() - start_time
                                print(f"  Done ({elapsed:.1f}s)")
                                break

            if total_time >= self.max_duration:
                print(f"  Max duration ({total_time:.1f}s)")

        except Exception as e:
            print(f"  Recording error: {e!r}")
            return None

        if not speech_started or speech_frames == 0:
            print("  No speech detected")
            return None

        audio = np.concatenate(all_audio)
        trim_chunks = max(0, silence_frames - 2)
        if trim_chunks > 0:
            trim_frames = int(trim_chunks * self.chunk_frames)
            if len(audio) > trim_frames:
                audio = audio[:-trim_frames]

        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp_path = tmp.name
            sf.write(tmp_path, audio, self.sample_rate)
            duration = len(audio) / self.sample_rate
            print(f"  Recorded {duration:.1f}s")
            return tmp_path
        except Exception as e:
            print(f"  Save error: {e!r}")
            return None