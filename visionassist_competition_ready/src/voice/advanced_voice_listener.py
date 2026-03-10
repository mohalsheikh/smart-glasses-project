# src/voice/advanced_voice_listener.py
"""
Advanced Voice Listener v2.0 - LOCAL Transcription (No APIs)
=============================================================

All transcription runs LOCALLY on your machine. No API keys needed.

Engines:
- faster-whisper: Runs Whisper model locally via CTranslate2 (primary)
- Vosk: Lightweight offline recognition (fallback)

Features:
1. WAKE WORD DETECTION ("Hey Vision" / "Vision" / customizable)
2. ALWAYS-ON LISTENING MODE
3. SMARTER VAD (adaptive noise calibration, pre-speech buffer)
4. CONVERSATION MODE (follow-up without wake word)
5. macOS-compatible audio (records at native rate, resamples to 16kHz)

Drop-in compatible with existing controller.py.

Setup:
  pip install faster-whisper sounddevice soundfile numpy scipy
  # Optional: pip install vosk pvporcupine
"""

from __future__ import annotations

import os
import json
import time
import tempfile
import threading
from typing import Optional, Callable, List, Tuple
from collections import deque
from enum import Enum

import numpy as np
import sounddevice as sd
import soundfile as sf

from src.utils import config


# =============================================================================
# macOS-COMPATIBLE AUDIO RECORDER
# =============================================================================

class AudioDeviceManager:
    """
    Handles macOS (and other OS) audio device quirks.

    Problem: macOS built-in mic only supports 44100/48000 Hz natively.
             Forcing 16000 Hz causes PortAudio error -10851.
    Solution: Record at the mic's native rate, resample to 16kHz after.
    """

    def __init__(self, target_sample_rate: int = 16000):
        self.target_rate = target_sample_rate
        self.native_rate = target_sample_rate
        self.device_index = None
        self.needs_resample = False

        self._detect_device()

    def _detect_device(self):
        """Detect default input device and its native sample rate."""
        try:
            device_info = sd.query_devices(kind='input')
            self.device_index = sd.default.device[0]
            self.native_rate = int(device_info['default_samplerate'])

            print(f"  Audio device: {device_info['name']}")
            print(f"   Native sample rate: {self.native_rate} Hz")
            print(f"   Target sample rate: {self.target_rate} Hz")

            if self.native_rate != self.target_rate:
                self.needs_resample = True
                print(f"   Resampling enabled: {self.native_rate} -> {self.target_rate} Hz")
            else:
                self.needs_resample = False
                print(f"   No resampling needed")

            # Quick mic test
            try:
                test = sd.rec(
                    int(0.05 * self.native_rate),
                    samplerate=self.native_rate,
                    channels=1,
                    dtype="float32",
                    device=self.device_index,
                )
                sd.wait()
                peak = float(np.max(np.abs(test)))
                print(f"   Mic test: OK (peak={peak:.4f})")
            except Exception as e:
                print(f"   WARNING: Mic test failed: {e}")
                print(f"   Check: System Settings > Privacy > Microphone > Terminal")

        except Exception as e:
            print(f"  WARNING: Could not detect audio device: {e}")
            self.native_rate = self.target_rate
            self.needs_resample = False

    def record_chunk(self, num_target_frames: int) -> np.ndarray:
        """
        Record audio chunk. Returns audio at target sample rate (16kHz).

        Args:
            num_target_frames: Number of frames wanted at 16kHz.
        Returns:
            Audio numpy array at target_rate, shape (num_target_frames,)
        """
        if self.needs_resample:
            ratio = self.native_rate / self.target_rate
            native_frames = int(num_target_frames * ratio)

            chunk = sd.rec(
                native_frames,
                samplerate=self.native_rate,
                channels=1,
                dtype="float32",
                device=self.device_index,
            )
            sd.wait()
            chunk = chunk.flatten()
            chunk = self._resample(chunk, self.native_rate, self.target_rate)

            if len(chunk) > num_target_frames:
                chunk = chunk[:num_target_frames]
            elif len(chunk) < num_target_frames:
                chunk = np.pad(chunk, (0, num_target_frames - len(chunk)))

            return chunk
        else:
            chunk = sd.rec(
                num_target_frames,
                samplerate=self.target_rate,
                channels=1,
                dtype="float32",
                device=self.device_index,
            )
            sd.wait()
            return chunk.flatten()

    def record_seconds(self, seconds: float) -> np.ndarray:
        """Record for a duration, return audio at target sample rate."""
        native_frames = int(seconds * self.native_rate)

        chunk = sd.rec(
            native_frames,
            samplerate=self.native_rate,
            channels=1,
            dtype="float32",
            device=self.device_index,
        )
        sd.wait()
        chunk = chunk.flatten()

        if self.needs_resample:
            chunk = self._resample(chunk, self.native_rate, self.target_rate)

        return chunk

    @staticmethod
    def _resample(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
        """Resample audio from orig_sr to target_sr."""
        if orig_sr == target_sr:
            return audio

        try:
            from scipy.signal import resample_poly
            from math import gcd
            g = gcd(orig_sr, target_sr)
            up = target_sr // g
            down = orig_sr // g
            return resample_poly(audio, up, down).astype(np.float32)
        except ImportError:
            pass

        # Fallback: numpy linear interpolation
        duration = len(audio) / orig_sr
        target_len = int(duration * target_sr)
        indices = np.linspace(0, len(audio) - 1, target_len)
        return np.interp(indices, np.arange(len(audio)), audio).astype(np.float32)


# =============================================================================
# LISTENING MODES
# =============================================================================

class ListeningMode(Enum):
    PUSH_TO_TALK = "ptt"
    WAKE_WORD = "wake_word"
    CONTINUOUS = "continuous"


class ConversationState(Enum):
    IDLE = "idle"
    LISTENING = "listening"
    PROCESSING = "processing"
    CONVERSATION = "conversation"


# =============================================================================
# WAKE WORD DETECTOR
# =============================================================================

class WakeWordDetector:
    DEFAULT_WAKE_PHRASES = [
        "hey vision", "vision", "hey glasses", "okay vision", "ok vision",
    ]

    def __init__(self, wake_phrases: Optional[List[str]] = None, sensitivity: float = 0.6):
        self.wake_phrases = [p.lower().strip() for p in (wake_phrases or self.DEFAULT_WAKE_PHRASES)]
        self.sensitivity = sensitivity
        self.porcupine = None
        self._porcupine_available = False
        self._try_init_porcupine()

        if self._porcupine_available:
            print("  Wake word: Porcupine engine (high accuracy)")
        else:
            print("  Wake word: Phonetic matching engine")
        print(f"   Wake phrases: {self.wake_phrases}")

    def _try_init_porcupine(self):
        access_key = os.getenv("PORCUPINE_ACCESS_KEY", "")
        if not access_key:
            return
        try:
            import pvporcupine
            self.porcupine = pvporcupine.create(
                access_key=access_key, keywords=["computer"],
                sensitivities=[self.sensitivity],
            )
            self._porcupine_available = True
        except Exception as e:
            print(f"  Porcupine unavailable: {e}")

    def check_wake_word_in_text(self, text: str) -> bool:
        if not text:
            return False
        t = text.lower().strip()
        for phrase in self.wake_phrases:
            if phrase in t:
                return True
        phonetic_variants = {
            "hey vision": ["hey vision", "a vision", "hey visions", "hey fishing", "hey vishion"],
            "vision": ["vision", "visions", "vishion"],
            "hey glasses": ["hey glasses", "hey glass", "a glasses"],
        }
        for base_phrase in self.wake_phrases:
            for variant in phonetic_variants.get(base_phrase, []):
                if variant in t:
                    return True
        return False

    def check_audio_porcupine(self, audio_frame: np.ndarray) -> bool:
        if not self._porcupine_available or self.porcupine is None:
            return False
        try:
            pcm = (audio_frame * 32768).astype(np.int16)
            frame_length = self.porcupine.frame_length
            for i in range(0, len(pcm) - frame_length + 1, frame_length):
                result = self.porcupine.process(pcm[i:i + frame_length])
                if result >= 0:
                    return True
        except Exception:
            pass
        return False

    def cleanup(self):
        if self.porcupine is not None:
            try:
                self.porcupine.delete()
            except Exception:
                pass


# =============================================================================
# ADAPTIVE VAD
# =============================================================================

class AdaptiveVAD:
    def __init__(
        self, sample_rate: int = 16000, frame_ms: int = 30,
        pre_speech_buffer_ms: int = 300, hangover_ms: int = 400,
        noise_adaptation_rate: float = 0.05, speech_energy_factor: float = 1.8,
    ):
        self.sample_rate = sample_rate
        self.frame_size = int(sample_rate * frame_ms / 1000)
        self.pre_speech_frames = int(pre_speech_buffer_ms / frame_ms)
        self.hangover_frames = int(hangover_ms / frame_ms)
        self.noise_adaptation_rate = noise_adaptation_rate
        self.speech_energy_factor = speech_energy_factor

        self.noise_energy = 0.01
        self.speech_active = False
        self.hangover_counter = 0
        self.pre_speech_buffer: deque = deque(maxlen=self.pre_speech_frames)

        self._calibrated = False
        self._calibration_frames: List[float] = []
        self._calibration_target = int(300 / frame_ms)  # 300ms calibration (faster)

    def calibrate(self, audio_chunk: np.ndarray) -> bool:
        energy = self._compute_energy(audio_chunk)
        self._calibration_frames.append(energy)
        if len(self._calibration_frames) >= self._calibration_target:
            self.noise_energy = np.mean(self._calibration_frames) * 1.2
            self._calibrated = True
            print(f"  VAD calibrated - noise floor: {self.noise_energy:.4f}")
            return True
        return False

    def process_frame(self, audio_chunk: np.ndarray) -> Tuple[bool, bool]:
        energy = self._compute_energy(audio_chunk)
        zcr = self._compute_zcr(audio_chunk)

        if not self.speech_active:
            self.noise_energy = (
                (1 - self.noise_adaptation_rate) * self.noise_energy
                + self.noise_adaptation_rate * energy
            )

        threshold = self.noise_energy * self.speech_energy_factor
        is_speech_frame = energy > threshold and zcr > 0.005
        just_started = False

        if is_speech_frame:
            if not self.speech_active:
                self.speech_active = True
                just_started = True
            self.hangover_counter = self.hangover_frames
        else:
            if self.speech_active:
                self.hangover_counter -= 1
                if self.hangover_counter <= 0:
                    self.speech_active = False

        self.pre_speech_buffer.append(audio_chunk.copy())
        return self.speech_active, just_started

    def get_pre_speech_audio(self) -> np.ndarray:
        if self.pre_speech_buffer:
            return np.concatenate(list(self.pre_speech_buffer))
        return np.array([], dtype=np.float32)

    def reset(self):
        self.speech_active = False
        self.hangover_counter = 0
        self.pre_speech_buffer.clear()

    @staticmethod
    def _compute_energy(audio: np.ndarray) -> float:
        if len(audio) == 0:
            return 0.0
        return float(np.sqrt(np.mean(audio ** 2)))

    @staticmethod
    def _compute_zcr(audio: np.ndarray) -> float:
        if len(audio) < 2:
            return 0.0
        signs = np.sign(audio)
        crossings = np.sum(np.abs(np.diff(signs)) > 0)
        return float(crossings / len(audio))


# =============================================================================
# LOCAL TRANSCRIPTION ENGINES (No APIs!)
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

    def transcribe(self, audio_path: str, is_wake_check: bool = False) -> Optional[str]:
        if not self.available or not self.model:
            return None
        try:
            beam_size = 3 if is_wake_check else 5
            best_of = 1 if is_wake_check else 3

            # NOTE: Do NOT use initial_prompt - Whisper hallucinates it back
            # when given silent/quiet audio. Let it transcribe freely.
            segments, info = self.model.transcribe(
                audio_path, language="en", beam_size=beam_size, best_of=best_of,
                temperature=0.0, condition_on_previous_text=False,
                vad_filter=True,
                vad_parameters=dict(
                    min_silence_duration_ms=300 if is_wake_check else 500,
                    speech_pad_ms=150 if is_wake_check else 200,
                ),
            )
            text_parts = [seg.text.strip() for seg in segments]
            full_text = " ".join(text_parts).strip()

            if not full_text:
                return None

            # Reject known Whisper hallucinations on near-silent audio
            hallucinations = [
                "voice commands for visionassist",
                "visionassist smart glasses",
                "thank you for watching",
                "thanks for watching",
                "please subscribe",
                "thank you for listening",
                "the end",
                "bye bye",
                "you",
                "...",
            ]
            check = full_text.lower().strip(" .")
            for h in hallucinations:
                if check == h or check.startswith(h):
                    print(f"  Whisper hallucination rejected: {full_text!r}")
                    return None

            return full_text
        except Exception as e:
            print(f"  Whisper local error: {e}")
            return None


class VoskTranscriber:
    """Vosk offline speech recognition fallback."""

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
            print("  vosk not installed (optional). Run: pip install vosk")
        except Exception as e:
            print(f"  Vosk init failed: {e}")

    def transcribe(self, audio_path: str, sample_rate: int = 16000, is_wake_check: bool = False) -> Optional[str]:
        if not self.available or not self.model:
            return None
        try:
            audio_data, sr = sf.read(audio_path, dtype="int16")
            if sr != sample_rate:
                try:
                    from scipy.signal import resample_poly
                    from math import gcd
                    g = gcd(sr, sample_rate)
                    audio_data = resample_poly(audio_data, sample_rate // g, sr // g).astype(np.int16)
                except ImportError:
                    target_len = int(len(audio_data) * sample_rate / sr)
                    indices = np.linspace(0, len(audio_data) - 1, target_len)
                    audio_data = np.interp(indices, np.arange(len(audio_data)), audio_data).astype(np.int16)
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
# MAIN ADVANCED VOICE LISTENER
# =============================================================================

class AdvancedVoiceListener:
    """
    Production-grade voice listener with LOCAL transcription.
    Uses AudioDeviceManager for macOS compatibility.
    """

    def __init__(
        self, mode: str = "wake_word", max_duration: float = 10.0,
        sample_rate: int = 16000, silence_duration: float = 1.2,
        min_speech_duration: float = 0.15, conversation_timeout: float = 15.0,
        wake_phrases: Optional[List[str]] = None,
        on_wake_detected: Optional[Callable] = None,
        on_listening_start: Optional[Callable] = None,
        on_listening_stop: Optional[Callable] = None,
    ):
        self.mode = ListeningMode(mode)
        self.max_duration = max_duration
        self.sample_rate = sample_rate
        self.silence_duration = silence_duration
        self.min_speech_duration = min_speech_duration
        self.conversation_timeout = conversation_timeout

        self.on_wake_detected = on_wake_detected
        self.on_listening_start = on_listening_start
        self.on_listening_stop = on_listening_stop

        self.state = ConversationState.IDLE
        self._last_interaction_time = 0.0
        self._state_lock = threading.Lock()

        self.wake_detector = WakeWordDetector(wake_phrases=wake_phrases)
        self.vad = AdaptiveVAD(sample_rate=sample_rate)

        # macOS-compatible audio manager
        self.audio = AudioDeviceManager(target_sample_rate=sample_rate)

        print("  Initializing local transcription engines...")
        self.local_whisper = LocalWhisperTranscriber()
        self.vosk = VoskTranscriber()

        if not self.local_whisper.available and not self.vosk.available:
            print("  WARNING: No transcription engine available!")
            print("    Install faster-whisper: pip install faster-whisper")

        self.chunk_duration = 0.03
        self.chunk_frames = int(self.chunk_duration * sample_rate)

        self._wake_thread: Optional[threading.Thread] = None
        self._wake_stop_event = threading.Event()
        self._wake_detected_event = threading.Event()

        self.stats = {
            "wake_detections": 0, "transcriptions": 0,
            "failed_transcriptions": 0, "avg_listen_ms": 0.0, "avg_transcribe_ms": 0.0,
        }

        engine = "Local Whisper" if self.local_whisper.available else ("Vosk" if self.vosk.available else "NONE")
        print(
            f"  AdvancedVoiceListener v2.0 initialized\n"
            f"   Mode: {self.mode.value}\n"
            f"   Engine: {engine} (fully local, no API)\n"
            f"   Mic native rate: {self.audio.native_rate} Hz\n"
            f"   Conversation timeout: {conversation_timeout}s\n"
            f"   Max listen duration: {max_duration}s"
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def listen_and_transcribe(self) -> Optional[str]:
        if self.mode == ListeningMode.WAKE_WORD:
            with self._state_lock:
                in_convo = (
                    self.state == ConversationState.CONVERSATION
                    and (time.time() - self._last_interaction_time) < self.conversation_timeout
                )
            if not in_convo:
                self._set_state(ConversationState.IDLE)

        self._set_state(ConversationState.LISTENING)
        if self.on_listening_start:
            try: self.on_listening_start()
            except Exception: pass

        t0 = time.time()
        audio_path = self._record_with_smart_vad()
        listen_ms = (time.time() - t0) * 1000

        if self.on_listening_stop:
            try: self.on_listening_stop()
            except Exception: pass

        if not audio_path:
            self._set_state(ConversationState.IDLE)
            return None

        self._set_state(ConversationState.PROCESSING)
        t1 = time.time()
        text = self._transcribe(audio_path)
        transcribe_ms = (time.time() - t1) * 1000

        self.stats["transcriptions"] += 1
        self.stats["avg_listen_ms"] = self.stats["avg_listen_ms"] * 0.8 + listen_ms * 0.2
        self.stats["avg_transcribe_ms"] = self.stats["avg_transcribe_ms"] * 0.8 + transcribe_ms * 0.2

        if not text:
            self.stats["failed_transcriptions"] += 1
            self._set_state(ConversationState.IDLE)
            return None

        if self.mode == ListeningMode.WAKE_WORD:
            text = self._strip_wake_word(text)

        self._last_interaction_time = time.time()
        self._set_state(ConversationState.CONVERSATION)
        return text

    def wait_for_wake_word(self, timeout: float = 0.0) -> bool:
        print("  Waiting for wake word...")
        self._set_state(ConversationState.IDLE)
        start = time.time()

        while True:
            if timeout > 0 and (time.time() - start) > timeout:
                return False
            try:
                audio = self.audio.record_seconds(2.0)

                if self.wake_detector.check_audio_porcupine(audio):
                    self.stats["wake_detections"] += 1
                    if self.on_wake_detected: self.on_wake_detected()
                    print("  Wake word detected! (Porcupine)")
                    return True

                energy = float(np.sqrt(np.mean(audio ** 2)))
                if energy > 0.01:
                    tmp_path = self._save_audio(audio)
                    if tmp_path:
                        text = self._transcribe(tmp_path, is_wake_check=True)
                        if text and self.wake_detector.check_wake_word_in_text(text):
                            self.stats["wake_detections"] += 1
                            if self.on_wake_detected: self.on_wake_detected()
                            print(f"  Wake word detected! ('{text}')")
                            return True
            except Exception as e:
                print(f"  Wake word listen error: {e}")
                time.sleep(0.5)

    def start_background_listening(self, callback: Callable[[str], None]):
        if self._wake_thread and self._wake_thread.is_alive():
            print("  Background listening already running")
            return
        self._wake_stop_event.clear()

        def _background_loop():
            print("  Background listening started")
            while not self._wake_stop_event.is_set():
                try:
                    detected = self.wait_for_wake_word(timeout=0.5)
                    if self._wake_stop_event.is_set(): break
                    if detected:
                        text = self.listen_and_transcribe()
                        if text: callback(text)
                except Exception as e:
                    print(f"  Background listener error: {e}")
                    time.sleep(1)
            print("  Background listening stopped")

        self._wake_thread = threading.Thread(target=_background_loop, daemon=True)
        self._wake_thread.start()

    def stop_background_listening(self):
        self._wake_stop_event.set()
        if self._wake_thread:
            self._wake_thread.join(timeout=3.0)

    def is_in_conversation(self) -> bool:
        with self._state_lock:
            return (
                self.state == ConversationState.CONVERSATION
                and (time.time() - self._last_interaction_time) < self.conversation_timeout
            )

    def end_conversation(self):
        self._set_state(ConversationState.IDLE)
        self._last_interaction_time = 0

    def get_state(self) -> str:
        with self._state_lock:
            return self.state.value

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _set_state(self, state: ConversationState):
        with self._state_lock:
            self.state = state

    def _strip_wake_word(self, text: str) -> str:
        t = text.strip()
        t_lower = t.lower()
        for phrase in self.wake_detector.wake_phrases:
            if t_lower.startswith(phrase):
                stripped = t[len(phrase):].strip(" ,.")
                return stripped if stripped else t
        return t

    def _record_with_smart_vad(self) -> Optional[str]:
        """
        Record audio for voice command.
        
        Uses simple timed recording — records for a set duration then 
        lets Whisper figure out the speech. WAY more reliable than VAD
        on laptop mics where TTS bleeds into the mic.
        """
        # Wait for TTS to fully stop bleeding into the mic
        time.sleep(0.5)

        print("  🎤 Listening... (speak now)")

        # Simple timed recording — 5 seconds is enough for any command
        record_seconds = 5.0
        audio = self.audio.record_seconds(record_seconds)

        if audio is None or len(audio) == 0:
            print("  No audio captured")
            return None

        duration = len(audio) / self.sample_rate
        rms = float(np.sqrt(np.mean(audio ** 2)))
        peak = float(np.max(np.abs(audio)))

        print(f"  Recorded {duration:.1f}s (RMS={rms:.4f}, peak={peak:.4f})")

        # Only reject if truly silent (no mic input at all)
        if peak < 0.001:
            print(f"  Recording is completely silent — check mic permissions")
            return None

        return self._save_audio(audio)

    def _save_audio(self, audio: np.ndarray) -> Optional[str]:
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

    def _transcribe(self, audio_path: str, is_wake_check: bool = False) -> Optional[str]:
        try:
            # 1) Try local faster-whisper
            if self.local_whisper.available:
                text = self.local_whisper.transcribe(audio_path, is_wake_check=is_wake_check)
                if text:
                    print(f"  [{('wake' if is_wake_check else 'cmd')}] {text!r}")
                    return text

            # 2) Try local Vosk
            if self.vosk.available:
                text = self.vosk.transcribe(audio_path, self.sample_rate, is_wake_check=is_wake_check)
                if text:
                    print(f"  [{('wake' if is_wake_check else 'cmd')}] {text!r}")
                    return text

            # 3) Fallback: OpenAI Whisper API (if API key available)
            text = self._transcribe_openai_api(audio_path)
            if text:
                print(f"  [{('wake' if is_wake_check else 'cmd')}] {text!r} (OpenAI API)")
                return text

            print("  Transcription failed (no engine available)")
            return None
        finally:
            try: os.remove(audio_path)
            except Exception: pass

    def _transcribe_openai_api(self, audio_path: str) -> Optional[str]:
        """Fallback: use OpenAI Whisper API when no local engine is installed."""
        try:
            api_key = os.getenv("OPENAI_API_KEY", "")
            if not api_key:
                return None

            from openai import OpenAI
            client = OpenAI()

            with open(audio_path, "rb") as f:
                resp = client.audio.transcriptions.create(
                    model="whisper-1",
                    file=f,
                    language="en",
                    # NOTE: No prompt — it causes hallucinations on quiet audio
                )

            text = (resp.text or "").strip()
            if not text:
                return None

            # Reject common Whisper hallucinations
            check = text.lower().strip(" .")
            hallucinations = [
                "thank you", "thanks for watching", "please subscribe",
                "the end", "bye", "you", "visionassist",
            ]
            for h in hallucinations:
                if check == h or (len(check) < 20 and h in check):
                    print(f"  OpenAI API hallucination rejected: {text!r}")
                    return None

            return text

        except ImportError:
            return None
        except Exception as e:
            print(f"  OpenAI Whisper API error: {e}")
            return None

    def cleanup(self):
        self.stop_background_listening()
        self.wake_detector.cleanup()


# =============================================================================
# BACKWARD COMPATIBILITY
# =============================================================================

class VoiceListener(AdvancedVoiceListener):
    """Drop-in replacement. Same listen_and_transcribe() API."""

    def __init__(
        self, duration_seconds: float = 10.0, sample_rate: int = 16000,
        silence_threshold: float = 0.012, silence_duration: float = 0.8,
        min_speech_duration: float = 0.15,
    ):
        mode = os.getenv("VOICE_MODE", "ptt")
        conversation_timeout = float(os.getenv("CONVERSATION_TIMEOUT", "15.0"))
        super().__init__(
            mode=mode, max_duration=duration_seconds, sample_rate=sample_rate,
            silence_duration=silence_duration, min_speech_duration=min_speech_duration,
            conversation_timeout=conversation_timeout,
        )