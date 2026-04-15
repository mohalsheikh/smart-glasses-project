# src/voice_listener.py
"""
Advanced Voice Listener with Deepgram + Whisper Fallback
=========================================================

Features:
- Deepgram Nova-3: Industry-leading accuracy, 50% lower word error rate
- Automatic silence detection (VAD)
- Keyword boosting for navigation commands
- Falls back to Whisper if Deepgram unavailable

Setup:
1. Get free API key at: https://console.deepgram.com (free $200 credit!)
2. Add to .env: DEEPGRAM_API_KEY=your_key_here
3. pip install deepgram-sdk

This is MUCH more accurate than Whisper alone, especially for short commands.
"""

import os
import tempfile
import time
from typing import Optional
import numpy as np

import sounddevice as sd
import soundfile as sf

from src.utils import config


# =============================================================================
# DEEPGRAM CLIENT (Primary - Much Better!)
# =============================================================================

class DeepgramTranscriber:
    """Deepgram Nova-3 transcription - industry-leading accuracy."""
    
    def __init__(self):
        self.client = None
        self.available = False
        
        api_key = os.getenv("DEEPGRAM_API_KEY", "")
        if not api_key:
            print("⚠️  DEEPGRAM_API_KEY not found - using Whisper fallback")
            return
        
        try:
            from deepgram import DeepgramClient, PrerecordedOptions
            self.client = DeepgramClient(api_key=api_key)
            self.PrerecordedOptions = PrerecordedOptions
            self.available = True
            print("✅ Deepgram Nova-3 initialized (high accuracy mode)")
        except ImportError:
            print("⚠️  deepgram-sdk not installed. Run: pip install deepgram-sdk")
        except Exception as e:
            print(f"⚠️  Deepgram init failed: {e}")
    
    def transcribe(self, audio_path: str) -> Optional[str]:
        """Transcribe audio file using Deepgram Nova-3."""
        if not self.available or not self.client:
            return None
        
        try:
            with open(audio_path, "rb") as f:
                audio_data = f.read()
            
            # Configure for best accuracy with smart glasses commands
            options = self.PrerecordedOptions(
                model="nova-2",  # Best accuracy model
                language="en-US",
                smart_format=True,  # Better formatting
                punctuate=True,
                # Boost recognition of navigation commands
                keywords=[
                    "directions:2",
                    "Target:2", 
                    "Starbucks:2",
                    "navigate:2",
                    "next:2",
                    "continue:2",
                    "stop:2",
                    "status:2",
                    "describe:2",
                    "read:2",
                    "weather:2",
                ],
            )
            
            response = self.client.listen.rest.v("1").transcribe_file(
                {"buffer": audio_data, "mimetype": "audio/wav"},
                options
            )
            
            # Extract transcript
            transcript = response.results.channels[0].alternatives[0].transcript
            return transcript.strip() if transcript else None
            
        except Exception as e:
            print(f"⚠️  Deepgram error: {e}")
            return None


# =============================================================================
# WHISPER CLIENT (Fallback)
# =============================================================================

class WhisperTranscriber:
    """OpenAI Whisper fallback transcription."""
    
    def __init__(self):
        try:
            from openai import OpenAI
            timeout = float(getattr(config, "OPENAI_TIMEOUT_SECONDS", 30.0))
            try:
                self.client = OpenAI(timeout=timeout)
            except TypeError:
                self.client = OpenAI()
            self.available = True
        except Exception as e:
            print(f"⚠️  Whisper init failed: {e}")
            self.client = None
            self.available = False
    
    def transcribe(self, audio_path: str) -> Optional[str]:
        """Transcribe using OpenAI Whisper."""
        if not self.available or not self.client:
            return None
        
        try:
            with open(audio_path, "rb") as f:
                response = self.client.audio.transcriptions.create(
                    model=getattr(config, "OPENAI_TRANSCRIBE_MODEL", "whisper-1"),
                    file=f,
                    language="en",
                    temperature=0.0,
                    prompt=(
                        "Voice commands for AI smart glasses. "
                        "Commands: give me directions to Target, "
                        "navigate to Starbucks, next step, continue, "
                        "stop navigation, what do you see, describe the scene, "
                        "read this, how far, status, weather."
                    ),
                    response_format="text",
                )
            return (response or "").strip() or None
        except Exception as e:
            print(f"⚠️  Whisper error: {e}")
            return None


# =============================================================================
# MAIN VOICE LISTENER
# =============================================================================

class VoiceListener:
    """
    Smart Voice Listener with VAD and dual transcription engines.
    
    Uses Deepgram Nova-3 (primary) with Whisper fallback.
    Deepgram is MUCH more accurate, especially for short commands.
    """

    def __init__(
        self,
        duration_seconds: float = 8.0,
        sample_rate: int = 16000,  # 16kHz for better compatibility
        silence_threshold: float = 0.012,
        silence_duration: float = 1.3,
        min_speech_duration: float = 0.15,
    ):
        self.max_duration = duration_seconds
        self.sample_rate = sample_rate
        self.silence_threshold = silence_threshold
        self.silence_duration = silence_duration
        self.min_speech_duration = min_speech_duration
        
        # Chunk settings
        self.chunk_duration = 0.15  # 150ms chunks for responsiveness
        self.chunk_frames = int(self.chunk_duration * sample_rate)
        
        # Initialize transcription engines
        self.deepgram = DeepgramTranscriber()
        self.whisper = WhisperTranscriber()
        
        engine = "Deepgram Nova-3" if self.deepgram.available else "Whisper"
        print(
            f"🎙 VoiceListener initialized\n"
            f"   Engine: {engine}\n"
            f"   Max duration: {duration_seconds}s\n"
            f"   Silence to stop: {silence_duration}s"
        )

    def listen_and_transcribe(self) -> Optional[str]:
        """Main entry point - record and transcribe."""
        audio_path = self._record_with_vad()
        if not audio_path:
            return None
        
        try:
            # Try Deepgram first (much better accuracy)
            if self.deepgram.available:
                print("📨 Transcribing with Deepgram...")
                text = self.deepgram.transcribe(audio_path)
                if text:
                    print(f"📝 Transcription: {text!r}")
                    return text
                print("⚠️  Deepgram returned empty, trying Whisper...")
            
            # Fallback to Whisper
            if self.whisper.available:
                print("📨 Transcribing with Whisper...")
                text = self.whisper.transcribe(audio_path)
                if text:
                    print(f"📝 Transcription: {text!r}")
                    return text
            
            print("❌ Transcription failed")
            return None
            
        finally:
            # Clean up temp file
            try:
                os.remove(audio_path)
            except Exception:
                pass

    def _calculate_energy(self, audio_chunk: np.ndarray) -> float:
        """Calculate RMS energy of audio chunk."""
        if len(audio_chunk) == 0:
            return 0.0
        return float(np.sqrt(np.mean(audio_chunk ** 2)))

    def _record_with_vad(self) -> Optional[str]:
        """Record audio with Voice Activity Detection."""
        print("🎙 Listening... (speak now)")
        
        all_audio = []
        speech_started = False
        speech_frames = 0
        silence_frames = 0
        total_time = 0.0
        
        threshold = self.silence_threshold
        silence_frames_needed = int(self.silence_duration / self.chunk_duration)
        min_speech_frames = int(self.min_speech_duration / self.chunk_duration)
        
        start_time = time.time()
        
        try:
            while total_time < self.max_duration:
                # Record chunk
                chunk = sd.rec(
                    self.chunk_frames,
                    samplerate=self.sample_rate,
                    channels=1,
                    dtype="float32",
                )
                sd.wait()
                
                chunk = chunk.flatten()
                all_audio.append(chunk)
                total_time += self.chunk_duration
                
                energy = self._calculate_energy(chunk)
                is_speech = energy > threshold
                
                if not speech_started:
                    if is_speech:
                        speech_started = True
                        speech_frames = 1
                        silence_frames = 0
                        print("🗣️  Speech detected...")
                else:
                    if is_speech:
                        speech_frames += 1
                        silence_frames = 0
                    else:
                        silence_frames += 1
                        
                        if silence_frames >= silence_frames_needed:
                            # Accept any amount of speech
                            if speech_frames > 0:
                                elapsed = time.time() - start_time
                                print(f"✅ Done ({elapsed:.1f}s)")
                                break
            
            if total_time >= self.max_duration:
                print(f"⏱️  Max duration ({total_time:.1f}s)")
                
        except Exception as e:
            print(f"❌ Recording error: {e!r}")
            return None
        
        # Check for speech
        if not speech_started or speech_frames == 0:
            print("🔇 No speech detected")
            return None
        
        # Combine audio
        audio = np.concatenate(all_audio)
        
        # Trim trailing silence
        trim_chunks = max(0, silence_frames - 2)
        if trim_chunks > 0:
            trim_frames = int(trim_chunks * self.chunk_frames)
            if len(audio) > trim_frames:
                audio = audio[:-trim_frames]
        
        # Save to temp file
        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp_path = tmp.name
            sf.write(tmp_path, audio, self.sample_rate)
            duration = len(audio) / self.sample_rate
            print(f"💾 Recorded {duration:.1f}s")
            return tmp_path
        except Exception as e:
            print(f"❌ Save error: {e!r}")
            return None