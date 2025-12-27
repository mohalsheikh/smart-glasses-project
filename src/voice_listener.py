# src/voice_listener.py

import os
import tempfile
from typing import Optional

import sounddevice as sd
import soundfile as sf
from openai import OpenAI

from src.utils import config


def _make_openai_client() -> OpenAI:
    """
    Create an OpenAI client with a default timeout if the SDK supports it.
    If not supported, fall back to OpenAI() with no kwargs.
    """
    try:
        return OpenAI(timeout=float(getattr(config, "OPENAI_TIMEOUT_SECONDS", 30.0)))
    except TypeError:
        return OpenAI()


client = _make_openai_client()


class VoiceListener:
    """
    Voice listener for your smart glasses.

    Flow:
      - Record audio
      - Transcribe with Whisper
      - Optionally clean transcript with GPT (for ASR mistakes)
    """

    def __init__(
        self,
        duration_seconds: float = 6.0,
        sample_rate: int = 22050,
    ):
        self.duration_seconds = duration_seconds
        self.sample_rate = sample_rate

        print(
            f"🎙 VoiceListener initialized "
            f"(duration={duration_seconds}s, sr={sample_rate}Hz)"
        )

    def listen_and_transcribe(self) -> Optional[str]:
        raw_text = self._record_and_transcribe_raw()
        if not raw_text:
            return None

        if self._is_simple_command(raw_text):
            cleaned = raw_text.strip()
            print(f"✨ Simple command detected, skipping GPT cleaning: {cleaned!r}")
            return cleaned

        cleaned = self._clean_transcription(raw_text)
        print(f"✨ Final cleaned query: {cleaned!r}")
        return cleaned

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _record_and_transcribe_raw(self) -> Optional[str]:
        print(f"🎙 Listening for {self.duration_seconds} seconds... speak now.")

        try:
            frames = int(self.duration_seconds * self.sample_rate)
            audio = sd.rec(
                frames,
                samplerate=self.sample_rate,
                channels=1,
                dtype="float32",
            )
            sd.wait()
        except Exception as e:
            print(f"❌ Error recording audio: {e!r}")
            return None

        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp_path = tmp.name
            sf.write(tmp_path, audio, self.sample_rate)
            print(f"💾 Saved recording to {tmp_path}")
        except Exception as e:
            print(f"❌ Error saving WAV file: {e!r}")
            return None

        try:
            with open(tmp_path, "rb") as f:
                print(f"📨 Sending audio to OpenAI transcription ({config.OPENAI_TRANSCRIBE_MODEL})...")
                resp = client.audio.transcriptions.create(
                    model=config.OPENAI_TRANSCRIBE_MODEL,
                    file=f,
                    language="en",
                    temperature=0,
                    prompt=(
                        "You are transcribing voice commands from a person "
                        "wearing smart glasses. They may ask about the weather, "
                        "directions, describing the environment around them, "
                        "reading text, or solving written questions.\n"
                        "Recognize phrases like:\n"
                        "- 'what is the weather like today'\n"
                        "- 'describe the environment'\n"
                        "- 'give me directions to'\n"
                        "- 'what do you see around me'\n"
                        "- 'read this'\n"
                        "- 'solve this question'\n"
                    ),
                    response_format="text",
                )
            text = (resp or "").strip()
            print(f"📝 Raw transcription: {text!r}")
            return text if text else None
        except Exception as e:
            print(f"❌ Error during transcription: {e!r}")
            return None
        finally:
            try:
                os.remove(tmp_path)
            except Exception:
                pass

    def _is_simple_command(self, raw: str) -> bool:
        if not raw:
            return False

        t = raw.lower().strip()
        words = t.split()

        if len(words) <= 7:
            trigger_phrases = [
                "solve this",
                "solve the question",
                "solve this question",
                "solve this problem",
                "answer this",
                "answer this question",
                "answer this problem",
                "read this",
                "read it",
            ]
            if any(p in t for p in trigger_phrases):
                return True

        return False

    def _clean_transcription(self, raw: str) -> str:
        try:
            print("🧹 Cleaning transcription with GPT...")
            resp = client.chat.completions.create(
                model=config.OPENAI_CHAT_MODEL,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You receive noisy ASR transcripts from a smart glasses microphone. "
                            "Correct misrecognized words and output the most likely user question.\n"
                            "Rules:\n"
                            "- Fix obvious ASR mistakes.\n"
                            "- Preserve intent.\n"
                            "- Output ONLY the corrected sentence."
                        ),
                    },
                    {"role": "user", "content": raw},
                ],
                max_tokens=64,
                temperature=0,
            )
            cleaned = (resp.choices[0].message.content or "").strip()
            print(f"🧹 Cleaned text: {cleaned!r}")
            return cleaned if cleaned else raw
        except Exception as e:
            print(f"⚠️ Error cleaning transcription with GPT, using raw text: {e!r}")
            return raw
