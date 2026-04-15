"""
Speech Engine - Smooth Raspberry Pi Voice (BEST QUALITY OFFLINE)
Uses pico2wave + sox processing for cleaner output
"""

import os
import subprocess
import tempfile


class SpeechEngine:
    def __init__(self):
        print("🔊 Smooth Speech Engine (pico2wave + sox) initialized")

    def speak(self, text):
        if not text:
            return

        print(f"🗣️ Speaking (smooth): {text}")

        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                raw_file = f.name
                processed_file = raw_file.replace(".wav", "_out.wav")

            # 1. Generate speech (pico2wave)
            subprocess.run([
                "pico2wave",
                "-w", raw_file,
                text
            ], check=True)

            # 2. Add silence + smoothing using sox
            subprocess.run([
                "sox",
                raw_file,
                processed_file,
                "pad", "0.3", "0"
            ], check=True)

            # 3. Play final audio
            subprocess.run(["aplay", processed_file], check=False)

            # cleanup
            os.remove(raw_file)
            os.remove(processed_file)

        except Exception as e:
            print(f"❌ Speech error: {e}")
