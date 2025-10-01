"""
Text-to-Speech wrapper using pyttsx3 (offline, works on macOS with NSSS).
"""

import pyttsx3


class SpeechEngine:
    def __init__(self, rate: int = 190, volume: float = 1.0):
        self.engine = pyttsx3.init()
        try:
            self.engine.setProperty("rate", rate)
            self.engine.setProperty("volume", volume)
        except Exception:
            pass

    def speak(self, text: str):
        if not text:
            return
        print(f"🔊 Speaking: {text}")
        self.engine.say(text)
        self.engine.runAndWait()
