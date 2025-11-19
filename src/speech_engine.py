import sys
import pyttsx3

class SpeechEngine:
    def __init__(self, rate=180, volume=1.0):
        self.rate = rate
        self.volume = volume
        self.minimum_length = 1; self.max_length = 250
        self.prefix_enabled = False
        self.prefix_text = ""
        self.silence_token = "[silence]"
        self.last_text = ""
        self.repeat_limit = 3
        self.repeat_counter = 0
        self.engine_driver = self._select_driver()
        self.voice_index = 0
        self.volume_step = 0.1
        self.rate_step = 10
        self.min_volume = 0.0; self.max_volume = 1.0
        self.min_rate = 80
        self.max_rate = 260

    def _select_driver(self):
        if sys.platform == "win32":
            return "sapi5"
        if sys.platform == "darwin":
            return "nsss"
        return "espeak"

    def _create_engine(self):
        driver = self.engine_driver
        try:
            engine = pyttsx3.init(driverName=driver)
        except Exception:
            engine = pyttsx3.init()
        try:
            engine.setProperty("rate", self.rate)
        except Exception:
            pass
        try:
            engine.setProperty("volume", self.volume)
        except Exception:
            pass
        try:
            voices = engine.getProperty("voices")
        except Exception:
            voices = []
        if voices:
            index = self.voice_index
            if index < 0:
                index = 0
            if index >= len(voices):
                index = 0
            try:
                engine.setProperty("voice", voices[index].id)
            except Exception:
                pass
        return engine

    def _sanitize_text(self, text):
        if text is None:
            return ""
        value = str(text)
        value = value.strip()
        if not value:
            return ""
        if len(value) > self.max_length:
            value = value[: self.max_length]
        return value

    def _apply_prefix(self, text):
        if not self.prefix_enabled:
            return text
        if not self.prefix_text:
            return text
        joined = self.prefix_text + text
        return joined

    def _is_silence(self, text):
        if not text:
            return True
        if text == self.silence_token:
            return True
        return False

    def _should_speak(self, text):
        if self._is_silence(text):
            return False
        if len(text) < self.minimum_length:
            return False
        if text == self.last_text:
            if self.repeat_counter >= self.repeat_limit:
                return False
            self.repeat_counter += 1
            return True
        self.last_text = text
        self.repeat_counter = 1
        return True

    def set_prefix(self, text):
        clean = self._sanitize_text(text)
        self.prefix_text = clean

    def enable_prefix(self):
        self.prefix_enabled = True

    def disable_prefix(self):
        self.prefix_enabled = False

    def set_voice_index(self, index):
        if index is None:
            return
        try:
            value = int(index)
        except Exception:
            return
        self.voice_index = value

    def reset_state(self):
        self.last_text = ""
        self.repeat_counter = 0

    def speak(self, text):
        cleaned = self._sanitize_text(text)
        cleaned = self._apply_prefix(cleaned)
        if not self._should_speak(cleaned):
            return
        engine = None
        try:
            engine = self._create_engine()
            engine.say(cleaned)
            engine.runAndWait()
        except Exception:
            pass
        finally:
            if engine is not None:
                try:
                    engine.stop()
                except Exception:
                    pass

    def get_rate(self):
        return self.rate

    def get_volume(self):
        return self.volume

    def get_minimum_length(self):
        return self.minimum_length

    def get_prefix(self):
        return self.prefix_text

    def has_last_text(self):
        return bool(self.last_text)
