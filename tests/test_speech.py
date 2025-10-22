import sys
import pyttsx3

def test_text_to_speech():
    # Pick the right driver explicitly (optional but helpful)
    driver = {"win32": "sapi5", "darwin": "nsss"}.get(sys.platform, "espeak")
    engine = pyttsx3.init(driverName=driver)

    # Tweak basics
    engine.setProperty("rate", 180)         # words per minute
    engine.setProperty("volume", 1.0)       # 0.0 to 1.0

    # Try to select a voice (first one found)
    voices = engine.getProperty("voices")
    if voices:
        engine.setProperty("voice", voices[0].id)

    engine.say("Hello, this is a test.")
    engine.runAndWait()
    engine.stop()

if __name__ == "__main__":
    test_text_to_speech()
