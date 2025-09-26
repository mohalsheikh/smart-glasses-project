"""
Controller module - orchestrates all components
"""

from src.camera_handler import CameraHandler
from src.object_detector import ObjectDetector
from src.currency_recognizer import CurrencyRecognizer
from src.ocr_engine import OCREngine
from src.speech_engine import SpeechEngine

class MainController:
    def __init__(self):
        self.camera = CameraHandler()
        self.detector = ObjectDetector()
        self.currency = CurrencyRecognizer()
        self.ocr = OCREngine()
        self.speech = SpeechEngine()

    def run(self):
        print("🚀 Smart Glasses System Starting...")
        # Example flow (pseudo-code for now):
        frame = self.camera.capture_frame()
        if frame is not None:
            objects = self.detector.detect(frame)
            text = self.ocr.read_text(frame)
            currency = self.currency.recognize(frame)

            message = f"Objects: {objects}, Text: {text}, Currency: {currency}"
            self.speech.speak(message)
