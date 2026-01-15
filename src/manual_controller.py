"""
Manual controller module created by Ethan. 
This activates our pipeline upon user input rather than periodically.
Currently, this user input is a key press, however in the future it will be voice commands.

This module is responsible for:
- Initializing hardware and core services
- Running the main processing loop
- Delegating description/summarization to helper utilities
"""

from src.camera_handler import CameraHandler
from src.currency_recognizer import CurrencyRecognizer
from src.object_detector import ObjectDetector
from src.ocr_engine import OCREngine
from src.speech_engine import SpeechEngine

import src.utils.config as config
from src.utils.object_description import summarize_detections

class MainController:
    def __init__(self) -> None:
        # Core components
        self.camera = CameraHandler()
        self.detector = ObjectDetector()
        self.currency = CurrencyRecognizer() # we probably don't need this separate component. ideally we should just let the object detector detect currency.
        self.ocr = OCREngine() # unfinished.
        self.speech = SpeechEngine()

        print("⚡ MANUAL Smart Glasses System Initialized")
        print(f"📊 Model: {config.DEFAULT_MODEL_NAME}")

    def run(self) -> None:    
        # showing the first frame of the camera to open the window.


        # instructions for the user
        print('Press Spacebar to process a frame. Press Ctrl+C to exit.')
        while True: # main loop
            if self.camera.wait_key_press('r'):  # if r is pressed...
                self.camera.capture_and_show_frame()
                print("Frame processed.")