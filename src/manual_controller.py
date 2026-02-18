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
from src.utils.object_description import summarize_detections, format_ocr_feedback

# voice input is optional (only used in manual mode for now)
try:
    from src.voice_input import VoiceInput
except Exception:
    VoiceInput = None  # type: ignore

class MainController:
    def __init__(self) -> None:
        # Core components
        self.camera = CameraHandler()
        self.detector = ObjectDetector()
        # self.currency = CurrencyRecognizer() # we probably don't need this separate component. ideally we should just let the object detector detect currency.
        self.ocr = OCREngine() # unfinished.
        self.speech = SpeechEngine()
        self.voice = VoiceInput()

        self.camera_frame_width = self.camera.frame_width # frame width from camera handler

        print("⚡ MANUAL Smart Glasses System Initialized")

    # helper to clean up Vosk's unknown token from results.
    # We replace it with empty string and then strip whitespace. "[unk]" becomes "", something like "detect [unk]" becomes "detect". 
    def _remove_unk(self, s: str) -> str:
        return s.replace("[unk]", "").strip()

    def run(self) -> None:    
        # frame variable used to hold the current frame from the camera.
        # initially showing the first frame of the camera to open the window.
        frame = self.camera.capture_and_show_frame()
        annotated_frame = frame.copy() if frame is not None else None

        # instructions for the user
        print('Press r to process a frame. Press Ctrl+C to exit.')

        while True: # main loop
            if self.camera.wait_key_press('r'):  # if r is pressed...
                self.speech.speak("I'm listening!")

                # print("r pressed.")
                # first, the camera handler obtains a frame from the camera...
                frame = self.camera.capture_frame() 
                # print("Got frame from camera.")

                if frame is None:
                    print("⚠️ No frame from camera.")
                    continue

                # If voice is enabled, start listening AFTER we take the frame.
                # This matches the behavior you asked for (capture first, then ask the question).
                if self.voice is not None:
                    print('\n🎙️ Listening...')

                    try:
                        transcript = self.voice.listen_command()
                    except Exception as e:
                        print(f"[VoiceInput ERROR] {type(e).__name__}: {e}")
                        transcript = ""
                        raise e

                    if not transcript:
                        print("🎙️ No speech detected.")
                        self.speech.speak("I didn't hear a command.")
                        continue
                    
                    print(f"🎙️ Heard: {transcript}")

                    cleaned_transcript = self._remove_unk(transcript)
                    print(f"🎙️ Cleaned transcript: {cleaned_transcript}")
                    
                    if not cleaned_transcript:
                        print("🎙️ Command not recognized. Try again.")
                        self.speech.speak("Sorry, I didn't catch that.")
                        continue

            
                # next, the object detector detects objects inside of the frame.
                # from this we get the detection results and update the frame with annotations.
                detections, annotated_frame = self.detector.detect(frame, annotate=True)
                # print("Detection complete.")
                
                # =========================================================================================
                # TEST: Per-object OCR attachment
                # =========================================================================================
                try:
                    detections = self.ocr.attach_crop_text_to_detected_objects(frame, detections)

                    print("\n=== Per-object OCR test ===")
                    for i, det in enumerate(detections):
                        print(
                            f"[{i}] "
                            f"label={det.get('label')} "
                            f"confidence={det.get('confidence')} "
                            f"bbox={det.get('bbox')} "
                            f"ocr_text='{det.get('ocr_text', '')}'"
                        )
                    print("=== End per-object OCR test ===\n")

                except Exception as e:
                    print(f"[Per-object OCR test ERROR] {type(e).__name__}: {e}")
                # =========================================================================================
                # Results: Successful attachment of OCR text to detected objects. But might have bad confidence keys.   
                # ==========================================================================================

                # Run OCR (Extract text) on the full frame and 
                # format confidence-based feedback for the user based on annotated confidence values.
                print("🔍 Running OCR on frame...")
                ocr_result = self.ocr.extract_text_with_confidence(frame)
                ocr_feedback = format_ocr_feedback(ocr_result)
                if ocr_result.get("text"):
                    print(ocr_feedback)
                else:
                    print("❌🔍 No text detected.\n")

                # finally, we summarize the detections and speak them out loud.
                description = summarize_detections(detections, frame_width=self.camera_frame_width)
                # print("Generated description.")

                self.speech.speak(description)

                print(f"Frame processed: {description}")
            
            if annotated_frame is not None:
                self.camera.show_image(annotated_frame) # just keep showing the last frame so that the window doesn't say not responding.
