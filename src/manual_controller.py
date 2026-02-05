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

        # Optional: offline speech-to-text for manual commands
        self.voice = None
        if getattr(config, "VOICE_INPUT_ENABLED", False) and VoiceInput is not None:
            try:
                self.voice = VoiceInput(
                    model_path=config.VOSK_MODEL_PATH,
                    device_index=config.VOICE_INPUT_DEVICE_INDEX,
                    target_rate=config.VOICE_INPUT_TARGET_RATE,
                    block_size=config.VOICE_INPUT_BLOCK_SIZE,
                )
                print("🎙️ Voice input ready (Vosk)")
            except Exception as e:
                # Keep the rest of the system working even if voice isn't set up.
                print(f"⚠️ Voice input disabled: {e}")
                self.voice = None

        self.camera_frame_width = self.camera.frame_width # frame width from camera handler

        print("⚡ MANUAL Smart Glasses System Initialized")

    def run(self) -> None:    
        # frame variable used to hold the current frame from the camera.
        # initially showing the first frame of the camera to open the window.
        frame = self.camera.capture_and_show_frame()
        annotated_frame = frame.copy() if frame is not None else None

        # instructions for the user
        if self.voice is None:
            print('Press r to process a frame. Press Ctrl+C to exit.')
        else:
            print('Press r to take a frame, then say: "hey what is in front of me". Press Ctrl+C to exit.')
        while True: # main loop
            if self.camera.wait_key_press('r'):  # if r is pressed...

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
                    print('\n🎙️ Listening... (say: "hey what is in front of me")')
                    transcript = self.voice.listen_once(timeout_seconds=config.VOICE_INPUT_TIMEOUT_SECONDS)
                    transcript = transcript.strip()

                    if not transcript:
                        print("🎙️ No speech detected.")
                        self.speech.speak("I didn't hear a command.")
                        continue

                    print(f"🎙️ Heard: {transcript}")
                    if not self.voice.matches_any(transcript, config.VOICE_DESCRIBE_COMMANDS):
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
