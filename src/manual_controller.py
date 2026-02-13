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

class MainController:
    def __init__(self) -> None:
        # Core components
        self.camera = CameraHandler()
        self.detector = ObjectDetector()
        self.currency = CurrencyRecognizer() # we probably don't need this separate component. ideally we should just let the object detector detect currency.
        self.ocr = OCREngine() # unfinished.
        self.speech = SpeechEngine()

        self.camera_frame_width = self.camera.frame_width # frame width from camera handler

        print("⚡ MANUAL Smart Glasses System Initialized")
        print(f"📊 Model: {config.DEFAULT_MODEL_NAME}")

    def run(self) -> None:    
        # frame variable used to hold the current frame from the camera.
        # initially showing the first frame of the camera to open the window.
        frame = self.camera.capture_and_show_frame()
        annotated_frame = frame.copy()

        # instructions for the user
        print('Press r to process a frame. Press Ctrl+C to exit.')
        while True: # main loop
            if self.camera.wait_key_press('r'):  # if r is pressed...


                # print("r pressed.")
                # first, the camera handler obtains a frame from the camera...
                frame = self.camera.capture_frame() 
                # print("Got frame from camera.")

                # detections, annotated_frame = self.detector.detect(frame, annotate=True)

                # Run OCR (Extract text) on the full frame and 
                # format confidence-based feedback for the user based on annotated confidence values.
                print()
                print("✅ OCR Ready!")
                print("🔍 Running OCR on frame...")
                # The object detector detects objects inside of the frame.
                # from this we get the detection results and update the frame with annotations.
                detections, annotated_frame = self.detector.detect(frame, annotate=True)
                print("✅ Detection complete.")

                ocr_result = self.ocr.extract_text_with_confidence(frame)
                ocr_feedback = format_ocr_feedback(ocr_result)
                if ocr_result.get("text"):
                    print(ocr_feedback)
                else:
                    print("❌🔍 No text detected.\n")
                # =========================================================================================
                # TEST: Per-object OCR attachment
                # =========================================================================================
                try:
                    detections = self.ocr.attach_crop_text_to_detected_objects(frame, detections)

                    print("ℹ️  Per-object OCR attachment results ℹ️")
                    print("====================================================")

                    for i, det in enumerate(detections):
                        # Extract Values
                        label = det.get("label")
                        confidence = det.get("confidence")
                        bbox = det.get("bbox")
                        ocr_text = det.get("ocr_text")
                        # Format confidence (round to 2 decimals)
                        confidence_str = f"{float(confidence):.2f}" if confidence is not None else "n/a"
                        # Simplify bounding box feedback (remove np.float32 text)
                        if bbox is not None:
                            try:
                                x1, y1, x2, y2 = bbox
                                bbox_str = f"({int(x1)}, {int(y1)}, {int(x2)}, {int(y2)})"
                            except Exception:
                                bbox_str = "n/a"
                        else:
                            bbox_str = "n/a"
                        # Format OCR text (strip whitespace, show (none) if empty)
                        ocr_text_str = (ocr_text or "").strip()
                        # Format label (strip whitespace, show n/a if empty)
                        label_str = str(label).strip() if label else "n/a"
                        # Output Results with Formtting 
                        print(f" ID [{i:02d}]")
                        print(f"   Label:         {label_str}")
                        print(f"   Conf:          {confidence_str}")
                        print(f"   BBox:          {bbox_str}")
                        if ocr_text_str:
                            print(f"   Attached_Text: '{ocr_text_str}'")
                        else:
                            print(f"   Attached_Text: none")
                    print("====================================================")

                except Exception as e:
                    print(f"[Per-object OCR attachment ERROR] {type(e).__name__}: {e}")
                # =========================================================================================
                # Results: Successful attachment of OCR text to detected objects. But might have bad confidence keys.   
                # ==========================================================================================

                # finally, we summarize the detections and speak them out loud.
                description = summarize_detections(detections, frame_width=self.camera_frame_width)
                # print("Generated description.")

                self.speech.speak(description)

                print(f"Frame processed: {description}")
            
            self.camera.show_image(annotated_frame) # just keep showing the last frame so that the window doesn't say not responding.