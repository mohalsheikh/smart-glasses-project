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
from src.voice_input import VoiceInput

import src.utils.config as config
from src.utils.object_description import summarize_detections, format_ocr_feedback

class MainController:
    def __init__(self) -> None:
        # Core components
        self.camera = CameraHandler()
        self.camera_frame_width = self.camera.frame_width # frame width from camera handler

        self.detector = ObjectDetector(model_name="yolov8n.pt")
        # self.currency = CurrencyRecognizer() # we probably don't need this separate component. ideally we should just let the object detector detect currency.
        self.ocr = OCREngine() # unfinished.
        self.speech = SpeechEngine()

        class_names_dict = self.detector.classes # this is a map of class ID to class name. these are the objects that our YOLO model(s) know
        self.voice = VoiceInput(model_class_names=class_names_dict) # pass the class names from the object detector to the voice input module so it can recognize them in commands.

        self.class_names = list(class_names_dict.values()) # list of class names from the object detector, used for command recognition in voice input.

        # set of all individual words that appear in class names, used for partial matching of class names in voice commands.
        self.partial_class_names = {word for name in self.class_names for word in name.split()} 

        print("⚡ MANUAL Smart Glasses System Initialized")

    # helper to clean up Vosk's unknown token from results.
    # We replace it with empty string and then strip whitespace. "[unk]" becomes "", something like "detect [unk]" becomes "detect". 
    def _remove_unk(self, s: str) -> str:
        return s.replace("[unk]", "").strip()
    
    def _print_ocr_feedback(self, detections):
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

    def run(self) -> None:    
        # frame variable used to hold the current frame from the camera.
        # initially showing the first frame of the camera to open the window.
        frame = self.camera.capture_and_show_frame()
        annotated_frame = frame.copy() if frame is not None else None

        while True: # main loop
            wake_word_input = self.voice.listen_wake_word(timeout_seconds=8.0)
            print(f"Wake word input: '{wake_word_input}'")  # Debug print for wake word input

            if 'vision' in wake_word_input:  # if the user said the wake word, we start the processing pipeline.
                self.speech.speak("I'm listening!")

                # first, the camera handler obtains a frame from the camera...
                frame = self.camera.capture_frame() 

                if frame is None:
                    print("⚠️ No frame from camera.")
                    continue

                # If voice is enabled, start listening AFTER we take the frame.
                # This matches the behavior you asked for (capture first, then ask the question).
                print('\n🎙️ Listening...')

                try:
                    transcript = self.voice.listen_command()
                except Exception as e:
                    print(f"[VoiceInput ERROR] {type(e).__name__}: {e}")
                    raise e

                # continue if we didn't get any transcript
                if not transcript:
                    print("🎙️ No speech detected.")
                    self.speech.speak("I didn't hear a command.")
                    continue
                
                # if we got a transcript, print it out for debugging and then clean it up by removing any "[unk]" tokens that Vosk might have included for unrecognized words.
                print(f"🎙️ Heard: {transcript}")

                cleaned_transcript = self._remove_unk(transcript)
                print(f"🎙️ Cleaned transcript: {cleaned_transcript}")

                # if after cleaning the transcript is empty, that means we didn't recognize any valid command words, so we should continue...
                if not cleaned_transcript:
                    print("🎙️ Command not recognized. Try again.")
                    self.speech.speak("Sorry, I didn't catch that.")
                    continue

                # command routing
                last_word = cleaned_transcript.split()[-1]

                description = None
                match last_word:
                    case "detect":
                        detections, annotated_frame = self.detector.detect(frame, annotate=True) # detect objects and get annotated frame
                        description = summarize_detections(detections, frame_width=self.camera_frame_width) # describe detections in natural language
                    case "read": 
                        detections, annotated_frame = self.detector.detect(frame, annotate=True) # detect objects and get annotated frame
                        detections = self.ocr.attach_crop_text_to_detected_objects(frame, detections) # read text on objects

                        self._print_ocr_feedback(detections)

                        detections = [det for det in detections if det.get("ocr_text")] # filter to just objects with text
                        description = summarize_detections(detections, frame_width=self.camera_frame_width) # describe detections of objects with text in natural language
                    case _:
                        # if this happens then the last word is a valid class name that YOLO can detect, or part of one.
                        # now we want to see what the user wants to do with this...

                        # we check for other terms in the cleaned transcript for two purposes: 
                        # 1) to look for command words like 'detect' or 'read'
                        # 2) to look for other class names that might indicate the user wants to deal with multiple classes at once
                        #    some class names are multiple words, so when we're iterating through the transcript we're checking for partial matches 
                        #    and building up potential multi-word class names that the user might be referring to. 
                        partial_obj_name = None 
                        objs_to_process = []

                        # processing last word first
                        if last_word in self.partial_class_names:
                            partial_obj_name = last_word # start building a potential multi-word class name with the last word if it's a partial match
                        elif last_word in self.class_names:
                            objs_to_process.append(last_word) # append last word to objects to process if it's a valid class name and not part of a larger class name
                        
                        for word in cleaned_transcript[:-1:-1].split(): # iterate through the cleaned transcript in reverse order, starting from the second to last word
                            # first check for partial match
                            if word in self.partial_class_names:
                                if partial_obj_name is None: # if we don't already have a partial object name, start one with the current word
                                    partial_obj_name = word
                                else:
                                    partial_obj_name = f"{word} {partial_obj_name}" # prepend word to the existing partial object name

                                    # if the new partial object name is a full match for a class name, add it to the objects to process and reset the partial object name
                                    if partial_obj_name in self.class_names: 
                                        objs_to_process.append(partial_obj_name)
                                        partial_obj_name = None

                            # if not partial match, check for full match
                            elif word in self.class_names: 
                                objs_to_process.append(word) # append word to objects to process if it's a valid class name and not part of a larger class name

                            # if neither, check for command words like 'detect' or 'read' 
                            elif word == "detect":
                                # TODO detect only the specified class(es)
                                break
                            elif word == "read":
                                # TODO read only the specified class(es)
                                break
                        
                        description = f"Final partial obj name was {partial_obj_name}, final objs to process: {objs_to_process}. yipee"
                        
                self.speech.speak(description)
                print(f"Frame processed: {description}")

                # # next, the object detector detects objects inside of the frame.
                # # from this we get the detection results and update the frame with annotations.
                
                # # print("Detection complete.")
                
                # # =========================================================================================
                # # TEST: Per-object OCR attachment
                # # =========================================================================================
                # try:
                #     detections = self.ocr.attach_crop_text_to_detected_objects(frame, detections)

                #     print("\n=== Per-object OCR test ===")
                #     for i, det in enumerate(detections):
                #         print(
                #             f"[{i}] "
                #             f"label={det.get('label')} "
                #             f"confidence={det.get('confidence')} "
                #             f"bbox={det.get('bbox')} "
                #             f"ocr_text='{det.get('ocr_text', '')}'"
                #         )
                #     print("=== End per-object OCR test ===\n")

                # except Exception as e:
                #     print(f"[Per-object OCR test ERROR] {type(e).__name__}: {e}")
                # # =========================================================================================
                # # Results: Successful attachment of OCR text to detected objects. But might have bad confidence keys.   
                # # ==========================================================================================

                # # Run OCR (Extract text) on the full frame and 
                # # format confidence-based feedback for the user based on annotated confidence values.
                # print("🔍 Running OCR on frame...")
                # ocr_result = self.ocr.extract_text_with_confidence(frame)
                # ocr_feedback = format_ocr_feedback(ocr_result)
                # if ocr_result.get("text"):
                #     print(ocr_feedback)
                # else:
                #     print("❌🔍 No text detected.\n")

                # # finally, we summarize the detections and speak them out loud.
                # description = summarize_detections(detections, frame_width=self.camera_frame_width)
                # # print("Generated description.")

                # self.speech.speak(description)

                # print(f"Frame processed: {description}")
            
            if annotated_frame is not None:
                self.camera.show_image(annotated_frame) # just keep showing the last frame so that the window doesn't say not responding.
