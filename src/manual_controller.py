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
from src.utils.object_description import summarize_detections, format_ocr_feedback, normalize_label

import threading
from queue import Queue
from collections import deque
import numpy as np
import enum

FRAME_COUNT = 3 # the number of frames provided to objectdetector and ocrengine for detection and reading.

class VoiceInputState(enum.Enum):
    WAITING_FOR_WAKE_WORD = 1,
    WAITING_FOR_COMMAND = 2

class MainController:
    def __init__(self) -> None:
        # Core components
        self.camera = CameraHandler()
        self.camera_frame_width = self.camera.frame_width # frame width from camera handler

        self.detector = ObjectDetector(model_names=["yolov8n.pt", "currency_detector.pt"])
        # To use multiple models, pass model_names instead:
        # self.detector = ObjectDetector(model_names=["yolov8n.pt", "currency_detector.pt"])

        # self.currency = CurrencyRecognizer() # we probably don't need this separate component. ideally we should just let the object detector detect currency.
        self.ocr = OCREngine()
        self.speech = SpeechEngine()

        class_names_dict = self.detector.classes # this is a map of class ID to class name. these are the objects that our YOLO model(s) know
        self.voice = VoiceInput(model_class_names=class_names_dict) # pass the class names from the object detector to the voice input module so it can recognize them in commands.
        self.commands = self.voice.commands # the command grammar minus "[unk]" and the class names.

        # Build the list of *normalized* class names used for command matching.
        # This keeps the controller in sync with the normalized grammar that
        # VoiceInput now builds (see voice_input.py).
        normalized_names = set()
        for name in class_names_dict.values():
            norm = normalize_label(name)
            if norm is not None:
                normalized_names.add(norm)

        self.class_names = list(normalized_names)

        # sort class names by number of words from greatest to least (determined by counting spaces).
        # this is done to ensure that when we do partial matching of class names in the transcript, we check for longer class names first (therefore they are matched first).
        # for instance if a hypothetical model knew "cell phone" and "phone", and the user said "detect cell phone", we match "cell phone" instead of phone.
        self.class_names.sort(key=lambda s: s.count(" "), reverse=True) 

        # set of all individual words that appear in (normalized) class names, used for partial matching of class names in voice commands.
        self.partial_class_names = {word for name in self.class_names for word in name.split()} 

        self.speech_queue = Queue() # queue for text to be spoken by the speech thread
        self.voice_input_result_q = Queue() # queue for results from voice input thread
        self.voice_input_state_q = Queue() # queue for state of voice input thread (waiting for wake word or waiting for command)

        print("⚡ MANUAL Smart Glasses System Initialized")

    # thread for speech output tasks, in parallel with the main loop.
    def _tts_worker(self):
        # only speaks if there is a new text in the speech queue, which is managed by the main thread.
        while True:
            try:
                text = self.speech_queue.get(timeout=0.1)
                self.speech.speak(text)
            except Exception:
                continue

    # thread for voice input tasks, in parallel with the main loop.
    # has two states: waiting for wake word and waiting for command.
    # states are updated by the main thread through the voice_input_state_q, and results are sent back to the main thread through the voice_input_result_q.
    def _voice_input_worker(self):
        waiting_for = VoiceInputState.WAITING_FOR_WAKE_WORD

        while True:
            try:
                waiting_for = self.voice_input_state_q.get(timeout=1)
            except Exception:
                pass

            match waiting_for:
                case VoiceInputState.WAITING_FOR_WAKE_WORD:
                    wake_word_input = self.voice.listen_wake_word(timeout_seconds=8.0)
                    self.voice_input_result_q.put((VoiceInputState.WAITING_FOR_WAKE_WORD, wake_word_input)) # put the wake word input in the queue for the main thread to process

                case VoiceInputState.WAITING_FOR_COMMAND:
                    command_input = self.voice.listen_command()
                    self.voice_input_result_q.put((VoiceInputState.WAITING_FOR_COMMAND, command_input)) # put the command input in the queue for the main thread to process
                    

    def _start_worker_threads(self):
        tts_thread = threading.Thread(target=self._tts_worker, name="TTSThread", daemon=True)
        voice_input_thread = threading.Thread(target=self._voice_input_worker, name="VoiceInputThread", daemon=True)

        tts_thread.start()
        voice_input_thread.start()

    # determines action to take based on the value of command, and returns a natural language description of the result to be spoken to the user.
    # objs is the list of objects that we want to process. If it is none, we are processing all objects that the ObjectDetector knows.
    def _route_command(self, command: str, cleaned_transcript: str, frames: list[np.ndarray], prev_desc: str, objs: list[str] = None):
        final_frames = None
        match command:
##############################################################################################################
# # Action commands
##############################################################################################################  
            case "detect":
                detections, final_frames = self.detector.detect(frames, annotate=True, objects=objs) # detect objects and get annotated frame
                description = summarize_detections(detections, frame_width=self.camera_frame_width) # describe detections in natural language 
            case "read": 
                detections, final_frames = self.detector.detect(frames, annotate=True, objects=objs) # detect objects and get annotated frame
                detections = [self.ocr.attach_crop_text_to_detected_objects(frames[i], det) for i, det in enumerate(detections)] # read text on objects

                for i, det in enumerate(detections): 
                    self._print_ocr_feedback(det) # print OCR feedback for each detected object in a readable format
                    detections[i] = [d for d in det if d.get("ocr_text") is not None] # filter to just objects with text for description

                text_detections = [det for det in detections if det.get("ocr_text")] # filter to just objects with text

                if not text_detections:
                    if detections:
                        description = "I found objects, but I couldn't read any text."
                    else:
                        description = "I don't see anything to read."
                else:
                    description = summarize_detections(text_detections, frame_width=self.camera_frame_width) # describe detections of objects with text in natural language
##############################################################################################################
# # Quit commands
##############################################################################################################       
            case "sleep" | "end" | "nevermind" | "thanks":
                description = "Going back to sleep..."
                final_frames = None # frames.copy() if frames is not None else None
##############################################################################################################
# # Repeat commands
##############################################################################################################   
            case "repeat":
                description = prev_desc
                final_frames = None # frames.copy() if frames is not None else None
##############################################################################################################
# # Default
############################################################################################################## 
            case _:
                # if this happens we assume the user wants to detect/read specific objects mentioned in the command.
                
                # find the index of the command word in the transcript so that we can extract the part of the transcript after the command word, which should contain the object names that the user wants to deal with.
                index_of_command = max([cleaned_transcript.rfind(cmd) for cmd in self.commands])
                transcript_with_command = cleaned_transcript[index_of_command:]
                
                # extract object names from the part of the transcript after the command word. 
                objs_to_process = self._extract_objs_from_transcript(transcript_with_command)

                command = transcript_with_command.split()[0]
                if command in self.commands and not len(objs_to_process) == 0: 
                    description, final_frames = self._route_command(command, cleaned_transcript, frames, prev_desc, objs=objs_to_process) # recursively call _route_command with the specific objects to process. 
                else:
                    description = f"Sorry, I didn't understand the command. I heard '{cleaned_transcript}'."
                    final_frames = None # frames.copy() if frames is not None else None

        return description, final_frames

    # helper to clean up Vosk's unknown token from results.
    # We replace it with empty string and then strip whitespace. "[unk]" becomes "", something like "detect [unk]" becomes "detect". 
    def _remove_unk(self, s: str) -> str:
        return s.replace("[unk]", "").strip()
    
    # helper to extract object names from the transcript for commands.
    def _extract_objs_from_transcript(self, transcript: str) -> list[str]:
        objs_to_process = set()

        for class_name in self.class_names:
            if class_name in transcript:
                objs_to_process.add(class_name)
                transcript = transcript.replace(class_name, "") # remove the class name from the transcript so that we can check for command words without the class names in the way

        return list(objs_to_process)
    
    # helper to print OCR feedback for each detected object in a readable format.
    def _print_ocr_feedback(self, detections):
        # =========================================================================================
        # TEST: Per-object OCR attachment
        # =========================================================================================
        try:
            print()
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

    def run(self) -> None:    
        # frames deque (used as queue) used to hold the FRAME_COUNT most recent frames from the camera.
        frames = deque()

        # initially populating with FRAME_COUNT frames from the camera.
        # this serves the purpose of making it so that we don't have to check the size of the frames deque when we add to it;
        # whenever we capture a new frame we always will add one and remove one, so the size will always be constant and equal to FRAME_COUNT after this initial population step.
        for _ in range(FRAME_COUNT):
            frame = self.camera.capture_and_show_frame("Live Camera Feed")
            frames.append(frame)
        
        # list of annotations from most recent detections, shown in detections window.
        # initialized to the initial FRAME_COUNT unannotated frames
        annotated_frames: list[np.ndarray] = list(frames)

        annotated_frames_index = 0 # index of the annotated frame to show in the detections window

        self._start_worker_threads() # start the display and speech worker threads
        description = "I have nothing to repeat." # used to store spoken descriptions, saved across loops to support the "repeat" command.

        while True: # main loop
            # capturing a new frame and replacing the oldest frame in the frames deque with it
            frame = self.camera.capture_and_show_frame(window_name="Live Camera Feed") # keep showing the camera feed for our own convenience
            frames.append(frame) 
            frames.popleft()

            self.camera.show_image(annotated_frames[annotated_frames_index], window_name="Detections")
            annotated_frames_index = (annotated_frames_index + 1) % FRAME_COUNT # index of next annotated frame for next loop iteration

            if self.camera.wait_key_press('q', delay=10): # if the user presses 'q', we quit the program.
                print("Exiting program.")
                break

            try:
                voice_input_result = self.voice_input_result_q.get_nowait() # check if there is a new result from the voice input thread
            except Exception:
                continue
            
            voice_input_state = voice_input_result[0]
            transcript = voice_input_result[1]
            match voice_input_state:
                case VoiceInputState.WAITING_FOR_WAKE_WORD:
                    print(f"🎙️ Wake word input: '{transcript}'")  # Debug print for wake word input
                    
                    if 'vision' in transcript:  # if the user said the wake word, we start the processing pipeline.
                        self.speech_queue.put("I'm listening!")
                        self.voice_input_state_q.put(VoiceInputState.WAITING_FOR_COMMAND) # tell the voice input thread to start listening for a command
                case VoiceInputState.WAITING_FOR_COMMAND:
                        # before processing the command, go back to waiting for the wake word, because right now this is always what we want.
                        self.voice_input_state_q.put(VoiceInputState.WAITING_FOR_WAKE_WORD) 

                        print(f"🎙️ Heard command: '{transcript}'")
                        
                        # continue if we didn't get any transcript
                        if not transcript:
                            print("🎙️ No speech detected.")
                            # self.speech.speak("I didn't hear a command.")
                            self.speech_queue.put("I didn't hear a command.")
                            continue
                        
                        # if we got a transcript, print it out for debugging and then clean it up by removing any "[unk]" tokens that Vosk might have included for unrecognized words.
                        print(f"🎙️ Heard: {transcript}")

                        cleaned_transcript = self._remove_unk(transcript)
                        print(f"🎙️ Cleaned transcript: {cleaned_transcript}")

                        # if after cleaning the transcript is empty, that means we didn't recognize any valid command words, so we should continue...
                        if not cleaned_transcript:
                            print("🎙️ Command not recognized. Try again.")
                            # self.speech.speak("Sorry, I didn't catch that.")
                            self.speech_queue.put("Sorry, I didn't catch that.")
                            continue

                        # command routing determines what action to take based on the transcript.
                        split_transcript = cleaned_transcript.split()
                        last_word = split_transcript[-1] # We assume the command is the last_word in the cleaned transcript... if it isn't, it is likely an object name that the user wants to detect/read.

                        results = self._route_command(last_word, cleaned_transcript, frames, description)
                        description = results[0]
                        annotated_frames = results[1] if results[1] is not None else annotated_frames # only updating frames if frames result from route command isn't None (i.e. if no new detections were called for).
        
                        self.speech_queue.put(description)
                        print(f"Output: {description}")

            # wake_word_input = self.voice.listen_wake_word(timeout_seconds=8.0)
            # print(f"Wake word input: '{wake_word_input}'")  # Debug print for wake word input

            # if 'vision' in wake_word_input:  # if the user said the wake word, we start the processing pipeline.
            #     # self.speech.speak("I'm listening!")
            #     self.speech_queue.put("I'm listening!")

            #     # first, the camera handler obtains a frame from the camera...
            #     frame = self.camera.capture_frame() 
            #     if frame is None:
            #         # continue to the next iteration of the loop if we didn't get a frame
            #         print("⚠️ No frame from camera.")
            #         continue

            #     # listen for voice input from the user.
            #     try:
            #         transcript = self.voice.listen_command()
            #     except Exception as e:
            #         print(f"[VoiceInput ERROR] {type(e).__name__}: {e}")
            #         raise e

           
            
            # if annotated_frame is not None:
            #     print("Displaying annotated frame.")
            #     self.camera.show_image(annotated_frame) # just keep showing the last frame so that the window doesn't say not responding.
