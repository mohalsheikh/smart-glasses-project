"""
Manual controller module created by Ethan. 
This activates our pipeline upon user input rather than periodically.
Currently, this user input is a key press, however in the future it will be voice commands.

UPDATED (Sprint 2):
- Always listening for wake phrase: "hey vision"
- After wake phrase, speech says: "I'm listening."
- Then listens for a command, captures a frame, runs YOLO/OCR/Currency depending on command,
  and speaks only what the user asked for.

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
from src.voice_listener import VoiceCommandListener

import src.utils.config as config
from src.utils.object_description import (
    summarize_detections,
    format_ocr_feedback,
    summarize_target_location,
    filter_detections_by_label,
)

from src.utils.command_router import parse_command


class MainController:
    def __init__(self) -> None:
        # Core components
        self.camera = CameraHandler()
        self.detector = ObjectDetector()
        self.currency = CurrencyRecognizer()
        self.ocr = OCREngine()  # unfinished but usable for reading text
        self.speech = SpeechEngine()

        self.camera_frame_width = self.camera.frame_width
        self.camera_frame_height = self.camera.frame_height

        # Voice listener (always-on)
        self.voice = None
        if config.VOICE_INPUT_ENABLED:
            self.voice = VoiceCommandListener()

        print("⚡ MANUAL Smart Glasses System Initialized")
        print(f"📊 Model: {config.DEFAULT_MODEL_NAME}")

    def _process_command(self, command_text: str) -> None:
        """
        Takes the recognized command string, captures a fresh frame,
        then runs the correct pipeline and speaks the result.
        """
        # Capture a fresh frame AFTER we have a command (matches your desired flow)
        frame = self.camera.capture_frame()
        if frame is None:
            self.speech.speak("Sorry, I couldn't capture a frame.")
            return

        cmd = parse_command(command_text)

        # Always run YOLO only when needed (performance-friendly)
        if cmd.intent in ["describe", "locate_object", "count_object", "read_object"]:
            detections, annotated_frame = self.detector.detect(frame, annotate=True)
        else:
            detections, annotated_frame = [], frame

        # -------------------------
        # Command: DESCRIBE SCENE
        # -------------------------
        if cmd.intent == "describe":
            description = summarize_detections(detections, frame_width=self.camera_frame_width)
            self.speech.speak(description)
            print(f"[VOICE] {cmd.intent}: {description}")
            return

        # -------------------------
        # Command: LOCATE OBJECT (target-only speech)
        # -------------------------
        if cmd.intent == "locate_object":
            target = cmd.target
            # If target is missing, ask user to try again
            if not target:
                self.speech.speak("Which object are you looking for? Try: where is my bottle.")
                return

            response = summarize_target_location(
                detections,
                frame_width=self.camera_frame_width,
                frame_height=self.camera_frame_height,
                target_label=target,
            )
            self.speech.speak(response)
            print(f"[VOICE] locate {target}: {response}")
            return

        # -------------------------
        # Command: COUNT/FILTER OBJECT
        # “are there any ____”
        # -------------------------
        if cmd.intent == "count_object":
            target = cmd.target
            if not target:
                self.speech.speak("Which object should I check for? Try: are there any bottles.")
                return

            matches = filter_detections_by_label(detections, target)
            if not matches:
                response = f"I don't see any {target}s. Want me to describe what I do see?"
            elif len(matches) == 1:
                response = f"Yes, I see one {target}."
            else:
                response = f"Yes, I see {len(matches)} {target}s."
            self.speech.speak(response)
            print(f"[VOICE] count {target}: {response}")
            return

        # -------------------------
        # Command: READ FULL FRAME TEXT
        # -------------------------
        if cmd.intent == "read_frame":
            print("🔍 Running OCR on frame (read this)...")
            ocr_result = self.ocr.extract_text_with_confidence(frame)
            if ocr_result.get("text"):
                # Speak the text (plus a short confidence hint)
                spoken = f"It says: {ocr_result['text']}"
                self.speech.speak(spoken)
                print(format_ocr_feedback(ocr_result))
            else:
                self.speech.speak("I don't see any readable text.")
            return

        # -------------------------
        # Command: READ TEXT ON A SPECIFIC OBJECT
        # -------------------------
        if cmd.intent == "read_object":
            target = cmd.target
            if not target:
                self.speech.speak("What should I read? Try: read the label or read the book.")
                return

            matches = filter_detections_by_label(detections, target)
            if not matches:
                self.speech.speak(f"I don't see a {target}. Want me to describe what I do see?")
                return

            # Choose the best match (largest bbox usually closest)
            best = max(matches, key=lambda d: (d.get("bbox")[2] - d.get("bbox")[0]) * (d.get("bbox")[3] - d.get("bbox")[1]))

            x1, y1, x2, y2 = [int(float(v)) for v in best["bbox"]]
            crop = frame[max(0, y1):max(0, y2), max(0, x1):max(0, x2)]

            text = self.ocr.extract_text_as_string(crop)
            if text:
                self.speech.speak(f"The {target} says: {text}")
            else:
                self.speech.speak(f"I don't see readable text on the {target}.")
            return

        # -------------------------
        # Command: CURRENCY
        # -------------------------
        if cmd.intent == "currency":
            result = self.currency.recognize(frame)
            self.speech.speak(result)
            print(f"[VOICE] currency: {result}")
            return

        # -------------------------
        # HELP / UNKNOWN
        # -------------------------
        if cmd.intent == "help" or cmd.intent == "unknown":
            help_text = (
                "Try saying: "
                "what do you see, "
                "where is my water bottle, "
                "read this, "
                "or how much money am I holding."
            )
            self.speech.speak(help_text)
            print(f"[VOICE] help/unknown: {command_text}")
            return

    def run(self) -> None:
        # show the first frame to open the window (if debug window is enabled)
        frame = self.camera.capture_frame()
        annotated_frame = frame.copy() if frame is not None else None

        # start voice listener
        if self.voice is not None:
            self.voice.start_background()

        print(f'Say "{config.WAKE_PHRASE}" to wake me up. Then say a command.')
        print("Example commands:")
        print('- "what do you see"')
        print('- "where is my water bottle"')
        print('- "read this"')
        print('- "how much money am I holding"')
        print("Press Ctrl+C to exit.\n")

        while True:
            # Keep the camera window responsive (optional)
            if config.SHOW_DEBUG_WINDOW:
                try:
                    live = self.camera.capture_frame()
                    if live is not None:
                        annotated_frame = live
                    if annotated_frame is not None:
                        self.camera.show_image(annotated_frame)
                    # Pump window events
                    self.camera.wait_key_press("~", delay=1)
                except Exception:
                    # If running headless (Pi without GUI), ignore window errors
                    pass

            # Voice loop (always-on)
            if self.voice is not None:
                # drive recognition
                self.voice.tick()

                event = self.voice.poll_event()
                if event:
                    if event["type"] == "wake":
                        self.speech.speak("I'm listening.")
                    elif event["type"] == "command":
                        self._process_command(event["text"])
