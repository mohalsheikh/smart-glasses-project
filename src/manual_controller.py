"""
Manual controller module created by Ethan. 
This activates our pipeline upon user input rather than periodically.
"""

from src.camera_handler import CameraHandler
from src.currency_recognizer import CurrencyRecognizer
from src.object_detector import ObjectDetector
from src.ocr_engine import OCREngine
from src.speech_engine import SpeechEngine
from src.voice_input import VoiceInput

import src.utils.config as config
from src.utils.object_description import summarize_detections, format_ocr_feedback

import threading
from queue import Queue
import enum


class VoiceInputState(enum.Enum):
    WAITING_FOR_WAKE_WORD = 1
    WAITING_FOR_COMMAND = 2


class MainController:
    def __init__(self) -> None:
        # Core components
        self.camera = CameraHandler()
        self.camera_frame_width = self.camera.frame_width

        self.detector = ObjectDetector(model_name="yolov8n.pt")
        self.ocr = OCREngine()
        self.speech = SpeechEngine()

        class_names_dict = self.detector.classes
        self.voice = VoiceInput(model_class_names=class_names_dict)
        self.commands = self.voice.commands

        self.class_names = list(class_names_dict.values())
        self.class_names.sort(key=lambda s: s.count(" "), reverse=True)

        self.partial_class_names = {word for name in self.class_names for word in name.split()}

        self.speech_queue = Queue()
        self.voice_input_result_q = Queue()
        self.voice_input_state_q = Queue()

        print("⚡ MANUAL Smart Glasses System Initialized")

    # =========================
    # TTS THREAD (FIXED)
    # =========================
    def _tts_worker(self):
        print("🔊 TTS worker started")

        while True:
            text = self.speech_queue.get()

            try:
                if text:
                    print(f"🗣️ TTS OUTPUT: {text}")
                    self.speech.speak(text)

            except Exception as e:
                print(f"❌ TTS ERROR: {e}")

    # =========================
    # VOICE INPUT THREAD
    # =========================
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
                    self.voice_input_result_q.put(
                        (VoiceInputState.WAITING_FOR_WAKE_WORD, wake_word_input)
                    )

                case VoiceInputState.WAITING_FOR_COMMAND:
                    command_input = self.voice.listen_command()
                    self.voice_input_result_q.put(
                        (VoiceInputState.WAITING_FOR_COMMAND, command_input)
                    )

    def _start_worker_threads(self):
        tts_thread = threading.Thread(target=self._tts_worker, daemon=True)
        voice_thread = threading.Thread(target=self._voice_input_worker, daemon=True)

        tts_thread.start()
        voice_thread.start()

    # =========================
    # ROUTE COMMAND (FIXED - THIS WAS MISSING)
    # =========================
    def _route_command(self, command: str, cleaned_transcript: str, frame, objs: list[str] = None):
        final_frame = None

        try:
            match command:

                case "detect":
                    detections, final_frame = self.detector.detect(
                        frame, annotate=True, objects=objs
                    )

                    description = summarize_detections(
                        detections,
                        frame_width=self.camera_frame_width
                    )

                case "read":
                    detections, final_frame = self.detector.detect(
                        frame, annotate=True, objects=objs
                    )

                    detections = self.ocr.attach_crop_text_to_detected_objects(
                        frame, detections
                    )

                    self._print_ocr_feedback(detections)

                    detections = [d for d in detections if d.get("ocr_text")]

                    description = summarize_detections(
                        detections,
                        frame_width=self.camera_frame_width
                    )

                case "sleep" | "end" | "nevermind" | "thanks":
                    description = "Going to sleep mode."
                    final_frame = frame.copy() if frame is not None else None
                    print("🛑 Vision system sleeping.")

                case _:
                    description = f"Sorry, I didn't understand '{cleaned_transcript}'."
                    final_frame = frame.copy() if frame is not None else None

        except Exception as e:
            print(f"❌ Command routing error: {e}")
            description = "Something went wrong processing the command."
            final_frame = frame.copy() if frame is not None else None

        return description, final_frame

    # =========================
    # REMOVE UNK TOKEN
    # =========================
    def _remove_unk(self, s: str) -> str:
        return s.replace("[unk]", "").strip()

    # =========================
    # OCR DEBUG OUTPUT
    # =========================
    def _print_ocr_feedback(self, detections):
        try:
            print("\nℹ️ Per-object OCR results")
            print("=" * 50)

            for i, det in enumerate(detections):
                label = det.get("label")
                confidence = det.get("confidence")
                bbox = det.get("bbox")
                ocr_text = det.get("ocr_text")

                confidence_str = f"{float(confidence):.2f}" if confidence else "n/a"

                if bbox:
                    try:
                        x1, y1, x2, y2 = bbox
                        bbox_str = f"({int(x1)}, {int(y1)}, {int(x2)}, {int(y2)})"
                    except:
                        bbox_str = "n/a"
                else:
                    bbox_str = "n/a"

                print(f"ID [{i:02d}]")
                print(f" Label: {label}")
                print(f" Conf: {confidence_str}")
                print(f" BBox: {bbox_str}")
                print(f" OCR: {ocr_text if ocr_text else 'none'}")

            print("=" * 50)

        except Exception as e:
            print(f"[OCR ERROR] {e}")

    # =========================
    # MAIN LOOP
    # =========================
    def run(self) -> None:
        frame = self.camera.capture_and_show_frame()

        if frame is None:
            print("❌ Failed to capture camera frame.")
            self.speech.speak("Camera initialization failed.")
            return

        annotated_frame = frame.copy()

        self._start_worker_threads()

        # 🔥 TEST AUDIO ON START
        self.speech_queue.put("Smart glasses audio system started")

        while True:
            self.camera.show_image(annotated_frame)

            if self.camera.wait_key_press('q', delay=10):
                print("Exiting program.")
                break

            try:
                voice_input_result = self.voice_input_result_q.get_nowait()
            except Exception:
                continue

            state = voice_input_result[0]
            transcript = voice_input_result[1]

            match state:

                case VoiceInputState.WAITING_FOR_WAKE_WORD:
                    print(f"🎙️ Wake word input: '{transcript}'")

                    if "vision" in transcript:
                        self.speech_queue.put("I'm listening!")
                        self.voice_input_state_q.put(
                            VoiceInputState.WAITING_FOR_COMMAND
                        )

                case VoiceInputState.WAITING_FOR_COMMAND:
                    self.voice_input_state_q.put(
                        VoiceInputState.WAITING_FOR_WAKE_WORD
                    )

                    print(f"🎙️ Heard command: '{transcript}'")

                    if not transcript:
                        self.speech_queue.put("I didn't hear a command.")
                        continue

                    cleaned = self._remove_unk(transcript)

                    if not cleaned:
                        self.speech_queue.put("Sorry, I didn't catch that.")
                        continue

                    frame = self.camera.capture_frame()

                    split = cleaned.split()
                    last_word = split[-1]

                    description, annotated_frame = self._route_command(
                        last_word,
                        cleaned,
                        frame
                    )

                    self.speech_queue.put(description)
                    print(f"Frame processed: {description}")
