"""
FAST Controller - optimized for real-time performance
Created by Mohammed
Speed-optimized: removed all slow features

This module is responsible for:
- Initializing hardware and core services
- Running the main processing loop
- Delegating description/summarization to helper utilities
"""

from collections import deque
import time

import cv2 as cv

from src.camera_handler import CameraHandler
from src.object_detector import ObjectDetector
from src.currency_recognizer import CurrencyRecognizer
from src.ocr_engine import OCREngine
from src.speech_engine import SpeechEngine
import src.utils.config as config
from src.utils.object_description import summarize_detections


class MainController:
    def __init__(self) -> None:
        # Core components
        self.camera = CameraHandler()
        self.detector = ObjectDetector()
        self.currency = CurrencyRecognizer()
        self.ocr = OCREngine()
        self.speech = SpeechEngine()

        # FPS tracking
        self.fps_queue: deque[float] = deque(maxlen=30)
        self.last_frame_time: float = time.time()

        # Last spoken sentence to avoid repetition
        self.last_spoken_sentence: str | None = None

        # Cache last detections for skipped frames
        self.last_detections = []
        self.last_annotated = None

    
        #---Eric
        # OCR state
        self.ocr_results_cache = []  # Store last OCR results
        self.ocr_cooldown_frames = 0  # Skip heavy detection while OCR just triggered
        #---

        print("⚡ FAST Smart Glasses System Initialized")
        print(f"📊 Model: {config.DEFAULT_MODEL_NAME}")
        print(f"🎯 Processing every {config.PROCESS_EVERY_N_FRAMES} frames")
        #---Eric
        print(f"🔤 Press 'R' to run OCR on detected objects")

    def _resize_frame_for_yolo(self, frame, max_side: int = 1280) -> tuple:
        """Resize frame so largest side <= max_side for faster YOLO inference.
        
        Returns (resized_frame, scale_factor).
        """
        h, w = frame.shape[:2]
        if max(h, w) > max_side:
            scale = max_side / max(h, w)
            new_w, new_h = int(w * scale), int(h * scale)
            resized = cv.resize(frame, (new_w, new_h), interpolation=cv.INTER_AREA)
            return resized, scale
        return frame, 1.0
    
    def _scale_detections_back(self, detections: list, scale: float) -> None:
        """Scale detection bboxes back to original frame size (modifies in-place)."""
        if scale >= 1.0:
            return
        for det in detections:
            if "bbox" in det:
                x1, y1, x2, y2 = det["bbox"]
                det["bbox"] = (int(x1 / scale), int(y1 / scale), int(x2 / scale), int(y2 / scale))
        #---

    def calculate_fps(self) -> float:
        """Calculate smoothed FPS using a moving average."""
        current_time = time.time()
        fps = 1.0 / (current_time - self.last_frame_time) if self.last_frame_time else 0.0
        self.last_frame_time = current_time
        self.fps_queue.append(fps)
        return sum(self.fps_queue) / len(self.fps_queue) if self.fps_queue else 0.0

    def run(self) -> None:
        """Main loop for the smart glasses system."""
        print("🚀 Smart Glasses System Starting...")
        print("Press 'q' to quit.")

        frame_idx = 0

        try:
            while True:
                frame = self.camera.capture_frame()
                if frame is None:
                    print("⚠️ No frame from camera, exiting.")
                    break

                frame_height, frame_width = frame.shape[:2]

                # Calculate FPS
                fps = self.calculate_fps()

                #---Eric edited*
                # Only run detection every Nth frame for speed (skip if in OCR cooldown)
                run_detection = (
                    self.ocr_cooldown_frames <= 0
                    and frame_idx % config.PROCESS_EVERY_N_FRAMES == 0
                )

                if run_detection:
                    # Resize frame for faster YOLO inference
                    yolo_frame, scale = self._resize_frame_for_yolo(frame, max_side=1280)
                    detections, annotated_frame = self.detector.detect(yolo_frame, annotate=True)
                    
                    # Scale bboxes back to original frame size
                    self._scale_detections_back(detections, scale)
                #---
                    self.last_detections = detections
                    self.last_annotated = annotated_frame
                else:
                    # Use cached detections/frame
                    detections = self.last_detections
                    annotated_frame = self.last_annotated if self.last_annotated is not None else frame

                # Overlays
                if config.SHOW_FPS:
                    cv.putText(
                        annotated_frame,
                        f"FPS: {fps:.1f}",
                        (10, 30),
                        cv.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 255, 0),
                        2,
                    )

                if config.SHOW_DETECTION_COUNT:
                    cv.putText(
                        annotated_frame,
                        f"Objects: {len(detections)}",
                        (10, 60),
                        cv.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 255, 0),
                        2,
                    )

                # Debug / visualization window
                if config.SHOW_DEBUG_WINDOW:
                    self.camera.show_image(
                        annotated_frame,
                        window_name="Smart Glasses - Fast Mode",
                    )

                # Keyboard input handling
                key_press = cv.waitKey(1) & 0xFF
                
                # Quit on 'q'
                if key_press == ord("q"):
                    print("👋 Exiting.")
                    break
                
                #---Eric
                # OCR on 'r' or 'R'
                if key_press == ord("r") or key_press == ord("R"):
                    print("\n🔤 OCR triggered! Running fresh detection and processing up to 4 objects...")
                    try:
                        # Force fresh YOLO detection for accurate OCR
                        yolo_frame, scale = self._resize_frame_for_yolo(frame, max_side=1280)
                        fresh_detections, _ = self.detector.detect(yolo_frame, annotate=False)
                        
                        # IMPORTANT: Pass RESIZED frame to OCR (not original)
                        # Smaller crops = much faster inference, prevents freezing
                        # Keep detections in resized space (don't scale back)
                        ocr_results = self.ocr.read_text_from_detections(
                            frame=yolo_frame,  # Use resized frame for faster OCR
                            detections=fresh_detections,  # Keep in resized space
                            frame_idx=frame_idx,
                            ocr_every_n_frames=1,  # Process immediately
                            max_crops=4,
                            min_box_dim=20,
                        )
                        
                        # Briefly pause new detections to reduce contention
                        self.ocr_cooldown_frames = 2

                        self.ocr_results_cache = ocr_results
                        
                        if ocr_results:
                            print(f"\n📝 OCR Results ({len(ocr_results)} objects):")
                            for i, result in enumerate(ocr_results, 1):
                                label = result.get("label", "unknown")
                                ocr_texts = result.get("ocr", [])
                                
                                if ocr_texts:
                                    print(f"  {i}. {label}:")
                                    for ocr_item in ocr_texts:
                                        text = ocr_item.get("text", "")
                                        conf = ocr_item.get("confidence", 0.0)
                                        print(f"     → '{text}' (conf: {conf:.2f})")
                                else:
                                    print(f"  {i}. {label}: [waiting for OCR processing...]")
                        else:
                            print(f"  ℹ️ No objects detected (total detections: {len(detections)})")
                            print("  ℹ️ OCR will process detected objects when available")
                    
                    except Exception as e:
                        print(f"  ⚠️ OCR error: {e}")

                # Decrement OCR cooldown if active
                if self.ocr_cooldown_frames > 0:
                    self.ocr_cooldown_frames -= 1
                #---

                frame_idx += 1

                # Periodic debug logging
                if frame_idx % config.DEBUG_PRINT_EVERY_N_FRAMES == 0 and detections:
                    print(
                        f"\n--- Frame {frame_idx} | FPS: {fps:.1f} "
                        f"| Objects: {len(detections)} ---"
                    )
                    for d in detections[:8]:
                        label = str(d.get("label", "object"))
                        conf = float(d.get("confidence", 0.0))
                        print(f"  {label:20s} conf={conf:.2f}")

                    #---Eric
                    # Periodic cache cleanup to prevent unbounded growth
                    self.ocr.cleanup_stale_cache()
                    #---

                # Periodic speech output
                if frame_idx % config.SPEAK_EVERY_N_FRAMES == 0 and detections:
                    objects_sentence = summarize_detections(
                        detections,
                        frame_width=frame_width,
                    )

                    if objects_sentence != self.last_spoken_sentence:
                        message = objects_sentence  # keep it simple for speed

                        try:
                            self.speech.speak(message)
                        except Exception as e:  # noqa: BLE001
                            print(f"⚠️ Speech error: {e}")

                        self.last_spoken_sentence = objects_sentence

        except KeyboardInterrupt:
            print("🛑 Interrupted by user.")

        finally:
            try:
                cv.destroyAllWindows()
            except Exception:  # noqa: BLE001
                pass
            #---Eric 
            # Cleanup OCR engine
            try:
                self.ocr.shutdown()
            except Exception as e:  # noqa: BLE001
                print(f"⚠️ OCR cleanup warning: {e}")
            #---

            avg_fps = sum(self.fps_queue) / len(self.fps_queue) if self.fps_queue else 0.0
            print("\n📊 Final Stats:")
            print(f"   Average FPS: {avg_fps:.1f}")
            print(f"   Total frames: {frame_idx}")
