"""
FAST Controller - optimized for real-time performance
Created by Mohammed
Speed-optimized: removed all slow features
"""

from collections import deque
from typing import List, Dict, Any, Optional
import time
import cv2 as cv
import numpy as np

from src.camera_handler import CameraHandler
from src.object_detector import ObjectDetector
from src.currency_recognizer import CurrencyRecognizer
from src.ocr_engine import OCREngine
from src.speech_engine import SpeechEngine
import src.utils.config as config


# Ignore noisy labels
IGNORE_LABELS = {
    "Clothing", "Human arm", "Human hair", "Human leg", "Human body",
    "Human head", "Human ear", "Human eye", "Human mouth", "Human nose",
    "Human hand", "Human foot", "Human face", "Fashion accessory"
}

# Merge similar labels
MERGE_LABELS = {
    "Human face": "person", "Man": "person", "Woman": "person",
    "Boy": "person", "Girl": "person", "Person": "person",
    "Laptop computer": "laptop", "Computer keyboard": "keyboard",
    "Computer mouse": "mouse", "Mobile phone": "phone",
    "Cellular telephone": "phone", "Telephone": "phone",
    "Television": "TV", "Drink": "beverage",
}

# Priority objects
PRIORITY_LABELS = {
    "person", "Door", "Door handle", "Stairs", "Chair", "Table",
    "Car", "Bus", "Truck", "Bicycle", "Motorcycle",
    "Traffic light", "Traffic sign", "Stop sign",
    "Laptop", "laptop", "phone", "Mug", "Bottle",
    "Toilet", "Sink", "Bed", "Couch"
}


def normalize_label(label: str) -> Optional[str]:
    """Normalize labels - fast version."""
    if label in IGNORE_LABELS:
        return None
    if label in MERGE_LABELS:
        return MERGE_LABELS[label]
    return label


def direction_from_center(center, frame_width: int) -> Optional[str]:
    """Get direction from center position."""
    if center is None or frame_width <= 0:
        return None
    
    x = center[0]
    left_thresh = frame_width / 3
    right_thresh = 2 * frame_width / 3
    
    if x < left_thresh:
        return "on your left"
    elif x > right_thresh:
        return "on your right"
    else:
        return "in front of you"


def add_indefinite_article(label: str) -> str:
    """Add a/an to label."""
    if not label:
        return label
    first_letter = label[0].lower()
    return f"an {label}" if first_letter in "aeiou" else f"a {label}"


def get_confidence_threshold(label: str) -> float:
    """Get confidence threshold based on object type."""
    if label in config.SMALL_OBJECTS:
        return config.CONFIDENCE_BY_CATEGORY["small_objects"]
    elif label in PRIORITY_LABELS:
        return config.CONFIDENCE_BY_CATEGORY["priority_objects"]
    else:
        return config.CONFIDENCE_BY_CATEGORY["general_objects"]


def summarize_detections(
    detections: List[Dict[str, Any]],
    frame_width: int,
    max_items: int = None,
) -> str:
    """
    Fast summary - no distance estimation.
    """
    if max_items is None:
        max_items = config.MAX_SPEECH_ITEMS
    
    # Filter by adaptive confidence
    filtered = []
    for d in detections:
        raw_label = d.get("label", "object")
        cleaned = normalize_label(raw_label)
        if cleaned is None:
            continue
        
        conf = float(d.get("confidence", 0.0))
        required_conf = get_confidence_threshold(cleaned)
        
        if conf >= required_conf:
            filtered.append({
                "label": cleaned,
                "confidence": conf,
                "center": d.get("center"),
            })
    
    if not filtered:
        return "I don't see any objects clearly."
    
    # Sort by confidence
    filtered.sort(key=lambda x: x["confidence"], reverse=True)
    
    # Priority first
    priority = [d for d in filtered if d["label"] in PRIORITY_LABELS]
    non_priority = [d for d in filtered if d["label"] not in PRIORITY_LABELS]
    
    combined = priority + non_priority
    combined = combined[:max_items]
    
    # Build phrases
    phrases = []
    for d in combined:
        label = d["label"]
        center = d["center"]
        direction = direction_from_center(center, frame_width)
        
        obj_phrase = add_indefinite_article(label)
        
        if direction:
            phrases.append(f"{obj_phrase} {direction}")
        else:
            phrases.append(obj_phrase)
    
    if not phrases:
        return "I don't see any objects clearly."
    
    # Natural sentence
    if len(phrases) == 1:
        return f"I see {phrases[0]}."
    elif len(phrases) == 2:
        return f"I see {phrases[0]} and {phrases[1]}."
    else:
        return f"I see {', '.join(phrases[:-1])}, and {phrases[-1]}."


class MainController:
    def __init__(self):
        self.camera = CameraHandler()
        self.detector = ObjectDetector()
        self.currency = CurrencyRecognizer()
        self.ocr = OCREngine()
        self.speech = SpeechEngine()
        
        # FPS tracking
        self.fps_queue = deque(maxlen=30)
        self.last_frame_time = time.time()
        
        # Last spoken to avoid repetition
        self.last_spoken_sentence = None
        
        # Cache last detections for skipped frames
        self.last_detections = []
        self.last_annotated = None
        
        print("⚡ FAST Smart Glasses System Initialized")
        print(f"📊 Model: {config.DEFAULT_MODEL_NAME}")
        print(f"🎯 Processing every {config.PROCESS_EVERY_N_FRAMES} frames")
    
    def calculate_fps(self) -> float:
        """Calculate FPS."""
        current_time = time.time()
        fps = 1.0 / (current_time - self.last_frame_time) if self.last_frame_time else 0
        self.last_frame_time = current_time
        self.fps_queue.append(fps)
        return sum(self.fps_queue) / len(self.fps_queue) if self.fps_queue else 0
    
    def run(self):
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
                
                # Only process every Nth frame for speed
                if frame_idx % config.PROCESS_EVERY_N_FRAMES == 0:
                    detections, annotated_frame = self.detector.detect(frame, annotate=True)
                    self.last_detections = detections
                    self.last_annotated = annotated_frame
                else:
                    # Use cached detections/frame
                    detections = self.last_detections
                    annotated_frame = self.last_annotated if self.last_annotated is not None else frame
                
                # Add FPS overlay
                if config.SHOW_FPS:
                    cv.putText(
                        annotated_frame, f"FPS: {fps:.1f}",
                        (10, 30), cv.FONT_HERSHEY_SIMPLEX,
                        0.7, (0, 255, 0), 2
                    )
                
                if config.SHOW_DETECTION_COUNT:
                    cv.putText(
                        annotated_frame, f"Objects: {len(detections)}",
                        (10, 60), cv.FONT_HERSHEY_SIMPLEX,
                        0.7, (0, 255, 0), 2
                    )
                
                # Show window
                if config.SHOW_DEBUG_WINDOW:
                    self.camera.show_image(
                        annotated_frame,
                        window_name="Smart Glasses - Fast Mode"
                    )
                
                # Quit on 'q'
                if cv.waitKey(1) & 0xFF == ord('q'):
                    print("👋 Exiting.")
                    break
                
                frame_idx += 1
                
                # Debug print (less frequent)
                if frame_idx % config.DEBUG_PRINT_EVERY_N_FRAMES == 0 and detections:
                    print(f"\n--- Frame {frame_idx} | FPS: {fps:.1f} | Objects: {len(detections)} ---")
                    for d in detections[:8]:
                        print(f"  {d.get('label'):20s} conf={d.get('confidence', 0):.2f}")
                
                # Speech (less frequent)
                if frame_idx % config.SPEAK_EVERY_N_FRAMES == 0 and detections:
                    objects_sentence = summarize_detections(
                        detections,
                        frame_width=frame_width,
                    )
                    
                    if objects_sentence != self.last_spoken_sentence:
                        # Simple message - no OCR/currency for speed
                        message = objects_sentence
                        
                        try:
                            self.speech.speak(message)
                        except Exception as e:
                            print(f"⚠️ Speech error: {e}")
                        
                        self.last_spoken_sentence = objects_sentence
        
        except KeyboardInterrupt:
            print("🛑 Interrupted.")
        
        finally:
            try:
                cv.destroyAllWindows()
            except:
                pass
            
            avg_fps = sum(self.fps_queue)/len(self.fps_queue) if self.fps_queue else 0
            print(f"\n📊 Final Stats:")
            print(f"   Average FPS: {avg_fps:.1f}")
            print(f"   Total frames: {frame_idx}")