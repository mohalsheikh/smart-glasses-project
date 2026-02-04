"""
Currency Recognition Module
=============================================
Created by Mohammed - CBU Capstone Project

This module uses a custom-trained YOLOv8 model to detect and identify
US currency bills in real-time.

The model should be placed in: models/currency_detector.pt
"""

import cv2 as cv
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from collections import Counter
import time

# Try to import YOLO
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("[WARN] ultralytics not installed. Install with: pip install ultralytics")


class CurrencyRecognizer:
    """
    Recognizes US currency bills using a custom-trained YOLOv8 model.
    
    Features:
    - Real-time detection at 25+ FPS
    - Supports $1, $5, $10, $20, $50, $100 bills
    - Detection stabilization to avoid flickering
    - Confidence thresholding
    - Multiple bill detection and counting
    """
    
    # Mapping of class indices to denominations (adjust based on your training)
    DENOMINATION_MAP = {
        0: 1,      # one_dollar
        1: 5,      # five_dollar
        2: 10,     # ten_dollar
        3: 20,     # twenty_dollar
        4: 50,     # fifty_dollar
        5: 100,    # hundred_dollar
    }
    
    # Class names (should match your training data)
    CLASS_NAMES = {
        0: 'one_dollar',
        1: 'five_dollar', 
        2: 'ten_dollar',
        3: 'twenty_dollar',
        4: 'fifty_dollar',
        5: 'hundred_dollar',
    }
    
    def __init__(
        self,
        model_path: str = None,
        confidence_threshold: float = 0.5,
        use_gpu: bool = True,
        stabilization_frames: int = 3
    ):
        """
        Initialize the currency recognizer.
        
        Args:
            model_path: Path to the trained YOLOv8 model (.pt file)
            confidence_threshold: Minimum confidence for detections
            use_gpu: Whether to use GPU acceleration
            stabilization_frames: Number of consistent frames before announcing
        """
        self.confidence_threshold = confidence_threshold
        self.use_gpu = use_gpu
        self.stabilization_frames = stabilization_frames
        
        # Detection history for stabilization
        self._detection_history: List[Dict] = []
        self._last_announcement: str = ""
        self._last_announcement_time: float = 0
        self._announcement_cooldown: float = 2.0  # seconds
        
        # Find and load model
        self.model = None
        self.model_loaded = False
        
        if model_path is None:
            # Try common locations
            possible_paths = [
                Path(__file__).parent.parent / 'models' / 'currency_detector.pt',
                Path('models/currency_detector.pt'),
                Path('./currency_detector.pt'),
            ]
            for path in possible_paths:
                if path.exists():
                    model_path = str(path)
                    break
        
        if model_path and Path(model_path).exists():
            self._load_model(model_path)
        else:
            print(f"[WARN] Currency model not found. Detection disabled.")
            print(f"       Train and export a model to: models/currency_detector.pt")
    
    def _load_model(self, model_path: str):
        """Load the YOLO model."""
        if not YOLO_AVAILABLE:
            print("[ERROR] Cannot load model - ultralytics not installed")
            return
        
        try:
            print(f"[INFO] Loading currency model: {model_path}")
            self.model = YOLO(model_path)
            
            # Set device
            if self.use_gpu:
                import torch
                if torch.cuda.is_available():
                    self.model.to('cuda')
                    print("[INFO] Currency model using GPU")
                else:
                    print("[INFO] CUDA not available, using CPU")
            
            self.model_loaded = True
            
            # Update class names from model if available
            if hasattr(self.model, 'names'):
                self._update_class_mapping(self.model.names)
            
            print("[INFO] Currency model loaded successfully")
            
        except Exception as e:
            print(f"[ERROR] Failed to load currency model: {e}")
            self.model = None
            self.model_loaded = False
    
    def _update_class_mapping(self, model_names: dict):
        """Update class mapping based on model's class names."""
        # Try to match model class names to denominations
        denomination_keywords = {
            'one': 1, '1': 1, 'single': 1,
            'five': 5, '5': 5,
            'ten': 10, '10': 10,
            'twenty': 20, '20': 20,
            'fifty': 50, '50': 50,
            'hundred': 100, '100': 100,
        }
        
        new_map = {}
        for idx, name in model_names.items():
            name_lower = name.lower()
            for keyword, value in denomination_keywords.items():
                if keyword in name_lower:
                    new_map[idx] = value
                    break
        
        if new_map:
            self.DENOMINATION_MAP = new_map
            self.CLASS_NAMES = model_names
            print(f"[INFO] Updated denomination mapping: {new_map}")
    
    def _stabilize_detection(self, current_detection: Dict) -> Optional[Dict]:
        """
        Stabilize detections to avoid flickering announcements.
        
        Returns the detection only if it's been consistent for several frames.
        """
        self._detection_history.append(current_detection)
        
        # Keep limited history
        if len(self._detection_history) > self.stabilization_frames * 2:
            self._detection_history.pop(0)
        
        if len(self._detection_history) < self.stabilization_frames:
            return None
        
        # Check if recent detections are consistent
        recent = self._detection_history[-self.stabilization_frames:]
        
        # Get the most common total value
        totals = [d.get('total', 0) for d in recent if d.get('detected')]
        
        if not totals:
            return None
        
        # If majority agree on the total, return it
        counter = Counter(totals)
        most_common_total, count = counter.most_common(1)[0]
        
        if count >= self.stabilization_frames - 1 and most_common_total > 0:
            # Find a detection with this total to return
            for d in reversed(recent):
                if d.get('total') == most_common_total:
                    return d
        
        return None
    
    def _should_announce(self, message: str) -> bool:
        """Check if we should announce this detection (rate limiting)."""
        now = time.time()
        
        # Don't repeat the same message too quickly
        if message == self._last_announcement:
            if now - self._last_announcement_time < self._announcement_cooldown * 2:
                return False
        
        # General cooldown
        if now - self._last_announcement_time < self._announcement_cooldown:
            return False
        
        self._last_announcement = message
        self._last_announcement_time = now
        return True
    
    def recognize(self, frame: np.ndarray) -> str:
        """
        Recognize currency in the frame.
        
        This is the simple interface for backward compatibility.
        Returns a string message about detected currency.
        
        Args:
            frame: BGR image from camera
            
        Returns:
            String describing detected currency (e.g., "$20 bill detected")
        """
        result = self.recognize_detailed(frame)
        return result['message']
    
    def recognize_detailed(
        self,
        frame: np.ndarray,
        annotate: bool = False
    ) -> Dict[str, Any]:
        """
        Recognize currency with detailed output.
        
        Args:
            frame: BGR image from camera
            annotate: If True, include annotated frame in result
            
        Returns:
            Dictionary with:
                - detected: bool
                - bills: list of detected denominations
                - total: sum of all bills
                - message: human-readable string
                - boxes: list of bounding boxes
                - frame: annotated frame (if annotate=True)
        """
        result = {
            'detected': False,
            'bills': [],
            'total': 0,
            'message': "No currency detected",
            'boxes': [],
            'frame': frame if annotate else None,
            'should_announce': False
        }
        
        if frame is None or not self.model_loaded:
            return result
        
        try:
            # Run inference
            detections = self.model(
                frame,
                conf=self.confidence_threshold,
                verbose=False
            )[0]
            
            detected_bills = []
            boxes = []
            
            # Process detections
            for box in detections.boxes:
                cls_id = int(box.cls[0])
                confidence = float(box.conf[0])
                xyxy = box.xyxy[0].cpu().numpy()
                
                # Get denomination value
                denomination = self.DENOMINATION_MAP.get(cls_id, 0)
                
                if denomination > 0:
                    detected_bills.append(denomination)
                    boxes.append({
                        'denomination': denomination,
                        'confidence': confidence,
                        'bbox': tuple(xyxy),
                        'class_name': self.CLASS_NAMES.get(cls_id, f'class_{cls_id}')
                    })
            
            if detected_bills:
                result['detected'] = True
                result['bills'] = detected_bills
                result['total'] = sum(detected_bills)
                result['boxes'] = boxes
                
                # Create message
                if len(detected_bills) == 1:
                    result['message'] = f"${detected_bills[0]} bill detected"
                else:
                    # Group and count bills
                    bill_counts = Counter(detected_bills)
                    parts = []
                    for denom, count in sorted(bill_counts.items(), reverse=True):
                        if count == 1:
                            parts.append(f"${denom}")
                        else:
                            parts.append(f"{count} ${denom} bills")
                    
                    result['message'] = f"{', '.join(parts)} - Total ${result['total']}"
            
            # Draw annotations if requested
            if annotate and result['frame'] is not None:
                result['frame'] = self._annotate_frame(frame.copy(), boxes)
            
        except Exception as e:
            print(f"[ERROR] Currency detection failed: {e}")
            result['message'] = "Detection error"
        
        # Apply stabilization
        stable_result = self._stabilize_detection(result)
        
        if stable_result and stable_result.get('detected'):
            # Check if we should announce
            stable_result['should_announce'] = self._should_announce(stable_result['message'])
            return stable_result
        
        result['should_announce'] = False
        return result
    
    def _annotate_frame(
        self,
        frame: np.ndarray,
        boxes: List[Dict]
    ) -> np.ndarray:
        """Draw detection boxes and labels on the frame."""
        for box in boxes:
            x1, y1, x2, y2 = map(int, box['bbox'])
            denom = box['denomination']
            conf = box['confidence']
            
            # Draw box
            color = (0, 255, 0)  # Green
            cv.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # Draw label
            label = f"${denom} ({conf:.0%})"
            font = cv.FONT_HERSHEY_SIMPLEX
            font_scale = 0.7
            thickness = 2
            
            # Get text size for background
            (text_width, text_height), baseline = cv.getTextSize(
                label, font, font_scale, thickness
            )
            
            # Draw background rectangle
            cv.rectangle(
                frame,
                (x1, y1 - text_height - 10),
                (x1 + text_width + 10, y1),
                color,
                -1
            )
            
            # Draw text
            cv.putText(
                frame,
                label,
                (x1 + 5, y1 - 5),
                font,
                font_scale,
                (0, 0, 0),
                thickness
            )
        
        return frame
    
    def get_status(self) -> Dict[str, Any]:
        """Get the recognizer status."""
        return {
            'model_loaded': self.model_loaded,
            'use_gpu': self.use_gpu,
            'confidence_threshold': self.confidence_threshold,
            'class_names': self.CLASS_NAMES,
            'denomination_map': self.DENOMINATION_MAP,
        }


# Factory function for easy instantiation
def get_currency_recognizer(**kwargs) -> CurrencyRecognizer:
    """
    Factory function to create a CurrencyRecognizer.
    
    Args:
        **kwargs: Arguments passed to CurrencyRecognizer.__init__
        
    Returns:
        CurrencyRecognizer instance
    """
    return CurrencyRecognizer(**kwargs)


# For backward compatibility and testing
if __name__ == '__main__':
    import sys
    
    print("Testing Currency Recognizer...")
    
    recognizer = CurrencyRecognizer()
    print(f"\nStatus: {recognizer.get_status()}")
    
    if len(sys.argv) > 1:
        # Test on provided image
        image_path = sys.argv[1]
        frame = cv.imread(image_path)
        
        if frame is not None:
            result = recognizer.recognize_detailed(frame, annotate=True)
            print(f"\nResult: {result['message']}")
            print(f"Bills: {result['bills']}")
            print(f"Total: ${result['total']}")
            
            if result['frame'] is not None:
                cv.imwrite('detected_output.jpg', result['frame'])
                print("Saved annotated image to: detected_output.jpg")
        else:
            print(f"Could not load image: {image_path}")
    else:
        print("\nTo test on an image:")
        print("  python currency_recognizer.py path/to/image.jpg")
