"""
OCR Engine using EasyOCR with background threading
Created by Eric Leon - Integrated two-stage YOLO→EasyOCR pipeline
"""

import time
import threading
import queue
from typing import List, Dict, Optional

import cv2
import numpy as np
import easyocr

from src.utils import preprocessing
import src.utils.config as config


def _get_reader(gpu: bool = False):
    """Create or return a cached EasyOCR reader (singleton pattern)."""
    if not hasattr(_get_reader, "reader"):
        _get_reader.reader = easyocr.Reader(["en"], gpu=gpu)
    return _get_reader.reader


def _preprocess_crop(crop: np.ndarray, fast: bool = True) -> np.ndarray:
    """Preprocess crop for OCR using existing helpers.
    
    Returns a single-channel (grayscale) image suitable for EasyOCR.
    """
    # sharpen -> gray
    try:
        sharp = preprocessing.sharpen_image(crop)
    except Exception:
        sharp = crop

    gray = preprocessing.bgr_to_gray(sharp)

    # optional denoise (skip in fast mode)
    if fast or not config.USE_ADAPTIVE_PREPROCESSING:
        denoised = gray
    else:
        denoised = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)

    return denoised


class OCREngine:
    """OCR Engine with background threading and caching for real-time performance."""
    
    def __init__(self, use_gpu: bool = False, cache_ttl: float = 5.0, queue_size: int = 6):
        """Initialize OCR engine with background worker.
        
        Args:
            use_gpu: Whether to use GPU for EasyOCR (if available)
            cache_ttl: Cache time-to-live in seconds
            queue_size: Maximum size of OCR processing queue
        """
        self.use_gpu = use_gpu
        self.cache_ttl = cache_ttl
        
        # Initialize reader in background to avoid blocking
        self.reader = None
        self.reader_ready = threading.Event()
        self.reader_init_thread = threading.Thread(target=self._init_reader, daemon=True)
        self.reader_init_thread.start()
        
        # Background OCR worker queue and cache
        self.ocr_queue: queue.Queue = queue.Queue(maxsize=queue_size)
        self.cache_lock = threading.Lock()
        self.ocr_cache: Dict[str, Dict] = {}  # key -> {results: [...], ts: float}
        self.stop_event = threading.Event()
        
        # Start background worker thread (will wait for reader to be ready)
        self.worker = threading.Thread(target=self._ocr_worker, daemon=True)
        self.worker.start()
        
        print("✅ OCR Engine initializing (loading models in background)...")
    
    def _init_reader(self):
        """Initialize EasyOCR reader in background thread."""
        try:
            print("⏳ Loading EasyOCR models (this may take 10-30 seconds)...")
            self.reader = _get_reader(gpu=self.use_gpu)
            self.reader_ready.set()
            print("✅ OCR Engine ready!")
        except Exception as e:
            print(f"❌ OCR initialization failed: {e}")
            self.reader_ready.set()  # Set anyway to unblock waiters
    
    def _iou(self, boxA: tuple, boxB: tuple) -> float:
        """Calculate Intersection over Union for two bounding boxes."""
        # boxes are (x1,y1,x2,y2)
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])
        interW = max(0, xB - xA)
        interH = max(0, yB - yA)
        interArea = interW * interH
        boxAArea = max(0, boxA[2] - boxA[0]) * max(0, boxA[3] - boxA[1])
        boxBArea = max(0, boxB[2] - boxB[0]) * max(0, boxB[3] - boxB[1])
        union = boxAArea + boxBArea - interArea
        return interArea / union if union > 0 else 0.0
    
    def _get_cache_key(self, detection: Dict) -> str:
        """Generate cache key for a detection, preferring track_id."""
        track_id = detection.get("track_id")
        if track_id is not None:
            return f"id_{track_id}"
        
        # Fallback to bbox-based key
        bbox = detection.get("bbox")
        if bbox:
            x1, y1, x2, y2 = map(int, bbox)
            return f"{x1}-{y1}-{x2}-{y2}"
        
        return None
    
    def _find_cache_key_by_iou(self, detection: Dict, threshold: float = 0.5) -> Optional[str]:
        """Find cache key for detection using IoU matching for objects without track_id."""
        bbox = detection.get("bbox")
        if not bbox:
            return None
        
        x1, y1, x2, y2 = map(int, bbox)
        current_box = (x1, y1, x2, y2)
        
        best_match = (None, 0.0)
        
        with self.cache_lock:
            for key in self.ocr_cache.keys():
                # Skip track_id-based keys
                if key.startswith("id_"):
                    continue
                
                try:
                    parts = key.split("-")
                    cached_box = tuple(map(int, parts))
                except Exception:
                    continue
                
                iou_score = self._iou(current_box, cached_box)
                if iou_score > best_match[1]:
                    best_match = (key, iou_score)
        
        if best_match[0] is not None and best_match[1] > threshold:
            return best_match[0]
        
        return None
    
    def _ocr_worker(self):
        """Background worker that processes OCR tasks from the queue."""
        # Wait for reader to be initialized
        self.reader_ready.wait()
        
        if self.reader is None:
            print("⚠️ OCR worker exiting - reader initialization failed")
            return
        
        while not self.stop_event.is_set():
            try:
                key, crop = self.ocr_queue.get(timeout=0.5)
            except queue.Empty:
                continue
            
            try:
                # Preprocess crop
                preprocessed = _preprocess_crop(crop, fast=not config.USE_ADAPTIVE_PREPROCESSING)
                
                # Run EasyOCR
                raw_results = self.reader.readtext(preprocessed)
                
                # Parse and filter results
                parsed_results = []
                for item in raw_results:
                    try:
                        _, text, conf = item
                    except Exception:
                        continue
                    
                    if conf >= config.DEFAULT_OCR_CONFIDENCE_THRESHOLD:
                        parsed_results.append({"text": text, "confidence": float(conf)})
                
                # Update cache
                with self.cache_lock:
                    self.ocr_cache[key] = {"results": parsed_results, "ts": time.time()}
                
            except Exception as e:
                # Store empty results on error
                with self.cache_lock:
                    self.ocr_cache[key] = {"results": [], "ts": time.time()}
            
            finally:
                try:
                    self.ocr_queue.task_done()
                except Exception:
                    pass
    
    def read_text_from_detections(
        self,
        frame: np.ndarray,
        detections: List[Dict],
        frame_idx: int,
        ocr_every_n_frames: int = 5,
        max_crops: int = 4,
        min_box_dim: int = 20,
    ) -> List[Dict]:
        """Process detections and return OCR results for all detected objects.
        
        Args:
            frame: Current video frame
            detections: List of YOLO detections from ObjectDetector
            frame_idx: Current frame index for periodic processing
            ocr_every_n_frames: Frequency to enqueue new OCR tasks
        
        Returns:
            List of detections with OCR results:
            [{"label": str, "confidence": float, "bbox": tuple, "ocr": [{"text": str, "confidence": float}]}]
        """
        # Check if reader is ready
        if not self.reader_ready.is_set():
            print("⏳ OCR still initializing, please wait...")
            return []
        
        if self.reader is None:
            print("❌ OCR not available")
            return []
        
        results = []
        do_enqueue = (frame_idx % ocr_every_n_frames) == 0

        # Process higher-confidence detections first and cap work per trigger
        ordered = sorted(detections, key=lambda d: float(d.get("confidence", 0.0)), reverse=True)
        if max_crops is not None and max_crops > 0:
            ordered = ordered[:max_crops]

        for det in ordered:
            # WHITELIST FILTERING REMOVED - Process all detected objects
            # Previously filtered by: config.OCR_WHITELIST
            
            bbox = det.get("bbox")
            if bbox is None:
                continue
            
            # Extract and validate bbox
            x1, y1, x2, y2 = map(int, bbox)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
            
            if x2 <= x1 or y2 <= y1:
                continue

            # Skip tiny boxes to avoid wasted OCR work
            if (x2 - x1) < min_box_dim or (y2 - y1) < min_box_dim:
                continue
            
            # Extract crop
            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                continue
            
            # Upscale small crops for better OCR
            if crop.shape[1] < 120:
                crop = cv2.resize(crop, (crop.shape[1] * 2, crop.shape[0] * 2), interpolation=cv2.INTER_CUBIC)
            
            # Get or find cache key
            cache_key = self._get_cache_key(det)
            if cache_key is None:
                cache_key = self._find_cache_key_by_iou(det)
            if cache_key is None:
                continue
            
            # Try IoU-based matching if no direct match
            mapped_key = cache_key
            with self.cache_lock:
                if cache_key not in self.ocr_cache:
                    iou_match = self._find_cache_key_by_iou(det)
                    if iou_match:
                        mapped_key = iou_match
            
            # Check if cache is stale
            is_stale = True
            with self.cache_lock:
                entry = self.ocr_cache.get(mapped_key)
                if entry and (time.time() - entry.get("ts", 0.0) < self.cache_ttl):
                    is_stale = False
            
            # Enqueue OCR task if needed
            if do_enqueue and is_stale:
                try:
                    # Non-blocking enqueue; copy crop to avoid mutation
                    self.ocr_queue.put_nowait((mapped_key, crop.copy()))
                except queue.Full:
                    pass  # Skip if queue is full
            
            # Get cached OCR results
            ocr_results = []
            with self.cache_lock:
                cached_entry = self.ocr_cache.get(mapped_key)
                if cached_entry:
                    ocr_results = cached_entry.get("results", [])
            
            # Build result
            result = {
                "label": det.get("label"),
                "confidence": float(det.get("confidence", 0.0)),
                "bbox": (x1, y1, x2, y2),
                "ocr": ocr_results
            }
            results.append(result)
        
        return results
    
    def read_text(self, frame: np.ndarray) -> str:
        """Simple synchronous OCR for backward compatibility.
        
        Args:
            frame: Image frame to perform OCR on
        
        Returns:
            Concatenated text from all detected text regions
        """
        try:
            preprocessed = _preprocess_crop(frame, fast=True)
            raw_results = self.reader.readtext(preprocessed)
            
            texts = []
            for item in raw_results:
                try:
                    _, text, conf = item
                    if conf >= config.DEFAULT_OCR_CONFIDENCE_THRESHOLD:
                        texts.append(text)
                except Exception:
                    continue
            
            return " ".join(texts) if texts else ""
        
        except Exception:
            return ""
    
    def cleanup_stale_cache(self, max_age: float = None) -> int:
        """Remove stale entries from cache.
        
        Args:
            max_age: Maximum age in seconds (defaults to 2x cache_ttl)
        
        Returns:
            Number of entries cleaned
        """
        if max_age is None:
            max_age = self.cache_ttl * 2
        
        current_time = time.time()
        
        with self.cache_lock:
            stale_keys = [
                key for key, entry in self.ocr_cache.items()
                if current_time - entry.get("ts", 0.0) > max_age
            ]
            for key in stale_keys:
                del self.ocr_cache[key]
        
        if stale_keys:
            print(f"🧹 Cleaned {len(stale_keys)} stale OCR cache entries")
        
        return len(stale_keys)
    
    def shutdown(self):
        """Gracefully shutdown the OCR engine and background worker."""
        print("🛑 Shutting down OCR engine...")
        
        # Signal worker to stop
        self.stop_event.set()
        
        # Drain queue
        try:
            while not self.ocr_queue.empty():
                try:
                    self.ocr_queue.get_nowait()
                    self.ocr_queue.task_done()
                except Exception:
                    pass
        except Exception:
            pass
        
        # Wait for worker to finish
        if self.worker.is_alive():
            self.worker.join(timeout=1.0)
        
        # Clear cache
        with self.cache_lock:
            self.ocr_cache.clear()
        
        print("✅ OCR engine shut down successfully")
    
    def __del__(self):
        """Cleanup on deletion."""
        try:
            self.shutdown()
        except Exception:
            pass