"""
Made by Eric Leon

OCR Engine: is meant to use EasyOCR to extract text 
 from object crops within the taken frames.

Notes:
- OCR results are stored as dictionaries to group related metadata:
- bounding box coordinates, detected text, and confidence score.
- NumPy arrays (np.ndarray) are used to represent image pixel data.
- Images and object crops are passed as arrays of pixel values (H x W x C),
   which EasyOCR consumes directly for text recognition.

"""
from __future__ import annotations
from typing import List, Dict, Any
import easyocr
import cv2 as cv # plan to use later for preprocessing..?
import numpy as np

DEFAULT_MIN_CONFIDENCE: float = 0.45

class OCREngine:
############################################################################################
    def __init__(self, languages: List[str] = None, gpu: bool = False):
        self.languages = languages or ["en"]
        self.gpu = gpu
        self.easyOCR_reader = self._create_reader()
############################################################################################
    def _create_reader(self):
        return easyocr.Reader(self.languages, gpu=self.gpu)
############################################################################################
    def _annotate_confidence(self, filtered_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Annotates and formats a summary of the confidence values that EasyOCR produced.
        
        Returns a dict with:
        - avg_conf: average confidence (primary trigger for low-confidence warnings)
        - min_conf: minimum confidence (secondary diagnostic signal)
        - count: number of detections that passed the min_conf filter
        
        If no detections, returns avg_conf=0.0, min_conf=0.0, count=0.
        """
        if not filtered_results:
            return {
                "avg_conf": 0.0,
                "min_conf": 0.0,
                "count": 0
            }
        
        confidences = [r["confidence"] for r in filtered_results]
        return {
            "avg_conf": sum(confidences) / len(confidences),
            "min_conf": min(confidences),
            "count": len(confidences)
        }
############################################################################################
    def _filter_and_sort_results(
        self,
        results: List[Dict[str, Any]],
        min_conf: float
    ) -> List[Dict[str, Any]]:
        """
        Filters OCR results by confidence and sorts them in reading order
        (top-to-bottom, left-to-right).
        """
        filtered = [r for r in results if r["confidence"] >= min_conf]
        if not filtered:
            return []

        def reading_order_key(r):
            bbox = r["bbox"]
            cy = sum(p[1] for p in bbox) / 4
            cx = sum(p[0] for p in bbox) / 4
            return (cy, cx)

        filtered.sort(key=reading_order_key)
        return filtered
############################################################################################
    def _join_text(self, results: List[Dict[str, Any]]) -> str:
        """
        Joins sorted OCR results into a single readable string.
        """
        return " ".join(r["text"] for r in results)
############################################################################################
    def _extract_text(self, image: np.ndarray) -> List[Dict[str, Any]]:
        # `image` is a NumPy array containing raw pixel data.
        # This array represents the image or object crop being analyzed by EasyOCR.
        raw = self.easyOCR_reader.readtext(image, detail=1)
        # detail=1: return full details (bbox, text, confidence) for each detection
        results = []
        # x and y coordinate pairs from the bounding box. 
        # EasyOCR's box is a list of 4 points (the corners of a rectangle):
        for box, txt, conf in raw:
            bbox = [(float(x), float(y)) for (x, y) in box]
            results.append({
                "bbox": bbox,
                "text": str(txt),
                "confidence": float(conf),
            })
        return results
############################################################################################
    def extract_text_as_string(self, image: np.ndarray, min_conf: float = DEFAULT_MIN_CONFIDENCE) -> str:
        """
        Extracts all text from image and returns a single readable string.
        Filters by minimum confidence and sorts in reading order (top to bottom, left to right).
        """
        results = self._extract_text(image)
        filtered = self._filter_and_sort_results(results, min_conf)
        if not filtered:
            return ""
        return self._join_text(filtered)
############################################################################################
    def extract_text_with_confidence(self, image: np.ndarray, min_conf: float = DEFAULT_MIN_CONFIDENCE) -> Dict[str, Any]:
        """
        Extracts all text from image and returns a dict with text and confidence metrics.
        
        Returns a dict with:
        - text: the extracted text string (sorted in reading order, filtered by min_conf)
        - avg_conf: average confidence of filtered detections (primary low-confidence trigger)
        - min_conf: minimum confidence of filtered detections (secondary diagnostic signal)
        - count: number of detections that passed the min_conf threshold
        
        The calling layer (e.g., controller, speech engine) uses avg_conf to decide
        whether to display/speak a "low confidence" warning.
        """
        results = self._extract_text(image)
        filtered = self._filter_and_sort_results(results, min_conf)
        
        # Get confidence metrics
        conf_metrics = self._annotate_confidence(filtered)
        
        # Extract and sort text
        text = self._join_text(filtered) if filtered else ""
        
        # Return structured result with confidence data
        return {
            "text": text,
            "avg_conf": conf_metrics["avg_conf"],
            "min_conf": conf_metrics["min_conf"],
            "count": conf_metrics["count"]
        }
############################################################################################
    def attach_crop_text_to_detected_objects(self, frame: np.ndarray, detections: List[Dict[str, Any]], min_conf: float = DEFAULT_MIN_CONFIDENCE) -> List[Dict[str, Any]]:
        """
        For each detected object's bounding-box crop, attaches the extracted
        text to the detection dictionary using the "ocr_text" field.
        """
        np_float_to_int = lambda x: int(x.item())

        for det in detections:
            coords = (np_float_to_int(det["bbox"][0]), np_float_to_int(det["bbox"][1]), # x1, y1
                      np_float_to_int(det["bbox"][2]), np_float_to_int(det["bbox"][3])) # x2, y2
            
            # grabs the crop from the frame using bbox coords.
            # OCR is run only on this cropped image, not the full frame.
            crop = frame[coords[1]:coords[3], coords[0]:coords[2]] 

            det["ocr_text"] = self.extract_text_as_string(crop, min_conf=min_conf)
            
        return detections
############################################################################################