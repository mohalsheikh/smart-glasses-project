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
import cv2 as cv # plan to use later..?
import numpy as np


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
    def extract_text_as_string(self, image: np.ndarray, min_conf: float = 0.45) -> str:
        """
        Extracts all text from image and returns a single readable string.
        Filters by minimum confidence and sorts in reading order (top to bottom, left to right).
        """
        results = self._extract_text(image)
        # Filter by confidence
        filtered = [r for r in results if r["confidence"] >= min_conf]
        if not filtered:
            return ""
        # Sorts by reading order (top to bottom, left to right)
        # cy = center y coordinate, cx = center x coordinate
        # p = corner point in bbox
        def reading_order_key(r):
            bbox = r["bbox"]
            # Calculate center y and x
            cy = sum(p[1] for p in bbox) / 4
            cx = sum(p[0] for p in bbox) / 4
            return (cy, cx)
        filtered.sort(key=reading_order_key)
        # Join all text with spaces
        return " ".join(r["text"] for r in filtered)
############################################################################################
    def attach_crop_text_to_detected_objects(self, detections: List[Dict[str, Any]], min_conf: float = 0.45) -> List[Dict[str, Any]]:
        """
        Adds the formatted text to the detected object data 
        via a "ocr_text" field where the extracted text is stored.
        Expects each detection to have a "crop" (np.ndarray) field.
        """
        for det in detections:
            crop = det.get("crop")
            # `crop` is a NumPy array representing the pixel region of a detected object.
            # OCR is run only on this cropped image, not the full frame.
            if crop is not None and isinstance(crop, np.ndarray):
                det["ocr_text"] = self.extract_text_as_string(crop, min_conf=min_conf)
            else:
                det["ocr_text"] = ""
        return detections
############################################################################################