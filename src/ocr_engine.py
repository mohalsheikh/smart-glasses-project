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
from src.utils import preprocessing
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
        # Convert BGR (OpenCV default) → RGB (what OCR expects)
        image_rgb = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        raw = self.easyOCR_reader.readtext(image_rgb, detail=1)
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
    # def extract_text_as_string(self, image: np.ndarray, min_conf: float = DEFAULT_MIN_CONFIDENCE) -> str:
    #     """
    #     Extracts all text from image and returns a single readable string.
    #     Filters by minimum confidence and sorts in reading order (top to bottom, left to right).
    #     Tries 4 deskew candidates and returns the best one.
    #     """
    #     best_text = ""
    #     best_score = -1.0  # higher is better
    #     preprocessed = preprocessing.gaussian_blur(image)
    #     preprocessed = preprocessing.enhance_contrast_clahe(preprocessed)
    #     preprocessed = preprocessing.sharpen_image(preprocessed)
    #     for i, candidate in enumerate(preprocessing.deskew_image(preprocessed), start=1):
    #         results = self._extract_text(candidate)
    #         filtered = self._filter_and_sort_results(results, min_conf)
    #         print(f"[OCR] deskew candidate {i}: raw={len(results)} filtered={len(filtered)}")
    #         if not filtered:
    #             continue
    #         # Uses confidence helper to score
    #         metrics = self._annotate_confidence(filtered)  # avg_conf, min_conf, count
    #         text = self._join_text(filtered)
    #         score = metrics["count"] + metrics["avg_conf"]
    #         print(f"[OCR] candidate {i} score={score:.3f} text: {text}")
    #         if score > best_score:
    #             best_score = score
    #             best_text = text
    #     if best_text:
    #         return best_text
    #     print("[OCR] no candidates produced filtered text")
    #     return ""

    def extract_text_as_string(self, image: np.ndarray, min_conf: float = DEFAULT_MIN_CONFIDENCE) -> str:
        """
        Extracts all text from image and returns a single readable string.
        Filters by minimum confidence and sorts in reading order (top to bottom, left to right).
        Tries 4 deskew candidates and returns the best one.
        """

        def _to_bgr(img):
            if len(img.shape) == 2:
                return cv.cvtColor(img, cv.COLOR_GRAY2BGR)
            return img.copy()

        def _label(img, text):
            out = _to_bgr(img)
            cv.putText(
                out,
                text,
                (10, 25),
                cv.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
                cv.LINE_AA
            )
            return out

        def _fit_in_cell(img, cell_w=420, cell_h=280, bg=(30, 30, 30)):
            img = _to_bgr(img)
            h, w = img.shape[:2]

            scale = min(cell_w / w, cell_h / h)
            new_w = max(1, int(w * scale))
            new_h = max(1, int(h * scale))

            resized = cv.resize(img, (new_w, new_h))

            canvas = np.full((cell_h, cell_w, 3), bg, dtype=np.uint8)

            x = (cell_w - new_w) // 2
            y = (cell_h - new_h) // 2
            canvas[y:y+new_h, x:x+new_w] = resized

            return canvas
        
        def _add_border(img, pad=10):
            return cv.copyMakeBorder(
                img,
                pad, pad, pad, pad,
                cv.BORDER_CONSTANT,
                value=(40, 40, 40)  # dark gray spacing
            )

        def _make_grid(images, cols=2):
            rows = []
            for i in range(0, len(images), cols):
                row = images[i:i+cols]
                rows.append(cv.hconcat(row))
            return cv.vconcat(rows)

        best_text = ""
        best_score = -1.0
        best_candidate = None

        # ===== preprocessing stages =====
        stage0 = image.copy()
        stage1 = preprocessing.gaussian_blur(stage0)
        stage2 = preprocessing.enhance_contrast_clahe(stage1)
        stage3 = preprocessing.sharpen_image(stage2)

        candidates = list(preprocessing.deskew_image(stage3))

        for i, candidate in enumerate(candidates, start=1):
            results = self._extract_text(candidate)
            filtered = self._filter_and_sort_results(results, min_conf)
            print(f"[OCR] deskew candidate {i}: raw={len(results)} filtered={len(filtered)}")

            if not filtered:
                continue

            metrics = self._annotate_confidence(filtered)
            text = self._join_text(filtered)
            score = metrics["count"] + metrics["avg_conf"]
            print(f"[OCR] candidate {i} score={score:.3f} text: {text}")

            if score > best_score:
                best_score = score
                best_text = text
                best_candidate = candidate.copy()

        # ===== build one debug collage =====
        top_row = [
            _add_border(_label(stage0, "0 Original")),
            _add_border(_label(stage1, "1 Blur")),
            _add_border(_label(stage2, "2 CLAHE")),
            _add_border(_label(stage3, "3 Sharpen")),
        ]

        top_row = [_fit_in_cell(img, cell_w=420, cell_h=280) for img in top_row]
        top_strip = _make_grid(top_row, cols=2)

        bottom_imgs = []
        for i, cand in enumerate(candidates, start=1):
            tag = f"Deskew {i}"
            if best_candidate is not None and np.array_equal(cand, best_candidate):
                tag += " (BEST)"
            bottom_imgs.append(_add_border(_label(cand, tag)))

        bottom_imgs = [_fit_in_cell(img, cell_w=420, cell_h=280) for img in bottom_imgs]

        if bottom_imgs:
            bottom_strip = _make_grid(bottom_imgs, cols=2)
            collage = cv.vconcat([top_strip, bottom_strip])
        else:
            collage = top_strip

        collage = cv.copyMakeBorder(
            collage, 20, 20, 20, 20,
            cv.BORDER_CONSTANT,
            value=(20, 20, 20)
        )

        cv.namedWindow("OCR Preprocessing Debug", cv.WINDOW_NORMAL)
        cv.resizeWindow("OCR Preprocessing Debug", 1600, 900)

        scale = 1.5
        h, w = collage.shape[:2]
        collage_resized = cv.resize(collage, (int(w * scale), int(h * scale)))

        cv.imshow("OCR Preprocessing Debug", collage_resized)
        cv.waitKey(1)

        if best_text:
            return best_text

        print("[OCR] no candidates produced filtered text")
        return ""

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