"""
OCR Enginee blah blah
"""

import easyocr
import cv2 as cv
import numpy as np

class OCREngine:
    def __init__(self):
        # initialize EasyOCR reader 
        self.easyOCR_reader = self._get_reader()

    """# given the list of detections from the object detector
    # reads text from each detected object and returns a list of the results. (why?? -Eric),
    # TODO create this function that, given the list of detections from object detector,
    # adds a "text" field to each detection with the extracted text from that object...
    # consider preforming some forms of preprocessing on each region like grayscale, dilate, deskew, etc. before passing to easyocr. """

    def read_objects(self, detections):
        pass

    # TODO look into gpu in case pi has a gpu?
    def _get_reader(self):
        return easyocr.Reader(['en'], gpu=False)
    
    def _extract_text(self, image: np.ndarray):
        # Unprocessed list of:
        # bounding box (bbox) coordinates (as np.int32), 
        # recognized text (e.g., "Eric's Notes"), 
        # and confidence scores (e.g., np.float64(0.86549)):
        raw = self.easyOCR_reader.readtext(image)

        results = []
        for box, txt, conf in raw:
            results.append({
                "bbox": [tuple(pt) for pt in box],
                "text": txt,
                "confidence": float(conf)
            })

        return results
    
    # new.. test to see if works
    def _debug_print_results(self, results, title="OCR results"):
        print(f"\n--- {title} ({len(results)}) ---")
        for i, r in enumerate(results[:10]):  # show first 10
            bbox = r["bbox"]
            print(f"{i:02d} conf={r['confidence']:.3f} text={r['text']!r} bbox0={bbox[0]}")
        if len(results) > 10:
            print(f"... (+{len(results)-10} more)")

    # new.. test to see if works
    def _filter_results(self, results, min_conf=0.45, min_len=2):
        filtered = []
        for r in results:
            t = (r["text"] or "").strip()
            if r["confidence"] < min_conf:
                continue
            if len(t) < min_len:
                continue
            filtered.append({**r, "text": t})
        return filtered

    # new.. test to see if works
    def _bbox_stats(self, bbox):
        xs = [p[0] for p in bbox]
        ys = [p[1] for p in bbox]
        x0, x1 = min(xs), max(xs)
        y0, y1 = min(ys), max(ys)
        w, h = (x1 - x0), (y1 - y0)
        cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
        return x0, y0, x1, y1, w, h, cx, cy

    def _merge_bboxes(self, bboxes):
        xs = [p[0] for b in bboxes for p in b]
        ys = [p[1] for b in bboxes for p in b]
        x0, y0, x1, y1 = min(xs), min(ys), max(xs), max(ys)
        return [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]

    # new.. test to see if works
    def _sort_reading_order(self, results):
        def key(r):
            x0, y0, x1, y1, w, h, cx, cy = self._bbox_stats(r["bbox"])
            return (cy, cx)
        return sorted(results, key=key)
    
    # new.. test to see if works
    def _as_single_block(self, results):
        if not results:
            return None
        text = " ".join(r["text"] for r in results)
        avg_conf = sum(r["confidence"] for r in results) / len(results)
        bbox = self._merge_bboxes([r["bbox"] for r in results])
        return {"text": text, "avg_confidence": avg_conf, "bbox": bbox}
    
    # new.. test to see if works
    def _group_by_vertical_gap(self, results, gap_multiplier=1.5):
        if not results:
            return []

        ordered = self._sort_reading_order(results)

        # estimate typical text height
        heights = []
        for r in ordered:
            *_, w, h, cx, cy = self._bbox_stats(r["bbox"])
            heights.append(h)
        median_h = sorted(heights)[len(heights)//2] if heights else 12
        y_gap_thresh = gap_multiplier * median_h

        blocks = []
        current = [ordered[0]]
        prev_cy = self._bbox_stats(ordered[0]["bbox"])[-1]

        for r in ordered[1:]:
            cy = self._bbox_stats(r["bbox"])[-1]
            if abs(cy - prev_cy) > y_gap_thresh:
                blocks.append(current)
                current = [r]
            else:
                current.append(r)
            prev_cy = cy

        blocks.append(current)
        return blocks


