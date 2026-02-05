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

    # given the list of detections from the object detector,
    # reads text from each detected object and returns a list of the results.
    # TODO create this function that, given the list of detections from object detector,
    # adds a "text" field to each detection with the extracted text from that object...
    # consider preforming some forms of preprocessing on each region like grayscale, dilate, deskew, etc. before passing to easyocr.
    def read_objects(self, detections):
        pass

    # TODO look into gpu in case pi has a gpu?
    def _get_reader(self):
        return easyocr.Reader(['en'], gpu=False)
    
    # uses easyocr to extract text from an image.
    def _extract_text(self, image: np.ndarray):
        return self.easyOCR_reader.readtext(image)