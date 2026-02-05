from src.ocr_engine import OCREngine
import cv2 as cv

ocr_engine = OCREngine()

img = cv.imread("Test_img1.png")

print(ocr_engine.read_text(img))