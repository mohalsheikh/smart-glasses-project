
"""
Text-detection using a basic EasyOCR implementation
Created by Eric Leon
"""

import cv2
import easyocr
import matplotlib.pyplot as plt
from src.utils.config import DEFAULT_OCR_CONFIDENCE_THRESHOLD

# read image
image_path = "C:\\Repos\\CAPSTONE\\smart-glasses-project\\Test_img1.png"  # replace with your image path
image = cv2.imread(image_path)

# instance text detector
reader = easyocr.Reader(['en'], gpu=False)

# detect text on image
text_ = reader.readtext(image)

threshold = DEFAULT_OCR_CONFIDENCE_THRESHOLD
# draw bbox and text

for t in text_:
    bbox, text, score = t

    if score > threshold :
        cv2.rectangle(image, bbox[0], bbox[2], (0, 255, 0), 5)
        cv2.putText(image, text, bbox[0], cv2.FONT_HERSHEY_DUPLEX, 0.65, (255, 0, 0), 2)

    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.show()