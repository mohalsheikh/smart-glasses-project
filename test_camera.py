import cv2 as cv
from src.camera_handler import CameraHandler
from src.object_detector import ObjectDetector

camera_handler = CameraHandler()
object_detector = ObjectDetector()

while True:
    frame = camera_handler.capture_frame()
    _, annotated = object_detector.detect(frame, annotate=True)
    camera_handler.show_image(annotated)

    # Press Q to quit window
    if cv.waitKey(1) & 0xFF == ord("q"):
        break