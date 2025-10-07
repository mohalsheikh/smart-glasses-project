import cv2 as cv
from src.camera_handler import CameraHandler
from src.object_detector import ObjectDetector

camera_handler = CameraHandler()
object_detector = ObjectDetector()

while True:
    camera_handler.capture_and_show_frame(window_name="burger")
    print(object_detector._track(frame=camera_handler.capture_frame())[0].boxes)

    # Press Q to quit window
    if cv.waitKey(1) & 0xFF == ord("q"):
        break