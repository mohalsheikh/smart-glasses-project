import cv2 as cv
from src.camera_handler import CameraHandler

camera_handler = CameraHandler()

while True:
    camera_handler.capture_and_show_frame(window_name="burger")

    # Press Q to quit window
    if cv.waitKey(1) & 0xFF == ord("q"):
        break