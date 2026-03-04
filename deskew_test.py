from src.camera_handler import CameraHandler
from src.utils.preprocessing import deskew_image

cam = CameraHandler()

frame = cam.capture_and_show_frame(window_name="Original")
cam.wait_key_press("q", delay=10000)

for deskew_frame in deskew_image(frame):
    cam.show_image(deskew_frame, window_name="Deskewed")
    cam.wait_key_press("q", delay=10000)