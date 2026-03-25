import time

from src.camera_handler import CameraHandler
from src.utils.preprocessing import rgb_to_bgr, bgr_to_rgb

time.sleep(3)

try:
    cam = CameraHandler()
except Exception as e:
    print(f"Error initializing camera: {e}")
    exit(1)

frame = cam.capture_and_show_frame()
print(frame[0][0]) # print the values of the top-left pixel
cam.wait_key_press("q", delay=10000)

rgb_frame = bgr_to_rgb(frame)
print(rgb_frame[0][0]) # print the RGB values of the top-left pixel
cam.show_image(rgb_frame)

cam.wait_key_press("q", delay=10000)

bgr_frame = rgb_to_bgr(rgb_frame)
print(bgr_frame[0][0]) # print the BGR values of the top-left pixel
cam.show_image(bgr_frame)

cam.wait_key_press("q", delay=10000)