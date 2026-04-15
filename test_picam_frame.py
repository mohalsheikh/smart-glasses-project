from picamera2 import Picamera2
import time
import cv2 as cv

W,H = 1280,720
picam2 = Picamera2()
cfg = picam2.create_video_configuration(main={"size": (W,H), "format": "XRGB8888"})
picam2.configure(cfg)
picam2.start()
time.sleep(0.2)

frame = picam2.capture_array("main")
print("Captured:", frame.shape, frame.dtype)

# Save a JPEG so we can verify it worked
bgr = cv.cvtColor(frame, cv.COLOR_RGB2BGR)
cv.imwrite("picam_test.jpg", bgr)
print("Saved picam_test.jpg")
picam2.stop()
