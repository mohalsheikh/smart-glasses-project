"""
Handles camera input and preprocessing
"""

import cv2

class CameraHandler:
    def __init__(self, camera_index=0):
        self.cap = cv2.VideoCapture(camera_index)

    def capture_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            print("⚠️ Failed to capture frame")
            return None
        return frame

    def release(self):
        self.cap.release()
        cv2.destroyAllWindows()
