"""
Handles camera input and basic resource management.
"""

from typing import Optional
import cv2
from src.utils.config import CAMERA_INDEX, FRAME_WIDTH, FRAME_HEIGHT


class CameraHandler:
    def __init__(self, camera_index: int = CAMERA_INDEX):
        self.cap: Optional[cv2.VideoCapture] = None
        self.index = camera_index
        self._open()

    def _open(self):
        self.cap = cv2.VideoCapture(self.index)
        if not self.cap or not self.cap.isOpened():
            print("⚠️ Failed to open camera. Check permissions in System Settings → Privacy & Security → Camera.")
            self.cap = None
            return
        # Set preferred resolution (best-effort)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

    def capture_frame(self):
        if self.cap is None:
            return None
        ok, frame = self.cap.read()
        if not ok:
            return None
        return frame

    def release(self):
        if self.cap is not None:
            try:
                self.cap.release()
            except Exception:
                pass
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass
