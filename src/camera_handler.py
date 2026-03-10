"""
Handles camera input and preprocessing
Created by Ethan - Updated for Raspberry Pi libcamera support
"""

import cv2 as cv
import numpy as np
import platform
import subprocess
import tempfile
import os
from src.utils.config import DEFAULT_FRAME_WIDTH, DEFAULT_FRAME_HEIGHT

DEFAULT_CAMERA_INDEX: int = 0

class CameraHandler:
    def __init__(
            self,
            camera_index: int = DEFAULT_CAMERA_INDEX,
            frame_width: int = DEFAULT_FRAME_WIDTH,
            frame_height: int = DEFAULT_FRAME_HEIGHT
            ):

        if camera_index is None:
            raise ValueError("camera_index must be set.")
        if camera_index < 0:
            raise ValueError("camera_index must be a non-negative integer.")
        if frame_width is None:
            raise ValueError("frame_width must be set.")
        if frame_width <= 0:
            raise ValueError("frame_width must be a positive integer.")
        if frame_height is None:
            raise ValueError("frame_height must be set.")
        if frame_height <= 0:
            raise ValueError("frame_height must be a positive integer.")

        self._frame_width = frame_width
        self._frame_height = frame_height
        self.use_rpicam = False
        self.cap = None
        self._tmp_path = os.path.join(tempfile.gettempdir(), "smartglasses_frame.jpg")

        # Try rpicam-still on Linux (Raspberry Pi with libcamera)
        if platform.system() == "Linux":
            try:
                result = subprocess.run(
                    ["rpicam-still", "--list-cameras"],
                    capture_output=True, text=True, timeout=5
                )
                if result.returncode == 0 and "imx" in result.stdout.lower():
                    self.use_rpicam = True
                    print(f"[CameraHandler] Using rpicam-still (libcamera)")
                    return
            except (FileNotFoundError, subprocess.TimeoutExpired):
                pass

        # Fallback to OpenCV
        self.cap = cv.VideoCapture(camera_index)
        if not self.cap or not self.cap.isOpened():
            raise RuntimeError(f"Could not open camera {camera_index} with OpenCV.")

        ret, test_frame = self.cap.read()
        if not ret or test_frame is None:
            self.cap.release()
            raise RuntimeError(
                f"Camera {camera_index} opened but cannot read frames. "
                "It may be in use by another application or have driver issues. "
                "Please check that no other applications are using the camera.\n"
            )

        self.cap.set(cv.CAP_PROP_FRAME_WIDTH, frame_width)
        self.cap.set(cv.CAP_PROP_FRAME_HEIGHT, frame_height)
        print(f"[CameraHandler] Using OpenCV VideoCapture (index {camera_index})")

    def __del__(self):
        self.release()

    def release(self):
        try:
            if self.cap is not None:
                self.cap.release()
                self.cap = None
        except Exception:
            pass
        try:
            if os.path.exists(self._tmp_path):
                os.remove(self._tmp_path)
        except Exception:
            pass
        try:
            cv.destroyAllWindows()
        except Exception:
            pass

    def capture_frame(self):
        if self.use_rpicam:
            try:
                subprocess.run(
                    [
                        "rpicam-still",
                        "-o", self._tmp_path,
                        "--width", str(self._frame_width),
                        "--height", str(self._frame_height),
                        "--nopreview",
                        "--immediate",
                        "-t", "1",
                    ],
                    capture_output=True,
                    timeout=10
                )
                if os.path.exists(self._tmp_path):
                    frame = cv.imread(self._tmp_path)
                    return frame
            except (subprocess.TimeoutExpired, Exception) as e:
                print(f"[CameraHandler] rpicam-still capture failed: {e}")
                return None
        elif self.cap is not None:
            if not self.cap.isOpened():
                return None
            ret, frame = self.cap.read()
            if not ret:
                return None
            return frame
        return None

    def show_image(self, image, window_name="Camera"):
        cv.imshow(window_name, image)

    def capture_and_show_frame(self, window_name="Camera"):
        frame = self.capture_frame()
        if frame is not None:
            self.show_image(frame, window_name)
        return frame

    def wait_key_press(self, key: str, delay: int = 1):
        return cv.waitKey(delay) & 0xFF == ord(key)

    @property
    def frame_width(self) -> int:
        return self._frame_width

    @property
    def frame_height(self) -> int:
        return self._frame_height
