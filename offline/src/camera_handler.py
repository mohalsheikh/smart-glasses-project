"""
camera_handler.py

Robust camera handler for Raspberry Pi 5 CSI cameras (libcamera/PiCamera2),
with fallback to OpenCV VideoCapture (V4L2/USB).

Key goals:
- Prefer PiCamera2 (best for Pi CSI cameras, e.g. imx708).
- Avoid opening extra GUI windows unless explicitly requested.
- Work in headless environments (SSH) without HighGUI crashes.
"""

from __future__ import annotations

import os
import time
from typing import Optional, Tuple

import cv2 as cv

import src.utils.config as config


def _has_display() -> bool:
    """Return True if a GUI display is likely available."""
    return bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))


class CameraHandler:
    """
    CameraHandler supports two backends:
      1) "picamera2" (preferred): Raspberry Pi CSI camera via libcamera
      2) "opencv": cv2.VideoCapture fallback (USB cams or V4L2 nodes)

    By default, it will try Picamera2 first, then fall back to OpenCV.
    """

    def __init__(
        self,
        camera_index: int = getattr(config, "DEFAULT_CAMERA_INDEX", 0),
        frame_width: int = getattr(config, "DEFAULT_FRAME_WIDTH", 1280),
        frame_height: int = getattr(config, "DEFAULT_FRAME_HEIGHT", 720),
        prefer_libcamera: bool = True,
        libcamera_cam: int = 0,
        fps: int = 30,
        display: bool = False,
        window_name: str = "Camera",
        flip: int = -1,
    ):
        """
        Args:
            camera_index: OpenCV camera index (fallback path).
            frame_width/frame_height: Desired frame size.
            prefer_libcamera: Try PiCamera2 first if True.
            libcamera_cam: PiCamera2 camera index (0 for CAM0, 1 for CAM1).
            fps: Target FPS for capture.
            display: If True, allows GUI preview via imshow.
            window_name: Window name if display=True.
            flip: OpenCV flip code (None disables). Common:
                  0 vertical, 1 horizontal, -1 both. Default -1 (often needed).
        """
        # Validate inputs
        if camera_index is None or camera_index < 0:
            raise ValueError("camera_index must be a non-negative integer.")
        if frame_width is None or frame_width <= 0:
            raise ValueError("frame_width must be a positive integer.")
        if frame_height is None or frame_height <= 0:
            raise ValueError("frame_height must be a positive integer.")
        if fps is None or fps <= 0:
            raise ValueError("fps must be a positive integer.")

        self.frame_width = int(frame_width)
        self.frame_height = int(frame_height)
        self.fps = int(fps)

        self.display = bool(display) and _has_display()
        self.window_name = window_name
        self.flip = flip  # -1, 0, 1, or None

        self.backend: str = "unknown"
        self._running = False

        # Backend internals
        self.cap: Optional[cv.VideoCapture] = None
        self.picam2 = None  # type: ignore

        self._last_frame_time = 0.0

        # Try preferred backend first
        if prefer_libcamera:
            if self._try_init_picamera2(cam=libcamera_cam):
                self.backend = "picamera2"
                self._running = True
                self._setup_window_if_needed()
                return

        # Fallback to OpenCV VideoCapture
        if self._init_opencv(camera_index=camera_index):
            self.backend = "opencv"
            self._running = True
            self._setup_window_if_needed()
            return

        raise RuntimeError(
            "Could not initialize any camera backend.\n"
            "Tried: PiCamera2 (libcamera) and OpenCV VideoCapture.\n\n"
            "Fix tips:\n"
            " - For CSI cameras on Pi 5, install Picamera2: sudo apt install -y python3-picamera2\n"
            " - Ensure camera is detected: rpicam-hello --list-cameras\n"
            " - If using OpenCV, try a different index or a /dev/video* device.\n"
        )

    # ---------------------------
    # Initialization helpers
    # ---------------------------

    def _try_init_picamera2(self, cam: int) -> bool:
        """Try to initialize PiCamera2 backend. Returns True on success."""
        try:
            from picamera2 import Picamera2  # type: ignore
        except Exception:
            return False

        try:
            self.picam2 = Picamera2(camera_num=int(cam))

            # Create a config for video frames
            # format "XRGB8888" is common for OpenCV conversion; Picamera2 returns RGB array.
            video_config = self.picam2.create_video_configuration(
                main={
                    "size": (self.frame_width, self.frame_height),
                    "format": "XRGB8888",
                }
            )
            self.picam2.configure(video_config)

            # Start camera
            self.picam2.start()
            time.sleep(0.15)  # small warmup
            return True
        except Exception as e:
            # If anything fails, clean up and return False
            try:
                if self.picam2 is not None:
                    self.picam2.stop()
            except Exception:
                pass
            self.picam2 = None
            print(f"📷 CameraHandler: PiCamera2 init failed: {e}")
            return False

    def _init_opencv(self, camera_index: int) -> bool:
        """Initialize OpenCV VideoCapture backend. Returns True on success."""
        try:
            # CAP_V4L2 is more stable on Linux
            self.cap = cv.VideoCapture(int(camera_index), cv.CAP_V4L2)
        except Exception:
            self.cap = cv.VideoCapture(int(camera_index))

        if not self.cap or not self.cap.isOpened():
            self.cap = None
            return False

        # Configure capture props (may be ignored by some devices)
        self.cap.set(cv.CAP_PROP_FRAME_WIDTH, self.frame_width)
        self.cap.set(cv.CAP_PROP_FRAME_HEIGHT, self.frame_height)
        self.cap.set(cv.CAP_PROP_FPS, self.fps)

        return True

    def _setup_window_if_needed(self):
        """Create a window only if display mode is enabled."""
        if self.display:
            try:
                cv.namedWindow(self.window_name, cv.WINDOW_NORMAL)
            except Exception as e:
                # If GUI not available, disable display
                print(f"⚠️ CameraHandler: GUI window disabled (no display?): {e}")
                self.display = False

    # ---------------------------
    # Public API
    # ---------------------------

    def is_running(self) -> bool:
        return self._running

    def release(self):
        """Explicitly release camera resources."""
        self._running = False
        # Close OpenCV capture
        try:
            if self.cap is not None:
                self.cap.release()
        except Exception:
            pass
        self.cap = None

        # Stop PiCamera2
        try:
            if self.picam2 is not None:
                self.picam2.stop()
        except Exception:
            pass
        self.picam2 = None

        # Close windows only if we created any
        if self.display:
            try:
                cv.destroyWindow(self.window_name)
            except Exception:
                pass
            self.display = False

    def __del__(self):
        try:
            self.release()
        except Exception:
            pass

    def capture_frame(self) -> Optional["cv.Mat"]:
        """
        Capture a single frame and return as OpenCV BGR image (np array).
        Returns None if capture failed.
        """
        if not self._running:
            return None

        frame = None

        if self.backend == "picamera2" and self.picam2 is not None:
            try:
                # Picamera2 returns an RGB (or XRGB) array
                rgb = self.picam2.capture_array("main")
                # Convert to BGR for OpenCV pipeline
                frame = cv.cvtColor(rgb, cv.COLOR_RGB2BGR)
            except Exception as e:
                print(f"📷 CameraHandler: PiCamera2 capture failed: {e}")
                return None

        elif self.backend == "opencv" and self.cap is not None:
            ret, img = self.cap.read()
            if not ret:
                return None
            frame = img
        else:
            return None

        # Optional flip (useful for certain camera mount orientations)
        if frame is not None and self.flip is not None:
            try:
                frame = cv.flip(frame, int(self.flip))
            except Exception:
                pass

        return frame

    def show_image(self, image, window_name: Optional[str] = None):
        """
        Show image in a GUI window ONLY if display=True and display is available.
        This prevents the “second random window” problem by default.
        """
        if not self.display:
            return
        name = window_name or self.window_name
        try:
            cv.imshow(name, image)
        except Exception:
            # If it fails (Wayland/Qt plugin issues), disable display
            self.display = False

    def capture_and_show_frame(self, window_name: Optional[str] = None) -> Optional["cv.Mat"]:
        """
        Convenience: capture a frame and show it (if display=True).
        """
        frame = self.capture_frame()
        if frame is not None:
            self.show_image(frame, window_name=window_name)
        return frame

    def wait_key(self, delay_ms: int = 1) -> int:
        """
        Process GUI events if display=True.
        Returns key code or -1 if display is disabled.
        """
        if not self.display:
            return -1
        try:
            return cv.waitKey(delay_ms) & 0xFF
        except Exception:
            self.display = False
            return -1

    def get_backend(self) -> str:
        """Return which backend is being used: 'picamera2' or 'opencv'."""
        return self.backend
