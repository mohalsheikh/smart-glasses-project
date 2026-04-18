"""
Body Tracker Overlay for VisionAssist
=======================================
Draws MediaPipe hand landmarks (21 points), face mesh, and pose skeleton
on every camera frame. Used in the main demo loop for visual effect.

Usage in demo_mode.py:
    from body_tracker import BodyTracker
    tracker = BodyTracker()
    
    # In the main loop:
    frame = tracker.draw(frame)  # draws landmarks on the frame
"""

import cv2 as cv
import numpy as np

_HAS_MEDIAPIPE = False
try:
    import mediapipe as mp
    _HAS_MEDIAPIPE = True
except ImportError:
    pass


class BodyTracker:
    """Draws hand, face, and pose landmarks on camera frames."""

    def __init__(self, draw_hands=True, draw_face=True, draw_pose=True):
        self._available = _HAS_MEDIAPIPE
        self._holistic = None
        self._draw_hands = draw_hands
        self._draw_face = draw_face
        self._draw_pose = draw_pose

        if self._available:
            self._holistic = mp.solutions.holistic.Holistic(
                static_image_mode=False,
                model_complexity=0,  # fastest
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
            )
            self._mp_drawing = mp.solutions.drawing_utils
            self._mp_hands = mp.solutions.hands
            self._mp_holistic = mp.solutions.holistic
            self._mp_face_mesh = mp.solutions.face_mesh

            # Custom styles — techy green/cyan look
            self._hand_dot = self._mp_drawing.DrawingSpec(
                color=(0, 255, 170), thickness=2, circle_radius=3)
            self._hand_line = self._mp_drawing.DrawingSpec(
                color=(0, 200, 255), thickness=2)
            self._face_dot = self._mp_drawing.DrawingSpec(
                color=(80, 255, 120), thickness=1, circle_radius=1)
            self._face_line = self._mp_drawing.DrawingSpec(
                color=(80, 200, 120), thickness=1)
            self._pose_dot = self._mp_drawing.DrawingSpec(
                color=(255, 100, 50), thickness=2, circle_radius=3)
            self._pose_line = self._mp_drawing.DrawingSpec(
                color=(255, 150, 80), thickness=2)

            print("🦴 Body tracker overlay ready (hands + face + pose)")
        else:
            print("⚠️ Body tracker not available (mediapipe not installed)")

    @property
    def available(self):
        return self._available

    def draw(self, frame: np.ndarray) -> np.ndarray:
        """
        Process frame and draw all landmarks on it.
        Returns the annotated frame. Fast enough for real-time (~15-20ms).
        """
        if not self._available or self._holistic is None:
            return frame

        rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        results = self._holistic.process(rgb)

        # Draw face mesh
        if self._draw_face and results.face_landmarks:
            self._mp_drawing.draw_landmarks(
                frame,
                results.face_landmarks,
                self._mp_face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=self._face_dot,
                connection_drawing_spec=self._face_line,
            )

        # Draw pose skeleton
        if self._draw_pose and results.pose_landmarks:
            self._mp_drawing.draw_landmarks(
                frame,
                results.pose_landmarks,
                self._mp_holistic.POSE_CONNECTIONS,
                landmark_drawing_spec=self._pose_dot,
                connection_drawing_spec=self._pose_line,
            )

        # Draw left hand
        if self._draw_hands and results.left_hand_landmarks:
            self._mp_drawing.draw_landmarks(
                frame,
                results.left_hand_landmarks,
                self._mp_hands.HAND_CONNECTIONS,
                landmark_drawing_spec=self._hand_dot,
                connection_drawing_spec=self._hand_line,
            )

        # Draw right hand
        if self._draw_hands and results.right_hand_landmarks:
            self._mp_drawing.draw_landmarks(
                frame,
                results.right_hand_landmarks,
                self._mp_hands.HAND_CONNECTIONS,
                landmark_drawing_spec=self._hand_dot,
                connection_drawing_spec=self._hand_line,
            )

        return frame

    def close(self):
        if self._holistic:
            self._holistic.close()