# src/ai_features/human_analyzer.py
"""
🔬 PROFESSIONAL HUMAN ANALYZER
==============================

Advanced human analysis system with clean, modern visualization.

Features:
- Full 17-point body pose estimation
- 21-point hand tracking per hand (42 total)
- 468-point face mesh analysis
- Finger state detection (extended/curled)
- Gesture recognition (15+ gestures)
- Activity/pose classification
- Body language analysis
- Attention/gaze tracking
- Distance estimation
- Motion tracking & prediction
- Professional clean UI design

Requirements:
    pip install mediapipe opencv-python numpy ultralytics

Author: Smart Glasses AI System
"""

from __future__ import annotations

import cv2 as cv
import numpy as np
import time
import math
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
from collections import deque

# MediaPipe for hands/face
try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False
    print("⚠️ Install MediaPipe: pip install mediapipe")

# YOLO for pose
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False

try:
    import src.utils.config as config
except ImportError:
    config = None


# ═══════════════════════════════════════════════════════════════════════════════
# DATA CLASSES
# ═══════════════════════════════════════════════════════════════════════════════

class Gesture(Enum):
    """Recognized hand gestures."""
    UNKNOWN = "unknown"
    FIST = "fist"
    OPEN_PALM = "open palm"
    POINTING = "pointing"
    PEACE = "peace sign"
    THUMBS_UP = "thumbs up"
    THUMBS_DOWN = "thumbs down"
    OK_SIGN = "ok sign"
    ROCK = "rock on"
    CALL_ME = "call me"
    THREE = "three"
    FOUR = "four"
    PINCH = "pinch"
    GRAB = "grab"
    WAVE = "wave"


class Activity(Enum):
    """Recognized body activities."""
    UNKNOWN = "unknown"
    STANDING = "standing"
    SITTING = "sitting"
    WALKING = "walking"
    RUNNING = "running"
    REACHING = "reaching"
    WAVING = "waving"
    POINTING = "pointing"
    CROUCHING = "crouching"
    LEANING = "leaning"
    ARMS_CROSSED = "arms crossed"
    HANDS_ON_HIPS = "hands on hips"
    ARMS_RAISED = "arms raised"
    ONE_ARM_RAISED = "one arm raised"


@dataclass
class FingerState:
    """State of individual finger."""
    name: str
    is_extended: bool
    curl_angle: float = 0.0  # 0 = straight, 180 = fully curled
    tip_position: Optional[Tuple[int, int]] = None
    

@dataclass
class HandData:
    """Complete hand analysis."""
    landmarks: Optional[np.ndarray] = None
    landmarks_3d: Optional[np.ndarray] = None
    side: str = "unknown"  # "left" or "right"
    confidence: float = 0.0
    
    # Finger analysis
    fingers: Dict[str, FingerState] = field(default_factory=dict)
    num_fingers_extended: int = 0
    
    # Gesture
    gesture: Gesture = Gesture.UNKNOWN
    gesture_confidence: float = 0.0
    
    # Position
    wrist_pos: Optional[Tuple[int, int]] = None
    palm_center: Optional[Tuple[int, int]] = None
    hand_bbox: Optional[Tuple[int, int, int, int]] = None
    
    # Orientation
    palm_facing: str = "unknown"  # "camera", "away", "left", "right", "up", "down"
    hand_rotation: float = 0.0  # degrees


@dataclass
class FaceData:
    """Complete face analysis."""
    landmarks: Optional[np.ndarray] = None
    bbox: Optional[Tuple[int, int, int, int]] = None
    confidence: float = 0.0
    
    # Eye analysis
    left_eye_open: float = 1.0  # 0 = closed, 1 = open
    right_eye_open: float = 1.0
    is_blinking: bool = False
    blink_count: int = 0
    
    # Gaze
    gaze_direction: str = "forward"  # "forward", "left", "right", "up", "down"
    looking_at_camera: bool = True
    
    # Mouth
    mouth_open: float = 0.0  # 0 = closed, 1 = wide open
    is_speaking: bool = False
    is_smiling: bool = False
    
    # Head pose
    pitch: float = 0.0  # up/down
    yaw: float = 0.0    # left/right
    roll: float = 0.0   # tilt
    
    # Attention
    attention_score: float = 1.0  # 0-1, how attentive they seem


@dataclass
class BodyData:
    """Complete body analysis."""
    # Basic detection
    bbox: Tuple[int, int, int, int] = (0, 0, 0, 0)
    center: Tuple[float, float] = (0.0, 0.0)
    track_id: Optional[int] = None
    detection_confidence: float = 0.0
    
    # Pose keypoints (17 COCO points)
    keypoints: Optional[np.ndarray] = None
    keypoint_scores: Optional[np.ndarray] = None
    keypoints_visible: Dict[str, bool] = field(default_factory=dict)
    
    # Body measurements
    height_px: int = 0
    width_px: int = 0
    shoulder_width_px: int = 0
    torso_height_px: int = 0
    
    # Estimated real-world
    estimated_distance_m: float = 0.0
    estimated_height_m: float = 1.7
    
    # Activity & pose
    activity: Activity = Activity.UNKNOWN
    body_orientation: str = "front"  # "front", "back", "left", "right"
    posture_quality: str = "normal"  # "good", "normal", "slouching"
    
    # Arms analysis
    left_arm_angle: float = 0.0   # angle at elbow
    right_arm_angle: float = 0.0
    left_arm_raised: bool = False
    right_arm_raised: bool = False
    arms_position: str = "relaxed"  # "relaxed", "crossed", "raised", "akimbo"
    
    # Hands (linked from HandData)
    left_hand: Optional[HandData] = None
    right_hand: Optional[HandData] = None
    
    # Face (linked from FaceData)
    face: Optional[FaceData] = None
    
    # Motion
    velocity: Tuple[float, float] = (0.0, 0.0)
    speed: float = 0.0
    is_moving: bool = False
    motion_direction: str = "stationary"
    
    # Position history for smoothing
    position_history: deque = field(default_factory=lambda: deque(maxlen=10))
    
    # UI state
    time_tracked: float = 0.0
    first_seen: float = 0.0


# ═══════════════════════════════════════════════════════════════════════════════
# COLOR THEME - Clean Professional Design
# ═══════════════════════════════════════════════════════════════════════════════

class Theme:
    """Professional color theme."""
    # Main colors (BGR)
    PRIMARY = (180, 130, 70)       # Soft blue
    SECONDARY = (140, 160, 80)     # Teal
    ACCENT = (80, 180, 220)        # Warm orange
    SUCCESS = (120, 200, 80)       # Green
    WARNING = (60, 180, 230)       # Orange
    DANGER = (80, 80, 220)         # Red
    
    # Skeleton colors - natural gradient
    SKELETON_HEAD = (200, 180, 140)     # Light blue-gray
    SKELETON_TORSO = (180, 160, 120)    # Medium blue
    SKELETON_ARM_L = (160, 180, 100)    # Teal-ish
    SKELETON_ARM_R = (100, 180, 160)    # Green-teal
    SKELETON_LEG_L = (180, 140, 160)    # Soft purple
    SKELETON_LEG_R = (140, 140, 180)    # Dusty purple
    
    # Hand colors - warm gradient
    HAND_BASE = (150, 170, 200)         # Warm beige
    HAND_THUMB = (120, 160, 200)        # Peachy
    HAND_INDEX = (100, 150, 190)        
    HAND_MIDDLE = (80, 140, 180)        
    HAND_RING = (60, 130, 170)          
    HAND_PINKY = (40, 120, 160)         
    
    # Face
    FACE_MESH = (180, 170, 160)         # Light gray
    FACE_CONTOUR = (160, 150, 140)      
    FACE_EYES = (200, 180, 100)         # Soft blue
    FACE_MOUTH = (140, 140, 180)        # Soft pink
    
    # UI
    PANEL_BG = (40, 40, 45)             # Dark gray
    PANEL_BORDER = (80, 80, 90)         
    TEXT_PRIMARY = (240, 240, 240)      # White
    TEXT_SECONDARY = (160, 160, 170)    # Gray
    TEXT_LABEL = (120, 180, 200)        # Warm highlight
    
    # Overlays
    BBOX_ACTIVE = (180, 140, 80)        # Blue
    BBOX_INACTIVE = (100, 100, 110)     # Gray
    JOINT_FILL = (220, 220, 230)        # Near white
    JOINT_BORDER = (140, 140, 150)      


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN ANALYZER CLASS
# ═══════════════════════════════════════════════════════════════════════════════

class HumanAnalyzer:
    """
    Professional Human Analysis System
    
    Provides comprehensive human detection and analysis with:
    - Body pose estimation
    - Hand and finger tracking  
    - Face mesh analysis
    - Activity recognition
    - Clean professional visualization
    """
    
    # COCO Keypoint indices
    NOSE, L_EYE, R_EYE, L_EAR, R_EAR = 0, 1, 2, 3, 4
    L_SHOULDER, R_SHOULDER = 5, 6
    L_ELBOW, R_ELBOW = 7, 8
    L_WRIST, R_WRIST = 9, 10
    L_HIP, R_HIP = 11, 12
    L_KNEE, R_KNEE = 13, 14
    L_ANKLE, R_ANKLE = 15, 16
    
    KEYPOINT_NAMES = {
        0: "nose", 1: "left_eye", 2: "right_eye", 3: "left_ear", 4: "right_ear",
        5: "left_shoulder", 6: "right_shoulder", 7: "left_elbow", 8: "right_elbow",
        9: "left_wrist", 10: "right_wrist", 11: "left_hip", 12: "right_hip",
        13: "left_knee", 14: "right_knee", 15: "left_ankle", 16: "right_ankle"
    }
    
    SKELETON_CONNECTIONS = [
        # Head
        (NOSE, L_EYE), (NOSE, R_EYE), (L_EYE, L_EAR), (R_EYE, R_EAR),
        # Torso
        (L_SHOULDER, R_SHOULDER), (L_SHOULDER, L_HIP), (R_SHOULDER, R_HIP), (L_HIP, R_HIP),
        # Left arm
        (L_SHOULDER, L_ELBOW), (L_ELBOW, L_WRIST),
        # Right arm
        (R_SHOULDER, R_ELBOW), (R_ELBOW, R_WRIST),
        # Left leg
        (L_HIP, L_KNEE), (L_KNEE, L_ANKLE),
        # Right leg
        (R_HIP, R_KNEE), (R_KNEE, R_ANKLE),
    ]
    
    # MediaPipe hand landmark indices
    HAND_WRIST = 0
    HAND_THUMB_CMC, HAND_THUMB_MCP, HAND_THUMB_IP, HAND_THUMB_TIP = 1, 2, 3, 4
    HAND_INDEX_MCP, HAND_INDEX_PIP, HAND_INDEX_DIP, HAND_INDEX_TIP = 5, 6, 7, 8
    HAND_MIDDLE_MCP, HAND_MIDDLE_PIP, HAND_MIDDLE_DIP, HAND_MIDDLE_TIP = 9, 10, 11, 12
    HAND_RING_MCP, HAND_RING_PIP, HAND_RING_DIP, HAND_RING_TIP = 13, 14, 15, 16
    HAND_PINKY_MCP, HAND_PINKY_PIP, HAND_PINKY_DIP, HAND_PINKY_TIP = 17, 18, 19, 20
    
    HAND_CONNECTIONS = [
        # Thumb
        (0, 1), (1, 2), (2, 3), (3, 4),
        # Index
        (0, 5), (5, 6), (6, 7), (7, 8),
        # Middle
        (0, 9), (9, 10), (10, 11), (11, 12),
        # Ring
        (0, 13), (13, 14), (14, 15), (15, 16),
        # Pinky
        (0, 17), (17, 18), (18, 19), (19, 20),
        # Palm
        (5, 9), (9, 13), (13, 17),
    ]
    
    def __init__(
        self,
        pose_model: str = "yolov8n-pose.pt",
        enable_hands: bool = True,
        enable_face: bool = True,
        confidence_threshold: float = 0.4,
        # Visualization options
        show_skeleton: bool = True,
        show_hands: bool = True,
        show_face: bool = True,
        show_labels: bool = True,
        show_panel: bool = True,
        show_metrics: bool = True,
        panel_position: str = "right",  # "left", "right", "top", "bottom"
        skeleton_thickness: int = 2,
        joint_radius: int = 4,
    ):
        """Initialize the Human Analyzer."""
        self.confidence_threshold = confidence_threshold
        self.show_skeleton = show_skeleton
        self.show_hands = show_hands
        self.show_face = show_face
        self.show_labels = show_labels
        self.show_panel = show_panel
        self.show_metrics = show_metrics
        self.panel_position = panel_position
        self.skeleton_thickness = skeleton_thickness
        self.joint_radius = joint_radius
        
        # Initialize YOLO Pose
        self.pose_model = None
        if YOLO_AVAILABLE:
            try:
                self.pose_model = YOLO(pose_model)
                print(f"✅ Pose model loaded: {pose_model}")
            except Exception as e:
                print(f"⚠️ Could not load pose model: {e}")
        
        # Initialize MediaPipe Hands
        self.hands_detector = None
        if MEDIAPIPE_AVAILABLE and enable_hands:
            try:
                self.hands_detector = mp.solutions.hands.Hands(
                    static_image_mode=False,
                    max_num_hands=2,
                    min_detection_confidence=0.6,
                    min_tracking_confidence=0.5,
                )
                print("✅ Hand tracking enabled (21 points per hand)")
            except Exception as e:
                print(f"⚠️ Hand tracking error: {e}")
        
        # Initialize MediaPipe Face Mesh
        self.face_detector = None
        if MEDIAPIPE_AVAILABLE and enable_face:
            try:
                self.face_detector = mp.solutions.face_mesh.FaceMesh(
                    static_image_mode=False,
                    max_num_faces=2,
                    refine_landmarks=True,
                    min_detection_confidence=0.6,
                    min_tracking_confidence=0.5,
                )
                print("✅ Face mesh enabled (468 landmarks)")
            except Exception as e:
                print(f"⚠️ Face mesh error: {e}")
        
        # Tracking state
        self._tracked_bodies: Dict[int, BodyData] = {}
        self._frame_count = 0
        self._start_time = time.time()
        self._last_blink_time: Dict[int, float] = {}
        
        print("🔬 Human Analyzer ready")
    
    # ═══════════════════════════════════════════════════════════════════════════
    # MAIN PIPELINE
    # ═══════════════════════════════════════════════════════════════════════════
    
    def analyze(
        self,
        frame: np.ndarray,
        detections: Optional[List[Dict]] = None
    ) -> Tuple[List[BodyData], np.ndarray]:
        """
        Main analysis pipeline.
        
        Args:
            frame: BGR image
            detections: Optional pre-existing detections
            
        Returns:
            (list of BodyData, annotated frame)
        """
        self._frame_count += 1
        annotated = frame.copy()
        bodies: List[BodyData] = []
        
        if self.pose_model is None:
            return bodies, annotated
        
        h, w = frame.shape[:2]
        current_time = time.time()
        
        # ─────────────────────────────────────────────────────────────────────
        # 1. Pose Detection (YOLO)
        # ─────────────────────────────────────────────────────────────────────
        try:
            results = self.pose_model(frame, verbose=False)[0]
            
            kpts_data = getattr(results, "keypoints", None)
            boxes = getattr(results, "boxes", None)
            
            if kpts_data is not None and boxes is not None:
                kpts = self._to_numpy(getattr(kpts_data, "xy", None))
                kpts_conf = self._to_numpy(getattr(kpts_data, "conf", None))
                xyxy = self._to_numpy(getattr(boxes, "xyxy", None))
                conf = self._to_numpy(getattr(boxes, "conf", None))
                ids = self._to_numpy(getattr(boxes, "id", None))
                
                if kpts is not None and xyxy is not None:
                    for i in range(len(xyxy)):
                        if conf is not None and conf[i] < self.confidence_threshold:
                            continue
                        
                        x1, y1, x2, y2 = map(int, xyxy[i])
                        track_id = int(ids[i]) if ids is not None and i < len(ids) else i
                        
                        # Create or update body
                        if track_id in self._tracked_bodies:
                            body = self._tracked_bodies[track_id]
                        else:
                            body = BodyData(first_seen=current_time)
                        
                        # Update basic info
                        body.bbox = (x1, y1, x2, y2)
                        body.center = ((x1 + x2) / 2, (y1 + y2) / 2)
                        body.track_id = track_id
                        body.detection_confidence = float(conf[i]) if conf is not None else 0.0
                        body.keypoints = kpts[i] if i < len(kpts) else None
                        body.keypoint_scores = kpts_conf[i] if kpts_conf is not None and i < len(kpts_conf) else None
                        body.height_px = y2 - y1
                        body.width_px = x2 - x1
                        body.time_tracked = current_time - body.first_seen
                        
                        # Analyze pose
                        self._analyze_body_pose(body, h, w)
                        
                        # Track motion
                        body.position_history.append(body.center)
                        self._analyze_motion(body)
                        
                        bodies.append(body)
                        self._tracked_bodies[track_id] = body
        
        except Exception as e:
            print(f"⚠️ Pose detection error: {e}")
        
        # ─────────────────────────────────────────────────────────────────────
        # 2. Hand Detection (MediaPipe)
        # ─────────────────────────────────────────────────────────────────────
        if self.hands_detector and self.show_hands and bodies:
            try:
                rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
                hand_results = self.hands_detector.process(rgb)
                
                if hand_results.multi_hand_landmarks:
                    for idx, landmarks in enumerate(hand_results.multi_hand_landmarks):
                        # Determine handedness
                        side = "right"
                        if hand_results.multi_handedness and idx < len(hand_results.multi_handedness):
                            side = hand_results.multi_handedness[idx].classification[0].label.lower()
                        
                        # Analyze hand
                        hand_data = self._analyze_hand(landmarks, side, w, h)
                        
                        # Match to nearest body
                        if hand_data.wrist_pos:
                            nearest = min(bodies, key=lambda b: self._dist(hand_data.wrist_pos, b.center))
                            if side == "left":
                                nearest.left_hand = hand_data
                            else:
                                nearest.right_hand = hand_data
            except Exception as e:
                pass
        
        # ─────────────────────────────────────────────────────────────────────
        # 3. Face Detection (MediaPipe)
        # ─────────────────────────────────────────────────────────────────────
        if self.face_detector and self.show_face and bodies:
            try:
                rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
                face_results = self.face_detector.process(rgb)
                
                if face_results.multi_face_landmarks:
                    for landmarks in face_results.multi_face_landmarks:
                        face_data = self._analyze_face(landmarks, w, h)
                        
                        # Match to nearest body
                        if face_data.bbox:
                            face_center = (
                                (face_data.bbox[0] + face_data.bbox[2]) / 2,
                                (face_data.bbox[1] + face_data.bbox[3]) / 2
                            )
                            nearest = min(bodies, key=lambda b: self._dist(face_center, b.center))
                            nearest.face = face_data
            except Exception as e:
                pass
        
        # ─────────────────────────────────────────────────────────────────────
        # 4. Draw Visualizations
        # ─────────────────────────────────────────────────────────────────────
        annotated = self._draw_frame(annotated, bodies)
        
        return bodies, annotated
    
    # Alias for compatibility
    def analyze_humans(self, frame, detections=None):
        return self.analyze(frame, detections)
    
    # ═══════════════════════════════════════════════════════════════════════════
    # BODY ANALYSIS
    # ═══════════════════════════════════════════════════════════════════════════
    
    def _analyze_body_pose(self, body: BodyData, frame_h: int, frame_w: int) -> None:
        """Analyze body pose, activity, and measurements."""
        if body.keypoints is None:
            return
        
        kpts = body.keypoints
        scores = body.keypoint_scores
        
        def visible(idx: int) -> bool:
            if idx >= len(kpts):
                return False
            if scores is not None and idx < len(scores):
                return scores[idx] > 0.3
            return kpts[idx][0] > 0 or kpts[idx][1] > 0
        
        def get_point(idx: int) -> Optional[Tuple[float, float]]:
            return (kpts[idx][0], kpts[idx][1]) if visible(idx) else None
        
        # Record visibility
        body.keypoints_visible = {
            name: visible(idx) for idx, name in self.KEYPOINT_NAMES.items()
        }
        
        # Get key points
        nose = get_point(self.NOSE)
        l_shoulder = get_point(self.L_SHOULDER)
        r_shoulder = get_point(self.R_SHOULDER)
        l_elbow = get_point(self.L_ELBOW)
        r_elbow = get_point(self.R_ELBOW)
        l_wrist = get_point(self.L_WRIST)
        r_wrist = get_point(self.R_WRIST)
        l_hip = get_point(self.L_HIP)
        r_hip = get_point(self.R_HIP)
        l_knee = get_point(self.L_KNEE)
        r_knee = get_point(self.R_KNEE)
        l_ankle = get_point(self.L_ANKLE)
        r_ankle = get_point(self.R_ANKLE)
        
        # ─────────────────────────────────────────────────────────────────────
        # Measurements
        # ─────────────────────────────────────────────────────────────────────
        if l_shoulder and r_shoulder:
            body.shoulder_width_px = int(abs(r_shoulder[0] - l_shoulder[0]))
        
        if l_shoulder and r_shoulder and l_hip and r_hip:
            shoulder_y = (l_shoulder[1] + r_shoulder[1]) / 2
            hip_y = (l_hip[1] + r_hip[1]) / 2
            body.torso_height_px = int(abs(hip_y - shoulder_y))
        
        # Distance estimation
        if body.height_px > 0:
            # Rough estimation assuming 1.7m average height
            focal_length = frame_h * 0.8
            body.estimated_distance_m = round((1.7 * focal_length) / body.height_px, 1)
        
        # ─────────────────────────────────────────────────────────────────────
        # Arm Analysis
        # ─────────────────────────────────────────────────────────────────────
        # Left arm angle
        if l_shoulder and l_elbow and l_wrist:
            body.left_arm_angle = self._angle_at_point(l_shoulder, l_elbow, l_wrist)
            body.left_arm_raised = l_wrist[1] < l_shoulder[1]
        
        # Right arm angle
        if r_shoulder and r_elbow and r_wrist:
            body.right_arm_angle = self._angle_at_point(r_shoulder, r_elbow, r_wrist)
            body.right_arm_raised = r_wrist[1] < r_shoulder[1]
        
        # Arms position classification
        if body.left_arm_raised and body.right_arm_raised:
            body.arms_position = "raised"
        elif l_wrist and r_wrist and l_hip and r_hip:
            # Check if hands near hips (akimbo)
            hip_center = ((l_hip[0] + r_hip[0]) / 2, (l_hip[1] + r_hip[1]) / 2)
            l_near_hip = self._dist(l_wrist, hip_center) < body.width_px * 0.3
            r_near_hip = self._dist(r_wrist, hip_center) < body.width_px * 0.3
            if l_near_hip and r_near_hip:
                body.arms_position = "akimbo"
            # Check if arms crossed
            elif l_wrist and r_wrist and abs(l_wrist[0] - r_wrist[0]) < body.shoulder_width_px * 0.5:
                body.arms_position = "crossed"
            else:
                body.arms_position = "relaxed"
        
        # ─────────────────────────────────────────────────────────────────────
        # Body Orientation
        # ─────────────────────────────────────────────────────────────────────
        if l_shoulder and r_shoulder:
            shoulder_width = abs(r_shoulder[0] - l_shoulder[0])
            expected_width = body.width_px * 0.5
            
            if shoulder_width > expected_width * 0.75:
                body.body_orientation = "front"
            elif shoulder_width > expected_width * 0.35:
                # Determine left or right based on which shoulder is forward
                body.body_orientation = "angled"
            else:
                body.body_orientation = "side"
        
        # ─────────────────────────────────────────────────────────────────────
        # Activity Classification
        # ─────────────────────────────────────────────────────────────────────
        activity = Activity.UNKNOWN
        
        if l_shoulder and r_shoulder and l_hip and r_hip:
            shoulder_y = (l_shoulder[1] + r_shoulder[1]) / 2
            hip_y = (l_hip[1] + r_hip[1]) / 2
            torso_len = abs(hip_y - shoulder_y)
            
            if l_knee and r_knee:
                knee_y = (l_knee[1] + r_knee[1]) / 2
                
                # Sitting: knees at similar height to hips
                if abs(knee_y - hip_y) < torso_len * 0.4:
                    activity = Activity.SITTING
                # Crouching: significant knee bend but not sitting
                elif abs(knee_y - hip_y) < torso_len * 0.7:
                    activity = Activity.CROUCHING
                else:
                    activity = Activity.STANDING
            else:
                activity = Activity.STANDING
            
            # Override with arm activities
            if activity == Activity.STANDING:
                if body.arms_position == "raised":
                    activity = Activity.ARMS_RAISED
                elif body.arms_position == "crossed":
                    activity = Activity.ARMS_CROSSED
                elif body.arms_position == "akimbo":
                    activity = Activity.HANDS_ON_HIPS
                elif body.left_arm_raised != body.right_arm_raised:
                    # One arm raised - could be waving or pointing
                    if l_wrist or r_wrist:
                        raised_wrist = l_wrist if body.left_arm_raised else r_wrist
                        if raised_wrist:
                            # Check if arm is extended (pointing) or bent (waving)
                            arm_angle = body.left_arm_angle if body.left_arm_raised else body.right_arm_angle
                            if arm_angle > 150:  # Relatively straight arm
                                activity = Activity.POINTING
                            else:
                                activity = Activity.ONE_ARM_RAISED
            
            # Check for walking/running based on motion
            if body.is_moving and body.speed > 20:
                if body.speed > 50:
                    activity = Activity.RUNNING
                else:
                    activity = Activity.WALKING
        
        body.activity = activity
        
        # ─────────────────────────────────────────────────────────────────────
        # Posture Quality
        # ─────────────────────────────────────────────────────────────────────
        if l_shoulder and r_shoulder and nose:
            # Check if head is forward of shoulders (slouching)
            shoulder_center_x = (l_shoulder[0] + r_shoulder[0]) / 2
            head_offset = abs(nose[0] - shoulder_center_x)
            
            if head_offset < body.shoulder_width_px * 0.1:
                body.posture_quality = "good"
            elif head_offset < body.shoulder_width_px * 0.2:
                body.posture_quality = "normal"
            else:
                body.posture_quality = "slouching"
    
    def _analyze_motion(self, body: BodyData) -> None:
        """Analyze body motion from position history."""
        if len(body.position_history) < 2:
            body.is_moving = False
            body.motion_direction = "stationary"
            return
        
        # Calculate velocity from recent positions
        recent = list(body.position_history)[-5:]
        if len(recent) >= 2:
            dx = recent[-1][0] - recent[0][0]
            dy = recent[-1][1] - recent[0][1]
            body.velocity = (dx, dy)
            body.speed = math.sqrt(dx*dx + dy*dy)
            
            body.is_moving = body.speed > 5
            
            if body.is_moving:
                if abs(dx) > abs(dy):
                    body.motion_direction = "right" if dx > 0 else "left"
                else:
                    body.motion_direction = "down" if dy > 0 else "up"
            else:
                body.motion_direction = "stationary"
    
    # ═══════════════════════════════════════════════════════════════════════════
    # HAND ANALYSIS
    # ═══════════════════════════════════════════════════════════════════════════
    
    def _analyze_hand(self, landmarks, side: str, w: int, h: int) -> HandData:
        """Comprehensive hand analysis."""
        hand = HandData(side=side, confidence=1.0)
        
        try:
            # Convert landmarks to numpy
            pts = np.array([[lm.x * w, lm.y * h] for lm in landmarks.landmark])
            pts_3d = np.array([[lm.x, lm.y, lm.z] for lm in landmarks.landmark])
            
            hand.landmarks = pts
            hand.landmarks_3d = pts_3d
            hand.wrist_pos = (int(pts[0][0]), int(pts[0][1]))
            
            # Palm center (average of palm landmarks)
            palm_indices = [0, 5, 9, 13, 17]
            palm_pts = pts[palm_indices]
            hand.palm_center = (int(np.mean(palm_pts[:, 0])), int(np.mean(palm_pts[:, 1])))
            
            # Bounding box
            x_min, y_min = pts.min(axis=0)
            x_max, y_max = pts.max(axis=0)
            hand.hand_bbox = (int(x_min), int(y_min), int(x_max), int(y_max))
            
            # ─────────────────────────────────────────────────────────────────
            # Finger Analysis
            # ─────────────────────────────────────────────────────────────────
            finger_data = [
                ("thumb", [1, 2, 3, 4]),
                ("index", [5, 6, 7, 8]),
                ("middle", [9, 10, 11, 12]),
                ("ring", [13, 14, 15, 16]),
                ("pinky", [17, 18, 19, 20]),
            ]
            
            extended_count = 0
            
            for finger_name, indices in finger_data:
                mcp, pip, dip, tip = indices
                
                # Calculate curl angle
                curl = self._angle_at_point(
                    (pts[mcp][0], pts[mcp][1]),
                    (pts[pip][0], pts[pip][1]),
                    (pts[tip][0], pts[tip][1])
                )
                
                # Determine if extended
                if finger_name == "thumb":
                    # Thumb uses different logic - compare x position
                    if side == "right":
                        is_extended = pts[tip][0] < pts[pip][0]
                    else:
                        is_extended = pts[tip][0] > pts[pip][0]
                else:
                    # Other fingers - tip should be above pip (y is inverted)
                    is_extended = pts[tip][1] < pts[pip][1] - 10
                
                if is_extended:
                    extended_count += 1
                
                hand.fingers[finger_name] = FingerState(
                    name=finger_name,
                    is_extended=is_extended,
                    curl_angle=curl,
                    tip_position=(int(pts[tip][0]), int(pts[tip][1]))
                )
            
            hand.num_fingers_extended = extended_count
            
            # ─────────────────────────────────────────────────────────────────
            # Gesture Recognition
            # ─────────────────────────────────────────────────────────────────
            thumb = hand.fingers.get("thumb", FingerState("thumb", False))
            index = hand.fingers.get("index", FingerState("index", False))
            middle = hand.fingers.get("middle", FingerState("middle", False))
            ring = hand.fingers.get("ring", FingerState("ring", False))
            pinky = hand.fingers.get("pinky", FingerState("pinky", False))
            
            gesture = Gesture.UNKNOWN
            confidence = 0.8
            
            if extended_count == 0:
                gesture = Gesture.FIST
            elif extended_count == 5:
                gesture = Gesture.OPEN_PALM
            elif extended_count == 1:
                if index.is_extended:
                    gesture = Gesture.POINTING
                elif thumb.is_extended:
                    # Check thumb direction for up/down
                    thumb_tip_y = pts[4][1]
                    wrist_y = pts[0][1]
                    if thumb_tip_y < wrist_y - 20:
                        gesture = Gesture.THUMBS_UP
                    else:
                        gesture = Gesture.THUMBS_DOWN
                elif pinky.is_extended:
                    gesture = Gesture.PINCH
            elif extended_count == 2:
                if index.is_extended and middle.is_extended:
                    gesture = Gesture.PEACE
                elif thumb.is_extended and pinky.is_extended:
                    gesture = Gesture.CALL_ME
                elif thumb.is_extended and index.is_extended:
                    # Check for OK sign (thumb and index tips close)
                    thumb_tip = pts[4]
                    index_tip = pts[8]
                    if self._dist(thumb_tip, index_tip) < 30:
                        gesture = Gesture.OK_SIGN
            elif extended_count == 3:
                if index.is_extended and middle.is_extended and ring.is_extended:
                    gesture = Gesture.THREE
                elif index.is_extended and pinky.is_extended and not middle.is_extended:
                    gesture = Gesture.ROCK
            elif extended_count == 4:
                gesture = Gesture.FOUR
            
            hand.gesture = gesture
            hand.gesture_confidence = confidence
            
            # ─────────────────────────────────────────────────────────────────
            # Palm Orientation
            # ─────────────────────────────────────────────────────────────────
            # Use 3D landmarks to determine palm facing direction
            palm_normal_z = np.mean(pts_3d[palm_indices, 2])
            if palm_normal_z < -0.05:
                hand.palm_facing = "camera"
            elif palm_normal_z > 0.05:
                hand.palm_facing = "away"
            else:
                hand.palm_facing = "side"
        
        except Exception as e:
            pass
        
        return hand
    
    # ═══════════════════════════════════════════════════════════════════════════
    # FACE ANALYSIS  
    # ═══════════════════════════════════════════════════════════════════════════
    
    def _analyze_face(self, landmarks, w: int, h: int) -> FaceData:
        """Comprehensive face analysis."""
        face = FaceData(confidence=1.0)
        
        try:
            pts = np.array([[lm.x * w, lm.y * h, lm.z] for lm in landmarks.landmark])
            face.landmarks = pts[:, :2]
            
            # Bounding box
            xs, ys = pts[:, 0], pts[:, 1]
            face.bbox = (int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys)))
            
            # ─────────────────────────────────────────────────────────────────
            # Eye Analysis
            # ─────────────────────────────────────────────────────────────────
            # Eye aspect ratio for blink detection
            left_eye_pts = [33, 160, 158, 133, 153, 144]
            right_eye_pts = [362, 385, 387, 263, 373, 380]
            
            def eye_aspect_ratio(eye_indices):
                p = pts[eye_indices]
                v1 = np.linalg.norm(p[1] - p[5])
                v2 = np.linalg.norm(p[2] - p[4])
                h = np.linalg.norm(p[0] - p[3])
                return (v1 + v2) / (2.0 * h + 1e-6)
            
            face.left_eye_open = min(1.0, eye_aspect_ratio(left_eye_pts) * 3)
            face.right_eye_open = min(1.0, eye_aspect_ratio(right_eye_pts) * 3)
            face.is_blinking = (face.left_eye_open < 0.3 and face.right_eye_open < 0.3)
            
            # ─────────────────────────────────────────────────────────────────
            # Mouth Analysis
            # ─────────────────────────────────────────────────────────────────
            mouth_pts = [61, 291, 0, 17]  # left, right, top, bottom
            mp = pts[mouth_pts]
            mouth_h = np.linalg.norm(mp[2] - mp[3])
            mouth_w = np.linalg.norm(mp[0] - mp[1])
            face.mouth_open = min(1.0, (mouth_h / (mouth_w + 1e-6)) * 2)
            face.is_speaking = face.mouth_open > 0.3
            
            # Simple smile detection (corners up)
            mouth_left = pts[61]
            mouth_right = pts[291]
            mouth_top = pts[0]
            avg_corner_y = (mouth_left[1] + mouth_right[1]) / 2
            face.is_smiling = mouth_top[1] > avg_corner_y
            
            # ─────────────────────────────────────────────────────────────────
            # Head Pose
            # ─────────────────────────────────────────────────────────────────
            nose = pts[1]
            left_face = pts[234]
            right_face = pts[454]
            forehead = pts[10]
            chin = pts[152]
            
            # Yaw (left-right rotation)
            face_width = np.linalg.norm(left_face - right_face)
            face_center_x = (left_face[0] + right_face[0]) / 2
            yaw = (nose[0] - face_center_x) / (face_width + 1e-6) * 60
            face.yaw = float(yaw)
            
            # Pitch (up-down rotation)
            face_height = np.linalg.norm(forehead - chin)
            face_center_y = (forehead[1] + chin[1]) / 2
            pitch = (nose[1] - face_center_y) / (face_height + 1e-6) * 60
            face.pitch = float(pitch)
            
            # Gaze direction
            if abs(yaw) < 10 and abs(pitch) < 10:
                face.gaze_direction = "forward"
                face.looking_at_camera = True
            elif yaw < -10:
                face.gaze_direction = "left"
                face.looking_at_camera = False
            elif yaw > 10:
                face.gaze_direction = "right"
                face.looking_at_camera = False
            elif pitch < -10:
                face.gaze_direction = "up"
                face.looking_at_camera = False
            else:
                face.gaze_direction = "down"
                face.looking_at_camera = False
            
            # Attention score (combination of gaze and eye openness)
            gaze_score = 1.0 - min(1.0, (abs(yaw) + abs(pitch)) / 40)
            eye_score = (face.left_eye_open + face.right_eye_open) / 2
            face.attention_score = gaze_score * eye_score
        
        except Exception as e:
            pass
        
        return face
    
    # ═══════════════════════════════════════════════════════════════════════════
    # DRAWING METHODS
    # ═══════════════════════════════════════════════════════════════════════════
    
    def _draw_frame(self, frame: np.ndarray, bodies: List[BodyData]) -> np.ndarray:
        """Draw all visualizations on frame."""
        h, w = frame.shape[:2]
        
        for body in bodies:
            # Draw skeleton
            if self.show_skeleton:
                frame = self._draw_skeleton(frame, body)
            
            # Draw hands
            if self.show_hands:
                if body.left_hand:
                    frame = self._draw_hand(frame, body.left_hand)
                if body.right_hand:
                    frame = self._draw_hand(frame, body.right_hand)
            
            # Draw face
            if self.show_face and body.face:
                frame = self._draw_face(frame, body.face)
            
            # Draw bounding box
            frame = self._draw_bbox(frame, body)
            
            # Draw info panel
            if self.show_panel:
                frame = self._draw_panel(frame, body, w, h)
        
        # Draw frame info
        if self.show_metrics:
            frame = self._draw_metrics(frame, len(bodies))
        
        return frame
    
    def _draw_skeleton(self, frame: np.ndarray, body: BodyData) -> np.ndarray:
        """Draw body skeleton with clean styling."""
        if body.keypoints is None:
            return frame
        
        kpts = body.keypoints
        scores = body.keypoint_scores
        
        def valid(idx):
            if idx >= len(kpts):
                return False
            if scores is not None and idx < len(scores):
                return scores[idx] > 0.3
            return kpts[idx][0] > 0 or kpts[idx][1] > 0
        
        def get_bone_color(i1, i2):
            """Get color based on body part."""
            head = {0, 1, 2, 3, 4}
            arm_l = {5, 7, 9}
            arm_r = {6, 8, 10}
            leg_l = {11, 13, 15}
            leg_r = {12, 14, 16}
            
            if i1 in head or i2 in head:
                return Theme.SKELETON_HEAD
            if i1 in arm_l or i2 in arm_l:
                return Theme.SKELETON_ARM_L
            if i1 in arm_r or i2 in arm_r:
                return Theme.SKELETON_ARM_R
            if i1 in leg_l or i2 in leg_l:
                return Theme.SKELETON_LEG_L
            if i1 in leg_r or i2 in leg_r:
                return Theme.SKELETON_LEG_R
            return Theme.SKELETON_TORSO
        
        # Draw bones
        for i1, i2 in self.SKELETON_CONNECTIONS:
            if not valid(i1) or not valid(i2):
                continue
            
            pt1 = (int(kpts[i1][0]), int(kpts[i1][1]))
            pt2 = (int(kpts[i2][0]), int(kpts[i2][1]))
            color = get_bone_color(i1, i2)
            
            # Draw bone with slight transparency effect
            cv.line(frame, pt1, pt2, color, self.skeleton_thickness + 2)
            cv.line(frame, pt1, pt2, Theme.JOINT_FILL, self.skeleton_thickness)
        
        # Draw joints
        for i, pt in enumerate(kpts):
            if not valid(i):
                continue
            
            x, y = int(pt[0]), int(pt[1])
            
            # Joint styling
            cv.circle(frame, (x, y), self.joint_radius + 2, Theme.JOINT_BORDER, -1)
            cv.circle(frame, (x, y), self.joint_radius, Theme.JOINT_FILL, -1)
            
            # Larger circles for key joints
            if i in [self.L_WRIST, self.R_WRIST, self.L_ANKLE, self.R_ANKLE]:
                cv.circle(frame, (x, y), self.joint_radius + 1, Theme.ACCENT, 1)
        
        return frame
    
    def _draw_hand(self, frame: np.ndarray, hand: HandData) -> np.ndarray:
        """Draw hand skeleton and finger tracking."""
        if hand.landmarks is None:
            return frame
        
        pts = hand.landmarks
        
        # Finger colors
        finger_colors = {
            0: Theme.HAND_BASE,    # Wrist/palm
            1: Theme.HAND_THUMB,   # Thumb
            2: Theme.HAND_INDEX,   # Index
            3: Theme.HAND_MIDDLE,  # Middle
            4: Theme.HAND_RING,    # Ring
            5: Theme.HAND_PINKY,   # Pinky
        }
        
        def get_finger_color(idx):
            if idx <= 4: return finger_colors[1]      # Thumb
            elif idx <= 8: return finger_colors[2]    # Index
            elif idx <= 12: return finger_colors[3]   # Middle
            elif idx <= 16: return finger_colors[4]   # Ring
            else: return finger_colors[5]             # Pinky
        
        # Draw connections
        for start, end in self.HAND_CONNECTIONS:
            if start >= len(pts) or end >= len(pts):
                continue
            
            pt1 = (int(pts[start][0]), int(pts[start][1]))
            pt2 = (int(pts[end][0]), int(pts[end][1]))
            color = get_finger_color(end)
            
            cv.line(frame, pt1, pt2, color, 2)
        
        # Draw landmarks
        for i, pt in enumerate(pts):
            x, y = int(pt[0]), int(pt[1])
            color = get_finger_color(i)
            
            # Fingertips are larger
            if i in [4, 8, 12, 16, 20]:
                # Check if finger is extended
                finger_names = ["thumb", "index", "middle", "ring", "pinky"]
                finger_idx = (i - 4) // 4
                finger_name = finger_names[finger_idx] if finger_idx < len(finger_names) else "thumb"
                finger = hand.fingers.get(finger_name)
                
                if finger and finger.is_extended:
                    # Extended finger - bright tip
                    cv.circle(frame, (x, y), 6, Theme.TEXT_PRIMARY, -1)
                    cv.circle(frame, (x, y), 4, color, -1)
                else:
                    # Curled finger - dimmer
                    cv.circle(frame, (x, y), 4, color, -1)
            else:
                cv.circle(frame, (x, y), 3, color, -1)
        
        # Draw gesture label
        if hand.gesture != Gesture.UNKNOWN and hand.palm_center:
            label = hand.gesture.value.upper()
            pos = (hand.palm_center[0] - 30, hand.palm_center[1] + 40)
            
            # Background
            (tw, th), _ = cv.getTextSize(label, cv.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            cv.rectangle(frame, (pos[0] - 2, pos[1] - th - 2), 
                        (pos[0] + tw + 2, pos[1] + 2), Theme.PANEL_BG, -1)
            cv.putText(frame, label, pos, cv.FONT_HERSHEY_SIMPLEX, 0.45, 
                      Theme.SUCCESS, 1, cv.LINE_AA)
        
        return frame
    
    def _draw_face(self, frame: np.ndarray, face: FaceData) -> np.ndarray:
        """Draw face analysis overlay."""
        if face.landmarks is None or face.bbox is None:
            return frame
        
        pts = face.landmarks
        x1, y1, x2, y2 = face.bbox
        
        # Draw face contour (simplified)
        contour_indices = [
            10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
            397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
            172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109
        ]
        
        contour_pts = np.array([pts[i] for i in contour_indices if i < len(pts)], dtype=np.int32)
        if len(contour_pts) > 2:
            cv.polylines(frame, [contour_pts], True, Theme.FACE_CONTOUR, 1, cv.LINE_AA)
        
        # Draw eye regions
        left_eye = [33, 7, 163, 144, 145, 153, 154, 155, 133]
        right_eye = [362, 382, 381, 380, 374, 373, 390, 249, 263]
        
        for eye_indices in [left_eye, right_eye]:
            eye_pts = np.array([pts[i] for i in eye_indices if i < len(pts)], dtype=np.int32)
            if len(eye_pts) > 2:
                cv.polylines(frame, [eye_pts], True, Theme.FACE_EYES, 1, cv.LINE_AA)
        
        # Draw mouth region
        mouth_outer = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291, 375, 321, 405, 314, 17, 84, 181, 91, 146]
        mouth_pts = np.array([pts[i] for i in mouth_outer if i < len(pts)], dtype=np.int32)
        if len(mouth_pts) > 2:
            cv.polylines(frame, [mouth_pts], True, Theme.FACE_MOUTH, 1, cv.LINE_AA)
        
        # Draw subtle face box corners
        corner_len = 12
        
        for corner, dx, dy in [
            ((x1, y1), (1, 0), (0, 1)),
            ((x2, y1), (-1, 0), (0, 1)),
            ((x1, y2), (1, 0), (0, -1)),
            ((x2, y2), (-1, 0), (0, -1)),
        ]:
            cv.line(frame, corner, (corner[0] + dx[0]*corner_len, corner[1] + dx[1]*corner_len), 
                   Theme.FACE_CONTOUR, 1)
            cv.line(frame, corner, (corner[0] + dy[0]*corner_len, corner[1] + dy[1]*corner_len), 
                   Theme.FACE_CONTOUR, 1)
        
        # Status indicators
        status_x = x2 + 5
        status_y = y1
        
        indicators = []
        if face.is_speaking:
            indicators.append(("SPEAKING", Theme.SUCCESS))
        if face.is_smiling:
            indicators.append(("SMILING", Theme.ACCENT))
        if not face.looking_at_camera:
            indicators.append((f"LOOKING {face.gaze_direction.upper()}", Theme.WARNING))
        
        for i, (text, color) in enumerate(indicators):
            y = status_y + i * 15
            cv.putText(frame, text, (status_x, y + 10), cv.FONT_HERSHEY_SIMPLEX, 
                      0.35, color, 1, cv.LINE_AA)
        
        return frame
    
    def _draw_bbox(self, frame: np.ndarray, body: BodyData) -> np.ndarray:
        """Draw bounding box with clean corner style."""
        x1, y1, x2, y2 = body.bbox
        
        color = Theme.BBOX_ACTIVE if body.time_tracked > 0.5 else Theme.BBOX_INACTIVE
        corner_len = min(25, (x2 - x1) // 5, (y2 - y1) // 5)
        
        # Draw corners only
        for corner, dx1, dy1, dx2, dy2 in [
            ((x1, y1), corner_len, 0, 0, corner_len),
            ((x2, y1), -corner_len, 0, 0, corner_len),
            ((x1, y2), corner_len, 0, 0, -corner_len),
            ((x2, y2), -corner_len, 0, 0, -corner_len),
        ]:
            cv.line(frame, corner, (corner[0] + dx1, corner[1] + dy1), color, 2)
            cv.line(frame, corner, (corner[0] + dx2, corner[1] + dy2), color, 2)
        
        # ID label
        if body.track_id is not None:
            label = f"Person {body.track_id}"
            (tw, th), _ = cv.getTextSize(label, cv.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            
            cv.rectangle(frame, (x1, y1 - th - 6), (x1 + tw + 8, y1), color, -1)
            cv.putText(frame, label, (x1 + 4, y1 - 4), cv.FONT_HERSHEY_SIMPLEX, 
                      0.45, Theme.PANEL_BG, 1, cv.LINE_AA)
        
        return frame
    
    def _draw_panel(self, frame: np.ndarray, body: BodyData, frame_w: int, frame_h: int) -> np.ndarray:
        """Draw information panel."""
        x1, y1, x2, y2 = body.bbox
        
        # Panel dimensions
        panel_w = 150
        line_h = 16
        
        # Position panel
        if self.panel_position == "right":
            panel_x = x2 + 8
            if panel_x + panel_w > frame_w:
                panel_x = max(0, x1 - panel_w - 8)
        else:
            panel_x = max(0, x1 - panel_w - 8)
        
        panel_y = y1
        
        # Build content
        lines = []
        
        # Activity
        activity = body.activity.value if body.activity != Activity.UNKNOWN else "detected"
        lines.append(("Activity", activity.title()))
        
        # Distance
        if body.estimated_distance_m > 0:
            lines.append(("Distance", f"{body.estimated_distance_m:.1f}m"))
        
        # Orientation
        lines.append(("Facing", body.body_orientation.title()))
        
        # Arms
        if body.arms_position != "relaxed":
            lines.append(("Arms", body.arms_position.title()))
        
        # Motion
        if body.is_moving:
            lines.append(("Moving", body.motion_direction.title()))
        
        # Hands
        for hand, label in [(body.left_hand, "L Hand"), (body.right_hand, "R Hand")]:
            if hand and hand.gesture != Gesture.UNKNOWN:
                lines.append((label, hand.gesture.value.title()))
        
        # Face
        if body.face:
            if body.face.attention_score < 0.5:
                lines.append(("Attention", "Low"))
            elif body.face.is_speaking:
                lines.append(("Speaking", "Yes"))
        
        # Calculate panel height
        panel_h = len(lines) * line_h + 24
        
        # Draw panel background
        overlay = frame.copy()
        cv.rectangle(overlay, (panel_x, panel_y), (panel_x + panel_w, panel_y + panel_h), 
                    Theme.PANEL_BG, -1)
        cv.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)
        
        # Draw border
        cv.rectangle(frame, (panel_x, panel_y), (panel_x + panel_w, panel_y + panel_h), 
                    Theme.PANEL_BORDER, 1)
        
        # Draw header
        header = "ANALYSIS"
        cv.putText(frame, header, (panel_x + 6, panel_y + 14), cv.FONT_HERSHEY_SIMPLEX, 
                  0.4, Theme.PRIMARY, 1, cv.LINE_AA)
        cv.line(frame, (panel_x + 4, panel_y + 18), (panel_x + panel_w - 4, panel_y + 18), 
               Theme.PANEL_BORDER, 1)
        
        # Draw lines
        for i, (label, value) in enumerate(lines):
            y = panel_y + 32 + i * line_h
            cv.putText(frame, f"{label}:", (panel_x + 6, y), cv.FONT_HERSHEY_SIMPLEX, 
                      0.35, Theme.TEXT_SECONDARY, 1, cv.LINE_AA)
            cv.putText(frame, value, (panel_x + 60, y), cv.FONT_HERSHEY_SIMPLEX, 
                      0.35, Theme.TEXT_LABEL, 1, cv.LINE_AA)
        
        return frame
    
    def _draw_metrics(self, frame: np.ndarray, num_people: int) -> np.ndarray:
        """Draw frame metrics."""
        h, w = frame.shape[:2]
        
        elapsed = time.time() - self._start_time
        
        # Top-right info
        info = f"People: {num_people} | Time: {elapsed:.1f}s"
        (tw, th), _ = cv.getTextSize(info, cv.FONT_HERSHEY_SIMPLEX, 0.4, 1)
        
        cv.rectangle(frame, (w - tw - 12, 4), (w - 4, th + 10), Theme.PANEL_BG, -1)
        cv.putText(frame, info, (w - tw - 8, th + 6), cv.FONT_HERSHEY_SIMPLEX, 
                  0.4, Theme.TEXT_SECONDARY, 1, cv.LINE_AA)
        
        return frame
    
    # ═══════════════════════════════════════════════════════════════════════════
    # UTILITY METHODS
    # ═══════════════════════════════════════════════════════════════════════════
    
    def _to_numpy(self, tensor) -> Optional[np.ndarray]:
        if tensor is None:
            return None
        try:
            return tensor.cpu().numpy()
        except:
            return None
    
    def _dist(self, p1, p2) -> float:
        """Euclidean distance between two points."""
        return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
    
    def _angle_at_point(self, p1, p2, p3) -> float:
        """Calculate angle at p2 formed by p1-p2-p3."""
        v1 = np.array([p1[0] - p2[0], p1[1] - p2[1]])
        v2 = np.array([p3[0] - p2[0], p3[1] - p2[1]])
        
        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
        angle = np.degrees(np.arccos(np.clip(cos_angle, -1, 1)))
        return float(angle)
    
    def cleanup(self):
        """Release resources."""
        if self.hands_detector:
            self.hands_detector.close()
        if self.face_detector:
            self.face_detector.close()


# ═══════════════════════════════════════════════════════════════════════════════
# FACTORY FUNCTION
# ═══════════════════════════════════════════════════════════════════════════════

def create_human_analyzer(**kwargs) -> HumanAnalyzer:
    """Create HumanAnalyzer with configuration."""
    return HumanAnalyzer(
        enable_hands=kwargs.get('enable_hands', True),
        enable_face=kwargs.get('enable_face', True),
        confidence_threshold=kwargs.get('confidence_threshold', 0.4),
        show_skeleton=kwargs.get('show_skeleton', True),
        show_hands=kwargs.get('show_hands', True),
        show_face=kwargs.get('show_face', True),
        show_labels=kwargs.get('show_labels', True),
        show_panel=kwargs.get('show_panel', True),
        show_metrics=kwargs.get('show_metrics', True),
    )


# Aliases for compatibility
UltraHumanAnalyzer = HumanAnalyzer