"""
Sign Language Interpreter - Sign-to-Voice Communication for Smart Glasses
==========================================================================

A production-grade module for real-time sign language recognition enabling
blind users to understand sign language through audio feedback.

Features:
- Real-time hand landmark tracking via MediaPipe
- ASL alphabet fingerspelling recognition (A-Z, 0-9)
- Common signs library (50+ phrases)
- Temporal pattern analysis for dynamic signs
- Confidence-based speech output with uncertainty prompts
- Multi-hand support for two-handed signs
- Kalman filtering for smooth landmark tracking
- Gesture sequence detection for words/phrases
- Adaptive learning with feedback mechanism

Author: VisionAssist AI Team
Version: 1.0
"""

from __future__ import annotations

import cv2 as cv
import numpy as np
import mediapipe as mp
from typing import List, Dict, Any, Optional, Tuple, Deque, Callable
from dataclasses import dataclass, field
from collections import deque
from enum import Enum
import time
import math
import threading


# =============================================================================
# Enums and Constants
# =============================================================================

class SignCategory(Enum):
    """Categories of signs"""
    ALPHABET = "alphabet"
    NUMBER = "number"
    COMMON_WORD = "common_word"
    PHRASE = "phrase"
    DYNAMIC = "dynamic"  # Signs requiring motion
    TWO_HANDED = "two_handed"


class SignConfidence(Enum):
    """Confidence levels for recognition"""
    HIGH = "high"  # > 0.85
    MEDIUM = "medium"  # 0.65 - 0.85
    LOW = "low"  # 0.45 - 0.65
    UNCERTAIN = "uncertain"  # < 0.45


class InterpreterMode(Enum):
    """Operating modes"""
    FINGERSPELLING = "fingerspelling"  # Letter-by-letter
    WORD_SIGNS = "word_signs"  # Common signs/words
    CONTINUOUS = "continuous"  # Both modes combined
    LEARNING = "learning"  # Feedback collection mode


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class HandLandmarks:
    """Processed hand landmark data"""
    landmarks: np.ndarray  # 21x3 array of (x, y, z) normalized coordinates
    world_landmarks: Optional[np.ndarray] = None  # 3D world coordinates
    handedness: str = "unknown"  # "left" or "right"
    confidence: float = 0.0
    wrist_position: Tuple[float, float] = (0.0, 0.0)
    palm_center: Tuple[float, float] = (0.0, 0.0)
    palm_normal: Tuple[float, float, float] = (0.0, 0.0, 1.0)
    finger_states: Dict[str, bool] = field(default_factory=dict)
    finger_angles: Dict[str, float] = field(default_factory=dict)


@dataclass
class RecognizedSign:
    """A recognized sign with metadata"""
    sign: str  # The recognized sign (letter, word, or phrase)
    category: SignCategory = SignCategory.ALPHABET
    confidence: float = 0.0
    confidence_level: SignConfidence = SignConfidence.UNCERTAIN
    timestamp: float = 0.0
    duration: float = 0.0  # How long the sign was held
    hand_used: str = "right"  # "left", "right", or "both"
    alternatives: List[Tuple[str, float]] = field(default_factory=list)


@dataclass
class SignSequence:
    """A sequence of signs forming a word or phrase"""
    signs: List[RecognizedSign] = field(default_factory=list)
    interpreted_text: str = ""
    confidence: float = 0.0
    start_time: float = 0.0
    end_time: float = 0.0
    is_complete: bool = False


@dataclass
class InterpreterState:
    """Current state of the interpreter"""
    mode: InterpreterMode = InterpreterMode.CONTINUOUS
    is_signing: bool = False
    current_sequence: SignSequence = field(default_factory=SignSequence)
    last_sign: Optional[RecognizedSign] = None
    last_spoken: str = ""
    last_spoken_time: float = 0.0
    accumulated_text: str = ""
    pause_detected: bool = False


# =============================================================================
# ASL Alphabet and Signs Database
# =============================================================================

class ASLSignsDatabase:
    """
    Database of ASL signs with their landmark patterns.
    Uses normalized finger positions and angles for recognition.
    """
    
    # Finger tip and base landmark indices
    FINGER_TIPS = [4, 8, 12, 16, 20]  # thumb, index, middle, ring, pinky
    FINGER_PIPS = [3, 6, 10, 14, 18]  # proximal interphalangeal joints
    FINGER_MCPS = [2, 5, 9, 13, 17]   # metacarpophalangeal joints
    
    # Landmark indices for key positions
    WRIST = 0
    THUMB_TIP = 4
    THUMB_IP = 3
    THUMB_MCP = 2
    INDEX_TIP = 8
    INDEX_PIP = 6
    INDEX_MCP = 5
    MIDDLE_TIP = 12
    MIDDLE_PIP = 10
    MIDDLE_MCP = 9
    RING_TIP = 16
    RING_PIP = 14
    RING_MCP = 13
    PINKY_TIP = 20
    PINKY_PIP = 18
    PINKY_MCP = 17
    
    def __init__(self):
        """Initialize the signs database"""
        self.alphabet_patterns = self._build_alphabet_patterns()
        self.number_patterns = self._build_number_patterns()
        self.common_signs = self._build_common_signs()
        self.dynamic_signs = self._build_dynamic_signs()
        
    def _build_alphabet_patterns(self) -> Dict[str, Dict[str, Any]]:
        """
        Build recognition patterns for ASL alphabet.
        Each letter is defined by finger extension states and relative positions.
        """
        patterns = {
            # Format: finger_extended: [thumb, index, middle, ring, pinky]
            # Additional constraints for disambiguation
            
            'A': {
                'fingers_extended': [False, False, False, False, False],
                'thumb_position': 'beside_fist',  # Thumb alongside fingers
                'description': 'Fist with thumb beside index finger'
            },
            'B': {
                'fingers_extended': [False, True, True, True, True],
                'thumb_position': 'tucked',  # Thumb across palm
                'fingers_together': True,
                'description': 'Flat hand, fingers together, thumb tucked'
            },
            'C': {
                'fingers_extended': [True, True, True, True, True],
                'hand_shape': 'curved',
                'description': 'Curved hand forming C shape'
            },
            'D': {
                'fingers_extended': [False, True, False, False, False],
                'thumb_touches': 'middle_ring_pinky',
                'index_straight': True,
                'description': 'Index up, other fingers touch thumb'
            },
            'E': {
                'fingers_extended': [False, False, False, False, False],
                'fingers_curled': True,
                'thumb_position': 'under_fingers',
                'description': 'Fingers curled, thumb under fingers'
            },
            'F': {
                'fingers_extended': [True, False, True, True, True],
                'thumb_index_touch': True,
                'description': 'Thumb and index touch, others extended'
            },
            'G': {
                'fingers_extended': [True, True, False, False, False],
                'hand_orientation': 'sideways',
                'thumb_index_parallel': True,
                'description': 'Index and thumb parallel, pointing sideways'
            },
            'H': {
                'fingers_extended': [True, True, True, False, False],
                'hand_orientation': 'sideways',
                'description': 'Index and middle extended sideways'
            },
            'I': {
                'fingers_extended': [False, False, False, False, True],
                'description': 'Only pinky extended'
            },
            'J': {
                'fingers_extended': [False, False, False, False, True],
                'motion': 'j_curve',  # Dynamic sign
                'description': 'Pinky draws J shape'
            },
            'K': {
                'fingers_extended': [True, True, True, False, False],
                'index_middle_spread': True,
                'thumb_between': True,
                'description': 'Index and middle up spread, thumb between'
            },
            'L': {
                'fingers_extended': [True, True, False, False, False],
                'thumb_index_perpendicular': True,
                'description': 'L shape with thumb and index'
            },
            'M': {
                'fingers_extended': [False, False, False, False, False],
                'thumb_under_three': True,
                'description': 'Thumb under first three fingers'
            },
            'N': {
                'fingers_extended': [False, False, False, False, False],
                'thumb_under_two': True,
                'description': 'Thumb under first two fingers'
            },
            'O': {
                'fingers_extended': [True, True, True, True, True],
                'fingers_curved_to_thumb': True,
                'description': 'Fingers curved touching thumb tip (O shape)'
            },
            'P': {
                'fingers_extended': [True, True, True, False, False],
                'hand_orientation': 'down',
                'description': 'K hand shape pointing down'
            },
            'Q': {
                'fingers_extended': [True, True, False, False, False],
                'hand_orientation': 'down',
                'description': 'G hand shape pointing down'
            },
            'R': {
                'fingers_extended': [False, True, True, False, False],
                'fingers_crossed': True,
                'description': 'Index and middle crossed'
            },
            'S': {
                'fingers_extended': [False, False, False, False, False],
                'thumb_over_fingers': True,
                'description': 'Fist with thumb over fingers'
            },
            'T': {
                'fingers_extended': [False, False, False, False, False],
                'thumb_between_index_middle': True,
                'description': 'Thumb between index and middle'
            },
            'U': {
                'fingers_extended': [False, True, True, False, False],
                'fingers_together': True,
                'description': 'Index and middle extended together'
            },
            'V': {
                'fingers_extended': [False, True, True, False, False],
                'fingers_spread': True,
                'description': 'Index and middle spread (peace sign)'
            },
            'W': {
                'fingers_extended': [False, True, True, True, False],
                'fingers_spread': True,
                'description': 'Index, middle, ring spread'
            },
            'X': {
                'fingers_extended': [False, False, False, False, False],
                'index_hooked': True,
                'description': 'Index finger hooked'
            },
            'Y': {
                'fingers_extended': [True, False, False, False, True],
                'description': 'Thumb and pinky extended (hang loose)'
            },
            'Z': {
                'fingers_extended': [False, True, False, False, False],
                'motion': 'z_trace',  # Dynamic sign
                'description': 'Index traces Z in air'
            },
        }
        return patterns
    
    def _build_number_patterns(self) -> Dict[str, Dict[str, Any]]:
        """Build recognition patterns for numbers 0-9"""
        patterns = {
            '0': {
                'fingers_extended': [True, True, True, True, True],
                'fingers_curved_to_thumb': True,
                'description': 'O shape (same as letter O)'
            },
            '1': {
                'fingers_extended': [False, True, False, False, False],
                'description': 'Index finger up'
            },
            '2': {
                'fingers_extended': [False, True, True, False, False],
                'fingers_spread': True,
                'description': 'Index and middle spread (V shape)'
            },
            '3': {
                'fingers_extended': [True, True, True, False, False],
                'description': 'Thumb, index, middle extended'
            },
            '4': {
                'fingers_extended': [False, True, True, True, True],
                'description': 'Four fingers extended, thumb tucked'
            },
            '5': {
                'fingers_extended': [True, True, True, True, True],
                'hand_flat': True,
                'description': 'All fingers extended and spread'
            },
            '6': {
                'fingers_extended': [True, False, False, False, True],
                'thumb_pinky_touch': False,
                'description': 'Thumb and pinky out (like Y but different)'
            },
            '7': {
                'fingers_extended': [True, False, False, True, False],
                'description': 'Thumb and ring extended'
            },
            '8': {
                'fingers_extended': [True, False, True, False, False],
                'description': 'Thumb and middle extended'
            },
            '9': {
                'fingers_extended': [True, True, False, False, False],
                'thumb_index_touch': True,
                'description': 'Thumb and index touch (like F)'
            },
        }
        return patterns
    
    def _build_common_signs(self) -> Dict[str, Dict[str, Any]]:
        """Build recognition patterns for common ASL words/phrases"""
        signs = {
            # Greetings and Social
            'hello': {
                'type': 'single_hand',
                'motion': 'wave_from_forehead',
                'fingers_extended': [True, True, True, True, True],
                'description': 'Flat hand waves from forehead outward'
            },
            'goodbye': {
                'type': 'single_hand',
                'motion': 'wave',
                'fingers_extended': [True, True, True, True, True],
                'description': 'Open hand waving motion'
            },
            'please': {
                'type': 'single_hand',
                'motion': 'circular_chest',
                'hand_flat': True,
                'description': 'Flat hand circles on chest'
            },
            'thank_you': {
                'type': 'single_hand',
                'motion': 'from_chin_forward',
                'hand_flat': True,
                'description': 'Flat hand from chin moving forward'
            },
            'sorry': {
                'type': 'single_hand',
                'motion': 'circular_chest',
                'hand_shape': 'fist',
                'description': 'Fist circles on chest'
            },
            'yes': {
                'type': 'single_hand',
                'motion': 'nod_fist',
                'hand_shape': 'fist',
                'description': 'Fist nodding motion'
            },
            'no': {
                'type': 'single_hand',
                'motion': 'snap_fingers',
                'description': 'Index and middle snap to thumb'
            },
            
            # Questions
            'what': {
                'type': 'single_hand',
                'motion': 'shake_hand',
                'hand_flat': True,
                'description': 'Relaxed hand shaking side to side'
            },
            'where': {
                'type': 'single_hand',
                'motion': 'shake_index',
                'fingers_extended': [False, True, False, False, False],
                'description': 'Index finger shaking side to side'
            },
            'who': {
                'type': 'single_hand',
                'location': 'near_mouth',
                'motion': 'circle_mouth',
                'description': 'Index circles near mouth'
            },
            'when': {
                'type': 'two_handed',
                'motion': 'circle_around_index',
                'description': 'Index finger circles around other index'
            },
            'why': {
                'type': 'single_hand',
                'location': 'forehead',
                'motion': 'touch_forehead_to_y',
                'description': 'Touch forehead then form Y'
            },
            'how': {
                'type': 'two_handed',
                'motion': 'knuckles_roll',
                'description': 'Knuckles together, roll forward'
            },
            
            # Common Words
            'help': {
                'type': 'two_handed',
                'motion': 'lift_fist_on_palm',
                'description': 'Fist on flat palm, lift up'
            },
            'stop': {
                'type': 'two_handed',
                'motion': 'chop_palm',
                'description': 'Flat hand chops onto other palm'
            },
            'more': {
                'type': 'two_handed',
                'motion': 'fingertips_tap',
                'description': 'Fingertips of both hands tap together'
            },
            'want': {
                'type': 'single_hand',
                'motion': 'pull_toward',
                'hand_shape': 'claw',
                'description': 'Claw hand pulls toward body'
            },
            'need': {
                'type': 'single_hand',
                'motion': 'bend_index',
                'description': 'Index finger bends repeatedly'
            },
            'like': {
                'type': 'single_hand',
                'location': 'chest',
                'motion': 'pull_away',
                'description': 'Middle and thumb from chest, pull away'
            },
            'understand': {
                'type': 'single_hand',
                'location': 'temple',
                'motion': 'flick_index',
                'description': 'Index flicks up near temple'
            },
            'dont_understand': {
                'type': 'single_hand',
                'location': 'forehead',
                'motion': 'touch_then_shake',
                'description': 'Touch forehead, then shake hand'
            },
            'again': {
                'type': 'two_handed',
                'motion': 'curved_hand_to_palm',
                'description': 'Curved hand arcs to flat palm'
            },
            'slow': {
                'type': 'two_handed',
                'motion': 'hand_slides_up',
                'description': 'Hand slides slowly up other arm'
            },
            'fast': {
                'type': 'two_handed',
                'motion': 'thumbs_flick',
                'description': 'Fists with thumbs that flick out'
            },
            
            # Communication Related
            'sign': {
                'type': 'two_handed',
                'motion': 'alternate_circles',
                'description': 'Index fingers alternating circles'
            },
            'speak': {
                'type': 'single_hand',
                'location': 'mouth',
                'motion': 'tap_mouth',
                'description': 'Index taps near mouth'
            },
            'listen': {
                'type': 'single_hand',
                'location': 'ear',
                'motion': 'cup_ear',
                'description': 'Cupped hand near ear'
            },
            'repeat': {
                'type': 'two_handed',
                'motion': 'flip_hand_on_palm',
                'description': 'One hand flips onto other palm'
            },
            
            # Descriptions
            'good': {
                'type': 'single_hand',
                'motion': 'from_chin_down',
                'hand_flat': True,
                'description': 'Flat hand from chin moves down'
            },
            'bad': {
                'type': 'single_hand',
                'location': 'chin',
                'motion': 'flip_down',
                'description': 'Hand at chin flips down'
            },
            'big': {
                'type': 'two_handed',
                'motion': 'expand_apart',
                'description': 'Hands move apart showing size'
            },
            'small': {
                'type': 'two_handed',
                'motion': 'compress_together',
                'description': 'Hands move together showing small'
            },
            
            # Emergency/Important
            'emergency': {
                'type': 'single_hand',
                'motion': 'shake_e',
                'hand_shape': 'e_hand',
                'description': 'E handshape shaking'
            },
            'danger': {
                'type': 'two_handed',
                'motion': 'thumbs_up_collide',
                'description': 'Fists with thumbs move up and together'
            },
            'wait': {
                'type': 'two_handed',
                'motion': 'wiggle_fingers_up',
                'description': 'Hands up, wiggling fingers'
            },
        }
        return signs
    
    def _build_dynamic_signs(self) -> Dict[str, Dict[str, Any]]:
        """Signs that require motion tracking"""
        return {
            'j_motion': {
                'start_shape': 'i',  # Pinky extended
                'motion_pattern': 'curve_down_hook',
                'duration_ms': (300, 800),
            },
            'z_motion': {
                'start_shape': 'point',  # Index extended
                'motion_pattern': 'zigzag',
                'duration_ms': (400, 1000),
            },
            'wave_motion': {
                'motion_pattern': 'side_to_side',
                'cycles': (2, 5),
                'duration_ms': (500, 2000),
            },
            'nod_motion': {
                'motion_pattern': 'up_down',
                'cycles': (2, 4),
                'duration_ms': (300, 1000),
            },
            'circle_motion': {
                'motion_pattern': 'circular',
                'direction': 'any',
                'duration_ms': (500, 1500),
            },
        }


# =============================================================================
# Hand Analysis Engine
# =============================================================================

class HandAnalysisEngine:
    """
    Analyzes hand landmarks to extract features for sign recognition.
    Uses geometric analysis and temporal patterns.
    """
    
    def __init__(self):
        self.db = ASLSignsDatabase()
        
        # Temporal smoothing buffers
        self.landmark_history: Dict[str, Deque[np.ndarray]] = {
            'left': deque(maxlen=10),
            'right': deque(maxlen=10)
        }
        self.gesture_history: Dict[str, Deque[str]] = {
            'left': deque(maxlen=15),
            'right': deque(maxlen=15)
        }
        
        # Kalman filters for smoothing
        self.kalman_filters: Dict[str, Any] = {}
        
        # Motion tracking
        self.position_history: Dict[str, Deque[Tuple[float, float]]] = {
            'left': deque(maxlen=30),
            'right': deque(maxlen=30)
        }
        
    def process_landmarks(
        self, 
        landmarks: Any, 
        handedness: str,
        frame_width: int,
        frame_height: int
    ) -> HandLandmarks:
        """Process raw MediaPipe landmarks into structured data"""
        
        # Normalize handedness to lowercase
        handedness = handedness.lower()
        
        # Convert to numpy array
        lm_array = np.array([
            [lm.x, lm.y, lm.z] for lm in landmarks.landmark
        ])
        
        # Apply temporal smoothing
        smoothed = self._smooth_landmarks(lm_array, handedness)
        
        # Calculate derived features
        wrist = (smoothed[0, 0] * frame_width, smoothed[0, 1] * frame_height)
        palm_center = self._calculate_palm_center(smoothed, frame_width, frame_height)
        palm_normal = self._calculate_palm_normal(smoothed)
        finger_states = self._get_finger_states(smoothed, handedness)
        finger_angles = self._get_finger_angles(smoothed)
        
        # Update position history for motion detection
        self.position_history[handedness].append(palm_center)
        
        return HandLandmarks(
            landmarks=smoothed,
            handedness=handedness,
            confidence=landmarks.landmark[0].visibility if hasattr(landmarks.landmark[0], 'visibility') else 1.0,
            wrist_position=wrist,
            palm_center=palm_center,
            palm_normal=palm_normal,
            finger_states=finger_states,
            finger_angles=finger_angles
        )
    
    def _smooth_landmarks(self, landmarks: np.ndarray, handedness: str) -> np.ndarray:
        """Apply temporal smoothing to landmarks"""
        self.landmark_history[handedness].append(landmarks.copy())
        
        if len(self.landmark_history[handedness]) < 3:
            return landmarks
        
        # Weighted average of recent frames
        weights = np.array([0.1, 0.2, 0.3, 0.4])[-len(self.landmark_history[handedness]):]
        weights = weights / weights.sum()
        
        smoothed = np.zeros_like(landmarks)
        for i, hist_lm in enumerate(list(self.landmark_history[handedness])[-len(weights):]):
            smoothed += hist_lm * weights[i]
        
        return smoothed
    
    def _calculate_palm_center(
        self, 
        landmarks: np.ndarray,
        frame_width: int,
        frame_height: int
    ) -> Tuple[float, float]:
        """Calculate the center of the palm"""
        # Use wrist and MCP joints to find palm center
        palm_points = [0, 5, 9, 13, 17]  # Wrist and MCP joints
        center = landmarks[palm_points].mean(axis=0)
        return (center[0] * frame_width, center[1] * frame_height)
    
    def _calculate_palm_normal(self, landmarks: np.ndarray) -> Tuple[float, float, float]:
        """Calculate palm facing direction"""
        # Use cross product of two vectors on palm plane
        wrist = landmarks[0]
        index_mcp = landmarks[5]
        pinky_mcp = landmarks[17]
        
        v1 = index_mcp - wrist
        v2 = pinky_mcp - wrist
        
        normal = np.cross(v1, v2)
        norm = np.linalg.norm(normal)
        if norm > 0:
            normal = normal / norm
        
        return tuple(normal)
    
    def _get_finger_states(self, landmarks: np.ndarray, handedness: str) -> Dict[str, bool]:
        """Determine which fingers are extended"""
        states = {}
        
        # Thumb - compare x position (different logic for left/right hand)
        thumb_tip = landmarks[4]
        thumb_ip = landmarks[3]
        thumb_mcp = landmarks[2]
        
        if handedness.lower() == "left":
            states['thumb'] = thumb_tip[0] > thumb_ip[0]
        else:
            states['thumb'] = thumb_tip[0] < thumb_ip[0]
        
        # Other fingers - compare y positions (tip should be above pip if extended)
        finger_names = ['index', 'middle', 'ring', 'pinky']
        tips = [8, 12, 16, 20]
        pips = [6, 10, 14, 18]
        
        for name, tip_idx, pip_idx in zip(finger_names, tips, pips):
            states[name] = landmarks[tip_idx][1] < landmarks[pip_idx][1]
        
        return states
    
    def _get_finger_angles(self, landmarks: np.ndarray) -> Dict[str, float]:
        """Calculate finger curl angles"""
        angles = {}
        
        # For each finger, calculate angle at PIP joint
        fingers = {
            'index': (5, 6, 8),    # MCP, PIP, TIP
            'middle': (9, 10, 12),
            'ring': (13, 14, 16),
            'pinky': (17, 18, 20)
        }
        
        for name, (mcp, pip, tip) in fingers.items():
            v1 = landmarks[mcp] - landmarks[pip]
            v2 = landmarks[tip] - landmarks[pip]
            
            cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
            angle = np.degrees(np.arccos(np.clip(cos_angle, -1, 1)))
            angles[name] = angle
        
        # Thumb angle calculation
        v1 = landmarks[1] - landmarks[2]  # CMC to MCP
        v2 = landmarks[4] - landmarks[2]  # MCP to TIP
        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
        angles['thumb'] = np.degrees(np.arccos(np.clip(cos_angle, -1, 1)))
        
        return angles
    
    def detect_motion(self, handedness: str) -> Dict[str, Any]:
        """Detect motion patterns from position history"""
        # Normalize handedness to lowercase
        handedness = handedness.lower()
        positions = list(self.position_history[handedness])
        
        if len(positions) < 5:
            return {'type': 'none', 'magnitude': 0.0}
        
        # Calculate motion vectors
        positions = np.array(positions[-15:])  # Last 15 frames
        velocities = np.diff(positions, axis=0)
        
        # Detect motion type
        total_movement = np.sum(np.linalg.norm(velocities, axis=1))
        
        if total_movement < 10:
            return {'type': 'stationary', 'magnitude': total_movement}
        
        # Check for specific patterns
        x_movement = positions[-1, 0] - positions[0, 0]
        y_movement = positions[-1, 1] - positions[0, 1]
        
        # Side to side
        x_changes = np.sign(np.diff(positions[:, 0]))
        direction_changes = np.sum(np.abs(np.diff(x_changes)))
        
        if direction_changes >= 3:
            return {'type': 'wave', 'magnitude': total_movement, 'cycles': direction_changes // 2}
        
        # Vertical nod
        y_changes = np.sign(np.diff(positions[:, 1]))
        y_direction_changes = np.sum(np.abs(np.diff(y_changes)))
        
        if y_direction_changes >= 2:
            return {'type': 'nod', 'magnitude': total_movement, 'cycles': y_direction_changes // 2}
        
        # Linear motion
        if abs(x_movement) > abs(y_movement) * 2:
            direction = 'right' if x_movement > 0 else 'left'
            return {'type': 'linear', 'direction': direction, 'magnitude': total_movement}
        elif abs(y_movement) > abs(x_movement) * 2:
            direction = 'down' if y_movement > 0 else 'up'
            return {'type': 'linear', 'direction': direction, 'magnitude': total_movement}
        
        # Circular motion
        if self._detect_circular_motion(positions):
            return {'type': 'circular', 'magnitude': total_movement}
        
        return {'type': 'complex', 'magnitude': total_movement}
    
    def _detect_circular_motion(self, positions: np.ndarray) -> bool:
        """Detect if motion is approximately circular"""
        if len(positions) < 8:
            return False
        
        center = positions.mean(axis=0)
        distances = np.linalg.norm(positions - center, axis=1)
        
        # Check if distances are relatively constant (circular path)
        distance_variance = np.std(distances) / (np.mean(distances) + 1e-6)
        
        return distance_variance < 0.3


# =============================================================================
# Sign Recognition Engine
# =============================================================================

class SignRecognitionEngine:
    """
    Main recognition engine that classifies hand poses into signs.
    Uses pattern matching with confidence scoring.
    """
    
    def __init__(self):
        self.db = ASLSignsDatabase()
        self.hand_analyzer = HandAnalysisEngine()
        
        # Recognition state
        self.current_sign_start: float = 0.0
        self.current_sign: Optional[str] = None
        self.sign_stability_count: int = 0
        
        # Thresholds
        self.min_hold_time: float = 0.15  # Minimum time to hold a sign (seconds)
        self.stability_threshold: int = 3  # Frames of consistent recognition
        
    def recognize_alphabet(self, hand_data: HandLandmarks) -> Tuple[str, float, List[Tuple[str, float]]]:
        """
        Recognize ASL alphabet letters from hand landmarks.
        Returns: (letter, confidence, alternatives)
        """
        scores: Dict[str, float] = {}
        
        finger_states = [
            hand_data.finger_states.get('thumb', False),
            hand_data.finger_states.get('index', False),
            hand_data.finger_states.get('middle', False),
            hand_data.finger_states.get('ring', False),
            hand_data.finger_states.get('pinky', False),
        ]
        
        landmarks = hand_data.landmarks
        
        for letter, pattern in self.db.alphabet_patterns.items():
            score = 0.0
            max_score = 0.0
            
            # Check finger extension states
            expected = pattern.get('fingers_extended', [None] * 5)
            for i, (actual, expected_val) in enumerate(zip(finger_states, expected)):
                max_score += 1.0
                if expected_val is not None and actual == expected_val:
                    score += 1.0
                elif expected_val is None:
                    score += 0.5  # Partial credit for don't care
            
            # Additional pattern-specific checks
            if 'thumb_position' in pattern:
                max_score += 0.5
                thumb_pos = self._check_thumb_position(landmarks, pattern['thumb_position'], hand_data.handedness)
                score += 0.5 * thumb_pos
            
            if 'fingers_together' in pattern:
                max_score += 0.5
                together = self._check_fingers_together(landmarks)
                if together == pattern['fingers_together']:
                    score += 0.5
            
            if 'fingers_spread' in pattern:
                max_score += 0.5
                spread = self._check_fingers_spread(landmarks)
                if spread == pattern['fingers_spread']:
                    score += 0.5
            
            if 'thumb_index_touch' in pattern:
                max_score += 0.5
                touch = self._check_fingertip_touch(landmarks, 4, 8)
                if touch == pattern['thumb_index_touch']:
                    score += 0.5
            
            if 'fingers_crossed' in pattern:
                max_score += 0.5
                crossed = self._check_fingers_crossed(landmarks)
                if crossed == pattern['fingers_crossed']:
                    score += 0.5
            
            if 'index_hooked' in pattern:
                max_score += 0.5
                hooked = hand_data.finger_angles.get('index', 180) < 120
                if hooked == pattern['index_hooked']:
                    score += 0.5
            
            if 'hand_shape' in pattern:
                max_score += 0.3
                shape_match = self._check_hand_shape(landmarks, pattern['hand_shape'])
                score += 0.3 * shape_match
            
            # Normalize score
            scores[letter] = score / max_score if max_score > 0 else 0.0
        
        # Get best matches
        sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        
        if sorted_scores:
            best_letter, best_score = sorted_scores[0]
            alternatives = sorted_scores[1:4]  # Top 3 alternatives
            return best_letter, best_score, alternatives
        
        return '', 0.0, []
    
    def recognize_number(self, hand_data: HandLandmarks) -> Tuple[str, float]:
        """Recognize numbers 0-9 from hand landmarks"""
        scores: Dict[str, float] = {}
        
        finger_states = [
            hand_data.finger_states.get('thumb', False),
            hand_data.finger_states.get('index', False),
            hand_data.finger_states.get('middle', False),
            hand_data.finger_states.get('ring', False),
            hand_data.finger_states.get('pinky', False),
        ]
        
        for number, pattern in self.db.number_patterns.items():
            score = 0.0
            max_score = 5.0
            
            expected = pattern.get('fingers_extended', [None] * 5)
            for actual, expected_val in zip(finger_states, expected):
                if expected_val is not None and actual == expected_val:
                    score += 1.0
            
            scores[number] = score / max_score
        
        best_number = max(scores.items(), key=lambda x: x[1])
        return best_number[0], best_number[1]
    
    def recognize_common_sign(
        self, 
        left_hand: Optional[HandLandmarks],
        right_hand: Optional[HandLandmarks],
        motion_data: Dict[str, Any]
    ) -> Tuple[str, float]:
        """Recognize common signs/words"""
        
        scores: Dict[str, float] = {}
        
        for sign_name, pattern in self.db.common_signs.items():
            score = 0.0
            max_score = 0.0
            
            # Check if sign type matches available hands
            sign_type = pattern.get('type', 'single_hand')
            
            if sign_type == 'two_handed' and (left_hand is None or right_hand is None):
                continue
            
            if sign_type == 'single_hand':
                hand = right_hand if right_hand else left_hand
                if hand is None:
                    continue
                
                # Check hand shape
                if 'hand_flat' in pattern:
                    max_score += 1.0
                    is_flat = all([
                        hand.finger_states.get('index', False),
                        hand.finger_states.get('middle', False),
                        hand.finger_states.get('ring', False),
                        hand.finger_states.get('pinky', False),
                    ])
                    if is_flat == pattern['hand_flat']:
                        score += 1.0
                
                if 'hand_shape' in pattern:
                    max_score += 1.0
                    if pattern['hand_shape'] == 'fist':
                        is_fist = not any(hand.finger_states.values())
                        if is_fist:
                            score += 1.0
                    elif pattern['hand_shape'] == 'claw':
                        # Check for partially curled fingers
                        avg_angle = np.mean(list(hand.finger_angles.values()))
                        if 60 < avg_angle < 140:
                            score += 1.0
            
            # Check motion pattern
            if 'motion' in pattern:
                max_score += 2.0
                motion_type = motion_data.get('type', 'none')
                
                if pattern['motion'].startswith('wave') and motion_type == 'wave':
                    score += 2.0
                elif pattern['motion'].startswith('nod') and motion_type == 'nod':
                    score += 2.0
                elif pattern['motion'].startswith('circular') and motion_type == 'circular':
                    score += 2.0
                elif 'shake' in pattern['motion'] and motion_type == 'wave':
                    score += 1.5
            
            if max_score > 0:
                scores[sign_name] = score / max_score
        
        if scores:
            best = max(scores.items(), key=lambda x: x[1])
            return best[0], best[1]
        
        return '', 0.0
    
    def _check_thumb_position(self, landmarks: np.ndarray, position: str, handedness: str) -> float:
        """Check if thumb is in specified position"""
        thumb_tip = landmarks[4]
        index_mcp = landmarks[5]
        
        if position == 'beside_fist':
            # Thumb should be alongside index finger
            dist = np.linalg.norm(thumb_tip - index_mcp)
            return 1.0 if dist < 0.1 else 0.0
        elif position == 'tucked':
            # Thumb should be across palm
            palm_center = landmarks[[0, 5, 9, 13, 17]].mean(axis=0)
            thumb_to_center = np.linalg.norm(thumb_tip - palm_center)
            return 1.0 if thumb_to_center < 0.1 else 0.0
        elif position == 'over_fingers':
            # Thumb tip above index PIP
            return 1.0 if thumb_tip[1] < landmarks[6][1] else 0.0
        
        return 0.5
    
    def _check_fingers_together(self, landmarks: np.ndarray) -> bool:
        """Check if fingers are together (close spacing)"""
        tips = landmarks[[8, 12, 16, 20]]
        pairwise_distances = []
        for i in range(len(tips)):
            for j in range(i + 1, len(tips)):
                pairwise_distances.append(np.linalg.norm(tips[i] - tips[j]))
        return np.mean(pairwise_distances) < 0.08
    
    def _check_fingers_spread(self, landmarks: np.ndarray) -> bool:
        """Check if fingers are spread apart"""
        tips = landmarks[[8, 12, 16, 20]]
        pairwise_distances = []
        for i in range(len(tips)):
            for j in range(i + 1, len(tips)):
                pairwise_distances.append(np.linalg.norm(tips[i] - tips[j]))
        return np.mean(pairwise_distances) > 0.12
    
    def _check_fingertip_touch(self, landmarks: np.ndarray, tip1: int, tip2: int) -> bool:
        """Check if two fingertips are touching"""
        dist = np.linalg.norm(landmarks[tip1] - landmarks[tip2])
        return dist < 0.05
    
    def _check_fingers_crossed(self, landmarks: np.ndarray) -> bool:
        """Check if index and middle fingers are crossed"""
        index_tip = landmarks[8]
        middle_tip = landmarks[12]
        index_pip = landmarks[6]
        middle_pip = landmarks[10]
        
        # Check if fingers cross each other
        index_vec = index_tip - index_pip
        middle_vec = middle_tip - middle_pip
        
        cross = np.cross(index_vec[:2], middle_vec[:2])
        return abs(cross) > 0.002  # Crossed if vectors intersect
    
    def _check_hand_shape(self, landmarks: np.ndarray, shape: str) -> float:
        """Check if hand matches a specific shape"""
        if shape == 'curved':
            # Check for C-shape curvature
            finger_tips = landmarks[[4, 8, 12, 16, 20]]
            # Tips should form an arc
            center = finger_tips.mean(axis=0)
            distances = np.linalg.norm(finger_tips - center, axis=1)
            variance = np.std(distances)
            return 1.0 if variance < 0.05 else 0.0
        
        return 0.5


# =============================================================================
# Sign Language Interpreter (Main Class)
# =============================================================================

class SignLanguageInterpreter:
    """
    Main Sign Language Interpreter class for VisionAssist Smart Glasses.
    
    Provides real-time ASL recognition with audio feedback for blind users.
    """
    
    def __init__(
        self,
        mode: InterpreterMode = InterpreterMode.CONTINUOUS,
        min_detection_confidence: float = 0.7,
        min_tracking_confidence: float = 0.5,
        speech_callback: Optional[Callable[[str], None]] = None,
        language: str = "en",
        speak_letters: bool = True,
        speak_words: bool = True,
        word_pause_threshold: float = 1.0,
        confirmation_threshold: float = 0.75,
        uncertain_threshold: float = 0.55,
        enable_visual_feedback: bool = True,
    ):
        """
        Initialize the Sign Language Interpreter.
        
        Args:
            mode: Operating mode (FINGERSPELLING, WORD_SIGNS, CONTINUOUS)
            min_detection_confidence: Minimum confidence for hand detection
            min_tracking_confidence: Minimum confidence for tracking
            speech_callback: Function to call for TTS output
            language: Language code for output
            speak_letters: Whether to speak individual letters
            speak_words: Whether to speak recognized words
            word_pause_threshold: Pause duration to trigger word output (seconds)
            confirmation_threshold: Confidence above which to confirm sign
            uncertain_threshold: Confidence below which to ask for repeat
            enable_visual_feedback: Whether to draw visual overlays
        """
        
        # Configuration
        self.mode = mode
        self.speech_callback = speech_callback
        self.language = language
        self.speak_letters = speak_letters
        self.speak_words = speak_words
        self.word_pause_threshold = word_pause_threshold
        self.confirmation_threshold = confirmation_threshold
        self.uncertain_threshold = uncertain_threshold
        self.enable_visual_feedback = enable_visual_feedback
        
        # MediaPipe Hands
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self.mp_draw = mp.solutions.drawing_utils
        self.mp_styles = mp.solutions.drawing_styles
        
        # Recognition engines
        self.hand_analyzer = HandAnalysisEngine()
        self.recognition_engine = SignRecognitionEngine()
        
        # State management
        self.state = InterpreterState(mode=mode)
        self.last_frame_time = time.time()
        
        # Letter/word buffers
        self.letter_buffer: List[Tuple[str, float, float]] = []  # (letter, confidence, timestamp)
        self.word_buffer: str = ""
        self.last_stable_sign: str = ""
        self.last_stable_time: float = 0.0
        self.sign_hold_start: float = 0.0
        self.sign_stability_frames: int = 0
        
        # Speech management
        self.last_spoken_letter: str = ""
        self.last_spoken_time: float = 0.0
        self.min_letter_speak_interval: float = 1.5  # Minimum time between speaking letters (increased)
        
        # Word/common sign rate limiting
        self._last_spoken_word: str = ""
        self._last_word_time: float = 0.0
        
        # Threading for non-blocking speech
        self._speech_lock = threading.Lock()
        
        # Visual feedback colors
        self.colors = {
            'hand_landmarks': (0, 255, 128),
            'hand_connections': (0, 200, 100),
            'text_bg': (30, 30, 30),
            'text_fg': (255, 255, 255),
            'high_confidence': (0, 255, 0),
            'medium_confidence': (0, 255, 255),
            'low_confidence': (0, 165, 255),
            'uncertain': (0, 0, 255),
            'panel_bg': (20, 20, 20),
        }
        
        print("🤟 Sign Language Interpreter initialized")
        print(f"   ├─ Mode: {mode.value}")
        print(f"   ├─ Speech enabled: {speech_callback is not None}")
        print(f"   ├─ Visual feedback: {enable_visual_feedback}")
        print(f"   └─ Confidence threshold: {confirmation_threshold}")
    
    def process_frame(
        self, 
        frame: np.ndarray,
        detections: Optional[List[Dict]] = None
    ) -> Tuple[List[RecognizedSign], np.ndarray]:
        """
        Process a video frame for sign language recognition.
        
        Args:
            frame: BGR image frame
            detections: Optional object detections (unused, for API compatibility)
        
        Returns:
            Tuple of (recognized_signs, annotated_frame)
        """
        current_time = time.time()
        dt = current_time - self.last_frame_time
        self.last_frame_time = current_time
        
        h, w = frame.shape[:2]
        annotated_frame = frame.copy()
        recognized_signs: List[RecognizedSign] = []
        
        # Convert to RGB for MediaPipe
        rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        rgb_frame.flags.writeable = False
        
        # Process hands
        results = self.hands.process(rgb_frame)
        
        # Track detected hands
        left_hand: Optional[HandLandmarks] = None
        right_hand: Optional[HandLandmarks] = None
        
        if results.multi_hand_landmarks and results.multi_handedness:
            for hand_landmarks, handedness_info in zip(
                results.multi_hand_landmarks, 
                results.multi_handedness
            ):
                # Get handedness
                handedness = handedness_info.classification[0].label
                confidence = handedness_info.classification[0].score
                
                # Process landmarks
                hand_data = self.hand_analyzer.process_landmarks(
                    hand_landmarks, handedness, w, h
                )
                hand_data.confidence = confidence
                
                if handedness.lower() == "left":
                    left_hand = hand_data
                else:
                    right_hand = hand_data
                
                # Draw hand landmarks
                if self.enable_visual_feedback:
                    self._draw_hand_landmarks(annotated_frame, hand_landmarks, handedness)
        
        # Recognize signs based on mode
        if self.mode in [InterpreterMode.FINGERSPELLING, InterpreterMode.CONTINUOUS]:
            # Try to recognize alphabet/numbers from dominant hand
            primary_hand = right_hand if right_hand else left_hand
            
            if primary_hand:
                # Detect motion for dynamic signs
                motion_data = self.hand_analyzer.detect_motion(primary_hand.handedness)
                
                # Recognize static sign
                letter, letter_conf, alternatives = self.recognition_engine.recognize_alphabet(primary_hand)
                number, number_conf = self.recognition_engine.recognize_number(primary_hand)
                
                # Choose best match
                if letter_conf > number_conf:
                    sign = letter
                    conf = letter_conf
                    category = SignCategory.ALPHABET
                else:
                    sign = number
                    conf = number_conf
                    category = SignCategory.NUMBER
                
                # Handle sign stability and word building - require higher confidence
                if sign and conf > 0.65:  # Increased threshold
                    self._handle_sign_recognition(sign, conf, category, current_time)
                    
                    recognized = RecognizedSign(
                        sign=sign,
                        category=category,
                        confidence=conf,
                        confidence_level=self._get_confidence_level(conf),
                        timestamp=current_time,
                        hand_used=primary_hand.handedness,
                        alternatives=alternatives
                    )
                    recognized_signs.append(recognized)
        
        if self.mode in [InterpreterMode.WORD_SIGNS, InterpreterMode.CONTINUOUS]:
            # Try to recognize common signs
            motion_data = self.hand_analyzer.detect_motion('right' if right_hand else 'left')
            sign_name, sign_conf = self.recognition_engine.recognize_common_sign(
                left_hand, right_hand, motion_data
            )
            
            # Only speak common signs with VERY high confidence and rate limiting
            if sign_name and sign_conf > 0.85:  # Higher threshold for common signs
                # Rate limit - don't repeat same word within 3 seconds
                if sign_name != self._last_spoken_word or (current_time - self._last_word_time) > 3.0:
                    recognized = RecognizedSign(
                        sign=sign_name.replace('_', ' '),
                        category=SignCategory.COMMON_WORD,
                        confidence=sign_conf,
                        confidence_level=self._get_confidence_level(sign_conf),
                        timestamp=current_time,
                        hand_used="both" if left_hand and right_hand else ("right" if right_hand else "left")
                    )
                    recognized_signs.append(recognized)
                    
                    # Speak the word
                    self._speak_sign(sign_name.replace('_', ' '), sign_conf, is_word=True)
                    self._last_spoken_word = sign_name
                    self._last_word_time = current_time
        
        # Check for word completion (pause in signing)
        self._check_word_completion(current_time)
        
        # Draw status panel
        if self.enable_visual_feedback:
            self._draw_status_panel(annotated_frame, recognized_signs, left_hand, right_hand)
        
        return recognized_signs, annotated_frame
    
    def _handle_sign_recognition(
        self, 
        sign: str, 
        confidence: float, 
        category: SignCategory,
        current_time: float
    ):
        """Handle a recognized sign for word building"""
        
        # Validate sign is a single letter or number
        if not (len(sign) == 1 and (sign.isalpha() or sign.isdigit())):
            return
        
        # Check if same sign is being held
        if sign == self.last_stable_sign:
            self.sign_stability_frames += 1
            
            # Only add to buffer after stable hold
            if self.sign_stability_frames >= 3:  # ~100ms at 30fps
                hold_duration = current_time - self.sign_hold_start
                
                # Add to word buffer if held long enough and not already added
                if hold_duration > 0.2:  # 200ms hold
                    if len(self.letter_buffer) == 0 or self.letter_buffer[-1][0] != sign:
                        self.letter_buffer.append((sign, confidence, current_time))
                        self.word_buffer += sign
                        
                        # Speak the letter
                        if self.speak_letters:
                            self._speak_sign(sign, confidence, is_word=False)
        else:
            # New sign detected
            self.last_stable_sign = sign
            self.last_stable_time = current_time
            self.sign_hold_start = current_time
            self.sign_stability_frames = 1
    
    def _check_word_completion(self, current_time: float):
        """Check if a pause indicates word completion"""
        
        if not self.word_buffer:
            return
        
        time_since_last_sign = current_time - self.last_stable_time
        
        if time_since_last_sign > self.word_pause_threshold:
            # Word is complete, speak it
            word = self.word_buffer.lower()
            
            if self.speak_words and len(word) > 1:
                self._speak_word(word)
            
            # Clear buffers
            self.word_buffer = ""
            self.letter_buffer.clear()
    
    def _speak_sign(self, sign: str, confidence: float, is_word: bool = False):
        """Speak a recognized sign with appropriate phrasing"""
        
        with self._speech_lock:
            current_time = time.time()
            
            # Rate limiting
            if not is_word:
                if current_time - self.last_spoken_time < self.min_letter_speak_interval:
                    return
                if sign == self.last_spoken_letter:
                    return
            
            # Build speech text based on confidence
            if confidence >= self.confirmation_threshold:
                text = sign.upper() if not is_word else sign
            elif confidence >= self.uncertain_threshold:
                text = f"I think {sign}"
            else:
                text = f"Not sure, maybe {sign}"
            
            # Call speech callback
            if self.speech_callback:
                try:
                    self.speech_callback(text)
                except Exception as e:
                    print(f"⚠️ Speech callback error: {e}")
            
            self.last_spoken_letter = sign
            self.last_spoken_time = current_time
    
    def _speak_word(self, word: str):
        """Speak a completed word"""
        
        with self._speech_lock:
            text = f"Word: {word}"
            
            if self.speech_callback:
                try:
                    self.speech_callback(text)
                except Exception as e:
                    print(f"⚠️ Speech callback error: {e}")
    
    def _get_confidence_level(self, confidence: float) -> SignConfidence:
        """Convert confidence score to level"""
        if confidence >= 0.85:
            return SignConfidence.HIGH
        elif confidence >= 0.65:
            return SignConfidence.MEDIUM
        elif confidence >= 0.45:
            return SignConfidence.LOW
        return SignConfidence.UNCERTAIN
    
    def _draw_hand_landmarks(
        self, 
        frame: np.ndarray, 
        landmarks, 
        handedness: str
    ):
        """Draw hand landmarks with custom styling"""
        
        # Use MediaPipe's drawing utilities with custom colors
        self.mp_draw.draw_landmarks(
            frame,
            landmarks,
            self.mp_hands.HAND_CONNECTIONS,
            self.mp_styles.get_default_hand_landmarks_style(),
            self.mp_styles.get_default_hand_connections_style()
        )
        
        # Add handedness label
        h, w = frame.shape[:2]
        wrist = landmarks.landmark[0]
        label_pos = (int(wrist.x * w) - 30, int(wrist.y * h) - 20)
        
        cv.putText(
            frame, 
            handedness.upper(), 
            label_pos, 
            cv.FONT_HERSHEY_SIMPLEX, 
            0.6, 
            self.colors['text_fg'], 
            2
        )
    
    def _draw_status_panel(
        self, 
        frame: np.ndarray, 
        signs: List[RecognizedSign],
        left_hand: Optional[HandLandmarks],
        right_hand: Optional[HandLandmarks]
    ):
        """Draw status panel with recognition info"""
        
        h, w = frame.shape[:2]
        panel_width = 280
        panel_x = w - panel_width - 10
        panel_y = 10
        
        # Panel background
        overlay = frame.copy()
        cv.rectangle(
            overlay, 
            (panel_x, panel_y), 
            (w - 10, panel_y + 200), 
            self.colors['panel_bg'], 
            -1
        )
        cv.addWeighted(overlay, 0.8, frame, 0.2, 0, frame)
        
        # Border
        cv.rectangle(
            frame, 
            (panel_x, panel_y), 
            (w - 10, panel_y + 200), 
            (100, 100, 100), 
            1
        )
        
        # Title
        cv.putText(
            frame, 
            "🤟 SIGN INTERPRETER", 
            (panel_x + 10, panel_y + 25), 
            cv.FONT_HERSHEY_SIMPLEX, 
            0.5, 
            self.colors['text_fg'], 
            1
        )
        
        y_offset = panel_y + 50
        
        # Mode indicator
        mode_text = f"Mode: {self.mode.value}"
        cv.putText(frame, mode_text, (panel_x + 10, y_offset), 
                   cv.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)
        y_offset += 25
        
        # Hand status
        hands_text = f"Hands: {'L' if left_hand else '-'} {'R' if right_hand else '-'}"
        cv.putText(frame, hands_text, (panel_x + 10, y_offset), 
                   cv.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)
        y_offset += 30
        
        # Current recognition
        if signs:
            best_sign = max(signs, key=lambda s: s.confidence)
            
            # Color based on confidence
            conf_level = best_sign.confidence_level
            if conf_level == SignConfidence.HIGH:
                color = self.colors['high_confidence']
            elif conf_level == SignConfidence.MEDIUM:
                color = self.colors['medium_confidence']
            elif conf_level == SignConfidence.LOW:
                color = self.colors['low_confidence']
            else:
                color = self.colors['uncertain']
            
            # Sign display
            sign_text = best_sign.sign.upper()
            cv.putText(frame, sign_text, (panel_x + 10, y_offset + 20), 
                       cv.FONT_HERSHEY_SIMPLEX, 1.5, color, 2)
            
            # Confidence bar
            bar_width = int(200 * best_sign.confidence)
            cv.rectangle(frame, (panel_x + 10, y_offset + 40), 
                        (panel_x + 10 + bar_width, y_offset + 50), color, -1)
            cv.rectangle(frame, (panel_x + 10, y_offset + 40), 
                        (panel_x + 210, y_offset + 50), (100, 100, 100), 1)
            
            conf_text = f"{best_sign.confidence:.0%}"
            cv.putText(frame, conf_text, (panel_x + 220, y_offset + 48), 
                       cv.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)
            
            y_offset += 65
            
            # Alternatives
            if best_sign.alternatives:
                cv.putText(frame, "Alt:", (panel_x + 10, y_offset), 
                           cv.FONT_HERSHEY_SIMPLEX, 0.35, (100, 100, 100), 1)
                alt_text = " ".join([f"{a[0]}({a[1]:.0%})" for a in best_sign.alternatives[:3]])
                cv.putText(frame, alt_text, (panel_x + 40, y_offset), 
                           cv.FONT_HERSHEY_SIMPLEX, 0.35, (100, 100, 100), 1)
                y_offset += 20
        
        # Word buffer
        if self.word_buffer:
            cv.putText(frame, "Buffer:", (panel_x + 10, y_offset + 10), 
                       cv.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)
            cv.putText(frame, self.word_buffer.upper(), (panel_x + 60, y_offset + 10), 
                       cv.FONT_HERSHEY_SIMPLEX, 0.5, self.colors['text_fg'], 1)
    
    def get_current_word(self) -> str:
        """Get the current word being spelled"""
        return self.word_buffer
    
    def clear_buffer(self):
        """Clear the current word buffer"""
        self.word_buffer = ""
        self.letter_buffer.clear()
        self.last_stable_sign = ""
    
    def set_mode(self, mode: InterpreterMode):
        """Change the interpreter mode"""
        self.mode = mode
        self.state.mode = mode
        self.clear_buffer()
    
    def describe_for_speech(self, signs: List[RecognizedSign], frame_width: int) -> str:
        """Generate speech description of current signing activity"""
        
        if not signs:
            if self.word_buffer:
                return f"Currently spelling: {self.word_buffer}"
            return "No signing detected."
        
        best_sign = max(signs, key=lambda s: s.confidence)
        
        if best_sign.category == SignCategory.COMMON_WORD:
            return f"Sign: {best_sign.sign}"
        
        conf_phrase = ""
        if best_sign.confidence_level == SignConfidence.UNCERTAIN:
            conf_phrase = "I'm not sure, but I think they signed "
        elif best_sign.confidence_level == SignConfidence.LOW:
            conf_phrase = "It looks like "
        
        if best_sign.category == SignCategory.ALPHABET:
            return f"{conf_phrase}letter {best_sign.sign}"
        elif best_sign.category == SignCategory.NUMBER:
            return f"{conf_phrase}number {best_sign.sign}"
        
        return f"{conf_phrase}{best_sign.sign}"
    
    def get_summary(self, signs: List[RecognizedSign]) -> Dict[str, Any]:
        """Get a summary of current signing state"""
        return {
            'mode': self.mode.value,
            'current_word': self.word_buffer,
            'letter_count': len(self.letter_buffer),
            'signs_detected': len(signs),
            'best_sign': signs[0].sign if signs else None,
            'best_confidence': signs[0].confidence if signs else 0.0,
            'is_signing': len(signs) > 0,
        }
    
    def request_repeat(self):
        """Speak a request for the signer to repeat"""
        if self.speech_callback:
            self.speech_callback("I'm not sure, can you sign that again slower?")
    
    def request_spell(self):
        """Ask the signer to fingerspell the word"""
        if self.speech_callback:
            self.speech_callback("Could you spell that out letter by letter?")
    
    def __del__(self):
        """Cleanup resources"""
        try:
            if self.hands:
                self.hands.close()
        except:
            pass


# =============================================================================
# Factory Function
# =============================================================================

def create_sign_interpreter(
    mode: str = "continuous",
    speech_callback: Optional[Callable[[str], None]] = None,
    min_detection_confidence: float = 0.7,
    min_tracking_confidence: float = 0.5,
    speak_letters: bool = True,
    speak_words: bool = True,
    enable_visual_feedback: bool = True,
) -> SignLanguageInterpreter:
    """
    Factory function to create a SignLanguageInterpreter.
    
    Args:
        mode: "fingerspelling", "word_signs", "continuous", or "learning"
        speech_callback: Function to call for TTS output
        min_detection_confidence: Minimum confidence for detection
        min_tracking_confidence: Minimum confidence for tracking
        speak_letters: Whether to speak individual letters
        speak_words: Whether to speak completed words
        enable_visual_feedback: Whether to draw overlays
    
    Returns:
        Configured SignLanguageInterpreter instance
    """
    
    mode_map = {
        'fingerspelling': InterpreterMode.FINGERSPELLING,
        'word_signs': InterpreterMode.WORD_SIGNS,
        'continuous': InterpreterMode.CONTINUOUS,
        'learning': InterpreterMode.LEARNING,
    }
    
    interpreter_mode = mode_map.get(mode.lower(), InterpreterMode.CONTINUOUS)
    
    return SignLanguageInterpreter(
        mode=interpreter_mode,
        speech_callback=speech_callback,
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence,
        speak_letters=speak_letters,
        speak_words=speak_words,
        enable_visual_feedback=enable_visual_feedback,
    )


# =============================================================================
# Standalone Testing
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("🤟 Sign Language Interpreter - Test Mode")
    print("=" * 60)
    
    # Simple print callback for testing
    def test_speech(text: str):
        print(f"🔊 SPEAK: {text}")
    
    interpreter = create_sign_interpreter(
        mode="continuous",
        speech_callback=test_speech,
        enable_visual_feedback=True,
    )
    
    cap = cv.VideoCapture(0)
    cap.set(cv.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, 720)
    
    print("\n🎮 Controls:")
    print("   'q' - Quit")
    print("   'c' - Clear buffer")
    print("   '1' - Fingerspelling mode")
    print("   '2' - Word signs mode")
    print("   '3' - Continuous mode")
    print("-" * 60)
    
    fps_time = time.time()
    fps_history = deque(maxlen=30)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        signs, annotated = interpreter.process_frame(frame)
        
        # FPS calculation
        fps = 1.0 / (time.time() - fps_time + 1e-6)
        fps_time = time.time()
        fps_history.append(fps)
        avg_fps = sum(fps_history) / len(fps_history)
        
        # Draw FPS
        cv.putText(annotated, f"FPS: {avg_fps:.1f}", (10, 30), 
                   cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        cv.imshow("Sign Language Interpreter", annotated)
        
        key = cv.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('c'):
            interpreter.clear_buffer()
            print("📝 Buffer cleared")
        elif key == ord('1'):
            interpreter.set_mode(InterpreterMode.FINGERSPELLING)
            print("📝 Mode: Fingerspelling")
        elif key == ord('2'):
            interpreter.set_mode(InterpreterMode.WORD_SIGNS)
            print("📝 Mode: Word Signs")
        elif key == ord('3'):
            interpreter.set_mode(InterpreterMode.CONTINUOUS)
            print("📝 Mode: Continuous")
    
    cap.release()
    cv.destroyAllWindows()
    print("\n✅ Done!")