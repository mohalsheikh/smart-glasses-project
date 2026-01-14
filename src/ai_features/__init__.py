"""
Advanced AI Features Module
Contains cutting-edge AI capabilities for smart glasses
"""

from .scene_memory import SceneMemoryEngine, MemoryEntry
from .emotion_analyzer import EmotionFaceAnalyzer, FaceData
from .advanced_yolo import AdvancedObjectDetector, EnhancedDetection
from .color_text_analyzer import ColorTextAnalyzer
from .proactive_assistant import ProactiveAssistant, ProactiveAlert
from .human_analyzer import (
    HumanAnalyzer,
    HumanData,
    FaceAnalysis,
    HandAnalysis,
    BodyPose,
    MotionData,
    InteractionData,
    Activity,
    Gesture,
    GazeDirection,
    Emotion,
    Engagement,
    Posture,
    DetectionQuality,
    create_human_analyzer,
)

__all__ = [
    'SceneMemoryEngine',
    'MemoryEntry',
    'EmotionFaceAnalyzer',
    'FaceData',
    'AdvancedObjectDetector',
    'EnhancedDetection',
    'ColorTextAnalyzer',
    'ProactiveAssistant',
    'ProactiveAlert',
    # Human Analyzer
    'HumanAnalyzer',
    'HumanData',
    'FaceAnalysis',
    'HandAnalysis',
    'BodyPose',
    'MotionData',
    'InteractionData',
    'Activity',
    'Gesture',
    'GazeDirection',
    'Emotion',
    'Engagement',
    'Posture',
    'DetectionQuality',
    'create_human_analyzer',
    # SignLanguage Interpreter
    'SignLanguageInterpreter',
    'SignCategory',
    'SignConfidence',
    'InterpreterMode',
    'RecognizedSign',
    'SignSequence',
    'HandLandmarks',
    'create_sign_interpreter',
]