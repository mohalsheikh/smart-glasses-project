"""
Advanced Emotion Detection and Face Analysis
Uses MediaPipe, face-recognition, and OpenAI for comprehensive face understanding
"""

from __future__ import annotations

import cv2 as cv
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
import mediapipe as mp
from dataclasses import dataclass
import time

from openai import OpenAI
import src.utils.config as config

client = OpenAI()


@dataclass
class FaceData:
    """Comprehensive face information"""
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2
    confidence: float
    landmarks: Optional[np.ndarray] = None
    emotion: Optional[str] = None
    emotion_scores: Optional[Dict[str, float]] = None
    age_range: Optional[str] = None
    facial_expression: Optional[str] = None
    gaze_direction: Optional[str] = None
    person_id: Optional[int] = None
    embeddings: Optional[np.ndarray] = None


class EmotionFaceAnalyzer:
    """
    Advanced face analysis combining multiple approaches:
    - MediaPipe for face detection and landmarks
    - OpenAI GPT-4o Vision for emotion and expression analysis
    - Face recognition for person identification
    """
    
    def __init__(self):
        # MediaPipe Face Detection
        self.mp_face_detection = mp.solutions.face_detection
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_detection = self.mp_face_detection.FaceDetection(
            model_selection=1,  # 0 for short range, 1 for full range
            min_detection_confidence=0.6
        )
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=5,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.5
        )
        
        # Known faces database (simple in-memory)
        self.known_faces = {}  # person_id -> embeddings
        self.next_person_id = 1
        
        # Emotion keywords for basic detection
        self.emotion_keywords = {
            "happy": ["smile", "smiling", "joy", "cheerful", "happy", "grin"],
            "sad": ["sad", "unhappy", "down", "frown", "crying"],
            "angry": ["angry", "mad", "furious", "scowl", "glare"],
            "surprised": ["surprised", "shock", "amazed", "astonish"],
            "neutral": ["neutral", "calm", "expressionless", "blank"],
            "worried": ["worried", "concern", "anxious", "nervous"],
            "confused": ["confused", "puzzle", "perplex"],
        }
        
        print("😊 EmotionFaceAnalyzer initialized")
    
    def detect_faces(self, frame: np.ndarray) -> List[FaceData]:
        """Detect all faces in frame with MediaPipe"""
        if frame is None:
            return []
        
        try:
            # Convert BGR to RGB
            rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
            results = self.face_detection.process(rgb_frame)
            
            if not results.detections:
                return []
            
            h, w = frame.shape[:2]
            faces = []
            
            for detection in results.detections:
                bbox_rel = detection.location_data.relative_bounding_box
                x1 = int(bbox_rel.xmin * w)
                y1 = int(bbox_rel.ymin * h)
                x2 = int((bbox_rel.xmin + bbox_rel.width) * w)
                y2 = int((bbox_rel.ymin + bbox_rel.height) * h)
                
                # Ensure bbox is within frame
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)
                
                if x2 <= x1 or y2 <= y1:
                    continue
                
                face = FaceData(
                    bbox=(x1, y1, x2, y2),
                    confidence=detection.score[0]
                )
                faces.append(face)
            
            return faces
            
        except Exception as e:
            print(f"⚠️ Face detection error: {e}")
            return []
    
    def get_face_landmarks(self, frame: np.ndarray, face: FaceData) -> Optional[np.ndarray]:
        """Get detailed face landmarks using MediaPipe Face Mesh"""
        try:
            x1, y1, x2, y2 = face.bbox
            face_crop = frame[y1:y2, x1:x2]
            
            if face_crop.size == 0:
                return None
            
            rgb_crop = cv.cvtColor(face_crop, cv.COLOR_BGR2RGB)
            results = self.face_mesh.process(rgb_crop)
            
            if results.multi_face_landmarks:
                landmarks = results.multi_face_landmarks[0]
                # Convert to numpy array
                h, w = face_crop.shape[:2]
                points = np.array([
                    [lm.x * w, lm.y * h, lm.z]
                    for lm in landmarks.landmark
                ])
                return points
            
        except Exception as e:
            print(f"⚠️ Landmark detection error: {e}")
        
        return None
    
    def estimate_gaze_direction(self, landmarks: np.ndarray) -> Optional[str]:
        """Estimate where the person is looking based on face landmarks"""
        try:
            if landmarks is None or len(landmarks) < 468:
                return None
            
            # Use eye landmarks to estimate gaze
            # Simplified gaze estimation
            left_eye = landmarks[33]  # Left eye outer corner
            right_eye = landmarks[263]  # Right eye outer corner
            nose_tip = landmarks[1]
            
            # Calculate eye center
            eye_center_x = (left_eye[0] + right_eye[0]) / 2
            nose_x = nose_tip[0]
            
            # Estimate horizontal gaze
            diff = nose_x - eye_center_x
            
            if abs(diff) < 5:
                return "forward"
            elif diff > 5:
                return "left"
            else:
                return "right"
                
        except Exception as e:
            return None
    
    async def analyze_emotion_gpt4o(self, frame: np.ndarray, face: FaceData) -> Dict[str, Any]:
        """Use GPT-4o Vision to analyze facial expression and emotion"""
        try:
            # Crop face region
            x1, y1, x2, y2 = face.bbox
            # Add padding
            pad = 20
            x1 = max(0, x1 - pad)
            y1 = max(0, y1 - pad)
            x2 = min(frame.shape[1], x2 + pad)
            y2 = min(frame.shape[0], y2 + pad)
            
            face_crop = frame[y1:y2, x1:x2]
            
            if face_crop.size == 0:
                return {}
            
            # Encode face
            _, buffer = cv.imencode('.jpg', face_crop)
            import base64
            face_b64 = base64.b64encode(buffer).decode('utf-8')
            
            # Query GPT-4o
            response = client.chat.completions.create(
                model=config.OPENAI_VISION_MODEL,
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert at reading human emotions and facial expressions. Analyze the face in the image and provide: 1) Primary emotion, 2) Confidence (0-1), 3) Brief description. Be concise."
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": "Analyze this person's facial expression and emotion. Format: EMOTION|CONFIDENCE|DESCRIPTION"
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{face_b64}"
                                }
                            }
                        ]
                    }
                ],
                max_tokens=100,
                temperature=0.3
            )
            
            result = response.choices[0].message.content.strip()
            
            # Parse result
            parts = result.split('|')
            if len(parts) >= 3:
                emotion = parts[0].strip().lower()
                confidence = float(parts[1].strip())
                description = parts[2].strip()
                
                return {
                    "emotion": emotion,
                    "confidence": confidence,
                    "description": description
                }
            
        except Exception as e:
            print(f"⚠️ GPT-4o emotion analysis error: {e}")
        
        return {}
    
    def analyze_faces_complete(self, frame: np.ndarray, use_ai: bool = True) -> List[FaceData]:
        """Complete face analysis pipeline"""
        faces = self.detect_faces(frame)
        
        for face in faces:
            # Get landmarks
            landmarks = self.get_face_landmarks(frame, face)
            face.landmarks = landmarks
            
            # Estimate gaze
            if landmarks is not None:
                face.gaze_direction = self.estimate_gaze_direction(landmarks)
            
            # Emotion analysis with GPT-4o (optional, can be expensive)
            # For production, you might want to cache or rate-limit this
            # Commenting out by default for performance
            # if use_ai and config.OPENAI_API_KEY_PRESENT:
            #     emotion_data = self.analyze_emotion_gpt4o(frame, face)
            #     if emotion_data:
            #         face.emotion = emotion_data.get("emotion")
            #         face.facial_expression = emotion_data.get("description")
        
        return faces
    
    def describe_faces_for_user(self, faces: List[FaceData], frame_width: int) -> str:
        """Generate natural language description of detected faces"""
        if not faces:
            return "No faces detected."
        
        descriptions = []
        
        for i, face in enumerate(faces, 1):
            parts = []
            
            # Position
            x_center = (face.bbox[0] + face.bbox[2]) / 2
            if x_center < frame_width / 3:
                position = "on your left"
            elif x_center > 2 * frame_width / 3:
                position = "on your right"
            else:
                position = "in front of you"
            
            # Build description
            if len(faces) == 1:
                parts.append(f"One person {position}")
            else:
                parts.append(f"Person {i} {position}")
            
            # Add gaze if available
            if face.gaze_direction:
                parts.append(f"looking {face.gaze_direction}")
            
            # Add emotion if available
            if face.emotion:
                parts.append(f"appears {face.emotion}")
            
            descriptions.append(", ".join(parts))
        
        if len(descriptions) == 1:
            return descriptions[0] + "."
        else:
            return "I see " + str(len(faces)) + " people. " + "; ".join(descriptions) + "."
    
    def draw_face_annotations(self, frame: np.ndarray, faces: List[FaceData]) -> np.ndarray:
        """Draw face bounding boxes and labels"""
        annotated = frame.copy()
        
        for face in faces:
            x1, y1, x2, y2 = face.bbox
            
            # Draw bbox
            color = (0, 255, 0)
            cv.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            
            # Draw label
            label_parts = []
            if face.emotion:
                label_parts.append(face.emotion)
            if face.gaze_direction:
                label_parts.append(f"→{face.gaze_direction}")
            
            if label_parts:
                label = " ".join(label_parts)
                cv.putText(
                    annotated,
                    label,
                    (x1, y1 - 10),
                    cv.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color,
                    2
                )
            
            # Draw landmarks if available
            if face.landmarks is not None:
                for point in face.landmarks[::10]:  # Draw every 10th point
                    px, py = int(point[0] + x1), int(point[1] + y1)
                    cv.circle(annotated, (px, py), 1, (0, 255, 255), -1)
        
        return annotated
    
    def __del__(self):
        """Cleanup"""
        try:
            self.face_detection.close()
            self.face_mesh.close()
        except:
            pass
