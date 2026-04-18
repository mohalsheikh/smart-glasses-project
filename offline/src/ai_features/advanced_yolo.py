"""
Enhanced YOLOv8 Detector with Pose, Segmentation, and Advanced Tracking
Leverages full YOLOv8 capabilities for maximum scene understanding
"""

from __future__ import annotations

from ultralytics import YOLO
import src.utils.config as config
import numpy as np
import cv2 as cv
from typing import Any, Optional, Tuple, List, Dict
from dataclasses import dataclass


@dataclass
class EnhancedDetection:
    """Enhanced detection with pose and segmentation data"""
    label: str
    confidence: float
    bbox: Tuple[int, int, int, int]
    center: Tuple[float, float]
    track_id: Optional[int] = None
    
    # Pose estimation (if person)
    keypoints: Optional[np.ndarray] = None  # 17 keypoints for COCO
    keypoint_confidence: Optional[np.ndarray] = None
    pose_action: Optional[str] = None  # sitting, standing, walking, etc.
    
    # Segmentation mask
    mask: Optional[np.ndarray] = None
    mask_area: Optional[float] = None
    
    # Distance estimation
    estimated_distance: Optional[float] = None
    size_ratio: Optional[float] = None


class AdvancedObjectDetector:
    """
    Enhanced YOLOv8 detector with:
    - Object detection and tracking
    - Pose estimation for people
    - Instance segmentation
    - Activity recognition
    - Distance estimation
    """
    
    # COCO keypoint indices
    NOSE = 0
    LEFT_EYE = 1
    RIGHT_EYE = 2
    LEFT_EAR = 3
    RIGHT_EAR = 4
    LEFT_SHOULDER = 5
    RIGHT_SHOULDER = 6
    LEFT_ELBOW = 7
    RIGHT_ELBOW = 8
    LEFT_WRIST = 9
    RIGHT_WRIST = 10
    LEFT_HIP = 11
    RIGHT_HIP = 12
    LEFT_KNEE = 13
    RIGHT_KNEE = 14
    LEFT_ANKLE = 15
    RIGHT_ANKLE = 16
    
    def __init__(
        self,
        detection_model: str = None,
        pose_model: str = "yolov8n-pose.pt",
        segmentation_model: str = None,
        conf: float = None,
        iou: float = None,
        imgsz: int = None,
        tracker: str = None,
        max_det: int = None,
        enable_pose: bool = True,
        enable_segmentation: bool = False,  # Can be heavy, optional
    ):
        # Load models
        detection_model = detection_model or config.DEFAULT_MODEL_NAME
        conf = conf if conf is not None else config.DEFAULT_YOLO_CONFIDENCE_THRESHOLD
        iou = iou if iou is not None else config.DEFAULT_IOU_THRESHOLD
        imgsz = imgsz if imgsz is not None else config.YOLO_INFERENCE_SIZE
        tracker = tracker or config.DEFAULT_TRACKER
        max_det = max_det if max_det is not None else config.DEFAULT_MAX_DETECTIONS
        
        self.conf = float(conf)
        self.iou = float(iou)
        self.imgsz = int(imgsz)
        self.tracker = tracker
        self.max_det = int(max_det)
        
        # Detection model
        try:
            self.detection_model = YOLO(detection_model)
            print(f"✅ Detection model loaded: {detection_model}")
        except Exception as e:
            raise RuntimeError(f"Failed to load detection model: {e}")
        
        # Pose model (for person activity recognition)
        self.enable_pose = enable_pose
        self.pose_model = None
        if enable_pose:
            try:
                self.pose_model = YOLO(pose_model)
                print(f"✅ Pose model loaded: {pose_model}")
            except Exception as e:
                print(f"⚠️ Pose model loading failed: {e}")
                self.enable_pose = False
        
        # Segmentation model (optional, heavier)
        self.enable_segmentation = enable_segmentation
        self.seg_model = None
        if enable_segmentation and segmentation_model:
            try:
                self.seg_model = YOLO(segmentation_model)
                print(f"✅ Segmentation model loaded: {segmentation_model}")
            except Exception as e:
                print(f"⚠️ Segmentation model loading failed: {e}")
                self.enable_segmentation = False
        
        # Activity recognition helpers
        self.prev_person_keypoints = {}  # track_id -> keypoints for motion detection
    
    def _tensor_to_numpy(self, obj):
        """Convert tensor to numpy safely"""
        return obj.cpu().numpy() if obj is not None else None
    
    def detect_with_tracking(self, frame: np.ndarray) -> Tuple[List[EnhancedDetection], np.ndarray]:
        """Detect objects with tracking"""
        try:
            results = self.detection_model.track(
                source=frame,
                persist=True,
                conf=self.conf,
                iou=self.iou,
                imgsz=self.imgsz,
                tracker=self.tracker,
                max_det=self.max_det,
                verbose=False,
            )[0]
        except Exception as e:
            raise RuntimeError(f"Detection failed: {e}")
        
        boxes = getattr(results, "boxes", None)
        if boxes is None:
            return [], frame
        
        xyxy = self._tensor_to_numpy(getattr(boxes, "xyxy", None))
        if xyxy is None or xyxy.size == 0:
            return [], frame
        
        centers = (xyxy[:, :2] + xyxy[:, 2:]) / 2.0
        conf = self._tensor_to_numpy(getattr(boxes, "conf", None))
        cls = self._tensor_to_numpy(getattr(boxes, "cls", None))
        ids = self._tensor_to_numpy(getattr(boxes, "id", None))
        
        conf = conf.astype(float) if conf is not None else np.zeros((len(xyxy),), dtype=float)
        cls = cls.astype(int) if cls is not None else np.zeros((len(xyxy),), dtype=int)
        
        # Get labels
        labels = []
        for c in cls:
            name = self.detection_model.names.get(int(c))
            labels.append(name if name is not None else str(int(c)))
        
        detections = []
        for i in range(len(xyxy)):
            x1, y1, x2, y2 = xyxy[i]
            bbox = (int(x1), int(y1), int(x2), int(y2))
            cx, cy = centers[i]
            track_id = int(ids[i]) if ids is not None and i < len(ids) else None
            
            detection = EnhancedDetection(
                label=labels[i],
                confidence=float(conf[i]),
                bbox=bbox,
                center=(float(cx), float(cy)),
                track_id=track_id,
            )
            
            # Calculate size ratio for distance estimation
            bbox_area = (x2 - x1) * (y2 - y1)
            frame_area = frame.shape[0] * frame.shape[1]
            detection.size_ratio = float(bbox_area / frame_area)
            
            detections.append(detection)
        
        return detections, results.plot()
    
    def detect_poses(self, frame: np.ndarray) -> Dict[int, np.ndarray]:
        """Detect poses for all people in frame"""
        if not self.enable_pose or self.pose_model is None:
            return {}
        
        try:
            results = self.pose_model(frame, verbose=False)[0]
            
            keypoints_data = getattr(results, "keypoints", None)
            if keypoints_data is None:
                return {}
            
            # Extract keypoints
            kpts = self._tensor_to_numpy(getattr(keypoints_data, "xy", None))
            kpts_conf = self._tensor_to_numpy(getattr(keypoints_data, "conf", None))
            
            if kpts is None:
                return {}
            
            poses = {}
            for i in range(len(kpts)):
                poses[i] = {
                    "keypoints": kpts[i],
                    "confidence": kpts_conf[i] if kpts_conf is not None else None
                }
            
            return poses
            
        except Exception as e:
            print(f"⚠️ Pose detection error: {e}")
            return {}
    
    def recognize_activity(self, keypoints: np.ndarray, confidence: Optional[np.ndarray] = None) -> str:
        """Recognize activity from pose keypoints"""
        try:
            if keypoints is None or len(keypoints) < 17:
                return "unknown"
            
            # Filter low confidence keypoints
            if confidence is not None:
                valid_mask = confidence > 0.5
                if not np.any(valid_mask):
                    return "unknown"
            
            # Extract key points
            left_shoulder = keypoints[self.LEFT_SHOULDER]
            right_shoulder = keypoints[self.RIGHT_SHOULDER]
            left_hip = keypoints[self.LEFT_HIP]
            right_hip = keypoints[self.RIGHT_HIP]
            left_knee = keypoints[self.LEFT_KNEE]
            right_knee = keypoints[self.RIGHT_KNEE]
            
            # Calculate angles and positions
            # Torso angle (vertical = standing, horizontal = lying)
            shoulder_mid = (left_shoulder + right_shoulder) / 2
            hip_mid = (left_hip + right_hip) / 2
            
            torso_angle = np.degrees(np.arctan2(
                hip_mid[1] - shoulder_mid[1],
                hip_mid[0] - shoulder_mid[0]
            ))
            
            # Knee angles
            def angle_between_points(p1, p2, p3):
                """Calculate angle at p2"""
                v1 = p1 - p2
                v2 = p3 - p2
                cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
                return np.degrees(np.arccos(np.clip(cos_angle, -1, 1)))
            
            # Determine activity
            if abs(torso_angle - 90) < 30:  # Standing
                # Check if knees are bent
                left_knee_angle = angle_between_points(left_hip, left_knee, keypoints[self.LEFT_ANKLE])
                right_knee_angle = angle_between_points(right_hip, right_knee, keypoints[self.RIGHT_ANKLE])
                
                avg_knee_angle = (left_knee_angle + right_knee_angle) / 2
                
                if avg_knee_angle < 140:
                    return "sitting"
                else:
                    return "standing"
            elif abs(torso_angle) < 30:  # Horizontal
                return "lying_down"
            else:
                return "bending"
                
        except Exception as e:
            return "unknown"
    
    def enhance_detections_with_pose(
        self,
        frame: np.ndarray,
        detections: List[EnhancedDetection]
    ) -> List[EnhancedDetection]:
        """Add pose information to person detections"""
        if not self.enable_pose:
            return detections
        
        # Detect all poses
        poses = self.detect_poses(frame)
        
        # Match poses to person detections
        person_detections = [d for d in detections if d.label.lower() == "person"]
        
        for det in person_detections:
            # Find closest pose (simple matching by position)
            best_match = None
            min_dist = float('inf')
            
            for pose_id, pose_data in poses.items():
                kpts = pose_data["keypoints"]
                if kpts is None or len(kpts) < 1:
                    continue
                
                # Use nose or shoulder center as reference
                if confidence is not None and pose_data["confidence"][0] > 0.5:
                    pose_center = kpts[self.NOSE]
                else:
                    pose_center = (kpts[self.LEFT_SHOULDER] + kpts[self.RIGHT_SHOULDER]) / 2
                
                dist = np.linalg.norm(np.array(det.center) - pose_center)
                if dist < min_dist:
                    min_dist = dist
                    best_match = pose_data
            
            if best_match and min_dist < 100:  # Reasonable threshold
                det.keypoints = best_match["keypoints"]
                det.keypoint_confidence = best_match["confidence"]
                det.pose_action = self.recognize_activity(
                    det.keypoints,
                    det.keypoint_confidence
                )
        
        return detections
    
    def estimate_distance(self, detection: EnhancedDetection, focal_length: float = 500) -> float:
        """Estimate distance to object (rough approximation)"""
        # This is a simplified distance estimation
        # For person: assume average height of 1.7m
        if detection.label.lower() == "person":
            bbox_height = detection.bbox[3] - detection.bbox[1]
            if bbox_height > 0:
                # Simplified: distance = (real_height * focal_length) / pixel_height
                real_height = 1.7  # meters
                distance = (real_height * focal_length) / bbox_height
                return distance
        
        # For other objects, use size ratio heuristic
        if detection.size_ratio:
            # Very rough estimation: larger ratio = closer
            if detection.size_ratio > 0.3:
                return 1.0  # Close
            elif detection.size_ratio > 0.1:
                return 3.0  # Medium
            else:
                return 6.0  # Far
        
        return None
    
    def detect_complete(
        self,
        frame: np.ndarray,
        annotate: bool = True
    ) -> Tuple[List[EnhancedDetection], np.ndarray]:
        """Complete detection pipeline with all features"""
        # Basic detection with tracking
        detections, annotated = self.detect_with_tracking(frame)
        
        # Enhance with pose estimation
        detections = self.enhance_detections_with_pose(frame, detections)
        
        # Add distance estimates
        for det in detections:
            det.estimated_distance = self.estimate_distance(det)
        
        return detections, annotated if annotate else frame
    
    def describe_detections_enhanced(
        self,
        detections: List[EnhancedDetection],
        frame_width: int
    ) -> str:
        """Generate enhanced natural language description"""
        if not detections:
            return "No objects detected."
        
        descriptions = []
        
        # Prioritize people with pose information
        people = [d for d in detections if d.label.lower() == "person"]
        objects = [d for d in detections if d.label.lower() != "person"]
        
        # Describe people with activities
        for person in people[:3]:  # Limit to 3 people
            parts = []
            
            # Position
            x_center = person.center[0]
            if x_center < frame_width / 3:
                pos = "on your left"
            elif x_center > 2 * frame_width / 3:
                pos = "on your right"
            else:
                pos = "ahead"
            
            # Activity
            if person.pose_action and person.pose_action != "unknown":
                parts.append(f"person {person.pose_action} {pos}")
            else:
                parts.append(f"person {pos}")
            
            # Distance
            if person.estimated_distance:
                if person.estimated_distance < 2:
                    parts.append("very close")
                elif person.estimated_distance < 4:
                    parts.append("nearby")
            
            if parts:
                descriptions.append(", ".join(parts))
        
        # Describe other important objects
        priority_objects = [d for d in objects if d.confidence > 0.5]
        for obj in priority_objects[:5]:
            x_center = obj.center[0]
            if x_center < frame_width / 3:
                pos = "left"
            elif x_center > 2 * frame_width / 3:
                pos = "right"
            else:
                pos = "center"
            
            descriptions.append(f"{obj.label} ({pos})")
        
        if not descriptions:
            return "Scene unclear."
        
        return "I see: " + "; ".join(descriptions) + "."
