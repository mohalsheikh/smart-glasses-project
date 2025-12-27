"""
Advanced YOLOv8 Object Detector with tracking and annotated output.
"""

from __future__ import annotations

from ultralytics import YOLO
import src.utils.config as config
import numpy as np
from typing import Any, Optional, Tuple, List, Dict


class ObjectDetector:
    # initializes the detector with specified parameters or defaults from config.
    def __init__(
        self,
        model_name: str = config.DEFAULT_MODEL_NAME,  # path to the YOLO model
        conf: float = config.DEFAULT_YOLO_CONFIDENCE_THRESHOLD,  # confidence threshold
        iou: float = config.DEFAULT_IOU_THRESHOLD,  # IoU threshold
        imgsz: int = (
            config.DEFAULT_FRAME_WIDTH
            if config.DEFAULT_FRAME_WIDTH > config.DEFAULT_FRAME_HEIGHT
            else config.DEFAULT_FRAME_HEIGHT
        ),
        tracker: str = config.DEFAULT_TRACKER,  # the tracker we're using
        max_det: int = config.DEFAULT_MAX_DETECTIONS,  # maximum number of objects to detect in a frame
    ):
        if model_name is None:
            raise ValueError("Model name must be set.")
        if conf is None:
            raise ValueError("Confidence threshold must be set.")
        if iou is None:
            raise ValueError("IoU threshold must be set.")
        if imgsz is None:
            raise ValueError("Image size must be set.")
        if tracker is None:
            raise ValueError("Tracker config must be set.")
        if max_det is None:
            raise ValueError("Max detections must be set.")

        self.conf = float(conf)
        self.iou = float(iou)
        self.imgsz = int(imgsz)
        self.tracker = tracker
        self.max_det = int(max_det)

        try:
            self.model = YOLO(model_name)
            print(f"✅ ObjectDetector loaded model: {model_name}")
        except Exception as e:
            raise RuntimeError(
                f"Failed to load model '{model_name}' with exception: {e}\n"
                "Make sure you specify a valid file path to a YOLO model."
            )

    def _track(self, frame: np.ndarray, persist: bool = True):
        return self.model.track(
            source=frame,
            persist=persist,
            conf=self.conf,
            iou=self.iou,
            imgsz=self.imgsz,
            tracker=self.tracker,
            max_det=self.max_det,
            verbose=False,
        )[0]

    @staticmethod
    def _tensor_to_numpy_array(obj):
        return obj.cpu().numpy() if obj is not None else None

    # returns tuple (detections, frame).
    # if annotate is True, frame is the annotated frame, otherwise it's the original frame.
    def detect(self, frame: np.ndarray, annotate: bool = False):
        try:
            track_result = self._track(frame)
        except Exception as e:
            raise RuntimeError(f"Tracking failed with exception: {e}")

        track_result_boxes = getattr(track_result, "boxes", None)
        if track_result_boxes is None:
            return [], frame

        xyxy = self._tensor_to_numpy_array(getattr(track_result_boxes, "xyxy", None))
        if xyxy is None or xyxy.size == 0:
            return [], frame

        center = (xyxy[:, :2] + xyxy[:, 2:]) / 2.0

        conf = self._tensor_to_numpy_array(getattr(track_result_boxes, "conf", None))
        cls = self._tensor_to_numpy_array(getattr(track_result_boxes, "cls", None))
        ids = self._tensor_to_numpy_array(getattr(track_result_boxes, "id", None))

        conf = conf.astype(float) if conf is not None else np.zeros((len(xyxy),), dtype=float)
        cls = cls.astype(int) if cls is not None else np.zeros((len(xyxy),), dtype=int)

        # class labels (safe fallback)
        labels = []
        for c in cls:
            name = self.model.names.get(int(c)) if hasattr(self.model, "names") else None
            labels.append(name if name is not None else str(int(c)))

        detections = []
        for i in range(len(xyxy)):
            x1, y1, x2, y2 = xyxy[i]
            bbox = (int(x1), int(y1), int(x2), int(y2))
            cx, cy = center[i]
            track_id = int(ids[i]) if ids is not None and i < len(ids) else None

            detections.append(
                {
                    "label": labels[i],
                    "confidence": float(conf[i]),
                    "bbox": bbox,                # ✅ ints (x1,y1,x2,y2)
                    "center": (float(cx), float(cy)),
                    "track_id": track_id,
                }
            )

        return detections, (track_result.plot() if annotate else frame)
