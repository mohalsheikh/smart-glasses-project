"""
Advanced YOLOv8 Object Detector with optional tracking and annotated output.
"""

from typing import List, Dict, Tuple, Optional
import time
import numpy as np
import cv2
from ultralytics import YOLO


class ObjectDetector:
    def __init__(
        self,
        model_name: str = "yolov8n.pt",
        conf: float = 0.35,
        iou: float = 0.45,
        imgsz: int = 640,
        device: Optional[str] = None,       # e.g. "mps" on Apple Silicon if torch+mps is installed
        track: bool = True,
        tracker: str = "bytetrack.yaml",
        classes: Optional[List[int]] = None,
        half: bool = False,
        max_det: int = 100
    ):
        print("📦 Loading YOLOv8 model...")
        self.model = YOLO(model_name)

        self.conf = conf
        self.iou = iou
        self.imgsz = imgsz
        self.device = device
        self.track_enabled = track
        self.tracker = tracker
        self.classes = classes
        self.half = half
        self.max_det = max_det

        self._predict_kwargs = {
            "conf": self.conf,
            "iou": self.iou,
            "imgsz": self.imgsz,
            "device": self.device,
            "classes": self.classes,
            "half": self.half,
            "max_det": self.max_det,
            "verbose": False
        }
        print("✅ YOLOv8 loaded successfully!")

    def _run(self, frame, do_track: bool):
        if do_track:
            return self.model.track(
                source=frame,
                persist=True,
                tracker=self.tracker,
                **self._predict_kwargs
            )
        else:
            return self.model.predict(source=frame, **self._predict_kwargs)

    @staticmethod
    def _to_int(x) -> Optional[int]:
        try:
            return int(float(x))
        except Exception:
            return None

    def detect(self, frame, visualize: bool = True) -> Tuple[List[Dict], np.ndarray]:
        """
        Returns (detections, annotated_frame)
        detections: list of dicts {label, conf, bbox:(x1,y1,x2,y2), center:(cx,cy), track_id}
        """
        try:
            results = self._run(frame, do_track=self.track_enabled)
        except Exception as e:
            print(f"⚠️ Tracking failed ({e}); falling back to predict()")
            results = self._run(frame, do_track=False)

        dets: List[Dict] = []
        if not results:
            return dets, frame

        r = results[0]
        annotated = r.plot() if visualize else frame

        boxes = getattr(r, "boxes", None)
        if boxes is None or boxes.xyxy is None:
            return dets, annotated

        xyxy = boxes.xyxy
        confs = boxes.conf
        clss = boxes.cls
        ids = getattr(boxes, "id", None)

        # ensure numpy
        to_np = lambda t: t.cpu().numpy() if hasattr(t, "cpu") else t
        xyxy = to_np(xyxy)
        confs = to_np(confs)
        clss = to_np(clss)
        ids = to_np(ids) if ids is not None else None

        for i in range(xyxy.shape[0]):
            x1, y1, x2, y2 = [int(v) for v in xyxy[i]]
            conf = float(confs[i])
            cls_id = int(clss[i])
            label = self.model.names.get(cls_id, str(cls_id))
            track_id = self._to_int(ids[i]) if ids is not None and i < len(ids) else None
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)

            dets.append({
                "label": label,
                "conf": conf,
                "bbox": (x1, y1, x2, y2),
                "center": (cx, cy),
                "track_id": track_id
            })

        return dets, annotated
