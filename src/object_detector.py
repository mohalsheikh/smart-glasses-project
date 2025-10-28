import cv2
import torch
from ultralytics import YOLO
from typing import List, Dict, Any, Optional, Tuple

from src.utils.config import (
    YOLO_WEIGHTS_PATH,
    DEVICE,
    DEFAULT_CONF_THRESHOLD,
    MAX_DETECTIONS_PER_FRAME,
)

class ObjectDetector:
    """
    Wrapper around your fine-tuned YOLO model.
    Handles:
    - loading weights
    - running inference
    - returning clean structured detections
    - drawing boxes (optional)
    """

    def __init__(
        self,
        weights_path: str = YOLO_WEIGHTS_PATH,
        device: str = DEVICE,
        conf_threshold: float = DEFAULT_CONF_THRESHOLD,
    ):
        self.device = device
        self.conf_threshold = conf_threshold

        print(f"[ObjectDetector] Loading YOLO model from {weights_path}")
        print(f"[ObjectDetector] Using device: {device}")

        self.model = YOLO(weights_path)

        # model.names is {class_id: "classname"}
        self.class_names = self.model.names

    def detect(self, frame_bgr) -> List[Dict[str, Any]]:
        """
        Run detection on one frame (BGR OpenCV frame).
        Returns a list of dict detections:
        {
            "class_id": int,
            "class_name": str,
            "confidence": float,
            "bbox_xyxy": [x1, y1, x2, y2],
            "bbox_xywh": [cx, cy, w, h]
        }
        """
        results = self.model.predict(
            source=frame_bgr,
            device=self.device,
            conf=self.conf_threshold,
            verbose=False,
        )

        r = results[0]

        boxes_xyxy = r.boxes.xyxy.detach().cpu().numpy()  # (N, 4)
        boxes_xywh = r.boxes.xywh.detach().cpu().numpy()  # (N, 4) cx,cy,w,h
        scores = r.boxes.conf.detach().cpu().numpy()      # (N,)
        class_ids = r.boxes.cls.detach().cpu().numpy().astype(int)  # (N,)

        detections = []
        for i in range(len(class_ids)):
            cid = class_ids[i]
            conf = float(scores[i])
            name = self.class_names.get(cid, f"class_{cid}")
            det = {
                "class_id": cid,
                "class_name": name,
                "confidence": conf,
                "bbox_xyxy": boxes_xyxy[i].tolist(),
                "bbox_xywh": boxes_xywh[i].tolist(),
            }
            detections.append(det)

        # sort high → low confidence
        detections.sort(key=lambda d: d["confidence"], reverse=True)

        # trim to not spam downstream audio, etc.
        return detections[:MAX_DETECTIONS_PER_FRAME]

    def draw_detections(
        self,
        frame_bgr,
        detections: List[Dict[str, Any]],
        box_color: Tuple[int, int, int] = (0, 255, 0),
    ):
        """
        Draw boxes + labels ON the frame, in-place.
        Returns the same frame for convenience.
        """
        for det in detections:
            x1, y1, x2, y2 = det["bbox_xyxy"]
            label = f'{det["class_name"]} {det["confidence"]:.2f}'

            # box
            cv2.rectangle(
                frame_bgr,
                (int(x1), int(y1)),
                (int(x2), int(y2)),
                box_color,
                2,
            )

            # label bg
            cv2.rectangle(
                frame_bgr,
                (int(x1), int(y1) - 20),
                (int(x1) + 200, int(y1)),
                box_color,
                -1,
            )

            # label text
            cv2.putText(
                frame_bgr,
                label,
                (int(x1) + 5, int(y1) - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 0),
                1,
                cv2.LINE_AA,
            )

        return frame_bgr


if __name__ == "__main__":
    # quick self-test if you run:
    # python src/object_detector.py
    import time

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: could not open camera 0, trying 1...")
        cap = cv2.VideoCapture(1)

    detector = ObjectDetector()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Camera read failed, exiting.")
            break

        dets = detector.detect(frame)
        frame_drawn = detector.draw_detections(frame, dets)

        cv2.imshow("ObjectDetector self-test", frame_drawn)

        # simple log of top detection
        if len(dets) > 0:
            top = dets[0]
            print(
                f"[{time.strftime('%H:%M:%S')}] {top['class_name']} {top['confidence']:.2f} "
                f"at {top['bbox_xyxy']}"
            )

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
