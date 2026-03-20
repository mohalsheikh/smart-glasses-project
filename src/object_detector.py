"""
Advanced YOLOv8 Object Detector with multi-model support, tracking,
and annotated output.
Created by Ethan
Multi-model support added by Mohammed
"""

from ultralytics import YOLO
from src.utils.config import DEFAULT_FRAME_WIDTH, DEFAULT_FRAME_HEIGHT
import numpy as np
import cv2 as cv
from typing import List, Dict, Tuple, Optional
from src.utils.object_description import normalize_label


class ObjectDetector:
    """
    Object detector that can load one or more YOLO models and merge their
    detections into a single unified result list.

    Usage (single model -- backward compatible):
        detector = ObjectDetector(model_name="yolov8n.pt")

    Usage (multiple models):
        detector = ObjectDetector(model_names=["yolov8n.pt", "currency_detector.pt"])

    The ``classes`` property returns a merged dict of all class names across
    every loaded model, and ``detect()`` returns the combined detections from
    all models.
    """

    def __init__(
        self,
        model_name: Optional[str] = None,        # single model (backward compat)
        model_names: Optional[List[str]] = None,  # multiple models
        conf: float = 0.20,
        iou: float = 0.45,
        imgsz: int = DEFAULT_FRAME_WIDTH if DEFAULT_FRAME_WIDTH > DEFAULT_FRAME_HEIGHT else DEFAULT_FRAME_HEIGHT,
        tracker: str = "bytetrack.yaml",
        max_det: int = 100,
    ):
        # ---------- validate params ----------
        if model_name is None and model_names is None:
            raise ValueError(
                "You must provide either model_name (str) or model_names (list of str)."
            )
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

        self.conf = conf
        self.iou = iou
        self.imgsz = imgsz
        self.tracker = tracker
        self.max_det = max_det

        # ---------- build the list of model paths ----------
        self.paths: List[str] = []
        if model_names is not None:
            self.paths.extend(model_names)
        if model_name is not None and model_name not in self.paths:
            self.paths.insert(0, model_name)

        if len(self.paths) == 0:
            raise ValueError("No model paths provided.")

        # ---------- load models and construct name to id converter and label demerger ----------
        self._models: List[YOLO] = []

        # maps models to a dictionary that maps internal class names to that classes' id
        self._name_to_id: dict[YOLO, dict[str, int]] = dict()
        
        # maps models to a dictionary that maps normalized label names to the class ids from that model that are merged into that label.
        # for instance, an example entry may be: {"currency_detector.pt": {"one dollar bill" : [class id of one-front, class id of one-back]}}
        self._demerge: dict[YOLO, dict[str, list[int]]] = dict() 

        for path in self.paths:
            # load model w/ YOLO constructor and add to _models
            try:
                model = YOLO(path)
                self._models.append(model)
            except Exception as e:
                raise RuntimeError(
                    f"Failed to load model '{path}' with exception: {e}\n"
                    "Make sure that you specify a valid file path to a YOLO model."
                )
            
            model_names_items = model.names.items()
            self._name_to_id[model] = {name: id for id, name in model_names_items}
            self._demerge[model] = dict()
            
            # for each class that is normalized into a normalized label other than its original name or None,
            # self._demerge[model][normalized_label] becomes a list of all of that model's class ids that are normalized into normalized_label on the user's end.
            for id, class_name in model_names_items:
                normalized_label = normalize_label(class_name)

                if normalized_label not in (class_name, None):
                    curr_model_demerge = self._demerge[model]

                    if not curr_model_demerge.get(normalized_label): # create dict entry if doesn't exist
                        curr_model_demerge[normalized_label] = [id]
                    else: # append to list if dict entry does exist
                        curr_model_demerge[normalized_label].append(id)

        print(f"[ObjectDetector] Loaded {len(self._models)} model(s): {self.paths}")

    # ------------------------------------------------------------------
    # Tracking helpers (per-model)
    # ------------------------------------------------------------------
    def _track_single_model(self, model: YOLO, frames: list[np.ndarray], persist: bool = True, objects: list[str] = None):
        """
        Run tracking on a single model and return the first Results object, or none if the current model does not know of any classes in objects (if objects is provided).
        If the objects parameter is provided, this only returns detections for the class ids that correspond to the contents of objects.
        """
        class_ids: list[int] = []

        # populating class_ids with the ids for each object in objects, if objects is provided.
        if objects is not None:

            for obj in objects:
                # two cases: either obj is a normalized label, or it isn't.
                # if obj is a normalized label then we extend class_ids w/ the demerged class ids.
                # otherwise, we add the id for the non-normalized label to class_ids, but ONLY if the provided object is the name of an internal class of the provided model.
                demerged = self._demerge[model].get(obj)

                if demerged is None:
                    id = self._name_to_id[model].get(obj)

                    if id is not None:
                        class_ids.append(self._name_to_id[model][obj])
                else:
                    class_ids.extend(demerged)

            if len(class_ids) == 0:
                return None
        else:
            # class ids is none if objects is None (track will track all classes)
            class_ids = None

        return [ model.track(
            source=frames[i],
            persist=persist,
            conf=self.conf,
            iou=self.iou,
            imgsz=self.imgsz,
            tracker=self.tracker,
            max_det=self.max_det,
            verbose=False,
            classes=class_ids
        )[0] for i in range(len(frames)) ]

    @staticmethod
    def _tensor_to_numpy_array(obj):
        return obj.cpu().numpy() if obj is not None else None

    # ------------------------------------------------------------------
    # Core detection -- runs all models and merges
    # ------------------------------------------------------------------
    def detect(
        self,
        frames: list[np.ndarray],
        annotate: bool = False,
        objects: list[str] = None
    ) -> Tuple[List[Dict], np.ndarray]:
        """
        Run all loaded models on *frame*, merge detections, and return them.

        Returns:
            (detections, output_frames)
            - detections: list of dicts with keys
              label, confidence, bbox, center, track_id, model_index
            - output_frames: list of annotated frames if *annotate* is True,
              otherwise the original frames.
        """
        # all detections from all models for each frame
        all_detections: List[List[Dict]] = [[] for _ in range(len(frames))]

        # run tracking on each frame with each model and merge results into all_detections.
        for model_idx, model in enumerate(self._models):
            try:
                track_results = self._track_single_model(model, frames, objects=objects)
            except Exception as e:
                raise RuntimeError(
                    f"Tracking failed on model {model_idx} with exception: {e}"
                )
            
            # if track_results is None, this means that the current model does not know of any classes in objects (if objects is provided), 
            # so in that case we skip this model and move on to the next one.
            if track_results is None:
                continue

            # extract detections from current model from track_results, add detections from each frameto all_detections
            for i, track_result in enumerate(track_results):
                track_result_boxes = getattr(track_result, "boxes", None)
                
                if track_result_boxes is None:
                    continue

                xyxy = self._tensor_to_numpy_array(track_result_boxes.xyxy)
                if xyxy is None or xyxy.size == 0:
                    continue

                center = (xyxy[:, :2] + xyxy[:, 2:]) / 2
                conf = self._tensor_to_numpy_array(track_result_boxes.conf).astype(float)
                cls = self._tensor_to_numpy_array(track_result_boxes.cls).astype(int)
                ids = self._tensor_to_numpy_array(
                    getattr(track_result_boxes, "id", None)
                )
                labels = [model.names.get(c, f"class_{c}") for c in cls]

                print(f"FRAME {i} - MODEL {model_idx}")
                for j in range(len(xyxy)):
                    all_detections[i].append(
                        {
                            "label": labels[j],
                            "confidence": conf[j],
                            "bbox": tuple(xyxy[j]),
                            "center": tuple(center[j]),
                            "track_id": f"{model_idx}.{int(ids[j])}" if ids is not None and j < len(ids) else f"{model_idx}.N/A", # prefixed w/ model idx to ensure global uniqueness across models
                            "model_index": model_idx,
                        }
                    )

                    print(f"[ObjectDetector] Model {model_idx} detected {labels[j]} with confidence {conf[j]:.2f} at {xyxy[j]} (track_id: {ids[j] if ids is not None and j < len(ids) else 'N/A'})")
                
            # reset model to reset track ids for next detection run
            model = YOLO(self.paths[model_idx])

        # ---------- annotation ----------
        if annotate:
            annotated_frames = [self._annotate_frame(frames[i].copy(), all_detections[i]) for i in range(len(frames))]
        else:
            annotated_frames = frames.copy()

        return all_detections, annotated_frames

    # ------------------------------------------------------------------
    # Simple annotation that works across multiple models
    # ------------------------------------------------------------------
    def _annotate_frame(
        self, frame: np.ndarray, detections: List[Dict]
    ) -> np.ndarray:
        """Draw bounding boxes and labels for all detections onto *frame*."""
        # Different colors per model index so they're visually distinguishable
        palette = [
            (0, 255, 0),    # green  -- model 0
            (255, 128, 0),  # orange -- model 1
            (0, 200, 255),  # cyan   -- model 2
            (200, 0, 255),  # purple -- model 3
        ]

        for det in detections:
            x1, y1, x2, y2 = map(int, det["bbox"])
            color = palette[det.get("model_index", 0) % len(palette)]
            label_text = f"{det['label']} {det['confidence']:.0%}"

            cv.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            font = cv.FONT_HERSHEY_SIMPLEX
            font_scale = 0.55
            thickness = 2
            (tw, th), _ = cv.getTextSize(label_text, font, font_scale, thickness)
            cv.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 6, y1), color, -1)
            cv.putText(
                frame, label_text, (x1 + 3, y1 - 4),
                font, font_scale, (0, 0, 0), thickness,
            )

        return frame

    # ------------------------------------------------------------------
    # Merged class names across all models
    # ------------------------------------------------------------------
    @property
    def classes(self) -> Dict[int, str]:
        """
        Return a merged dictionary of class names from all loaded models.

        Keys are globally unique -- if two models share an index the second
        model's classes are offset so there are no collisions.
        """
        merged: Dict[int, str] = {}
        offset = 0
        for model in self._models:
            for idx, name in model.names.items():
                merged[offset + idx] = name
            # offset the next model's indices past the current model's max
            if model.names:
                offset += max(model.names.keys()) + 1
        return merged