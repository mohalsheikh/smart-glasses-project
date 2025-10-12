"""
Advanced YOLOv8 Object Detector with tracking and annotated output.
"""

from ultralytics import YOLO
import src.utils.config as config
import numpy as np

class ObjectDetector:
    # initializes the detector with specified parameters or defaults from config.
    def __init__(
            self,
            model_name: str = config.DEFAULT_MODEL_NAME, # path to the YOLO model
            conf: float = config.DEFAULT_CONFIDENCE_THRESHOLD, # confidence threshold
            iou: float = config.DEFAULT_IOU_THRESHOLD, # IoU threshold
            imgsz: int = config.FRAME_WIDTH if config.FRAME_WIDTH > config.FRAME_HEIGHT else config.FRAME_HEIGHT, # image size for model input
            tracker: str = config.DEFAULT_TRACKER, # the tracker we're using
            max_det: int = config.DEFAULT_MAX_DETECTIONS # maximum number of objects to detect in a frame
        ):

        # parameters cannot be None and must be of the types specified in the function signature.
        # the ifs below ensure that the parameters are not None by raising an error if they are.

        # error if no path to model is provided
        if model_name is None:
            raise ValueError("Model name must be set.")
        
        # error if confidence threshold is not provided
        if conf is None:
           raise ValueError("Confidence threshold must be set.")
        
        # error if iou threshold is not provided
        if iou is None:
            raise ValueError("IoU threshold must be set.")
        
        # error if image size is not provided
        if imgsz is None:
            raise ValueError("Image size must be set.")

        # error if tracker config is not provided
        if tracker is None:
            raise ValueError("Tracker config must be set.")

        # error if max detections is not provided
        if max_det is None:
            raise ValueError("Max detections must be set.")
        
        # store parameters
        self.conf = conf
        self.iou = iou
        self.imgsz = imgsz
        self.tracker = tracker
        self.max_det = max_det

        # attempt to load the model. if loading fails, raise an error
        try:
            self.model = YOLO(model_name)
        except Exception as e:
            raise RuntimeError(f"Failed to load model '{model_name}' with exception: {e}\nMake sure that you specify a valid file path to a YOLO model.")

    # wrapper for the model's track method with the parameters set in __init__.
    # the yolo track function, when given one frame, returns a list of one result every time if it is successful,
    # so we just return the first element of that list
    def _track(self, frame: np.ndarray, persist: bool = True):
        return self.model.track(
            source = frame,
            persist = persist,
            conf = self.conf,
            iou = self.iou,
            imgsz = self.imgsz,
            tracker = self.tracker,
            max_det = self.max_det,
            verbose = False
        )[0]
    
    @staticmethod
    def _tensor_to_numpy_array(obj):
        return obj.cpu().numpy() if obj is not None else None

    # returns tuple (detections, frame).
    # if annotate is True, frame is the annotated frame, otherwise it's the original frame.
    def detect(self, frame: np.ndarray, annotate: bool = False):
        # attempt to track objects in the frame. if tracking fails, raise an error
        try:
            track_result = self._track(frame)
        except Exception as e:
            raise RuntimeError(f"Tracking failed with exception: {e}")
        
        track_result_boxes = getattr(track_result, 'boxes', None) # get the boxes attribute from the track result

        if track_result_boxes is None: # if there is no boxes attribute, return empty list and original frame
            return [], frame
        
        xyxy = self._tensor_to_numpy_array(track_result_boxes.xyxy) # bounding box coordinates

        if xyxy.size == 0: # if no detections, return empty list and original frame
            return [], frame

        center = (xyxy[:, :2] + xyxy[:, 2:]) / 2
        conf = self._tensor_to_numpy_array(track_result_boxes.conf).astype(float) # confidence scores
        cls = self._tensor_to_numpy_array(track_result_boxes.cls).astype(int) # class indices
        id = self._tensor_to_numpy_array(getattr(track_result_boxes, 'id', None)) # track IDs, if available
        label = [self.model.names.get(c) for c in cls] # class labels

        # comphrehension to create list of detections. 
        # we format detections as dictionaries with keys: label, confidence, bbox, center, track_id
        detections = [
            {
            "label": label[i],
            "confidence": conf[i],
            "bbox": tuple(xyxy[i]), # convert numpy array to list for easier serialization
            "center": tuple(center[i]),
            "track_id": int(id[i]) if id is not None and i < len(id) else None
            } 
            for i in range(len(xyxy))
        ]

        # always return detections.
        # if annotate is true, return annotated frame, otherwise return original frame
        return detections, track_result.plot() if annotate else frame