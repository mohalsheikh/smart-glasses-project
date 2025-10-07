"""
Advanced YOLOv8 Object Detector with tracking and annotated output.
"""

# TODO look at detect method from our demo code, learn how it works, see what we want to do in this actual implementation...

from ultralytics import YOLO
import src.utils.config as config

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
            raise ValueError(f"Failed to load model '{model_name}' with exception: {e}\nMake sure that you specify a valid file path to a YOLO model.")

    # wrapper for the model's track method with the parameters set in __init__
    def _track(self, frame, persist: bool = True):
        return self.model.track(
            source = frame,
            persist = persist,
            conf = self.conf,
            iou = self.iou,
            imgsz = self.imgsz,
            tracker = self.tracker,
            max_det = self.max_det
        )