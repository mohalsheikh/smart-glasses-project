"""
Advanced YOLOv8 Object Detector with optional tracking and annotated output.
"""

from ultralytics import YOLO
import src.utils.config as config

class ObjectDetector:
    def __init__(self):
        if config.MODEL is None:
            raise ValueError("Model path must be set in config.")
        self.model = YOLO(config.MODEL)

        if config.CONFIDENCE_THRESHOLD is None:
            raise ValueError("Confidence threshold must be set in config.")
        self.conf = config.CONFIDENCE_THRESHOLD

        if config.IOU_THRESHOLD is None:
            raise ValueError("IoU threshold must be set in config.")
        self.iou = config.IOU_THRESHOLD

        if config.IMGSZ is None:
            raise ValueError("Image size must be set in config.")
        self.imgsz = config.IMGSZ

        if config.TRACKER is None:
            raise ValueError("Tracker config must be set in config.")
        self.tracker = config.TRACKER

        if config.MAX_DETECTIONS is None:
            raise ValueError("Max detections must be set in config.")
        self.max_det = config.MAX_DETECTIONS