"""
Text-detection using a two-stage YOLO -> EasyOCR pipeline.
This file implements a minimal, self-contained runner that:
 - Loads a YOLO detector from `src.object_detector.ObjectDetector`
 - Detects objects, crops detections, preprocesses crops
 - Runs EasyOCR per-crop and aggregates text results

Created by Eric Leon
"""

import os
import gc
# Force CPU-only execution and limit thread usage for CPU-only devices.
# Must set before importing heavy libraries that may initialize CUDA or BLAS threads.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
from typing import List, Dict, Tuple

import cv2
import numpy as np
import matplotlib.pyplot as plt
import easyocr

from src.object_detector import ObjectDetector
from src.utils import preprocessing
import src.utils.config as config


def _get_reader(gpu: bool = False):
	"""Create or return a cached EasyOCR reader."""
	# keep a module-level cache
	if not hasattr(_get_reader, "reader"):
		_get_reader.reader = easyocr.Reader(["en"], gpu=gpu)
	return _get_reader.reader


def _resize_image_if_large(image: np.ndarray, max_side: int) -> Tuple[np.ndarray, float]:
	"""Resize image so the largest side <= max_side. Returns (img, scale)."""
	h, w = image.shape[:2]
	if max(h, w) > max_side:
		scale = max_side / max(h, w)
		new_w, new_h = int(w * scale), int(h * scale)
		resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
		return resized, scale
	return image, 1.0


def _preprocess_crop(crop: np.ndarray, fast: bool = True) -> np.ndarray:
	"""Preprocess crop for OCR using existing helpers.

	Returns a single-channel (grayscale) image suitable for EasyOCR.
	"""
	# sharpen -> gray
	try:
		sharp = preprocessing.sharpen_image(crop)
	except Exception:
		sharp = crop

	gray = preprocessing.bgr_to_gray(sharp)

	# optional denoise (skip in fast mode)
	if fast or not config.USE_ADAPTIVE_PREPROCESSING:
		denoised = gray
	else:
		denoised = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)

	return denoised


def process_image(image_path: str, show: bool = False, use_gpu: bool = False) -> List[Dict]:
	"""Run two-stage detection on a single image and return aggregated results.

	Returns a list of detections with keys: label, confidence, bbox, ocr (list).
	"""
	if not os.path.exists(image_path):
		raise FileNotFoundError(f"Image not found: {image_path}")

	img = cv2.imread(image_path)
	if img is None:
		raise ValueError(f"Failed to read image: {image_path}")

	# Resize for YOLO inference using config.YOLO_INFERENCE_SIZE if present
	max_side = getattr(config, "YOLO_INFERENCE_SIZE", 1280)
	img_resized, scale = _resize_image_if_large(img, max_side)

	# Initialize detector and reader
	detector = ObjectDetector(model_name=getattr(config, "DEFAULT_MODEL_NAME", None))
	reader = _get_reader(gpu=use_gpu)

	# Run YOLO detector on resized image
	detections, _ = detector.detect(img_resized, annotate=False)

	results = []

	for det in detections:
		bbox = det.get("bbox")
		if bbox is None:
			continue

		# bbox is (x1,y1,x2,y2) on resized image; convert to ints and ensure ordering
		x1, y1, x2, y2 = map(int, bbox)

		# Clip to image
		x1, y1 = max(0, x1), max(0, y1)
		x2, y2 = min(img_resized.shape[1], x2), min(img_resized.shape[0], y2)

		if x2 <= x1 or y2 <= y1:
			continue

		crop = img_resized[y1:y2, x1:x2]

		# If crop is very small, scale it up for OCR
		if crop.shape[1] < 120:
			crop = cv2.resize(crop, (crop.shape[1] * 2, crop.shape[0] * 2), interpolation=cv2.INTER_CUBIC)

		pre = _preprocess_crop(crop, fast=not config.USE_ADAPTIVE_PREPROCESSING)

		# EasyOCR expects color or gray; pass the preprocessed grayscale
		try:
			ocr_raw = reader.readtext(pre)
		except Exception as e:
			ocr_raw = []

		# Format OCR results
		ocr_results = []
		for item in ocr_raw:
			# item = (bbox, text, conf)
			try:
				_, text, conf = item
			except Exception:
				continue
			if conf >= config.DEFAULT_OCR_CONFIDENCE_THRESHOLD:
				ocr_results.append({"text": text, "confidence": float(conf)})

		det_out = {
			"label": det.get("label"),
			"confidence": float(det.get("confidence", 0.0)),
			"bbox": (x1, y1, x2, y2),
			"ocr": ocr_results,
		}

		results.append(det_out)

		# Optional visualization
		if show:
			cv2.rectangle(img_resized, (x1, y1), (x2, y2), (0, 255, 0), 2)
			if ocr_results:
				cv2.putText(img_resized, ocr_results[0]["text"], (x1, max(y1 - 8, 10)), cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 0), 2)

	# Visualization
	if show:
		plt.figure(figsize=(10, 8))
		plt.imshow(cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB))
		plt.axis("off")
		plt.title(f"Detections: {len(results)}")
		plt.show()

	# Cleanup large objects
	plt.close("all")
	del img, img_resized
	gc.collect()

	return results


def process_camera(
	camera_index: int = 0,
	show: bool = True,
	use_gpu: bool = False,
	ocr_every_n_frames: int = 5,
	max_side: int = None,
):
	"""Run live two-stage detection on a camera stream.

	- Reuses a single `ObjectDetector` and EasyOCR reader for efficiency.
	- Runs YOLO on each frame; runs EasyOCR on each detected crop only every
	  `ocr_every_n_frames` to reduce CPU load.
	"""
	cap = cv2.VideoCapture(camera_index)
	if not cap.isOpened():
		raise RuntimeError(f"Failed to open camera index {camera_index}")

	detector = ObjectDetector(model_name=getattr(config, "DEFAULT_MODEL_NAME", None))
	reader = _get_reader(gpu=use_gpu)

	frame_count = 0
	ocr_cache = {}  # maps bbox string -> last OCR result

	try:
		while True:
			ret, frame = cap.read()
			if not ret:
				break

			# Resize for YOLO inference if requested
			ms = max_side if max_side is not None else getattr(config, "YOLO_INFERENCE_SIZE", 1280)
			frame_resized, scale = _resize_image_if_large(frame, ms)

			# Run YOLO detector (reuses model)
			try:
				detections, _ = detector.detect(frame_resized, annotate=False)
			except Exception:
				detections = []

			annotated = frame_resized.copy()

			for det in detections:
				bbox = det.get("bbox")
				if bbox is None:
					continue
				x1, y1, x2, y2 = map(int, bbox)
				x1, y1 = max(0, x1), max(0, y1)
				x2, y2 = min(annotated.shape[1], x2), min(annotated.shape[0], y2)
				if x2 <= x1 or y2 <= y1:
					continue

				crop = frame_resized[y1:y2, x1:x2]
				if crop.size == 0:
					continue

				key = f"{x1}-{y1}-{x2}-{y2}"
				do_ocr = (frame_count % max(1, ocr_every_n_frames)) == 0

				if do_ocr or key not in ocr_cache:
					small = False
					if crop.shape[1] < 120:
						crop = cv2.resize(crop, (crop.shape[1] * 2, crop.shape[0] * 2), interpolation=cv2.INTER_CUBIC)
						small = True

					pre = _preprocess_crop(crop, fast=not config.USE_ADAPTIVE_PREPROCESSING)
					try:
						ocr_raw = reader.readtext(pre)
					except Exception:
						ocr_raw = []

					ocr_results = []
					for item in ocr_raw:
						try:
							_, text, conf = item
						except Exception:
							continue
						if conf >= config.DEFAULT_OCR_CONFIDENCE_THRESHOLD:
							ocr_results.append({"text": text, "confidence": float(conf)})

					ocr_cache[key] = ocr_results

				# draw box and put first OCR line if available
				cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
				r = ocr_cache.get(key, [])
				if r:
					text = r[0]["text"]
					cv2.putText(annotated, text, (x1, max(y1 - 8, 10)), cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 0), 2)

			if show:
				cv2.imshow("YOLO -> EasyOCR (press 'q' to quit)", annotated)

			key = cv2.waitKey(1) & 0xFF
			if key == ord("q") or key == 27:
				break

			frame_count += 1

	finally:
		cap.release()
		cv2.destroyAllWindows()
		del detector
		del reader
		gc.collect()



if __name__ == "__main__":
	import argparse

	parser = argparse.ArgumentParser(description="YOLO → EasyOCR two-stage demo")
	parser.add_argument("image", nargs="?", help="Path to input image")
	parser.add_argument("--show", action="store_true", help="Show visualization")
	parser.add_argument("--gpu", action="store_true", help="Use GPU for EasyOCR reader if available")
	parser.add_argument("--camera", action="store_true", help="Run live camera loop instead of single image")
	parser.add_argument("--camera-index", type=int, default=0, help="Camera index for cv2.VideoCapture")
	parser.add_argument("--ocr-every-n-frames", type=int, default=5, help="Run OCR every N frames per detected bbox")
	args = parser.parse_args()

	if args.camera:
		process_camera(
			camera_index=args.camera_index,
			show=args.show,
			use_gpu=args.gpu,
			ocr_every_n_frames=max(1, args.ocr_every_n_frames),
			max_side=getattr(config, "YOLO_INFERENCE_SIZE", None),
		)
	else:
		if not args.image:
			parser.error("Please provide an image path or use --camera for live mode.")

		out = process_image(args.image, show=args.show, use_gpu=args.gpu)

		print("\nCombined Results:\n")
		for i, d in enumerate(out, 1):
			print(f"{i}. Label: {d['label']} (conf: {d['confidence']:.2f})")
			print(f"   BBox: {d['bbox']}")
			print(f"   OCR: {d['ocr']}")
			print()

