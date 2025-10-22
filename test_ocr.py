"""
Created by Eric Leon
"""
# import cv2
# from ultralytics import YOLO
# import easyocr
# import os

# # --- 1. Load YOLO model ---
# # Try to load license plate detector, fall back to yolov8n.pt if not found
# model_options = [
#     "../models/text_detector.pt",
#     "../models/license_plate_detector.pt",
#     "license_plate_detector.pt",
#     "../yolov8n.pt",
#     "yolov8n.pt"
# ]

# model = None
# for model_path in model_options:
#     if os.path.exists(model_path):
#         print(f"✅ Loading model: {model_path}")
#         model = YOLO(model_path)
#         break

# if model is None:
#     print("❌ No model found. Trying to download yolov8n.pt...")
#     model = YOLO("yolov8n.pt")  # This will auto-download if not present

# # --- 2. Initialize EasyOCR reader ---
# reader = easyocr.Reader(['en'], gpu=False)

# # --- 3. Capture a single frame from webcam ---
# cam = cv2.VideoCapture(0)
# if not cam.isOpened():
#     print("❌ Cannot access webcam.")
#     exit()

# print("📸 Press SPACE to capture a frame...")
# while True:
#     ret, frame = cam.read()
#     cv2.imshow("Webcam - Press SPACE to Capture", frame)
#     key = cv2.waitKey(1) & 0xFF
#     if key == 32:  # SPACE key
#         captured_frame = frame.copy()
#         break
#     elif key == 27:  # ESC to exit
#         cam.release()
#         cv2.destroyAllWindows()
#         exit()

# cam.release()
# cv2.destroyAllWindows()
# print("✅ Frame captured!")

# # --- 4. Run YOLO detection on the frame ---
# print("\n🔍 Running YOLO detection...")
# results = model(captured_frame)

# # --- 5. Draw bounding boxes and OCR each detection ---
# print("📝 Extracting text from detected regions...\n")
# all_text_results = []

# for r in results:
#     boxes = r.boxes.xyxy
#     print(f"Found {len(boxes)} region(s)")
    
#     for i, box in enumerate(boxes):
#         x1, y1, x2, y2 = map(int, box)
#         crop = captured_frame[y1:y2, x1:x2]

#         if crop.size == 0:
#             continue

#         # --- Preprocess crop for better OCR accuracy ---
#         gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
#         _, thresh = cv2.threshold(gray, 64, 255, cv2.THRESH_BINARY_INV)

#         # Run OCR on preprocessed image
#         ocr_result = reader.readtext(thresh)
#         if ocr_result:
#             text, conf = ocr_result[0][1], ocr_result[0][2]
#             print(f"  Region {i+1}: '{text}' (confidence: {conf:.2f})")
#             all_text_results.append({"text": text, "confidence": round(conf, 2)})
#             cv2.putText(captured_frame, text, (x1, y1 - 10),
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
#         else:
#             print(f"  Region {i+1}: No text detected")
#         cv2.rectangle(captured_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

# # --- 6. Display final results ---
# print("\n" + "=" * 60)
# print("📊 FINAL OCR RESULTS")
# print("=" * 60)
# if all_text_results:
#     print(all_text_results)
#     print("\nFormatted output:")
#     for i, result in enumerate(all_text_results, 1):
#         print(f"  {i}. '{result['text']}' (confidence: {result['confidence']})")
# else:
#     print("No text detected in any region.")
# print("=" * 60)

# cv2.imshow("Detected Text - Press any key to close", captured_frame)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
