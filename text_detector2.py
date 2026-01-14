"""
Text-detection using a basic EasyOCR implementation
Created by Eric Leon
"""

import os
import cv2
import easyocr
import matplotlib.pyplot as plt
import src.utils.config as config

# read image with error handling
image_path = "Test_img2.jpg"  # replace with your image path

# Check if file exists
if not os.path.exists(image_path):
    print(f"❌ Error: Image file not found at '{image_path}'")
    print(f"📁 Current working directory: {os.getcwd()}")
    print(f"\n💡 Available image files in current directory:")
    for file in os.listdir('.'):
        if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
            print(f"   - {file}")
    exit(1)

# Load image
image = cv2.imread(image_path)

# Verify image loaded successfully
if image is None:
    print(f"❌ Error: Failed to load image from '{image_path}'")
    print(f"💡 Make sure the file exists and is a valid image format")
    exit(1)

print(f"✅ Image loaded successfully: {image_path}")
print(f"📐 Image size: {image.shape[1]}x{image.shape[0]}")

# instance text detector
print("🔧 Initializing EasyOCR reader...")
reader = easyocr.Reader(['en'], gpu=False)
print("✅ EasyOCR ready!")

# detect text on image
print("🔍 Detecting text...")
text_ = reader.readtext(image)
print(f"📝 Found {len(text_)} text region(s)")

threshold = config.DEFAULT_OCR_CONFIDENCE_THRESHOLD

# draw bbox and text
detections_above_threshold = 0

for i, t in enumerate(text_):
    bbox, text, score = t
    
    print(f"\nRegion {i+1}: '{text}' (confidence: {score:.2f})", end="")

    if score > threshold:
        print(" ✅ ACCEPTED")
        detections_above_threshold += 1
        
        # Convert bbox coordinates to integers
        top_left = tuple(map(int, bbox[0]))
        bottom_right = tuple(map(int, bbox[2]))
        
        # Draw green bounding box
        cv2.rectangle(image, top_left, bottom_right, (0, 255, 0), 3)
        
        # Calculate text position (slightly above bbox)
        text_x = int(bbox[0][0])
        text_y = max(int(bbox[0][1]) - 10, 20)
        text_position = (text_x, text_y)
        
        # Add black outline for better readability
        cv2.putText(image, text, text_position, cv2.FONT_HERSHEY_DUPLEX, 
                    0.65, (0, 0, 0), 4)  # Black outline
        cv2.putText(image, text, text_position, cv2.FONT_HERSHEY_DUPLEX, 
                    0.65, (0, 255, 255), 2)  # Cyan text
        
        # Add confidence score
        score_text = f"{score:.0%}"
        score_x = int(bbox[0][0])
        score_y = int(bbox[2][1]) + 25
        score_position = (score_x, score_y)
        cv2.putText(image, score_text, score_position, cv2.FONT_HERSHEY_DUPLEX, 
                    0.5, (0, 255, 0), 2)
    else:
        print(f" ❌ REJECTED (below threshold: {threshold})")

print(f"\n{'='*60}")
print(f"📊 SUMMARY: {detections_above_threshold}/{len(text_)} detections above threshold ({threshold})")
print(f"{'='*60}\n")

# Display result
plt.figure(figsize=(12, 8))
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title(f"Text Detection Results - {detections_above_threshold} detections", fontsize=14)
plt.axis('off')
plt.tight_layout()
plt.show()

print("✅ Detection complete!")