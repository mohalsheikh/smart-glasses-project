"""
Text-detection using a basic EasyOCR implementation with preprocessing
OPTIMIZED VERSION for better performance
Created by Eric Leon
"""

import cv2
import easyocr
import matplotlib.pyplot as plt
import os
import numpy as np
import time
from src.utils.config import DEFAULT_OCR_CONFIDENCE_THRESHOLD
from src.utils.preprocessing import sharpen_image, bgr_to_gray, gray_to_bgr

# Global reader to initialize once
_reader = None

def get_reader():
    """Initialize EasyOCR reader once and reuse"""
    global _reader
    if _reader is None:
        print("🔧 Initializing EasyOCR reader (one-time setup)...")
        start = time.time()
        _reader = easyocr.Reader(['en'], gpu=False)
        print(f"✅ EasyOCR ready! (took {time.time()-start:.1f}s)")
    return _reader

def resize_image_if_large(image, max_dimension=1920):
    """
    Resize image if it's too large to speed up processing
    """
    height, width = image.shape[:2]
    
    if max(height, width) > max_dimension:
        scale = max_dimension / max(height, width)
        new_width = int(width * scale)
        new_height = int(height * scale)
        resized = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
        print(f"📐 Resized from {width}x{height} to {new_width}x{new_height} for faster processing")
        return resized, scale
    
    return image, 1.0

def preprocess_for_ocr(image, show_steps=False, fast_mode=True):
    """
    Apply preprocessing pipeline to improve OCR accuracy
    
    Args:
        fast_mode: If True, skip denoising (faster but slightly less accurate)
    """
    start_time = time.time()
    original = image.copy()
    
    # Step 1: Sharpen using existing function
    sharpened = sharpen_image(image)
    
    # Step 2: Convert to grayscale using existing function
    gray = bgr_to_gray(sharpened)
    
    # Step 3: Denoise (OPTIONAL - skip in fast mode)
    if fast_mode:
        print("⚡ Fast mode: Skipping denoising")
        denoised = gray
    else:
        print("🔧 Applying denoising (slow)...")
        denoised = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)
    
    # Step 4: Adaptive threshold for better text contrast
    thresh = cv2.adaptiveThreshold(
        denoised, 255, 
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY, 11, 2
    )
    
    # Step 5: Morphological operations to reduce noise
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    
    # Convert back to BGR for display/processing using existing function
    processed = gray_to_bgr(morph)
    
    print(f"✅ Preprocessing complete! (took {time.time()-start_time:.2f}s)")
    
    # Show preprocessing steps if requested (non-blocking)
    if show_steps:
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        axes[0, 0].imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
        axes[0, 0].set_title('1. Original')
        axes[0, 0].axis('off')
        
        axes[0, 1].imshow(cv2.cvtColor(sharpened, cv2.COLOR_BGR2RGB))
        axes[0, 1].set_title('2. Sharpened')
        axes[0, 1].axis('off')
        
        axes[0, 2].imshow(gray, cmap='gray')
        axes[0, 2].set_title('3. Grayscale')
        axes[0, 2].axis('off')
        
        axes[1, 0].imshow(denoised, cmap='gray')
        axes[1, 0].set_title('4. Denoised' if not fast_mode else '4. Denoised (skipped)')
        axes[1, 0].axis('off')
        
        axes[1, 1].imshow(thresh, cmap='gray')
        axes[1, 1].set_title('5. Thresholded')
        axes[1, 1].axis('off')
        
        axes[1, 2].imshow(morph, cmap='gray')
        axes[1, 2].set_title('6. Morphological')
        axes[1, 2].axis('off')
        
        plt.tight_layout()
        # Replace non-blocking show with blocking show so this is the only window
        plt.show()
    
    return processed

# Configuration
FAST_MODE = False  # Set to False for better accuracy but slower
SHOW_PREPROCESSING_STEPS = True  # Set to True to see preprocessing
MAX_IMAGE_SIZE = 1920  # Resize larger images for speed
SHOW_FINAL_RESULTS = False  # NEW: show original/result figure

# read image with error handling
image_path = "Test_img4.jpg"  # replace with your image path

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
total_start = time.time()
image = cv2.imread(image_path)

# Verify image loaded successfully
if image is None:
    print(f"❌ Error: Failed to load image from '{image_path}'")
    print(f"💡 Make sure the file exists and is a valid image format")
    exit(1)

print(f"✅ Image loaded successfully: {image_path}")
print(f"📐 Original image size: {image.shape[1]}x{image.shape[0]}")

# Resize if image is too large
image, scale_factor = resize_image_if_large(image, MAX_IMAGE_SIZE)

# Apply preprocessing
print(f"🔧 Applying preprocessing {'(FAST MODE)' if FAST_MODE else '(QUALITY MODE)'}...")
preprocessed_image = preprocess_for_ocr(image, show_steps=SHOW_PREPROCESSING_STEPS, fast_mode=FAST_MODE)

# Get reader (initializes once)
reader = get_reader()

# detect text on PREPROCESSED image
print("🔍 Detecting text on preprocessed image...")
ocr_start = time.time()
text_ = reader.readtext(preprocessed_image)
ocr_time = time.time() - ocr_start
print(f"📝 Found {len(text_)} text region(s) (OCR took {ocr_time:.2f}s)")

threshold = DEFAULT_OCR_CONFIDENCE_THRESHOLD

# draw bbox and text on ORIGINAL image for better visualization
result_image = image.copy()
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
        cv2.rectangle(result_image, top_left, bottom_right, (0, 255, 0), 3)
        
        # Calculate text position (slightly above bbox)
        text_x = int(bbox[0][0])
        text_y = max(int(bbox[0][1]) - 10, 20)
        text_position = (text_x, text_y)
        
        # Add black outline for better readability
        cv2.putText(result_image, text, text_position, cv2.FONT_HERSHEY_DUPLEX, 
                    0.65, (0, 0, 0), 4)  # Black outline
        cv2.putText(result_image, text, text_position, cv2.FONT_HERSHEY_DUPLEX, 
                    0.65, (0, 255, 255), 2)  # Cyan text
        
        # Add confidence score
        score_text = f"{score:.0%}"
        score_x = int(bbox[0][0])
        score_y = int(bbox[2][1]) + 25
        score_position = (score_x, score_y)
        cv2.putText(result_image, score_text, score_position, cv2.FONT_HERSHEY_DUPLEX, 
                    0.5, (0, 255, 0), 2)
    else:
        print(f" ❌ REJECTED (below threshold: {threshold})")

total_time = time.time() - total_start

print(f"\n{'='*60}")
print(f"📊 SUMMARY: {detections_above_threshold}/{len(text_)} detections above threshold ({threshold})")
print(f"⏱️  Total processing time: {total_time:.2f}s")
print(f"    - OCR inference: {ocr_time:.2f}s ({ocr_time/total_time*100:.0f}%)")
print(f"{'='*60}\n")

# Display result
if SHOW_FINAL_RESULTS:
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    axes[0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    axes[0].set_title('Original Image', fontsize=14)
    axes[0].axis('off')

    axes[1].imshow(cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB))
    axes[1].set_title(f'Text Detection Results - {detections_above_threshold} detections', fontsize=14)
    axes[1].axis('off')

    plt.tight_layout()
    plt.show()

print("✅ Detection complete!")