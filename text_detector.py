"""
Text-detection using a basic EasyOCR implementation
Created by Eric Leon
"""

import os

# limit BLAS / OpenMP threads early
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")

# safe imports and cleanup
import gc
try:
    import torch
except Exception:
    torch = None

# If a GPU is available and torch loaded, clear CUDA cache (safe-guard)
if torch is not None:
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass

# If previous runs left large objects in the global namespace (e.g. in an interactive session),
# delete them only if they exist to avoid NameError.
if 'model' in globals():
    try:
        del globals()['model']
    except Exception:
        pass

if 'reader' in globals():
    try:
        del globals()['reader']
    except Exception:
        pass

# Force a collection after conditional deletions
gc.collect()

import cv2
import easyocr
import matplotlib.pyplot as plt
import src.utils.config as config
import psutil, os
p = psutil.Process(os.getpid()); print(p.memory_info().rss)

# Performance flags
FAST_MODE = True               # True = skip denoising (faster). Set False for higher quality.
MAX_IMAGE_DIMENSION = 1280         # Resize largest side to this if original is bigger.

def resize_image_if_large(image, max_dim: int = MAX_IMAGE_DIMENSION):
    """Resize image if the largest side exceeds max_dim. Returns (image, scale)."""
    h, w = image.shape[:2]
    if max(h, w) > max_dim:
        scale = max_dim / max(h, w)
        new_w, new_h = int(w * scale), int(h * scale)
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
        print(f"📐 Resized {w}x{h} -> {new_w}x{new_h}")
        return resized, scale
    return image, 1.0

# read image (use config-driven threshold)
image_path = "Test_img2.jpg"  # replace with your image path
if not os.path.exists(image_path):
    print(f"❌ Error: Image file not found at '{image_path}'")
    print(f"📁 Current working directory: {os.getcwd()}")
    exit(1)

image = cv2.imread(image_path)
if image is None:
    print(f"❌ Failed to load image: {image_path}")
    exit(1)

# Resize early to reduce work/memory (important for large photos)
image, _ = resize_image_if_large(image, MAX_IMAGE_DIMENSION)

# instance text detector (init once if reused elsewhere)
reader = easyocr.Reader(['en'], gpu=False)

# create a lightweight preprocessing flow: sharpen -> grayscale -> optional denoise
sharpened = cv2.cvtColor(sharpen_image(image), cv2.COLOR_BGR2RGB) if 'sharpen_image' in globals() else image
gray = cv2.cvtColor(sharpened, cv2.COLOR_BGR2GRAY)
if FAST_MODE:
    denoised = gray
else:
    denoised = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)

# Use denoised (grayscale) for OCR — don't feed aggressive binary images to EasyOCR
text_ = reader.readtext(denoised)
threshold = config.DEFAULT_OCR_CONFIDENCE_THRESHOLD

# draw bbox and text (convert coords to int, show once)
detections = 0
for bbox, txt, score in text_:
    if score > threshold:
        detections += 1
        top_left = tuple(map(int, bbox[0]))
        bottom_right = tuple(map(int, bbox[2]))
        cv2.rectangle(image, top_left, bottom_right, (0, 255, 0), 5)
        text_pos = (top_left[0], max(top_left[1] - 10, 20))
        cv2.putText(image, txt, text_pos, cv2.FONT_HERSHEY_DUPLEX, 0.65, (255, 0, 0), 2)

print(f"📊 Detections above threshold ({threshold}): {detections}/{len(text_)}")
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.show()

# Close any open matplotlib figures and free large objects
plt.close("all")
del image, denoised, gray, sharpened, text_
gc.collect()
print("✅ Done — resources freed.")