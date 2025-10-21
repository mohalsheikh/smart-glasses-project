<!-- # Test OCR Script Guide

## Fixed Issues ✅

1. **Model Not Found Error** - Now the script:
   - Tries multiple model locations in order
   - Falls back to `yolov8n.pt` if custom detector not found
   - Auto-downloads `yolov8n.pt` if needed

2. **Added Preprocessing** - As specified in requirements:
   ```python
   gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
   _, thresh = cv2.threshold(gray, 64, 255, cv2.THRESH_BINARY_INV)
   ```

3. **Better Output** - Now shows:
   - Progress indicators
   - Results summary in JSON format
   - Individual result details

## How to Run

```bash
cd C:\Repos\CAPSTONE\smart-glasses-project\src
python test_ocr.py
```

## What It Does

1. **Loads YOLO Model** - Tries these models in order:
   - `../models/text_detector.pt`
   - `../models/license_plate_detector.pt`
   - `license_plate_detector.pt`
   - `../yolov8n.pt`
   - `yolov8n.pt` (auto-download)

2. **Initializes EasyOCR** - English language, CPU mode

3. **Captures Frame** - From webcam
   - Press **SPACE** to capture
   - Press **ESC** to cancel

4. **Detects Regions** - Using YOLO

5. **Extracts Text** - With preprocessing for each region

6. **Shows Results** - Visual display + JSON output

## Expected Output

```
✅ Loading model: ../yolov8n.pt
✅ Frame captured!

🔍 Running YOLO detection...
📝 Extracting text from detected regions...

Found 2 region(s)
  Region 1: 'ABC1234' (confidence: 0.92)
  Region 2: 'STOP' (confidence: 0.87)

============================================================
📊 FINAL OCR RESULTS
============================================================
[{'text': 'ABC1234', 'confidence': 0.92}, {'text': 'STOP', 'confidence': 0.87}]

Formatted output:
  1. 'ABC1234' (confidence: 0.92)
  2. 'STOP' (confidence: 0.87)
============================================================
```

## Adding a Custom Model

To use a license plate detector:

1. Place your model file:
   ```
   C:\Repos\CAPSTONE\smart-glasses-project\models\license_plate_detector.pt
   ```

2. Run the script - it will automatically find and use it!

## Troubleshooting

### "Cannot access webcam"
- Check if another app is using the camera
- Try unplugging/replugging the camera

### "No text detected"
- Ensure text is visible and well-lit
- Hold text closer to camera
- Try high-contrast text (dark on light background)

### Slow performance
- EasyOCR is running in CPU mode (gpu=False)
- To enable GPU: Change `gpu=False` to `gpu=True` on line 28

## Next Steps

For production use, consider using the full `OCREngine` class:
```python
from ocr_engine import OCREngine
ocr = OCREngine()
results = ocr.read_text(frame)
```

See `GETTING_STARTED_OCR.md` for more details. -->
