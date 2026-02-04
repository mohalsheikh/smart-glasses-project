# VisionAssist Currency Model Training Guide

## 🎯 Quick Start (TL;DR)

```bash
# 1. Clone your repo on the lab computer
git clone https://github.com/YOUR_USERNAME/visionassist.git
cd visionassist/training

# 2. Set up environment
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt

# 3. Download dataset (get API key from roboflow.com - it's free!)
python train_currency_model.py --download --api-key YOUR_ROBOFLOW_API_KEY

# 4. Train (takes 1-3 hours depending on GPU)
python train_currency_model.py --train

# 5. Export the model
python train_currency_model.py --export

# 6. Push to GitHub
git add models/
git commit -m "Add trained currency detection model"
git push
```

---

## 📋 Complete Step-by-Step Instructions

### Step 1: Get Your Roboflow API Key (Do This First!)

1. Go to [roboflow.com](https://roboflow.com)
2. Click "Sign Up" (it's FREE)
3. After signing in, click your profile icon → "Settings"
4. Go to "API Key" tab
5. Copy your API key (looks like: `abc123xyz789`)
6. Save it somewhere - you'll need it!

### Step 2: Set Up the Lab Computer

Open a terminal and run:

```bash
# Check if CUDA/GPU is available
nvidia-smi

# You should see something like:
# +-----------------------------------------------------------------------------+
# | NVIDIA-SMI 525.60.11    Driver Version: 525.60.11    CUDA Version: 12.0     |
# | GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
# | Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
# |   0  NVIDIA GeForce ...  Off  | 00000000:01:00.0  On |                  N/A |
# | 30%   45C    P8    15W / 250W |    500MiB / 11264MiB |      0%      Default |
# +-----------------------------------------------------------------------------+
```

If you see the GPU info, you're good! If not, you might need to use a different lab computer.

### Step 3: Clone Your Repository

```bash
cd ~
git clone https://github.com/YOUR_USERNAME/visionassist.git
cd visionassist
```

### Step 4: Create Virtual Environment (Recommended)

```bash
# Create a virtual environment
python -m venv venv

# Activate it
source venv/bin/activate  # Linux/Mac
# OR
.\venv\Scripts\activate   # Windows

# Verify Python
which python  # Should show your venv path
```

### Step 5: Install Dependencies

```bash
# Install PyTorch with CUDA support
# For CUDA 11.8 (most common):
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# For CUDA 12.1:
# pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Install other requirements
cd training
pip install -r requirements.txt

# Verify CUDA is working
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}')"
```

### Step 6: Download the Dataset

```bash
# Replace YOUR_API_KEY with your actual Roboflow API key
python train_currency_model.py --download --api-key YOUR_API_KEY
```

This downloads ~360 labeled images of US dollar bills in YOLOv8 format.

### Step 7: Train the Model

```bash
# Start training (takes 1-3 hours)
python train_currency_model.py --train

# If training gets interrupted, resume with:
python train_currency_model.py --train --resume
```

**What to expect:**
- Training shows progress bars for each epoch
- mAP50 should climb to 0.85+ (85%+ accuracy)
- Final results saved to `runs/currency_detector/`

**If you get CUDA out of memory:**
```bash
# Use smaller batch size
python train_currency_model.py --train --batch-size 8
```

### Step 8: Validate the Model

```bash
python train_currency_model.py --validate
```

You should see:
```
VALIDATION RESULTS
==================
  mAP50: 0.95+      (aim for >0.90)
  mAP50-95: 0.85+   (aim for >0.80)
  Precision: 0.90+  (aim for >0.85)
  Recall: 0.90+     (aim for >0.85)
```

### Step 9: Export for Deployment

```bash
python train_currency_model.py --export
```

This creates:
- `models/currency_detector.pt` - Main PyTorch model
- `models/currency_detector.onnx` - Cross-platform version

### Step 10: Test on Sample Images

```bash
# Download a test image of a dollar bill
# Or take a photo with your phone and transfer it

python train_currency_model.py --test --image path/to/dollar_bill.jpg
```

### Step 11: Push to GitHub

```bash
cd ..  # Back to repo root

# Add the trained model
git add models/currency_detector.pt
git add training/

# Commit
git commit -m "Add trained currency detection model (mAP50: XX%)"

# Push
git push origin main
```

### Step 12: Pull on Your Laptop

On your laptop:
```bash
cd visionassist
git pull

# The model is now at models/currency_detector.pt
```

---

## 🔧 Troubleshooting

### "CUDA out of memory"
```bash
# Reduce batch size
python train_currency_model.py --train --batch-size 8

# Or use a smaller model
# Edit train_currency_model.py: MODEL_SIZE = 'yolov8s.pt'
```

### "No module named 'ultralytics'"
```bash
pip install ultralytics
```

### "roboflow.errors.HTTPError"
- Check your API key is correct
- Make sure you have internet access
- Try again (sometimes Roboflow has temporary issues)

### Model file too large for GitHub (>100MB)
```bash
# Use Git LFS (Large File Storage)
git lfs install
git lfs track "*.pt"
git add .gitattributes
git add models/currency_detector.pt
git commit -m "Add model with LFS"
git push
```

Or use a smaller model:
```bash
# Edit CONFIG in train_currency_model.py
MODEL_SIZE = 'yolov8n.pt'  # Nano model, ~6MB
```

### Training takes too long
```bash
# Reduce epochs for quick testing
python train_currency_model.py --train --epochs 20
```

---

## 📊 Expected Results

After training, you should achieve:

| Metric | Target | Excellent |
|--------|--------|-----------|
| mAP50 | >90% | >95% |
| mAP50-95 | >75% | >85% |
| Precision | >85% | >95% |
| Recall | >85% | >95% |
| Inference | <50ms | <30ms |

---

## 📁 File Structure After Training

```
visionassist/
├── models/
│   └── currency_detector.pt    # ← Your trained model!
├── training/
│   ├── train_currency_model.py
│   ├── requirements.txt
│   ├── datasets/               # Downloaded training data
│   └── runs/                   # Training results
│       └── currency_detector/
│           ├── weights/
│           │   ├── best.pt     # Best model checkpoint
│           │   └── last.pt     # Latest checkpoint
│           └── results.png     # Training curves
└── src/
    └── currency_recognizer.py  # Uses the trained model
```

---

## 🚀 Using the Model in VisionAssist

Once the model is trained and in `models/currency_detector.pt`, the `CurrencyRecognizer` class in `src/currency_recognizer.py` will automatically find and use it.

```python
from src.currency_recognizer import CurrencyRecognizer

# Initialize (automatically loads models/currency_detector.pt)
recognizer = CurrencyRecognizer()

# Use in your main loop
result = recognizer.recognize(frame)
print(result)  # "$20 bill detected"
```

---

## 📞 Need Help?

1. Check the troubleshooting section above
2. Run with `--help` for all options: `python train_currency_model.py --help`
3. Check YOLO docs: https://docs.ultralytics.com
4. Check Roboflow docs: https://docs.roboflow.com

Good luck! 🍀
