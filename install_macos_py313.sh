#!/bin/bash
# macOS ARM Installation Script for Enhanced Smart Glasses
# Updated for Python 3.13 compatibility

set -e  # Exit on error

echo "🍎 Enhanced Smart Glasses - macOS ARM Installation"
echo "=================================================="
echo ""

# Check Python version
echo "📋 Checking Python version..."
python3 --version
if [ $? -ne 0 ]; then
    echo "❌ Python 3 not found. Please install Python 3.8 or higher."
    exit 1
fi

# Create virtual environment
echo ""
echo "🔧 Creating virtual environment..."
python3 -m venv venv
source venv/bin/activate

# Upgrade pip
echo ""
echo "⬆️  Upgrading pip..."
pip install --upgrade pip setuptools wheel

# Install in stages to avoid conflicts
echo ""
echo "📦 Stage 1: Core dependencies..."
pip install numpy==1.26.4
pip install pillow==10.4.0

echo ""
echo "📦 Stage 2: PyTorch (this may take a while)..."
# Use latest PyTorch compatible with Python 3.13
pip install torch torchvision

echo ""
echo "📦 Stage 3: Computer Vision..."
pip install opencv-python
pip install ultralytics
pip install mediapipe==0.10.14

echo ""
echo "📦 Stage 4: OpenAI..."
pip install openai
pip install tiktoken

echo ""
echo "📦 Stage 5: NLP & Transformers..."
pip install sentence-transformers
pip install transformers

echo ""
echo "📦 Stage 6: OCR..."
pip install easyocr
pip install pytesseract

echo ""
echo "📦 Stage 7: Audio..."
pip install pyttsx3
pip install SpeechRecognition

echo ""
echo "📦 Stage 8: Data Science..."
pip install pandas
pip install scipy
pip install scikit-learn
pip install scikit-image

echo ""
echo "📦 Stage 9: Image Processing..."
pip install albumentations

echo ""
echo "📦 Stage 10: Utilities..."
pip install python-dotenv
pip install pydantic
pip install colorama
pip install tqdm
pip install requests
pip install geopy
pip install joblib

echo ""
echo "✅ Installation complete!"
echo ""
echo "🧪 Testing imports..."
python3 -c "
import ultralytics
import cv2
import numpy
import torch
import openai
import mediapipe
import sentence_transformers
print('✅ All critical packages imported successfully!')
"

if [ $? -eq 0 ]; then
    echo ""
    echo "🎉 Success! All packages installed and working."
    echo ""
    echo "Next steps:"
    echo "1. Configure: cp .env.example .env"
    echo "2. Add your OPENAI_API_KEY to .env"
    echo "3. Run: python quick_start.py"
else
    echo ""
    echo "⚠️  Some packages failed to import. Check the error messages above."
fi
