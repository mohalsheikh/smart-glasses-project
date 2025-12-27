#!/bin/bash
# macOS ARM Installation Script for Enhanced Smart Glasses
# This script will install dependencies step-by-step to avoid conflicts

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
pip install torch==2.0.1 torchvision==0.15.2

echo ""
echo "📦 Stage 3: Computer Vision..."
pip install opencv-python==4.10.0.84
pip install ultralytics==8.3.0
pip install mediapipe==0.10.14

echo ""
echo "📦 Stage 4: OpenAI..."
pip install openai==1.55.0
pip install tiktoken==0.8.0

echo ""
echo "📦 Stage 5: NLP & Transformers..."
pip install sentence-transformers==3.2.0
pip install transformers==4.45.0

echo ""
echo "📦 Stage 6: OCR..."
pip install easyocr==1.7.2
pip install pytesseract==0.3.13

echo ""
echo "📦 Stage 7: Audio..."
pip install pyttsx3==2.90
pip install SpeechRecognition==3.10.4

echo ""
echo "📦 Stage 8: Data Science..."
pip install pandas==2.2.3
pip install scipy==1.14.1
pip install scikit-learn==1.5.2
pip install scikit-image==0.24.0

echo ""
echo "📦 Stage 9: Image Processing..."
pip install albumentations==1.4.20

echo ""
echo "📦 Stage 10: Utilities..."
pip install python-dotenv==1.0.1
pip install pydantic==2.9.2
pip install colorama==0.4.6
pip install tqdm==4.66.5
pip install requests==2.32.3
pip install geopy==2.4.1
pip install joblib==1.4.2

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
