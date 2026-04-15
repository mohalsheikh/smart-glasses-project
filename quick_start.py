#!/usr/bin/env python3
"""
Quick Start Script for Enhanced Smart Glasses
Helps users get started quickly with sensible defaults
"""

import sys
import os
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()  # Load .env file BEFORE importing other modules
# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))


def check_requirements():
    """Check if all requirements are installed"""
    print("📋 Checking requirements...")
    
    required_packages = [
        "ultralytics",
        "cv2",  # opencv-python
        "openai",
        "numpy",
        "torch",
        "mediapipe",
        "sentence_transformers",
    ]
    
    missing = []
    for package in required_packages:
        try:
            __import__(package.replace("-", "_"))
        except ImportError:
            missing.append(package)
    
    if missing:
        print(f"❌ Missing packages: {', '.join(missing)}")
        print(f"\n📦 Install with: pip install {' '.join(missing)}")
        return False
    
    print("✅ All required packages installed")
    return True


def check_api_key():
    """Check if OpenAI API key is set"""
    print("\n🔑 Checking OpenAI API key...")
    
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("❌ OPENAI_API_KEY not set")
        print("\n📝 To set your API key:")
        print("   1. Copy .env.example to .env")
        print("   2. Add your OpenAI API key")
        print("   3. Run: source .env  (or set in your shell)")
        return False
    
    print("✅ OpenAI API key found")
    return True


def select_preset():
    """Let user select a configuration preset"""
    print("\n🎯 Select Configuration Preset:")
    print("   1. Real-time (Fast, good for testing)")
    print("   2. Balanced (Good balance of speed and accuracy)")
    print("   3. Maximum Accuracy (Slow, best quality)")
    print("   4. Custom (Use .env file settings)")
    
    choice = input("\nEnter choice (1-4) [default: 1]: ").strip() or "1"
    
    presets = {
        "1": "real_time",
        "2": "balanced",
        "3": "maximum_accuracy",
        "4": None
    }
    
    preset = presets.get(choice)
    if preset:
        print(f"✅ Using '{preset}' preset")
        os.environ["CONFIG_PRESET"] = preset
    else:
        print("✅ Using custom configuration from .env")
    
    return preset


def download_models():
    """Download required YOLO models"""
    print("\n📥 Checking YOLO models...")
    
    from ultralytics import YOLO
    
    models = ["yolov8n.pt", "yolov8n-pose.pt"]
    
    for model_name in models:
        try:
            print(f"   Loading {model_name}...")
            YOLO(model_name)
            print(f"   ✅ {model_name} ready")
        except Exception as e:
            print(f"   ❌ Failed to load {model_name}: {e}")
            return False
    
    return True


def print_usage_tips():
    """Print helpful usage tips"""
    print("\n" + "="*60)
    print("🕶️  ENHANCED SMART GLASSES - QUICK START")
    print("="*60)
    print("\n📚 Keyboard Controls:")
    print("   q - Quit")
    print("   d - Describe scene")
    print("   v - Voice command mode")
    print("   r - Read text")
    print("   f - Analyze faces/emotions")
    print("   c - Describe colors")
    print("   m - Memory: 'What did I see?'")
    print("   s - Toggle safety warnings")
    print("   p - Toggle proactive mode")
    
    print("\n🗣️  Voice Commands (examples):")
    print("   'What do you see?'")
    print("   'Is anyone here?'")
    print("   'Where did I leave my keys?'")
    print("   'What color is this?'")
    print("   'Read this sign'")
    
    print("\n📖 Documentation:")
    print("   README.md - Full documentation")
    print("   FEATURES.md - Feature guide with examples")
    print("   .env.example - Configuration options")
    
    print("\n🚀 Starting in 3 seconds...")
    print("="*60 + "\n")


def main():
    """Main quick start function"""
    print("\n" + "="*60)
    print("🕶️  Enhanced Smart Glasses - Quick Start Setup")
    print("="*60 + "\n")
    
    # Check requirements
    if not check_requirements():
        print("\n❌ Please install missing requirements first.")
        print("   Run: pip install -r requirements.txt")
        sys.exit(1)
    
    # Check API key
    api_key_ok = check_api_key()
    if not api_key_ok:
        print("\n⚠️  Some features will be limited without OpenAI API key.")
        cont = input("Continue anyway? (y/n) [default: y]: ").strip().lower() or "y"
        if cont != "y":
            sys.exit(0)
    
    # Select preset
    preset = select_preset()
    
    # Download models
    if not download_models():
        print("\n❌ Model download failed. Please check your internet connection.")
        sys.exit(1)
    
    # Print tips
    print_usage_tips()
    
    # Import and run
    try:
        import time
        time.sleep(3)
        
        print("🚀 Launching Enhanced Smart Glasses...\n")
        
        # Import controller
        from src.controller import MainController
        
        # Create and run
        controller = MainController()
        controller.run()
        
    except KeyboardInterrupt:
        print("\n\n👋 Goodbye!")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
