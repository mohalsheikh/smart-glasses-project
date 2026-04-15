#!/usr/bin/env python3
"""
MediaPipe Diagnostic Script
===========================
Run this script to diagnose MediaPipe installation issues on your system.

Usage:
    python diagnose_mediapipe.py
"""

import sys
import platform

def main():
    print("=" * 60)
    print("MediaPipe Diagnostic Report")
    print("=" * 60)
    
    # System info
    print(f"\n📱 System Information:")
    print(f"   Python: {platform.python_version()}")
    print(f"   Platform: {platform.system()} {platform.release()}")
    print(f"   Architecture: {platform.machine()}")
    print(f"   Processor: {platform.processor()}")
    
    # Check if running on Apple Silicon
    is_apple_silicon = platform.system() == "Darwin" and platform.machine() == "arm64"
    if is_apple_silicon:
        print(f"\n⚠️  Running on Apple Silicon (ARM64)")
        print(f"   This may require special MediaPipe configuration.")
    
    # Try to import mediapipe
    print(f"\n📦 Checking MediaPipe installation...")
    try:
        import mediapipe
        print(f"   ✅ MediaPipe imported successfully")
        print(f"   Version: {mediapipe.__version__}")
        print(f"   Location: {mediapipe.__file__}")
    except ImportError as e:
        print(f"   ❌ MediaPipe NOT installed: {e}")
        print(f"\n   To install: pip install mediapipe")
        return
    except Exception as e:
        print(f"   ❌ Error importing MediaPipe: {e}")
        return
    
    # Check what's in the mediapipe module
    print(f"\n🔍 Checking MediaPipe modules...")
    mp_attrs = dir(mediapipe)
    print(f"   Available attributes: {len(mp_attrs)}")
    
    # Check for solutions
    has_solutions = hasattr(mediapipe, 'solutions')
    print(f"   Has 'solutions': {'✅ Yes' if has_solutions else '❌ No'}")
    
    if not has_solutions:
        print(f"\n   ⚠️  The 'solutions' module is missing!")
        print(f"   This can happen with certain MediaPipe builds.")
        print(f"\n   Try reinstalling MediaPipe:")
        print(f"   pip uninstall mediapipe")
        print(f"   pip install mediapipe")
        
        if is_apple_silicon:
            print(f"\n   For Apple Silicon, you may need:")
            print(f"   pip install mediapipe-silicon")
            print(f"   OR")
            print(f"   pip install --upgrade mediapipe")
        return
    
    # Check solutions contents
    print(f"\n🔍 Checking MediaPipe solutions...")
    solutions_attrs = dir(mediapipe.solutions)
    print(f"   Available solutions: {len(solutions_attrs)}")
    
    required_modules = ['hands', 'pose', 'face_mesh', 'drawing_utils']
    for mod in required_modules:
        has_mod = hasattr(mediapipe.solutions, mod)
        status = '✅' if has_mod else '❌'
        print(f"   {status} {mod}")
    
    # Try to create instances
    print(f"\n🧪 Testing MediaPipe components...")
    
    # Test Hands
    try:
        hands = mediapipe.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        print(f"   ✅ Hands - Works!")
        hands.close()
    except Exception as e:
        print(f"   ❌ Hands - Error: {e}")
    
    # Test Pose
    try:
        pose = mediapipe.solutions.pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        print(f"   ✅ Pose - Works!")
        pose.close()
    except Exception as e:
        print(f"   ❌ Pose - Error: {e}")
    
    # Test Face Mesh
    try:
        face = mediapipe.solutions.face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        print(f"   ✅ Face Mesh - Works!")
        face.close()
    except Exception as e:
        print(f"   ❌ Face Mesh - Error: {e}")
    
    # Test drawing utils
    try:
        draw = mediapipe.solutions.drawing_utils
        styles = mediapipe.solutions.drawing_styles
        print(f"   ✅ Drawing utilities - Available!")
    except Exception as e:
        print(f"   ❌ Drawing utilities - Error: {e}")
    
    print(f"\n" + "=" * 60)
    print("Diagnostic complete!")
    print("=" * 60)
    
    # Final recommendations
    print(f"\n💡 Recommendations:")
    if is_apple_silicon:
        print(f"""
For Apple Silicon Macs, try these steps if MediaPipe isn't working:

1. Ensure you have the latest pip:
   python -m pip install --upgrade pip

2. Install/reinstall MediaPipe:
   pip uninstall mediapipe
   pip install mediapipe

3. If that doesn't work, try:
   pip install --upgrade --force-reinstall mediapipe

4. Alternative for Apple Silicon:
   pip install mediapipe-silicon

5. Make sure you're using Python 3.9-3.11 (3.12 may have issues)
   Consider using a virtual environment with Python 3.11:
   python3.11 -m venv venv
   source venv/bin/activate
   pip install mediapipe opencv-python numpy
""")
    else:
        print(f"""
If MediaPipe isn't working:

1. Ensure you have the latest versions:
   pip install --upgrade mediapipe opencv-python numpy

2. Try reinstalling:
   pip uninstall mediapipe
   pip install mediapipe

3. Check Python version compatibility (3.9-3.11 recommended)
""")


if __name__ == "__main__":
    main()