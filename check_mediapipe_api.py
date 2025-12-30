#!/usr/bin/env python3
"""
Check what's available in MediaPipe 0.10.31+
"""

import mediapipe as mp

print("MediaPipe version:", mp.__version__)
print("\nTop-level attributes:")
for attr in sorted(dir(mp)):
    if not attr.startswith('_'):
        print(f"  - {attr}")

# Check if tasks API is available (new API in 0.10+)
if hasattr(mp, 'tasks'):
    print("\n✅ 'tasks' module found (new API)")
    print("\nTasks sub-modules:")
    for attr in sorted(dir(mp.tasks)):
        if not attr.startswith('_'):
            print(f"  - {attr}")
    
    # Check for vision tasks
    if hasattr(mp.tasks, 'vision'):
        print("\nVision tasks:")
        for attr in sorted(dir(mp.tasks.vision)):
            if not attr.startswith('_'):
                print(f"  - {attr}")

# Try importing via the python.solutions path
print("\n" + "="*50)
print("Trying alternative import paths...")
print("="*50)

try:
    from mediapipe.python.solutions import hands
    print("✅ from mediapipe.python.solutions import hands - WORKS")
except ImportError as e:
    print(f"❌ mediapipe.python.solutions.hands - {e}")

try:
    from mediapipe.python.solutions import pose
    print("✅ from mediapipe.python.solutions import pose - WORKS")
except ImportError as e:
    print(f"❌ mediapipe.python.solutions.pose - {e}")

try:
    from mediapipe.python.solutions import face_mesh
    print("✅ from mediapipe.python.solutions import face_mesh - WORKS")
except ImportError as e:
    print(f"❌ mediapipe.python.solutions.face_mesh - {e}")

try:
    from mediapipe.python.solutions import drawing_utils
    print("✅ from mediapipe.python.solutions import drawing_utils - WORKS")
except ImportError as e:
    print(f"❌ mediapipe.python.solutions.drawing_utils - {e}")

# Try the tasks API for hands
print("\n" + "="*50)
print("Trying Tasks API (new MediaPipe API)...")
print("="*50)

try:
    from mediapipe.tasks.python import vision
    print("✅ mediapipe.tasks.python.vision - Available")
    
    if hasattr(vision, 'HandLandmarker'):
        print("✅ HandLandmarker found!")
    if hasattr(vision, 'PoseLandmarker'):
        print("✅ PoseLandmarker found!")
    if hasattr(vision, 'FaceLandmarker'):
        print("✅ FaceLandmarker found!")
except ImportError as e:
    print(f"❌ Tasks API not available: {e}")