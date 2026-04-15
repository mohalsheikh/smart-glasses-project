# 🚀 QUICK REFERENCE - Enhanced Smart Glasses

## Installation (5 minutes)

```bash
# 1. Extract the zip
unzip enhanced_smart_glasses_v2.zip
cd enhanced_smart_glasses

# 2. Create virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY=sk-...

# 5. Run!
python quick_start.py
```

## Essential Commands

### Voice Commands
```
"What do you see?"                    # Scene description
"Where did I leave my phone?"         # Memory search
"Is anyone smiling?"                  # Emotion detection
"What are they doing?"                # Activity recognition
"What color is this?"                 # Color identification
"Read this"                           # Text reading
"Translate to Spanish"                # Translation
"What brand is this?"                 # Product recognition
"Have I been here before?"            # Scene memory
```

### Keyboard Shortcuts
```
q - Quit
d - Describe scene
v - Voice mode
r - Read text
f - Analyze faces
c - Describe colors
m - Search memory
s - Toggle warnings
p - Proactive mode
```

## New Features Cheat Sheet

### Scene Memory
```python
from src.ai_features import SceneMemoryEngine

memory = SceneMemoryEngine()
memory.store_scene(description, detections)
results = memory.recall_similar("where's my laptop?")
```

### Emotion Detection
```python
from src.ai_features import EmotionFaceAnalyzer

analyzer = EmotionFaceAnalyzer()
faces = analyzer.analyze_faces_complete(frame)
description = analyzer.describe_faces_for_user(faces, frame_width)
```

### Advanced YOLO
```python
from src.ai_features import AdvancedObjectDetector

detector = AdvancedObjectDetector(enable_pose=True)
detections, frame = detector.detect_complete(frame)
# Each detection has: label, pose_action, estimated_distance
```

### Color Analysis
```python
from src.ai_features import ColorTextAnalyzer

analyzer = ColorTextAnalyzer()
colors = analyzer.describe_colors_gpt4o(frame)
brand = analyzer.identify_brand_or_product(frame, bbox)
```

### Proactive Assistant
```python
from src.ai_features import ProactiveAssistant

assistant = ProactiveAssistant()
alerts = assistant.analyze_scene_for_proactive_alerts(detections, frame_size)
# Automatic safety warnings!
```

## Configuration Presets

**In .env file:**
```bash
# Fast (30+ FPS) - Good for testing
CONFIG_PRESET=real_time

# Balanced (15-20 FPS) - Recommended
CONFIG_PRESET=balanced

# Best Quality (5-10 FPS)
CONFIG_PRESET=maximum_accuracy

# Minimal (fastest)
CONFIG_PRESET=low_power
```

## Troubleshooting

### Issue: Slow performance
```bash
# Solution 1: Use faster preset
CONFIG_PRESET=real_time

# Solution 2: Reduce frame processing
PROCESS_EVERY_N_FRAMES=3

# Solution 3: Lower resolution
FRAME_WIDTH=640
FRAME_HEIGHT=480
```

### Issue: Out of memory
```bash
# Disable heavy features
ENABLE_SEGMENTATION=0
MAX_SCENE_MEMORIES=100

# Use smaller YOLO model
YOLO_MODEL=yolov8n.pt
```

### Issue: API errors
```bash
# Check API key
echo $OPENAI_API_KEY

# Reduce API usage
USE_GPT4O_FOR_EMOTIONS=0
USE_GPT4O_FOR_COLORS=0
```

### Issue: Camera not found
```bash
# Try different camera index
CAMERA_INDEX=1  # or 2, 3, etc.

# Check available cameras
python -c "import cv2; print([i for i in range(10) if cv2.VideoCapture(i).isOpened()])"
```

## Key Files

```
README.md              # Complete documentation
FEATURES.md            # Feature guide with examples
UPGRADE_SUMMARY.md     # What's new
.env.example          # Configuration options
quick_start.py        # Easy launcher
src/ai_features/      # New AI modules
```

## Performance Tips

1. **GPU**: Ensure CUDA is available for 5-10x speedup
2. **Model size**: yolov8n (fastest) → yolov8x (most accurate)
3. **Frame rate**: Higher PROCESS_EVERY_N_FRAMES = faster
4. **Resolution**: Lower = faster, higher = more accurate
5. **Features**: Disable unused features for speed

## Best Practices

✅ **Start with real_time preset**
✅ **Enable features gradually**
✅ **Use GPU if available**
✅ **Read FEATURES.md for advanced usage**
✅ **Customize to your needs**

## Getting Help

1. Check README.md
2. Read FEATURES.md examples
3. Review .env.example comments
4. Check code docstrings
5. Review console error messages

## Quick Examples

**Complete scene analysis:**
```python
# One-liner scene understanding
report = detector.describe_detections_enhanced(detections, width)
faces = emotion.describe_faces_for_user(faces, width)
colors = color_analyzer.describe_colors_simple(frame)
```

**Memory-based assistance:**
```python
# Store observations
memory.store_scene("Kitchen with keys on counter", detections)

# Later retrieve
results = memory.recall_by_object("keys")
print(f"Keys last seen: {results[0].description}")
```

**Proactive safety:**
```python
# Automatic alerts
alerts = proactive.analyze_scene_for_proactive_alerts(detections, frame_size)
for alert in alerts:
    if alert.priority == "high":
        speak(alert.message)  # "Stairs ahead!"
```

---

## 🎉 You're Ready!

This is a powerful system with many features. Start simple and explore gradually.

**Recommended first steps:**
1. Run `python quick_start.py`
2. Try basic voice commands
3. Test keyboard shortcuts
4. Read FEATURES.md when ready
5. Customize .env for your needs

**Happy exploring! 🕶️**
