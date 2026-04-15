# 🎯 Feature Guide - Enhanced Smart Glasses

## Table of Contents
1. [Quick Feature Overview](#quick-feature-overview)
2. [Voice Command Examples](#voice-command-examples)
3. [Advanced Features Deep Dive](#advanced-features-deep-dive)
4. [API Usage Examples](#api-usage-examples)
5. [Performance Tuning](#performance-tuning)
6. [Troubleshooting](#troubleshooting)

---

## Quick Feature Overview

### 🎨 What Makes This Version Special?

| Feature | Description | Use Case |
|---------|-------------|----------|
| **Scene Memory** | AI remembers places and objects | "Where did I leave my keys?" |
| **Emotion Detection** | Reads facial expressions | "Is anyone smiling?" |
| **Pose Estimation** | Knows what people are doing | "Is anyone sitting?" |
| **Proactive Alerts** | Warns before you ask | Stairs ahead, person waiting |
| **Color Analysis** | Describes colors naturally | "What color is this shirt?" |
| **Brand Recognition** | Identifies products | "What brand is this?" |
| **Multi-language** | Translates text on the fly | Signs in foreign languages |
| **Advanced Tracking** | Follows objects over time | Track moving people |

---

## Voice Command Examples

### 🗣️ Basic Commands

**Scene Understanding:**
```
"What's in front of me?"
"Describe what you see"
"What's around me?"
"Give me a detailed description"
```

**Object Detection:**
```
"Is there a chair nearby?"
"Where's the door?"
"Count the people"
"What objects are on the table?"
```

**Navigation:**
```
"Any obstacles?"
"Is it safe to walk forward?"
"Guide me to the door"
"What's on my left?"
```

### 🆕 Enhanced Commands

**Scene Memory:**
```
"Have I been in this room before?"
"What was here last time?"
"Where did I see my laptop?"
"When did I last see my phone?"
"Show me places where I've seen books"
```

**People & Emotions:**
```
"Is anyone here?"
"How many people?"
"Are they happy or sad?"
"What are they doing?"
"Is anyone looking at me?"
"Describe the person's expression"
```

**Color Analysis:**
```
"What color is this?"
"Describe the colors you see"
"What color is the wall?"
"Is this shirt blue or green?"
"What are the main colors here?"
```

**Text & Translation:**
```
"Read this"
"Translate this to Spanish"
"What language is this?"
"Read the sign"
"What does this label say?"
```

**Product Recognition:**
```
"What brand is this?"
"Identify this product"
"What's in this package?"
"Is this Coca-Cola?"
```

**Clothing:**
```
"What am I wearing?"
"Describe my shirt"
"What color are my pants?"
"Do these colors match?"
```

**Advanced Queries:**
```
"What's the mood in this room?"
"Describe the atmosphere"
"Is this a safe place to sit?"
"What's the most interesting thing here?"
"Give me context about this scene"
```

---

## Advanced Features Deep Dive

### 1. Scene Memory System

**How it works:**
- Uses OpenAI embeddings to create semantic representations of scenes
- Stores up to 500 memories by default
- Searches using similarity matching
- Categorizes scenes automatically (kitchen, outdoor, etc.)

**Example Usage:**
```python
from src.ai_features import SceneMemoryEngine

memory = SceneMemoryEngine()

# Store a scene
memory.store_scene(
    description="Living room with laptop on coffee table",
    detections=detections,
    location="home_living_room",
    importance=2.0  # Higher = more important
)

# Search for similar scenes
results = memory.recall_similar(
    query="room with computer",
    top_k=5,
    time_window_hours=24,  # Last 24 hours
    min_similarity=0.7
)

# Find specific objects
laptop_scenes = memory.recall_by_object("laptop", top_k=5)

# Get context summary
summary = memory.get_context_summary(recent_minutes=5)
print(summary)  # "Recent scenes: living_room. Objects seen: laptop, phone, book"
```

**Advanced Features:**
- Export memories to JSON for backup
- Clear old memories (configurable retention)
- Access frequency tracking
- Importance weighting

### 2. Emotion & Face Analysis

**Capabilities:**
- MediaPipe face detection (fast, real-time)
- 468-point face mesh landmarks
- Gaze direction estimation
- GPT-4o emotion analysis (optional, accurate)

**Example Usage:**
```python
from src.ai_features import EmotionFaceAnalyzer

analyzer = EmotionFaceAnalyzer()

# Detect all faces
faces = analyzer.detect_faces(frame)
print(f"Found {len(faces)} faces")

# Get detailed analysis
faces = analyzer.analyze_faces_complete(frame, use_ai=True)

for face in faces:
    print(f"Position: {face.bbox}")
    print(f"Confidence: {face.confidence}")
    print(f"Emotion: {face.emotion}")
    print(f"Looking: {face.gaze_direction}")
    print(f"Expression: {face.facial_expression}")

# Natural language description
description = analyzer.describe_faces_for_user(faces, frame_width)
# Output: "One person in front of you, looking forward, appears happy."
```

**Use Cases:**
- Social situations: "Is anyone uncomfortable?"
- Meetings: "Is the audience engaged?"
- Customer service: "Is the person satisfied?"
- Personal: "Am I smiling in this photo?"

### 3. Advanced YOLO with Pose

**Enhanced Detection:**
- Standard object detection (80+ classes)
- Human pose estimation (17 keypoints)
- Activity recognition (sitting, standing, walking, lying down, bending)
- Distance estimation
- Instance segmentation (optional)

**Example Usage:**
```python
from src.ai_features import AdvancedObjectDetector

detector = AdvancedObjectDetector(
    enable_pose=True,
    enable_segmentation=False
)

# Complete detection
detections, annotated = detector.detect_complete(frame, annotate=True)

for det in detections:
    print(f"Object: {det.label}")
    print(f"Confidence: {det.confidence:.2f}")
    
    if det.pose_action:
        print(f"Activity: {det.pose_action}")
    
    if det.estimated_distance:
        print(f"Distance: {det.estimated_distance:.1f}m")
    
    if det.mask is not None:
        print(f"Mask area: {det.mask_area} pixels")

# Natural language description
description = detector.describe_detections_enhanced(detections, frame_width)
# Output: "I see: person standing ahead, very close; person sitting left, nearby; car (right)"
```

**Activities Recognized:**
- standing
- sitting  
- walking (with motion tracking)
- lying_down
- bending
- unknown (when unclear)

### 4. Proactive AI Assistant

**What it does:**
- Monitors scene continuously
- Generates contextual alerts
- Prioritizes safety warnings
- Learns user patterns
- Provides unsolicited helpful information

**Alert Types:**
1. **Safety** (high priority): Stairs, obstacles, vehicles
2. **Navigation** (medium): Doors, exits, paths
3. **Information** (low): Text visible, people waiting
4. **Assistance** (low): Items left behind, suggestions

**Example Usage:**
```python
from src.ai_features import ProactiveAssistant

assistant = ProactiveAssistant(enable_proactive=True)

# Update context
assistant.update_scene_context(detections, location_type="kitchen")

# Get alerts
alerts = assistant.analyze_scene_for_proactive_alerts(
    detections,
    frame_size=(720, 1280)
)

for alert in alerts:
    print(f"[{alert.priority}] {alert.category}: {alert.message}")
    
    # Check if urgent
    if assistant.should_interrupt_for_safety(alert):
        speak_immediately(alert.message)

# Get contextual suggestions
suggestions = assistant.get_contextual_suggestions(user_query)

# Generate smart response
response = assistant.generate_smart_response(
    user_query="What should I do?",
    scene_detections=detections,
    conversation_history=history
)
```

**Customization:**
- Adjust cooldown periods per alert type
- Set priority levels
- Configure interruption rules
- Add custom triggers

### 5. Color & Text Analysis

**Color Capabilities:**
- Dominant color extraction (k-means clustering)
- 30+ color names (red, blue, navy, turquoise, etc.)
- GPT-4o natural descriptions ("vibrant red with hints of orange")
- Color matching and comparison
- Clothing color identification

**Example Usage:**
```python
from src.ai_features import ColorTextAnalyzer

analyzer = ColorTextAnalyzer()

# Simple color description
colors = analyzer.describe_colors_simple(frame)
# Output: "Mostly blue and white."

# Advanced AI description
colors_ai = analyzer.describe_colors_gpt4o(frame)
# Output: "The image features a vibrant blue sky with white puffy clouds."

# Extract dominant colors
dominant = analyzer.extract_dominant_colors(frame, n_colors=5)
for color, percentage in dominant:
    name = analyzer.get_basic_color_name(color)
    print(f"{name}: {percentage*100:.1f}%")

# Object-specific color
object_bbox = (100, 100, 300, 400)
color_desc = analyzer.analyze_color_of_object(frame, object_bbox, use_ai=True)

# Clothing analysis
person_bbox = (50, 50, 200, 500)
clothing = analyzer.describe_clothing_colors(frame, person_bbox)
# Output: "blue top, black bottom"

# Brand recognition
brand = analyzer.identify_brand_or_product(frame, bbox)
# Output: "Nike running shoe"

# Translation
translation = analyzer.translate_visible_text(
    "Hola, ¿cómo estás?",
    target_language="en"
)
print(translation)
# {
#   "original": "Hola, ¿cómo estás?",
#   "translated": "Hello, how are you?",
#   "language": "Spanish"
# }
```

**Text Analysis:**
- Multi-language OCR
- Context understanding
- Real-time translation
- Text type classification (sign, label, document)

---

## API Usage Examples

### Integration Example: Complete Scene Analysis

```python
from src.ai_features import *
import cv2 as cv

# Initialize all systems
detector = AdvancedObjectDetector(enable_pose=True)
memory = SceneMemoryEngine()
emotion = EmotionFaceAnalyzer()
colors = ColorTextAnalyzer()
proactive = ProactiveAssistant()

def analyze_complete_scene(frame):
    """Comprehensive scene analysis"""
    
    # 1. Object Detection with Pose
    detections, annotated = detector.detect_complete(frame, annotate=True)
    
    # 2. Face & Emotion Analysis
    faces = emotion.analyze_faces_complete(frame, use_ai=False)
    
    # 3. Color Analysis
    color_desc = colors.describe_colors_simple(frame)
    
    # 4. Check Memory
    scene_desc = detector.describe_detections_enhanced(detections, frame.shape[1])
    similar_scenes = memory.recall_similar(scene_desc, top_k=3)
    
    # 5. Store Current Scene
    memory.store_scene(scene_desc, detections, importance=1.0)
    
    # 6. Proactive Alerts
    alerts = proactive.analyze_scene_for_proactive_alerts(
        detections,
        frame.shape
    )
    
    # 7. Compile Report
    report = {
        "objects": len(detections),
        "people": len(faces),
        "description": scene_desc,
        "colors": color_desc,
        "faces": emotion.describe_faces_for_user(faces, frame.shape[1]),
        "alerts": [a.message for a in alerts],
        "seen_before": len(similar_scenes) > 0,
        "similar_scenes": [(s.description, score) for s, score in similar_scenes]
    }
    
    return report, annotated

# Usage
cap = cv.VideoCapture(0)
ret, frame = cap.read()

report, viz = analyze_complete_scene(frame)
print(report)
cv.imshow("Analysis", viz)
cv.waitKey(0)
```

### Integration Example: Real-time Monitoring

```python
def real_time_monitoring():
    """Real-time scene monitoring with all features"""
    
    cap = cv.VideoCapture(0)
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        # Process every 3 frames
        if frame_count % 3 == 0:
            # Quick detection
            detections, annotated = detector.detect_with_tracking(frame)
            
            # Update proactive context
            proactive.update_scene_context(detections)
            
            # Check for urgent alerts
            alerts = proactive.analyze_scene_for_proactive_alerts(
                detections,
                frame.shape
            )
            
            for alert in alerts:
                if alert.priority == "high":
                    print(f"⚠️ {alert.message}")
            
            # Show visualization
            cv.imshow("Monitoring", annotated)
        
        # Process every 30 frames (1 second at 30fps)
        if frame_count % 30 == 0:
            # Deep analysis
            faces = emotion.detect_faces(frame)
            if faces:
                print(f"👥 Detected {len(faces)} people")
            
            # Store in memory
            desc = detector.describe_detections_enhanced(detections, frame.shape[1])
            memory.store_scene(desc, detections)
        
        if cv.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv.destroyAllWindows()
```

---

## Performance Tuning

### Speed vs Quality Trade-offs

**Maximum Speed (30+ FPS):**
```python
# Configuration
YOLO_MODEL = "yolov8n.pt"
ENABLE_POSE = False
ENABLE_SEGMENTATION = False
PROCESS_EVERY_N_FRAMES = 3
FRAME_WIDTH = 480
FRAME_HEIGHT = 360
USE_GPU = True
```

**Balanced (15-20 FPS):**
```python
YOLO_MODEL = "yolov8s.pt"
ENABLE_POSE = True
ENABLE_SEGMENTATION = False
PROCESS_EVERY_N_FRAMES = 2
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
```

**Maximum Quality (5-10 FPS):**
```python
YOLO_MODEL = "yolov8x.pt"
YOLO_POSE_MODEL = "yolov8x-pose.pt"
ENABLE_SEGMENTATION = True
PROCESS_EVERY_N_FRAMES = 1
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720
USE_GPT4O_FOR_EMOTIONS = True
```

### GPU Optimization

```python
import torch

# Check CUDA availability
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA devices: {torch.cuda.device_count()}")

# Use half precision for 2x speed
USE_HALF_PRECISION = True

# Enable TensorRT for even faster inference (requires setup)
# model.export(format="engine")  # One-time export
# model = YOLO("model.engine")  # Load TensorRT engine
```

---

## Troubleshooting

### Common Issues

**1. "CUDA out of memory"**
- Reduce `FRAME_WIDTH` and `FRAME_HEIGHT`
- Increase `PROCESS_EVERY_N_FRAMES`
- Use smaller YOLO model (n instead of x)
- Disable segmentation

**2. "Slow performance"**
- Enable GPU: `USE_GPU=1`
- Use half precision: `USE_HALF_PRECISION=1`
- Process fewer frames: `PROCESS_EVERY_N_FRAMES=3`
- Close other GPU applications

**3. "OpenAI API errors"**
- Check API key is set correctly
- Verify internet connection
- Check rate limits
- Reduce API calls (disable GPT-4o for emotions/colors)

**4. "Camera not opening"**
- Try different `CAMERA_INDEX` (0, 1, 2)
- Check camera permissions
- Close other apps using camera
- Try lower resolution

**5. "Memory errors"**
- Reduce `MAX_SCENE_MEMORIES`
- Clear old memories: `memory.clear_old_memories(days=7)`
- Disable features you don't need

**6. "Poor detection accuracy"**
- Improve lighting
- Use higher resolution
- Use larger YOLO model
- Lower confidence threshold

### Performance Monitoring

```python
import time

def benchmark_features():
    """Benchmark each feature"""
    frame = capture_frame()
    
    # Detection
    start = time.time()
    detections, _ = detector.detect_complete(frame)
    print(f"Detection: {(time.time()-start)*1000:.1f}ms")
    
    # Pose
    start = time.time()
    poses = detector.detect_poses(frame)
    print(f"Pose: {(time.time()-start)*1000:.1f}ms")
    
    # Faces
    start = time.time()
    faces = emotion.detect_faces(frame)
    print(f"Faces: {(time.time()-start)*1000:.1f}ms")
    
    # Colors
    start = time.time()
    colors = color_analyzer.extract_dominant_colors(frame)
    print(f"Colors: {(time.time()-start)*1000:.1f}ms")
```

---

## Best Practices

1. **Start Simple**: Begin with core features, add advanced ones gradually
2. **Profile First**: Identify bottlenecks before optimizing
3. **Use Presets**: Try built-in configuration presets
4. **Cache Results**: Don't recompute expensive operations
5. **Batch Process**: Process multiple frames together when possible
6. **Monitor Memory**: Track RAM/VRAM usage
7. **Test Thoroughly**: Validate in real-world conditions

---

For more information, see:
- [README.md](README.md) - Main documentation
- [API Documentation](docs/api.md) - Detailed API reference
- [Examples](examples/) - More code examples
