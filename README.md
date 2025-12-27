# 🕶️ Enhanced Smart Glasses for the Blind - Next Generation AI Assistant

A revolutionary assistive technology system combining **YOLOv8**, **OpenAI GPT-4o**, **MediaPipe**, and advanced AI to provide comprehensive environmental awareness for blind and visually impaired users.

## 🌟 Key Features

### **Core Vision Capabilities**

- **Advanced YOLOv8 Detection**: Object detection, tracking, and classification
- **Pose Estimation**: Human activity recognition (standing, sitting, walking, etc.)
- **Instance Segmentation**: Precise object boundary detection
- **Face Detection & Analysis**: MediaPipe-powered face tracking with landmarks
- **Emotion Recognition**: AI-powered facial expression analysis

### **AI-Powered Intelligence**

- **GPT-4o Vision**: Advanced scene understanding and description
- **Scene Memory System**: Semantic memory with embeddings for "Have I seen this before?" queries
- **Proactive AI Assistant**: Context-aware suggestions and safety warnings
- **Multi-language Support**: Real-time translation and language detection
- **Brand/Product Recognition**: Identify products, logos, and brands

### **Enhanced Navigation & Safety**

- **Real-time Obstacle Detection**: Multi-layer safety system
- **Depth Estimation**: Distance calculation using YOLOv8 and optional MiDaS
- **Guidance Engine**: Continuous walking guidance with spatial audio cues
- **Fall Detection**: Pose-based safety monitoring
- **Emergency Situation Detection**: Proactive alerts for dangerous scenarios

### **Advanced Text & Color Analysis**

- **Hybrid OCR**: EasyOCR, Tesseract, and GPT-4o vision
- **Color Identification**: Dominant color extraction and natural language description
- **Text Translation**: Real-time translation to any language
- **Context Analysis**: Understanding what text means in its environment
- **Clothing Description**: Identify clothing colors and styles

### **Personal Object Learning**

- **Object Memory**: Learn and remember user's personal items
- **Location Tracking**: "Where did I leave my keys?"
- **Usage Patterns**: Learn frequently used items and locations
- **Embeddings-based Search**: Semantic search for objects

### **Communication & Interaction**

- **Voice Commands**: Natural language understanding
- **Real-time Speech**: Whisper-powered transcription
- **Text-to-Speech**: Natural voice output
- **Conversation Memory**: Context-aware dialogue
- **Multi-turn Conversations**: Maintains conversation history

## 🚀 What's New in This Enhanced Version

### **1. Advanced YOLO Features**

```python
from src.ai_features import AdvancedObjectDetector

detector = AdvancedObjectDetector(
    enable_pose=True,           # Human pose estimation
    enable_segmentation=True    # Instance segmentation
)

detections, frame = detector.detect_complete(frame, annotate=True)

# Access enhanced information
for det in detections:
    print(f"Object: {det.label}")
    print(f"Activity: {det.pose_action}")  # e.g., "sitting", "standing"
    print(f"Distance: {det.estimated_distance}m")
```

### **2. Scene Memory System**

```python
from src.ai_features import SceneMemoryEngine

memory = SceneMemoryEngine()

# Store current scene
memory.store_scene(
    description="Kitchen with stove and sink",
    detections=current_detections,
    importance=1.0
)

# Recall similar scenes
similar = memory.recall_similar("Where did I see a microwave?", top_k=3)

# Find specific objects
keys_locations = memory.recall_by_object("keys")
```

### **3. Emotion & Face Analysis**

```python
from src.ai_features import EmotionFaceAnalyzer

analyzer = EmotionFaceAnalyzer()

# Detect and analyze faces
faces = analyzer.analyze_faces_complete(frame, use_ai=True)

for face in faces:
    print(f"Emotion: {face.emotion}")
    print(f"Looking: {face.gaze_direction}")
    print(f"Expression: {face.facial_expression}")
```

### **4. Proactive AI Assistant**

```python
from src.ai_features import ProactiveAssistant

assistant = ProactiveAssistant(enable_proactive=True)

# Automatically generates alerts
alerts = assistant.analyze_scene_for_proactive_alerts(detections, frame_size)

# Contextual responses
response = assistant.generate_smart_response(
    user_query="What's around me?",
    scene_detections=detections,
    conversation_history=history
)
```

### **5. Color & Text Analysis**

```python
from src.ai_features import ColorTextAnalyzer

analyzer = ColorTextAnalyzer()

# Describe colors naturally
color_desc = analyzer.describe_colors_gpt4o(frame)  # "Mostly blue with yellow accents"

# Identify brands
brand = analyzer.identify_brand_or_product(frame, bbox)  # "Coca-Cola bottle"

# Translate text
translation = analyzer.translate_visible_text(text, target_language="es")

# Analyze clothing
clothing = analyzer.describe_clothing_colors(frame, person_bbox)
```

## 📋 Installation

### Prerequisites

- Python 3.8+
- CUDA-capable GPU (recommended for real-time performance)
- Webcam or camera device
- OpenAI API key

### Quick Setup

```bash
# Clone or extract the project
cd enhanced_smart_glasses

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download YOLO models (optional, auto-downloads on first run)
python -c "from ultralytics import YOLO; YOLO('yolov8n.pt'); YOLO('yolov8n-pose.pt'); YOLO('yolov8n-seg.pt')"

# Set up environment variables
cp .env.example .env
# Edit .env and add your OpenAI API key
```

### Environment Variables

Create a `.env` file:

```bash
# OpenAI Configuration
OPENAI_API_KEY=your_api_key_here
OPENAI_VISION_MODEL=gpt-4o
OPENAI_CHAT_MODEL=gpt-4o-mini

# Camera Settings
CAMERA_INDEX=0
FRAME_WIDTH=1280
FRAME_HEIGHT=720

# YOLO Settings
YOLO_MODEL=yolov8n.pt
YOLO_POSE_MODEL=yolov8n-pose.pt
YOLO_CONF=0.25
ENABLE_POSE=1
ENABLE_SEGMENTATION=0

# AI Features
ENABLE_SCENE_MEMORY=1
ENABLE_PROACTIVE_ASSISTANT=1
ENABLE_EMOTION_DETECTION=1
ENABLE_FACE_ANALYSIS=1

# Performance
USE_GPU=1
PROCESS_EVERY_N_FRAMES=2

# Navigation
ORS_API_KEY=your_openrouteservice_key  # Optional for navigation
```

## 🎮 Usage

### Basic Usage

```bash
# Run the enhanced smart glasses system
python src/controller.py

# Or run with specific features
python src/controller.py --enable-all-features
```

### Keyboard Controls

| Key   | Function                    |
| ----- | --------------------------- |
| `q`   | Quit application            |
| `d`   | Describe current scene      |
| `v`   | Voice interaction mode      |
| `r`   | Read text on screen         |
| `s`   | Toggle safety warnings      |
| `f`   | Analyze faces and emotions  |
| `c`   | Describe colors             |
| `m`   | Memory: "Have I seen this?" |
| `p`   | Toggle proactive mode       |
| `t`   | Translate visible text      |
| `1-3` | Change reading mode         |

### Voice Commands

The system understands natural language:

**Scene Understanding:**

- "What do you see?"
- "Describe the scene"
- "Is anyone in front of me?"
- "What colors do you see?"

**Object Finding:**

- "Where are my keys?"
- "Find my phone"
- "Is there a chair nearby?"

**Safety & Navigation:**

- "Any obstacles?"
- "Is it safe to walk?"
- "Where's the door?"
- "Guide me"

**Text & Reading:**

- "Read this"
- "What does this sign say?"
- "Translate this to Spanish"

**Memory & Context:**

- "Have I been here before?"
- "What was in this room last time?"
- "Where did I see my laptop?"

**People & Social:**

- "Who's there?"
- "How many people?"
- "Is anyone smiling?"
- "What are they doing?"

**Advanced:**

- "What brand is this?"
- "Identify this product"
- "Describe this person's clothing"
- "What's the mood here?"

## 🏗️ Architecture

```
enhanced_smart_glasses/
├── src/
│   ├── ai_features/          # 🆕 Advanced AI capabilities
│   │   ├── scene_memory.py
│   │   ├── emotion_analyzer.py
│   │   ├── advanced_yolo.py
│   │   ├── color_text_analyzer.py
│   │   └── proactive_assistant.py
│   ├── brain/                # Decision making
│   │   ├── assistant_brain_impl.py
│   │   ├── intent_detection.py
│   │   ├── openai_client.py
│   │   └── handlers/
│   ├── safety/               # Safety systems
│   │   ├── obstacle_layer.py
│   │   ├── depth_estimator.py
│   │   └── guidance_engine.py
│   ├── utils/                # Utilities
│   ├── controller.py         # 🆕 Enhanced main controller
│   ├── object_detector.py
│   ├── scene_ai_client.py
│   ├── ocr_engine.py
│   └── ...
├── models/                   # YOLO models
├── runtime/                  # Runtime data
└── requirements.txt
```

## 🎯 Advanced Examples

### Example 1: Complete Scene Analysis

```python
from src.ai_features import *
from src.controller import MainController

controller = MainController()

# Initialize enhanced features
detector = AdvancedObjectDetector(enable_pose=True)
memory = SceneMemoryEngine()
emotion_analyzer = EmotionFaceAnalyzer()
color_analyzer = ColorTextAnalyzer()
proactive = ProactiveAssistant()

# Capture and analyze
frame = controller.camera.capture_frame()

# Detect everything
detections, annotated = detector.detect_complete(frame, annotate=True)

# Analyze faces and emotions
faces = emotion_analyzer.analyze_faces_complete(frame)

# Get colors
colors = color_analyzer.describe_colors_gpt4o(frame)

# Store in memory
memory.store_scene(
    description=f"Scene with {len(detections)} objects, {colors}",
    detections=detections
)

# Check for proactive alerts
alerts = proactive.analyze_scene_for_proactive_alerts(detections, frame.shape)

# Generate comprehensive description
description = detector.describe_detections_enhanced(detections, frame.shape[1])
face_description = emotion_analyzer.describe_faces_for_user(faces, frame.shape[1])

print(f"Scene: {description}")
print(f"People: {face_description}")
print(f"Colors: {colors}")
```

### Example 2: Memory-Based Assistance

```python
# Store observations over time
memory.store_scene("Living room with laptop on table", detections, importance=2.0)
memory.store_scene("Kitchen with keys on counter", detections, importance=3.0)
memory.store_scene("Bedroom with phone on nightstand", detections, importance=2.0)

# Later, user asks: "Where did I leave my phone?"
results = memory.recall_by_object("phone", top_k=5)
if results:
    last_seen = results[0]
    print(f"Phone last seen: {last_seen.description}")
    print(f"Location: {last_seen.location}")
    print(f"Time: {time.time() - last_seen.timestamp:.0f} seconds ago")
```

### Example 3: Proactive Safety Monitoring

```python
proactive = ProactiveAssistant(enable_proactive=True)

while True:
    frame = capture_frame()
    detections = detect_objects(frame)

    # Update context
    proactive.update_scene_context(detections, location_type="outdoor")

    # Get alerts
    alerts = proactive.analyze_scene_for_proactive_alerts(
        detections,
        frame.shape
    )

    # Handle urgent alerts immediately
    for alert in alerts:
        if proactive.should_interrupt_for_safety(alert):
            speak_immediately(alert.message)
        else:
            queue_alert(alert)
```

## 🔧 Configuration

Key configuration options in `src/utils/config.py`:

```python
# AI Features
ENABLE_SCENE_MEMORY = True
ENABLE_EMOTION_DETECTION = True
ENABLE_PROACTIVE_ASSISTANT = True
MAX_MEMORIES = 500

# YOLO Enhancement
ENABLE_POSE_ESTIMATION = True
ENABLE_SEGMENTATION = False  # Heavy, disable for real-time
YOLO_POSE_MODEL = "yolov8n-pose.pt"

# OpenAI Settings
OPENAI_VISION_MODEL = "gpt-4o"
OPENAI_CHAT_MODEL = "gpt-4o-mini"
SCENE_AI_MAX_TOKENS = 300

# Performance
PROCESS_EVERY_N_FRAMES = 2
USE_GPU = True
USE_HALF_PRECISION = True

# Proactive Assistant
PROACTIVE_SAFETY_PRIORITY = "high"
PROACTIVE_INFO_COOLDOWN = 30  # seconds
```

## 📊 Performance Optimization

**For Real-time Performance:**

1. **Use lighter YOLO models**: `yolov8n.pt` instead of `yolov8x.pt`
2. **Reduce frame processing frequency**: `PROCESS_EVERY_N_FRAMES = 3`
3. **Disable heavy features**: Set `ENABLE_SEGMENTATION = False`
4. **Use GPU**: Ensure CUDA is properly installed
5. **Lower resolution**: `FRAME_WIDTH = 640`

**For Maximum Accuracy:**

1. **Use larger models**: `yolov8x.pt`, `yolov8x-pose.pt`
2. **Process every frame**: `PROCESS_EVERY_N_FRAMES = 1`
3. **Enable all features**: Segmentation, emotion detection, etc.
4. **Higher resolution**: `FRAME_WIDTH = 1280`

## 🤝 Contributing

Contributions are welcome! Areas for improvement:

- [ ] Add more language models for offline operation
- [ ] Implement actual hardware ToF sensor support
- [ ] Add GPS integration for outdoor navigation
- [ ] Develop mobile app companion
- [ ] Create custom training pipeline for personal object recognition
- [ ] Add support for AR glasses displays

## 📄 License

MIT License - See LICENSE file for details

## 🙏 Acknowledgments

- **Ultralytics** - YOLOv8 framework
- **OpenAI** - GPT-4o Vision and language models
- **MediaPipe** - Face detection and pose estimation
- **EasyOCR** - Text recognition
- **The blind and visually impaired community** for feedback and testing

## 📞 Support

For issues, questions, or suggestions:

- GitHub Issues: [Link to repo]
- Email: support@example.com
- Discord: [Community link]

---

**Made with ❤️ for the blind and visually impaired community**

_This enhanced version pushes the boundaries of what's possible with AI-assisted vision technology._
