# VisionAssist AI - Smart Glasses for the Visually Impaired

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.11+-blue.svg" alt="Python">
  <img src="https://img.shields.io/badge/OpenCV-4.8+-green.svg" alt="OpenCV">
  <img src="https://img.shields.io/badge/YOLOv8-Ultralytics-red.svg" alt="YOLO">
  <img src="https://img.shields.io/badge/MediaPipe-Latest-orange.svg" alt="MediaPipe">
  <img src="https://img.shields.io/badge/Platform-macOS%20%7C%20Linux-lightgrey.svg" alt="Platform">
</p>

An AI-powered smart glasses system designed to assist visually impaired users through real-time computer vision, natural language processing, and audio feedback. The system provides object detection, scene description, text reading, navigation assistance, sign language interpretation, and more.

---

## 🌟 Features

### Core Features

| Feature                   | Description                                                           |
| ------------------------- | --------------------------------------------------------------------- |
| **🔍 Object Detection**   | Real-time detection of 600+ objects using YOLOv8 with Open Images V7  |
| **🗣️ Scene Description**  | AI-powered natural language descriptions of surroundings using GPT-4o |
| **📖 Text Reading (OCR)** | Read text from documents, signs, labels with multiple modes           |
| **🎤 Voice Commands**     | Hands-free control with natural language voice commands               |
| **🧭 Navigation**         | Turn-by-turn walking directions with OpenRouteService                 |
| **🌤️ Weather**            | Real-time weather information for your location                       |

### Advanced AI Features

| Feature                          | Description                                             |
| -------------------------------- | ------------------------------------------------------- |
| **🤟 Sign Language Interpreter** | Real-time ASL alphabet and common signs recognition     |
| **🔬 Human Analyzer**            | Pose estimation, gesture recognition, emotion detection |
| **🧠 Scene Memory**              | Remember and recall locations of objects                |
| **⚡ Proactive Assistant**       | Automatic alerts for important changes                  |
| **🎨 Color & Text Analyzer**     | Identify colors and analyze text in images              |
| **🚧 Obstacle Detection**        | Safety warnings for nearby obstacles                    |

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        VisionAssist AI                          │
├─────────────────────────────────────────────────────────────────┤
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────────────┐ │
│  │  Camera  │→ │  YOLO    │→ │  Scene   │→ │  Speech Engine   │ │
│  │  Handler │  │ Detector │  │ AI/GPT-4 │  │  (TTS Output)    │ │
│  └──────────┘  └──────────┘  └──────────┘  └──────────────────┘ │
│       ↓              ↓             ↓               ↑            │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────────────┐ │
│  │  Human   │  │   Sign   │  │   OCR    │  │  Voice Listener  │ │
│  │ Analyzer │  │ Language │  │  Engine  │  │  (STT Input)     │ │
│  └──────────┘  └──────────┘  └──────────┘  └──────────────────┘ │
│       ↓              ↓             ↓               ↓            │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │                    Main Controller                          ││
│  │  • Frame processing loop (25+ FPS)                          ││
│  │  • Feature coordination                                     ││
│  │  • Safety monitoring                                        ││
│  └─────────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────────┘
```

---

## 📁 Project Structure

```
visionassist/
├── src/
│   ├── controller.py           # Main application controller
│   ├── camera_handler.py       # Camera capture and display
│   ├── object_detector.py      # YOLOv8 object detection
│   ├── scene_ai_client.py      # GPT-4o scene descriptions
│   ├── speech_engine.py        # Text-to-speech output
│   ├── voice_listener.py       # Voice command input
│   ├── ocr_engine.py           # Optical character recognition
│   ├── document_reader.py      # Document reading modes
│   ├── navigation_client.py    # Navigation and routing
│   ├── weather_client.py       # Weather information
│   │
│   ├── ai_features/            # Advanced AI modules
│   │   ├── human_analyzer.py       # Pose, gesture, emotion detection
│   │   ├── sign_language_interpreter.py  # ASL recognition
│   │   ├── scene_memory.py         # Spatial memory system
│   │   ├── emotion_analyzer.py     # Facial emotion analysis
│   │   ├── proactive_assistant.py  # Automatic alerts
│   │   ├── advanced_yolo.py        # Enhanced YOLO features
│   │   └── color_text_analyzer.py  # Color identification
│   │
│   ├── brain/                  # Intent processing
│   │   ├── assistant_brain_impl.py # Main brain logic
│   │   ├── intent_detection.py     # Command classification
│   │   └── handlers/               # Command handlers
│   │
│   ├── safety/                 # Safety features
│   │   ├── obstacle_layer.py       # Obstacle detection
│   │   ├── guidance_engine.py      # Navigation guidance
│   │   └── depth_estimator.py      # Distance estimation
│   │
│   └── utils/                  # Utilities
│       ├── config.py               # Configuration settings
│       ├── telemetry.py            # Performance logging
│       └── telemetry_dashboard.py  # Real-time dashboard
│
├── telemetry/                  # Performance logs
├── runtime/                    # Runtime data (location, etc.)
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

---

## 🚀 Installation

### Prerequisites

- Python 3.11 or higher
- macOS (Apple Silicon recommended) or Linux
- Webcam or USB camera
- Microphone (for voice commands)
- Speakers/headphones (for audio output)

### Step 1: Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/visionassist.git
cd visionassist
```

### Step 2: Create Virtual Environment

```bash
python3.11 -m venv venv311
source venv311/bin/activate  # On macOS/Linux
```

### Step 3: Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### Step 4: Set Up API Keys

Create a `.env` file or export environment variables:

```bash
export OPENAI_API_KEY="your-openai-api-key"
export OPENROUTE_API_KEY="your-openroute-api-key"  # Optional, for navigation
export OPENWEATHER_API_KEY="your-openweather-api-key"  # Optional, for weather
```

### Step 5: Run the Application

```bash
cd src
python controller.py
```

---

## ⌨️ Keyboard Controls

| Key | Action                              |
| --- | ----------------------------------- |
| `q` | Quit application                    |
| `d` | Describe current scene              |
| `v` | Activate voice command              |
| `r` | Read text in view                   |
| `s` | Toggle safety warnings              |
| `f` | Toggle fullscreen                   |
| `p` | Toggle mirror mode                  |
| `h` | Toggle human analyzer visualization |
| `g` | Toggle sign language interpreter    |
| `c` | Clear sign language buffer          |
| `x` | Save scene to memory                |
| `z` | Show memory stats                   |
| `1` | OCR: Local-only mode                |
| `2` | OCR: Hybrid mode                    |
| `3` | OCR: AI-only mode                   |
| `m` | Cycle OCR modes                     |

---

## 🎤 Voice Commands

### Scene & Objects

- _"What do you see?"_ - Describe the scene
- _"What's in front of me?"_ - Describe objects ahead
- _"Where is my [object]?"_ - Find specific object
- _"How many people?"_ - Count people in view

### Reading

- _"Read this"_ - Read text in view
- _"Read the sign"_ - Read signage
- _"Stop reading"_ - Stop current reading

### Navigation

- _"Navigate to [place]"_ - Get walking directions
- _"How do I get to [place]?"_ - Get directions
- _"Where am I?"_ - Current location

### Weather

- _"What's the weather?"_ - Current conditions
- _"Will it rain today?"_ - Weather forecast

### Sign Language

- _"Enable sign language"_ - Turn on interpreter
- _"Disable sign language"_ - Turn off interpreter
- _"What did they sign?"_ - Get current word buffer

### System

- _"Help"_ - List available commands
- _"Louder"_ / _"Quieter"_ - Adjust volume
- _"Repeat that"_ - Repeat last message

---

## 🔬 Human Analyzer Features

The Human Analyzer V3 provides comprehensive human understanding:

### Detection Capabilities

- **33-point pose estimation** with sub-pixel accuracy
- **21-point hand tracking** per hand (up to 4 hands)
- **468-point face mesh** for detailed facial analysis

### Activity Recognition (20 types)

Standing, Sitting, Walking, Running, Waving, Pointing, Arms Raised, Arms Crossed, Bending, Crouching, Lying Down, Leaning, Jumping, Reaching, Kneeling, Stretching, Typing, On Phone, Eating, Unknown

### Gesture Recognition (20 types)

Open Palm, Fist, Pointing, Peace Sign, Thumbs Up, Thumbs Down, OK Sign, Rock Sign, Call Me, Wave, Grab, Pinch, Finger Gun, Three, Four, Stop, Clap, Prayer, Heart, None

### Face Analysis

- **Emotion Detection**: Happy, Surprised, Focused, Tired, Confused, Interested, Neutral
- **Engagement Levels**: Highly Engaged, Engaged, Partial, Distracted, Disengaged
- **Gaze Direction**: Forward, At You, Left, Right, Up, Down, At Phone, Away
- **Drowsiness Detection**: PERCLOS algorithm with yawn detection

### Tracking Features

- Kalman filtering for smooth position tracking
- Multi-person tracking with persistent IDs
- Re-identification after occlusion
- Temporal voting for stable classifications

---

## 🤟 Sign Language Interpreter

Real-time ASL recognition for communication:

### Supported Signs

- **Full ASL Alphabet** (A-Z)
- **Numbers** (0-9)
- **Common Words**: Hello, Goodbye, Please, Thank You, Sorry, Yes, No, Help, Stop, More, Want, Need, Like, Love, Friend, Family, Work, Home, Food, Water

### How to Use

1. Press `g` to enable sign language mode
2. Position your hand clearly in front of the camera
3. Hold each sign steady for ~0.5 seconds
4. The system will speak recognized signs
5. Letters accumulate into words automatically
6. Press `c` to clear the buffer

### Tips for Best Results

- Good lighting on your hands
- Plain background
- Hand 1-2 feet from camera
- Hold signs steady
- Full hand visible in frame

---

## ⚙️ Configuration

Edit `src/utils/config.py` to customize:

### Camera Settings

```python
DEFAULT_CAMERA_INDEX = 0        # Camera device index
DEFAULT_FRAME_WIDTH = 640       # Frame width
DEFAULT_FRAME_HEIGHT = 480      # Frame height
```

### Performance Settings

```python
PROCESS_EVERY_N_FRAMES = 3      # Process every N frames (higher = faster)
YOLO_INFERENCE_SIZE = 480       # YOLO input size (smaller = faster)
USE_GPU = True                  # Use GPU acceleration
```

### AI Settings

```python
OPENAI_VISION_MODEL = "gpt-4o"  # Vision model for scene descriptions
OPENAI_CHAT_MODEL = "gpt-4o-mini"  # Chat model for conversations
```

### Display Settings

```python
FULLSCREEN_WINDOW = True        # Start in fullscreen
MIRROR_MODE = False             # Mirror the display
SHOW_FPS = True                 # Show FPS counter
```

---

## 📊 Telemetry Dashboard

Real-time performance monitoring:

```bash
# In a separate terminal
cd src
python -m utils.telemetry_dashboard
```

Dashboard shows:

- FPS over time
- Detection latency
- Object detection counts
- AI response times
- System resource usage

---

## 🔧 Troubleshooting

### Camera Not Found

```bash
# List available cameras
python -c "import cv2; print([i for i in range(10) if cv2.VideoCapture(i).isOpened()])"
```

### Slow Performance

1. Reduce frame size: `FRAME_WIDTH=480 FRAME_HEIGHT=360`
2. Increase skip frames: `PROCESS_EVERY_N_FRAMES=5`
3. Use smaller YOLO model: `YOLO_MODEL=yolov8n.pt`

### No Audio Output (macOS)

```bash
# Test TTS
say "Hello world"
```

### OpenAI API Errors

- Verify API key: `echo $OPENAI_API_KEY`
- Check API status: https://status.openai.com

### MediaPipe Errors

```bash
pip install --upgrade mediapipe
```

---

## 📋 Requirements

### Core Dependencies

```
opencv-python>=4.8.0
ultralytics>=8.0.0
mediapipe>=0.10.0
openai>=1.0.0
numpy>=1.24.0
```

### Audio/Speech

```
sounddevice>=0.4.6
soundfile>=0.12.0
pyttsx3>=2.90  # Fallback TTS
```

### Optional

```
openrouteservice  # Navigation
requests          # Weather API
pillow            # Image processing
```

---

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- [Ultralytics](https://ultralytics.com/) for YOLOv8
- [MediaPipe](https://mediapipe.dev/) for pose/hand/face detection
- [OpenAI](https://openai.com/) for GPT-4o vision capabilities
- [OpenRouteService](https://openrouteservice.org/) for navigation
- The visually impaired community for inspiration and feedback

---

## 📞 Contact

**Mohammed Alsheikh**

- GitHub: [@YOUR_USERNAME](https://github.com/YOUR_USERNAME)
- Project Link: [https://github.com/YOUR_USERNAME/visionassist](https://github.com/YOUR_USERNAME/visionassist)

---

<p align="center">
  Made with ❤️ for accessibility
</p>
