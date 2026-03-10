# VisionAssist - Competition Demo Guide

## Quick Start (Monday Demo)

### Option A: Laptop Demo (Recommended for presentation)
```bash
# Make sure you have your OpenAI API key set
export OPENAI_API_KEY="sk-..."

# Run the demo
python demo_mode.py
```

### Option B: Raspberry Pi Demo
```bash
# Install TTS on Pi (if not already)
sudo apt install espeak-ng

# Run
export OPENAI_API_KEY="sk-..."
python demo_mode.py --webcam 0
```

### Option C: No Voice (keyboard-only demo)
```bash
python demo_mode.py --no-voice
```

## Demo Controls
| Key | Action |
|-----|--------|
| D | Describe scene (AI narration) |
| V | Voice command |
| R | Read text in view |
| C | Identify currency |
| P | Describe people |
| F | Find/identify objects |
| T | Current time |
| W | Weather info |
| SPACE | Toggle auto-narration |
| H | Show/hide help overlay |
| Q | Quit |

## Suggested Demo Script (5 minutes)

### 1. Object Detection (30 sec)
- Point camera around the room
- Show real-time YOLO detection (bounding boxes, labels, FPS counter)
- "VisionAssist detects and tracks objects in real-time at 25+ FPS"

### 2. Scene Description (45 sec)
- Press D to describe the scene
- "The AI creates natural language descriptions of the environment"
- Show different angles/scenes

### 3. Text Reading (45 sec)
- Hold up a document, menu, or book
- Press R to read it
- "It reads printed text aloud using hybrid OCR"

### 4. Currency Recognition (30 sec)
- Hold up a bill
- Press C to identify it
- "Users who can't see denominations can identify their money"

### 5. Voice Commands (60 sec)
- Press V and ask natural questions:
  - "What do you see?"
  - "How many people are in the room?"
  - "Is there a chair nearby?"
  - "What time is it?"
- "Fully conversational - users just speak naturally"

### 6. People Description (30 sec)
- Press P with people in view
- "Describes clothing, position, and activity"

### 7. Navigation (mention only)
- "Full turn-by-turn walking navigation via OpenRouteService"
- "GPS integration through phone companion app"

### 8. Sign Language Recognition (mention only)
- "ASL fingerspelling and word sign recognition"
- "Bridges communication between deaf and blind users"

## Key Talking Points
- **Origin Story**: Built for Ethan, a resident with significant vision loss
- **Technical Scale**: 10,000+ lines, 15+ AI modules
- **Real Hardware**: Raspberry Pi 5 + Pi Camera Module 3 + ESP32 sensors
- **Fully Local**: Voice transcription runs offline (faster-whisper + vosk)
- **Cost**: Under $150 hardware BOM vs $3,000+ commercial alternatives
- **Market**: 2.2 billion people globally with vision impairment (WHO)

## Troubleshooting
- **No camera?** Try `python demo_mode.py --webcam 1` for external webcam
- **No sound?** Check `espeak-ng --version` on Pi, `say "test"` on Mac
- **No API?** System works without OpenAI (falls back to local detection descriptions)
- **Slow?** Set `export PROCESS_EVERY_N_FRAMES=4` for slower machines
