# Smart Glasses Project Overview

## Purpose
Assistive real-time vision pipeline for visually impaired users. The system listens for a wake word and command, captures camera frames, detects objects/currency, optionally reads text on detections, and speaks a natural-language response.

## Active Runtime Architecture
Current executable path:

1. `main.py` creates `MainController` and starts the app loop.
2. `src/manual_controller.py` orchestrates camera capture, voice state machine, command routing, detection/OCR calls, and speech output.
3. Worker threads:
   - TTS worker: dequeues text and calls `src/speech_engine.py`.
   - Voice worker: listens for wake word/command via `src/voice_input.py`.
4. Per command:
   - `detect`: `src/object_detector.py` runs YOLO tracking across recent frames.
   - `read`: detection first, then `src/ocr_engine.py` OCR on detected crops.
5. `src/utils/object_description.py` normalizes labels and summarizes detections into spoken output.
6. `src/camera_handler.py` handles OpenCV capture/display and input key checks.

## Active Project Structure (Concise)

```text
smart-glasses-project/
├── main.py                          # entrypoint (manual mode)
├── requirements.txt                 # pinned environment (broad lockfile)
├── yolov8n.pt                       # general object model
├── currency_detector.pt             # custom currency model
├── vosk-model-small-en-us-0.15/     # offline speech recognition model
├── src/
│   ├── manual_controller.py         # runtime orchestrator (active)
│   ├── camera_handler.py            # camera I/O and display
│   ├── object_detector.py           # multi-model YOLO detection/tracking
│   ├── ocr_engine.py                # EasyOCR over object crops
│   ├── voice_input.py               # Vosk + sounddevice command input
│   ├── speech_engine.py             # pyttsx3 text-to-speech
│   ├── currency_recognizer.py       # legacy/optional component (not in active flow)
│   └── utils/
│       ├── config.py                # camera defaults + direction enum
│       ├── object_description.py    # label normalization + summaries
│       └── preprocessing.py         # OCR helper preprocessing
├── tests/                           # tests/experiments
├── tests_Eric/                      # OCR/object-detection experiments
└── training/                        # model training pipeline (separate from runtime)
```

## What Is Not in Active Runtime
The following are present but not part of the current main execution path:

- `src/ocr_engine2.py`
- files under `tests_Eric/`
- most files under `tests/` (used for testing, not app runtime)
- `training/` pipeline files (used to train/export models, not run inference app)

## Project Requirements

### Runtime software requirements (active app)
- Python 3.10+ (recommended)
- OpenCV (`opencv-python`)
- NumPy (`numpy`)
- Ultralytics YOLO (`ultralytics`)
- EasyOCR (`easyocr`)
- Text-to-speech (`pyttsx3`)
- Offline speech recognition (`vosk`)
- Microphone input (`sounddevice`)
- Label formatting (`inflect`)

### Runtime assets required
- YOLO model files at project root:
  - `yolov8n.pt`
  - `currency_detector.pt`
- Vosk model directory:
  - `vosk-model-small-en-us-0.15/` (must contain `am/`, `conf/`, `graph/`)

### Hardware requirements
- Webcam accessible by OpenCV
- Microphone for wake word and command input
- Speakers/headphones for spoken output

### Notes on requirements.txt
`requirements.txt` is a broad pinned lockfile and includes many platform-specific packages (for example, many macOS `pyobjc*` packages). For this Windows project runtime, the core requirement set above reflects the actual active dependency surface.

### Training requirements (separate workflow)
`training/requirements.txt` defines model-training dependencies (Roboflow, pandas, onnx, etc.) used only for training/export, not for normal runtime inference.
