# Smart Glasses Project Overview

## Project Goal

Provide an offline-capable assistive vision workflow for visually impaired users by combining:

- spoken interaction (wake word + commands)
- real-time object detection
- OCR on detected objects
- spoken natural-language scene feedback

## Current Runtime State

The active executable path is manual command-driven mode:

1. `main.py` instantiates `MainController` and starts the main loop.
2. `src/manual_controller.py` owns the runtime state machine and orchestration.

Key behavior currently active:

- Wake word is `vision`.
- Commands are processed after wake word detection.
- A rolling 3-frame buffer is used for detection/OCR commands.
- Two windows are displayed: live camera feed + last annotated detections.
- `q` key exits the app.

## Runtime Architecture

### Core control flow

1. Camera frames are continuously captured via `src/camera_handler.py`.
2. Voice input worker thread listens using Vosk through `src/voice_input.py`.
3. TTS worker thread speaks queued responses via `src/speech_engine.py`.
4. Main thread routes command text in `src/manual_controller.py`.

### Command execution paths

- `detect`
  - Runs `src/object_detector.py` on the recent frame buffer.
  - Uses multi-model YOLO loading (currently `yolov8n.pt` and `currency_detector.pt`).
  - Produces merged detections and optional annotations.

- `read`
  - Runs detection first.
  - Runs `src/ocr_engine.py` on each detected crop.
  - Attaches OCR text per detection when confidence filtering passes.

### Summarization and language shaping

`src/utils/object_description.py`:

- normalizes labels (including currency front/back label merging)
- applies confidence thresholds by object category
- condenses repeated track IDs across frame history
- generates natural language with directional phrasing (left/front/right)

## Active Command Surface

- Actions: `detect`, `read`
- Direction modifiers: `left`, `front`, `right`
- Session controls: `sleep`, `end`, `nevermind`, `thanks`, `repeat`
- Help/tutorial: `commands`, `command`, `help`, `yes`, `no`

The help flow can read:

- `commands_user_facing.txt`
- `commands_directional_exit.txt` (optional follow-up if user says yes)

## Project Structure (Current, Concise)

```text
smart-glasses-project/
├── main.py
├── requirements.txt
├── tutorial.txt
├── commands_user_facing.txt
├── commands_directional_exit.txt
├── yolov8n.pt
├── currency_detector.pt
├── vosk-model-small-en-us-0.15/
├── docs/
│   ├── project_overview.md
│   ├── setup_guide.md
│   ├── user_manual.md
│   └── api_reference.md
├── src/
│   ├── manual_controller.py
│   ├── camera_handler.py
│   ├── object_detector.py
│   ├── ocr_engine.py
│   ├── voice_input.py
│   ├── speech_engine.py
│   ├── currency_recognizer.py
│   └── utils/
│       ├── config.py
│       ├── object_description.py
│       ├── preprocessing.py
│       └── logger.py
└── training/
```

## Components Not In Active Runtime Path

- `src/currency_recognizer.py` exists but is not currently used by `main.py` + `manual_controller.py`.
- `training/` is a separate model training/export workflow.
- Other top-level assets like alternate YOLO weights (`yolov8n-oiv7.pt`, `yolov8s-oiv7.pt`) are present but not loaded by default runtime.

## Requirements

### Runtime software

- Python 3.10+
- `opencv-python`
- `numpy`
- `ultralytics`
- `torch`, `torchvision`
- `easyocr`
- `pyttsx3`
- `vosk`
- `sounddevice`
- `inflect`

### Runtime model/assets

- `yolov8n.pt` at project root
- `currency_detector.pt` at project root
- `vosk-model-small-en-us-0.15/` at project root with subfolders:
  - `am/`
  - `conf/`
  - `graph/`

### Hardware

- Webcam
- Microphone
- Speaker/headphones

## Dependency Notes

- `requirements.txt` is a broad pinned lockfile.
- It includes packages not required for all platforms (for example macOS `pyobjc*` packages).
- For Windows runtime, the core dependency surface is the set listed above.
- Current code imports `vosk`, `sounddevice`, and `inflect`, but these are not pinned in the present root `requirements.txt` and may need manual installation.
