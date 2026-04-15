# Smart Glasses Project

Assistive real-time vision pipeline for visually impaired users. The application listens for a wake word, accepts spoken commands, detects objects (including currency), optionally reads text from detected objects, and speaks natural-language feedback.

## Current Status

- Active runtime entrypoint: `main.py`
- Active controller: `src/manual_controller.py`
- Runtime mode: wake-word + command driven (manual trigger), not periodic auto-loop commands
- Multi-model detection is active (`yolov8n.pt` + `currency_detector.pt`)

## What The System Does

- Captures a short rolling frame window from camera (3 frames)
- Waits for wake word: `vision`
- Listens for a command after wake word
- Routes command to detection/OCR pipeline
- Speaks summarized results via text-to-speech

Supported command families include:

- Action: detect, read
- Directional: left, front, right
- Session/control: repeat, sleep, end, nevermind, thanks
- Help/tutorial: commands, command, help (+ yes/no follow-up)

## Runtime Architecture

1. `main.py` creates `MainController` and starts `run()`.
2. `src/manual_controller.py` orchestrates the app loop and command routing.
3. Two daemon worker threads run continuously:
	- TTS worker -> `src/speech_engine.py`
	- Voice input worker -> `src/voice_input.py`
4. Detection path:
	- `src/object_detector.py` runs YOLO tracking and merges multi-model detections.
5. OCR path (`read` command):
	- Detect first, then `src/ocr_engine.py` extracts text from detected object crops.
6. Output shaping:
	- `src/utils/object_description.py` normalizes labels and builds spoken summaries.

## Required Runtime Assets

Place these at project root (current expected locations):

- `yolov8n.pt`
- `currency_detector.pt`
- `vosk-model-small-en-us-0.15/` (must contain `am/`, `conf/`, `graph/`)

## System Requirements

- Python 3.10+
- Webcam accessible by OpenCV
- Microphone for wake-word and command capture
- Speakers/headphones for speech output

## Core Runtime Libraries

- `opencv-python` (camera capture/display)
- `numpy` (frame and numeric ops)
- `ultralytics` (YOLO detection/tracking)
- `torch`, `torchvision` (model runtime)
- `easyocr` (text extraction)
- `pyttsx3` (offline text-to-speech)
- `vosk` (offline speech recognition)
- `sounddevice` (audio input)
- `inflect` (natural language phrasing)

Note: `requirements.txt` is broad and includes many platform-specific packages (including macOS `pyobjc*` packages). On Windows, the core runtime set above is the practical dependency surface.

Important: the current runtime code imports `vosk`, `sounddevice`, and `inflect`, but those are not pinned in the current root `requirements.txt`. If they are missing in your environment, install them manually.

## Quick Start (Windows PowerShell)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
# If missing at runtime, install these explicitly:
# pip install vosk sounddevice inflect
python main.py
```

## Project Layout (Relevant Runtime Files)

```text
smart-glasses-project/
├── main.py
├── requirements.txt
├── yolov8n.pt
├── currency_detector.pt
├── vosk-model-small-en-us-0.15/
├── tutorial.txt
├── commands_user_facing.txt
├── commands_directional_exit.txt
└── src/
	 ├── manual_controller.py
	 ├── camera_handler.py
	 ├── object_detector.py
	 ├── ocr_engine.py
	 ├── voice_input.py
	 ├── speech_engine.py
	 ├── currency_recognizer.py
	 └── utils/
		  ├── config.py
		  ├── object_description.py
		  └── preprocessing.py
```

## Notes

- `src/currency_recognizer.py` is currently not used in the active runtime path.
- The training workflow under `training/` is separate from normal runtime inference.