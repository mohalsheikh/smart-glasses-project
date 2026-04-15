# Smart Glasses Setup Guide

This guide covers setup for the current active runtime path:

- Entrypoint: `main.py`
- Controller: `src/manual_controller.py`
- Mode: wake-word and command-driven manual interaction

## 1. Prerequisites

- Windows 10/11 (recommended for this guide)
- Python 3.10+
- Webcam available to OpenCV
- Microphone available to sounddevice/Vosk
- Speaker or headphones for TTS output

## 2. Required Runtime Assets

Place these in the project root:

- `yolov8n.pt`
- `currency_detector.pt`
- `vosk-model-small-en-us-0.15/`

The Vosk folder must contain:

- `am/`
- `conf/`
- `graph/`

## 3. Create and Activate Virtual Environment

From project root:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

## 4. Install Dependencies

Install the repository requirements first:

```powershell
pip install -r requirements.txt
```

Important: runtime code imports `vosk`, `sounddevice`, and `inflect`, but these are not currently pinned in the root `requirements.txt`.

If needed, install them explicitly:

```powershell
pip install vosk sounddevice inflect
```

## 5. Run the Application

```powershell
python main.py
```

Expected behavior at startup:

- App opens camera windows.
- Startup tutorial may play from `tutorial.txt`.
- System waits for wake word `vision`.

## 6. Quick Verification Checklist

- Camera feed appears.
- No immediate model loading errors.
- Wake word `vision` is recognized.
- A simple command such as `detect` returns spoken output.
- Press `q` in the OpenCV window to exit.

## 7. Troubleshooting

### Camera does not open

- Close other apps using the camera.
- Check camera permissions in Windows privacy settings.

### Wake word or commands are not recognized

- Confirm microphone input device is available.
- Speak clearly and close to microphone.
- Check that `vosk-model-small-en-us-0.15/` exists at project root.

### Missing import error at runtime

- Install missing packages into the active virtual environment.
- Most common missing packages: `vosk`, `sounddevice`, `inflect`.

### Slow performance

- Ensure no heavy background processes are competing for CPU/GPU.
- Default resolution is already reduced in `src/utils/config.py` for speed.

## 8. Notes on Non-Active Components

- `src/currency_recognizer.py` is present but not part of the active runtime path.
- `training/` is for model training/export and is separate from runtime inference.
