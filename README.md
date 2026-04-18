````md
# VisionAssist

This project has 2 versions:

- offline version
- online version

The offline version is the main smart glasses project.
The online version is inside the `online` folder and is mainly for the competition/demo setup.

---

## Project Structure

- `src/` = offline version
- `online/` = online version

---

## What the project does

VisionAssist is an AI smart glasses project that helps users by doing things like:

- object detection
- reading text
- currency recognition
- scene description
- voice features
- weather and navigation features
- sign language / people analysis features

Some features work fully locally, and some online features need API keys.

---

## Offline Version

The offline version is the code inside the main `src` folder.

### Main file to run

```bash
python src/controller.py
````

### Setup

Make sure you are in the project folder first.

```bash
cd smart-glasses-project
```

Then install the packages you need. At minimum, this project uses things like:

```bash
pip install opencv-python numpy ultralytics easyocr sounddevice soundfile
```

Depending on what features you want to use, you may also need other packages like:

* openai
* tesseract
* deepgram-sdk
* vosk / whisper related packages

### Notes

* If you want OpenAI features, add your API key to your environment.
* If you want Deepgram voice transcription, add your Deepgram API key too.
* Some camera/audio features may depend on your device.

### Run

```bash
python src/controller.py
```

---

## Online Version

The online version is in the `online` folder.

### Go into the folder

```bash
cd online
```

### Main file to run

```bash
python demo_mode.py
```

### Setup

Install the packages you need:

```bash
pip install opencv-python numpy ultralytics easyocr sounddevice soundfile
```

You may also need:

* openai
* faster-whisper
* scipy

### API Keys

For the online version, you need to add your own API keys as environment variables.

Example:

```bash
export OPENAI_API_KEY="your_key_here"
```

If you are using other services, add those keys too.

### Run

```bash
python demo_mode.py
```

### Optional

Run without voice:

```bash
python demo_mode.py --no-voice
```

If using a webcam:

```bash
python demo_mode.py --webcam 0
```

---

## Important

* The offline version is in `src`
* The online version is in `online`
* For the online version, make sure you `cd online` first before running it
* Do not use real API keys in the repo
* If needed, create your own `.env` or export your keys in the terminal

---

## Quick Run Summary

### Offline

```bash
cd smart-glasses-project
python src/controller.py
```

### Online

```bash
cd smart-glasses-project/online
python demo_mode.py
```

---

## Troubleshooting

### Camera not working

Try changing the webcam index or check that your camera is connected.

### No audio

Make sure your mic/speakers are connected and the audio packages are installed.

### OpenAI features not working

Make sure your `OPENAI_API_KEY` is set correctly.

### Missing package error

Install the missing package with pip and run again.

```
```
