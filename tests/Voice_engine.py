# vosk_test.py (Windows)
# pip install vosk sounddevice

import json
import queue
from pathlib import Path

import sounddevice as sd
from vosk import Model, KaldiRecognizer

MODEL_PATH = r"C:\Repos\vosk-model-en-us-0.22\vosk-model-en-us-0.22"
TARGET_RATE = 16000
BLOCK_SIZE = 8000

q = queue.Queue()

def callback(indata, frames, time, status):
    if status:
        print("Audio status:", status)
    q.put(bytes(indata))

def list_input_devices():
    hostapis = sd.query_hostapis()
    print("\nInput devices (pick one that looks like a microphone):")
    for i, d in enumerate(sd.query_devices()):
        if d["max_input_channels"] > 0:
            api = hostapis[d["hostapi"]]["name"]
            print(f"  {i}: {d['name']}  (inputs={d['max_input_channels']}, "
                  f"default_sr={d['default_samplerate']}, api={api})")

def main():
    # ---- Verify model folder ----
    p = Path(MODEL_PATH)
    print("MODEL_PATH =", p)
    print("exists?   =", p.exists())
    print("is_dir?   =", p.is_dir())
    print("has am/?  =", (p / "am").exists())
    print("has conf/?=", (p / "conf").exists())
    print("has graph/?", (p / "graph").exists())

    if not (p.exists() and (p / "am").exists() and (p / "conf").exists() and (p / "graph").exists()):
        raise RuntimeError("MODEL_PATH must point to the folder that contains am/, conf/, graph/.")

    # ---- Pick mic ----
    list_input_devices()
    default_in = sd.default.device[0]
    print(f"\nDefault input device index: {default_in}")

    s = input("Enter mic device index (press Enter to use default): ").strip()
    device = int(s) if s else default_in

    info = sd.query_devices(device)
    print("\nChosen device:", device, "-", info["name"])
    print("Device default sample rate:", info["default_samplerate"])

    # ---- Pick a sample rate that actually opens ----
    sr = TARGET_RATE
    try:
        sd.check_input_settings(device=device, samplerate=sr, channels=1, dtype="int16")
    except Exception as e:
        print("\n16000 Hz didn't work for this mic, falling back to device default sample rate.")
        sr = int(info["default_samplerate"])
        sd.check_input_settings(device=device, samplerate=sr, channels=1, dtype="int16")

    print("Opening mic at sample rate:", sr)

    # ---- Load model + recognizer ----
    print("\nLoading Vosk model...")
    model = Model(str(p))
    rec = KaldiRecognizer(model, sr)

    # ---- Start listening ----
    print("\nListening... speak clearly. (Ctrl+C to stop)\n")
    with sd.RawInputStream(
        device=device,          # <-- IMPORTANT: force the chosen device
        samplerate=sr,
        blocksize=BLOCK_SIZE,
        dtype="int16",
        channels=1,
        callback=callback,
    ):
        while True:
            data = q.get()
            if rec.AcceptWaveform(data):
                result = json.loads(rec.Result())
                text = result.get("text", "").strip()
                if text:
                    print("✅ Final:", text)
            else:
                partial = json.loads(rec.PartialResult()).get("partial", "").strip()
                if partial:
                    print("…", partial)

if __name__ == "__main__":
    main()
