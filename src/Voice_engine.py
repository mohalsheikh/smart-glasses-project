# vosk_spk_gate.py (Windows)
# pip install vosk sounddevice numpy

import json
import queue
from pathlib import Path

import numpy as np
import sounddevice as sd
from vosk import Model, KaldiRecognizer, SpkModel

# === Paths ===
ASR_MODEL_PATH = r"C:\Repos\vosk-model-small-en-us-0.15\vosk-model-small-en-us-0.15"
SPK_MODEL_PATH = r"C:\Repos\vosk-model-spk-0.4\vosk-model-spk-0.4"
VOICEPRINT_FILE = "voiceprint.json"

# === Audio settings ===
TARGET_RATE = 16000
BLOCK_SIZE = 8000  # frames per callback chunk

# === Gates / thresholds ===
# Cosine similarity: higher = more likely the same speaker
# Typical starting range: 0.70–0.85 (you must tune this)
SIM_THRESHOLD = 0.78

# Loudness gate helps ignore far voices (tune per mic)
# RMS of int16 normalized to [0..1]. Start around 0.015–0.040
RMS_THRESHOLD = 0.020

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
            print(f"  {i}: {d['name']} (inputs={d['max_input_channels']}, "
                  f"default_sr={d['default_samplerate']}, api={api})")

def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(np.float32)
    b = b.astype(np.float32)
    na = np.linalg.norm(a) + 1e-9
    nb = np.linalg.norm(b) + 1e-9
    return float(np.dot(a, b) / (na * nb))

def rms_from_bytes_int16(raw: bytes) -> float:
    if not raw:
        return 0.0
    x = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    return float(np.sqrt(np.mean(x * x)) + 1e-12)

def pick_device_and_rate():
    list_input_devices()
    default_in = sd.default.device[0]
    print(f"\nDefault input device index: {default_in}")

    s = input("Enter mic device index (press Enter to use default): ").strip()
    device = int(s) if s else default_in

    info = sd.query_devices(device)
    print("\nChosen device:", device, "-", info["name"])
    print("Device default sample rate:", info["default_samplerate"])

    sr = TARGET_RATE
    try:
        sd.check_input_settings(device=device, samplerate=sr, channels=1, dtype="int16")
    except Exception:
        print("\n16000 Hz didn't work for this mic, falling back to device default sample rate.")
        sr = int(info["default_samplerate"])
        sd.check_input_settings(device=device, samplerate=sr, channels=1, dtype="int16")

    print("Opening mic at sample rate:", sr)
    return device, sr

def load_models():
    asr_p = Path(ASR_MODEL_PATH)
    spk_p = Path(SPK_MODEL_PATH)

    if not asr_p.exists():
        raise RuntimeError(f"ASR_MODEL_PATH not found: {asr_p}")
    if not spk_p.exists():
        raise RuntimeError(f"SPK_MODEL_PATH not found: {spk_p}")

    print("\nLoading ASR model...")
    asr_model = Model(str(asr_p))

    print("Loading speaker model...")
    spk_model = SpkModel(str(spk_p))

    return asr_model, spk_model

def make_recognizer(asr_model, spk_model, sr):
    rec = KaldiRecognizer(asr_model, sr)
    rec.SetWords(True)
    rec.SetSpkModel(spk_model)   # <-- key line
    return rec

def enroll_voiceprint(device, sr, asr_model, spk_model):
    """
    Collect multiple 'spk' vectors from your speech and average them.
    Tip: speak in short phrases with brief pauses (so AcceptWaveform finalizes).
    """
    print("\n=== ENROLL MODE ===")
    print("Speak 8–15 short phrases (with small pauses).")
    print("Goal: collect ~10 embeddings.\n")

    rec = make_recognizer(asr_model, spk_model, sr)

    collected = []
    chunk_bytes = bytearray()

    with sd.RawInputStream(
        device=device,
        samplerate=sr,
        blocksize=BLOCK_SIZE,
        dtype="int16",
        channels=1,
        callback=callback,
    ):
        while len(collected) < 10:
            data = q.get()
            chunk_bytes.extend(data)

            if rec.AcceptWaveform(data):
                res = json.loads(rec.Result())
                spk = res.get("spk")
                text = (res.get("text") or "").strip()

                # Only keep embeddings when there is real speech
                chunk_rms = rms_from_bytes_int16(bytes(chunk_bytes))
                chunk_bytes.clear()

                if spk and chunk_rms >= RMS_THRESHOLD:
                    v = np.array(spk, dtype=np.float32)
                    collected.append(v)
                    print(f"[{len(collected)}/10] captured embedding | rms={chunk_rms:.3f} | text='{text}'")
                else:
                    print(f"[skip] rms={chunk_rms:.3f} | text='{text}' (too quiet or no spk vector)")

    # Average + normalize
    mat = np.stack(collected, axis=0)
    avg = np.mean(mat, axis=0)
    avg = avg / (np.linalg.norm(avg) + 1e-9)

    with open(VOICEPRINT_FILE, "w", encoding="utf-8") as f:
        json.dump({"voiceprint": avg.tolist()}, f)

    print(f"\n✅ Saved voiceprint to {VOICEPRINT_FILE}")

def load_voiceprint():
    p = Path(VOICEPRINT_FILE)
    if not p.exists():
        raise RuntimeError(f"Missing {VOICEPRINT_FILE}. Run enroll first.")
    obj = json.loads(p.read_text(encoding="utf-8"))
    vp = np.array(obj["voiceprint"], dtype=np.float32)
    vp = vp / (np.linalg.norm(vp) + 1e-9)
    return vp

def listen_with_gate(device, sr, asr_model, spk_model):
    print("\n=== LISTEN MODE ===")
    print(f"Gate: SIM_THRESHOLD={SIM_THRESHOLD}, RMS_THRESHOLD={RMS_THRESHOLD}")
    print("Only printing commands when speaker matches.\n")

    target_vp = load_voiceprint()
    rec = make_recognizer(asr_model, spk_model, sr)

    chunk_bytes = bytearray()

    with sd.RawInputStream(
        device=device,
        samplerate=sr,
        blocksize=BLOCK_SIZE,
        dtype="int16",
        channels=1,
        callback=callback,
    ):
        while True:
            data = q.get()
            chunk_bytes.extend(data)

            if rec.AcceptWaveform(data):
                res = json.loads(rec.Result())
                text = (res.get("text") or "").strip()
                spk = res.get("spk")
                chunk_rms = rms_from_bytes_int16(bytes(chunk_bytes))
                chunk_bytes.clear()

                if not text:
                    continue

                if chunk_rms < RMS_THRESHOLD:
                    print(f"🚫 rejected (too quiet) rms={chunk_rms:.3f} | '{text}'")
                    continue

                if not spk:
                    print(f"🚫 rejected (no spk vector) rms={chunk_rms:.3f} | '{text}'")
                    continue

                v = np.array(spk, dtype=np.float32)
                v = v / (np.linalg.norm(v) + 1e-9)

                sim = cosine_sim(target_vp, v)

                if sim >= SIM_THRESHOLD:
                    print(f"✅ ACCEPT sim={sim:.3f} rms={chunk_rms:.3f} | {text}")
                else:
                    print(f"🚫 rejected sim={sim:.3f} rms={chunk_rms:.3f} | '{text}'")

            else:
                # Optional: show partial text (no speaker vector here)
                partial = json.loads(rec.PartialResult()).get("partial", "").strip()
                if partial:
                    print("…", partial)

def main():
    device, sr = pick_device_and_rate()
    asr_model, spk_model = load_models()

    print("\nChoose mode:")
    print("  1) Enroll my voice (create voiceprint.json)")
    print("  2) Listen with speaker gate")
    mode = input("Enter 1 or 2: ").strip()

    if mode == "1":
        enroll_voiceprint(device, sr, asr_model, spk_model)
    else:
        listen_with_gate(device, sr, asr_model, spk_model)

if __name__ == "__main__":
    main()
