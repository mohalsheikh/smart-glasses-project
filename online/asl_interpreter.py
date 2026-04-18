"""
ASL Sign Language Interpreter for VisionAssist
================================================
Production-grade continuous ASL interpretation using GPT-4o Vision.

Features:
- Continuous monitoring mode (captures every 2s while active)
- Smart sentence building from fingerspelled letters
- Avoids repeating the same detection
- Pauses during TTS playback
- Three detection modes: alphabet, phrase, auto
- Natural spoken output for blind users

Usage:
  interpreter.start(frame_provider)  → starts continuous mode
  interpreter.stop()                 → stops continuous mode
  interpreter.interpret_once(frame)  → single-shot interpretation
"""

from __future__ import annotations

import base64
import cv2 as cv
import os
import time
import threading
from typing import Optional, Callable


class ASLInterpreter:
    """
    Continuous ASL sign language interpreter using GPT-4o Vision.
    Designed for blind users who need real-time sign language translation.
    """

    def __init__(self, speech_engine=None):
        self._client = None
        self._available = False
        self._speech = speech_engine

        # Continuous mode state
        self._running = False
        self._thread = None
        self._get_frame: Optional[Callable] = None
        self._capture_interval = 2.5  # seconds between captures

        # Sentence building
        self._current_word = ""
        self._current_sentence = ""
        self._last_detection = ""
        self._last_detection_time = 0.0
        self._repeat_cooldown = 3.0  # don't repeat same detection within 3s
        self._no_sign_count = 0  # track consecutive "no sign" results

        try:
            from openai import OpenAI
            self._client = OpenAI()
            self._available = True
            print("🤟 ASL Interpreter initialized (GPT-4o Vision)")
        except Exception as e:
            print(f"⚠️ ASL Interpreter not available: {e}")

    @property
    def available(self) -> bool:
        return self._available

    @property
    def is_active(self) -> bool:
        return self._running
    
    def feed_frame(self, frame):
        """Accept frame from main loop (compatibility). This version uses _get_frame callback instead."""
        pass

    # ------------------------------------------------------------------
    # CONTINUOUS MODE
    # ------------------------------------------------------------------
    def start(self, get_frame_fn: Callable, speech_engine=None):
        """
        Start continuous ASL monitoring.
        get_frame_fn: callable that returns the latest camera frame (numpy array)
        """
        if self._running:
            return
        if not self._available:
            return

        self._get_frame = get_frame_fn
        if speech_engine:
            self._speech = speech_engine
        self._running = True
        self._current_word = ""
        self._current_sentence = ""
        self._last_detection = ""
        self._no_sign_count = 0

        self._thread = threading.Thread(target=self._continuous_loop, daemon=True)
        self._thread.start()
        print("🤟 ASL continuous mode: STARTED")

    def stop(self) -> str:
        """Stop continuous ASL monitoring. Returns the accumulated sentence."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=3.0)
            self._thread = None

        # Build final output
        final = self._build_final_sentence()
        self._current_word = ""
        self._current_sentence = ""
        self._last_detection = ""
        print("🤟 ASL continuous mode: STOPPED")
        return final

    def _continuous_loop(self):
        """Background loop that continuously captures and interprets."""
        while self._running:
            try:
                # Wait for TTS to finish before capturing (avoid picking up nothing while speaking)
                if self._speech and self._speech.is_speaking():
                    time.sleep(0.5)
                    continue

                # Get latest frame
                frame = self._get_frame() if self._get_frame else None
                if frame is None:
                    time.sleep(0.5)
                    continue

                # Interpret the frame
                result = self._interpret_frame_internal(frame)

                if result and result.get("detected"):
                    sign_type = result["type"]  # "letter", "word", "phrase", "none"
                    value = result["value"]
                    raw = result["raw"]

                    # Skip if same as last detection (within cooldown)
                    now = time.time()
                    if value == self._last_detection and (now - self._last_detection_time) < self._repeat_cooldown:
                        time.sleep(self._capture_interval)
                        continue

                    self._last_detection = value
                    self._last_detection_time = now
                    self._no_sign_count = 0

                    # Handle based on type
                    if sign_type == "letter":
                        self._current_word += value
                        # Speak the letter
                        self._say(f"{value}")

                    elif sign_type == "word" or sign_type == "phrase":
                        # If we were building a word, finalize it first
                        if self._current_word:
                            self._current_sentence += self._current_word + " "
                            self._current_word = ""
                        self._say(raw)

                    elif sign_type == "space":
                        # Space sign — finalize current word
                        if self._current_word:
                            word = self._current_word
                            self._current_sentence += word + " "
                            self._current_word = ""
                            self._say(f"Word: {word}")

                else:
                    self._no_sign_count += 1

                    # After 3 consecutive no-signs, if we have a word, speak it
                    if self._no_sign_count >= 3 and self._current_word:
                        word = self._current_word
                        self._current_sentence += word + " "
                        self._current_word = ""
                        self._say(f"Word: {word}")

                time.sleep(self._capture_interval)

            except Exception as e:
                if self._running:
                    print(f"🤟 ASL error: {e}")
                    time.sleep(2.0)

    def _build_final_sentence(self) -> str:
        """Build the final sentence from accumulated words."""
        sentence = self._current_sentence.strip()
        if self._current_word:
            sentence += (" " if sentence else "") + self._current_word
        return sentence.strip() if sentence else "No signs were detected."

    def _say(self, text: str):
        """Speak text through the speech engine."""
        if self._speech and text:
            print(f"🤟 ASL: {text}")
            self._speech.speak(text)

    # ------------------------------------------------------------------
    # SINGLE-SHOT MODE
    # ------------------------------------------------------------------
    def interpret_once(self, frame) -> str:
        """
        One-shot ASL interpretation. Returns spoken description.
        """
        if not self._available or self._client is None:
            return "Sign language interpreter not available."
        if frame is None:
            return "No image available."

        result = self._interpret_frame_internal(frame)
        if result and result.get("raw"):
            return result["raw"]
        return "I don't see anyone signing right now."

    # ------------------------------------------------------------------
    # CORE INTERPRETATION
    # ------------------------------------------------------------------
    def _interpret_frame_internal(self, frame) -> Optional[dict]:
        """
        Core interpretation. Returns dict with:
          detected: bool
          type: "letter" | "word" | "phrase" | "space" | "none"
          value: normalized value (uppercase letter, word, etc.)
          raw: raw spoken text from GPT-4o
        """
        if not self._available or self._client is None or frame is None:
            return None

        try:
            # Encode frame
            _, buffer = cv.imencode('.jpg', frame, [cv.IMWRITE_JPEG_QUALITY, 90])
            b64_image = base64.b64encode(buffer).decode('utf-8')

            prompt = (
                "You are an expert American Sign Language (ASL) interpreter. "
                "A blind person is wearing smart glasses with a camera, and someone in front of them "
                "is communicating using ASL. Your job is to translate what they see.\n\n"
                "CAREFULLY examine this image for ANY hand gestures or signs:\n\n"
                "1. Look for hands — even partially visible ones\n"
                "2. Analyze hand shape, finger positions, thumb placement\n"
                "3. Check hand orientation (palm facing camera, away, sideways)\n"
                "4. Note position relative to body (near face, chest, waist)\n"
                "5. Consider facial expression (important in ASL)\n\n"
                "RESPOND IN EXACTLY THIS FORMAT:\n"
                "TYPE: [letter/word/phrase/none]\n"
                "SIGN: [what is being signed]\n"
                "SPOKEN: [natural sentence to tell the blind person]\n\n"
                "Examples:\n"
                "TYPE: letter\nSIGN: A\nSPOKEN: The letter A\n\n"
                "TYPE: word\nSIGN: hello\nSPOKEN: They are saying hello\n\n"
                "TYPE: phrase\nSIGN: thank you\nSPOKEN: They are saying thank you\n\n"
                "TYPE: letter\nSIGN: I love you\nSPOKEN: They are signing I love you\n\n"
                "TYPE: none\nSIGN: none\nSPOKEN: No signing detected\n\n"
                "IMPORTANT:\n"
                "- If you see a hand making ANY deliberate shape, try to interpret it as ASL\n"
                "- Common ASL signs: hello, thank you, please, sorry, help, yes, no, "
                "I love you, good, bad, more, stop, go, eat, drink, water, name, "
                "what, where, when, how, why, who, understand, don't understand\n"
                "- For fingerspelling: identify the specific letter (A-Z)\n"
                "- Be confident in your interpretation — the blind person relies on you\n"
                "- Only say 'none' if there are truly NO hands visible at all"
            )

            response = self._client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{b64_image}",
                                    "detail": "high",
                                },
                            },
                        ],
                    }
                ],
                max_tokens=200,
                temperature=0.1,
            )

            text = response.choices[0].message.content.strip()
            return self._parse_response(text)

        except Exception as e:
            print(f"🤟 ASL interpretation error: {e}")
            return None

    def _parse_response(self, text: str) -> dict:
        """Parse the structured GPT-4o response."""
        result = {
            "detected": False,
            "type": "none",
            "value": "",
            "raw": "",
        }

        lines = text.strip().split("\n")
        for line in lines:
            line = line.strip()
            if line.upper().startswith("TYPE:"):
                result["type"] = line.split(":", 1)[1].strip().lower()
            elif line.upper().startswith("SIGN:"):
                result["value"] = line.split(":", 1)[1].strip()
            elif line.upper().startswith("SPOKEN:"):
                result["raw"] = line.split(":", 1)[1].strip()

        # If parsing failed, use the raw text
        if not result["raw"] and text:
            result["raw"] = text
            # Try to detect type from content
            t = text.lower()
            if "letter" in t:
                result["type"] = "letter"
            elif any(w in t for w in ["signing", "saying", "sign"]):
                result["type"] = "word"

        # Determine if a sign was detected
        if result["type"] != "none" and result["value"].lower() not in ["none", "no", ""]:
            result["detected"] = True

            # Normalize letter values
            if result["type"] == "letter" and len(result["value"]) == 1:
                result["value"] = result["value"].upper()

        return result