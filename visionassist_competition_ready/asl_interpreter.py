"""
ASL Sign Language Interpreter for VisionAssist
================================================
Uses GPT-4o Vision to interpret American Sign Language gestures
from camera frames. Supports:
- ASL alphabet fingerspelling (A-Z)
- Common ASL signs and phrases
- Number signs (0-9)
- Full sentence interpretation from sequences

Much more powerful than MediaPipe + classifier approach because
GPT-4o understands context, hand shapes, AND body language.
"""

from __future__ import annotations

import base64
import cv2 as cv
import os
import time
from typing import Optional


class ASLInterpreter:
    """
    Interprets ASL sign language from camera frames using GPT-4o Vision.
    """

    def __init__(self):
        self._client = None
        self._available = False
        self._last_signs: list = []
        self._max_history = 20  # Keep last 20 detected signs for sentence building

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

    def interpret_frame(self, frame, mode: str = "auto") -> str:
        """
        Interpret ASL sign language from a camera frame.

        Args:
            frame: OpenCV BGR image (numpy array)
            mode: "auto" (detect what's being signed),
                  "alphabet" (expect fingerspelling),
                  "phrase" (expect common signs/phrases)

        Returns:
            String describing what was signed, or error message.
        """
        if not self._available or self._client is None:
            return "Sign language interpreter not available."

        if frame is None:
            return "No image available to interpret."

        try:
            # Encode frame to base64 JPEG
            _, buffer = cv.imencode('.jpg', frame, [cv.IMWRITE_JPEG_QUALITY, 85])
            b64_image = base64.b64encode(buffer).decode('utf-8')

            # Build prompt based on mode
            if mode == "alphabet":
                prompt = (
                    "You are an expert ASL (American Sign Language) interpreter. "
                    "Look at this image and identify what ASL fingerspelling letter is being shown. "
                    "Focus on the hand shape, finger positions, and orientation. "
                    "Respond with ONLY the letter or number being signed. "
                    "If no hand is visible or the sign is unclear, say 'No sign detected'. "
                    "Be concise — just the letter/number."
                )
            elif mode == "phrase":
                prompt = (
                    "You are an expert ASL (American Sign Language) interpreter. "
                    "Look at this image and identify what ASL sign or phrase is being shown. "
                    "Consider hand shape, position relative to the body, facial expression, "
                    "and movement implied by the pose. "
                    "Common signs include: hello, thank you, please, sorry, help, "
                    "yes, no, I love you, good, bad, eat, drink, water, more, stop, go. "
                    "Respond with a short description of what is being signed. "
                    "If no sign is detected, say 'No sign detected'."
                )
            else:  # auto
                prompt = (
                    "You are an expert ASL (American Sign Language) interpreter helping "
                    "a blind person understand what a deaf person is signing to them. "
                    "Look at this image carefully and identify any ASL sign language being performed. "
                    "This could be:\n"
                    "- A fingerspelled letter (A-Z)\n"
                    "- A number (0-9)\n"
                    "- A common ASL sign or phrase\n"
                    "- A gesture or expression\n\n"
                    "Focus on: hand shape, finger positions, hand orientation, "
                    "position relative to the body, and facial expression.\n\n"
                    "Respond naturally as if telling a blind person what someone is signing. "
                    "For example: 'They are signing the letter A' or "
                    "'They are signing hello' or 'They are signing I love you'.\n\n"
                    "If no hands are visible or no sign is being made, say "
                    "'I don't see anyone signing right now'.\n"
                    "Keep your response to one short sentence."
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
                max_tokens=150,
                temperature=0.2,
            )

            result = response.choices[0].message.content.strip()

            # Track sign history for sentence building
            if result and "don't see" not in result.lower() and "no sign" not in result.lower():
                self._last_signs.append({
                    "text": result,
                    "time": time.time(),
                })
                # Trim history
                if len(self._last_signs) > self._max_history:
                    self._last_signs = self._last_signs[-self._max_history:]

            return result

        except Exception as e:
            return f"Sorry, I couldn't interpret the sign. Error: {str(e)}"

    def get_recent_signs(self, seconds: float = 30.0) -> str:
        """Get summary of recently detected signs."""
        if not self._last_signs:
            return "No signs detected recently."

        cutoff = time.time() - seconds
        recent = [s["text"] for s in self._last_signs if s["time"] > cutoff]

        if not recent:
            return "No signs detected in the last 30 seconds."

        return "Recent signs: " + " → ".join(recent)

    def clear_history(self):
        """Clear sign detection history."""
        self._last_signs.clear()
