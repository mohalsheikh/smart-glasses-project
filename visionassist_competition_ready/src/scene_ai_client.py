# src/scene_ai_client.py

from __future__ import annotations

import os
import time
import base64
from typing import List, Dict, Any
import cv2 as cv

from openai import OpenAI
from src.utils import config

client = OpenAI()


class SceneAIClient:
    """
    Wrapper around the vision model for scene understanding.

    Upgrades:
      ✅ JPEG instead of PNG (smaller payload, faster)
      ✅ Resize image before upload (lower latency)
      ✅ Proper OCR-mode system prompting
      ✅ Strong fallbacks when API unavailable
      ✅ Retries + timeout for better reliability (no crashes)
      ✅ Safer detection text formatting and truncation
    """

    def __init__(self, model: str | None = None):
        self.model = model or config.OPENAI_VISION_MODEL
        self.enabled = bool(os.getenv("OPENAI_API_KEY"))

        if self.enabled:
            print(f"🧠 SceneAIClient initialized with model: {self.model}")
        else:
            print("🧠 SceneAIClient disabled (no OPENAI_API_KEY). Vision will use fallbacks only.")

    def describe_scene(
        self,
        *,
        frame,
        detections: List[Dict[str, Any]],
        question: str,
        fallback_text: str,
        mode: str = "narration",
    ) -> str:
        if not self.enabled:
            if config.DEBUG:
                print("⚠️ SceneAIClient.describe_scene called but AI is disabled; returning fallback.")
            return fallback_text

        if frame is None:
            if config.DEBUG:
                print("⚠️ SceneAIClient.describe_scene called with empty frame; returning fallback.")
            return fallback_text

        try:
            image_url = self._frame_to_data_url(frame)
        except Exception as e:
            print(f"❌ Failed to encode frame to image: {e!r}")
            return fallback_text

        detections_text = self._format_detections(detections)
        system_prompt = self._system_prompt_for_mode(mode)

        safe_question = (question or "").strip()
        if len(safe_question) > 2000:
            safe_question = safe_question[:2000].rstrip() + "..."

        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            f"{safe_question}\n\n"
                            "Here is structured information from an object detector "
                            "(labels may be imperfect, but can help):\n"
                            f"{detections_text}"
                        ),
                    },
                    {"type": "image_url", "image_url": {"url": image_url}},
                ],
            },
        ]

        retries = max(0, int(config.SCENE_AI_RETRIES))
        base_delay = max(0.1, float(config.SCENE_AI_RETRY_BASE_DELAY_S))

        for attempt in range(retries + 1):
            try:
                try:
                    resp = client.chat.completions.create(
                        model=self.model,
                        messages=messages,
                        max_tokens=config.SCENE_AI_MAX_TOKENS,
                        temperature=config.SCENE_AI_TEMPERATURE,
                        timeout=config.SCENE_AI_TIMEOUT_S,
                    )
                except TypeError:
                    resp = client.chat.completions.create(
                        model=self.model,
                        messages=messages,
                        max_tokens=config.SCENE_AI_MAX_TOKENS,
                        temperature=config.SCENE_AI_TEMPERATURE,
                    )

                answer = (resp.choices[0].message.content or "").strip()
                return answer if answer else fallback_text

            except Exception as e:
                if attempt < retries:
                    delay = base_delay * (2 ** attempt)
                    if config.DEBUG:
                        print(f"⚠️ SceneAIClient API error (attempt {attempt+1}/{retries+1}): {e!r}")
                        print(f"↩️ Retrying in {delay:.2f}s...")
                    time.sleep(delay)
                    continue

                print(f"❌ SceneAIClient: API error: {e!r}")
                return fallback_text

        return fallback_text

    # ------------------------------------------------------------------

    def _frame_to_data_url(self, frame) -> str:
        if config.SCENE_AI_FORCE_RGB:
            frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

        h, w = frame.shape[:2]
        max_w = max(64, int(config.SCENE_AI_MAX_WIDTH))
        if w > max_w:
            scale = max_w / float(w)
            new_w = max(1, int(w * scale))
            new_h = max(1, int(h * scale))
            frame = cv.resize(frame, (new_w, new_h), interpolation=cv.INTER_AREA)

        quality = int(config.SCENE_AI_JPEG_QUALITY)
        quality = max(30, min(95, quality))

        success, encoded = cv.imencode(".jpg", frame, [int(cv.IMWRITE_JPEG_QUALITY), quality])
        if not success:
            raise RuntimeError("Failed to encode frame as JPEG.")

        b64 = base64.b64encode(encoded.tobytes()).decode("utf-8")
        return f"data:image/jpeg;base64,{b64}"

    def _format_detections(self, detections: List[Dict[str, Any]]) -> str:
        if not detections:
            return "No detections available."

        lines: List[str] = []
        max_items = max(1, int(config.SCENE_AI_DETECTIONS_MAX_ITEMS))

        for d in detections[:max_items]:
            label = str(d.get("label", "object"))
            try:
                conf = float(d.get("confidence", 0.0))
            except Exception:
                conf = 0.0

            center = d.get("center")
            x_str = y_str = "?"
            if isinstance(center, (list, tuple)) and len(center) == 2:
                try:
                    x_str = f"{float(center[0]):.1f}"
                    y_str = f"{float(center[1]):.1f}"
                except Exception:
                    x_str = y_str = "?"

            lines.append(f"- {label} (conf={conf:.2f}, center=({x_str}, {y_str}))")

        if len(detections) > max_items:
            lines.append(f"- ... and {len(detections) - max_items} more objects.")

        return "\n".join(lines)

    def _system_prompt_for_mode(self, mode: str) -> str:
        base = (
            "You are the eyes for a visually impaired person through their smart glasses.\n"
            "Speak directly to the user in SECOND PERSON, like a trusted friend walking with them.\n"
            "Use short, warm, natural sentences.\n"
            "Do NOT mention you are an AI, an image, or a camera.\n"
            "Speak as if you can see their world in real time.\n"
        )

        mode = (mode or "").strip().lower()

        if mode == "navigation":
            extra = (
                "\nNAVIGATION MODE:\n"
                "- Focus on safety: obstacles, stairs, doors, curbs, vehicles, people.\n"
                "- Use natural directions: on your left, just ahead, a few steps to your right.\n"
                "- Prioritize warnings. Be concise — 1-3 sentences.\n"
            )
        elif mode == "qa":
            extra = (
                "\nQA MODE:\n"
                "- Answer the user's specific question about what you can see.\n"
                "- If uncertain, say so honestly and suggest a better angle or getting closer.\n"
                "- For safety-critical questions (stove, traffic, hazards), err on the side of caution.\n"
                "- Keep the answer clear and actionable.\n"
            )
        elif mode == "ocr":
            extra = (
                "\nOCR MODE:\n"
                "- Focus ONLY on reading visible text.\n"
                "- Preserve the words exactly as written.\n"
                "- If text is too blurry or unclear, say so.\n"
                "- Do not describe the surroundings unless asked.\n"
            )
        else:
            extra = (
                "\nNARRATION MODE:\n"
                "- Describe what's around the user naturally — like a friend telling them what you see.\n"
                "- Start with the most important thing (people, obstacles, key objects).\n"
                "- Mention spatial relationships: 'right in front of you', 'off to your left'.\n"
                "- Include relevant text you can see on signs, screens, or labels.\n"
                "- Keep it to 3-5 sentences max. Don't ramble.\n"
            )

        return base + extra
