from __future__ import annotations

from typing import Optional, List, Dict, Any
import json

import src.utils.config as config
from src.brain.openai_client import client


class IntentDetectionMixin:
    # -----------------------------
    # Small helpers
    # -----------------------------

    def _looks_like_appearance_question(self, text: str) -> bool:
        t = text.lower()
        phrases = [
            "do i look good",
            "do i look pretty",
            "do i look handsome",
            "am i pretty",
            "am i good looking",
            "am i handsome",
            "am i ugly",
            "do i look okay",
            "do i look nice",
        ]
        return any(p in t for p in phrases)

    def _safe_json_loads(self, raw: str) -> Optional[Dict[str, Any]]:
        """
        Parse JSON safely. If model returns junk around JSON, attempt to extract the JSON object.
        """
        if not raw:
            return None

        raw = raw.strip()

        # First attempt: direct JSON
        try:
            obj = json.loads(raw)
            return obj if isinstance(obj, dict) else None
        except Exception:
            pass

        # Second attempt: extract substring between first '{' and last '}'
        try:
            start = raw.find("{")
            end = raw.rfind("}")
            if start != -1 and end != -1 and end > start:
                candidate = raw[start : end + 1]
                obj = json.loads(candidate)
                return obj if isinstance(obj, dict) else None
        except Exception:
            return None

        return None

    def _normalize_intent_data(self, intent_data: Dict[str, Any], original_text: str) -> Dict[str, Any]:
        """
        Ensure fields exist and are well-typed.
        """
        if not isinstance(intent_data, dict):
            intent_data = {}

        primary = intent_data.get("primary_intent") or "general_question"
        if not isinstance(primary, str):
            primary = "general_question"

        confidence = intent_data.get("confidence", 0.0)
        try:
            confidence = float(confidence)
        except Exception:
            confidence = 0.0

        secondary = intent_data.get("secondary_intents", []) or []
        if not isinstance(secondary, list):
            secondary = []
        secondary = [s for s in secondary if isinstance(s, str)]

        requires_vision = bool(intent_data.get("requires_vision", False))

        params = intent_data.get("parameters", {}) or {}
        if not isinstance(params, dict):
            params = {}

        natural_query = intent_data.get("natural_query") or original_text
        if not isinstance(natural_query, str):
            natural_query = original_text

        normalized = {
            "primary_intent": primary.strip(),
            "confidence": max(0.0, min(1.0, confidence)),
            "secondary_intents": secondary,
            "requires_vision": requires_vision,
            "parameters": params,
            "natural_query": natural_query.strip() or original_text,
        }

        # Patch appearance questions misclassified as greeting
        if normalized["primary_intent"] == "greeting" and self._looks_like_appearance_question(original_text):
            normalized["primary_intent"] = "appearance_opinion"
            normalized["confidence"] = max(normalized["confidence"], 0.9)
            normalized["requires_vision"] = True

        return normalized

    # ------------------------------------------------------------------
    # Smart Intent Detection using GPT + fallback
    # ------------------------------------------------------------------

    def _detect_intents(self, text: str) -> Dict[str, Any]:
        """
        Use GPT to intelligently detect user intent(s) from natural language.
        Returns a structured intent classification with confidence and parameters.
        """
        model = config.OPENAI_CHAT_MODEL

        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "system",
                        "content": """You are an intent classifier for a smart glasses assistant.
Analyze the user's query and return a JSON object with detected intents.

Available intents:
- read_text: User wants to read visible text (signs, labels, documents)
- describe_person: User wants description of a person (appearance, clothing, actions)
- people_count: User wants to know how many people are present
- describe_environment: User wants general scene/surroundings description
- identify_object: User wants to know what a specific object is
- find_object: User wants to locate an object in view
- answer_visible_question: User wants you to read and SOLVE a written question or problem
  that is visible in front of them (on paper, a book, screen, or board)
- translate_text: User wants text translated (often saying things like 'translate this to Arabic')
- weather: User wants weather information
- directions: User wants navigation/directions
- time: User wants current time
- scene_change: User wants to know what changed recently in their surroundings
- appearance_opinion: User asks if they or someone else looks good/pretty/handsome/ugly
- mode_change: User wants shorter or longer responses (e.g., "quick mode on")
- vision_qa: User asks a question about the current visual scene or the state of something they are looking at (e.g., "Is the stove on?", "Is this door open?", "Which button should I press?")
- general_question: General knowledge or conversation
- greeting: Casual greeting or small talk
- help: User needs assistance understanding capabilities

Important rules:
- If the user asks whether they or someone looks good, attractive, pretty, ugly, etc.,
  classify as "appearance_opinion", NOT "greeting".
- If the user asks "what just changed", "did anything change", etc., classify as "scene_change".
- If they say "quick mode on/off", "shorter answers", "more detailed answers", etc.,
  classify as "mode_change" and set parameters.mode_action appropriately
  (e.g. "quick_on" or "quick_off").
- If they say things like "solve this question in front of me", "answer this problem here",
  and refer to something they are looking at, classify as "answer_visible_question".
- If they say things like "translate this", "translate this to Arabic/English/Spanish",
  classify as "translate_text" and include the target language guess in parameters.target_language.

- If the user asks a question that can only be answered by looking at what is in front of them
  (state like on/off, open/closed, full/empty, which item is which, etc.), classify as "vision_qa"
  and set requires_vision=true.

Return JSON format:
{
  "primary_intent": "intent_name",
  "confidence": 0.0-1.0,
  "secondary_intents": ["intent2", "intent3"],
  "requires_vision": true/false,
  "parameters": {
    "object_name": "...",
    "destination": "...",
    "mode_action": "quick_on" or "quick_off",
    "target_language": "arabic" | "english" | "spanish" | ...,
    "location": "...",
    ...
  },
  "natural_query": "rephrased user question for clarity"
}""",
                    },
                    {"role": "user", "content": text},
                ],
                response_format={"type": "json_object"},
                max_tokens=260,
                temperature=0,
            )

            raw = (resp.choices[0].message.content or "").strip()
            parsed = self._safe_json_loads(raw)
            if not parsed:
                print("⚠️ Intent model returned non-JSON; using fallback detection.")
                return self._fallback_intent_detection(text)

            intent_data = self._normalize_intent_data(parsed, text)

            print(
                f"🎯 Detected intent: {intent_data.get('primary_intent')} "
                f"(confidence: {intent_data.get('confidence', 0)})"
            )
            return intent_data

        except Exception as e:
            print(f"⚠️ Intent detection failed, using fallback: {e!r}")
            return self._fallback_intent_detection(text)

    def _fallback_intent_detection(self, text: str) -> Dict[str, Any]:
        """Keyword-based fallback when GPT intent detection fails."""
        t = text.lower()

        # Visual state / scene questions (best-effort fallback)
        visual_state_markers = [
            "on or off", "on/off", "is it on", "is it off",
            "open or closed", "open/closed", "is it open", "is it closed",
            "is this the", "which one", "what is this", "what am i looking at",
            "is there", "do you see", "can you see",
            "is the stove", "is the oven", "is the light", "is the burner",
            "is the door", "is the window", "is the sink",
        ]
        if any(m in t for m in visual_state_markers):
            return {
                "primary_intent": "vision_qa",
                "confidence": 0.75,
                "secondary_intents": [],
                "requires_vision": True,
                "parameters": {},
                "natural_query": text,
            }

        if any(
            phrase in t
            for phrase in [
                "solve this",
                "solve the question",
                "solve this question",
                "solve this problem",
                "answer this",
                "answer this question",
                "answer this problem",
                "can you answer this question",
                "can you answer this problem",
            ]
        ):
            return {
                "primary_intent": "answer_visible_question",
                "confidence": 0.9,
                "secondary_intents": [],
                "requires_vision": True,
                "parameters": {},
                "natural_query": text,
            }

        if "translate" in t or "translation" in t:
            return {
                "primary_intent": "translate_text",
                "confidence": 0.9,
                "secondary_intents": [],
                "requires_vision": True,
                "parameters": {"target_language": self._parse_target_language(text)},
                "natural_query": text,
            }

        if self._looks_like_appearance_question(text):
            return {
                "primary_intent": "appearance_opinion",
                "confidence": 0.9,
                "secondary_intents": [],
                "requires_vision": True,
                "parameters": {},
                "natural_query": text,
            }

        if any(k in t for k in ["what changed", "what just changed", "did anything change", "has anything changed"]):
            return {
                "primary_intent": "scene_change",
                "confidence": 0.9,
                "secondary_intents": [],
                "requires_vision": True,
                "parameters": {},
                "natural_query": text,
            }

        if "quick mode" in t or "shorter answers" in t or "less detail" in t:
            return {
                "primary_intent": "mode_change",
                "confidence": 0.9,
                "secondary_intents": [],
                "requires_vision": False,
                "parameters": {"mode_action": "quick_on"},
                "natural_query": text,
            }
        if "more detail" in t or "longer answers" in t or "detailed answers" in t:
            return {
                "primary_intent": "mode_change",
                "confidence": 0.9,
                "secondary_intents": [],
                "requires_vision": False,
                "parameters": {"mode_action": "quick_off"},
                "natural_query": text,
            }

        if any(k in t for k in ["read", "what does this say", "what's written"]):
            return {
                "primary_intent": "read_text",
                "confidence": 0.8,
                "secondary_intents": [],
                "requires_vision": True,
                "parameters": {},
                "natural_query": text,
            }

        if any(k in t for k in ["describe person", "who is", "who's in front"]):
            return {
                "primary_intent": "describe_person",
                "confidence": 0.8,
                "secondary_intents": [],
                "requires_vision": True,
                "parameters": {},
                "natural_query": text,
            }

        if any(k in t for k in ["how many people", "is there anyone", "are there people"]):
            return {
                "primary_intent": "people_count",
                "confidence": 0.8,
                "secondary_intents": [],
                "requires_vision": True,
                "parameters": {},
                "natural_query": text,
            }

        if any(k in t for k in ["what do you see", "describe", "what's around", "what's in front"]):
            secondary = []
            if any(k in t for k in ["weather", "temperature", "hot", "cold", "rain", "sunny"]):
                secondary.append("weather")

            return {
                "primary_intent": "describe_environment",
                "confidence": 0.7,
                "secondary_intents": secondary,
                "requires_vision": True,
                "parameters": {},
                "natural_query": text,
            }

        if any(k in t for k in ["what is this", "what am i looking at", "identify"]):
            return {
                "primary_intent": "identify_object",
                "confidence": 0.8,
                "secondary_intents": [],
                "requires_vision": True,
                "parameters": {},
                "natural_query": text,
            }

        if any(k in t for k in ["where is", "find", "locate", "can you see"]):
            obj = self._extract_object_name(t)
            return {
                "primary_intent": "find_object",
                "confidence": 0.7,
                "secondary_intents": [],
                "requires_vision": True,
                "parameters": {"object_name": obj},
                "natural_query": text,
            }

        if any(k in t for k in ["weather", "temperature", "hot", "cold", "rain", "sunny"]):
            return {
                "primary_intent": "weather",
                "confidence": 0.9,
                "secondary_intents": [],
                "requires_vision": False,
                "parameters": {},
                "natural_query": text,
            }

        if any(k in t for k in ["directions to", "way to", "how do i get to", "navigate"]):
            dest = self._extract_destination(text)
            return {
                "primary_intent": "directions",
                "confidence": 0.9,
                "secondary_intents": [],
                "requires_vision": False,
                "parameters": {"destination": dest},
                "natural_query": text,
            }

        if any(k in t for k in ["what time", "time is it", "current time"]):
            return {
                "primary_intent": "time",
                "confidence": 0.9,
                "secondary_intents": [],
                "requires_vision": False,
                "parameters": {},
                "natural_query": text,
            }

        if any(k in t for k in ["hello", "hi", "hey", "good morning", "good afternoon"]):
            return {
                "primary_intent": "greeting",
                "confidence": 0.9,
                "secondary_intents": [],
                "requires_vision": False,
                "parameters": {},
                "natural_query": text,
            }

        if any(k in t for k in ["help", "what can you do", "capabilities", "how do you work"]):
            return {
                "primary_intent": "help",
                "confidence": 0.9,
                "secondary_intents": [],
                "requires_vision": False,
                "parameters": {},
                "natural_query": text,
            }

        return {
            "primary_intent": "general_question",
            "confidence": 0.5,
            "secondary_intents": [],
            "requires_vision": False,
            "parameters": {},
            "natural_query": text,
        }

    # ------------------------------------------------------------------
    # Parameter Extraction Helpers
    # ------------------------------------------------------------------

    def _extract_destination(self, text: str) -> Optional[str]:
        """Extract destination from navigation queries."""
        t = text.lower()

        patterns = [
            ("directions to ", "directions to"),
            ("way to ", "way to"),
            ("how do i get to ", "how do i get to"),
            ("navigate to ", "navigate to"),
            ("go to ", "go to"),
            ("take me to ", "take me to"),
        ]

        for pattern, prefix in patterns:
            if pattern in t:
                idx = t.find(pattern)
                dest = text[idx + len(prefix) :].strip(" .?!")
                return dest or None

        return None

    def _extract_object_name(self, text: str) -> Optional[str]:
        """Extract object name from queries like 'where is my phone' or 'find my keys'."""
        t = text.lower()

        patterns = [
            "where is my ",
            "where's my ",
            "find my ",
            "locate my ",
            "where is the ",
            "where's the ",
            "find the ",
            "can you see my ",
            "can you see the ",
            "do you see my ",
            "do you see the ",
        ]

        for pattern in patterns:
            if pattern in t:
                idx = t.find(pattern)
                obj = t[idx + len(pattern) :].strip(" .?!")
                for end in [" in this room", " around here", " nearby"]:
                    if obj.endswith(end):
                        obj = obj[: -len(end)].strip()
                return obj or None

        return None

    def _count_people_from_detections(self, detections: Optional[List[Dict[str, Any]]]) -> int:
        """Count people from YOLO detections."""
        if not detections:
            return 0

        count = 0
        for d in detections:
            label = (d.get("label") or "").lower()
            if any(
                k in label
                for k in ["person", "man", "woman", "boy", "girl", "human face", "human body", "people"]
            ):
                count += 1
        return count

    def _parse_target_language(self, text: str) -> str:
        """
        Very simple language guess: looks for 'to Arabic', 'into Arabic', etc.
        Defaults to 'english' if nothing obvious is found.
        """
        t = text.lower()

        languages = [
            "arabic",
            "english",
            "spanish",
            "french",
            "german",
            "italian",
            "portuguese",
            "chinese",
            "japanese",
            "korean",
            "hindi",
        ]

        for lang in languages:
            if f"to {lang}" in t or f"into {lang}" in t or f"in {lang}" in t:
                return lang

        return "english"
