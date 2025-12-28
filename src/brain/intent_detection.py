# src/brain/intent_detection.py
"""
Enhanced Intent Detection with Navigation Support

Handles all voice command classification including navigation-specific commands
like "continue", "next step", "repeat", etc.
"""

from __future__ import annotations

from typing import Optional, List, Dict, Any
import json
import re

import src.utils.config as config
from src.brain.openai_client import client


class IntentDetectionMixin:
    """Mixin providing intent detection methods for AssistantBrain."""

    # ------------------------------------------------------------------
    # Navigation State Tracking
    # ------------------------------------------------------------------
    
    def _is_navigating(self) -> bool:
        """Check if there's an active navigation session."""
        try:
            if hasattr(self, 'navigation') and self.navigation:
                return hasattr(self.navigation, '_session') and self.navigation._session is not None
        except Exception:
            pass
        return False

    # ------------------------------------------------------------------
    # Pre-classification: Check for navigation commands FIRST
    # ------------------------------------------------------------------

    def _is_navigation_command(self, text: str) -> Optional[Dict[str, Any]]:
        """
        Check if this is a navigation-related command.
        Returns intent dict if yes, None if no.
        
        This runs BEFORE GPT classification to catch navigation commands
        that might be misclassified (like "continue" -> "help").
        """
        t = (text or "").lower().strip()
        
        # Navigation continuation commands (during active navigation)
        continuation_commands = [
            # Continue
            ("continue", ["continue", "continue directions", "continue navigation", 
                         "keep going", "go on", "more directions", "more steps",
                         "what's next", "then what", "and then", "next steps"]),
            # Next step
            ("next_step", ["next step", "next", "next please", "next direction", 
                          "what next", "proceed", "following step"]),
            # Repeat
            ("repeat", ["repeat", "repeat step", "say that again", "again",
                       "what was that", "repeat that", "one more time",
                       "say again", "repeat please", "pardon"]),
            # Full directions
            ("full", ["full directions", "all directions", "read all",
                     "all steps", "full route", "complete directions",
                     "read full directions", "give me all"]),
            # Status
            ("status", ["where am i going", "navigation status", "current route",
                       "where are we going", "what's my destination", "my destination"]),
            # Stop
            ("stop", ["stop navigation", "cancel navigation", "end navigation",
                     "stop directions", "cancel directions", "end directions",
                     "stop route", "cancel route", "i'm done navigating",
                     "never mind navigation"]),
        ]
        
        # Check if any continuation command matches
        for cmd_type, patterns in continuation_commands:
            for pattern in patterns:
                if pattern in t or t == pattern.split()[0]:  # Match "continue" alone too
                    # These commands only make sense during navigation
                    if cmd_type in ["continue", "next_step", "repeat", "full", "status"] and not self._is_navigating():
                        # Not navigating - don't treat as nav command
                        continue
                    
                    return {
                        "primary_intent": "directions",
                        "confidence": 0.95,
                        "secondary_intents": [],
                        "requires_vision": False,
                        "parameters": {"nav_action": cmd_type},
                        "natural_query": text,
                    }
        
        # Transport mode changes
        mode_patterns = [
            (r"i'?m walking", "walking"),
            (r"i'?ll walk", "walking"),
            (r"on foot", "walking"),
            (r"walking mode", "walking"),
            (r"wheelchair", "wheelchair"),
            (r"accessible", "wheelchair"),
            (r"i'?m in an? uber", "rideshare"),
            (r"i'?m taking an? uber", "rideshare"),
            (r"i'?m in an? lyft", "rideshare"),
            (r"i'?m taking an? taxi", "rideshare"),
            (r"i'?m taking the bus", "transit"),
            (r"i'?m on the bus", "transit"),
            (r"public transit", "transit"),
        ]
        
        for pattern, mode in mode_patterns:
            if re.search(pattern, t, re.IGNORECASE):
                return {
                    "primary_intent": "directions",
                    "confidence": 0.95,
                    "secondary_intents": [],
                    "requires_vision": False,
                    "parameters": {"nav_action": "mode_change", "mode": mode},
                    "natural_query": text,
                }
        
        # Destination selection
        if any(p in t for p in ["option 1", "option 2", "option 3", "the closest", "the nearest", "first one", "second one"]):
            option_num = self._extract_option_number(t)
            return {
                "primary_intent": "directions",
                "confidence": 0.95,
                "secondary_intents": [],
                "requires_vision": False,
                "parameters": {"nav_action": "select", "option": option_num or 1},
                "natural_query": text,
            }
        
        # New navigation requests
        nav_triggers = [
            "directions to", "navigate to", "take me to", "how do i get to",
            "route to", "way to", "get to", "go to", "walking directions",
            "driving directions", "transit directions"
        ]
        
        for trigger in nav_triggers:
            if trigger in t:
                dest = self._extract_destination(text)
                return {
                    "primary_intent": "directions",
                    "confidence": 0.9,
                    "secondary_intents": [],
                    "requires_vision": False,
                    "parameters": {"destination": dest},
                    "natural_query": text,
                }
        
        return None

    def _extract_option_number(self, text: str) -> Optional[int]:
        """Extract option number from text like 'option 1' or 'the second one'."""
        match = re.search(r"option\s*(\d+)", text, re.IGNORECASE)
        if match:
            return int(match.group(1))
        
        match = re.search(r"(?:choose|select|pick)\s*(?:option\s*)?(\d+)", text, re.IGNORECASE)
        if match:
            return int(match.group(1))
        
        ordinals = {"first": 1, "second": 2, "third": 3, "1st": 1, "2nd": 2, "3rd": 3}
        for word, num in ordinals.items():
            if word in text.lower():
                return num
        
        if "closest" in text.lower() or "nearest" in text.lower():
            return 1
        
        return None

    # ------------------------------------------------------------------
    # Small helpers
    # ------------------------------------------------------------------

    def _looks_like_appearance_question(self, text: str) -> bool:
        t = text.lower()
        phrases = [
            "do i look good", "do i look pretty", "do i look handsome",
            "am i pretty", "am i good looking", "am i handsome", "am i ugly",
            "do i look okay", "do i look nice",
        ]
        return any(p in t for p in phrases)

    def _safe_json_loads(self, raw: str) -> Optional[Dict[str, Any]]:
        """Parse JSON safely with fallback extraction."""
        if not raw:
            return None

        raw = raw.strip()

        try:
            obj = json.loads(raw)
            return obj if isinstance(obj, dict) else None
        except Exception:
            pass

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
        """Ensure fields exist and are well-typed."""
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
        # FIRST: Check for navigation commands (before GPT)
        nav_intent = self._is_navigation_command(text)
        if nav_intent:
            print(f"🎯 Detected intent: {nav_intent.get('primary_intent')} [nav command] "
                  f"(confidence: {nav_intent.get('confidence', 0)})")
            return nav_intent

        model = config.OPENAI_CHAT_MODEL

        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "system",
                        "content": """You are an intent classifier for a smart glasses assistant for blind users.
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
- directions: User wants navigation/directions to somewhere, OR is continuing navigation
  (includes: "continue", "next step", "repeat", "stop navigation", mode changes like "I'm walking")
- time: User wants current time
- scene_change: User wants to know what changed recently in their surroundings
- appearance_opinion: User asks if they or someone else looks good/pretty/handsome/ugly
- mode_change: User wants shorter or longer responses (e.g., "quick mode on")
- vision_qa: User asks a question about the current visual scene or the state of something
- general_question: General knowledge or conversation
- greeting: Casual greeting or small talk
- help: User needs assistance understanding capabilities

IMPORTANT for navigation:
- "continue", "next", "repeat", "next step", "keep going" during navigation → directions intent
- "I'm walking", "I'm in an Uber", "wheelchair mode" → directions intent (mode change)
- "option 1", "the closest one" → directions intent (destination selection)
- "stop navigation", "cancel directions" → directions intent
- If unsure whether it's navigation-related and user might be navigating, use directions intent

Return JSON format:
{
  "primary_intent": "intent_name",
  "confidence": 0.0-1.0,
  "secondary_intents": ["intent2", "intent3"],
  "requires_vision": true/false,
  "parameters": {
    "object_name": "...",
    "destination": "...",
    "nav_action": "continue|next_step|repeat|stop|mode_change|select",
    "mode_action": "quick_on" or "quick_off",
    "target_language": "...",
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

        # Check navigation commands first (even in fallback)
        nav_intent = self._is_navigation_command(text)
        if nav_intent:
            return nav_intent

        # Visual state / scene questions
        visual_state_markers = [
            "on or off", "on/off", "is it on", "is it off",
            "open or closed", "open/closed", "is it open", "is it closed",
            "is this the", "which one", "what is this", "what am i looking at",
            "is there", "do you see", "can you see",
        ]
        if any(m in t for m in visual_state_markers):
            return {
                "primary_intent": "vision_qa",
                "confidence": 0.85,
                "secondary_intents": [],
                "requires_vision": True,
                "parameters": {},
                "natural_query": text,
            }

        # Mode changes
        if any(k in t for k in ["quick mode", "short mode", "brief mode", "shorter answers"]):
            action = "quick_on" if any(k in t for k in ["on", "enable", "start", "shorter"]) else "quick_off"
            return {
                "primary_intent": "mode_change",
                "confidence": 0.9,
                "secondary_intents": [],
                "requires_vision": False,
                "parameters": {"mode_action": action},
                "natural_query": text,
            }

        # Scene change
        if any(k in t for k in ["what changed", "did anything change", "what's different", "what just happened"]):
            return {
                "primary_intent": "scene_change",
                "confidence": 0.85,
                "secondary_intents": [],
                "requires_vision": True,
                "parameters": {},
                "natural_query": text,
            }

        # Translation
        if any(k in t for k in ["translate", "translation", "in arabic", "in spanish", "in french"]):
            lang = self._parse_target_language(text)
            return {
                "primary_intent": "translate_text",
                "confidence": 0.9,
                "secondary_intents": [],
                "requires_vision": True,
                "parameters": {"target_language": lang},
                "natural_query": text,
            }

        # Read text
        if any(k in t for k in ["read", "what does it say", "what's written", "text"]):
            return {
                "primary_intent": "read_text",
                "confidence": 0.8,
                "secondary_intents": [],
                "requires_vision": True,
                "parameters": {},
                "natural_query": text,
            }

        # Answer visible question
        if any(k in t for k in ["solve", "answer this", "what's the answer", "help me solve"]):
            return {
                "primary_intent": "answer_visible_question",
                "confidence": 0.85,
                "secondary_intents": [],
                "requires_vision": True,
                "parameters": {},
                "natural_query": text,
            }

        # Describe person
        if any(k in t for k in ["describe person", "who is", "who's in front"]):
            return {
                "primary_intent": "describe_person",
                "confidence": 0.8,
                "secondary_intents": [],
                "requires_vision": True,
                "parameters": {},
                "natural_query": text,
            }

        # People count
        if any(k in t for k in ["how many people", "is there anyone", "are there people"]):
            return {
                "primary_intent": "people_count",
                "confidence": 0.8,
                "secondary_intents": [],
                "requires_vision": True,
                "parameters": {},
                "natural_query": text,
            }

        # Describe environment
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

        # Identify object
        if any(k in t for k in ["what is this", "what am i looking at", "identify"]):
            return {
                "primary_intent": "identify_object",
                "confidence": 0.8,
                "secondary_intents": [],
                "requires_vision": True,
                "parameters": {},
                "natural_query": text,
            }

        # Find object
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

        # Weather
        if any(k in t for k in ["weather", "temperature", "hot", "cold", "rain", "sunny"]):
            return {
                "primary_intent": "weather",
                "confidence": 0.9,
                "secondary_intents": [],
                "requires_vision": False,
                "parameters": {},
                "natural_query": text,
            }

        # Directions (basic check - nav commands checked earlier)
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

        # Time
        if any(k in t for k in ["what time", "time is it", "current time"]):
            return {
                "primary_intent": "time",
                "confidence": 0.9,
                "secondary_intents": [],
                "requires_vision": False,
                "parameters": {},
                "natural_query": text,
            }

        # Greeting
        if any(k in t for k in ["hello", "hi", "hey", "good morning", "good afternoon"]):
            return {
                "primary_intent": "greeting",
                "confidence": 0.9,
                "secondary_intents": [],
                "requires_vision": False,
                "parameters": {},
                "natural_query": text,
            }

        # Help
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
        text_clean = (text or "").strip()
        t = text_clean.lower()

        patterns = [
            r"(?:directions?|navigate|route|way|get)\s+to\s+(.+?)(?:\s+by\s+|\s+using\s+|\s*$)",
            r"(?:how\s+(?:do\s+i|can\s+i|to)\s+get\s+to)\s+(.+?)(?:\s+by\s+|\s+using\s+|\s*$)",
            r"(?:take\s+me\s+to)\s+(.+?)(?:\s+by\s+|\s+using\s+|\s*$)",
            r"(?:i\s+(?:want|need)\s+to\s+go\s+to)\s+(.+?)(?:\s+by\s+|\s+using\s+|\s*$)",
            r"(?:going\s+to)\s+(.+?)(?:\s+by\s+|\s+using\s+|\s*$)",
            r"(?:find)\s+(.+?)(?:\s+near|\s+nearby|\s*$)",
            r"(?:nearest|closest)\s+(.+?)(?:\s*$)",
        ]

        for pattern in patterns:
            match = re.search(pattern, t, re.IGNORECASE)
            if match:
                dest = match.group(1).strip()
                dest = re.sub(r"\s*(please|thanks|thank you|now|right now)\s*$", "", dest, flags=re.IGNORECASE)
                dest = dest.strip("?.!,")
                if dest and len(dest) > 1:
                    return dest

        # Simple extraction fallback
        simple_patterns = [
            ("directions to ", "directions to"),
            ("way to ", "way to"),
            ("how do i get to ", "how do i get to"),
            ("navigate to ", "navigate to"),
            ("go to ", "go to"),
            ("take me to ", "take me to"),
        ]

        for pattern, prefix in simple_patterns:
            if pattern in t:
                idx = t.find(pattern)
                dest = text_clean[idx + len(prefix):].strip(" .?!")
                return dest or None

        return None

    def _extract_object_name(self, text: str) -> Optional[str]:
        """Extract object name from queries like 'where is my phone' or 'find my keys'."""
        t = text.lower()

        patterns = [
            "where is my ", "where's my ", "find my ", "locate my ",
            "where is the ", "where's the ", "find the ",
            "can you see my ", "can you see the ",
            "do you see my ", "do you see the ",
        ]

        for pattern in patterns:
            if pattern in t:
                idx = t.find(pattern)
                obj = t[idx + len(pattern):].strip(" .?!")
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
            if any(k in label for k in ["person", "man", "woman", "boy", "girl", "human face", "human body", "people"]):
                count += 1
        return count

    def _parse_target_language(self, text: str) -> str:
        """Parse target language from translation request."""
        t = text.lower()

        languages = [
            "arabic", "english", "spanish", "french", "german",
            "italian", "portuguese", "chinese", "japanese", "korean", "hindi",
        ]

        for lang in languages:
            if f"to {lang}" in t or f"into {lang}" in t or f"in {lang}" in t:
                return lang

        return "english"