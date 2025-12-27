from __future__ import annotations

from typing import Optional, List, Dict, Any
import time

from src.scene_ai_client import SceneAIClient
from src.weather_client import WeatherClient
from src.navigation_client import NavigationClient

from src.brain.intent_detection import IntentDetectionMixin
from src.brain.handlers.vision import VisionHandlersMixin
from src.brain.handlers.navigation import NavigationHandlersMixin
from src.brain.handlers.system import SystemHandlersMixin


class AssistantBrain(IntentDetectionMixin, VisionHandlersMixin, NavigationHandlersMixin, SystemHandlersMixin):
    """
    Enhanced high-level brain for smart glasses with improved natural language understanding.
    Feature-based split:
      - intent_detection.py: intent parsing + helpers
      - handlers/vision.py: all vision handlers
      - handlers/navigation.py: navigation handler(s)
      - handlers/system.py: time/weather/greeting/mode/general chat
    """

    _MAX_HISTORY_TURNS: int = 6  # user+assistant pairs (so ~12 messages)

    def __init__(
        self,
        scene_ai: SceneAIClient,
        weather_client: WeatherClient,
        navigation_client: NavigationClient,
    ):
        self.scene_ai = scene_ai
        self.weather = weather_client
        self.navigation = navigation_client

        # Conversation / behavior flags
        self.conversation_history: List[Dict[str, Any]] = []
        self.quick_mode: bool = False  # If True → shorter answers where possible

        # Scene context for "what just changed?"
        self.last_scene_detections: Optional[List[Dict[str, Any]]] = None
        self.prev_scene_detections: Optional[List[Dict[str, Any]]] = None
        self.last_scene_time: float = 0.0
        self.prev_scene_time: float = 0.0

        print("🧠 Enhanced AssistantBrain initialized.")

    # ------------------------------------------------------------------
    # Scene context (called from controller each time YOLO runs)
    # ------------------------------------------------------------------

    def update_scene_context(
        self,
        *,
        frame,
        detections: Optional[List[Dict[str, Any]]],
    ) -> None:
        now = time.time()
        self.prev_scene_detections = self.last_scene_detections
        self.prev_scene_time = self.last_scene_time

        self.last_scene_detections = list(detections or [])
        self.last_scene_time = now

    # ------------------------------------------------------------------
    # History helpers (bounded)
    # ------------------------------------------------------------------

    def _append_history(self, role: str, content: str) -> None:
        if not content:
            return
        self.conversation_history.append({"role": role, "content": content})

        max_msgs = self._MAX_HISTORY_TURNS * 2
        if len(self.conversation_history) > max_msgs:
            self.conversation_history = self.conversation_history[-max_msgs:]

    def _history_messages(self) -> List[Dict[str, str]]:
        return list(self.conversation_history)

    # ------------------------------------------------------------------
    # Main Entry Point
    # ------------------------------------------------------------------

    def handle_query(
        self,
        text: str,
        *,
        frame,
        detections: Optional[List[Dict[str, Any]]] = None,
    ) -> str:
        if not text or not text.strip():
            return "I didn't hear anything. Could you repeat that?"

        text = text.strip()
        print(f"🧠 AssistantBrain.handle_query: {text!r}")

        # Store user message in bounded history
        self._append_history("user", text)

        intent_data = self._detect_intents(text)
        primary_intent = intent_data.get("primary_intent", "general_question")
        parameters = intent_data.get("parameters", {}) or {}
        secondary_intents = intent_data.get("secondary_intents", []) or []

        # If an intent requires vision but we have no frame, respond gracefully
        requires_vision = bool(intent_data.get("requires_vision", False))

        # If the classifier says this needs vision but labeled it as a general question,
        # route through the general-purpose vision QA handler.
        if primary_intent == "general_question" and requires_vision:
            primary_intent = "vision_qa"
        if requires_vision and frame is None:
            msg = "I need a clearer view to help with that. Try holding the camera steady and closer."
            self._append_history("assistant", msg)
            return msg

        handlers = {
            "read_text": lambda: self._handle_read_text(frame, detections),
            "describe_person": lambda: self._handle_describe_person(frame, detections),
            "people_count": lambda: self._handle_people_presence(detections),
            "describe_environment": lambda: self._handle_describe_env(frame, detections),
            "identify_object": lambda: self._handle_identify_object(frame, detections, text),
            "find_object": lambda: self._handle_find_object(frame, detections, parameters.get("object_name"), text),
            "answer_visible_question": lambda: self._handle_answer_visible_question(frame, detections, text),
            "vision_qa": lambda: self._handle_vision_qa(frame, detections, text),
            "translate_text": lambda: self._handle_translate_visible_text(
                frame,
                detections,
                text,
                parameters.get("target_language"),
            ),
            "weather": lambda: self._handle_weather(text),
            "directions": lambda: self._handle_directions(parameters.get("destination"), text),
            "time": lambda: self._handle_time(),
            "greeting": lambda: self._handle_greeting(text),
            "help": lambda: self._handle_help(),
            "scene_change": lambda: self._handle_scene_change(detections),
            "appearance_opinion": lambda: self._handle_appearance_opinion(frame, detections, text),
            "mode_change": lambda: self._handle_mode_change(parameters.get("mode_action"), text),
            "general_question": lambda: self._handle_general_chat(text, intent_data),
        }

        handler = handlers.get(primary_intent, lambda: self._handle_general_chat(text, intent_data))

        try:
            primary_answer = handler()
        except Exception as e:
            print(f"❌ Error in handler for {primary_intent}: {e!r}")
            primary_answer = "I'm sorry, I ran into an issue processing that. Could you try asking again?"

        # Light multi-intent:
        extra_parts: List[str] = []

        try:
            primary_is_vision = primary_intent in {
                "read_text",
                "describe_person",
                "people_count",
                "describe_environment",
                "identify_object",
                "find_object",
                "answer_visible_question",
                "translate_text",
                "scene_change",
                "appearance_opinion",
            }

            if secondary_intents and len(secondary_intents) <= 3:
                for sec in secondary_intents:
                    if not isinstance(sec, str):
                        continue
                    sec = sec.strip()

                    if not sec or sec == primary_intent:
                        continue

                    # Skip secondary vision intents to avoid extra calls/latency
                    if primary_is_vision and sec in {
                        "read_text",
                        "describe_person",
                        "people_count",
                        "describe_environment",
                        "identify_object",
                        "find_object",
                        "answer_visible_question",
                        "translate_text",
                        "scene_change",
                        "appearance_opinion",
                    }:
                        continue

                    sec_handler = handlers.get(sec)
                    if sec_handler:
                        sec_answer = sec_handler()
                        if sec_answer:
                            extra_parts.append(sec_answer)

        except Exception as e:
            print(f"⚠️ Multi-intent secondary handling error: {e!r}")

        final_answer = primary_answer
        if extra_parts:
            final_answer = primary_answer.strip()
            for part in extra_parts[:2]:
                part = part.strip()
                if not part:
                    continue
                if part.lower() in final_answer.lower():
                    continue
                final_answer += " Also, " + part

        final_answer = (final_answer or "").strip() or "I'm not sure. Could you rephrase that?"

        # Store assistant answer in history
        self._append_history("assistant", final_answer)
        return final_answer
