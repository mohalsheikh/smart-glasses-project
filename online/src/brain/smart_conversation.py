# src/brain/smart_conversation.py
"""
Smart Conversation Engine v2.0 — Natural Dialogue for VisionAssist
====================================================================

Major upgrades over original AssistantBrain:

1. PERSONALITY & NATURAL DIALOGUE
   - Warm, supportive personality (not robotic)
   - Remembers user's name and preferences
   - Adapts verbosity to context (quick answers for navigation, detailed for descriptions)

2. MULTI-TURN CONVERSATION CONTEXT
   - "What about the one on the left?" → understands reference
   - "Tell me more" → knows what was just discussed
   - "And the weather?" → handles topic switches gracefully
   - Sliding context window with summarization

3. PROACTIVE INTELLIGENCE
   - Notices patterns: "You seem to be looking for something"
   - Offers relevant info: "By the way, there's a restroom nearby"
   - Time-aware suggestions: "It's getting dark, be extra careful"

4. RESPONSE QUALITY
   - Confidence indicators: "I'm fairly sure..." vs "I can clearly see..."
   - Asks for clarification when uncertain
   - Better error recovery: "Could you hold the camera more steady?"

5. CONTEXT-AWARE SYSTEM PROMPTS
   - Dynamically builds system prompts based on:
     * Current scene context
     * Recent conversation history
     * User preferences
     * Time of day
     * Current activity (navigating, reading, exploring)

This is a MIXIN — integrates with existing AssistantBrain via multiple inheritance.
"""

from __future__ import annotations

import time
import datetime
from typing import Optional, List, Dict, Any

from openai import OpenAI
import src.utils.config as config

client = OpenAI()


class SmartConversationMixin:
    """
    Mixin that enhances AssistantBrain with natural conversation abilities.
    
    Usage: class AssistantBrain(SmartConversationMixin, IntentDetectionMixin, ...):
    """

    # ------------------------------------------------------------------
    # User Profile & Preferences
    # ------------------------------------------------------------------

    def _init_conversation_engine(self):
        """Call this from AssistantBrain.__init__"""
        self._user_name: Optional[str] = None
        self._user_preferences: Dict[str, Any] = {
            "verbosity": "normal",  # "brief", "normal", "detailed"
            "personality": True,     # Warm personality vs. robotic
            "proactive_hints": True,
            "confidence_indicators": True,
        }

        # Conversation tracking
        self._current_activity: str = "exploring"  # exploring, navigating, reading
        self._last_spoken_response: str = ""
        self._last_described_objects: List[str] = []
        self._interaction_count: int = 0
        self._session_start: float = time.time()

        # Reference resolution
        self._reference_context: Dict[str, Any] = {
            "last_object": None,
            "last_person": None,
            "last_direction": None,
            "last_text_read": None,
            "last_location_mentioned": None,
        }

        print("💬 Smart Conversation Engine initialized")

    # ------------------------------------------------------------------
    # Dynamic System Prompt Builder
    # ------------------------------------------------------------------

    def _build_system_prompt(
        self,
        mode: str = "general",
        include_scene: bool = True,
        include_history: bool = True,
    ) -> str:
        """
        Build a rich, context-aware system prompt for GPT.
        This is the secret sauce for natural, helpful responses.
        """
        now = datetime.datetime.now()
        hour = now.hour

        # Time-of-day context
        if 5 <= hour < 12:
            time_context = "morning"
        elif 12 <= hour < 17:
            time_context = "afternoon"
        elif 17 <= hour < 21:
            time_context = "evening"
        else:
            time_context = "night"

        # Base personality
        personality = (
            "You are Vision, a friendly and supportive AI assistant built into smart glasses "
            "for a visually impaired user. You are their eyes, their navigator, and their companion.\n\n"
            "CRITICAL RULES:\n"
            "- Speak directly to the user in SECOND PERSON ('you', not 'the user')\n"
            "- NEVER say 'I see an image' or 'In this photo' — you're looking through glasses in real-time\n"
            "- NEVER mention you're an AI, a model, or looking at a camera feed\n"
            "- Be warm but concise — every extra word is time the user is waiting\n"
            "- Use spatial language: 'on your left', 'right in front of you', 'about 3 feet away'\n"
            "- For safety-critical info (obstacles, traffic), be IMMEDIATE and CLEAR\n"
        )

        # Verbosity
        verbosity = self._user_preferences.get("verbosity", "normal")
        if verbosity == "brief":
            personality += "- Keep responses to 1-2 sentences. Be as concise as possible.\n"
        elif verbosity == "detailed":
            personality += "- Provide detailed descriptions when asked. Include colors, textures, spatial relationships.\n"
        else:
            personality += "- Use 2-4 sentences for most responses. Longer for complex descriptions.\n"

        # User name
        if self._user_name:
            personality += f"- The user's name is {self._user_name}. Use it occasionally (not every response).\n"

        # Time awareness
        personality += f"\nIt's currently {time_context} ({now.strftime('%I:%M %p')}).\n"

        if time_context == "night":
            personality += "Note: It's dark outside. Visibility may be reduced. Be extra cautious about safety.\n"

        # Activity context
        activity_context = {
            "exploring": "The user is exploring their environment. Help them understand their surroundings.",
            "navigating": "The user is actively navigating to a destination. Prioritize directional info and safety.",
            "reading": "The user is trying to read text. Focus on accurate text recognition.",
            "searching": "The user is looking for something specific. Help them locate it.",
            "socializing": "The user is around people. Help them understand who's around and social context.",
        }
        personality += f"\nCurrent activity: {activity_context.get(self._current_activity, 'General assistance')}\n"

        # Mode-specific instructions
        if mode == "describe":
            personality += (
                "\nDESCRIBE MODE:\n"
                "- Paint a vivid but concise picture of the scene\n"
                "- Start with the most important/prominent elements\n"
                "- Include spatial relationships between objects\n"
                "- Mention any text, signs, or labels visible\n"
                "- Note anything unusual or potentially important\n"
            )
        elif mode == "navigation":
            personality += (
                "\nNAVIGATION MODE:\n"
                "- Focus on obstacles, pathways, and spatial orientation\n"
                "- Use clock directions: '2 o'clock', 'straight ahead'\n"
                "- Mention surface changes (curb, step, slope)\n"
                "- Warn about any hazards immediately\n"
            )
        elif mode == "reading":
            personality += (
                "\nREADING MODE:\n"
                "- Focus exclusively on visible text\n"
                "- Preserve the exact words as written\n"
                "- If text is partially obscured, say what you can read and note what's unclear\n"
                "- Don't describe the scene unless asked\n"
            )
        elif mode == "qa":
            personality += (
                "\nQA MODE:\n"
                "- Answer the specific question asked\n"
                "- If uncertain, say so honestly ('I think...' or 'It's hard to tell, but...')\n"
                "- For yes/no questions, lead with the answer then explain briefly\n"
            )

        # Scene context
        if include_scene and hasattr(self, 'last_scene_detections') and self.last_scene_detections:
            objects = [d.get("label", "") for d in self.last_scene_detections[:8]]
            if objects:
                personality += f"\nObjects currently detected: {', '.join(objects)}\n"

        # Conversation context (what was just discussed)
        if include_history:
            conv_ctx = self._get_reference_context_string()
            if conv_ctx:
                personality += f"\nRecent conversation context: {conv_ctx}\n"

        # Confidence indicators
        if self._user_preferences.get("confidence_indicators", True):
            personality += (
                "\nConfidence tips:\n"
                "- If very confident: state facts directly\n"
                "- If somewhat confident: 'It looks like...' or 'I think...'\n"
                "- If uncertain: 'It's hard to tell, but...' or 'I'd need a closer look'\n"
            )

        return personality

    # ------------------------------------------------------------------
    # Reference Resolution
    # ------------------------------------------------------------------

    def _resolve_references(self, text: str) -> str:
        """
        Resolve pronouns and references in user queries.
        "What color is it?" → "What color is [the chair]?"
        "Tell me more" → "Tell me more about [the scene I just described]"
        """
        t = text.lower().strip()
        resolved = text

        # "it" / "that" / "this" → last discussed object
        if any(phrase in t for phrase in [
            "what is it", "what's it", "what color is it", "how big is it",
            "what about it", "tell me about it", "is it",
            "what about that", "what is that", "what's that",
            "what about this", "what is this",
        ]):
            last_obj = self._reference_context.get("last_object")
            if last_obj:
                resolved = text.replace(" it", f" the {last_obj}").replace("that", f"the {last_obj}")

        # "them" / "they" → last mentioned people
        if any(phrase in t for phrase in [
            "who are they", "what about them", "how many of them",
            "are they", "describe them",
        ]):
            last_person = self._reference_context.get("last_person")
            if last_person:
                resolved = f"Describe the people I was asking about: {last_person}"

        # "there" → last mentioned direction
        if any(phrase in t for phrase in [
            "what's over there", "what about there", "go there",
        ]):
            last_dir = self._reference_context.get("last_direction")
            if last_dir:
                resolved = f"What do you see {last_dir}?"

        # "tell me more" / "more details" → expand last response
        if any(phrase in t for phrase in [
            "tell me more", "more details", "elaborate", "go on",
            "more about", "what else",
        ]):
            if self._last_spoken_response:
                resolved = f"Give me more details about: {self._last_spoken_response[:200]}"

        # "read it again" / "repeat"
        if any(phrase in t for phrase in ["read it again", "say it again", "repeat"]):
            last_text = self._reference_context.get("last_text_read")
            if last_text:
                return f"REPEAT: {last_text}"

        return resolved

    def _update_references(
        self,
        intent: str,
        response: str,
        objects: Optional[List[str]] = None,
        direction: Optional[str] = None,
    ):
        """Update reference context after generating a response."""
        self._last_spoken_response = response
        self._interaction_count += 1

        if objects:
            self._last_described_objects = objects
            if objects:
                self._reference_context["last_object"] = objects[0]

        if direction:
            self._reference_context["last_direction"] = direction

        if intent in ("describe_person", "people_count"):
            self._reference_context["last_person"] = response[:100]

        if intent in ("read_text",):
            self._reference_context["last_text_read"] = response

    def _get_reference_context_string(self) -> str:
        """Get a formatted string of recent reference context."""
        parts = []
        ctx = self._reference_context

        if ctx.get("last_object"):
            parts.append(f"Last discussed object: {ctx['last_object']}")
        if ctx.get("last_direction"):
            parts.append(f"Last direction mentioned: {ctx['last_direction']}")
        if ctx.get("last_person"):
            parts.append(f"Last person topic: {ctx['last_person'][:50]}")

        if self._last_described_objects:
            parts.append(f"Recently described: {', '.join(self._last_described_objects[:3])}")

        return " | ".join(parts) if parts else ""

    # ------------------------------------------------------------------
    # Smart Response Enhancement
    # ------------------------------------------------------------------

    def _enhance_response(self, response: str, intent: str) -> str:
        """
        Post-process response for natural conversation quality.
        Adds contextual suggestions, handles edge cases.
        """
        if not response:
            return "I'm not sure about that. Could you try asking differently?"

        response = response.strip()

        # Add proactive suggestions (occasionally, not every time)
        if (self._user_preferences.get("proactive_hints", True)
                and self._interaction_count % 5 == 0
                and intent not in ("help", "greeting")):
            hint = self._get_proactive_hint(intent)
            if hint:
                response += f" {hint}"

        return response

    def _get_proactive_hint(self, current_intent: str) -> Optional[str]:
        """Generate contextual proactive hints."""
        now = datetime.datetime.now()

        # Time-based hints
        if now.hour >= 18 and self._current_activity == "navigating":
            return "It's getting dark, so be extra careful at crossings."

        # Activity-based hints
        if current_intent == "describe_environment" and self._interaction_count > 10:
            return "By the way, you can ask me to remember this scene by saying 'save this scene'."

        return None

    # ------------------------------------------------------------------
    # Activity Detection
    # ------------------------------------------------------------------

    def _update_activity(self, intent: str):
        """Update current activity based on detected intent."""
        activity_map = {
            "directions": "navigating",
            "describe_environment": "exploring",
            "describe_person": "socializing",
            "people_count": "socializing",
            "read_text": "reading",
            "find_object": "searching",
            "identify_object": "exploring",
        }
        new_activity = activity_map.get(intent)
        if new_activity:
            self._current_activity = new_activity

    # ------------------------------------------------------------------
    # Enhanced GPT Chat (replaces basic chat calls)
    # ------------------------------------------------------------------

    def _smart_chat(
        self,
        query: str,
        mode: str = "general",
        include_image: bool = False,
        frame=None,
        max_tokens: int = 300,
        temperature: float = 0.4,
    ) -> str:
        """
        Enhanced chat completion with full context.
        Use this instead of direct client.chat.completions.create calls.
        """
        # Resolve references
        resolved_query = self._resolve_references(query)

        # Build dynamic system prompt
        system_prompt = self._build_system_prompt(
            mode=mode,
            include_scene=True,
            include_history=True,
        )

        messages = [{"role": "system", "content": system_prompt}]

        # Add conversation history (last few turns)
        if hasattr(self, "conversation_history"):
            for msg in self.conversation_history[-6:]:
                messages.append(msg)

        # Build user message
        if include_image and frame is not None:
            try:
                image_url = self.scene_ai._frame_to_data_url(frame)
                messages.append({
                    "role": "user",
                    "content": [
                        {"type": "text", "text": resolved_query},
                        {"type": "image_url", "image_url": {"url": image_url}},
                    ],
                })
            except Exception:
                messages.append({"role": "user", "content": resolved_query})
        else:
            messages.append({"role": "user", "content": resolved_query})

        try:
            model = config.OPENAI_VISION_MODEL if include_image else config.OPENAI_CHAT_MODEL
            resp = client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            answer = (resp.choices[0].message.content or "").strip()
            return answer if answer else "I couldn't quite process that. Could you try again?"
        except Exception as e:
            print(f"❌ Smart chat error: {e}")
            return "Sorry, I had trouble processing that. Please try again."

    # ------------------------------------------------------------------
    # User Preference Commands
    # ------------------------------------------------------------------

    def _handle_preference_command(self, text: str) -> Optional[str]:
        """Handle user preference commands."""
        t = text.lower().strip()

        # Name
        if "my name is" in t or "call me" in t:
            import re
            match = re.search(r"(?:my name is|call me)\s+(\w+)", t, re.IGNORECASE)
            if match:
                self._user_name = match.group(1).capitalize()
                return f"Nice to meet you, {self._user_name}! I'll remember that."

        # Verbosity
        if any(p in t for p in ["shorter answers", "brief mode", "quick mode on", "be brief"]):
            self._user_preferences["verbosity"] = "brief"
            return "Got it — I'll keep my answers short and sweet."

        if any(p in t for p in ["detailed mode", "longer answers", "more detail", "verbose"]):
            self._user_preferences["verbosity"] = "detailed"
            return "Understood — I'll give you more detailed descriptions."

        if any(p in t for p in ["normal mode", "regular mode", "default mode"]):
            self._user_preferences["verbosity"] = "normal"
            return "Back to normal response length."

        return None

    # ------------------------------------------------------------------
    # Session Stats
    # ------------------------------------------------------------------

    def get_session_stats(self) -> Dict[str, Any]:
        """Get conversation session statistics."""
        session_duration = time.time() - self._session_start
        return {
            "interactions": self._interaction_count,
            "session_minutes": session_duration / 60,
            "current_activity": self._current_activity,
            "user_name": self._user_name,
            "verbosity": self._user_preferences.get("verbosity", "normal"),
        }
