from __future__ import annotations

from typing import Optional, List, Dict, Any

import src.utils.config as config
from src.brain.openai_client import client


class SystemHandlersMixin:
    def _handle_weather(self, text: str) -> str:
        """Handle weather queries (supports 'weather in London' + live GPS)."""
        try:
            # IMPORTANT: pass the user's text so the WeatherClient can detect city overrides
            return self.weather.get_weather_summary(text)
        except Exception as e:
            print(f"❌ Error in _handle_weather: {e!r}")
            return "I'm having trouble getting the weather information right now."

    def _handle_time(self) -> str:
        from datetime import datetime

        now = datetime.now()
        time_str = now.strftime("%I:%M %p")
        if time_str.startswith("0"):
            time_str = time_str[1:]
        return f"It's {time_str}."

    def _handle_greeting(self, text: str) -> str:
        from datetime import datetime

        hour = datetime.now().hour
        t = text.lower()

        if "how are you" in t:
            return "I'm doing great, thanks for asking! How can I help you today?"

        if "thank" in t:
            return "You're very welcome! Let me know if you need anything else."

        if "bye" in t or "goodbye" in t:
            return "Goodbye! Have a great day!"

        if hour < 12:
            greeting = "Good morning"
        elif hour < 17:
            greeting = "Good afternoon"
        else:
            greeting = "Good evening"

        return f"{greeting}! How can I assist you today?"

    def _handle_help(self) -> str:
        return (
            "Sure! Here's what I can do for you. "
            "Just say things like: "
            "'what's around me' to hear about your surroundings, "
            "'read this' to read any text you're holding, "
            "'how much money' to identify your bills, "
            "'who's in front of me' to learn about people nearby, "
            "'what time is it' or 'what's the weather', "
            "or just ask me anything and I'll do my best to help."
        )

    def _handle_mode_change(self, mode_action: Optional[str], text: str) -> str:
        t = text.lower()

        if not mode_action:
            if "quick" in t and any(w in t for w in ["off", "normal", "regular", "longer", "more detail"]):
                mode_action = "quick_off"
            elif "quick" in t or "shorter" in t or "less detail" in t:
                mode_action = "quick_on"

        if mode_action == "quick_on":
            self.quick_mode = True
            return "Got it. I'll keep my descriptions shorter and more to the point."
        if mode_action == "quick_off":
            self.quick_mode = False
            return "Okay. I'll go back to giving fuller, more detailed descriptions."

        return "You can say 'quick mode on' for shorter answers or 'quick mode off' for more detailed ones."

    def _handle_general_chat(self, text: str, intent_data: Dict[str, Any]) -> str:
        model = config.OPENAI_CHAT_MODEL

        try:
            system_msg = (
                "You are a warm, helpful voice assistant built into smart glasses for a visually impaired user. "
                "You're like a trusted friend who happens to be able to see for them. "
                "Give clear, concise answers that sound natural when spoken aloud. "
                "Keep responses under 3–4 sentences unless more detail is needed. "
                "Be friendly, confident, and empathetic. "
                "\n\nIMPORTANT: Never mention you are an AI or that you're looking at an image or camera. "
                "If the user asks for something visual and you don't have enough info, "
                "gently suggest they point the camera at what they want to know about."
            )

            messages: List[Dict[str, str]] = [{"role": "system", "content": system_msg}]
            messages.extend(self._history_messages())

            if not messages or messages[-1].get("role") != "user":
                messages.append({"role": "user", "content": text})

            resp = client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=250,
                temperature=0.7,
            )

            answer = (resp.choices[0].message.content or "").strip()
            return answer or "I'm not quite sure how to answer that. Could you rephrase it?"

        except Exception as e:
            print(f"❌ Error in _handle_general_chat: {e!r}")
            return "I'm having trouble thinking of an answer right now. Could you ask me something else?"
