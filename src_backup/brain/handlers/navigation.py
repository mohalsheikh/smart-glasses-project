from __future__ import annotations

from typing import Optional


class NavigationHandlersMixin:
    def _handle_directions(self, destination: Optional[str], text: str) -> str:
        """Handle navigation/directions queries (start/continue/repeat/next/stop)."""
        t = (text or "").lower().strip()

        # Stop / cancel navigation
        if any(
            p in t
            for p in [
                "stop navigation",
                "cancel navigation",
                "end navigation",
                "stop directions",
                "cancel directions",
            ]
        ):
            try:
                return self.navigation.stop_navigation()
            except Exception as e:
                print(f"❌ Error stopping navigation: {e!r}")
                return "Navigation stopped."

        # Repeat current step
        if any(p in t for p in ["repeat", "repeat step", "say that again", "again"]):
            try:
                return self.navigation.repeat_step()
            except Exception as e:
                print(f"❌ Error repeating step: {e!r}")
                return "I couldn't repeat that step right now."

        # Continue directions (chunked)
        if any(p in t for p in ["continue directions", "continue navigation", "keep going", "go on", "continue route"]):
            try:
                if hasattr(self.navigation, "continue_directions"):
                    return self.navigation.continue_directions()
                return self.navigation.next_step()
            except Exception as e:
                print(f"❌ Error continuing directions: {e!r}")
                return "I couldn't continue directions right now."

        # Full directions (remaining, chunked)
        if any(
            p in t
            for p in [
                "full directions",
                "read full directions",
                "read all directions",
                "all directions",
                "all steps",
                "full route",
            ]
        ):
            try:
                if hasattr(self.navigation, "read_full_directions"):
                    return self.navigation.read_full_directions()
                return self.navigation.next_step()
            except Exception as e:
                print(f"❌ Error reading full directions: {e!r}")
                return "I couldn't read the full directions right now."

        # Next step (single-step mode)
        if any(p in t for p in ["next step", "next", "next please", "continue"]):
            try:
                return self.navigation.next_step()
            except Exception as e:
                print(f"❌ Error getting next step: {e!r}")
                return "I couldn't get the next step right now."

        # Start navigation (one-shot)
        if not destination:
            destination = self._extract_destination(text)

        if not destination:
            return "Where would you like to go?"

        try:
            return self.navigation.get_directions_summary(destination)
        except Exception as e:
            print(f"❌ Error in _handle_directions: {e!r}")
            return f"I'm having trouble getting directions to {destination} right now."
