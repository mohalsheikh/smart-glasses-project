# src/brain/handlers/navigation.py
"""
Enhanced Navigation Handlers for Blind/Visually Impaired Users

Handles all navigation-related voice commands with accessibility-first design.
"""

from __future__ import annotations

import re
from typing import Optional


class NavigationHandlersMixin:
    """Mixin class providing navigation handling methods for AssistantBrain."""

    def _handle_directions(self, destination: Optional[str], text: str) -> str:
        """
        Handle navigation/directions queries with full accessibility support.
        
        Supported commands:
        - Start: "directions to Walmart", "navigate to CVS", "how do I get to..."
        - Continue: "continue directions", "keep going", "next steps"
        - Next step: "next step", "next", "what's next"
        - Repeat: "repeat", "say that again", "repeat step"
        - Status: "where am I going", "navigation status"
        - Stop: "stop navigation", "cancel directions"
        - Mode: "I'm walking", "I'm in an Uber", "wheelchair mode"
        - Options: "option 1", "the closest one", "choose first"
        """
        t = (text or "").lower().strip()

        # ------------------------------------------------------------------
        # Transport Mode Changes
        # ------------------------------------------------------------------
        
        # "I'm walking" / "I'm in an Uber" / "set mode to wheelchair"
        mode_change = self._detect_transport_mode_change(t)
        if mode_change:
            try:
                result = self.navigation.set_transport_mode(mode_change)
                return result
            except Exception as e:
                print(f"❌ Error setting transport mode: {e!r}")
                return f"I couldn't change the transport mode. It's currently set to {self.navigation.get_transport_mode()}."

        # "what transport modes" / "list modes"
        if any(p in t for p in ["what modes", "list modes", "transport modes", "available modes", "what are my options for"]):
            try:
                return self.navigation.list_transport_modes()
            except Exception:
                return (
                    "You can navigate by walking, wheelchair, public transit, or rideshare. "
                    "Say something like 'I'm walking' or 'I'm taking an Uber' to change modes."
                )

        # ------------------------------------------------------------------
        # Destination Selection (from previous search)
        # ------------------------------------------------------------------
        
        # "option 1" / "the first one" / "choose option 2"
        option_num = self._extract_option_number(t)
        if option_num:
            try:
                return self.navigation.select_destination(option_num)
            except Exception as e:
                print(f"❌ Error selecting destination: {e!r}")
                return "I couldn't select that option. Please try your search again."

        # "the closest one" / "nearest" / "go to closest"
        if any(p in t for p in ["closest one", "nearest one", "closest", "nearest", "the first"]):
            try:
                return self.navigation.select_destination(1)
            except Exception as e:
                print(f"❌ Error selecting closest: {e!r}")
                return "I couldn't select the closest option. Where would you like to go?"

        # ------------------------------------------------------------------
        # Stop / Cancel Navigation
        # ------------------------------------------------------------------
        
        if any(p in t for p in [
            "stop navigation", "cancel navigation", "end navigation",
            "stop directions", "cancel directions", "end directions",
            "stop route", "cancel route", "i'm done navigating",
            "never mind", "forget it"
        ]):
            try:
                return self.navigation.stop_navigation()
            except Exception as e:
                print(f"❌ Error stopping navigation: {e!r}")
                return "Navigation stopped."

        # ------------------------------------------------------------------
        # Navigation Status
        # ------------------------------------------------------------------
        
        if any(p in t for p in [
            "where am i going", "navigation status", "current route",
            "where are we going", "what's my destination", "status"
        ]):
            try:
                return self.navigation.status()
            except Exception as e:
                print(f"❌ Error getting status: {e!r}")
                return "I couldn't get the navigation status."

        if any(p in t for p in ["where am i going", "my destination", "heading to"]):
            try:
                return self.navigation.where_am_i_going()
            except Exception:
                return "You're not currently navigating anywhere."

        # ------------------------------------------------------------------
        # Repeat Current Step
        # ------------------------------------------------------------------
        
        if any(p in t for p in [
            "repeat", "repeat step", "say that again", "again",
            "what was that", "repeat that", "one more time",
            "say again", "repeat please"
        ]):
            try:
                return self.navigation.repeat_step()
            except Exception as e:
                print(f"❌ Error repeating step: {e!r}")
                return "I couldn't repeat that step."

        # ------------------------------------------------------------------
        # Continue Directions (Multiple Steps)
        # ------------------------------------------------------------------
        
        if any(p in t for p in [
            "continue directions", "continue navigation", "continue route",
            "keep going", "go on", "more directions", "more steps",
            "what's next", "then what", "and then"
        ]):
            try:
                return self.navigation.continue_directions()
            except Exception as e:
                print(f"❌ Error continuing directions: {e!r}")
                return "I couldn't continue the directions."

        # ------------------------------------------------------------------
        # Full Directions (Read All Remaining)
        # ------------------------------------------------------------------
        
        if any(p in t for p in [
            "full directions", "all directions", "read all",
            "all steps", "full route", "complete directions",
            "read full directions", "give me all"
        ]):
            try:
                return self.navigation.read_full_directions()
            except Exception as e:
                print(f"❌ Error reading full directions: {e!r}")
                return "I couldn't read the full directions."

        # ------------------------------------------------------------------
        # Next Step (Single Step)
        # ------------------------------------------------------------------
        
        if any(p in t for p in [
            "next step", "next", "next please", "continue",
            "next direction", "what next", "proceed"
        ]):
            try:
                return self.navigation.next_step()
            except Exception as e:
                print(f"❌ Error getting next step: {e!r}")
                return "I couldn't get the next step."

        # ------------------------------------------------------------------
        # Start New Navigation
        # ------------------------------------------------------------------
        
        # Extract destination from text if not provided
        if not destination:
            destination = self._extract_destination(text)

        if not destination:
            # Check if they're asking how to navigate
            if any(p in t for p in ["how do i navigate", "how do i use navigation", "help with directions"]):
                return (
                    "To get directions, just tell me where you want to go. "
                    "For example, say 'directions to Walmart' or 'how do I get to the nearest coffee shop'. "
                    "You can also specify how you're traveling, like 'walking directions to the park' "
                    "or 'I'm taking an Uber to the airport'."
                )
            return "Where would you like to go? You can say a place name, address, or type of business."

        # Check for transport mode in the request
        mode = self._extract_transport_mode_from_request(t)
        
        try:
            if mode:
                return self.navigation.get_directions(destination, mode=mode)
            return self.navigation.get_directions(destination)
        except Exception as e:
            print(f"❌ Error getting directions: {e!r}")
            return f"I'm having trouble getting directions to {destination}. Please try again."

    def _detect_transport_mode_change(self, text: str) -> Optional[str]:
        """Detect if user is changing transport mode."""
        patterns = [
            # Walking
            (r"i'?m walking", "walking"),
            (r"i'?ll walk", "walking"),
            (r"on foot", "walking"),
            (r"walking mode", "walking"),
            (r"set mode to walk", "walking"),
            
            # Wheelchair
            (r"wheelchair", "wheelchair"),
            (r"accessible", "wheelchair"),
            (r"i use a wheelchair", "wheelchair"),
            (r"mobility", "wheelchair"),
            
            # Rideshare
            (r"i'?m in an? uber", "rideshare"),
            (r"i'?m taking an? uber", "rideshare"),
            (r"i'?m in an? lyft", "rideshare"),
            (r"i'?m taking an? lyft", "rideshare"),
            (r"i'?m in an? taxi", "rideshare"),
            (r"i'?m taking an? taxi", "rideshare"),
            (r"i'?m in an? cab", "rideshare"),
            (r"i'?m taking an? cab", "rideshare"),
            (r"rideshare mode", "rideshare"),
            (r"car mode", "rideshare"),
            (r"driving mode", "rideshare"),
            (r"by car", "rideshare"),
            
            # Transit
            (r"i'?m taking the bus", "transit"),
            (r"i'?m on the bus", "transit"),
            (r"i'?m taking the train", "transit"),
            (r"i'?m taking the metro", "transit"),
            (r"i'?m taking the subway", "transit"),
            (r"public transit", "transit"),
            (r"transit mode", "transit"),
            (r"by bus", "transit"),
            (r"by train", "transit"),
        ]
        
        for pattern, mode in patterns:
            if re.search(pattern, text, re.IGNORECASE):
                return mode
        
        return None

    def _extract_transport_mode_from_request(self, text: str) -> Optional[str]:
        """Extract transport mode from a navigation request."""
        patterns = [
            (r"walking directions", "walking"),
            (r"walk to", "walking"),
            (r"wheelchair directions", "wheelchair"),
            (r"accessible route", "wheelchair"),
            (r"driving directions", "rideshare"),
            (r"by uber", "rideshare"),
            (r"by lyft", "rideshare"),
            (r"by taxi", "rideshare"),
            (r"by cab", "rideshare"),
            (r"transit directions", "transit"),
            (r"bus directions", "transit"),
            (r"by bus", "transit"),
            (r"by train", "transit"),
        ]
        
        for pattern, mode in patterns:
            if re.search(pattern, text, re.IGNORECASE):
                return mode
        
        return None

    def _extract_option_number(self, text: str) -> Optional[int]:
        """Extract option number from text like 'option 1' or 'the second one'."""
        # "option 1", "option 2", etc.
        match = re.search(r"option\s*(\d+)", text, re.IGNORECASE)
        if match:
            return int(match.group(1))
        
        # "choose 1", "select 2", etc.
        match = re.search(r"(?:choose|select|pick)\s*(?:option\s*)?(\d+)", text, re.IGNORECASE)
        if match:
            return int(match.group(1))
        
        # Ordinal words
        ordinals = {
            "first": 1, "second": 2, "third": 3,
            "1st": 1, "2nd": 2, "3rd": 3,
        }
        
        for word, num in ordinals.items():
            if word in text:
                return num
        
        return None

    def _extract_destination(self, text: str) -> Optional[str]:
        """Extract destination from navigation request text."""
        text = (text or "").strip()
        if not text:
            return None

        # Common patterns for destination extraction
        patterns = [
            r"(?:directions?|navigate|route|way|get)\s+to\s+(.+?)(?:\s+by\s+|\s+using\s+|\s*$)",
            r"(?:how\s+(?:do\s+i|can\s+i|to)\s+get\s+to)\s+(.+?)(?:\s+by\s+|\s+using\s+|\s*$)",
            r"(?:take\s+me\s+to)\s+(.+?)(?:\s+by\s+|\s+using\s+|\s*$)",
            r"(?:i\s+(?:want|need)\s+to\s+go\s+to)\s+(.+?)(?:\s+by\s+|\s+using\s+|\s*$)",
            r"(?:going\s+to)\s+(.+?)(?:\s+by\s+|\s+using\s+|\s*$)",
            r"(?:find)\s+(.+?)(?:\s+near|\s+nearby|\s*$)",
            r"(?:where\s+is)\s+(.+?)(?:\s*\?|\s*$)",
            r"(?:nearest|closest)\s+(.+?)(?:\s*$)",
        ]

        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                dest = match.group(1).strip()
                # Clean up common suffixes
                dest = re.sub(r"\s*(please|thanks|thank you|now|right now)\s*$", "", dest, flags=re.IGNORECASE)
                dest = dest.strip("?.!,")
                if dest and len(dest) > 1:
                    return dest

        return None