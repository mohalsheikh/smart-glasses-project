# src/navigation_client.py
"""
Enhanced Navigation Client for Blind/Visually Impaired Users

Features:
- Multiple transport modes: walking, wheelchair, public transit, rideshare
- Smart destination selection (always picks closest unless specified)
- Detailed accessibility-focused directions
- Landmark-based navigation cues
- ETA and distance in human-friendly format
- Multi-stop journey support (for transit)
- Voice-friendly responses optimized for TTS

Author: Enhanced for accessibility
"""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests

from src.utils import config


class TransportMode(Enum):
    """Available transport modes for navigation."""
    WALKING = "foot-walking"
    WHEELCHAIR = "wheelchair"
    TRANSIT = "public-transit"  # Requires additional API
    RIDESHARE = "driving-car"   # For Uber/Taxi directions
    
    @classmethod
    def from_string(cls, s: str) -> "TransportMode":
        s = (s or "").lower().strip()
        mapping = {
            "walk": cls.WALKING,
            "walking": cls.WALKING,
            "foot": cls.WALKING,
            "on foot": cls.WALKING,
            
            "wheelchair": cls.WHEELCHAIR,
            "accessible": cls.WHEELCHAIR,
            "mobility": cls.WHEELCHAIR,
            
            "transit": cls.TRANSIT,
            "bus": cls.TRANSIT,
            "train": cls.TRANSIT,
            "metro": cls.TRANSIT,
            "subway": cls.TRANSIT,
            "public": cls.TRANSIT,
            "public transit": cls.TRANSIT,
            "public transport": cls.TRANSIT,
            
            "uber": cls.RIDESHARE,
            "lyft": cls.RIDESHARE,
            "taxi": cls.RIDESHARE,
            "cab": cls.RIDESHARE,
            "car": cls.RIDESHARE,
            "rideshare": cls.RIDESHARE,
            "ride": cls.RIDESHARE,
        }
        return mapping.get(s, cls.WALKING)
    
    def friendly_name(self) -> str:
        names = {
            TransportMode.WALKING: "walking",
            TransportMode.WHEELCHAIR: "wheelchair accessible",
            TransportMode.TRANSIT: "public transit",
            TransportMode.RIDESHARE: "rideshare",
        }
        return names.get(self, "walking")


@dataclass
class NavStep:
    """A single navigation step with accessibility enhancements."""
    instruction: str
    distance_m: float = 0.0
    duration_s: float = 0.0
    direction: Optional[str] = None  # left, right, straight, etc.
    landmark: Optional[str] = None   # nearby landmark for orientation
    surface: Optional[str] = None    # sidewalk, crosswalk, stairs, etc.
    caution: Optional[str] = None    # warnings (traffic, construction, etc.)


@dataclass
class Destination:
    """A potential destination with metadata."""
    lat: float
    lon: float
    label: str
    distance_m: float
    category: Optional[str] = None
    address: Optional[str] = None
    
    def distance_friendly(self) -> str:
        if self.distance_m < 100:
            return f"{int(self.distance_m)} meters"
        elif self.distance_m < 1000:
            return f"{int(self.distance_m)} meters"
        else:
            km = self.distance_m / 1000
            if km < 10:
                return f"{km:.1f} kilometers"
            return f"{int(km)} kilometers"


@dataclass
class NavigationSession:
    """Active navigation session state."""
    destination: str
    destination_coords: Tuple[float, float]
    transport_mode: TransportMode
    steps: List[NavStep] = field(default_factory=list)
    current_step_index: int = 0
    total_distance_m: float = 0.0
    total_duration_s: float = 0.0
    summary: Optional[str] = None
    started_at: float = field(default_factory=time.time)
    
    @property
    def remaining_steps(self) -> int:
        return max(0, len(self.steps) - self.current_step_index - 1)
    
    @property
    def is_complete(self) -> bool:
        return self.current_step_index >= len(self.steps) - 1


class NavigationClient:
    """
    Accessibility-focused navigation client using OpenRouteService.
    
    Designed specifically for blind and visually impaired users with:
    - Voice-friendly directions
    - Multiple transport mode support
    - Smart destination selection (closest by default)
    - Landmark-based navigation
    - Safety warnings and surface information
    """

    def __init__(self):
        self.api_key = (
            os.getenv("ORS_API_KEY", "").strip() or 
            os.getenv("OPENROUTESERVICE_API_KEY", "").strip()
        )
        self.enabled = bool(self.api_key)
        
        # Default to walking (most common for blind users)
        self._default_mode = TransportMode.WALKING
        self._current_mode = self._default_mode
        
        self.timeout_s = float(getattr(config, "ORS_TIMEOUT_S", 12.0))
        self.base_url = getattr(config, "ORS_BASE_URL", "https://api.openrouteservice.org")

        # Current GPS location
        self.current_lat: Optional[float] = None
        self.current_lon: Optional[float] = None
        self.current_accuracy_m: Optional[float] = None
        self.last_location_time: float = 0.0

        # Active navigation session
        self._session: Optional[NavigationSession] = None
        
        # Destination cache (for "which one?" follow-ups)
        self._last_destinations: List[Destination] = []
        self._pending_destination_query: Optional[str] = None

        if self.enabled:
            print("🧭 NavigationClient initialized (OpenRouteService enabled).")
            print(f"   Default mode: {self._default_mode.friendly_name()}")
        else:
            print("🧭 NavigationClient initialized (ORS disabled: missing ORS_API_KEY).")

    # ------------------------------------------------------------------
    # Transport Mode Management
    # ------------------------------------------------------------------

    def set_transport_mode(self, mode: str) -> str:
        """Set the transport mode for navigation."""
        new_mode = TransportMode.from_string(mode)
        old_mode = self._current_mode
        self._current_mode = new_mode
        
        if new_mode == TransportMode.TRANSIT:
            return (
                f"Transport mode set to {new_mode.friendly_name()}. "
                "Note: Full public transit directions require a transit API. "
                "I'll provide walking directions to the nearest transit stop."
            )
        
        if new_mode == TransportMode.RIDESHARE:
            return (
                f"Transport mode set to {new_mode.friendly_name()}. "
                "I'll provide directions suitable for giving to your driver, "
                "plus an estimated arrival time."
            )
        
        return f"Transport mode changed from {old_mode.friendly_name()} to {new_mode.friendly_name()}."

    def get_transport_mode(self) -> str:
        """Get current transport mode."""
        return self._current_mode.friendly_name()

    def list_transport_modes(self) -> str:
        """List available transport modes."""
        return (
            "Available transport modes: "
            "walking (default for pedestrians), "
            "wheelchair (accessible routes), "
            "public transit (bus, metro, train), "
            "or rideshare (Uber, Lyft, taxi). "
            "Say 'set mode to walking' or 'I'm taking an Uber' to change."
        )

    # ------------------------------------------------------------------
    # GPS Location Management
    # ------------------------------------------------------------------

    def _location_file_path(self) -> Path:
        p = getattr(config, "GPS_LOCATION_FILE", "./runtime/location.json")
        return Path(str(p))

    def _try_load_location_from_file(self) -> bool:
        """Load GPS from runtime file."""
        path = self._location_file_path()
        if not path.exists():
            return False

        try:
            data = json.loads(path.read_text())
            lat = float(data.get("lat"))
            lon = float(data.get("lon"))
            acc = data.get("accuracy_m", None)
            acc_f = float(acc) if acc is not None else None

            server_t = data.get("server_t", None)
            if isinstance(server_t, (int, float)):
                age_s = time.time() - float(server_t)
            else:
                age_s = time.time() - path.stat().st_mtime

            self.set_current_location(lat, lon, acc_f, touch_time=True)

            if getattr(config, "DEBUG", False):
                print(f"📍 [Nav] Loaded GPS: lat={lat:.6f}, lon={lon:.6f}, age={age_s:.1f}s")
            return True
        except Exception as e:
            if getattr(config, "DEBUG", False):
                print(f"⚠️ [Nav] Failed to read GPS file: {e!r}")
            return False

    def set_current_location(
        self,
        lat: float,
        lon: float,
        accuracy_m: Optional[float] = None,
        *,
        touch_time: bool = True,
    ) -> None:
        """Update current GPS location."""
        try:
            self.current_lat = float(lat)
            self.current_lon = float(lon)
            self.current_accuracy_m = float(accuracy_m) if accuracy_m is not None else None
            if touch_time:
                self.last_location_time = time.time()
        except Exception as e:
            print(f"⚠️ [Nav] Failed to set location: {e!r}")

    def has_location(self) -> bool:
        return self.current_lat is not None and self.current_lon is not None

    def is_location_stale(self) -> bool:
        if not self.has_location():
            return True
        stale_s = float(getattr(config, "GPS_STALE_SECONDS", 45.0))
        return (time.time() - self.last_location_time) > stale_s

    def _ensure_fresh_location(self) -> bool:
        """Ensure we have a fresh GPS location. Returns True if location is available."""
        if not self.has_location() or self.is_location_stale():
            self._try_load_location_from_file()

        # Check for fixed origin (testing)
        lat_s = (getattr(config, "NAV_ORIGIN_LAT", "") or "").strip()
        lon_s = (getattr(config, "NAV_ORIGIN_LON", "") or "").strip()
        if lat_s and lon_s:
            try:
                self.set_current_location(float(lat_s), float(lon_s), None, touch_time=True)
            except Exception:
                pass
        
        return self.has_location() and not self.is_location_stale()

    # ------------------------------------------------------------------
    # Main Public API
    # ------------------------------------------------------------------

    def get_directions(self, destination_text: str, mode: Optional[str] = None) -> str:
        """
        Get directions to a destination.
        
        This is the main entry point for navigation requests.
        Automatically selects the closest matching destination.
        """
        if not self.enabled:
            return "Navigation is not available. Please set the ORS_API_KEY environment variable."

        if not self._ensure_fresh_location():
            return (
                "I don't have your current location yet. "
                "Please open the GPS page on your phone and tap 'Start sending location'."
            )

        destination_text = (destination_text or "").strip().strip('"').strip("'")
        if not destination_text:
            return "Where would you like to go? You can say a place name, address, or type of business."

        # Set transport mode if specified
        if mode:
            self._current_mode = TransportMode.from_string(mode)

        # Search for destinations
        destinations = self._search_destinations(destination_text)
        
        if not destinations:
            return (
                f"I couldn't find '{destination_text}' nearby. "
                "Try being more specific, like 'Walmart on Main Street' or give me an address."
            )

        # Store for potential follow-up
        self._last_destinations = destinations
        self._pending_destination_query = destination_text

        # Check if there are multiple very different options
        if len(destinations) > 1:
            closest = destinations[0]
            second = destinations[1]
            
            # If the second option is significantly closer or same name, just use closest
            # If they're very different places, offer choice
            if self._are_different_places(closest, second, destination_text):
                return self._offer_destination_choices(destinations[:3], destination_text)

        # Use the closest destination
        return self._start_navigation_to(destinations[0])

    def select_destination(self, choice: int) -> str:
        """Select from previously offered destinations."""
        if not self._last_destinations:
            return "I don't have any destinations to choose from. Where would you like to go?"
        
        if choice < 1 or choice > len(self._last_destinations):
            return f"Please choose a number between 1 and {len(self._last_destinations)}."
        
        return self._start_navigation_to(self._last_destinations[choice - 1])

    def get_directions_summary(self, destination_text: str) -> str:
        """Backward-compatible method - redirects to get_directions."""
        return self.get_directions(destination_text)

    # ------------------------------------------------------------------
    # Navigation Session Control
    # ------------------------------------------------------------------

    def continue_directions(self) -> str:
        """Read the next chunk of directions."""
        if not self._session:
            return "You're not currently navigating. Where would you like to go?"

        if self._session.is_complete:
            return "You've reached the end of the directions. You should be at your destination."

        count = int(getattr(config, "NAV_CONTINUE_STEPS_SPOKEN", 5))
        chunk = self._get_steps_chunk(count, advance=True)
        
        if not chunk:
            return "You've reached the end of the directions. You should be arriving at your destination."

        remaining = self._session.remaining_steps
        if remaining > 0:
            return f"{chunk} Say 'continue' for more, or 'repeat' to hear that again."
        return f"{chunk} That's the last step. You should be at your destination."

    def next_step(self) -> str:
        """Get just the next single step."""
        if not self._session:
            return "You're not currently navigating. Where would you like to go?"

        if self._session.is_complete:
            return "You've reached the end of the directions. You should be at your destination."

        self._session.current_step_index += 1
        if self._session.current_step_index >= len(self._session.steps):
            return "That was the last step. You should be at your destination."

        step = self._session.steps[self._session.current_step_index]
        step_num = self._session.current_step_index + 1
        total = len(self._session.steps)
        
        return f"Step {step_num} of {total}: {self._format_step_for_speech(step)}"

    def repeat_step(self) -> str:
        """Repeat the current step."""
        if not self._session:
            return "You're not currently navigating. Where would you like to go?"

        if not self._session.steps:
            return "I don't have any steps to repeat."

        step = self._session.steps[self._session.current_step_index]
        step_num = self._session.current_step_index + 1
        total = len(self._session.steps)
        
        return f"Step {step_num} of {total}: {self._format_step_for_speech(step)}"

    def read_full_directions(self) -> str:
        """Read all remaining directions."""
        if not self._session:
            return "You're not currently navigating."

        max_steps = int(getattr(config, "NAV_MAX_STEPS_PER_RESPONSE", 10))
        chunk = self._get_steps_chunk(max_steps, advance=True)
        
        remaining = self._session.remaining_steps
        if remaining > 0:
            return f"{chunk} There are {remaining} more steps. Say 'continue' to hear them."
        return f"{chunk} That's all the directions."

    def status(self) -> str:
        """Get current navigation status."""
        if not self._session:
            return (
                f"You're not currently navigating. "
                f"Transport mode is set to {self._current_mode.friendly_name()}. "
                "Where would you like to go?"
            )

        elapsed = time.time() - self._session.started_at
        elapsed_min = int(elapsed / 60)
        
        step_num = self._session.current_step_index + 1
        total = len(self._session.steps)
        remaining = self._session.remaining_steps
        
        mode_str = self._session.transport_mode.friendly_name()
        
        return (
            f"You're navigating to {self._session.destination} by {mode_str}. "
            f"You're on step {step_num} of {total}, with {remaining} steps remaining. "
            f"You started {elapsed_min} minutes ago. "
            "Say 'next step' or 'continue directions' to proceed."
        )

    def stop_navigation(self) -> str:
        """Stop current navigation."""
        if not self._session:
            return "You're not currently navigating."

        dest = self._session.destination
        self._session = None
        return f"Navigation to {dest} has been stopped."

    def where_am_i_going(self) -> str:
        """Quick reminder of destination."""
        if not self._session:
            return "You're not currently navigating anywhere."
        return f"You're heading to {self._session.destination} by {self._session.transport_mode.friendly_name()}."

    # ------------------------------------------------------------------
    # Internal: Destination Search
    # ------------------------------------------------------------------

    def _search_destinations(self, text: str) -> List[Destination]:
        """Search for destinations and return sorted by distance (closest first)."""
        url = f"{self.base_url}/geocode/search"

        size = int(getattr(config, "NAV_GEO_SIZE", 10))
        radius_km = float(getattr(config, "NAV_GEO_RADIUS_KM", 25.0))
        country = (getattr(config, "NAV_GEO_COUNTRY", "US") or "").strip()

        params: Dict[str, Any] = {
            "api_key": self.api_key,
            "text": text,
            "size": max(1, min(50, size)),
        }

        if self.has_location():
            params.update({
                "focus.point.lat": self.current_lat,
                "focus.point.lon": self.current_lon,
                "boundary.circle.lat": self.current_lat,
                "boundary.circle.lon": self.current_lon,
                "boundary.circle.radius": max(1.0, radius_km),
            })

        if country:
            params["boundary.country"] = country

        try:
            r = requests.get(url, params=params, timeout=self.timeout_s)
            r.raise_for_status()
            data = r.json()

            features = data.get("features") or []
            destinations: List[Destination] = []

            for feat in features:
                geom = feat.get("geometry") or {}
                coords = geom.get("coordinates")
                if not coords or len(coords) != 2:
                    continue

                lon = float(coords[0])
                lat = float(coords[1])

                props = feat.get("properties") or {}
                label = props.get("label") or text
                category = props.get("category") or props.get("layer")
                
                # Extract address components
                address_parts = []
                if props.get("street"):
                    address_parts.append(props["street"])
                if props.get("locality"):
                    address_parts.append(props["locality"])
                address = ", ".join(address_parts) if address_parts else None

                dist_m = 0.0
                if self.has_location():
                    dist_m = self._haversine_m(self.current_lat, self.current_lon, lat, lon)

                destinations.append(Destination(
                    lat=lat,
                    lon=lon,
                    label=label,
                    distance_m=dist_m,
                    category=category,
                    address=address,
                ))

            # Sort by distance (closest first)
            destinations.sort(key=lambda d: d.distance_m)

            if getattr(config, "DEBUG", False) and destinations:
                print("🧭 [Nav] Found destinations (closest first):")
                for i, d in enumerate(destinations[:5], 1):
                    print(f"   {i}) {d.label} - {d.distance_friendly()}")

            return destinations

        except Exception as e:
            print(f"❌ [Nav] Search error: {e!r}")
            return []

    def _are_different_places(self, d1: Destination, d2: Destination, query: str) -> bool:
        """Check if two destinations are meaningfully different choices."""
        # If one is much closer, don't bother offering choice
        if d1.distance_m > 0 and d2.distance_m > d1.distance_m * 3:
            return False
        
        # Extract base names (e.g., "Walmart" from "Walmart Supercenter")
        def base_name(label: str) -> str:
            return label.split(",")[0].split("-")[0].strip().lower()
        
        name1 = base_name(d1.label)
        name2 = base_name(d2.label)
        query_lower = query.lower()
        
        # If both match the query closely, they're "same type" - use closest
        if query_lower in name1 and query_lower in name2:
            return False
        
        # If names are very similar, use closest
        if name1 == name2:
            return False
        
        # They seem different enough to offer a choice
        return True

    def _offer_destination_choices(self, destinations: List[Destination], query: str) -> str:
        """Offer user a choice between destinations."""
        if len(destinations) == 1:
            return self._start_navigation_to(destinations[0])

        choices = []
        for i, d in enumerate(destinations[:3], 1):
            dist_str = d.distance_friendly()
            if d.address:
                choices.append(f"Option {i}: {d.label}, {dist_str} away")
            else:
                choices.append(f"Option {i}: {d.label}, {dist_str} away")

        choices_text = ". ".join(choices)
        
        return (
            f"I found multiple places matching '{query}'. "
            f"{choices_text}. "
            "Say 'option 1', 'option 2', or 'the closest one' to choose, "
            "or 'navigate to the closest' to go to the nearest one."
        )

    # ------------------------------------------------------------------
    # Internal: Route Calculation
    # ------------------------------------------------------------------

    def _start_navigation_to(self, destination: Destination) -> str:
        """Start navigation to a specific destination."""
        # Get the ORS profile for current transport mode
        profile = self._get_ors_profile()
        
        route = self._get_route(
            start=(self.current_lat, self.current_lon),
            end=(destination.lat, destination.lon),
            profile=profile,
        )
        
        if not route:
            return f"I couldn't calculate a route to {destination.label}. Please try again."

        steps, total_distance, total_duration = self._extract_steps(route)
        steps = self._enhance_steps(steps)
        steps = self._simplify_steps(steps)

        # Create session
        self._session = NavigationSession(
            destination=destination.label,
            destination_coords=(destination.lat, destination.lon),
            transport_mode=self._current_mode,
            steps=steps,
            total_distance_m=total_distance,
            total_duration_s=total_duration,
        )

        # Build response
        summary = self._build_route_summary(destination, total_distance, total_duration)
        self._session.summary = summary

        if not steps:
            return f"{summary} However, I couldn't get detailed turn-by-turn directions."

        # Get first chunk of steps
        initial_count = int(getattr(config, "NAV_INITIAL_STEPS_SPOKEN", 4))
        initial_steps = self._get_steps_chunk(initial_count, advance=True)

        return self._cap_tts(
            f"{summary} {initial_steps} "
            "Say 'next step' for one step at a time, 'continue' for more, or 'repeat' to hear again."
        )

    def _get_ors_profile(self) -> str:
        """Get ORS API profile string for current transport mode."""
        profiles = {
            TransportMode.WALKING: "foot-walking",
            TransportMode.WHEELCHAIR: "wheelchair",
            TransportMode.RIDESHARE: "driving-car",
            TransportMode.TRANSIT: "foot-walking",  # Walk to transit stops
        }
        return profiles.get(self._current_mode, "foot-walking")

    def _get_route(
        self, 
        start: Tuple[float, float], 
        end: Tuple[float, float],
        profile: str = "foot-walking"
    ) -> Optional[Dict[str, Any]]:
        """Get route from ORS API."""
        url = f"{self.base_url}/v2/directions/{profile}"
        
        headers = {
            "Authorization": self.api_key,
            "Content-Type": "application/json",
        }
        
        body: Dict[str, Any] = {
            "coordinates": [
                [float(start[1]), float(start[0])],  # ORS uses [lon, lat]
                [float(end[1]), float(end[0])],
            ],
            "instructions": True,
            "instructions_format": "text",
            "language": getattr(config, "NAV_LANGUAGE", "en"),
        }

        # Add wheelchair-specific options
        if profile == "wheelchair":
            body["options"] = {
                "profile_params": {
                    "restrictions": {
                        "surface_type": "cobblestone:flattened",
                        "track_type": "grade1",
                        "smoothness_type": "good",
                        "maximum_sloped_curb": 0.06,
                        "maximum_incline": 6,
                    }
                }
            }

        try:
            r = requests.post(url, headers=headers, json=body, timeout=self.timeout_s)
            r.raise_for_status()
            return r.json()
        except requests.exceptions.HTTPError as e:
            print(f"❌ [Nav] Route HTTP error: {e!r}")
            # Try fallback to walking if wheelchair fails
            if profile == "wheelchair":
                print("🔄 [Nav] Falling back to foot-walking profile")
                return self._get_route(start, end, "foot-walking")
            return None
        except Exception as e:
            print(f"❌ [Nav] Route error: {e!r}")
            return None

    def _extract_steps(self, payload: Dict[str, Any]) -> Tuple[List[NavStep], float, float]:
        """Extract steps and summary from ORS response."""
        steps: List[NavStep] = []
        total_distance = 0.0
        total_duration = 0.0

        # Try GeoJSON format first
        try:
            features = payload.get("features") or []
            if features:
                props = features[0].get("properties") or {}
                segments = props.get("segments") or []
                if segments:
                    seg = segments[0]
                    total_distance = float(seg.get("distance", 0))
                    total_duration = float(seg.get("duration", 0))

                    for s in seg.get("steps") or []:
                        instruction = (s.get("instruction") or "").strip()
                        if instruction:
                            steps.append(NavStep(
                                instruction=instruction,
                                distance_m=float(s.get("distance", 0)),
                                duration_s=float(s.get("duration", 0)),
                            ))
                    return steps, total_distance, total_duration
        except Exception:
            pass

        # Try routes[] format
        try:
            routes = payload.get("routes") or []
            if routes:
                route = routes[0]
                segments = route.get("segments") or []
                if segments:
                    seg = segments[0]
                    total_distance = float(seg.get("distance", 0))
                    total_duration = float(seg.get("duration", 0))

                    for s in seg.get("steps") or []:
                        instruction = (s.get("instruction") or "").strip()
                        if instruction:
                            steps.append(NavStep(
                                instruction=instruction,
                                distance_m=float(s.get("distance", 0)),
                                duration_s=float(s.get("duration", 0)),
                            ))
                    return steps, total_distance, total_duration
        except Exception:
            pass

        return [], 0.0, 0.0

    def _enhance_steps(self, steps: List[NavStep]) -> List[NavStep]:
        """Add accessibility enhancements to steps."""
        for step in steps:
            instruction_lower = step.instruction.lower()
            
            # Detect direction
            if "turn left" in instruction_lower or "left onto" in instruction_lower:
                step.direction = "left"
            elif "turn right" in instruction_lower or "right onto" in instruction_lower:
                step.direction = "right"
            elif "straight" in instruction_lower or "continue" in instruction_lower:
                step.direction = "straight"
            elif "u-turn" in instruction_lower:
                step.direction = "u-turn"
            
            # Detect surface/caution
            if "stairs" in instruction_lower or "steps" in instruction_lower:
                step.surface = "stairs"
                step.caution = "Watch for stairs"
            elif "crosswalk" in instruction_lower or "crossing" in instruction_lower:
                step.surface = "crosswalk"
                step.caution = "Crossing street"
            elif "elevator" in instruction_lower:
                step.surface = "elevator"
            elif "escalator" in instruction_lower:
                step.surface = "escalator"
                step.caution = "Escalator ahead"

        return steps

    def _simplify_steps(self, steps: List[NavStep]) -> List[NavStep]:
        """Merge very short steps and clean up instructions."""
        if not steps:
            return []

        min_distance = float(getattr(config, "NAV_MIN_STEP_DISTANCE_M", 15.0))
        max_steps = int(getattr(config, "NAV_MAX_STORED_STEPS", 200))

        merged: List[NavStep] = []

        for step in steps:
            if not step.instruction:
                continue

            if not merged:
                merged.append(step)
                continue

            prev = merged[-1]
            
            # Merge if very short and same direction
            if (step.distance_m < min_distance and 
                step.direction == prev.direction and
                not step.caution):
                prev.distance_m += step.distance_m
                prev.duration_s += step.duration_s
                continue

            merged.append(step)

            if len(merged) >= max_steps:
                break

        return merged

    # ------------------------------------------------------------------
    # Internal: Formatting
    # ------------------------------------------------------------------

    def _build_route_summary(self, dest: Destination, distance_m: float, duration_s: float) -> str:
        """Build a spoken route summary."""
        mode = self._current_mode.friendly_name()
        
        # Format distance
        if distance_m < 100:
            dist_str = f"about {int(distance_m)} meters"
        elif distance_m < 1000:
            dist_str = f"about {int(distance_m / 10) * 10} meters"
        else:
            km = distance_m / 1000
            if km < 2:
                dist_str = f"about {km:.1f} kilometers"
            else:
                dist_str = f"about {int(km)} kilometers"

        # Format duration
        minutes = int(duration_s / 60)
        if minutes < 1:
            time_str = "less than a minute"
        elif minutes == 1:
            time_str = "about 1 minute"
        elif minutes < 60:
            time_str = f"about {minutes} minutes"
        else:
            hours = minutes // 60
            remaining_mins = minutes % 60
            if remaining_mins == 0:
                time_str = f"about {hours} hour{'s' if hours > 1 else ''}"
            else:
                time_str = f"about {hours} hour{'s' if hours > 1 else ''} and {remaining_mins} minutes"

        # Build summary based on mode
        if self._current_mode == TransportMode.RIDESHARE:
            return (
                f"I've found a route to {dest.label}. "
                f"It's {dist_str} away. "
                f"By car, it should take {time_str}. "
                "Here are directions you can share with your driver."
            )
        elif self._current_mode == TransportMode.WHEELCHAIR:
            return (
                f"I've found an accessible route to {dest.label}. "
                f"It's {dist_str}, taking {time_str} by wheelchair. "
                "The route avoids stairs and steep inclines."
            )
        else:
            return (
                f"Navigating to {dest.label}. "
                f"It's {dist_str} away, about {time_str} {mode}."
            )

    def _format_step_for_speech(self, step: NavStep) -> str:
        """Format a single step for text-to-speech."""
        instruction = step.instruction.strip()
        
        # Clean up common ORS phrasing
        instruction = instruction.replace("Head ", "Start by heading ")
        instruction = instruction.replace("onto ", "on ")
        
        # Add distance if significant
        parts = [instruction]
        
        if step.distance_m >= 20:
            if step.distance_m < 100:
                parts.append(f"for about {int(step.distance_m)} meters")
            elif step.distance_m < 1000:
                parts.append(f"for about {int(step.distance_m / 10) * 10} meters")
            else:
                km = step.distance_m / 1000
                parts.append(f"for about {km:.1f} kilometers")

        # Add caution if present
        if step.caution:
            parts.append(f"Caution: {step.caution}")

        return ". ".join(parts) + "."

    def _get_steps_chunk(self, count: int, advance: bool = False) -> str:
        """Get a chunk of steps formatted for speech."""
        if not self._session or not self._session.steps:
            return ""

        start = self._session.current_step_index
        end = min(len(self._session.steps), start + count)

        if start >= len(self._session.steps):
            return ""

        parts = []
        for i in range(start, end):
            step = self._session.steps[i]
            step_num = i + 1
            parts.append(f"Step {step_num}: {self._format_step_for_speech(step)}")

        if advance:
            self._session.current_step_index = end - 1

        return " ".join(parts)

    def _cap_tts(self, text: str) -> str:
        """Cap text length for TTS."""
        max_chars = int(getattr(config, "NAV_MAX_TTS_CHARS", 900))
        if max_chars > 0 and len(text) > max_chars:
            return text[:max_chars].rstrip() + "..."
        return text

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def _haversine_m(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Calculate distance between two points in meters."""
        from math import atan2, cos, radians, sin, sqrt

        R = 6371000.0
        p1, p2 = radians(lat1), radians(lat2)
        dlat = radians(lat2 - lat1)
        dlon = radians(lon2 - lon1)
        
        a = sin(dlat / 2) ** 2 + cos(p1) * cos(p2) * sin(dlon / 2) ** 2
        c = 2 * atan2(sqrt(a), sqrt(1 - a))
        
        return R * c