# src/navigation_client.py
"""
Google Maps Navigation Client for VisionAssist
===============================================

Production-grade navigation using Google Maps Platform APIs:
- Routes API: Accurate turn-by-turn directions
- Places API: Smart place search ("Target", "Starbucks", etc.)
- Geocoding API: Address to coordinates conversion

Optimized for visually impaired users with:
- Natural language directions for TTS
- Safety alerts (crossings, stairs)
- Progress tracking with dynamic ETA
- Multi-modal transport support

Version: 3.0.0 (Google Maps Edition)
"""

from __future__ import annotations

import json
import os
import re
import time
import math
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Deque
from collections import deque
import urllib.parse

import requests

from src.utils import config


# =============================================================================
# ENUMS
# =============================================================================

class TransportMode(Enum):
    """Transport modes supported by Google Maps Routes API."""
    WALKING = "WALK"
    DRIVING = "DRIVE"
    BICYCLE = "BICYCLE"
    TRANSIT = "TRANSIT"
    TWO_WHEELER = "TWO_WHEELER"
    
    @classmethod
    def from_string(cls, s: str) -> "TransportMode":
        s = (s or "").lower().strip()
        mapping = {
            "walk": cls.WALKING, "walking": cls.WALKING, "foot": cls.WALKING,
            "on foot": cls.WALKING, "by foot": cls.WALKING, "pedestrian": cls.WALKING,
            "drive": cls.DRIVING, "driving": cls.DRIVING, "car": cls.DRIVING,
            "uber": cls.DRIVING, "lyft": cls.DRIVING, "taxi": cls.DRIVING,
            "rideshare": cls.DRIVING, "cab": cls.DRIVING,
            "bike": cls.BICYCLE, "bicycle": cls.BICYCLE, "cycling": cls.BICYCLE,
            "transit": cls.TRANSIT, "bus": cls.TRANSIT, "train": cls.TRANSIT,
            "metro": cls.TRANSIT, "subway": cls.TRANSIT, "public": cls.TRANSIT,
            "scooter": cls.TWO_WHEELER, "motorcycle": cls.TWO_WHEELER,
        }
        return mapping.get(s, cls.WALKING)
    
    def friendly_name(self) -> str:
        return {
            TransportMode.WALKING: "walking",
            TransportMode.DRIVING: "driving",
            TransportMode.BICYCLE: "cycling",
            TransportMode.TRANSIT: "public transit",
            TransportMode.TWO_WHEELER: "two-wheeler",
        }.get(self, "walking")


class StepType(Enum):
    """Navigation maneuver types from Google Maps."""
    TURN_LEFT = "TURN_LEFT"
    TURN_RIGHT = "TURN_RIGHT"
    TURN_SLIGHT_LEFT = "TURN_SLIGHT_LEFT"
    TURN_SLIGHT_RIGHT = "TURN_SLIGHT_RIGHT"
    TURN_SHARP_LEFT = "TURN_SHARP_LEFT"
    TURN_SHARP_RIGHT = "TURN_SHARP_RIGHT"
    U_TURN_LEFT = "UTURN_LEFT"
    U_TURN_RIGHT = "UTURN_RIGHT"
    STRAIGHT = "STRAIGHT"
    DEPART = "DEPART"
    ARRIVE = "ARRIVE"
    MERGE = "MERGE"
    RAMP_LEFT = "RAMP_LEFT"
    RAMP_RIGHT = "RAMP_RIGHT"
    FORK_LEFT = "FORK_LEFT"
    FORK_RIGHT = "FORK_RIGHT"
    ROUNDABOUT_LEFT = "ROUNDABOUT_LEFT"
    ROUNDABOUT_RIGHT = "ROUNDABOUT_RIGHT"
    FERRY = "FERRY"
    UNKNOWN = "UNKNOWN"
    
    @classmethod
    def from_google(cls, maneuver: str) -> "StepType":
        maneuver = (maneuver or "").upper().replace("-", "_")
        try:
            return cls(maneuver)
        except ValueError:
            return cls.UNKNOWN


class AlertLevel(Enum):
    INFO = 1
    CAUTION = 2
    WARNING = 3


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class NavStep:
    """A single navigation step with rich metadata."""
    instruction: str
    distance_m: float = 0.0
    duration_s: float = 0.0
    step_type: StepType = StepType.UNKNOWN
    street_name: Optional[str] = None
    maneuver: Optional[str] = None
    start_lat: Optional[float] = None
    start_lon: Optional[float] = None
    end_lat: Optional[float] = None
    end_lon: Optional[float] = None
    polyline: Optional[str] = None
    
    # Accessibility metadata
    has_crossing: bool = False
    has_stairs: bool = False
    caution: Optional[str] = None
    alert_level: AlertLevel = AlertLevel.INFO


@dataclass
class Destination:
    """A place/destination from Google Places API."""
    place_id: str
    name: str
    address: str
    lat: float
    lon: float
    distance_m: float = 0.0
    types: List[str] = field(default_factory=list)
    rating: Optional[float] = None
    is_open: Optional[bool] = None
    
    def distance_friendly(self) -> str:
        if self.distance_m < 50:
            return "very close"
        elif self.distance_m < 100:
            return f"about {int(round(self.distance_m, -1))} meters"
        elif self.distance_m < 1000:
            return f"about {int(round(self.distance_m / 100) * 100)} meters"
        else:
            km = self.distance_m / 1000
            if km < 10:
                return f"about {km:.1f} kilometers"
            return f"about {int(round(km))} kilometers"


@dataclass
class NavigationSession:
    """Active navigation session state."""
    destination_name: str
    destination_address: str
    destination_coords: Tuple[float, float]
    transport_mode: TransportMode
    steps: List[NavStep] = field(default_factory=list)
    current_step_index: int = 0
    total_distance_m: float = 0.0
    total_duration_s: float = 0.0
    remaining_distance_m: float = 0.0
    remaining_duration_s: float = 0.0
    started_at: float = field(default_factory=time.time)
    polyline: Optional[str] = None  # Full route polyline for visualization
    
    # Route metadata
    has_tolls: bool = False
    has_highways: bool = False
    warnings: List[str] = field(default_factory=list)
    
    @property
    def remaining_steps(self) -> int:
        return max(0, len(self.steps) - self.current_step_index - 1)
    
    @property
    def is_complete(self) -> bool:
        return self.current_step_index >= len(self.steps) - 1
    
    @property
    def progress_percent(self) -> int:
        if self.total_distance_m <= 0:
            return 0
        completed = self.total_distance_m - self.remaining_distance_m
        return min(100, int((completed / self.total_distance_m) * 100))
    
    @property
    def elapsed_time_s(self) -> float:
        return time.time() - self.started_at
    
    def update_progress(self, step_index: int) -> None:
        """Update progress after reading steps."""
        # Move to the NEXT step (so we don't repeat)
        self.current_step_index = step_index + 1
        
        # Calculate completed distance based on steps we've passed
        completed_dist = sum(s.distance_m for s in self.steps[:self.current_step_index])
        self.remaining_distance_m = max(0, self.total_distance_m - completed_dist)
        
        # Recalculate ETA based on actual pace
        if self.elapsed_time_s > 0 and completed_dist > 0:
            speed = completed_dist / self.elapsed_time_s
            if speed > 0:
                self.remaining_duration_s = self.remaining_distance_m / speed


# =============================================================================
# MAIN NAVIGATION CLIENT
# =============================================================================

class NavigationClient:
    """
    Google Maps Navigation Client for VisionAssist
    
    Uses Google Maps Platform APIs for accurate, accessible navigation.
    """
    
    # API Endpoints
    ROUTES_API = "https://routes.googleapis.com/directions/v2:computeRoutes"
    PLACES_API = "https://places.googleapis.com/v1/places:searchText"
    PLACES_NEARBY_API = "https://places.googleapis.com/v1/places:searchNearby"
    GEOCODE_API = "https://maps.googleapis.com/maps/api/geocode/json"
    
    # Safety keywords for step analysis
    CROSSING_KEYWORDS = ["cross", "crossing", "crosswalk", "intersection", "traffic light", "pedestrian"]
    STAIRS_KEYWORDS = ["stairs", "steps", "staircase", "stairway"]

    def __init__(self):
        self.api_key = os.getenv("GOOGLE_MAPS_API_KEY", "").strip()
        self.enabled = bool(self.api_key)
        
        # Also check for legacy ORS key for backwards compatibility
        if not self.enabled:
            ors_key = os.getenv("ORS_API_KEY", "").strip()
            if ors_key:
                print("⚠️  [Nav] ORS key found but Google Maps key missing. Using limited functionality.")
        
        self._current_mode = TransportMode.WALKING
        self.timeout_s = 15.0
        
        # GPS state
        self.current_lat: Optional[float] = None
        self.current_lon: Optional[float] = None
        self.current_accuracy_m: Optional[float] = None
        self.last_location_time: float = 0.0
        self.location_history: Deque[Tuple[float, float, float]] = deque(maxlen=10)
        
        # Session state
        self._session: Optional[NavigationSession] = None
        self._last_destinations: List[Destination] = []
        self._pending_query: Optional[str] = None
        
        # Settings
        self.safety_alerts_enabled = True
        self.avoid_highways = False
        self.avoid_tolls = False
        
        if self.enabled:
            print("🧭 NavigationClient initialized (Google Maps Platform).")
            print(f"   Default mode: {self._current_mode.friendly_name()}")
        else:
            print("🧭 NavigationClient initialized (DISABLED: missing GOOGLE_MAPS_API_KEY).")

    # =========================================================================
    # TRANSPORT MODE
    # =========================================================================

    def set_transport_mode(self, mode: str) -> str:
        new_mode = TransportMode.from_string(mode)
        old_mode = self._current_mode
        self._current_mode = new_mode
        
        if new_mode == old_mode:
            return f"Already set to {new_mode.friendly_name()}."
        
        responses = {
            TransportMode.WALKING: "Switched to walking. I'll give pedestrian directions with crosswalk alerts.",
            TransportMode.DRIVING: "Switched to driving. I'll provide car directions.",
            TransportMode.BICYCLE: "Switched to cycling. I'll find bike-friendly routes.",
            TransportMode.TRANSIT: "Switched to transit. I'll include bus and train options.",
        }
        return responses.get(new_mode, f"Mode: {new_mode.friendly_name()}.")

    def get_transport_mode(self) -> str:
        return self._current_mode.friendly_name()

    def list_transport_modes(self) -> str:
        return "Available: walking, driving, cycling, or transit. Say 'switch to walking'."

    # =========================================================================
    # GPS LOCATION
    # =========================================================================

    def _location_file_path(self) -> Path:
        return Path(str(getattr(config, "GPS_LOCATION_FILE", "./runtime/location.json")))

    def _try_load_location_from_file(self) -> bool:
        path = self._location_file_path()
        if not path.exists():
            return False
        try:
            data = json.loads(path.read_text())
            lat = float(data.get("lat"))
            lon = float(data.get("lon"))
            acc = data.get("accuracy_m")
            self.set_current_location(lat, lon, float(acc) if acc else None)
            return True
        except Exception:
            return False

    def set_current_location(self, lat: float, lon: float, accuracy_m: Optional[float] = None) -> None:
        self.current_lat = float(lat)
        self.current_lon = float(lon)
        self.current_accuracy_m = float(accuracy_m) if accuracy_m else None
        self.last_location_time = time.time()
        self.location_history.append((lat, lon, self.last_location_time))

    def has_location(self) -> bool:
        return self.current_lat is not None and self.current_lon is not None

    def is_location_stale(self) -> bool:
        if not self.has_location():
            return True
        stale_s = float(getattr(config, "GPS_STALE_SECONDS", 60.0))
        return (time.time() - self.last_location_time) > stale_s

    def _ensure_fresh_location(self) -> bool:
        if not self.has_location() or self.is_location_stale():
            self._try_load_location_from_file()
        
        # Check for fixed test origin
        lat_s = (getattr(config, "NAV_ORIGIN_LAT", "") or "").strip()
        lon_s = (getattr(config, "NAV_ORIGIN_LON", "") or "").strip()
        if lat_s and lon_s:
            try:
                self.set_current_location(float(lat_s), float(lon_s))
            except Exception:
                pass
        
        return self.has_location()

    # =========================================================================
    # MAIN API: GET DIRECTIONS
    # =========================================================================

    def get_directions(self, destination_text: str, mode: Optional[str] = None) -> str:
        """Main entry point - get directions to a destination."""
        if not self.enabled:
            return "Navigation unavailable. Please add GOOGLE_MAPS_API_KEY to your .env file."

        if not self._ensure_fresh_location():
            return "I need your location. Please enable GPS."

        destination_text = (destination_text or "").strip().strip('"\'')
        if not destination_text:
            return "Where would you like to go?"

        if mode:
            self._current_mode = TransportMode.from_string(mode)

        # Search for places matching the query
        destinations = self._search_places(destination_text)
        
        if not destinations:
            # Try geocoding as fallback (for addresses)
            dest = self._geocode_address(destination_text)
            if dest:
                destinations = [dest]
        
        if not destinations:
            return f"Couldn't find '{destination_text}'. Try the full address or business name."

        self._last_destinations = destinations
        self._pending_query = destination_text

        # If multiple different places, offer choices
        if len(destinations) > 1 and self._should_offer_choices(destinations):
            return self._offer_choices(destinations[:3])

        # Navigate to closest
        return self._start_navigation(destinations[0])

    def select_destination(self, choice: int) -> str:
        """Select from previously offered destinations."""
        if not self._last_destinations:
            return "No options available. Where would you like to go?"
        
        if choice < 1 or choice > len(self._last_destinations):
            return f"Please say 1 to {len(self._last_destinations)}."
        
        return self._start_navigation(self._last_destinations[choice - 1])

    # =========================================================================
    # NAVIGATION CONTROL
    # =========================================================================

    def continue_directions(self) -> str:
        """Get next chunk of directions."""
        if not self._session:
            return "You're not navigating. Where would you like to go?"
        if self._session.is_complete:
            return self._arrival_message()
        
        chunk = self._get_steps_chunk(3, advance=True)
        if not chunk:
            return self._arrival_message()

        progress = self._session.progress_percent
        remaining = self._session.remaining_steps
        
        if remaining > 0:
            return f"{chunk} {progress}% there. Say 'next' for more."
        return f"{chunk} You're arriving."

    def next_step(self) -> str:
        """Get just the next step."""
        if not self._session:
            return "You're not navigating. Where to?"
        if self._session.is_complete:
            return self._arrival_message()

        self._session.current_step_index += 1
        if self._session.current_step_index >= len(self._session.steps):
            return self._arrival_message()

        step = self._session.steps[self._session.current_step_index]
        self._session.update_progress(self._session.current_step_index)
        return self._format_step(step, self._session.current_step_index)

    def repeat_step(self) -> str:
        """Repeat current step."""
        if not self._session or not self._session.steps:
            return "No directions to repeat."
        step = self._session.steps[self._session.current_step_index]
        return self._format_step(step, self._session.current_step_index)

    def read_full_directions(self) -> str:
        """Read all remaining directions."""
        if not self._session:
            return "You're not navigating."
        if self._session.remaining_steps == 0:
            return self._arrival_message()
        
        chunk = self._get_steps_chunk(10, advance=True)
        remaining = self._session.remaining_steps
        if remaining > 0:
            return f"{chunk} {remaining} more steps after that."
        return chunk

    def status(self) -> str:
        """Get navigation status."""
        if not self._session:
            return f"Not navigating. Mode: {self._current_mode.friendly_name()}."

        progress = self._session.progress_percent
        dist = self._format_distance(self._session.remaining_distance_m)
        time_str = self._format_duration(self._session.remaining_duration_s)
        
        return f"Heading to {self._session.destination_name}. {progress}% there, {dist} to go, {time_str}."

    def stop_navigation(self) -> str:
        """Stop current navigation."""
        if not self._session:
            return "You're not navigating."
        name = self._session.destination_name
        progress = self._session.progress_percent
        self._session = None
        return f"Navigation to {name} stopped." + (f" You were {progress}% there." if progress > 20 else "")

    def where_am_i_going(self) -> str:
        """Quick destination reminder."""
        if not self._session:
            return "You're not navigating."
        time_str = self._format_duration(self._session.remaining_duration_s)
        return f"Heading to {self._session.destination_name}, {time_str} away."

    def how_far(self) -> str:
        """Get remaining distance."""
        if not self._session:
            return "You're not navigating."
        dist = self._format_distance(self._session.remaining_distance_m)
        time_str = self._format_duration(self._session.remaining_duration_s)
        return f"{dist} to go, {time_str}."

    def get_upcoming_alerts(self) -> str:
        """Get safety alerts for upcoming steps."""
        if not self._session:
            return "You're not navigating."
        
        alerts = []
        start = self._session.current_step_index
        end = min(len(self._session.steps), start + 3)
        
        for i in range(start, end):
            step = self._session.steps[i]
            if step.has_crossing:
                alerts.append("street crossing")
            if step.has_stairs:
                alerts.append("stairs")
        
        if not alerts:
            return "Path looks clear ahead."
        return "Heads up: " + ", ".join(set(alerts)) + " ahead."

    # =========================================================================
    # GOOGLE PLACES API - SEARCH
    # =========================================================================

    def _search_places(self, query: str) -> List[Destination]:
        """Search for places using Google Places API (New)."""
        if not self.has_location():
            return []
        
        headers = {
            "Content-Type": "application/json",
            "X-Goog-Api-Key": self.api_key,
            "X-Goog-FieldMask": "places.id,places.displayName,places.formattedAddress,places.location,places.types,places.rating,places.currentOpeningHours"
        }
        
        body = {
            "textQuery": query,
            "locationBias": {
                "circle": {
                    "center": {
                        "latitude": self.current_lat,
                        "longitude": self.current_lon
                    },
                    "radius": 50000.0  # 50km radius
                }
            },
            "maxResultCount": 10
        }
        
        try:
            r = requests.post(self.PLACES_API, headers=headers, json=body, timeout=self.timeout_s)
            r.raise_for_status()
            data = r.json()
            
            destinations = []
            for place in data.get("places", []):
                loc = place.get("location", {})
                lat = loc.get("latitude")
                lon = loc.get("longitude")
                
                if lat is None or lon is None:
                    continue
                
                name = place.get("displayName", {}).get("text", query)
                address = place.get("formattedAddress", "")
                
                # Calculate distance
                dist = self._haversine_m(self.current_lat, self.current_lon, lat, lon)
                
                # Check if open
                is_open = None
                hours = place.get("currentOpeningHours", {})
                if hours:
                    is_open = hours.get("openNow")
                
                destinations.append(Destination(
                    place_id=place.get("id", ""),
                    name=name,
                    address=address,
                    lat=lat,
                    lon=lon,
                    distance_m=dist,
                    types=place.get("types", []),
                    rating=place.get("rating"),
                    is_open=is_open,
                ))
            
            # Sort by distance
            destinations.sort(key=lambda d: d.distance_m)
            return destinations
            
        except Exception as e:
            print(f"❌ [Nav] Places search error: {e!r}")
            return []

    def _geocode_address(self, address: str) -> Optional[Destination]:
        """Geocode an address using Google Geocoding API."""
        params = {
            "address": address,
            "key": self.api_key,
        }
        
        if self.has_location():
            params["bounds"] = f"{self.current_lat-0.5},{self.current_lon-0.5}|{self.current_lat+0.5},{self.current_lon+0.5}"
        
        try:
            r = requests.get(self.GEOCODE_API, params=params, timeout=self.timeout_s)
            r.raise_for_status()
            data = r.json()
            
            results = data.get("results", [])
            if not results:
                return None
            
            result = results[0]
            loc = result.get("geometry", {}).get("location", {})
            lat = loc.get("lat")
            lon = loc.get("lng")
            
            if lat is None or lon is None:
                return None
            
            dist = 0.0
            if self.has_location():
                dist = self._haversine_m(self.current_lat, self.current_lon, lat, lon)
            
            return Destination(
                place_id=result.get("place_id", ""),
                name=result.get("formatted_address", address).split(",")[0],
                address=result.get("formatted_address", ""),
                lat=lat,
                lon=lon,
                distance_m=dist,
                types=result.get("types", []),
            )
            
        except Exception as e:
            print(f"❌ [Nav] Geocoding error: {e!r}")
            return None

    # =========================================================================
    # GOOGLE ROUTES API - DIRECTIONS
    # =========================================================================

    def _get_route(self, dest: Destination) -> Optional[Dict[str, Any]]:
        """Get route using Google Routes API."""
        headers = {
            "Content-Type": "application/json",
            "X-Goog-Api-Key": self.api_key,
            "X-Goog-FieldMask": "routes.duration,routes.distanceMeters,routes.polyline,routes.legs,routes.warnings,routes.travelAdvisory"
        }
        
        body = {
            "origin": {
                "location": {
                    "latLng": {
                        "latitude": self.current_lat,
                        "longitude": self.current_lon
                    }
                }
            },
            "destination": {
                "location": {
                    "latLng": {
                        "latitude": dest.lat,
                        "longitude": dest.lon
                    }
                }
            },
            "travelMode": self._current_mode.value,
            "computeAlternativeRoutes": False,
            "languageCode": "en-US",
            "units": "METRIC"
        }
        
        # Add routing preferences
        if self._current_mode == TransportMode.DRIVING:
            modifiers = {}
            if self.avoid_highways:
                modifiers["avoidHighways"] = True
            if self.avoid_tolls:
                modifiers["avoidTolls"] = True
            if modifiers:
                body["routeModifiers"] = modifiers
        
        try:
            r = requests.post(self.ROUTES_API, headers=headers, json=body, timeout=self.timeout_s)
            r.raise_for_status()
            return r.json()
        except requests.exceptions.HTTPError as e:
            print(f"❌ [Nav] Routes API error: {e!r}")
            # Try to get error details
            try:
                error_data = e.response.json()
                print(f"   Error details: {error_data}")
            except:
                pass
            return None
        except Exception as e:
            print(f"❌ [Nav] Routes error: {e!r}")
            return None

    def _parse_route(self, route_data: Dict[str, Any], dest: Destination) -> Optional[NavigationSession]:
        """Parse Google Routes API response into a NavigationSession."""
        routes = route_data.get("routes", [])
        if not routes:
            return None
        
        route = routes[0]
        
        # Get total distance and duration
        total_distance = float(route.get("distanceMeters", 0))
        duration_str = route.get("duration", "0s")
        total_duration = self._parse_duration(duration_str)
        
        # Get polyline for visualization
        polyline = route.get("polyline", {}).get("encodedPolyline")
        
        # Get warnings
        warnings = route.get("warnings", [])
        
        # Check for tolls/highways
        advisory = route.get("travelAdvisory", {})
        has_tolls = bool(advisory.get("tollInfo"))
        
        # Parse steps from legs
        steps: List[NavStep] = []
        legs = route.get("legs", [])
        
        for leg in legs:
            for step_data in leg.get("steps", []):
                step = self._parse_step(step_data)
                if step:
                    steps.append(step)
        
        # Simplify/merge short steps
        steps = self._simplify_steps(steps)
        
        return NavigationSession(
            destination_name=dest.name,
            destination_address=dest.address,
            destination_coords=(dest.lat, dest.lon),
            transport_mode=self._current_mode,
            steps=steps,
            total_distance_m=total_distance,
            total_duration_s=total_duration,
            remaining_distance_m=total_distance,
            remaining_duration_s=total_duration,
            polyline=polyline,
            has_tolls=has_tolls,
            warnings=warnings,
        )

    def _parse_step(self, step_data: Dict[str, Any]) -> Optional[NavStep]:
        """Parse a single step from Routes API response."""
        # Get instruction
        nav_instruction = step_data.get("navigationInstruction", {})
        instruction = nav_instruction.get("instructions", "")
        maneuver = nav_instruction.get("maneuver", "")
        
        if not instruction:
            return None
        
        # Clean HTML from instruction
        instruction = re.sub(r'<[^>]+>', '', instruction)
        instruction = instruction.strip()
        
        # Get distance and duration
        distance = float(step_data.get("distanceMeters", 0))
        duration_str = step_data.get("staticDuration", "0s")
        duration = self._parse_duration(duration_str)
        
        # Get coordinates
        start_loc = step_data.get("startLocation", {}).get("latLng", {})
        end_loc = step_data.get("endLocation", {}).get("latLng", {})
        
        # Get step polyline
        polyline = step_data.get("polyline", {}).get("encodedPolyline")
        
        # Classify step type
        step_type = StepType.from_google(maneuver)
        
        # Create step
        step = NavStep(
            instruction=instruction,
            distance_m=distance,
            duration_s=duration,
            step_type=step_type,
            maneuver=maneuver,
            start_lat=start_loc.get("latitude"),
            start_lon=start_loc.get("longitude"),
            end_lat=end_loc.get("latitude"),
            end_lon=end_loc.get("longitude"),
            polyline=polyline,
        )
        
        # Analyze for safety alerts
        self._analyze_step_safety(step)
        
        return step

    def _analyze_step_safety(self, step: NavStep) -> None:
        """Analyze step for safety concerns."""
        instr_lower = step.instruction.lower()
        
        if any(kw in instr_lower for kw in self.CROSSING_KEYWORDS):
            step.has_crossing = True
            step.caution = "Street crossing"
            step.alert_level = AlertLevel.CAUTION
        
        if any(kw in instr_lower for kw in self.STAIRS_KEYWORDS):
            step.has_stairs = True
            step.caution = "Stairs"
            step.alert_level = AlertLevel.WARNING

    def _simplify_steps(self, steps: List[NavStep]) -> List[NavStep]:
        """Merge very short consecutive steps."""
        if not steps:
            return []
        
        merged: List[NavStep] = []
        min_dist = 10.0  # Minimum distance to keep a step
        
        for step in steps:
            if not step.instruction:
                continue
            
            if not merged:
                merged.append(step)
                continue
            
            prev = merged[-1]
            
            # Merge if very short and same general direction
            if (step.distance_m < min_dist and 
                step.step_type == StepType.STRAIGHT and
                not step.caution):
                prev.distance_m += step.distance_m
                prev.duration_s += step.duration_s
                continue
            
            merged.append(step)
        
        return merged[:100]  # Limit total steps

    def _parse_duration(self, duration_str: str) -> float:
        """Parse duration string like '300s' to seconds."""
        if not duration_str:
            return 0.0
        # Remove 's' suffix and convert
        try:
            return float(duration_str.rstrip('s'))
        except ValueError:
            return 0.0

    # =========================================================================
    # START NAVIGATION
    # =========================================================================

    def _start_navigation(self, dest: Destination) -> str:
        """Start navigation to a destination."""
        route_data = self._get_route(dest)
        
        if not route_data:
            return f"Found {dest.name} but couldn't calculate a route. Please try again."
        
        session = self._parse_route(route_data, dest)
        
        if not session or not session.steps:
            return f"Found {dest.name} but couldn't get directions."
        
        self._session = session
        
        # Build summary
        dist = self._format_distance(session.total_distance_m)
        time_str = self._format_duration(session.total_duration_s)
        mode = self._current_mode.friendly_name()
        
        summary = f"Heading to {dest.name}. {dist} away, {time_str} {mode}."
        
        # Add opening hours info if available
        if dest.is_open is not None:
            if dest.is_open:
                summary += " It's currently open."
            else:
                summary += " Note: It might be closed."
        
        # Get first steps
        initial_steps = self._get_steps_chunk(3, advance=True)
        
        # Add safety note
        safety = ""
        crossing_count = sum(1 for s in session.steps if s.has_crossing)
        if crossing_count > 0 and self.safety_alerts_enabled:
            safety = f" Route has {crossing_count} crossing{'s' if crossing_count > 1 else ''}."
        
        # Add warnings from Google
        if session.warnings:
            safety += f" Note: {session.warnings[0]}"
        
        return self._cap_tts(f"{summary}{safety} {initial_steps} Say 'next' or 'continue' for more.")

    # =========================================================================
    # CHOICE HANDLING
    # =========================================================================

    def _should_offer_choices(self, destinations: List[Destination]) -> bool:
        """Determine if we should offer choices."""
        if len(destinations) < 2:
            return False
        
        d1, d2 = destinations[0], destinations[1]
        
        # If closest is much closer, auto-select it
        if d1.distance_m > 0 and d2.distance_m > d1.distance_m * 3:
            return False
        
        # If same name (e.g., two Targets), pick closest
        if d1.name.lower() == d2.name.lower():
            return False
        
        # Check if names start with same word
        n1_words = d1.name.lower().split()
        n2_words = d2.name.lower().split()
        if n1_words and n2_words and n1_words[0] == n2_words[0]:
            return False
        
        return True

    def _offer_choices(self, destinations: List[Destination]) -> str:
        """Offer destination choices to user."""
        choices = []
        for i, d in enumerate(destinations[:3], 1):
            dist = d.distance_friendly()
            # Simplify address
            addr_short = d.address.split(",")[0] if d.address else ""
            
            if addr_short and addr_short != d.name:
                choices.append(f"Option {i}: {d.name} on {addr_short}, {dist}")
            else:
                choices.append(f"Option {i}: {d.name}, {dist}")
        
        return "Found multiple places. " + " ".join(choices) + " Say 'option 1' or 'the closest'."

    # =========================================================================
    # FORMATTING
    # =========================================================================

    def _format_step(self, step: NavStep, index: int) -> str:
        """Format a single step for TTS."""
        total = len(self._session.steps) if self._session else 1
        instr = step.instruction
        
        parts = []
        
        # Add context occasionally
        if index == 0:
            parts.append("First")
        elif step.step_type == StepType.ARRIVE:
            parts.append("Finally")
        elif index > 0 and index % 5 == 0:
            parts.append(f"Step {index + 1} of {total}")
        
        parts.append(instr)
        
        # Add distance for significant steps
        if step.distance_m >= 30 and step.step_type != StepType.ARRIVE:
            parts.append(f"for {self._format_distance(step.distance_m)}")
        
        # Add caution
        if step.caution and self.safety_alerts_enabled:
            parts.append(f"Caution: {step.caution}")
        
        result = ", ".join(parts) + "."
        
        # Add progress occasionally - calculate based on step index for accuracy
        if self._session and index > 0 and index % 4 == 0:
            # Calculate progress based on distance of steps completed
            completed_dist = sum(s.distance_m for s in self._session.steps[:index + 1])
            if self._session.total_distance_m > 0:
                progress = min(99, int((completed_dist / self._session.total_distance_m) * 100))
                if progress < 95:
                    result += f" {progress}% there."
        
        return result

    def _get_steps_chunk(self, count: int, advance: bool = False) -> str:
        """Get a chunk of formatted steps."""
        if not self._session or not self._session.steps:
            return ""
        
        start = self._session.current_step_index
        end = min(len(self._session.steps), start + count)
        
        if start >= len(self._session.steps):
            return ""
        
        parts = [self._format_step(self._session.steps[i], i) for i in range(start, end)]
        
        if advance:
            # Update progress - pass the last step index we read
            self._session.update_progress(end - 1)
        
        return " ".join(parts)

    def _format_distance(self, meters: float) -> str:
        """Format distance for speech."""
        if meters < 30:
            return "a few steps"
        elif meters < 100:
            return f"about {int(round(meters / 10) * 10)} meters"
        elif meters < 1000:
            return f"about {int(round(meters / 50) * 50)} meters"
        else:
            km = meters / 1000
            if km < 10:
                return f"about {km:.1f} kilometers"
            return f"about {int(round(km))} kilometers"

    def _format_duration(self, seconds: float) -> str:
        """Format duration for speech."""
        mins = int(seconds / 60)
        if mins < 1:
            return "less than a minute"
        elif mins == 1:
            return "about a minute"
        elif mins < 60:
            if mins <= 5:
                return f"about {mins} minutes"
            return f"about {int(round(mins / 5) * 5)} minutes"
        else:
            hours = mins // 60
            rem = mins % 60
            if rem < 5:
                return f"about {hours} hour{'s' if hours > 1 else ''}"
            elif rem < 35:
                return f"about {hours} and a half hours"
            return f"about {hours + 1} hour{'s' if hours + 1 > 1 else ''}"

    def _arrival_message(self) -> str:
        """Build arrival announcement."""
        if not self._session:
            return "You've arrived."
        mins = int(self._session.elapsed_time_s / 60)
        time_taken = "less than a minute" if mins < 1 else f"about {mins} minutes"
        return f"You've arrived at {self._session.destination_name}! That took {time_taken}."

    def _cap_tts(self, text: str) -> str:
        """Cap text length for TTS."""
        max_chars = 1000
        if len(text) > max_chars:
            return text[:max_chars].rstrip() + "..."
        return text

    # =========================================================================
    # UTILITIES
    # =========================================================================

    def _haversine_m(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Calculate distance between two points in meters."""
        R = 6371000.0
        p1, p2 = math.radians(lat1), math.radians(lat2)
        dlat, dlon = math.radians(lat2 - lat1), math.radians(lon2 - lon1)
        a = math.sin(dlat / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlon / 2) ** 2
        return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    def is_navigating(self) -> bool:
        """Check if navigation is active."""
        return self._session is not None

    def get_session_info(self) -> Optional[Dict[str, Any]]:
        """Get session info for external use."""
        if not self._session:
            return None
        return {
            "destination": self._session.destination_name,
            "destination_address": self._session.destination_address,
            "destination_coords": self._session.destination_coords,
            "transport_mode": self._session.transport_mode.friendly_name(),
            "progress_percent": self._session.progress_percent,
            "remaining_distance_m": self._session.remaining_distance_m,
            "remaining_duration_s": self._session.remaining_duration_s,
            "current_step": self._session.current_step_index,
            "total_steps": len(self._session.steps),
            "polyline": self._session.polyline,
        }

    def get_route_polyline(self) -> Optional[str]:
        """Get the encoded polyline for map visualization."""
        if not self._session:
            return None
        return self._session.polyline