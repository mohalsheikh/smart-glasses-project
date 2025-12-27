from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests

from src.utils import config


@dataclass
class NavStep:
    instruction: str
    distance_m: float = 0.0
    duration_s: float = 0.0


class NavigationClient:
    """
    Real navigation client using OpenRouteService (ORS).

    UX improvements:
      - Speaks multiple steps on start (not just step 1)
      - "continue directions" reads the next chunk of steps
      - "full directions" reads remaining steps in chunks (no huge TTS)
      - Reads GPS from file automatically (runtime/location.json)
    """

    def __init__(self):
        self.api_key = (os.getenv("ORS_API_KEY", "").strip() or os.getenv("OPENROUTESERVICE_API_KEY", "").strip())
        self.enabled = bool(self.api_key)

        self.profile = getattr(config, "NAV_PROFILE", "foot-walking")
        self.timeout_s = float(getattr(config, "ORS_TIMEOUT_S", 12.0))
        self.base_url = getattr(config, "ORS_BASE_URL", "https://api.openrouteservice.org")

        # Current GPS location
        self.current_lat: Optional[float] = None
        self.current_lon: Optional[float] = None
        self.current_accuracy_m: Optional[float] = None
        self.last_location_time: float = 0.0

        # Active navigation session
        self._active_destination: Optional[str] = None
        self._active_steps: List[NavStep] = []
        self._step_index: int = 0
        self._active_summary: Optional[str] = None

        if self.enabled:
            print("🧭 NavigationClient initialized (OpenRouteService enabled).")
        else:
            print("🧭 NavigationClient initialized (ORS disabled: missing ORS_API_KEY).")

    # ------------------------------------------------------------------
    # GPS: load from runtime file
    # ------------------------------------------------------------------

    def _location_file_path(self) -> Path:
        p = getattr(config, "GPS_LOCATION_FILE", "./runtime/location.json")
        return Path(str(p))

    def _try_load_location_from_file(self) -> bool:
        """
        Loads last GPS from runtime file and updates current_lat/current_lon.
        Returns True if loaded successfully.
        """
        path = self._location_file_path()
        if not path.exists():
            return False

        try:
            data = json.loads(path.read_text())
            lat = float(data.get("lat"))
            lon = float(data.get("lon"))
            acc = data.get("accuracy_m", None)
            acc_f = float(acc) if acc is not None else None

            # Age: prefer server_t (seconds), fallback to file mtime
            age_s = None
            server_t = data.get("server_t", None)
            if isinstance(server_t, (int, float)):
                age_s = time.time() - float(server_t)
            else:
                age_s = time.time() - path.stat().st_mtime

            self.set_current_location(lat, lon, acc_f, touch_time=True)

            if getattr(config, "DEBUG", False):
                print(
                    f"📍 [NavigationClient] Loaded GPS from file -> "
                    f"lat={lat}, lon={lon}, acc={acc_f}, age={age_s:.1f}s"
                )
            return True
        except Exception as e:
            if getattr(config, "DEBUG", False):
                print(f"⚠️ [NavigationClient] Failed to read GPS file: {e!r}")
            return False

    def set_current_location(
        self,
        lat: float,
        lon: float,
        accuracy_m: Optional[float] = None,
        *,
        touch_time: bool = True,
    ) -> None:
        try:
            self.current_lat = float(lat)
            self.current_lon = float(lon)
            self.current_accuracy_m = float(accuracy_m) if accuracy_m is not None else None
            if touch_time:
                self.last_location_time = time.time()

            if getattr(config, "DEBUG", False):
                print(
                    f"📍 [NavigationClient] Location updated: "
                    f"lat={self.current_lat}, lon={self.current_lon}, acc={self.current_accuracy_m}"
                )
        except Exception as e:
            print(f"⚠️ [NavigationClient] Failed to set location: {e!r}")

    def has_location(self) -> bool:
        return self.current_lat is not None and self.current_lon is not None

    def is_location_stale(self) -> bool:
        if not self.has_location():
            return True
        stale_s = float(getattr(config, "GPS_STALE_SECONDS", 45.0))
        return (time.time() - float(self.last_location_time)) > stale_s

    def _ensure_fresh_location(self) -> None:
        """
        If we don't have a location (or it's stale), try reading from file.
        """
        if not self.has_location() or self.is_location_stale():
            self._try_load_location_from_file()

        # Optional fixed origin for laptop testing
        lat_s = (getattr(config, "NAV_ORIGIN_LAT", "") or "").strip()
        lon_s = (getattr(config, "NAV_ORIGIN_LON", "") or "").strip()
        if lat_s and lon_s:
            try:
                self.set_current_location(float(lat_s), float(lon_s), None, touch_time=True)
                if getattr(config, "DEBUG", False):
                    print("🧭 [NavigationClient] Using fixed NAV_ORIGIN_LAT/LON for testing.")
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Public API used by AssistantBrain
    # ------------------------------------------------------------------

    def get_directions_summary(self, destination_text: str) -> str:
        """
        Start navigation: speak summary + first chunk of steps (not just step 1).
        """
        if not self.enabled:
            return "Navigation is not enabled. Please set the ORS_API_KEY environment variable."

        self._ensure_fresh_location()
        if not self.has_location() or self.is_location_stale():
            return "I don't have your current location yet. Open the GPS page and press 'Start sending location'."

        destination_text = (destination_text or "").strip().strip('"').strip("'")
        if not destination_text:
            return "Where would you like to go?"

        # 1) Geocode destination (biased to your GPS)
        dest = self._geocode_destination(destination_text)
        if not dest:
            return f"I couldn't find '{destination_text}'. Try a more specific place name or an address."

        dest_lat, dest_lon, dest_label, dest_dist_m = dest

        # sanity check if it looks insanely far
        max_km = float(getattr(config, "NAV_MAX_REASONABLE_KM", 150.0))
        if dest_dist_m > 0 and (dest_dist_m / 1000.0) > max_km:
            return (
                f"I found a '{destination_text}', but it looks very far away "
                f"(about {dest_dist_m/1000:.0f} km). "
                "Try saying the city too, like: 'Walmart in Riverside' or give an address."
            )

        # 2) Route from current -> destination
        route_payload = self._get_route(
            start=(self.current_lat, self.current_lon),
            end=(dest_lat, dest_lon),
        )
        if not route_payload:
            return f"I couldn't get a route to {destination_text} right now."

        # 3) Parse steps
        steps, total_distance_m, total_duration_s = self._extract_steps_and_summary(route_payload)

        # 4) Simplify micro-steps
        steps = self._simplify_steps(steps)

        summary = self._format_route_summary(total_distance_m, total_duration_s, dest_label or destination_text)
        self._start_session(destination_text, steps, summary)

        if not steps:
            return f"{summary} I couldn't extract turn-by-turn steps."

        # Speak first chunk
        n0 = int(getattr(config, "NAV_INITIAL_STEPS_SPOKEN", 4))
        n0 = max(1, min(n0, int(getattr(config, "NAV_MAX_STEPS_PER_RESPONSE", 10))))
        chunk = self._get_steps_chunk(start_index=0, count=n0, advance_index=True)

        msg = f"{summary} {chunk} Say 'continue directions' for the next steps."
        return self._cap_tts(msg)

    def continue_directions(self) -> str:
        """
        Speak the next chunk of steps (5 by default).
        """
        if not self._active_destination:
            return "You're not navigating right now. You can say: give me directions to Walmart."

        if not self._active_steps:
            return f"You're navigating to {self._active_destination}, but I don't have turn-by-turn steps."

        count = int(getattr(config, "NAV_CONTINUE_STEPS_SPOKEN", 5))
        count = max(1, min(count, int(getattr(config, "NAV_MAX_STEPS_PER_RESPONSE", 10))))
        chunk = self._get_steps_chunk(start_index=self._step_index, count=count, advance_index=True)

        if not chunk:
            return "There are no more steps. You're almost there."

        remaining = (len(self._active_steps) - 1) - self._step_index
        if remaining <= 0:
            return self._cap_tts(f"{chunk} That was the final step. You're almost there.")

        return self._cap_tts(f"{chunk} Say 'continue directions' for more, or 'repeat'.")

    def read_full_directions(self) -> str:
        """
        Reads the remaining steps in *one response*, but capped (so TTS doesn't explode).
        If there are more steps left beyond the cap, it will tell you to continue.
        """
        if not self._active_destination:
            return "You're not navigating right now."

        if not self._active_steps:
            return f"You're navigating to {self._active_destination}, but I don't have turn-by-turn steps."

        max_steps = int(getattr(config, "NAV_MAX_STEPS_PER_RESPONSE", 10))
        chunk = self._get_steps_chunk(start_index=self._step_index, count=max_steps, advance_index=True)

        remaining = (len(self._active_steps) - 1) - self._step_index
        if remaining > 0:
            return self._cap_tts(f"{chunk} I have more steps. Say 'continue directions' to keep going.")
        return self._cap_tts(f"{chunk} That’s the final step. You're almost there.")

    def next_step(self) -> str:
        """
        Single-step mode (for when the user explicitly wants one step at a time).
        """
        if not self._active_destination:
            return "You're not navigating right now. You can say: give me directions to Walmart."

        if not self._active_steps:
            return f"You're navigating to {self._active_destination}, but I don't have turn-by-turn steps."

        if self._step_index >= len(self._active_steps) - 1:
            return "You're at the final step. You're almost there."

        self._step_index += 1
        step = self._active_steps[self._step_index]
        return self._cap_tts(f"Step {self._step_index + 1}: {self._format_step_spoken(step)}")

    def repeat_step(self) -> str:
        if not self._active_destination:
            return "You're not navigating right now."

        if not self._active_steps:
            return f"You're navigating to {self._active_destination}, but I don't have turn-by-turn steps."

        step = self._active_steps[self._step_index]
        return self._cap_tts(f"Step {self._step_index + 1}: {self._format_step_spoken(step)}")

    def status(self) -> str:
        if not self._active_destination:
            return "You're not navigating right now."
        if self._active_summary:
            return self._cap_tts(self._active_summary + " Say 'continue directions' for steps.")
        return f"You're navigating to {self._active_destination}."

    def stop_navigation(self) -> str:
        if not self._active_destination:
            return "Navigation is already stopped."

        dest = self._active_destination
        self._active_destination = None
        self._active_steps = []
        self._step_index = 0
        self._active_summary = None
        return f"Stopped navigation to {dest}."

    # ------------------------------------------------------------------
    # Internals: session + formatting
    # ------------------------------------------------------------------

    def _start_session(self, destination: str, steps: List[NavStep], summary: Optional[str]) -> None:
        self._active_destination = destination
        self._active_steps = steps
        self._step_index = 0
        self._active_summary = summary

        if getattr(config, "DEBUG", False):
            print(f"🧭 [NavigationClient] Session started: dest={destination!r}, steps={len(steps)}")

    def _format_route_summary(self, distance_m: float, duration_s: float, dest: str) -> str:
        if distance_m <= 0 and duration_s <= 0:
            return f"Okay. Navigating to {dest}."

        if distance_m >= 1000:
            dist_str = f"{distance_m/1000:.1f} kilometers"
        else:
            dist_str = f"{int(distance_m)} meters"

        mins = int(round(duration_s / 60.0)) if duration_s > 0 else 0
        if mins >= 60:
            hours = mins // 60
            rem = mins % 60
            dur_str = f"{hours} hours {rem} minutes" if rem else f"{hours} hours"
        else:
            dur_str = f"{mins} minutes" if mins else "a short time"

        return f"Route to {dest}. About {dist_str}, around {dur_str}."

    def _format_step_spoken(self, step: NavStep) -> str:
        instr = (step.instruction or "").strip() or "Continue."
        instr = instr.replace("\n", " ").strip()
        if len(instr) > 160:
            instr = instr[:160].rstrip() + "..."

        if step.distance_m and step.distance_m > 0:
            if step.distance_m >= 1000:
                d = f"{step.distance_m/1000:.1f} kilometers"
            else:
                d = f"{int(step.distance_m)} meters"
            return f"{instr} for {d}."
        return f"{instr}."

    def _cap_tts(self, text: str) -> str:
        max_chars = int(getattr(config, "NAV_MAX_TTS_CHARS", 900))
        if max_chars > 0 and len(text) > max_chars:
            return text[:max_chars].rstrip() + "..."
        return text

    def _get_steps_chunk(self, start_index: int, count: int, *, advance_index: bool) -> str:
        """
        Build a chunk: "Step X: ... Step Y: ..."
        If advance_index=True, moves internal index to last spoken step.
        """
        if not self._active_steps:
            return ""

        start_index = max(0, int(start_index))
        count = max(1, int(count))
        end = min(len(self._active_steps), start_index + count)

        if start_index >= len(self._active_steps):
            return ""

        parts: List[str] = []
        for i in range(start_index, end):
            step = self._active_steps[i]
            parts.append(f"Step {i + 1}: {self._format_step_spoken(step)}")

        if advance_index:
            self._step_index = end - 1

        return " ".join(parts)

    # ------------------------------------------------------------------
    # Helpers: distance + simplification
    # ------------------------------------------------------------------

    def _haversine_m(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        from math import atan2, cos, radians, sin, sqrt

        R = 6371000.0
        p1 = radians(lat1)
        p2 = radians(lat2)
        dlat = radians(lat2 - lat1)
        dlon = radians(lon2 - lon1)
        a = sin(dlat / 2) ** 2 + cos(p1) * cos(p2) * sin(dlon / 2) ** 2
        c = 2 * atan2(sqrt(a), sqrt(1 - a))
        return R * c

    def _simplify_steps(self, steps: List[NavStep]) -> List[NavStep]:
        if not steps:
            return []

        min_m = float(getattr(config, "NAV_MIN_STEP_DISTANCE_M", 18.0))
        max_store = int(getattr(config, "NAV_MAX_STORED_STEPS", 200))

        merged: List[NavStep] = []

        for s in steps:
            instr = (s.instruction or "").strip()
            dist = float(s.distance_m or 0.0)
            dur = float(s.duration_s or 0.0)

            if not instr:
                continue

            if not merged:
                merged.append(NavStep(instruction=instr, distance_m=dist, duration_s=dur))
                continue

            prev = merged[-1]
            if dist < min_m or instr.lower() == (prev.instruction or "").strip().lower():
                prev.distance_m += dist
                prev.duration_s += dur
                continue

            merged.append(NavStep(instruction=instr, distance_m=dist, duration_s=dur))

            if len(merged) >= max_store:
                break

        return merged

    # ------------------------------------------------------------------
    # ORS calls
    # ------------------------------------------------------------------

    def _geocode_destination(self, text: str) -> Optional[Tuple[float, float, str, float]]:
        url = f"{self.base_url}/geocode/search"

        have_origin = self.current_lat is not None and self.current_lon is not None
        size = int(getattr(config, "NAV_GEO_SIZE", 8))
        radius_km = float(getattr(config, "NAV_GEO_RADIUS_KM", 25.0))
        country = (getattr(config, "NAV_GEO_COUNTRY", "US") or "").strip()

        params: Dict[str, Any] = {
            "api_key": self.api_key,
            "text": text,
            "size": max(1, min(50, size)),
        }

        if have_origin:
            params.update(
                {
                    "focus.point.lat": float(self.current_lat),
                    "focus.point.lon": float(self.current_lon),
                    "boundary.circle.lat": float(self.current_lat),
                    "boundary.circle.lon": float(self.current_lon),
                    "boundary.circle.radius": max(1.0, radius_km),
                }
            )

        if country:
            params["boundary.country"] = country

        try:
            r = requests.get(url, params=params, timeout=self.timeout_s)
            r.raise_for_status()
            data = r.json()

            feats = data.get("features") or []
            if not feats:
                return None

            candidates: List[Tuple[float, float, str, float]] = []

            for feat in feats:
                geom = feat.get("geometry") or {}
                coords = geom.get("coordinates")
                if not coords or len(coords) != 2:
                    continue

                lon = float(coords[0])
                lat = float(coords[1])

                props = feat.get("properties") or {}
                label = props.get("label") or text

                dist_m = 0.0
                if have_origin:
                    dist_m = self._haversine_m(float(self.current_lat), float(self.current_lon), lat, lon)

                candidates.append((lat, lon, label, dist_m))

            if not candidates:
                return None

            if have_origin:
                candidates.sort(key=lambda x: x[3])

                if getattr(config, "DEBUG", False):
                    print("🧭 [NavigationClient] Geocode candidates (closest first):")
                    for i, c in enumerate(candidates[:5], start=1):
                        print(f"   {i}) {c[2]}  dist≈{c[3]/1000:.2f} km")

                return candidates[0]

            return candidates[0]

        except Exception as e:
            print(f"❌ [NavigationClient] Geocoding error: {e!r}")
            return None

    def _get_route(self, start: Tuple[float, float], end: Tuple[float, float]) -> Optional[Dict[str, Any]]:
        url = f"{self.base_url}/v2/directions/{self.profile}"
        headers = {
            "Authorization": self.api_key,
            "Content-Type": "application/json",
        }
        body = {
            "coordinates": [
                [float(start[1]), float(start[0])],
                [float(end[1]), float(end[0])],
            ],
            "instructions": True,
        }

        try:
            r = requests.post(url, headers=headers, json=body, timeout=self.timeout_s)
            r.raise_for_status()
            payload = r.json()

            if getattr(config, "DEBUG", False):
                print(f"🧭 [NavigationClient] ORS payload keys: {list(payload.keys())[:12]}")

            return payload
        except Exception as e:
            print(f"❌ [NavigationClient] Route error: {e!r}")
            return None

    def _extract_steps_and_summary(self, payload: Dict[str, Any]) -> Tuple[List[NavStep], float, float]:
        steps: List[NavStep] = []
        total_distance_m = 0.0
        total_duration_s = 0.0

        # GeoJSON features shape
        try:
            features = payload.get("features") or []
            if features:
                props = (features[0].get("properties") or {})
                segments = props.get("segments") or []
                if segments:
                    seg0 = segments[0]
                    total_distance_m = float(seg0.get("distance", 0.0) or 0.0)
                    total_duration_s = float(seg0.get("duration", 0.0) or 0.0)

                    raw_steps = seg0.get("steps") or []
                    for s in raw_steps:
                        instr = (s.get("instruction") or "").strip()
                        dist = float(s.get("distance", 0.0) or 0.0)
                        dur = float(s.get("duration", 0.0) or 0.0)
                        if instr:
                            steps.append(NavStep(instruction=instr, distance_m=dist, duration_s=dur))
                    return steps, total_distance_m, total_duration_s
        except Exception:
            pass

        # routes[] shape
        try:
            routes = payload.get("routes") or []
            if routes:
                r0 = routes[0]
                segments = r0.get("segments") or []
                if segments:
                    seg0 = segments[0]
                    total_distance_m = float(seg0.get("distance", 0.0) or 0.0)
                    total_duration_s = float(seg0.get("duration", 0.0) or 0.0)

                    raw_steps = seg0.get("steps") or []
                    for s in raw_steps:
                        instr = (s.get("instruction") or "").strip()
                        dist = float(s.get("distance", 0.0) or 0.0)
                        dur = float(s.get("duration", 0.0) or 0.0)
                        if instr:
                            steps.append(NavStep(instruction=instr, distance_m=dist, duration_s=dur))
                    return steps, total_distance_m, total_duration_s
        except Exception:
            pass

        return [], 0.0, 0.0
