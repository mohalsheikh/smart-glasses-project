# src/weather_client.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict, Any, Tuple
import json
import time
from datetime import datetime

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

import src.utils.config as config


@dataclass
class _WeatherCache:
    key: str
    summary: str
    ts: float  # epoch seconds


class WeatherClient:
    """
    Resilient Open-Meteo weather client with GREAT UX.

    ✅ Uses LIVE GPS from config.GPS_LOCATION_FILE when fresh
    ✅ If GPS is stale, uses LAST-KNOWN GPS for a grace window (instead of snapping to Riverside)
    ✅ If user asks "weather in London", overrides GPS and uses that place
    ✅ Retries with backoff + caching for speed and reliability
    """

    # Weather caching (fast repeated asks)
    WEATHER_CACHE_TTL_S = 60 * 12         # 12 minutes
    FALLBACK_CACHE_TTL_S = 60 * 90        # 90 minutes (use cached summary if network fails)

    # Geocode caching
    GEOCODE_TTL_S = 60 * 60 * 24 * 7      # 7 days

    # If GPS is older than this, we still *may* use it as "last known location"
    # (prevents falling back to Riverside when your GPS server isn't updating every ~45s)
    GPS_WEATHER_GRACE_MAX_AGE_S = float(
        getattr(config, "GPS_STALE_SECONDS", 45)
    ) + (60 * 60 * 6)  # GPS_STALE_SECONDS + 6 hours grace

    # Network timeouts (connect, read)
    CONNECT_TIMEOUT_S = 3.0
    READ_TIMEOUT_S = 12.0

    def __init__(self, location: str = "Riverside,CA,US", *, units: Optional[str] = None, use_live_gps: bool = True):
        self.location = location
        self.use_live_gps = use_live_gps

        self.units = (units or "fahrenheit").lower().strip()
        if self.units not in ("fahrenheit", "celsius"):
            self.units = "fahrenheit"

        self._weather_cache: Optional[_WeatherCache] = None
        self._geocode_cache: Dict[str, Tuple[float, float, float]] = {}  # key -> (lat, lon, ts)

        self.session = self._build_session()

        print(f"🌦 WeatherClient initialized for location '{self.location}' (live_gps={self.use_live_gps})")

    # ---------------------------------------------------------------------
    # HTTP + retries
    # ---------------------------------------------------------------------

    def _build_session(self) -> requests.Session:
        session = requests.Session()
        retry = Retry(
            total=3,
            connect=3,
            read=3,
            backoff_factor=0.6,
            status_forcelist=(429, 500, 502, 503, 504),
            allowed_methods=("GET",),
            raise_on_status=False,
        )
        adapter = HTTPAdapter(max_retries=retry, pool_connections=10, pool_maxsize=10)
        session.mount("https://", adapter)
        session.mount("http://", adapter)
        return session

    def _timeout(self) -> Tuple[float, float]:
        return (self.CONNECT_TIMEOUT_S, self.READ_TIMEOUT_S)

    # ---------------------------------------------------------------------
    # Parsing helpers
    # ---------------------------------------------------------------------

    def _extract_city(self) -> str:
        parts = [p.strip() for p in self.location.split(",") if p.strip()]
        return parts[0] if parts else self.location

    def _minutes_ago_phrase(self, seconds: float) -> str:
        mins = max(1, int(seconds / 60))
        if mins < 60:
            return f"{mins} minute{'s' if mins != 1 else ''} ago"
        hrs = max(1, int(mins / 60))
        return f"{hrs} hour{'s' if hrs != 1 else ''} ago"

    def _parse_epoch_ts(self, ts_val: Any) -> Optional[float]:
        """
        Accepts:
          - seconds epoch (float/int)
          - milliseconds epoch (big int)
          - ISO string like "2025-12-22T03:15:22" (or with 'Z')
        """
        if ts_val is None:
            return None

        # numeric
        try:
            if isinstance(ts_val, (int, float)):
                ts = float(ts_val)
                # milliseconds?
                if ts > 1e12:
                    ts = ts / 1000.0
                return ts
        except Exception:
            pass

        # string
        if isinstance(ts_val, str):
            s = ts_val.strip()
            # numeric string?
            try:
                ts = float(s)
                if ts > 1e12:
                    ts = ts / 1000.0
                return ts
            except Exception:
                pass

            # ISO-ish
            try:
                s2 = s.replace("Z", "+00:00")
                dt = datetime.fromisoformat(s2)
                return dt.timestamp()
            except Exception:
                return None

        return None

    def _extract_location_override(self, query: Optional[str]) -> Optional[str]:
        """
        Pulls a location from questions like:
          - "weather in London"
          - "what's the weather like in New York right now"
          - "weather for Paris"
        If query is just "weather" / "weather here", returns None.
        """
        if not query:
            return None

        q = query.strip().lower()
        if not q:
            return None

        # If they explicitly mean "here", don't override.
        if any(p in q for p in ["weather here", "around here", "where i am", "right here", "my location"]):
            return None

        triggers = [" weather in ", "weather in ", " weather for ", "weather for ", " weather at ", "weather at ", "weather near ", " weather near "]
        for trig in triggers:
            if trig in q:
                idx = q.find(trig)
                loc = query[idx + len(trig):].strip(" .?!")
                # trim common trailing words
                for tail in [" right now", " now", " today", " currently", " please"]:
                    if loc.lower().endswith(tail):
                        loc = loc[: -len(tail)].strip()
                return loc or None

        # Also handle: "What is the weather in London now?"
        if " in " in q and "weather" in q:
            # crude but useful: take after last " in "
            loc = query.lower().rsplit(" in ", 1)[-1].strip(" .?!")
            for tail in [" right now", " now", " today", " currently", " please"]:
                if loc.endswith(tail):
                    loc = loc[: -len(tail)].strip()
            # re-use original casing approximately
            return loc or None

        return None

    # ---------------------------------------------------------------------
    # Geocoding + GPS
    # ---------------------------------------------------------------------

    def _geocode_place(self, place: str) -> Tuple[float, float]:
        now = time.time()
        key = place.strip().lower()

        cached = self._geocode_cache.get(key)
        if cached:
            lat, lon, ts = cached
            if (now - ts) <= self.GEOCODE_TTL_S:
                return float(lat), float(lon)

        geo_url = "https://geocoding-api.open-meteo.com/v1/search"
        geo_params = {"name": place, "count": 1, "language": "en", "format": "json"}

        resp = self.session.get(geo_url, params=geo_params, timeout=self._timeout())
        if resp.status_code != 200:
            raise RuntimeError(f"Geocoding HTTP {resp.status_code}")

        data: Dict[str, Any] = resp.json()
        results = data.get("results") or []
        if not results:
            raise RuntimeError("No geocoding results")

        r0 = results[0]
        lat = float(r0["latitude"])
        lon = float(r0["longitude"])

        self._geocode_cache[key] = (lat, lon, now)
        return lat, lon

    def _read_gps_file(self) -> Optional[Tuple[float, float, float]]:
        """
        Returns (lat, lon, age_seconds) if GPS file exists and is parseable.
        We do NOT reject stale here; caller decides how to use it.
        """
        if not self.use_live_gps:
            return None

        path = getattr(config, "GPS_LOCATION_FILE", None) or getattr(config, "LOCATION_JSON_PATH", None)
        if not path:
            return None

        try:
            with open(path, "r", encoding="utf-8") as f:
                raw = f.read().strip()
            if not raw:
                return None

            data = json.loads(raw)
            if not isinstance(data, dict):
                return None

            lat = data.get("lat", data.get("latitude"))
            lon = data.get("lon", data.get("longitude"))

            ts_val = (
                data.get("ts")
                or data.get("timestamp")
                or data.get("time")
                or data.get("updated_at")
                or data.get("updatedAt")
            )

            if lat is None or lon is None or ts_val is None:
                return None

            lat = float(lat)
            lon = float(lon)

            ts = self._parse_epoch_ts(ts_val)
            if ts is None:
                return None

            age = time.time() - ts
            if age < 0:
                age = 0.0

            return lat, lon, age

        except Exception:
            return None

    # ---------------------------------------------------------------------
    # Weather summary formatting
    # ---------------------------------------------------------------------

    def _weather_code_to_description(self, code: Any) -> str:
        mapping = {
            0: "clear",
            1: "mostly clear",
            2: "partly cloudy",
            3: "overcast",
            45: "fog",
            48: "freezing fog",
            51: "light drizzle",
            53: "drizzle",
            55: "heavy drizzle",
            61: "light rain",
            63: "rain",
            65: "heavy rain",
            71: "light snow",
            73: "snow",
            75: "heavy snow",
            80: "rain showers",
            81: "heavy showers",
            82: "violent showers",
            95: "thunderstorms",
            96: "thunderstorms with hail",
            99: "severe thunderstorms with hail",
        }
        try:
            return mapping.get(int(code), "mixed conditions")
        except Exception:
            return "mixed conditions"

    def _format_temp(self, c: Any) -> Optional[str]:
        if c is None:
            return None
        try:
            c = float(c)
        except Exception:
            return None

        if self.units == "fahrenheit":
            f = (c * 9.0 / 5.0) + 32.0
            return f"{round(f)}°F"
        return f"{round(c)}°C"

    def _format_wind(self, kmh: Any) -> Optional[str]:
        if kmh is None:
            return None
        try:
            kmh = float(kmh)
        except Exception:
            return None
        return f"{round(kmh)} km/h"

    def _build_spoken_summary(self, label: str, current: Dict[str, Any], *, note: Optional[str] = None) -> str:
        desc = self._weather_code_to_description(current.get("weathercode"))
        temp = self._format_temp(current.get("temperature"))
        wind = self._format_wind(current.get("windspeed"))

        hint: Optional[str] = None
        try:
            code = int(current.get("weathercode")) if current.get("weathercode") is not None else None
        except Exception:
            code = None

        if code in (95, 96, 99):
            hint = "Heads up—there may be thunderstorms."
        elif code in (61, 63, 65, 80, 81, 82, 51, 53, 55):
            hint = "You might want a jacket or umbrella."

        parts = [f"{label}: it’s {desc}."]
        if temp:
            parts.append(f"About {temp}.")
        if wind:
            parts.append(f"Winds around {wind}.")
        if hint:
            parts.append(hint)
        if note:
            parts.append(note)

        return " ".join(parts)

    # ---------------------------------------------------------------------
    # Public API
    # ---------------------------------------------------------------------

    def get_weather_summary(self, query: Optional[str] = None) -> str:
        """
        If query contains a location ("weather in London"), uses that place.
        Otherwise uses GPS location (fresh preferred; stale allowed within grace window).
        Otherwise falls back to default city.
        """
        now = time.time()

        # A) Explicit place override from query
        override = self._extract_location_override(query)
        if override:
            place = override
            key = f"place:{place.lower()}"
            label = f"In {place}"
            coords = None
            note = None
        else:
            place = None
            key = ""
            label = ""
            coords = None
            note = None

        # B) Otherwise try GPS (fresh or last-known)
        gps = None if override else self._read_gps_file()
        if gps:
            lat, lon, age = gps
            coords = (lat, lon)
            key = f"gps:{lat:.4f},{lon:.4f}"
            if age <= float(getattr(config, "GPS_STALE_SECONDS", 45)):
                label = "Right where you are"
                note = None
            elif age <= self.GPS_WEATHER_GRACE_MAX_AGE_S:
                label = "Near your last known location"
                note = f"(location from {self._minutes_ago_phrase(age)})"
            else:
                coords = None  # too old to trust
                gps = None

        # C) Final fallback city
        if coords is None and not override:
            city = self._extract_city()
            key = f"city:{city.lower()}"
            label = f"In {city}"
            note = None

        # Cache-first
        if self._weather_cache and self._weather_cache.key == key:
            age_s = now - self._weather_cache.ts
            if age_s <= self.WEATHER_CACHE_TTL_S:
                return self._weather_cache.summary

        try:
            # Resolve coords
            if coords is None:
                # override place or fallback city
                if override:
                    lat, lon = self._geocode_place(override)
                else:
                    lat, lon = self._geocode_place(self._extract_city())
            else:
                lat, lon = coords

            # Fetch current weather
            weather_url = "https://api.open-meteo.com/v1/forecast"
            params = {"latitude": lat, "longitude": lon, "current_weather": True}

            resp = self.session.get(weather_url, params=params, timeout=self._timeout())
            if resp.status_code != 200:
                raise RuntimeError(f"Weather HTTP {resp.status_code}")

            data: Dict[str, Any] = resp.json()
            current = data.get("current_weather") or {}

            summary = self._build_spoken_summary(label, current, note=note)

            self._weather_cache = _WeatherCache(key=key, summary=summary, ts=now)
            return summary

        except Exception as e:
            print(f"❌ WeatherClient error: {e!r}")

            # Use cached if we have it for this key
            if self._weather_cache and self._weather_cache.key == key:
                age_s = now - self._weather_cache.ts
                if age_s <= self.FALLBACK_CACHE_TTL_S:
                    return f"I can’t reach live weather right now. As of {self._minutes_ago_phrase(age_s)}: {self._weather_cache.summary}"

            # Friendly fallback
            if override:
                return f"I’m having trouble getting live weather for {override} right now. Try again in a moment."
            if gps:
                return "I’m having trouble getting live weather right now. Try again in a moment."
            return f"I’m having trouble getting live weather for {self._extract_city()} right now. Try again in a moment."
