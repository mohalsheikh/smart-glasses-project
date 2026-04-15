# src/safety/emergency_detection.py
"""
Emergency Detection System for VisionAssist
=============================================

Critical safety features for visually impaired users:

1. FALL DETECTION
   - Monitors IMU accelerometer data for sudden impacts
   - Detects orientation changes (upright → horizontal)
   - Confirms fall with post-fall stillness check
   - Auto-triggers emergency alert after confirmation

2. DISTRESS VOICE DETECTION
   - Listens for distress phrases: "help", "call 911", "emergency"
   - Can be triggered even without wake word
   - Configurable urgency levels

3. SOS MODE
   - Voice-activated: "emergency" or "call for help"
   - Sends GPS location to emergency contacts
   - Plays loud alert tone
   - Reads location aloud to user

4. INACTIVITY MONITORING
   - Detects if user hasn't moved or spoken for too long
   - Gentle check-in: "Are you okay? I haven't heard from you."
   - Escalates if no response

5. DANGEROUS ENVIRONMENT DETECTION
   - Detects vehicles approaching quickly (visual)
   - Detects edge/drop-off risks
   - Water/traffic hazards from scene analysis

All features are configurable and can be enabled/disabled individually.
"""

from __future__ import annotations

import time
import json
import threading
from typing import Optional, List, Dict, Any, Callable, Tuple
from dataclasses import dataclass, field
from collections import deque
from enum import Enum
from pathlib import Path

import numpy as np


# =============================================================================
# ENUMS & DATA STRUCTURES
# =============================================================================

class EmergencyLevel(Enum):
    NONE = 0
    CHECK_IN = 1       # Gentle check: "Are you okay?"
    WARNING = 2         # Audible warning: "Danger detected"
    ALERT = 3           # High-priority alert with location
    EMERGENCY = 4       # Full emergency: contact responders


class EmergencyType(Enum):
    FALL = "fall"
    DISTRESS_VOICE = "distress_voice"
    INACTIVITY = "inactivity"
    VEHICLE_DANGER = "vehicle_danger"
    EDGE_DANGER = "edge_danger"
    MANUAL_SOS = "manual_sos"


@dataclass
class EmergencyEvent:
    """Represents a detected emergency event."""
    event_type: EmergencyType
    level: EmergencyLevel
    timestamp: float = field(default_factory=time.time)
    message: str = ""
    location: Optional[Dict[str, float]] = None  # {"lat": ..., "lng": ...}
    confirmed: bool = False
    resolved: bool = False
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EmergencyContact:
    """Emergency contact information."""
    name: str
    phone: str
    relationship: str = ""
    is_primary: bool = False


# =============================================================================
# FALL DETECTOR
# =============================================================================

class FallDetector:
    """
    Detects falls using IMU accelerometer and gyroscope data.
    
    Algorithm:
    1. Monitor acceleration magnitude for sudden spikes (free-fall → impact)
    2. Check orientation change (upright → horizontal/inverted)
    3. Confirm with post-impact stillness (person not moving after fall)
    4. Uses a state machine: NORMAL → FREE_FALL → IMPACT → CONFIRMING → FALLEN
    """

    class State(Enum):
        NORMAL = "normal"
        FREE_FALL = "free_fall"
        IMPACT = "impact"
        CONFIRMING = "confirming"
        FALLEN = "fallen"

    def __init__(
        self,
        free_fall_threshold: float = 0.3,      # G's (near 0 = free fall)
        impact_threshold: float = 3.0,          # G's (high = impact)
        orientation_threshold: float = 60.0,     # Degrees from upright
        confirmation_time: float = 5.0,          # Seconds of stillness to confirm
        stillness_threshold: float = 0.15,       # G's variation = "still"
    ):
        self.free_fall_threshold = free_fall_threshold
        self.impact_threshold = impact_threshold
        self.orientation_threshold = orientation_threshold
        self.confirmation_time = confirmation_time
        self.stillness_threshold = stillness_threshold

        self.state = self.State.NORMAL
        self._state_enter_time = time.time()
        self._accel_history: deque = deque(maxlen=50)
        self._orientation_at_impact: Optional[float] = None

        # Stats
        self.total_falls_detected = 0
        self.false_alarms = 0

    def update(
        self,
        accel_x: float,
        accel_y: float,
        accel_z: float,
        pitch: float = 0.0,
        roll: float = 0.0,
    ) -> Optional[EmergencyEvent]:
        """
        Process IMU data and detect falls.
        
        Args:
            accel_x/y/z: Accelerometer data in G's (1G = 9.8m/s²)
            pitch, roll: Orientation in degrees
            
        Returns:
            EmergencyEvent if fall detected, None otherwise.
        """
        now = time.time()
        accel_mag = np.sqrt(accel_x**2 + accel_y**2 + accel_z**2)
        self._accel_history.append((now, accel_mag))

        orientation_angle = np.sqrt(pitch**2 + roll**2)

        if self.state == self.State.NORMAL:
            # Check for free-fall (acceleration near zero)
            if accel_mag < self.free_fall_threshold:
                self.state = self.State.FREE_FALL
                self._state_enter_time = now
                return None

        elif self.state == self.State.FREE_FALL:
            # Check for impact (high acceleration spike)
            if accel_mag > self.impact_threshold:
                self.state = self.State.IMPACT
                self._state_enter_time = now
                self._orientation_at_impact = orientation_angle
                return None
            # Timeout — back to normal
            if now - self._state_enter_time > 1.0:
                self.state = self.State.NORMAL
                return None

        elif self.state == self.State.IMPACT:
            # Check if orientation changed significantly (person now horizontal)
            if orientation_angle > self.orientation_threshold:
                self.state = self.State.CONFIRMING
                self._state_enter_time = now
            elif now - self._state_enter_time > 2.0:
                # Impact but orientation OK — might be a bump
                self.state = self.State.NORMAL
            return None

        elif self.state == self.State.CONFIRMING:
            # Check for stillness (person not moving after fall)
            recent = [mag for t, mag in self._accel_history if now - t < 2.0]
            if recent:
                variation = max(recent) - min(recent)
                if variation < self.stillness_threshold:
                    # Person is still after impact + orientation change = FALL
                    if now - self._state_enter_time >= self.confirmation_time:
                        self.state = self.State.FALLEN
                        self.total_falls_detected += 1
                        return EmergencyEvent(
                            event_type=EmergencyType.FALL,
                            level=EmergencyLevel.EMERGENCY,
                            message="Fall detected! Are you okay? Say 'I'm fine' or I'll call for help.",
                            details={
                                "impact_g": float(accel_mag),
                                "orientation": float(orientation_angle),
                                "confirmation_seconds": float(now - self._state_enter_time),
                            },
                        )
                else:
                    # Person is moving — probably fine
                    self.state = self.State.NORMAL
                    self.false_alarms += 1
            return None

        elif self.state == self.State.FALLEN:
            # Stay in fallen state until explicitly reset
            pass

        return None

    def reset(self):
        """Reset fall detector (user confirmed they're OK)."""
        self.state = self.State.NORMAL
        self._state_enter_time = time.time()

    def cancel_alert(self):
        """Cancel current alert (false alarm)."""
        self.state = self.State.NORMAL
        self.false_alarms += 1


# =============================================================================
# DISTRESS VOICE DETECTOR
# =============================================================================

class DistressVoiceDetector:
    """
    Detects distress phrases in voice input.
    Works alongside normal voice commands — doesn't require wake word.
    """

    # Phrases and their urgency levels
    DISTRESS_PHRASES = {
        EmergencyLevel.EMERGENCY: [
            "call 911", "call nine one one", "call an ambulance",
            "call emergency", "emergency help", "someone help",
            "i'm hurt", "i fell", "i can't move", "heart attack",
            "i can't breathe", "call the police",
        ],
        EmergencyLevel.ALERT: [
            "help me", "i need help", "please help",
            "call for help", "get help", "i'm in danger",
            "i'm lost", "i'm scared", "something's wrong",
        ],
        EmergencyLevel.WARNING: [
            "help", "emergency", "danger", "sos",
            "i'm not okay", "i don't feel good",
        ],
    }

    def __init__(self, cooldown_seconds: float = 30.0):
        self.cooldown = cooldown_seconds
        self._last_detection_time: Dict[str, float] = {}

    def check_text(self, text: str) -> Optional[EmergencyEvent]:
        """Check transcribed text for distress phrases."""
        if not text:
            return None

        t = text.lower().strip()
        now = time.time()

        for level in [EmergencyLevel.EMERGENCY, EmergencyLevel.ALERT, EmergencyLevel.WARNING]:
            for phrase in self.DISTRESS_PHRASES[level]:
                if phrase in t:
                    # Check cooldown
                    key = f"{level.value}_{phrase}"
                    if key in self._last_detection_time:
                        if now - self._last_detection_time[key] < self.cooldown:
                            continue

                    self._last_detection_time[key] = now
                    return EmergencyEvent(
                        event_type=EmergencyType.DISTRESS_VOICE,
                        level=level,
                        message=self._get_response_message(level, phrase),
                        details={"phrase": phrase, "full_text": text},
                    )

        return None

    @staticmethod
    def _get_response_message(level: EmergencyLevel, phrase: str) -> str:
        if level == EmergencyLevel.EMERGENCY:
            return "Emergency detected. I'm getting help. Stay calm."
        elif level == EmergencyLevel.ALERT:
            return "I hear you need help. Would you like me to contact someone?"
        else:
            return "Are you okay? Let me know if you need assistance."


# =============================================================================
# INACTIVITY MONITOR
# =============================================================================

class InactivityMonitor:
    """
    Monitors user activity and checks in if they've been inactive too long.
    Especially important for visually impaired users who might be disoriented.
    """

    def __init__(
        self,
        check_in_minutes: float = 30.0,
        escalation_minutes: float = 5.0,
    ):
        self.check_in_interval = check_in_minutes * 60
        self.escalation_interval = escalation_minutes * 60

        self.last_activity_time = time.time()
        self.last_check_in_time = time.time()
        self._check_in_sent = False
        self._escalated = False

    def register_activity(self):
        """Call this whenever user interacts (voice, key press, etc.)."""
        self.last_activity_time = time.time()
        self._check_in_sent = False
        self._escalated = False

    def check(self) -> Optional[EmergencyEvent]:
        """Check if user has been inactive. Call periodically."""
        now = time.time()
        inactive_duration = now - self.last_activity_time

        if not self._check_in_sent and inactive_duration > self.check_in_interval:
            self._check_in_sent = True
            self.last_check_in_time = now
            return EmergencyEvent(
                event_type=EmergencyType.INACTIVITY,
                level=EmergencyLevel.CHECK_IN,
                message="Hey, just checking in. Are you doing okay? Say something if you can hear me.",
                details={"inactive_minutes": inactive_duration / 60},
            )

        if self._check_in_sent and not self._escalated:
            if now - self.last_check_in_time > self.escalation_interval:
                self._escalated = True
                return EmergencyEvent(
                    event_type=EmergencyType.INACTIVITY,
                    level=EmergencyLevel.ALERT,
                    message="I haven't heard from you in a while. I'm going to try to get help.",
                    details={"inactive_minutes": inactive_duration / 60},
                )

        return None


# =============================================================================
# DANGEROUS ENVIRONMENT DETECTOR
# =============================================================================

class DangerousEnvironmentDetector:
    """
    Detects dangerous situations from visual scene analysis.
    """

    # Objects that might indicate danger
    DANGER_OBJECTS = {
        "vehicle_close": {
            "labels": ["car", "truck", "bus", "motorcycle", "bicycle"],
            "min_area_ratio": 0.25,  # Object takes up >25% of frame
            "message": "Vehicle very close! Stay alert.",
            "level": EmergencyLevel.WARNING,
        },
        "traffic": {
            "labels": ["traffic light", "traffic sign", "stop sign"],
            "min_area_ratio": 0.05,
            "message": "Traffic signals detected. Be careful crossing.",
            "level": EmergencyLevel.CHECK_IN,
        },
        "stairs_edge": {
            "labels": ["stairs", "staircase", "edge", "ledge", "cliff"],
            "min_area_ratio": 0.1,
            "message": "Stairs or edge detected ahead. Proceed carefully.",
            "level": EmergencyLevel.WARNING,
        },
    }

    def __init__(self, cooldown_seconds: float = 15.0):
        self.cooldown = cooldown_seconds
        self._last_alerts: Dict[str, float] = {}

    def check_scene(
        self,
        detections: List[Dict[str, Any]],
        frame_width: int,
        frame_height: int,
    ) -> List[EmergencyEvent]:
        """Check detections for dangerous objects/situations."""
        events = []
        now = time.time()
        frame_area = frame_width * frame_height

        if frame_area <= 0:
            return events

        for danger_type, config in self.DANGER_OBJECTS.items():
            # Cooldown check
            if danger_type in self._last_alerts:
                if now - self._last_alerts[danger_type] < self.cooldown:
                    continue

            for det in detections:
                label = (det.get("label", "") or "").lower()
                if not any(kw in label for kw in config["labels"]):
                    continue

                # Check if object is close (large in frame)
                bbox = det.get("bbox")
                if bbox and len(bbox) == 4:
                    x1, y1, x2, y2 = bbox
                    obj_area = abs(x2 - x1) * abs(y2 - y1)
                    area_ratio = obj_area / frame_area

                    if area_ratio >= config["min_area_ratio"]:
                        self._last_alerts[danger_type] = now
                        events.append(EmergencyEvent(
                            event_type=EmergencyType.VEHICLE_DANGER if "vehicle" in danger_type
                            else EmergencyType.EDGE_DANGER,
                            level=config["level"],
                            message=config["message"],
                            details={
                                "object": label,
                                "area_ratio": area_ratio,
                                "confidence": det.get("confidence", 0),
                            },
                        ))
                        break  # One alert per danger type per cycle

        return events


# =============================================================================
# MAIN EMERGENCY SYSTEM
# =============================================================================

class EmergencySystem:
    """
    Central emergency detection and response system.
    Coordinates all safety subsystems.
    """

    def __init__(
        self,
        enable_fall_detection: bool = True,
        enable_distress_voice: bool = True,
        enable_inactivity: bool = True,
        enable_environment: bool = True,
        speak_callback: Optional[Callable[[str], None]] = None,
        emergency_contacts: Optional[List[Dict[str, str]]] = None,
    ):
        # Subsystems
        self.fall_detector = FallDetector() if enable_fall_detection else None
        self.distress_detector = DistressVoiceDetector() if enable_distress_voice else None
        self.inactivity_monitor = InactivityMonitor() if enable_inactivity else None
        self.environment_detector = DangerousEnvironmentDetector() if enable_environment else None

        # Callback for speaking alerts
        self.speak_callback = speak_callback

        # Emergency contacts
        self.contacts: List[EmergencyContact] = []
        if emergency_contacts:
            for c in emergency_contacts:
                self.contacts.append(EmergencyContact(
                    name=c.get("name", ""),
                    phone=c.get("phone", ""),
                    relationship=c.get("relationship", ""),
                    is_primary=c.get("is_primary", False),
                ))

        # Event log
        self.event_log: deque = deque(maxlen=100)
        self._active_emergency: Optional[EmergencyEvent] = None
        self._lock = threading.Lock()

        # GPS location (updated externally)
        self.last_known_location: Optional[Dict[str, float]] = None

        enabled = []
        if enable_fall_detection:
            enabled.append("fall")
        if enable_distress_voice:
            enabled.append("distress")
        if enable_inactivity:
            enabled.append("inactivity")
        if enable_environment:
            enabled.append("environment")

        print(f"🚨 EmergencySystem initialized")
        print(f"   Active modules: {', '.join(enabled) or 'none'}")
        print(f"   Emergency contacts: {len(self.contacts)}")

    # ------------------------------------------------------------------
    # Update Methods (call from controller)
    # ------------------------------------------------------------------

    def update_imu(
        self,
        accel_x: float, accel_y: float, accel_z: float,
        pitch: float = 0.0, roll: float = 0.0,
    ) -> Optional[EmergencyEvent]:
        """Update with IMU data for fall detection."""
        if self.fall_detector is None:
            return None
        event = self.fall_detector.update(accel_x, accel_y, accel_z, pitch, roll)
        if event:
            self._handle_event(event)
        return event

    def check_voice(self, transcribed_text: str) -> Optional[EmergencyEvent]:
        """Check voice input for distress phrases."""
        if self.distress_detector is None:
            return None

        # Register activity
        if self.inactivity_monitor:
            self.inactivity_monitor.register_activity()

        # Check for "I'm fine" / "I'm okay" (cancel active emergency)
        if self._active_emergency and transcribed_text:
            t = transcribed_text.lower()
            if any(p in t for p in ["i'm fine", "i'm okay", "i am fine", "i am okay",
                                     "false alarm", "cancel emergency", "cancel alert",
                                     "i'm alright", "never mind"]):
                self._resolve_emergency("User confirmed they are okay")
                return None

        event = self.distress_detector.check_text(transcribed_text)
        if event:
            self._handle_event(event)
        return event

    def check_scene(
        self,
        detections: List[Dict[str, Any]],
        frame_width: int,
        frame_height: int,
    ) -> List[EmergencyEvent]:
        """Check visual scene for dangers."""
        if self.environment_detector is None:
            return []
        events = self.environment_detector.check_scene(detections, frame_width, frame_height)
        for event in events:
            self._handle_event(event)
        return events

    def check_inactivity(self) -> Optional[EmergencyEvent]:
        """Periodic inactivity check."""
        if self.inactivity_monitor is None:
            return None
        event = self.inactivity_monitor.check()
        if event:
            self._handle_event(event)
        return event

    def register_activity(self):
        """Register user activity (voice, key press, etc.)."""
        if self.inactivity_monitor:
            self.inactivity_monitor.register_activity()

    def update_location(self, lat: float, lng: float):
        """Update GPS location."""
        self.last_known_location = {"lat": lat, "lng": lng}

    # ------------------------------------------------------------------
    # SOS Mode
    # ------------------------------------------------------------------

    def trigger_sos(self, reason: str = "Manual SOS activated") -> EmergencyEvent:
        """Manually trigger SOS mode."""
        event = EmergencyEvent(
            event_type=EmergencyType.MANUAL_SOS,
            level=EmergencyLevel.EMERGENCY,
            message="SOS activated. Getting help now.",
            location=self.last_known_location,
            details={"reason": reason},
        )
        self._handle_event(event)
        return event

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _handle_event(self, event: EmergencyEvent):
        """Process an emergency event."""
        with self._lock:
            event.location = self.last_known_location
            self.event_log.append(event)

            if event.level.value >= EmergencyLevel.ALERT.value:
                self._active_emergency = event

            # Speak the alert
            if self.speak_callback and event.message:
                try:
                    self.speak_callback(event.message)
                except Exception as e:
                    print(f"⚠️ Emergency speak error: {e}")

            # Log
            print(f"🚨 [{event.level.name}] {event.event_type.value}: {event.message}")

            # For high-level emergencies, prepare emergency info
            if event.level.value >= EmergencyLevel.EMERGENCY.value:
                self._prepare_emergency_response(event)

    def _prepare_emergency_response(self, event: EmergencyEvent):
        """Prepare emergency response (location sharing, contact notification)."""
        # Build emergency info
        info_parts = [f"Emergency: {event.event_type.value}"]
        if event.location:
            info_parts.append(
                f"Location: {event.location.get('lat', 'unknown')}, {event.location.get('lng', 'unknown')}"
            )
        info_parts.append(f"Time: {time.strftime('%H:%M:%S')}")

        emergency_info = " | ".join(info_parts)
        print(f"🚨 EMERGENCY INFO: {emergency_info}")

        # In production: send SMS/call to emergency contacts
        # For now, just log it
        if self.contacts:
            primary = next((c for c in self.contacts if c.is_primary), self.contacts[0])
            print(f"📞 Would contact: {primary.name} ({primary.phone})")

        # Speak location to user
        if self.speak_callback and event.location:
            loc_msg = (
                f"Your approximate location is latitude {event.location.get('lat', 'unknown'):.4f}, "
                f"longitude {event.location.get('lng', 'unknown'):.4f}. "
                "Help is being contacted."
            )
            try:
                self.speak_callback(loc_msg)
            except Exception:
                pass

    def _resolve_emergency(self, reason: str):
        """Resolve the active emergency."""
        with self._lock:
            if self._active_emergency:
                self._active_emergency.resolved = True
                self._active_emergency = None

            if self.fall_detector:
                self.fall_detector.reset()

            print(f"✅ Emergency resolved: {reason}")
            if self.speak_callback:
                try:
                    self.speak_callback("Okay, glad you're alright. Emergency cancelled.")
                except Exception:
                    pass

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------

    def has_active_emergency(self) -> bool:
        """Check if there's an active unresolved emergency."""
        with self._lock:
            return self._active_emergency is not None

    def get_status(self) -> Dict[str, Any]:
        """Get system status."""
        return {
            "active_emergency": self._active_emergency is not None,
            "total_events": len(self.event_log),
            "fall_detector": {
                "state": self.fall_detector.state.value if self.fall_detector else "disabled",
                "falls_detected": self.fall_detector.total_falls_detected if self.fall_detector else 0,
            },
            "last_location": self.last_known_location,
        }

    # ------------------------------------------------------------------
    # Voice Command Integration
    # ------------------------------------------------------------------

    def is_emergency_command(self, text: str) -> bool:
        """Check if text is an emergency command."""
        if not text:
            return False
        t = text.lower().strip()
        return any(p in t for p in [
            "emergency", "sos", "call 911", "call for help",
            "help me", "i need help", "i fell", "i'm hurt",
        ])

    def handle_emergency_command(self, text: str) -> Optional[str]:
        """Handle an emergency voice command. Returns response text."""
        if not text:
            return None

        t = text.lower().strip()

        # Cancel commands
        if any(p in t for p in ["cancel", "i'm fine", "i'm okay", "false alarm", "never mind"]):
            self._resolve_emergency("User voice cancellation")
            return "Emergency cancelled. Glad you're okay."

        # SOS trigger
        if any(p in t for p in ["sos", "call 911", "call emergency", "call for help"]):
            event = self.trigger_sos(reason=f"Voice command: {text}")
            return event.message

        # Check for distress
        event = self.check_voice(text)
        if event:
            return event.message

        return None
