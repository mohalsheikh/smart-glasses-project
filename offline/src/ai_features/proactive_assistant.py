"""
Proactive AI Assistant - Contextually Aware Helper
Monitors scenes and proactively provides helpful information without being asked
"""

from __future__ import annotations

import time
from typing import List, Dict, Any, Optional, Set
from dataclasses import dataclass, field
import numpy as np

from openai import OpenAI
import src.utils.config as config

client = OpenAI()


@dataclass
class ProactiveAlert:
    """Alert to be spoken to the user"""
    message: str
    priority: str  # "low", "medium", "high", "critical"
    category: str  # "safety", "navigation", "information", "assistance"
    timestamp: float = field(default_factory=time.time)
    spoken: bool = False


class ProactiveAssistant:
    """
    AI assistant that proactively helps the user by:
    - Detecting potentially important situations
    - Providing contextual information
    - Suggesting actions
    - Warning about potential issues
    """
    
    def __init__(self, enable_proactive: bool = True):
        self.enable_proactive = enable_proactive
        self.alert_queue: List[ProactiveAlert] = []
        
        # Track what we've already alerted about (to avoid spam)
        self.recent_alerts: Set[str] = set()
        self.alert_history_ttl = 30  # seconds
        self.last_cleanup = time.time()
        
        # Context tracking
        self.scene_context = {
            "current_location_type": "unknown",  # indoor, outdoor, kitchen, etc.
            "time_in_location": 0,
            "last_scene_change": time.time(),
            "user_seems_stationary": False,
            "objects_nearby": [],
            "people_nearby": 0,
        }
        
        # Proactive triggers
        self.proactive_rules = {
            # Safety triggers
            "stairs_detected": {
                "cooldown": 10,
                "priority": "high",
                "category": "safety",
                "message": "Stairs detected ahead. Use caution.",
            },
            "obstacle_close": {
                "cooldown": 5,
                "priority": "high",
                "category": "safety",
                "message": "Obstacle very close. Please be careful.",
            },
            "vehicle_approaching": {
                "cooldown": 8,
                "priority": "high",
                "category": "safety",
                "message": "Vehicle approaching.",
            },
            "door_nearby": {
                "cooldown": 15,
                "priority": "medium",
                "category": "navigation",
                "message": "Door nearby",
            },
            
            # Helpful information
            "person_waiting": {
                "cooldown": 20,
                "priority": "medium",
                "category": "information",
                "message": "Someone appears to be waiting near you.",
            },
            "item_left_behind": {
                "cooldown": 30,
                "priority": "medium",
                "category": "assistance",
                "message": "You may have left something behind.",
            },
            "text_visible": {
                "cooldown": 20,
                "priority": "low",
                "category": "information",
                "message": "Text visible. Would you like me to read it?",
            },
        }
        
        self.last_trigger_times: Dict[str, float] = {}
        
        print("🤖 ProactiveAssistant initialized")
    
    def update_scene_context(
        self,
        detections: List[Dict[str, Any]],
        location_type: Optional[str] = None
    ):
        """Update scene context for proactive awareness"""
        if location_type:
            if location_type != self.scene_context["current_location_type"]:
                self.scene_context["last_scene_change"] = time.time()
                self.scene_context["time_in_location"] = 0
            else:
                self.scene_context["time_in_location"] = time.time() - self.scene_context["last_scene_change"]
            
            self.scene_context["current_location_type"] = location_type
        
        # Track nearby objects and people
        self.scene_context["objects_nearby"] = [d.get("label", "") for d in detections]
        self.scene_context["people_nearby"] = sum(
            1 for d in detections if d.get("label", "").lower() == "person"
        )
    
    def analyze_scene_for_proactive_alerts(
        self,
        detections: List[Dict[str, Any]],
        frame_size: tuple
    ) -> List[ProactiveAlert]:
        """Analyze scene and generate proactive alerts"""
        if not self.enable_proactive:
            return []
        
        alerts = []
        current_time = time.time()
        
        # Clean up old alerts from history
        if current_time - self.last_cleanup > self.alert_history_ttl:
            self.recent_alerts.clear()
            self.last_cleanup = current_time
        
        # Check for various situations
        labels = [d.get("label", "").lower() for d in detections]
        
        # Safety: Stairs
        if any("stair" in label for label in labels):
            self._maybe_trigger("stairs_detected", alerts, current_time)
        
        # Safety: Close obstacles
        close_objects = [
            d for d in detections
            if d.get("confidence", 0) > 0.5 and self._is_object_close(d, frame_size)
        ]
        if close_objects:
            # Check if any are potentially dangerous
            dangerous_keywords = ["car", "vehicle", "truck", "bike", "pole", "edge"]
            if any(any(kw in label for kw in dangerous_keywords) for label in labels):
                self._maybe_trigger("obstacle_close", alerts, current_time)
        
        # Navigation: Door nearby
        if any("door" in label for label in labels):
            self._maybe_trigger("door_nearby", alerts, current_time)
        
        # Information: People waiting
        people = [d for d in detections if d.get("label", "").lower() == "person"]
        if len(people) > 0:
            # Check if person seems stationary (simplified)
            for person in people:
                if self._person_seems_stationary(person):
                    self._maybe_trigger("person_waiting", alerts, current_time)
                    break
        
        return alerts
    
    def _is_object_close(self, detection: Dict[str, Any], frame_size: tuple) -> bool:
        """Check if object is close based on bbox size"""
        bbox = detection.get("bbox")
        if not bbox:
            return False
        
        x1, y1, x2, y2 = bbox
        bbox_area = (x2 - x1) * (y2 - y1)
        frame_area = frame_size[0] * frame_size[1]
        
        area_ratio = bbox_area / frame_area
        return area_ratio > 0.2  # Object takes up >20% of frame
    
    def _person_seems_stationary(self, person: Dict[str, Any]) -> bool:
        """Simple heuristic: if person has track_id and pose suggests standing"""
        # This is simplified - in production, you'd track movement over time
        return person.get("track_id") is not None
    
    def _maybe_trigger(
        self,
        trigger_name: str,
        alerts: List[ProactiveAlert],
        current_time: float
    ):
        """Trigger alert if cooldown has passed"""
        rule = self.proactive_rules.get(trigger_name)
        if not rule:
            return
        
        # Check cooldown
        last_trigger = self.last_trigger_times.get(trigger_name, 0)
        if current_time - last_trigger < rule["cooldown"]:
            return
        
        # Check if recently alerted
        alert_key = f"{trigger_name}_{int(current_time / 60)}"  # Group by minute
        if alert_key in self.recent_alerts:
            return
        
        # Create alert
        alert = ProactiveAlert(
            message=rule["message"],
            priority=rule["priority"],
            category=rule["category"],
        )
        
        alerts.append(alert)
        self.last_trigger_times[trigger_name] = current_time
        self.recent_alerts.add(alert_key)
    
    def get_contextual_suggestions(self, user_query: str) -> Optional[str]:
        """Provide contextual suggestions based on current scene"""
        suggestions = []
        
        # Based on location
        location = self.scene_context["current_location_type"]
        time_in_loc = self.scene_context["time_in_location"]
        
        if location == "kitchen" and time_in_loc > 120:  # 2 minutes
            suggestions.append("Would you like help identifying items in the kitchen?")
        
        if self.scene_context["people_nearby"] > 0:
            suggestions.append("There are people nearby if you need assistance.")
        
        # Based on objects
        objects = self.scene_context["objects_nearby"]
        if "phone" in objects or "laptop" in objects:
            suggestions.append("I can help you locate your device if needed.")
        
        return ". ".join(suggestions) if suggestions else None
    
    def generate_smart_response(
        self,
        user_query: str,
        scene_detections: List[Dict[str, Any]],
        conversation_history: List[Dict[str, str]]
    ) -> str:
        """Generate contextually aware response using GPT-4o"""
        try:
            # Build context
            context_parts = [
                f"Current location type: {self.scene_context['current_location_type']}",
                f"People nearby: {self.scene_context['people_nearby']}",
                f"Objects visible: {', '.join(self.scene_context['objects_nearby'][:10])}",
            ]
            context_str = "\n".join(context_parts)
            
            # Build messages
            messages = [
                {
                    "role": "system",
                    "content": (
                        "You are a proactive AI assistant for smart glasses worn by a blind person. "
                        "You have context about their environment and proactively help them. "
                        "Be concise, helpful, and anticipate their needs. "
                        "Use second person and natural language. "
                        f"\n\nCurrent context:\n{context_str}"
                    )
                }
            ]
            
            # Add conversation history (last 3 turns)
            for msg in conversation_history[-6:]:
                messages.append(msg)
            
            # Add current query
            messages.append({"role": "user", "content": user_query})
            
            response = client.chat.completions.create(
                model=config.OPENAI_CHAT_MODEL,
                messages=messages,
                max_tokens=200,
                temperature=0.7,
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            print(f"⚠️ Smart response generation error: {e}")
            return "I'm here to help. Could you clarify what you need?"
    
    def should_interrupt_for_safety(self, alert: ProactiveAlert) -> bool:
        """Determine if alert is urgent enough to interrupt user"""
        return alert.priority in ["high", "critical"] and alert.category == "safety"
    
    def get_next_alert(self) -> Optional[ProactiveAlert]:
        """Get next unspoken alert, prioritized"""
        unspoken = [a for a in self.alert_queue if not a.spoken]
        if not unspoken:
            return None
        
        # Sort by priority
        priority_order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
        unspoken.sort(key=lambda a: priority_order.get(a.priority, 999))
        
        alert = unspoken[0]
        alert.spoken = True
        return alert
    
    def clear_old_alerts(self, max_age_seconds: float = 60):
        """Remove old alerts from queue"""
        current_time = time.time()
        self.alert_queue = [
            a for a in self.alert_queue
            if current_time - a.timestamp < max_age_seconds
        ]
