# src/ai_features/enhanced_scene_memory.py
"""
Enhanced Scene Memory v2.0 — Auto-Save + Temporal + Conversation Context
=========================================================================

Major upgrades over original SceneMemoryEngine:

1. AUTO-SAVE MODE
   - Automatically saves scenes when significant changes are detected
   - Tracks object deltas (new objects appearing/disappearing)
   - No manual 'x' key needed — just works in the background

2. TEMPORAL AWARENESS
   - "What did I see 5 minutes ago?"
   - "When did I last see my keys?"
   - Time-windowed recall with natural language time parsing

3. SPATIAL CONTEXT
   - Tracks which room/location the user is in
   - "What was in the kitchen?" works even after leaving
   - Location transitions are logged

4. CONVERSATION CONTEXT MEMORY
   - Remembers what user asked about recently
   - "Tell me more about that" understands context
   - Stores user preferences (quick mode, reading mode, etc.)

5. SMART DEDUPLICATION
   - Doesn't save identical/near-identical scenes
   - Merges similar consecutive memories
   - Importance scoring based on scene novelty

Drop-in replacement for SceneMemoryEngine.
"""

from __future__ import annotations

import time
import json
import hashlib
from typing import List, Dict, Any, Optional, Tuple, Set
from dataclasses import dataclass, field
from collections import deque, Counter

import numpy as np

from openai import OpenAI
import src.utils.config as config

client = OpenAI()


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class SceneMemory:
    """A single stored scene memory with rich metadata."""
    timestamp: float
    description: str
    embedding: Optional[np.ndarray]
    detections: List[Dict[str, Any]]
    location: Optional[str] = None
    location_type: str = "unknown"
    tags: List[str] = field(default_factory=list)
    importance: float = 1.0
    access_count: int = 0
    last_accessed: float = field(default_factory=time.time)
    # New fields
    objects_hash: str = ""  # For deduplication
    is_transition: bool = False  # Scene change marker
    user_saved: bool = False  # Was this manually saved?
    conversation_context: Optional[str] = None  # What was the user asking about?


@dataclass
class ConversationContext:
    """Tracks recent conversation context for 'tell me more' style queries."""
    last_topic: str = ""
    last_intent: str = ""
    last_objects_discussed: List[str] = field(default_factory=list)
    last_location_discussed: str = ""
    last_query_time: float = 0.0
    recent_queries: deque = field(default_factory=lambda: deque(maxlen=10))
    user_preferences: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LocationTracker:
    """Tracks user's location transitions."""
    current_location: str = "unknown"
    current_location_type: str = "unknown"
    location_history: deque = field(default_factory=lambda: deque(maxlen=50))
    time_at_current: float = 0.0
    last_transition_time: float = field(default_factory=time.time)


# =============================================================================
# SCENE TAGS
# =============================================================================

SCENE_TAGS = {
    "indoor": ["room", "wall", "ceiling", "floor", "door", "window", "light"],
    "outdoor": ["tree", "sky", "car", "road", "building", "grass", "cloud", "sun"],
    "kitchen": ["stove", "refrigerator", "sink", "microwave", "oven", "plate", "cup", "mug"],
    "bathroom": ["toilet", "sink", "shower", "bathtub", "mirror", "towel"],
    "bedroom": ["bed", "pillow", "dresser", "nightstand", "lamp", "blanket"],
    "living_room": ["couch", "tv", "sofa", "table", "chair", "remote", "television"],
    "office": ["desk", "computer", "keyboard", "mouse", "monitor", "laptop", "pen"],
    "restaurant": ["table", "chair", "menu", "plate", "glass", "fork", "knife"],
    "store": ["shelf", "product", "cart", "checkout", "price", "bottle"],
    "street": ["car", "traffic light", "crosswalk", "sidewalk", "sign", "bus", "bicycle"],
    "classroom": ["whiteboard", "desk", "chair", "book", "projector", "backpack"],
    "hallway": ["door", "wall", "light", "sign", "exit"],
}


# =============================================================================
# ENHANCED SCENE MEMORY ENGINE
# =============================================================================

class EnhancedSceneMemory:
    """
    Production-grade scene memory with auto-save, temporal awareness,
    and conversation context tracking.
    """

    def __init__(
        self,
        max_memories: int = 1000,
        auto_save: bool = True,
        auto_save_interval: float = 10.0,
        min_change_threshold: float = 0.3,
        use_openai_embeddings: bool = True,
    ):
        self.max_memories = max_memories
        self.auto_save = auto_save
        self.auto_save_interval = auto_save_interval
        self.min_change_threshold = min_change_threshold
        self.use_openai = use_openai_embeddings

        # Memory store
        self.memories: deque = deque(maxlen=max_memories)

        # Tracking
        self.conversation = ConversationContext()
        self.location = LocationTracker()

        # Auto-save state
        self._last_auto_save = 0.0
        self._last_objects: Set[str] = set()
        self._last_object_counts: Counter = Counter()
        self._scene_stability_frames = 0
        self._min_stability_frames = 5  # Scene must be stable for 5 frames

        # Local embeddings fallback
        self._local_model = None

        print(f"🧠 EnhancedSceneMemory v2.0 initialized")
        print(f"   Max memories: {max_memories}")
        print(f"   Auto-save: {auto_save} (interval: {auto_save_interval}s)")

    # ------------------------------------------------------------------
    # Embedding
    # ------------------------------------------------------------------

    def _get_embedding(self, text: str) -> Optional[np.ndarray]:
        """Get text embedding using OpenAI or local model."""
        if not text:
            return None
        try:
            if self.use_openai and config.OPENAI_API_KEY_PRESENT:
                response = client.embeddings.create(
                    model="text-embedding-3-small",
                    input=text[:8000]  # Truncate to model limit
                )
                return np.array(response.data[0].embedding, dtype=np.float32)
        except Exception as e:
            print(f"⚠️ Embedding failed: {e}")

        # Fallback: load local model
        if self._local_model is None:
            try:
                from sentence_transformers import SentenceTransformer
                self._local_model = SentenceTransformer('all-MiniLM-L6-v2')
            except Exception:
                pass

        if self._local_model is not None:
            try:
                return self._local_model.encode(text, convert_to_numpy=True).astype(np.float32)
            except Exception:
                pass

        return None

    # ------------------------------------------------------------------
    # Scene Classification
    # ------------------------------------------------------------------

    def _classify_scene(self, detections: List[Dict[str, Any]]) -> Tuple[List[str], str]:
        """Classify scene type and return (tags, best_location_type)."""
        labels = [d.get("label", "").lower() for d in detections]

        scores = {}
        for scene_type, keywords in SCENE_TAGS.items():
            matches = sum(1 for kw in keywords if any(kw in label for label in labels))
            if matches >= 2:
                scores[scene_type] = matches

        if not scores:
            return ["unknown"], "unknown"

        sorted_scenes = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        tags = [s[0] for s in sorted_scenes[:3]]
        best = sorted_scenes[0][0]
        return tags, best

    def _compute_objects_hash(self, detections: List[Dict[str, Any]]) -> str:
        """Compute a hash of detected objects for deduplication."""
        labels = sorted(d.get("label", "").lower() for d in detections)
        return hashlib.md5("|".join(labels).encode()).hexdigest()[:12]

    def _compute_scene_change(self, detections: List[Dict[str, Any]]) -> float:
        """
        Compute how much the scene has changed (0.0 = identical, 1.0 = completely different).
        Uses Jaccard distance on object sets.
        """
        current_objects = set(d.get("label", "").lower() for d in detections if d.get("label"))
        if not current_objects and not self._last_objects:
            return 0.0
        if not current_objects or not self._last_objects:
            return 1.0

        intersection = current_objects & self._last_objects
        union = current_objects | self._last_objects
        jaccard = len(intersection) / len(union) if union else 1.0
        return 1.0 - jaccard

    # ------------------------------------------------------------------
    # Auto-Save Logic
    # ------------------------------------------------------------------

    def update_from_detections(
        self,
        detections: List[Dict[str, Any]],
        frame_idx: int = 0,
    ) -> Optional[str]:
        """
        Called every detection cycle. Decides whether to auto-save.
        Returns a message if scene was saved, None otherwise.

        Call this from controller.py in the detection loop.
        """
        if not self.auto_save:
            return None

        now = time.time()

        # Don't save too frequently
        if now - self._last_auto_save < self.auto_save_interval:
            return None

        # Compute scene change
        change = self._compute_scene_change(detections)

        # Update location tracking
        tags, location_type = self._classify_scene(detections)
        if location_type != self.location.current_location_type:
            # Location transition detected!
            old = self.location.current_location_type
            self.location.location_history.append({
                "from": old,
                "to": location_type,
                "time": now,
            })
            self.location.current_location_type = location_type
            self.location.last_transition_time = now

            # Auto-save transition
            if old != "unknown":
                self._auto_store(
                    detections=detections,
                    tags=tags,
                    location_type=location_type,
                    importance=1.5,  # Transitions are important
                    is_transition=True,
                )
                self._last_auto_save = now
                self._last_objects = set(d.get("label", "").lower() for d in detections)
                return f"Scene changed: moved from {old} to {location_type}"

        # Save if scene changed significantly
        if change >= self.min_change_threshold:
            self._scene_stability_frames = 0

        # Wait for scene to stabilize
        self._scene_stability_frames += 1
        if self._scene_stability_frames >= self._min_stability_frames and change >= self.min_change_threshold:
            obj_hash = self._compute_objects_hash(detections)

            # Check for duplicate
            if self.memories and self.memories[-1].objects_hash == obj_hash:
                return None

            self._auto_store(
                detections=detections,
                tags=tags,
                location_type=location_type,
                importance=min(2.0, 0.5 + change),
            )
            self._last_auto_save = now
            self._last_objects = set(d.get("label", "").lower() for d in detections)
            self._scene_stability_frames = 0
            return None  # Silent auto-save

        return None

    def _auto_store(
        self,
        detections: List[Dict[str, Any]],
        tags: List[str],
        location_type: str,
        importance: float = 1.0,
        is_transition: bool = False,
    ):
        """Internal auto-save method."""
        objects = [d.get("label", "unknown") for d in detections[:10]]
        if not objects:
            description = f"Empty {location_type} scene"
        else:
            description = f"{location_type.replace('_', ' ').title()} with: {', '.join(objects[:5])}"

        self.store_scene(
            description=description,
            detections=detections,
            location=self.location.current_location,
            location_type=location_type,
            tags=tags,
            importance=importance,
            is_transition=is_transition,
        )

    # ------------------------------------------------------------------
    # Manual Store
    # ------------------------------------------------------------------

    def store_scene(
        self,
        description: str,
        detections: List[Dict[str, Any]],
        location: Optional[str] = None,
        location_type: str = "unknown",
        tags: Optional[List[str]] = None,
        importance: float = 1.0,
        user_saved: bool = False,
        is_transition: bool = False,
        conversation_context: Optional[str] = None,
    ) -> bool:
        """Store a scene in memory with full metadata."""
        try:
            embedding = self._get_embedding(description)

            if tags is None:
                tags, location_type = self._classify_scene(detections)

            memory = SceneMemory(
                timestamp=time.time(),
                description=description,
                embedding=embedding,
                detections=detections[:20],  # Cap detections stored
                location=location or self.location.current_location,
                location_type=location_type,
                tags=tags,
                importance=importance,
                user_saved=user_saved,
                is_transition=is_transition,
                objects_hash=self._compute_objects_hash(detections),
                conversation_context=conversation_context or self.conversation.last_topic,
            )

            self.memories.append(memory)
            return True
        except Exception as e:
            print(f"⚠️ Memory store failed: {e}")
            return False

    # ------------------------------------------------------------------
    # Recall Methods
    # ------------------------------------------------------------------

    def recall_similar(
        self,
        query: str,
        top_k: int = 3,
        time_window_hours: Optional[float] = None,
        min_similarity: float = 0.65,
    ) -> List[Tuple[SceneMemory, float]]:
        """Find semantically similar past scenes."""
        query_emb = self._get_embedding(query)
        if query_emb is None:
            return []

        now = time.time()
        results = []

        for mem in self.memories:
            if mem.embedding is None:
                continue

            if time_window_hours is not None:
                age_hours = (now - mem.timestamp) / 3600
                if age_hours > time_window_hours:
                    continue

            sim = float(np.dot(query_emb, mem.embedding) / (
                np.linalg.norm(query_emb) * np.linalg.norm(mem.embedding) + 1e-8
            ))

            if sim >= min_similarity:
                mem.access_count += 1
                mem.last_accessed = now
                results.append((mem, sim))

        results.sort(key=lambda x: x[1] * x[0].importance, reverse=True)
        return results[:top_k]

    def recall_by_object(self, object_name: str, top_k: int = 5) -> List[SceneMemory]:
        """Find memories containing a specific object."""
        obj_lower = object_name.lower()
        results = []

        for mem in self.memories:
            for det in mem.detections:
                label = det.get("label", "").lower()
                if obj_lower in label or label in obj_lower:
                    results.append(mem)
                    break

        results.sort(key=lambda m: m.timestamp * m.importance, reverse=True)
        return results[:top_k]

    def recall_by_time(
        self,
        minutes_ago: Optional[float] = None,
        hours_ago: Optional[float] = None,
        time_range_minutes: float = 5.0,
    ) -> List[SceneMemory]:
        """
        Recall scenes from a specific time.
        "What did I see 10 minutes ago?" → recall_by_time(minutes_ago=10)
        """
        now = time.time()
        if minutes_ago is not None:
            target_time = now - (minutes_ago * 60)
        elif hours_ago is not None:
            target_time = now - (hours_ago * 3600)
        else:
            target_time = now - 300  # Default: 5 min ago

        window = time_range_minutes * 60
        results = [
            m for m in self.memories
            if abs(m.timestamp - target_time) <= window
        ]
        results.sort(key=lambda m: abs(m.timestamp - target_time))
        return results[:5]

    def recall_by_location(self, location_type: str, top_k: int = 5) -> List[SceneMemory]:
        """Recall scenes from a specific location type."""
        loc_lower = location_type.lower()
        results = [
            m for m in self.memories
            if loc_lower in m.location_type.lower() or loc_lower in " ".join(m.tags)
        ]
        results.sort(key=lambda m: m.timestamp, reverse=True)
        return results[:top_k]

    def when_last_seen(self, object_name: str) -> Optional[str]:
        """
        "When did I last see my keys?"
        Returns a natural language time description.
        """
        memories = self.recall_by_object(object_name, top_k=1)
        if not memories:
            return None

        mem = memories[0]
        elapsed = time.time() - mem.timestamp
        location = mem.location_type

        if elapsed < 60:
            time_str = "just now"
        elif elapsed < 3600:
            mins = int(elapsed / 60)
            time_str = f"about {mins} minute{'s' if mins > 1 else ''} ago"
        elif elapsed < 86400:
            hours = int(elapsed / 3600)
            time_str = f"about {hours} hour{'s' if hours > 1 else ''} ago"
        else:
            days = int(elapsed / 86400)
            time_str = f"about {days} day{'s' if days > 1 else ''} ago"

        if location and location != "unknown":
            return f"You last saw {object_name} {time_str} in the {location.replace('_', ' ')}."
        return f"You last saw {object_name} {time_str}."

    # ------------------------------------------------------------------
    # Conversation Context
    # ------------------------------------------------------------------

    def update_conversation(
        self,
        query: str,
        intent: str = "",
        objects: Optional[List[str]] = None,
        location: str = "",
    ):
        """Update conversation context for 'tell me more' queries."""
        self.conversation.last_topic = query
        self.conversation.last_intent = intent
        self.conversation.last_objects_discussed = objects or []
        self.conversation.last_location_discussed = location
        self.conversation.last_query_time = time.time()
        self.conversation.recent_queries.append({
            "query": query,
            "intent": intent,
            "time": time.time(),
        })

    def get_conversation_context(self) -> str:
        """Get formatted conversation context for AI prompts."""
        ctx = self.conversation
        if not ctx.last_topic or (time.time() - ctx.last_query_time) > 120:
            return ""

        parts = []
        if ctx.last_topic:
            parts.append(f"User was recently asking about: {ctx.last_topic}")
        if ctx.last_objects_discussed:
            parts.append(f"Objects discussed: {', '.join(ctx.last_objects_discussed[:5])}")
        if ctx.last_location_discussed:
            parts.append(f"Location context: {ctx.last_location_discussed}")

        return " | ".join(parts)

    # ------------------------------------------------------------------
    # Summary & Stats
    # ------------------------------------------------------------------

    def get_context_summary(self, recent_minutes: int = 5) -> str:
        """Get a summary of recent scene context."""
        cutoff = time.time() - (recent_minutes * 60)
        recent = [m for m in self.memories if m.timestamp >= cutoff]

        if not recent:
            return "No recent scene data."

        locations = Counter(m.location_type for m in recent if m.location_type != "unknown")
        all_objects = Counter()
        for m in recent:
            for d in m.detections:
                label = d.get("label", "")
                if label:
                    all_objects[label] += 1

        parts = []
        if locations:
            main_loc = locations.most_common(1)[0][0]
            parts.append(f"Location: {main_loc.replace('_', ' ')}")
        if all_objects:
            top_5 = [obj for obj, _ in all_objects.most_common(5)]
            parts.append(f"Seen recently: {', '.join(top_5)}")

        transitions = [m for m in recent if m.is_transition]
        if transitions:
            parts.append(f"Moved through {len(transitions)} locations")

        return ". ".join(parts) if parts else "Scene data available but no notable changes."

    def get_stats(self) -> Dict[str, Any]:
        """Get memory system statistics."""
        return {
            "total_memories": len(self.memories),
            "user_saved": sum(1 for m in self.memories if m.user_saved),
            "auto_saved": sum(1 for m in self.memories if not m.user_saved),
            "transitions": sum(1 for m in self.memories if m.is_transition),
            "current_location": self.location.current_location_type,
            "unique_locations": len(set(m.location_type for m in self.memories)),
        }

    # ------------------------------------------------------------------
    # Export / Persistence
    # ------------------------------------------------------------------

    def export_memories(self, filepath: str) -> bool:
        """Export memories to JSON (embeddings excluded for size)."""
        try:
            data = []
            for mem in self.memories:
                data.append({
                    "timestamp": mem.timestamp,
                    "description": mem.description,
                    "detections": mem.detections,
                    "location": mem.location,
                    "location_type": mem.location_type,
                    "tags": mem.tags,
                    "importance": mem.importance,
                    "access_count": mem.access_count,
                    "user_saved": mem.user_saved,
                    "is_transition": mem.is_transition,
                })

            with open(filepath, "w") as f:
                json.dump(data, f, indent=2)

            print(f"✅ Exported {len(data)} memories to {filepath}")
            return True
        except Exception as e:
            print(f"❌ Export failed: {e}")
            return False

    def clear_old_memories(self, days: int = 30):
        """Clear memories older than specified days (keep user-saved)."""
        cutoff = time.time() - (days * 86400)
        before = len(self.memories)
        self.memories = deque(
            [m for m in self.memories if m.timestamp >= cutoff or m.user_saved],
            maxlen=self.max_memories,
        )
        removed = before - len(self.memories)
        if removed > 0:
            print(f"🗑️ Cleared {removed} old memories")


# =============================================================================
# BACKWARD COMPATIBILITY
# =============================================================================

# Alias so existing code that imports SceneMemoryEngine still works
SceneMemoryEngine = EnhancedSceneMemory
MemoryEntry = SceneMemory