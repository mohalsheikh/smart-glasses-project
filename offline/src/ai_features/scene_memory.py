# """
# Advanced Scene Memory System using OpenAI Embeddings
# Remembers scenes, objects, and contexts for intelligent assistance
# """

# from __future__ import annotations

# import time
# import numpy as np
# from typing import List, Dict, Any, Optional, Tuple
# from dataclasses import dataclass, field
# from collections import deque
# import json

# from openai import OpenAI
# from sentence_transformers import SentenceTransformer
# import src.utils.config as config

# client = OpenAI()


# @dataclass
# class MemoryEntry:
#     """Represents a stored memory with embeddings"""
#     timestamp: float
#     description: str
#     embedding: np.ndarray
#     detections: List[Dict[str, Any]]
#     location: Optional[str] = None
#     tags: List[str] = field(default_factory=list)
#     importance: float = 1.0
#     access_count: int = 0
#     last_accessed: float = field(default_factory=time.time)


# class SceneMemoryEngine:
#     """
#     Maintains semantic memory of scenes and objects using embeddings.
#     Enables "Have I seen this before?" "What was in this room?" queries.
#     """
    
#     def __init__(self, max_memories: int = 500, use_openai: bool = True):
#         self.max_memories = max_memories
#         self.use_openai = use_openai
#         self.memories: deque = deque(maxlen=max_memories)
        
#         # Use sentence transformer as fallback if OpenAI fails
#         self.local_model = None
#         if not use_openai:
#             try:
#                 self.local_model = SentenceTransformer('all-MiniLM-L6-v2')
#                 print("🧠 SceneMemory: Using local embeddings model")
#             except Exception as e:
#                 print(f"⚠️ SceneMemory: Failed to load local model: {e}")
        
#         # Scene tags tracking
#         self.scene_tags = {
#             "indoor": ["room", "wall", "ceiling", "floor", "door", "window"],
#             "outdoor": ["tree", "sky", "car", "road", "building", "grass"],
#             "kitchen": ["stove", "refrigerator", "sink", "microwave", "oven"],
#             "bathroom": ["toilet", "sink", "shower", "bathtub", "mirror"],
#             "bedroom": ["bed", "pillow", "dresser", "nightstand"],
#             "living_room": ["couch", "tv", "table", "chair"],
#             "office": ["desk", "computer", "keyboard", "mouse", "monitor"],
#             "restaurant": ["table", "chair", "menu", "plate", "glass"],
#             "store": ["shelf", "product", "cart", "checkout"],
#             "street": ["car", "traffic light", "crosswalk", "sidewalk", "sign"],
#         }
        
#         print("🧠 SceneMemoryEngine initialized")
    
#     def _get_embedding(self, text: str) -> Optional[np.ndarray]:
#         """Get embedding for text using OpenAI or local model"""
#         try:
#             if self.use_openai and config.OPENAI_API_KEY_PRESENT:
#                 response = client.embeddings.create(
#                     model="text-embedding-3-small",
#                     input=text
#                 )
#                 return np.array(response.data[0].embedding)
#             elif self.local_model is not None:
#                 return self.local_model.encode(text, convert_to_numpy=True)
#         except Exception as e:
#             print(f"⚠️ SceneMemory: Embedding failed: {e}")
#         return None
    
#     def _classify_scene(self, detections: List[Dict[str, Any]]) -> List[str]:
#         """Classify scene type based on detected objects"""
#         tags = []
#         detected_labels = [d.get("label", "").lower() for d in detections]
        
#         for scene_type, keywords in self.scene_tags.items():
#             matches = sum(1 for kw in keywords if any(kw in label for label in detected_labels))
#             if matches >= 2:
#                 tags.append(scene_type)
        
#         return tags if tags else ["unknown"]
    
#     def store_scene(
#         self,
#         description: str,
#         detections: List[Dict[str, Any]],
#         location: Optional[str] = None,
#         importance: float = 1.0
#     ) -> bool:
#         """Store a scene in memory"""
#         try:
#             embedding = self._get_embedding(description)
#             if embedding is None:
#                 return False
            
#             tags = self._classify_scene(detections)
            
#             memory = MemoryEntry(
#                 timestamp=time.time(),
#                 description=description,
#                 embedding=embedding,
#                 detections=detections,
#                 location=location,
#                 tags=tags,
#                 importance=importance
#             )
            
#             self.memories.append(memory)
#             return True
#         except Exception as e:
#             print(f"⚠️ SceneMemory: Failed to store: {e}")
#             return False
    
#     def recall_similar(
#         self,
#         query: str,
#         top_k: int = 3,
#         time_window_hours: Optional[float] = None,
#         min_similarity: float = 0.7
#     ) -> List[Tuple[MemoryEntry, float]]:
#         """Find similar past scenes"""
#         try:
#             query_embedding = self._get_embedding(query)
#             if query_embedding is None:
#                 return []
            
#             current_time = time.time()
#             results = []
            
#             for memory in self.memories:
#                 # Time filter
#                 if time_window_hours is not None:
#                     age_hours = (current_time - memory.timestamp) / 3600
#                     if age_hours > time_window_hours:
#                         continue
                
#                 # Calculate cosine similarity
#                 similarity = np.dot(query_embedding, memory.embedding) / (
#                     np.linalg.norm(query_embedding) * np.linalg.norm(memory.embedding)
#                 )
                
#                 if similarity >= min_similarity:
#                     memory.access_count += 1
#                     memory.last_accessed = current_time
#                     results.append((memory, float(similarity)))
            
#             # Sort by similarity * importance
#             results.sort(key=lambda x: x[1] * x[0].importance, reverse=True)
#             return results[:top_k]
            
#         except Exception as e:
#             print(f"⚠️ SceneMemory: Recall failed: {e}")
#             return []
    
#     def recall_by_object(self, object_name: str, top_k: int = 5) -> List[MemoryEntry]:
#         """Find memories containing a specific object"""
#         results = []
#         object_lower = object_name.lower()
        
#         for memory in self.memories:
#             # Check detections
#             for det in memory.detections:
#                 label = det.get("label", "").lower()
#                 if object_lower in label or label in object_lower:
#                     results.append(memory)
#                     break
        
#         # Sort by recency and importance
#         results.sort(key=lambda m: m.timestamp * m.importance, reverse=True)
#         return results[:top_k]
    
#     def recall_by_location(self, location: str, top_k: int = 5) -> List[MemoryEntry]:
#         """Find memories from a specific location"""
#         results = [m for m in self.memories if m.location and location.lower() in m.location.lower()]
#         results.sort(key=lambda m: m.timestamp, reverse=True)
#         return results[:top_k]
    
#     def recall_by_scene_type(self, scene_type: str, top_k: int = 5) -> List[MemoryEntry]:
#         """Find memories of a specific scene type (kitchen, outdoor, etc.)"""
#         scene_lower = scene_type.lower()
#         results = [m for m in self.memories if scene_lower in m.tags]
#         results.sort(key=lambda m: m.timestamp, reverse=True)
#         return results[:top_k]
    
#     def get_context_summary(self, recent_minutes: int = 5) -> str:
#         """Get summary of recent context"""
#         cutoff = time.time() - (recent_minutes * 60)
#         recent = [m for m in self.memories if m.timestamp >= cutoff]
        
#         if not recent:
#             return "No recent context available."
        
#         # Group by scene type
#         scene_counts = {}
#         objects_seen = set()
        
#         for memory in recent:
#             for tag in memory.tags:
#                 scene_counts[tag] = scene_counts.get(tag, 0) + 1
#             for det in memory.detections:
#                 objects_seen.add(det.get("label", ""))
        
#         summary_parts = []
#         if scene_counts:
#             main_scene = max(scene_counts.items(), key=lambda x: x[1])[0]
#             summary_parts.append(f"Recent scenes: {main_scene}")
        
#         if objects_seen:
#             top_objects = list(objects_seen)[:5]
#             summary_parts.append(f"Objects seen: {', '.join(top_objects)}")
        
#         return ". ".join(summary_parts)
    
#     def export_memories(self, filepath: str) -> bool:
#         """Export memories to JSON"""
#         try:
#             data = []
#             for memory in self.memories:
#                 data.append({
#                     "timestamp": memory.timestamp,
#                     "description": memory.description,
#                     "detections": memory.detections,
#                     "location": memory.location,
#                     "tags": memory.tags,
#                     "importance": memory.importance,
#                     "access_count": memory.access_count,
#                 })
            
#             with open(filepath, 'w') as f:
#                 json.dump(data, f, indent=2)
            
#             print(f"✅ Exported {len(data)} memories to {filepath}")
#             return True
#         except Exception as e:
#             print(f"❌ Memory export failed: {e}")
#             return False
    
#     def clear_old_memories(self, days: int = 30):
#         """Clear memories older than specified days"""
#         cutoff = time.time() - (days * 24 * 3600)
#         original_count = len(self.memories)
#         self.memories = deque(
#             [m for m in self.memories if m.timestamp >= cutoff],
#             maxlen=self.max_memories
#         )
#         removed = original_count - len(self.memories)
#         if removed > 0:
#             print(f"🗑️ Cleared {removed} old memories")
