# src/assistant_brain.py
from __future__ import annotations

# Backwards-compatible re-export:
# Any existing imports like `from src.assistant_brain import AssistantBrain`
# will still work without changing other files.
from src.brain.assistant_brain_impl import AssistantBrain

__all__ = ["AssistantBrain"]
