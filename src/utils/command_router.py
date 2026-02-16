"""
Command router
Created by Team Visionary (Nathan + Ethan + Mohammed + Eric)

This module converts recognized speech into a structured command so
the controller can decide what pipeline to run (YOLO, OCR, currency, etc.)
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import re

import src.utils.config as config


@dataclass
class Command:
    intent: str
    target: Optional[str] = None
    raw: str = ""


def _clean_text(text: str) -> str:
    if not text:
        return ""
    t = text.strip().lower()
    # remove wake phrase if it shows up in the transcript
    t = t.replace(config.WAKE_PHRASE.lower(), "").strip()
    # normalize punctuation
    t = re.sub(r"[^a-z0-9\s']", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def _extract_target_after(text: str, phrase: str) -> str:
    """
    Extracts target after a trigger phrase.
    Example: "where is my water bottle" after "where is" -> "my water bottle"
    """
    idx = text.find(phrase)
    if idx == -1:
        return ""
    tail = text[idx + len(phrase):].strip()

    # common filler words
    for prefix in ["my ", "the ", "a ", "an "]:
        if tail.startswith(prefix):
            tail = tail[len(prefix):].strip()

    return tail.strip()


def _canonical_target(target_phrase: str) -> Optional[str]:
    """
    Convert the user's requested target phrase into a canonical label
    we can match against normalized YOLO labels.
    """
    if not target_phrase:
        return None

    phrase = target_phrase.strip().lower()

    # Try direct match
    if phrase in config.OBJECT_SYNONYMS:
        return config.OBJECT_SYNONYMS[phrase]

    # Try “best effort” match (e.g., user says "water bottle please")
    # We'll check known keys inside the phrase.
    for k, v in config.OBJECT_SYNONYMS.items():
        if k in phrase:
            return v

    return None


def parse_command(text: str) -> Command:
    """
    Parse recognized speech into intent + optional target.
    Start small and expand later.
    """
    cleaned = _clean_text(text)
    if not cleaned:
        return Command(intent="unknown", raw=text)

    # HELP
    if cleaned in ["help", "commands", "what can i say"]:
        return Command(intent="help", raw=text)

    # DESCRIBE SCENE
    if ("what do you see" in cleaned) or ("what is in front of me" in cleaned) or ("what's in front of me" in cleaned):
        return Command(intent="describe", raw=text)

    # READ FULL FRAME
    if cleaned == "read this" or "what does this say" in cleaned or cleaned.startswith("read this"):
        return Command(intent="read_frame", raw=text)

    # CURRENCY
    if "how much money" in cleaned or "money am i holding" in cleaned:
        return Command(intent="currency", raw=text)

    # COUNT/FILTER MODE: “are there any ____”
    if "are there any" in cleaned:
        tgt = _extract_target_after(cleaned, "are there any")
        return Command(intent="count_object", target=_canonical_target(tgt), raw=text)

    # LOCATE: “where is ____”
    if "where is" in cleaned:
        tgt = _extract_target_after(cleaned, "where is")
        return Command(intent="locate_object", target=_canonical_target(tgt), raw=text)

    # READ TARGET OBJECT: “read the ____”
    if cleaned.startswith("read "):
        tgt = _extract_target_after(cleaned, "read")
        return Command(intent="read_object", target=_canonical_target(tgt), raw=text)

    return Command(intent="unknown", raw=text)
