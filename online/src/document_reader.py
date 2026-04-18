# src/document_reader.py
"""
Smart Document Reader v2.0
============================
Intelligent text reading with auto-summarization for long content.

Behavior:
  - SHORT text (< 150 words): reads everything directly
  - MEDIUM text (150-300 words): reads with brief context
  - LONG text (300+ words): auto-summarizes, then asks user if they
    want the full text read aloud

Uses GPT-4o-mini for summarization (fast + cheap).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple
import time

import src.utils.config as config
from src.ocr_engine import OCREngine

try:
    from src.brain.openai_client import client
except Exception:
    client = None


# ---------------------------------------------------------------------------
# Thresholds
# ---------------------------------------------------------------------------

SHORT_WORD_LIMIT = 150    # Just read everything
MEDIUM_WORD_LIMIT = 300   # Read with brief intro
# Above MEDIUM = summarize first, offer full read


@dataclass
class DocumentState:
    paragraphs: List[str]
    full_text: str
    idx: int
    ts: float
    engine: str
    conf: float
    mode: str
    word_count: int
    was_summarized: bool = False
    user_wants_full_read: bool = False


class DocumentReader:
    """
    Smart document reader:
      - start(frame, mode): OCR + intelligent reading strategy
      - next(): read next paragraph
      - repeat(): repeat current paragraph
      - summarize(): AI summary of full page
      - read_full(): read the complete text (after summary)
    """

    def __init__(self, ocr: OCREngine):
        self.ocr = ocr
        self.state: Optional[DocumentState] = None

    def _expired(self) -> bool:
        if not self.state:
            return True
        ttl = float(getattr(config, "DOC_CACHE_TTL_S", 120.0) or 120.0)
        return (time.time() - self.state.ts) > ttl

    def _normalize_mode(self, mode: Optional[str]) -> str:
        raw = (mode or "").strip().lower()
        if not raw:
            raw = (getattr(config, "READING_MODE_DEFAULT", "") or "").strip().lower()
        if not raw:
            raw = (getattr(config, "OCR_MODE", "hybrid") or "hybrid").strip().lower()

        if raw in {"offline", "local", "local_only", "localonly"}:
            return "local_only"
        if raw in {"ai", "scene", "openai", "scene_only", "sceneonly"}:
            return "scene_only"
        if raw in {"hybrid", "local_only", "scene_only"}:
            return raw
        return "hybrid"

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def start(self, frame_bgr, mode: Optional[str] = None) -> str:
        """
        Smart read:
          - Short text → read it all
          - Long text → summarize, offer full read
        """
        mode_final = self._normalize_mode(mode)

        paras, conf, engine = self.ocr.read_paragraphs(
            frame_bgr, mode=mode_final, force_ai=True
        )
        paras = [p.strip() for p in (paras or []) if p.strip()]

        if not paras:
            self.state = None
            return (
                "I can't read any clear text right now. "
                "Try holding it steadier, a bit closer, and make sure there's good lighting."
            )

        full_text = "\n\n".join(paras)
        word_count = len(full_text.split())

        self.state = DocumentState(
            paragraphs=paras,
            full_text=full_text,
            idx=0,
            ts=time.time(),
            engine=engine,
            conf=conf,
            mode=mode_final,
            word_count=word_count,
            was_summarized=False,
            user_wants_full_read=False,
        )

        # ---- Smart reading strategy ----

        if word_count <= SHORT_WORD_LIMIT:
            # Short text: just read it all naturally
            reading = self._humanize_all(paras)
            return f"I can see some text. It says: {reading}"

        if word_count <= MEDIUM_WORD_LIMIT:
            # Medium text: read with brief context
            reading = self._humanize_all(paras)
            return (
                f"I see a block of text, about {word_count} words. "
                f"Here's what it says: {reading}"
            )

        # Long text: summarize first
        summary = self._ai_summarize(full_text)
        self.state.was_summarized = True

        if summary:
            return (
                f"I see a longer document, about {word_count} words. "
                f"Here's a quick summary: {summary} "
                f"Would you like me to read the full text?"
            )
        else:
            # AI summarization failed, read first paragraph
            first = self._humanize(paras[0])
            remaining = len(paras) - 1
            return (
                f"I see a longer document with {len(paras)} sections. "
                f"Starting from the top: {first} "
                f"There {'is' if remaining == 1 else 'are'} {remaining} more "
                f"{'section' if remaining == 1 else 'sections'}. "
                f"Say 'next' to continue or 'read everything' for the full text."
            )

    # ------------------------------------------------------------------
    # Navigation
    # ------------------------------------------------------------------

    def next(self) -> str:
        if not self.state or self._expired():
            return "I'm not in reading mode right now. Say 'read this' while pointing at text."

        if self.state.idx + 1 >= len(self.state.paragraphs):
            return "That's the end of the text. Say 'repeat' to hear it again, or 'summarize' for an overview."

        self.state.idx += 1
        self.state.ts = time.time()

        current = self.state.paragraphs[self.state.idx]
        remaining = len(self.state.paragraphs) - self.state.idx - 1

        text = self._humanize(current)
        if remaining > 0:
            text += f" {remaining} more {'section' if remaining == 1 else 'sections'} remaining."

        return text

    def repeat(self) -> str:
        if not self.state or self._expired():
            return "I'm not in reading mode right now. Say 'read this' while pointing at text."
        return self._humanize(
            self.state.paragraphs[self.state.idx],
            prefix="Sure, here it is again."
        )

    def read_full(self) -> str:
        """Read the complete text — called when user says yes to full read."""
        if not self.state or self._expired():
            return "I'm not in reading mode right now. Say 'read this' while pointing at text."

        self.state.user_wants_full_read = True
        self.state.idx = 0
        reading = self._humanize_all(self.state.paragraphs)
        return f"Okay, reading everything. {reading}"

    def summarize(self) -> str:
        if not self.state or self._expired():
            return "I'm not in reading mode right now. Say 'read this' while pointing at text."

        full = self.state.full_text.strip()
        if not full:
            return "I don't have enough text to summarize."

        summary = self._ai_summarize(full)
        if summary:
            return summary
        return self._fallback_summary(full)

    # ------------------------------------------------------------------
    # AI Summarization
    # ------------------------------------------------------------------

    def _ai_summarize(self, text: str) -> Optional[str]:
        """Use GPT-4o-mini to create a natural, spoken-friendly summary."""
        if client is None:
            return None

        try:
            prompt = (
                "You are the voice of smart glasses for a visually impaired user. "
                "Summarize the following text in 2-3 clear, natural sentences. "
                "Make it sound like you're speaking to the user directly. "
                "Mention the type of content (e.g., article, notice, menu, label, letter). "
                "Focus on the most important information.\n\n"
                f"TEXT:\n{text[:6000]}"
            )

            resp = client.chat.completions.create(
                model=getattr(config, "OPENAI_CHAT_MODEL", "gpt-4o-mini"),
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a helpful voice assistant for visually impaired users. "
                            "You summarize text naturally and concisely. "
                            "Speak in second person. Be warm and clear."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                max_tokens=200,
                temperature=0.3,
            )

            out = (resp.choices[0].message.content or "").strip()
            out = " ".join(out.split())
            return out if out else None

        except Exception as e:
            print(f"⚠️ AI summarize failed: {e!r}")
            return None

    def _fallback_summary(self, text: str) -> str:
        """Simple first-two-sentences summary when AI is unavailable."""
        s = " ".join(text.split())
        parts = []
        buff = ""
        for ch in s:
            buff += ch
            if ch in ".!?":
                parts.append(buff.strip())
                buff = ""
                if len(parts) >= 2:
                    break
        if len(parts) < 2 and buff.strip():
            parts.append(buff.strip())
        if not parts:
            return "I don't have enough to summarize."
        return " ".join(parts[:2])

    # ------------------------------------------------------------------
    # Text formatting for speech
    # ------------------------------------------------------------------

    def _humanize(self, paragraph: str, prefix: str = "") -> str:
        p = " ".join((paragraph or "").split()).strip()
        if not p:
            return "I couldn't read that part clearly."
        # Keep it comfortable for audio
        if len(p) > 600:
            p = p[:600].rsplit(" ", 1)[0] + "..."
        if prefix:
            return f"{prefix} {p}"
        return p

    def _humanize_all(self, paragraphs: List[str]) -> str:
        """Join all paragraphs into a single spoken-friendly string."""
        parts = []
        for p in paragraphs:
            cleaned = " ".join((p or "").split()).strip()
            if cleaned:
                parts.append(cleaned)

        if not parts:
            return "I couldn't read the text clearly."

        combined = " ... ".join(parts)

        # Cap at ~800 chars for audio comfort
        if len(combined) > 800:
            combined = combined[:800].rsplit(" ", 1)[0] + "... and there's more after that."

        return combined

    # ------------------------------------------------------------------
    # Compat: set_mode / get_mode
    # ------------------------------------------------------------------

    def set_mode(self, mode: str):
        pass  # mode is per-call now

    def get_mode(self) -> str:
        if self.state:
            return self.state.mode
        return self._normalize_mode(None)
