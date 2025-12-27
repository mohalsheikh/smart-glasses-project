# src/document_reader.py

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional
import time

import src.utils.config as config
from src.ocr_engine import OCREngine

try:
    from src.brain.openai_client import client
except Exception:
    client = None


@dataclass
class DocumentState:
    paragraphs: List[str]
    idx: int
    ts: float
    engine: str
    conf: float
    mode: str  # "local_only" | "hybrid" | "scene_only"


class DocumentReader:
    """
    Document mode:
      - start(frame, mode): OCR + read paragraph 1
      - next(): read next paragraph
      - repeat(): repeat current paragraph
      - summarize(): 2-sentence summary of full page
    """

    def __init__(self, ocr: OCREngine):
        self.ocr = ocr
        self.state: Optional[DocumentState] = None

    def _expired(self) -> bool:
        if not self.state:
            return True
        ttl = float(getattr(config, "DOC_CACHE_TTL_S", 60.0) or 60.0)
        return (time.time() - self.state.ts) > ttl

    def _normalize_mode(self, mode: Optional[str]) -> str:
        """
        Accepts user-friendly aliases and maps to:
          - "local_only" (offline)
          - "hybrid"
          - "scene_only" (ai)
        """
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

    def start(self, frame_bgr, mode: Optional[str] = None) -> str:
        """
        mode (aliases accepted):
          - "offline" / "local" / "local_only"
          - "hybrid"
          - "ai" / "scene" / "scene_only"

        IMPORTANT:
          For explicit "read this" commands we pass force_ai=True so
          AI mode/hybrid escalation won't be blocked by the cooldown.
        """
        mode_final = self._normalize_mode(mode)

        paras, conf, engine = self.ocr.read_paragraphs(frame_bgr, mode=mode_final, force_ai=True)
        paras = [p.strip() for p in (paras or []) if p.strip()]

        if not paras:
            self.state = None
            return "I can’t read any clear text right now. Try holding it steadier and closer."

        self.state = DocumentState(
            paragraphs=paras,
            idx=0,
            ts=time.time(),
            engine=engine,
            conf=conf,
            mode=mode_final,
        )

        # Make it sound human + clear
        return self._humanize(paras[0], prefix="Okay. Starting from the top.")

    def next(self) -> str:
        if not self.state or self._expired():
            return "I’m not in document mode right now. Say: read this page."

        if self.state.idx + 1 >= len(self.state.paragraphs):
            return "That’s the end of the page."

        self.state.idx += 1
        self.state.ts = time.time()

        # Keep reading from the cached paragraphs (no new OCR call)
        return self._humanize(self.state.paragraphs[self.state.idx])

    def repeat(self) -> str:
        if not self.state or self._expired():
            return "I’m not in document mode right now. Say: read this page."
        return self._humanize(self.state.paragraphs[self.state.idx], prefix="Sure. Here it is again.")

    def refresh(self, frame_bgr) -> str:
        """
        Re-run OCR on the current page (useful if user moved closer / got focus).
        Uses the last selected mode.
        """
        if not self.state:
            return "I’m not in document mode right now. Say: read this page."

        mode_final = self._normalize_mode(self.state.mode)
        paras, conf, engine = self.ocr.read_paragraphs(frame_bgr, mode=mode_final, force_ai=True)
        paras = [p.strip() for p in (paras or []) if p.strip()]

        if not paras:
            return "I still can’t read it clearly. Try more light and hold it steady."

        self.state = DocumentState(
            paragraphs=paras,
            idx=min(self.state.idx, max(0, len(paras) - 1)),
            ts=time.time(),
            engine=engine,
            conf=conf,
            mode=mode_final,
        )
        return self._humanize(self.state.paragraphs[self.state.idx], prefix="Okay. Re-reading that part.")

    def summarize(self) -> str:
        if not self.state or self._expired():
            return "I’m not in document mode right now. Say: read this page."

        full = "\n\n".join(self.state.paragraphs).strip()
        if not full:
            return "I don’t have enough text to summarize."

        if client is None:
            return self._fallback_summary(full)

        try:
            prompt = (
                "Summarize the following text in exactly 2 short sentences. "
                "Make it easy to understand when spoken aloud.\n\n"
                f"TEXT:\n{full[:6000]}"
            )

            resp = client.chat.completions.create(
                model=getattr(config, "OPENAI_CHAT_MODEL", "gpt-4o-mini"),
                messages=[
                    {"role": "system", "content": "You summarize text for a voice assistant. Be concise and clear."},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=120,
                temperature=0.2,
            )

            out = (resp.choices[0].message.content or "").strip()
            out = " ".join(out.split())
            if not out:
                return self._fallback_summary(full)
            return out

        except Exception as e:
            print(f"⚠️ summarize() failed: {e!r}")
            return self._fallback_summary(full)

    def _fallback_summary(self, text: str) -> str:
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
            return "I don’t have enough to summarize."
        if len(parts) == 1:
            return parts[0]
        return f"{parts[0]} {parts[1]}"

    def _humanize(self, paragraph: str, prefix: str = "") -> str:
        p = " ".join((paragraph or "").split()).strip()
        if not p:
            return "I couldn’t read that part clearly."
        # Keep it comfortable for audio
        if len(p) > 520:
            p = p[:520].rsplit(" ", 1)[0] + "..."
        if prefix:
            return f"{prefix} {p}"
        return p
