# src/safety/guidance_engine.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, List, Optional, Tuple
import time

import numpy as np
import src.utils.config as config


def _now() -> float:
    return time.time()


def _normalize_label(label: str) -> str:
    return (label or "object").strip().lower()


def _direction_from_center(center: Optional[Tuple[float, float]], frame_w: int) -> str:
    if not center or frame_w <= 0:
        return "straight ahead"
    x = center[0]
    if x < frame_w / 3:
        return "to your left"
    if x > 2 * frame_w / 3:
        return "to your right"
    return "straight ahead"


def _short_label(label: str) -> str:
    m = {
        "traffic sign": "street sign",
        "traffic light": "traffic light",
        "stop sign": "stop sign",
        "door handle": "door",
        "bicycle": "bike",
        "bike": "bike",
        "motorcycle": "motorcycle",
        "person": "person",
        "pedestrian crossing": "crosswalk",
    }
    return m.get(label, label)


def _is_hazard_keyword(label: str) -> bool:
    kws = getattr(config, "OBSTACLE_PRIORITY_KEYWORDS", ())
    l = label.lower()
    return any(k in l for k in kws)


def _bbox_area_ratio(bbox: Tuple[float, float, float, float], frame_w: int, frame_h: int) -> float:
    x1, y1, x2, y2 = bbox
    w = max(0.0, float(x2) - float(x1))
    h = max(0.0, float(y2) - float(y1))
    area = w * h
    denom = float(frame_w * frame_h) if frame_w * frame_h > 0 else 1.0
    return area / denom


def _depth_median_for_bbox(depth_map: Optional[np.ndarray], bbox: Tuple[int, int, int, int]) -> Optional[float]:
    if depth_map is None:
        return None
    H, W = depth_map.shape[:2]
    x1, y1, x2, y2 = bbox
    x1 = max(0, min(W - 1, int(x1)))
    y1 = max(0, min(H - 1, int(y1)))
    x2 = max(0, min(W, int(x2)))
    y2 = max(0, min(H, int(y2)))
    if x2 <= x1 or y2 <= y1:
        return None
    roi = depth_map[y1:y2, x1:x2]
    if roi.size == 0:
        return None
    return float(np.median(roi))


@dataclass
class GuidanceEvent:
    key: str
    text: str
    severity: int  # 0=info, 1=near, 2=close, 3=danger
    ts: float
    is_info: bool


class GuidanceEngine:
    """
    “Real guide” continuous cues.

    Goals:
      - Short, actionable, human phrasing
      - Only hazards + navigation affordances
      - Dedup 8–10s unless it becomes more dangerous
      - Quiet mode: only real danger
      - Depth-aware escalation (but only when depth is reliable)
    """

    def __init__(self):
        self.last_spoken_ts: float = 0.0
        self.last_event_ts_by_key: Dict[str, float] = {}
        self.last_event_sev_by_key: Dict[str, int] = {}
        self.last_info_ts: float = 0.0

    def _allowed_labels(self, profile: str) -> set[str]:
        if profile == "quiet":
            return set()

        if profile == "indoor":
            return {
                "stairs", "stair", "steps", "step",
                "door", "door handle",
                "elevator",
                "person",
            }

        # outdoor
        return {
            "stairs", "stair", "steps", "step",
            "curb", "kerb",
            "pole", "post", "bollard",
            "bike", "bicycle", "motorcycle",
            "car", "bus", "truck",
            "traffic light", "traffic sign", "stop sign",
            "crosswalk", "pedestrian crossing",
            "person",
        }

    def _estimate_severity(
        self,
        det: Dict[str, Any],
        frame_w: int,
        frame_h: int,
        depth_map: Optional[np.ndarray],
        depth_quality: float,
    ) -> int:
        bbox = det.get("bbox")
        if not bbox or len(bbox) != 4:
            return 0

        x1, y1, x2, y2 = bbox
        bbox_int = (int(x1), int(y1), int(x2), int(y2))

        # Use depth only when quality is good enough (prevents spam)
        min_q = float(getattr(config, "DEPTH_MIN_QUALITY", 0.35) or 0.35)
        if depth_map is not None and depth_quality >= min_q:
            depth_val = _depth_median_for_bbox(depth_map, bbox_int)
            if depth_val is not None:
                close_t = float(getattr(config, "DEPTH_CLOSE_THRESH", 0.55) or 0.55)
                very_t = float(getattr(config, "DEPTH_VERY_CLOSE_THRESH", 0.70) or 0.70)
                if depth_val >= very_t:
                    return 3
                if depth_val >= close_t:
                    return 2
                # else fall through to bbox for near/info

        r = _bbox_area_ratio((float(x1), float(y1), float(x2), float(y2)), frame_w, frame_h)

        near_r = float(getattr(config, "GUIDANCE_BBOX_NEAR_AREA_RATIO", 0.10) or 0.10)
        close_r = float(getattr(config, "GUIDANCE_BBOX_CLOSE_AREA_RATIO", 0.18) or 0.18)
        danger_r = float(getattr(config, "GUIDANCE_BBOX_DANGER_AREA_RATIO", 0.28) or 0.28)

        if r >= danger_r:
            return 3
        if r >= close_r:
            return 2
        if r >= near_r:
            return 1
        return 0

    def _human_line(self, label: str, direction: str, sev: int) -> Tuple[str, bool]:
        """
        Returns (spoken_text, is_info)
        """
        l = label

        # Navigation affordances (info-ish unless close)
        if l in ("door", "door handle"):
            base = f"Door {direction}."
            is_info = True
        elif l in ("traffic light", "traffic sign", "stop sign"):
            base = f"{_short_label(l).title()} {direction}."
            is_info = True
        elif l in ("crosswalk", "pedestrian crossing"):
            base = f"Crosswalk {direction}."
            is_info = True

        # Hazards / obstacles
        elif "stairs" in l or "steps" in l or l == "step":
            base = "Watch your step—stairs ahead."
            is_info = False
        elif "curb" in l or "kerb" in l:
            base = "Curb ahead."
            is_info = False
        elif "pole" in l or "post" in l or "bollard" in l:
            base = f"Heads up—pole {direction}."
            is_info = False
        elif l in ("car", "bus", "truck", "motorcycle", "bike", "bicycle"):
            base = f"{_short_label(l).title()} {direction}."
            is_info = False
        else:
            base = f"{_short_label(l).title()} {direction}."
            is_info = False

        # Gentle escalation (only when it matters)
        if sev >= 3:
            # Keep it short; “very close” is enough.
            if base.endswith("."):
                base = base[:-1]
            base = f"{base}, very close."
        elif sev == 2:
            if base.endswith("."):
                base = base[:-1]
            base = f"{base}, close."

        return base, is_info

    def update(
        self,
        detections: List[Dict[str, Any]],
        frame_w: int,
        frame_h: int,
        *,
        profile: Optional[str] = None,
        depth_map: Optional[np.ndarray] = None,
        depth_quality: float = 0.0,
        now: Optional[float] = None,
    ) -> Optional[str]:
        if not getattr(config, "GUIDANCE_ENABLED", False):
            return None

        profile = (profile or getattr(config, "GUIDANCE_PROFILE", "indoor") or "indoor").strip().lower()
        now = now if now is not None else _now()

        cooldown = float(getattr(config, "GUIDANCE_COOLDOWN_S", 2.2) or 2.2)
        if (now - self.last_spoken_ts) < cooldown:
            return None

        repeat_after = float(getattr(config, "GUIDANCE_REPEAT_AFTER_S", 9.0) or 9.0)
        escalate_gap = float(getattr(config, "GUIDANCE_ESCALATE_MIN_GAP_S", 1.2) or 1.2)
        info_gap = float(getattr(config, "GUIDANCE_INFO_MIN_GAP_S", 18.0) or 18.0)
        min_conf = float(getattr(config, "GUIDANCE_MIN_CONF", 0.20) or 0.20)

        allowed = self._allowed_labels(profile)
        candidates: List[GuidanceEvent] = []

        for d in detections or []:
            conf = float(d.get("confidence", 1.0) or 0.0)
            if conf < min_conf:
                continue

            raw_label = d.get("label") or "object"
            label = _normalize_label(raw_label)

            # Profile gating
            if profile == "quiet":
                if not _is_hazard_keyword(label):
                    continue
            else:
                if (label not in allowed) and (not _is_hazard_keyword(label)):
                    continue

            bbox = d.get("bbox")
            if not bbox or len(bbox) != 4:
                continue

            direction = _direction_from_center(d.get("center"), frame_w)
            track_id = d.get("track_id")
            key = f"{label}:{direction}:{track_id or ''}".strip()

            sev = self._estimate_severity(
                d, frame_w, frame_h,
                depth_map=depth_map,
                depth_quality=depth_quality,
            )

            # Quiet mode: only real danger
            if profile == "quiet" and sev < 3:
                continue

            text, is_info = self._human_line(label, direction, sev)

            # Make “info” rare
            if is_info and sev <= 1 and (now - self.last_info_ts) < info_gap:
                continue

            # Dedup + escalation
            last_ts = self.last_event_ts_by_key.get(key, 0.0)
            last_sev = self.last_event_sev_by_key.get(key, -1)

            if (now - last_ts) < repeat_after:
                if sev > last_sev and (now - last_ts) >= escalate_gap:
                    pass
                else:
                    continue

            candidates.append(GuidanceEvent(key=key, text=text, severity=sev, ts=now, is_info=is_info))

        if not candidates:
            return None

        # Always prefer the most dangerous/close message
        candidates.sort(key=lambda e: (e.severity, e.ts), reverse=True)
        chosen = candidates[0]

        self.last_event_ts_by_key[chosen.key] = now
        self.last_event_sev_by_key[chosen.key] = chosen.severity
        self.last_spoken_ts = now

        if chosen.is_info and chosen.severity <= 1:
            self.last_info_ts = now

        return chosen.text
