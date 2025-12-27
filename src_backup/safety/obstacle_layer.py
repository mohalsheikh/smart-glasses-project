# src/safety/obstacle_layer.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, List, Optional, Tuple
import time
import re

import numpy as np
import src.utils.config as config


@dataclass
class _LastAlert:
    key: str
    ts: float


class ObstacleLayer:
    """
    Obstacle + trip-hazard warnings.

    Modes:
      - bbox: fast heuristic using bbox size + bottom-of-frame
      - depth: uses MiDaS depth map (normalized inverse depth 0..1, higher=closer)
              + step-down detection in walking path region

    Returns short spoken warnings. One warning at a time.
    """

    def __init__(self):
        self._last_global_ts: float = 0.0
        self._last_by_track: Dict[int, float] = {}
        self._last_alert: Optional[_LastAlert] = None

        self._patterns = {
            "step": re.compile(r"(stairs?|steps?|staircase|curb)", re.I),
            "pole": re.compile(r"(pole|post|bollard|street\s*light|lamp\s*post|sign\s*post)", re.I),
            "chair": re.compile(r"(chair|stool)", re.I),
            "table": re.compile(r"(table|desk)", re.I),
            "bike": re.compile(r"(bicycle|bike|motorcycle)", re.I),
            "pet": re.compile(r"(dog|cat|pet)", re.I),
        }

    def _direction(self, cx: float, frame_w: int) -> str:
        if frame_w <= 0:
            return "in front of you"
        if cx < frame_w / 3:
            return "on your left"
        if cx > 2 * frame_w / 3:
            return "on your right"
        return "in front of you"

    def _hazard_type(self, label: str) -> Optional[str]:
        if not label:
            return None
        for k, pat in self._patterns.items():
            if pat.search(label):
                return k
        return None

    # ---------------------------
    # bbox mode closeness (fast)
    # ---------------------------
    def _closeness_bbox(self, bbox: Tuple[int, int, int, int], frame_w: int, frame_h: int) -> Optional[str]:
        x1, y1, x2, y2 = bbox
        w = max(0, x2 - x1)
        h = max(0, y2 - y1)
        if frame_w <= 0 or frame_h <= 0:
            return None

        area_ratio = (w * h) / float(frame_w * frame_h)
        bottom_ratio = y2 / float(frame_h)

        near_area = float(getattr(config, "OBSTACLE_NEAR_AREA_RATIO", 0.10) or 0.10)
        very_near_area = float(getattr(config, "OBSTACLE_VERY_NEAR_AREA_RATIO", 0.18) or 0.18)
        bottom_gate = float(getattr(config, "OBSTACLE_BOTTOM_Y_RATIO", 0.62) or 0.62)

        if bottom_ratio < bottom_gate:
            near_area *= 1.35
            very_near_area *= 1.35

        if area_ratio >= very_near_area:
            return "very close"
        if area_ratio >= near_area:
            return "close"
        return None

    # ---------------------------
    # depth mode closeness (better)
    # ---------------------------
    def _closeness_depth(
        self,
        depth_map: np.ndarray,
        bbox: Tuple[int, int, int, int],
    ) -> Optional[str]:
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

        val = float(np.median(roi))  # 0..1, higher=closer
        close_t = float(getattr(config, "DEPTH_CLOSE_THRESH", 0.55) or 0.55)
        very_t = float(getattr(config, "DEPTH_VERY_CLOSE_THRESH", 0.70) or 0.70)

        if val >= very_t:
            return "very close"
        if val >= close_t:
            return "close"
        return None

    def _detect_step_down(self, depth_map: np.ndarray) -> bool:
        """
        Depth-only trip hazard detector.
        We look at a walking-path ROI in the center-bottom:
          - if bottom band is significantly "farther" than upper band, likely a drop/step down.
        depth_map: normalized inverse depth (higher=closer)
        """
        if depth_map is None:
            return False
        if not getattr(config, "DEPTH_STEP_DOWN_ENABLED", True):
            return False

        H, W = depth_map.shape[:2]
        if H < 40 or W < 40:
            return False

        x1 = int(W * 0.35)
        x2 = int(W * 0.65)

        upper_y1 = int(H * 0.62)
        upper_y2 = int(H * 0.78)

        bottom_y1 = int(H * 0.84)
        bottom_y2 = int(H * 0.97)

        upper = depth_map[upper_y1:upper_y2, x1:x2]
        bottom = depth_map[bottom_y1:bottom_y2, x1:x2]
        if upper.size == 0 or bottom.size == 0:
            return False

        upper_med = float(np.median(upper))
        bottom_med = float(np.median(bottom))

        # upper should not be super far (avoid noise)
        min_upper = float(getattr(config, "DEPTH_STEP_DOWN_MIN_UPPER", 0.45) or 0.45)
        if upper_med < min_upper:
            return False

        drop_ratio = float(getattr(config, "DEPTH_STEP_DOWN_DROP_RATIO", 0.22) or 0.22)

        # If bottom is much LOWER inverse-depth => farther => drop
        return bottom_med < (upper_med * (1.0 - drop_ratio))

    # ---------------------------
    # public update
    # ---------------------------
    def update(
        self,
        frame,
        detections: List[Dict[str, Any]],
        *,
        depth_map: Optional[np.ndarray] = None,
        now: Optional[float] = None,
    ) -> Optional[str]:
        if not getattr(config, "OBSTACLE_ENABLED", False):
            return None

        if now is None:
            now = time.time()

        min_interval = float(getattr(config, "OBSTACLE_MIN_INTERVAL_S", 2.0) or 2.0)
        per_object_cooldown = float(getattr(config, "OBSTACLE_REPEAT_COOLDOWN_S", 6.0) or 6.0)

        # throttle
        if (now - self._last_global_ts) < min_interval:
            return None

        if frame is None:
            return None

        frame_h, frame_w = frame.shape[:2]
        mode = (getattr(config, "OBSTACLE_MODE", "bbox") or "bbox").strip().lower()

        # 1) Depth-only step-down (highest priority)
        if mode == "depth" and depth_map is not None:
            if self._detect_step_down(depth_map):
                msg = "Step down ahead."
                if not (self._last_alert and self._last_alert.key == msg and (now - self._last_alert.ts) < (min_interval * 2)):
                    self._last_global_ts = now
                    self._last_alert = _LastAlert(key=msg, ts=now)
                    return msg

        if not detections:
            return None

        candidates = []
        for d in detections:
            label = str(d.get("label") or "")
            hazard = self._hazard_type(label)
            if hazard is None:
                continue

            bbox = d.get("bbox")
            if not bbox or len(bbox) != 4:
                continue

            x1, y1, x2, y2 = bbox
            bbox_int = (int(x1), int(y1), int(x2), int(y2))

            if mode == "depth" and depth_map is not None:
                close = self._closeness_depth(depth_map, bbox_int)
            else:
                close = self._closeness_bbox(bbox_int, frame_w, frame_h)

            if close is None:
                continue

            cx, cy = d.get("center", (None, None))
            if cx is None:
                cx = (bbox_int[0] + bbox_int[2]) / 2.0

            direction = self._direction(float(cx), frame_w)

            track_id = d.get("track_id")
            if isinstance(track_id, int):
                last_ts = self._last_by_track.get(track_id, 0.0)
                if (now - last_ts) < per_object_cooldown:
                    continue

            priority = {
                "step": 100,
                "pole": 90,
                "pet": 85,
                "bike": 80,
                "chair": 70,
                "table": 65,
            }.get(hazard, 50)

            if close == "very close":
                priority += 20

            candidates.append((priority, hazard, close, direction, track_id))

        if not candidates:
            return None

        candidates.sort(key=lambda x: x[0], reverse=True)
        _, hazard, close, direction, track_id = candidates[0]

        # Messages
        if hazard == "step":
            msg = "Stairs ahead." if direction == "in front of you" else f"Stairs {direction}."
            if close == "very close":
                msg += " Very close."
            else:
                msg += " Close."
        elif hazard == "pole":
            msg = f"Pole {direction}, {close}."
        elif hazard == "pet":
            msg = f"Pet {direction}, {close}."
        elif hazard == "bike":
            msg = f"Bike {direction}, {close}."
        elif hazard == "chair":
            msg = f"Chair {direction}, {close}."
        elif hazard == "table":
            msg = f"Table {direction}, {close}."
        else:
            msg = f"Obstacle {direction}, {close}."

        # anti-spam
        if self._last_alert and self._last_alert.key == msg and (now - self._last_alert.ts) < (min_interval * 2):
            return None

        self._last_global_ts = now
        self._last_alert = _LastAlert(key=msg, ts=now)
        if isinstance(track_id, int):
            self._last_by_track[track_id] = now

        return msg
