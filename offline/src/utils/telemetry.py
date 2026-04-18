# src/utils/telemetry.py
"""
Enhanced Telemetry System for Smart Glasses

Logs comprehensive metrics for visualization and analysis:
- Frame processing (FPS, latency, detections)
- Voice interactions (transcription, response times)
- Safety events (obstacles, guidance)
- AI operations (scene description, OCR)
- Speech output (TTS timing)
- System health (memory, errors)

Each line is one JSON object in JSONL format.
"""

from __future__ import annotations

import json
import time
import threading
import traceback
from pathlib import Path
from typing import Any, Dict, Optional, List
from queue import SimpleQueue, Empty
from dataclasses import dataclass, asdict
from contextlib import contextmanager

# Try to import psutil for system metrics
try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False


def now_ms() -> int:
    return int(time.time() * 1000)


def now_s() -> float:
    return time.time()


# ----------------------------
# Global logger (optional convenience)
# ----------------------------

_GLOBAL_LOGGER: Optional["TelemetryLogger"] = None
_GLOBAL_LOCK = threading.Lock()


def set_global_logger(logger: Optional["TelemetryLogger"]) -> None:
    global _GLOBAL_LOGGER
    with _GLOBAL_LOCK:
        _GLOBAL_LOGGER = logger


def get_global_logger() -> Optional["TelemetryLogger"]:
    with _GLOBAL_LOCK:
        return _GLOBAL_LOGGER


# ----------------------------
# Convenience functions
# ----------------------------

def log_event(name: str, payload: Optional[Dict[str, Any]] = None) -> None:
    """Log a general event."""
    lg = get_global_logger()
    if lg is None:
        return
    lg.log_event(name, payload)


def log_frame(payload: Dict[str, Any]) -> None:
    """Log frame-level telemetry."""
    lg = get_global_logger()
    if lg is None:
        return
    lg.log_frame(payload)


def log_voice(
    req_id: str,
    phase: str,
    listen_ms: Optional[float] = None,
    process_ms: Optional[float] = None,
    total_ms: Optional[float] = None,
    text_len: Optional[int] = None,
    response_len: Optional[int] = None,
    command_type: Optional[str] = None,
    success: bool = True,
) -> None:
    """Log voice interaction telemetry."""
    lg = get_global_logger()
    if lg is None:
        return
    payload = {
        "type": "voice",
        "req_id": req_id,
        "phase": phase,
        "success": success,
    }
    if listen_ms is not None:
        payload["listen_ms"] = listen_ms
    if process_ms is not None:
        payload["process_ms"] = process_ms
    if total_ms is not None:
        payload["total_ms"] = total_ms
    if text_len is not None:
        payload["text_len"] = text_len
    if response_len is not None:
        payload["response_len"] = response_len
    if command_type is not None:
        payload["command_type"] = command_type
    lg.log(payload)


def log_safety(
    event_type: str,
    hazard_type: Optional[str] = None,
    severity: int = 0,
    direction: Optional[str] = None,
    distance: Optional[str] = None,
    message: Optional[str] = None,
    depth_quality: Optional[float] = None,
) -> None:
    """Log safety/obstacle telemetry."""
    lg = get_global_logger()
    if lg is None:
        return
    payload = {
        "type": "safety",
        "event_type": event_type,
        "severity": severity,
    }
    if hazard_type is not None:
        payload["hazard_type"] = hazard_type
    if direction is not None:
        payload["direction"] = direction
    if distance is not None:
        payload["distance"] = distance
    if message is not None:
        payload["message"] = message[:200] if message else None
    if depth_quality is not None:
        payload["depth_quality"] = depth_quality
    lg.log(payload)


def log_ai(
    operation: str,
    latency_ms: float,
    success: bool = True,
    model: Optional[str] = None,
    mode: Optional[str] = None,
    result_len: Optional[int] = None,
    input_tokens: Optional[int] = None,
    output_tokens: Optional[int] = None,
) -> None:
    """Log AI operation telemetry (scene AI, OCR)."""
    lg = get_global_logger()
    if lg is None:
        return
    payload = {
        "type": "ai",
        "operation": operation,
        "latency_ms": latency_ms,
        "success": success,
    }
    if model is not None:
        payload["model"] = model
    if mode is not None:
        payload["mode"] = mode
    if result_len is not None:
        payload["result_len"] = result_len
    if input_tokens is not None:
        payload["input_tokens"] = input_tokens
    if output_tokens is not None:
        payload["output_tokens"] = output_tokens
    lg.log(payload)


def log_speech(
    text_len: int,
    duration_ms: float,
    source: str,
    queued: bool = False,
    skipped_dedupe: bool = False,
) -> None:
    """Log TTS output telemetry."""
    lg = get_global_logger()
    if lg is None:
        return
    lg.log({
        "type": "speech",
        "text_len": text_len,
        "duration_ms": duration_ms,
        "source": source,
        "queued": queued,
        "skipped_dedupe": skipped_dedupe,
    })


def log_error(error: Exception, context: Optional[str] = None) -> None:
    """Log an error event."""
    lg = get_global_logger()
    if lg is None:
        return
    lg.log({
        "type": "error",
        "error_type": type(error).__name__,
        "error_msg": str(error)[:500],
        "context": context,
        "traceback": traceback.format_exc()[:2000],
    })


def log_system_health() -> None:
    """Log current system health metrics."""
    lg = get_global_logger()
    if lg is None:
        return
    if not HAS_PSUTIL:
        return
    try:
        process = psutil.Process()
        lg.log({
            "type": "system",
            "cpu_percent": psutil.cpu_percent(interval=None),
            "memory_percent": psutil.virtual_memory().percent,
            "process_memory_mb": process.memory_info().rss / (1024 * 1024),
            "process_threads": process.num_threads(),
        })
    except Exception:
        pass


@contextmanager
def timed_operation(name: str, extra: Optional[Dict[str, Any]] = None):
    """Context manager to time an operation and log it."""
    t0 = time.perf_counter()
    result = {"success": True, "error": None}
    try:
        yield result
    except Exception as e:
        result["success"] = False
        result["error"] = str(e)
        raise
    finally:
        duration_ms = (time.perf_counter() - t0) * 1000.0
        payload = {
            "operation": name,
            "duration_ms": duration_ms,
            **result,
        }
        if extra:
            payload.update(extra)
        log_event(f"timed_{name}", payload)


# ----------------------------
# Main Logger Class
# ----------------------------

class TelemetryLogger:
    """
    Fast JSONL logger with a background writer thread.
    Each line is one JSON object.

    record["type"] = "meta" | "frame" | "event" | "voice" | "safety" | "ai" | "speech" | "system" | "error"
    """

    def __init__(self, out_path: str | Path, max_queue: int = 20000):
        self.out_path = Path(out_path)
        self.out_path.parent.mkdir(parents=True, exist_ok=True)

        self._q: SimpleQueue[Dict[str, Any]] = SimpleQueue()
        self._stop = threading.Event()
        self._writer = threading.Thread(target=self._writer_loop, daemon=True)
        self._max_queue = max_queue

        self._queued_count = 0
        self._lock = threading.Lock()
        
        # Stats tracking
        self._stats = {
            "frames_logged": 0,
            "events_logged": 0,
            "errors_logged": 0,
            "dropped_records": 0,
        }

        self._writer.start()

    def log(self, record: Dict[str, Any]) -> None:
        record = dict(record)
        record.setdefault("ts_ms", now_ms())

        # best-effort: avoid runaway memory if something goes wrong
        with self._lock:
            if self._queued_count >= self._max_queue:
                self._stats["dropped_records"] += 1
                return
            self._queued_count += 1

        self._q.put(record)

    def log_meta(self, meta: Dict[str, Any]) -> None:
        self.log({"type": "meta", **meta})

    def log_event(self, name: str, payload: Optional[Dict[str, Any]] = None) -> None:
        rec: Dict[str, Any] = {"type": "event", "name": name}
        if payload:
            rec.update(payload)
        self.log(rec)
        with self._lock:
            self._stats["events_logged"] += 1

    def log_frame(self, payload: Dict[str, Any]) -> None:
        self.log({"type": "frame", **payload})
        with self._lock:
            self._stats["frames_logged"] += 1
    
    def log_dataclass(self, obj) -> None:
        """Log a dataclass object."""
        data = asdict(obj) if hasattr(obj, '__dataclass_fields__') else obj
        rec_type = type(obj).__name__.lower().replace("telemetry", "")
        self.log({"type": rec_type, **data})

    def get_stats(self) -> Dict[str, int]:
        """Get logging statistics."""
        with self._lock:
            return dict(self._stats)

    def close(self) -> None:
        self._stop.set()
        self._writer.join(timeout=2.0)

    def _writer_loop(self) -> None:
        try:
            with self.out_path.open("a", encoding="utf-8") as f:
                while not self._stop.is_set():
                    try:
                        rec = self._q.get(timeout=0.25)
                    except Empty:
                        continue

                    try:
                        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    except Exception:
                        pass
                    finally:
                        with self._lock:
                            self._queued_count = max(0, self._queued_count - 1)

                # drain remaining items
                while True:
                    try:
                        rec = self._q.get_nowait()
                    except Empty:
                        break
                    try:
                        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    except Exception:
                        pass
                    finally:
                        with self._lock:
                            self._queued_count = max(0, self._queued_count - 1)
        except Exception:
            # do not crash the app if telemetry fails
            return


# ----------------------------
# Telemetry Aggregator (for real-time stats)
# ----------------------------

class TelemetryAggregator:
    """
    Aggregates telemetry for real-time dashboard updates.
    Maintains rolling windows of metrics.
    """
    
    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self._lock = threading.Lock()
        
        # Rolling windows
        self.fps_window: List[float] = []
        self.latency_window: List[float] = []
        self.detection_window: List[int] = []
        self.confidence_window: List[float] = []
        
        # Counters
        self.total_frames = 0
        self.total_voice_interactions = 0
        self.total_safety_events = 0
        self.total_ai_calls = 0
        
        # Label tracking
        self.label_counts: Dict[str, int] = {}
        
        # Error tracking
        self.error_counts: Dict[str, int] = {}
    
    def add_frame(self, fps: float, latency_ms: float, n_detections: int, 
                  top_conf: float, labels: List[str]) -> None:
        with self._lock:
            self.total_frames += 1
            
            self.fps_window.append(fps)
            self.latency_window.append(latency_ms)
            self.detection_window.append(n_detections)
            if top_conf > 0:
                self.confidence_window.append(top_conf)
            
            # Trim windows
            if len(self.fps_window) > self.window_size:
                self.fps_window = self.fps_window[-self.window_size:]
            if len(self.latency_window) > self.window_size:
                self.latency_window = self.latency_window[-self.window_size:]
            if len(self.detection_window) > self.window_size:
                self.detection_window = self.detection_window[-self.window_size:]
            if len(self.confidence_window) > self.window_size:
                self.confidence_window = self.confidence_window[-self.window_size:]
            
            # Track labels
            for label in labels:
                self.label_counts[label] = self.label_counts.get(label, 0) + 1
    
    def add_voice(self) -> None:
        with self._lock:
            self.total_voice_interactions += 1
    
    def add_safety(self) -> None:
        with self._lock:
            self.total_safety_events += 1
    
    def add_ai_call(self) -> None:
        with self._lock:
            self.total_ai_calls += 1
    
    def add_error(self, error_type: str) -> None:
        with self._lock:
            self.error_counts[error_type] = self.error_counts.get(error_type, 0) + 1
    
    def get_summary(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "total_frames": self.total_frames,
                "total_voice": self.total_voice_interactions,
                "total_safety": self.total_safety_events,
                "total_ai": self.total_ai_calls,
                "avg_fps": sum(self.fps_window) / len(self.fps_window) if self.fps_window else 0,
                "avg_latency_ms": sum(self.latency_window) / len(self.latency_window) if self.latency_window else 0,
                "avg_detections": sum(self.detection_window) / len(self.detection_window) if self.detection_window else 0,
                "avg_confidence": sum(self.confidence_window) / len(self.confidence_window) if self.confidence_window else 0,
                "top_labels": sorted(self.label_counts.items(), key=lambda x: x[1], reverse=True)[:10],
                "errors": dict(self.error_counts),
            }