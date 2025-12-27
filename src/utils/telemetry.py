# src/utils/telemetry.py
from __future__ import annotations

import json
import time
import threading
from pathlib import Path
from typing import Any, Dict, Optional
from queue import SimpleQueue, Empty


def now_ms() -> int:
    return int(time.time() * 1000)


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


def log_event(name: str, payload: Optional[Dict[str, Any]] = None) -> None:
    lg = get_global_logger()
    if lg is None:
        return
    lg.log_event(name, payload)


def log_frame(payload: Dict[str, Any]) -> None:
    lg = get_global_logger()
    if lg is None:
        return
    lg.log_frame(payload)


class TelemetryLogger:
    """
    Fast JSONL logger with a background writer thread.
    Each line is one JSON object.

    record["type"] = "meta" | "frame" | "event"
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

        self._writer.start()

    def log(self, record: Dict[str, Any]) -> None:
        record = dict(record)
        record.setdefault("ts_ms", now_ms())

        # best-effort: avoid runaway memory if something goes wrong
        with self._lock:
            if self._queued_count >= self._max_queue:
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

    def log_frame(self, payload: Dict[str, Any]) -> None:
        self.log({"type": "frame", **payload})

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
