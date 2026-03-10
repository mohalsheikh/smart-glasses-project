from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
import time


@dataclass
class ToFReading:
    meters: float
    ts: float


class ToFSensor:
    """Interface."""
    def read(self) -> Optional[ToFReading]:
        raise NotImplementedError


class NoopToFSensor(ToFSensor):
    def read(self) -> Optional[ToFReading]:
        return None


class VL53L1XSensor(ToFSensor):
    """
    VL53L1X ToF sensor (Raspberry Pi / I2C).
    Requires:
      pip install adafruit-circuitpython-vl53l1x adafruit-blinka
    """
    def __init__(self, i2c_bus: int = 1, timing_budget_ms: int = 50):
        self._ready = False
        self._sensor = None
        self._last_fail_ts = 0.0

        try:
            import board  # type: ignore
            import busio  # type: ignore
            import adafruit_vl53l1x  # type: ignore

            # board.I2C() uses default bus; busio.I2C can also be used
            # We keep it simple for Raspberry Pi
            i2c = board.I2C()
            sensor = adafruit_vl53l1x.VL53L1X(i2c)
            sensor.timing_budget = timing_budget_ms
            sensor.start_ranging()

            self._sensor = sensor
            self._ready = True
            print("📡 ToF sensor ready (VL53L1X)")

        except Exception as e:
            self._ready = False
            print(f"⚠️ ToF sensor disabled (init failed): {e!r}")

    def read(self) -> Optional[ToFReading]:
        if not self._ready or self._sensor is None:
            return None

        now = time.time()
        if (now - self._last_fail_ts) < 2.0:
            return None

        try:
            # distance in cm
            dist_cm = self._sensor.distance
            if dist_cm is None:
                return None

            meters = float(dist_cm) / 100.0
            if meters <= 0:
                return None

            return ToFReading(meters=meters, ts=now)

        except Exception as e:
            self._last_fail_ts = time.time()
            print(f"⚠️ ToF read failed: {e!r}")
            return None
