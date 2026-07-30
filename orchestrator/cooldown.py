"""
cooldown.py — thermal-baseline waiter between rungs.

Kept deliberately dumb: poll NVML from Python at ~1 Hz, wait until GPU
temperature drops to `baseline_c + delta_c`, or until `max_sec` has
elapsed. This is fine to run in Python — the measurement window is
already closed by the time we get here.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass

log = logging.getLogger(__name__)

try:
    import pynvml
except ImportError:  # pragma: no cover
    pynvml = None


@dataclass
class CooldownConfig:
    baseline_c: float = 40.0
    delta_c: float = 5.0
    min_sec: float = 10.0
    max_sec: float = 120.0
    poll_sec: float = 1.0


@dataclass
class CooldownResult:
    waited_sec: float
    final_temp_c: float
    reached_target: bool


def capture_baseline(device_index: int, sample_sec: float = 60.0, poll_sec: float = 1.0) -> float:
    """Sample idle temperature for `sample_sec` and return the median."""
    if pynvml is None:
        raise RuntimeError("pynvml not installed; cannot capture baseline")
    pynvml.nvmlInit()
    try:
        h = pynvml.nvmlDeviceGetHandleByIndex(device_index)
        temps: list[float] = []
        end = time.monotonic() + sample_sec
        while time.monotonic() < end:
            t = pynvml.nvmlDeviceGetTemperature(h, pynvml.NVML_TEMPERATURE_GPU)
            temps.append(float(t))
            time.sleep(poll_sec)
        temps.sort()
        median = temps[len(temps) // 2] if temps else 40.0
        return median
    finally:
        pynvml.nvmlShutdown()


def wait(device_index: int, cfg: CooldownConfig) -> CooldownResult:
    if pynvml is None:
        # Best-effort fallback: sleep min_sec and return.
        time.sleep(cfg.min_sec)
        return CooldownResult(waited_sec=cfg.min_sec, final_temp_c=-1.0, reached_target=False)

    pynvml.nvmlInit()
    try:
        h = pynvml.nvmlDeviceGetHandleByIndex(device_index)
        target = cfg.baseline_c + cfg.delta_c
        start = time.monotonic()
        last_temp = float(pynvml.nvmlDeviceGetTemperature(h, pynvml.NVML_TEMPERATURE_GPU))
        while True:
            elapsed = time.monotonic() - start
            last_temp = float(pynvml.nvmlDeviceGetTemperature(h, pynvml.NVML_TEMPERATURE_GPU))
            if elapsed >= cfg.min_sec and last_temp <= target:
                return CooldownResult(elapsed, last_temp, True)
            if elapsed >= cfg.max_sec:
                log.warning("cooldown timeout: temp=%.1fC target=%.1fC", last_temp, target)
                return CooldownResult(elapsed, last_temp, False)
            time.sleep(cfg.poll_sec)
    finally:
        pynvml.nvmlShutdown()
