"""Servo backends: a simulated one (no hardware) and a PCA9685 one for the Pi.

Backends do no safety limiting: :class:`roco.body.head.HeadController` clamps first.
"""

from __future__ import annotations

import logging
import threading
from collections import deque
from collections.abc import Mapping
from typing import Any

from roco.body.base import ServoBackend

log = logging.getLogger(__name__)


def _check_range(channel: int, angle: float) -> float:
    angle = float(angle)
    if not 0.0 <= angle <= 180.0:
        raise ValueError(f"servo angle {angle:.1f} out of 0..180 on channel {channel}")
    return angle


class SimulatedServos(ServoBackend):
    """Tracks commanded angles in memory and logs them; perfect for the PC and tests."""

    def __init__(self, history: int = 1000) -> None:
        self._angles: dict[int, float] = {}
        self.history: deque[tuple[int, float]] = deque(maxlen=history)
        self.released: set[int] = set()
        self.closed = False
        self._lock = threading.Lock()

    def set_angle(self, channel: int, angle: float) -> None:
        angle = _check_range(channel, angle)
        with self._lock:
            self._angles[channel] = angle
            self.history.append((channel, angle))
            self.released.discard(channel)
        log.debug("servo[%d] -> %.1f°", channel, angle)

    def get_angle(self, channel: int) -> float | None:
        with self._lock:
            return self._angles.get(channel)

    def release(self, channel: int) -> None:
        with self._lock:
            self.released.add(channel)
        log.debug("servo[%d] released", channel)

    def close(self) -> None:
        if not self.closed:
            self.closed = True
            log.debug("simulated servos closed at %s", self._angles)


class PCA9685Servos(ServoBackend):
    """Adafruit PCA9685 16-channel driver via ``adafruit-circuitpython-servokit``.

    Install with ``pip install -e ".[pi]"`` and enable I2C on the Pi.
    """

    def __init__(
        self,
        i2c_address: int = 0x40,
        channels: int = 16,
        pulse_min_us: int = 500,
        pulse_max_us: int = 2500,
    ) -> None:
        from adafruit_servokit import ServoKit  # lazy: only on the Pi

        self._kit = ServoKit(channels=channels, address=i2c_address)
        self._pulse = (pulse_min_us, pulse_max_us)
        self._configured: set[int] = set()
        self._angles: dict[int, float] = {}
        self._lock = threading.Lock()
        self._closed = False
        log.info("PCA9685 ready at 0x%02X", i2c_address)

    def set_angle(self, channel: int, angle: float) -> None:
        angle = _check_range(channel, angle)
        with self._lock:
            servo = self._kit.servo[channel]
            if channel not in self._configured:
                servo.set_pulse_width_range(*self._pulse)
                self._configured.add(channel)
            servo.angle = angle
            self._angles[channel] = angle

    def get_angle(self, channel: int) -> float | None:
        with self._lock:
            return self._angles.get(channel)

    def release(self, channel: int) -> None:
        with self._lock:
            self._kit.servo[channel].angle = None  # stop sending pulses: servo goes limp

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        for channel in list(self._configured):
            try:
                self.release(channel)
            except Exception:  # pragma: no cover - hardware only
                log.exception("Could not release servo %d", channel)


def create_servo_backend(cfg: Mapping[str, Any] | None = None) -> ServoBackend:
    """Build the backend named by ``servos.backend`` (``simulated`` | ``pca9685``).

    ``cfg`` may be the full config or just its ``servos`` section. If the PCA9685
    can't be opened (library missing, no I2C), falls back to the simulator loudly.
    """
    section = _servo_section(cfg)
    backend = str(section.get("backend", "simulated")).lower()
    if backend == "simulated":
        return SimulatedServos()
    if backend == "pca9685":
        try:
            return PCA9685Servos(
                i2c_address=int(section.get("i2c_address", 0x40)),
                pulse_min_us=int(section.get("pulse_min_us", 500)),
                pulse_max_us=int(section.get("pulse_max_us", 2500)),
            )
        except Exception as exc:
            log.error(
                "PCA9685 unavailable (%s); using simulated servos. "
                'On the Pi: pip install -e ".[pi]" and enable I2C.',
                exc,
            )
            return SimulatedServos()
    raise ValueError(f"Unknown servos.backend {backend!r} (expected simulated or pca9685)")


def _servo_section(cfg: Mapping[str, Any] | None) -> Mapping[str, Any]:
    if not cfg:
        return {}
    servos = cfg.get("servos")
    return servos if isinstance(servos, Mapping) else cfg
