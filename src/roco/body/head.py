"""Pan/tilt head: safe limits, smooth stepped motion and non-blocking gestures.

Angles: ``pan`` and ``tilt`` offsets are relative to the configured ``center``.
Positive pan turns to ROCO's left (viewer's right) and positive tilt looks *down*;
set ``servos.pan.invert`` / ``servos.tilt.invert: true`` if your bracket is mounted the
other way round. Every commanded angle is clamped to the axis ``min``/``max``.

All gestures run on one worker thread. Starting a gesture cancels the running one
within a step (~20 ms); only one gesture ever drives the servos at a time. While the
head is marked idle (:meth:`HeadController.set_idle`) it looks around by itself, and
picks that up again after any one-shot gesture.
"""

from __future__ import annotations

import logging
import random
import threading
import time
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass
from typing import Any

from roco.body.base import ServoBackend
from roco.body.expression import ease_in_out_cubic

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class AxisConfig:
    channel: int
    min: float
    max: float
    center: float
    invert: bool = False

    def __post_init__(self) -> None:
        if not 0 <= self.min <= self.center <= self.max <= 180:
            raise ValueError(f"need 0 <= min <= center <= max <= 180, got {self}")

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any], channel: int) -> AxisConfig:
        return cls(
            channel=int(data.get("channel", channel)),
            min=float(data.get("min", 0)),
            max=float(data.get("max", 180)),
            center=float(data.get("center", 90)),
            invert=bool(data.get("invert", False)),
        )

    def clamp(self, angle: float) -> float:
        return min(self.max, max(self.min, float(angle)))

    def angle_for(self, offset: float) -> float:
        """Absolute, clamped servo angle for an offset from centre."""
        return self.clamp(self.center + (-offset if self.invert else offset))


@dataclass(frozen=True)
class Keyframe:
    """Move to (pan, tilt) offsets over ``seconds``, then hold for ``hold`` seconds."""

    pan: float
    tilt: float
    seconds: float
    hold: float = 0.0


@dataclass(frozen=True)
class Gesture:
    name: str
    keyframes: tuple[Keyframe, ...] = ()
    # For endless gestures (look-around): a generator of keyframes.
    generator: Callable[[random.Random], Iterator[Keyframe]] | None = None


def motion_point(
    start: tuple[float, float], end: tuple[float, float], progress: float
) -> tuple[float, float]:
    """Eased (pan, tilt) at ``progress`` (0..1, clamped) of the way from ``start`` to ``end``."""
    t = ease_in_out_cubic(progress)
    return (start[0] + (end[0] - start[0]) * t, start[1] + (end[1] - start[1]) * t)


def _look_around(rng: random.Random) -> Iterator[Keyframe]:
    while True:
        if rng.random() < 0.3:
            pan, tilt = 0.0, 0.0
        else:
            pan, tilt = rng.uniform(-28, 28), rng.uniform(-8, 6)
        yield Keyframe(pan, tilt, seconds=rng.uniform(0.7, 1.3), hold=rng.uniform(2.0, 5.0))


GESTURES: dict[str, Gesture] = {
    "center": Gesture("center", (Keyframe(0, 0, 0.45),)),
    "nod": Gesture(
        "nod",
        (
            Keyframe(0, 10, 0.17),
            Keyframe(0, -4, 0.17),
            Keyframe(0, 8, 0.15),
            Keyframe(0, 0, 0.2),
        ),
    ),
    "shake": Gesture(
        "shake",
        (
            Keyframe(-14, 0, 0.17),
            Keyframe(14, 0, 0.25),
            Keyframe(-10, 0, 0.22),
            Keyframe(0, 0, 0.2),
        ),
    ),
    "tilt": Gesture(  # curious head tilt: turn a little and look up, hold, come back
        "tilt",
        (Keyframe(12, -7, 0.4, hold=1.8), Keyframe(0, 0, 0.5)),
    ),
    "startle": Gesture("startle", (Keyframe(0, -9, 0.1, hold=0.5), Keyframe(0, 0, 0.4))),
    "look_down": Gesture("look_down", (Keyframe(0, 12, 0.8),)),  # sad: stays down
    "sleep": Gesture("sleep", (Keyframe(0, 90, 2.5),)),  # all the way down (clamped)
    "look_around": Gesture("look_around", generator=_look_around),
}


class HeadController:
    """Drives pan/tilt servos through a single gesture worker thread."""

    def __init__(
        self,
        servos: ServoBackend,
        pan: AxisConfig,
        tilt: AxisConfig,
        *,
        step_s: float = 0.02,
        rng: random.Random | None = None,
        idle_delay_s: float = 1.5,
        time_scale: float = 1.0,
        clock: Callable[[], float] = time.monotonic,
        on_gesture: Callable[[str], None] | None = None,
    ) -> None:
        self.servos = servos
        self.pan = pan
        self.tilt = tilt
        self.step_s = step_s
        self.rng = rng or random.Random()
        self.idle_delay_s = idle_delay_s
        self.time_scale = time_scale  # < 1 speeds every gesture up (tests)
        self.clock = clock
        self.on_gesture = on_gesture
        self.history: list[str] = []  # names of gestures started, for tests and the demo
        self._pos = (0.0, 0.0)  # current offsets
        self._cond = threading.Condition()
        self._pending: Gesture | None = None
        self._current: str | None = None
        self._cancel = threading.Event()
        self._idle = False
        self._closed = False
        self._thread: threading.Thread | None = None
        self._write(0.0, 0.0)

    @classmethod
    def from_config(
        cls, servos: ServoBackend, cfg: Mapping[str, Any] | None = None, **kwargs: Any
    ) -> HeadController:
        section: Mapping[str, Any] = (cfg or {}).get("servos", cfg or {})  # full cfg or section
        pan = AxisConfig.from_mapping(section.get("pan", {}), channel=0)
        tilt = AxisConfig.from_mapping(section.get("tilt", {}), channel=1)
        return cls(servos, pan, tilt, **kwargs)

    # -- queries ----------------------------------------------------------------------
    @property
    def current_gesture(self) -> str | None:
        return self._current

    @property
    def angles(self) -> tuple[float, float]:
        """Current absolute (pan, tilt) servo angles."""
        return self.pan.angle_for(self._pos[0]), self.tilt.angle_for(self._pos[1])

    @property
    def idle(self) -> bool:
        return self._idle

    # -- commands (non-blocking) ------------------------------------------------------
    def play(self, name: str) -> None:
        """Start gesture ``name``, cancelling whatever is running."""
        try:
            gesture = GESTURES[name]
        except KeyError:
            raise ValueError(f"unknown gesture {name!r}; known: {sorted(GESTURES)}") from None
        with self._cond:
            if self._closed:
                return
            self._pending = gesture
            self._cancel.set()
            self._ensure_thread()
            self._cond.notify_all()

    def nod(self) -> None:
        self.play("nod")

    def shake(self) -> None:
        self.play("shake")

    def curious_tilt(self) -> None:
        self.play("tilt")

    def center(self) -> None:
        self.play("center")

    def look_around(self) -> None:
        self.play("look_around")

    def set_idle(self, idle: bool) -> None:
        """Idle heads look around; leaving idle stops the look-around (not other gestures)."""
        with self._cond:
            self._idle = idle
            if idle:
                if self._current is None and self._pending is None:
                    self._pending = GESTURES["look_around"]
                    self._ensure_thread()
                    self._cond.notify_all()
            else:
                if self._pending is not None and self._pending.name == "look_around":
                    self._pending = None
                if self._current == "look_around":
                    self._cancel.set()

    def cancel(self) -> None:
        """Stop where we are (the head keeps its current position)."""
        with self._cond:
            self._pending = None
            self._cancel.set()

    def wait_idle(self, timeout: float = 5.0) -> bool:
        """Block until no gesture is running or pending (tests, shutdown)."""
        with self._cond:
            return self._cond.wait_for(
                lambda: self._current is None and self._pending is None, timeout
            )

    def close(self, recenter: bool = True) -> None:
        with self._cond:
            if self._closed:
                return
            self._closed = True
            self._pending = None
            self._cancel.set()
            self._cond.notify_all()
        if self._thread is not None:
            self._thread.join(timeout=2)
        if recenter:
            self._write(0.0, 0.0)

    # -- worker -----------------------------------------------------------------------
    def _ensure_thread(self) -> None:
        if self._thread is None or not self._thread.is_alive():
            self._thread = threading.Thread(target=self._worker, name="roco-head", daemon=True)
            self._thread.start()

    def _worker(self) -> None:
        while True:
            with self._cond:
                self._cond.wait_for(lambda: self._pending is not None or self._closed)
                if self._closed:
                    self._current = None
                    self._cond.notify_all()
                    return
                gesture, self._pending = self._pending, None
                assert gesture is not None
                self._cancel.clear()
                self._current = gesture.name
                self.history.append(gesture.name)
            log.info("head gesture: %s", gesture.name)
            if self.on_gesture is not None:
                try:
                    self.on_gesture(gesture.name)
                except Exception:
                    log.exception("on_gesture callback failed")
            try:
                self._run(gesture)
            except Exception:
                log.exception("head gesture %s failed", gesture.name)
            with self._cond:
                self._current = None
                if (
                    self._idle
                    and self._pending is None
                    and not self._closed
                    and gesture.name != "look_around"
                ):
                    self._pending = GESTURES["look_around"]
                self._cond.notify_all()

    def _run(self, gesture: Gesture) -> None:
        if gesture.generator is not None:
            # Let the face settle first so look-around doesn't fight a fresh gesture.
            if self._cancel.wait(self.idle_delay_s * self.time_scale):
                return
            frames: Iterator[Keyframe] = gesture.generator(self.rng)
        else:
            frames = iter(gesture.keyframes)
        for frame in frames:
            target = (self._limit_pan(frame.pan), self._limit_tilt(frame.tilt))
            if not self._move(target, frame.seconds * self.time_scale):
                return
            if frame.hold > 0 and self._cancel.wait(frame.hold * self.time_scale):
                return

    def _move(self, target: tuple[float, float], seconds: float) -> bool:
        """Step towards ``target``; timed by the clock so the OS timer granularity
        (~15 ms on Windows) can't stretch a gesture. False if cancelled."""
        start, t0 = self._pos, self.clock()
        while True:
            progress = 1.0 if seconds <= 0 else (self.clock() - t0) / seconds
            if self._cancel.is_set():
                return False
            self._write(*motion_point(start, target, progress))
            if progress >= 1.0:
                return True
            if self._cancel.wait(self.step_s):
                return False

    def _limit_pan(self, offset: float) -> float:
        return self._offset_within(self.pan, offset)

    def _limit_tilt(self, offset: float) -> float:
        return self._offset_within(self.tilt, offset)

    @staticmethod
    def _offset_within(axis: AxisConfig, offset: float) -> float:
        """Clamp an offset so that center+offset stays inside the axis limits."""
        return (axis.angle_for(offset) - axis.center) * (-1 if axis.invert else 1)

    def _write(self, pan: float, tilt: float) -> None:
        self._pos = (pan, tilt)
        self.servos.set_angle(self.pan.channel, self.pan.angle_for(pan))
        self.servos.set_angle(self.tilt.channel, self.tilt.angle_for(tilt))
