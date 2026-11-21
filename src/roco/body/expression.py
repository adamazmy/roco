"""Pure face-expression logic: emotion shapes, easing, blinking and the animator.

Nothing here imports pygame, so it is fully testable in CI. :mod:`roco.body.face`
turns the :class:`FaceFrame` produced by :class:`FaceAnimator` into pixels.

Geometry conventions (all relative, the renderer scales them to the window):

- ``EyeShape.width/height`` multiply the base eye size.
- ``upper``/``lower`` are the fraction of the eye's height hidden by that eyelid,
  measured at the eye's centre line.
- Lid tilts are in degrees and expressed relative to the *outer* corner, so a
  table entry is mirrored automatically for the other eye: ``upper_tilt > 0``
  drops the upper lid's outer corner (sad droop), ``lower_tilt > 0`` raises the
  lower lid's outer corner.
- ``lower_curve`` bulges the lower lid upward in the middle (happy "^ ^" eyes).
- ``dy`` moves the eye vertically by a fraction of the base eye height (- = up).
- ``gaze_x/gaze_y`` are in -1..1 and move both eyes together (- = left / up).
"""

from __future__ import annotations

import math
import random
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field, fields, replace
from typing import TypeVar

from roco.bus import Emotion, State

T = TypeVar("T")

Clock = Callable[[], float]


# --------------------------------------------------------------------------- easing


def clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return lo if value < lo else hi if value > hi else value


def lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def ease_in_out_cubic(t: float) -> float:
    t = clamp(t)
    return 4 * t * t * t if t < 0.5 else 1 - (-2 * t + 2) ** 3 / 2


def approach(current: float, target: float, dt: float, rate: float) -> float:
    """Exponential smoothing that behaves the same at any frame rate.

    ``rate`` is in 1/s: after ``1/rate`` seconds ~63 % of the gap is closed.
    """
    if dt <= 0:
        return current
    return target + (current - target) * math.exp(-rate * dt)


def lerp_dataclass(a: T, b: T, t: float) -> T:
    """Field-wise interpolation of two instances of the same (nested) float dataclass."""
    values = {}
    for f in fields(a):  # type: ignore[arg-type]
        va, vb = getattr(a, f.name), getattr(b, f.name)
        if isinstance(va, float | int) and not isinstance(va, bool):
            values[f.name] = lerp(float(va), float(vb), t)
        elif hasattr(va, "__dataclass_fields__"):
            values[f.name] = lerp_dataclass(va, vb, t)
        else:
            values[f.name] = vb if t >= 0.5 else va
    return replace(a, **values)  # type: ignore[type-var]


# --------------------------------------------------------------------------- shapes


@dataclass(frozen=True)
class EyeShape:
    width: float = 1.0
    height: float = 1.0
    upper: float = 0.0
    upper_tilt: float = 0.0
    lower: float = 0.0
    lower_tilt: float = 0.0
    lower_curve: float = 0.0
    dy: float = 0.0


@dataclass(frozen=True)
class FaceParams:
    """Everything that eases between expressions."""

    left: EyeShape = field(default_factory=EyeShape)  # screen-left eye
    right: EyeShape = field(default_factory=EyeShape)
    gaze_x: float = 0.0
    gaze_y: float = 0.0
    openness: float = 1.0  # 0 = eyes closed (sleeping)
    scale: float = 1.0  # listening widens the eyes a little
    glow: float = 0.0  # listening halo, 0..1
    mouth: float = 0.0  # mouth visibility, 0..1 (speaking)
    bounce: float = 0.0  # vertical bob amplitude, 0..1 (excited)
    brightness: float = 1.0


def _both(shape: EyeShape, **extra: float) -> FaceParams:
    return FaceParams(left=shape, right=shape, **extra)


EMOTIONS: dict[Emotion, FaceParams] = {
    Emotion.NEUTRAL: _both(EyeShape()),
    Emotion.HAPPY: _both(
        EyeShape(height=0.95, lower=0.24, lower_curve=0.3, lower_tilt=4, dy=-0.04)
    ),
    Emotion.EXCITED: _both(
        EyeShape(width=1.1, height=1.08, lower=0.18, lower_curve=0.34, dy=-0.07),
        bounce=1.0,
    ),
    Emotion.CURIOUS: FaceParams(
        left=EyeShape(width=0.94, height=0.84, upper=0.24, upper_tilt=-8, lower=0.08),
        right=EyeShape(width=1.1, height=1.14, dy=-0.05),
        gaze_x=0.35,
        gaze_y=-0.1,
    ),
    Emotion.CONFUSED: FaceParams(
        left=EyeShape(height=1.08, upper=0.04, upper_tilt=10, dy=-0.06),
        right=EyeShape(
            width=0.95, height=0.9, upper=0.3, upper_tilt=-16, lower=0.14, lower_tilt=-8, dy=0.04
        ),
        gaze_x=-0.2,
    ),
    Emotion.SAD: _both(
        EyeShape(height=0.88, upper=0.2, upper_tilt=17, dy=0.1),
        gaze_y=0.35,
        brightness=0.85,
    ),
    Emotion.SLEEPY: _both(
        EyeShape(height=0.92, upper=0.5, upper_tilt=4, lower=0.1, dy=0.08),
        gaze_y=0.2,
        brightness=0.8,
    ),
    Emotion.SURPRISED: _both(EyeShape(width=1.14, height=1.24, dy=-0.06)),
}


def target_params(state: State, emotion: Emotion) -> FaceParams:
    """Combine the emotion's shape with the modifiers of the conversational state."""
    p = EMOTIONS[emotion]
    if state is State.LISTENING:
        return replace(p, scale=1.07, glow=1.0, gaze_x=p.gaze_x * 0.3, gaze_y=p.gaze_y * 0.3)
    if state is State.THINKING:
        # Eyes glance up and aside, slightly narrowed as if pondering.
        def ponder(e: EyeShape) -> EyeShape:
            return replace(e, height=e.height * 0.92, upper=max(e.upper, 0.06))

        return replace(
            p, left=ponder(p.left), right=ponder(p.right), gaze_x=0.5, gaze_y=-0.8, bounce=0.0
        )
    if state is State.SPEAKING:
        return replace(p, mouth=1.0)
    if state is State.SLEEPING:
        return replace(EMOTIONS[Emotion.NEUTRAL], openness=0.0, gaze_y=0.3, brightness=0.6)
    return p


# --------------------------------------------------------------------------- blinking


class BlinkScheduler:
    """Random blinks every ``min_interval``..``max_interval`` seconds.

    Pure: time comes in through ``amount(now)``; randomness from the injected ``rng``.
    """

    def __init__(
        self,
        rng: random.Random | None = None,
        min_interval: float = 3.0,
        max_interval: float = 6.0,
        duration: float = 0.16,
        double_chance: float = 0.15,
    ) -> None:
        if not 0 < min_interval <= max_interval:
            raise ValueError("need 0 < min_interval <= max_interval")
        self.rng = rng or random.Random()
        self.min_interval = min_interval
        self.max_interval = max_interval
        self.duration = duration
        self.double_chance = double_chance
        self.next_blink: float | None = None

    def reset(self, now: float) -> None:
        """Schedule the next blink a full interval from ``now``."""
        self.next_blink = now + self.rng.uniform(self.min_interval, self.max_interval)

    def amount(self, now: float) -> float:
        """0 = eyes open, 1 = fully closed."""
        if self.next_blink is None:
            self.reset(now)
        assert self.next_blink is not None
        elapsed = now - self.next_blink
        if elapsed < 0:
            return 0.0
        if elapsed >= self.duration:
            # Blink finished (possibly several if frames were skipped): plan the next one.
            end = self.next_blink + self.duration
            if self.rng.random() < self.double_chance:
                self.next_blink = end + 0.09
            else:
                self.next_blink = end + self.rng.uniform(self.min_interval, self.max_interval)
            return self.amount(now) if now >= self.next_blink else 0.0
        t = elapsed / self.duration
        # Close fast (40 %), open a little slower (60 %).
        return ease_in_out_cubic(t / 0.4) if t < 0.4 else 1 - ease_in_out_cubic((t - 0.4) / 0.6)


class IdleGaze:
    """Small random eye darts while idle, so the face never looks frozen."""

    def __init__(self, rng: random.Random | None = None) -> None:
        self.rng = rng or random.Random()
        self.x = 0.0
        self.y = 0.0
        self._next = 0.0

    def update(self, now: float) -> tuple[float, float]:
        if now >= self._next:
            if self.rng.random() < 0.45:
                self.x, self.y = 0.0, 0.0
            else:
                self.x = self.rng.uniform(-0.5, 0.5)
                self.y = self.rng.uniform(-0.25, 0.25)
            self._next = now + self.rng.uniform(1.2, 4.0)
        return self.x, self.y


# --------------------------------------------------------------------------- animator


@dataclass(frozen=True)
class FaceFrame:
    """A fully resolved frame: what the renderer draws, in relative units."""

    params: FaceParams
    blink: float  # 0..1, multiplies eye openness down
    mouth_open: float  # 0..1 smoothed amplitude
    offset_x: float  # extra gaze from idle darts, -1..1
    offset_y: float
    bob: float  # vertical bob in base-eye-heights (excited bounce / sleep breathing)
    glow_pulse: float  # 0..1
    time: float
    sleeping: bool
    state: State
    emotion: Emotion

    def eye_openness(self) -> float:
        return clamp(self.params.openness * (1.0 - self.blink))


class FaceAnimator:
    """Holds targets (thread-safe setters) and eases towards them in :meth:`update`.

    ``clock`` and ``rng`` are injectable so tests can drive time deterministically.
    """

    def __init__(
        self,
        clock: Clock = time.monotonic,
        rng: random.Random | None = None,
        transition_s: float = 0.28,
    ) -> None:
        self.clock = clock
        self.rng = rng or random.Random()
        self.transition_s = transition_s
        self.blinker = BlinkScheduler(self.rng)
        self.idle_gaze = IdleGaze(self.rng)
        self._lock = threading.Lock()
        self._state = State.IDLE
        self._emotion = Emotion.NEUTRAL
        self._mouth_target = 0.0
        self._mouth = 0.0
        self._gaze = (0.0, 0.0)
        self._last: float | None = None
        start = self.clock()
        self._from = target_params(self._state, self._emotion)
        self._to = self._from
        self._t0 = start

    # -- thread-safe setters -----------------------------------------------------------
    def set_state(self, state: State) -> None:
        with self._lock:
            if state is not self._state:
                self._retarget(state, self._emotion)

    def set_emotion(self, emotion: Emotion) -> None:
        with self._lock:
            if emotion is not self._emotion:
                self._retarget(self._state, emotion)

    def set_mouth(self, amplitude: float) -> None:
        self._mouth_target = clamp(float(amplitude))  # atomic assignment

    @property
    def state(self) -> State:
        return self._state

    @property
    def emotion(self) -> Emotion:
        return self._emotion

    def _retarget(self, state: State, emotion: Emotion) -> None:
        now = self.clock()
        self._from = self._current(now)  # start from wherever we are mid-transition
        self._to = target_params(state, emotion)
        self._t0 = now
        self._state, self._emotion = state, emotion

    def _current(self, now: float) -> FaceParams:
        duration = self.transition_s
        if self._to.openness < self._from.openness:
            duration *= 2.5  # falling asleep is slow
        t = 1.0 if duration <= 0 else ease_in_out_cubic((now - self._t0) / duration)
        return self._to if t >= 1.0 else lerp_dataclass(self._from, self._to, t)

    # -- per frame ---------------------------------------------------------------------
    def update(self) -> FaceFrame:
        now = self.clock()
        dt = 0.0 if self._last is None else max(0.0, min(now - self._last, 0.25))
        self._last = now
        with self._lock:
            params = self._current(now)
            state, emotion = self._state, self._emotion

        sleeping = state is State.SLEEPING
        if sleeping:
            blink = 0.0
            self.blinker.reset(now)
        else:
            blink = self.blinker.amount(now)

        # Mouth: open fast, close a bit slower, so speech looks lively but not jittery.
        target = self._mouth_target if state is State.SPEAKING else 0.0
        rate = 30.0 if target > self._mouth else 14.0
        self._mouth = approach(self._mouth, target, dt, rate)

        darts = self.idle_gaze.update(now) if state is State.IDLE else (0.0, 0.0)
        gx = approach(self._gaze[0], darts[0], dt, 18.0)
        gy = approach(self._gaze[1], darts[1], dt, 18.0)
        self._gaze = (gx, gy)

        if sleeping:
            bob = 0.035 * math.sin(now * 2 * math.pi / 4.5)  # slow breathing
        else:
            bob = -0.05 * params.bounce * abs(math.sin(now * 2 * math.pi * 1.1))

        return FaceFrame(
            params=params,
            blink=blink,
            mouth_open=clamp(self._mouth),
            offset_x=gx,
            offset_y=gy,
            bob=bob,
            glow_pulse=0.5 + 0.5 * math.sin(now * 2 * math.pi / 1.6),
            time=now,
            sleeping=sleeping,
            state=state,
            emotion=emotion,
        )
