"""Face, ServoBackend and Body interfaces."""

from __future__ import annotations

import threading
from abc import ABC, abstractmethod
from collections.abc import Callable

from roco.bus import Emotion, EventBus, State, Topic


class Face(ABC):
    """The animated face. Owns the main thread while :meth:`run` executes.

    Contract for implementations:

    - Setters are called from other threads; they only store target values
      (under a lock or as atomic assignments). All drawing happens in ``run``.
    - ``run`` blocks until the window is closed or :meth:`stop` is called, then
      returns. It must be called from the main thread (pygame requirement).
    """

    @abstractmethod
    def run(self) -> None: ...

    @abstractmethod
    def stop(self) -> None:
        """Ask ``run`` to return soon. Thread-safe."""

    @abstractmethod
    def set_state(self, state: State) -> None: ...

    @abstractmethod
    def set_emotion(self, emotion: Emotion) -> None: ...

    @abstractmethod
    def set_mouth(self, amplitude: float) -> None:
        """0 = closed, 1 = fully open."""


class NullFace(Face):
    """Headless face: records the latest values; ``run`` just waits for ``stop``."""

    def __init__(self) -> None:
        self.state = State.IDLE
        self.emotion = Emotion.NEUTRAL
        self.mouth = 0.0
        self._stopped = threading.Event()

    def run(self) -> None:
        self._stopped.wait()

    def stop(self) -> None:
        self._stopped.set()

    def set_state(self, state: State) -> None:
        self.state = state

    def set_emotion(self, emotion: Emotion) -> None:
        self.emotion = emotion

    def set_mouth(self, amplitude: float) -> None:
        self.mouth = amplitude


class ServoBackend(ABC):
    """Low-level servo output. Angles in degrees, 0..180.

    Backends do no safety limiting of their own: the head controller (workstream C)
    clamps to the per-axis ``min``/``max`` from config before calling ``set_angle``.
    """

    @abstractmethod
    def set_angle(self, channel: int, angle: float) -> None: ...

    @abstractmethod
    def get_angle(self, channel: int) -> float | None:
        """Last commanded angle, or None if never set."""

    def release(self, channel: int) -> None:
        """Stop driving the servo (let it go limp). Optional."""

    def close(self) -> None:
        """Release hardware. Safe to call twice."""


class Body:
    """Maps bus events onto the face and head gestures.

    The base class does the bus wiring; subclasses override the ``on_*`` hooks
    (e.g. happy -> nod, confused -> curious tilt). Handlers run in the publisher's
    thread, so they must be quick and non-blocking (gestures run on their own thread).
    """

    def __init__(self, face: Face, servos: ServoBackend | None = None) -> None:
        self.face = face
        self.servos = servos
        self._unsubscribers: list[Callable[[], None]] = []

    def attach(self, bus: EventBus) -> None:
        self._unsubscribers += [
            bus.subscribe(Topic.STATE, self.on_state),
            bus.subscribe(Topic.EMOTION, self.on_emotion),
            bus.subscribe(Topic.MOUTH, self.on_mouth),
        ]

    def detach(self) -> None:
        for unsubscribe in self._unsubscribers:
            unsubscribe()
        self._unsubscribers.clear()

    def on_state(self, state: State) -> None:
        self.face.set_state(state)

    def on_emotion(self, emotion: Emotion) -> None:
        self.face.set_emotion(emotion)

    def on_mouth(self, amplitude: float) -> None:
        self.face.set_mouth(amplitude)

    def close(self) -> None:
        self.detach()
        if self.servos is not None:
            self.servos.close()
