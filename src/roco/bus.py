"""In-process event bus that decouples the brain, voice and body.

Handlers run synchronously in the publisher's thread. A handler that needs to do
slow or main-thread-only work (pygame drawing, servo motion) must hand it off
itself, e.g. by storing the latest value for its own loop to pick up. A failing
handler is logged and never breaks the publisher or other handlers.

Core topics and payloads:

=========  ==================  ==============================================
topic      payload             meaning
=========  ==================  ==============================================
state      :class:`State`      idle | listening | thinking | speaking | sleeping
emotion    :class:`Emotion`    emotion parsed from the reply's leading tag
mouth      ``float`` 0..1      speech amplitude, published per audio block
announce   ``str``             text ROCO should say unprompted (reminders)
=========  ==================  ==============================================
"""

from __future__ import annotations

import logging
import threading
from collections import defaultdict
from collections.abc import Callable
from enum import StrEnum
from typing import Any

log = logging.getLogger(__name__)

Handler = Callable[[Any], None]
WILDCARD = "*"


class Topic(StrEnum):
    STATE = "state"
    EMOTION = "emotion"
    MOUTH = "mouth"
    ANNOUNCE = "announce"


class State(StrEnum):
    IDLE = "idle"
    LISTENING = "listening"
    THINKING = "thinking"
    SPEAKING = "speaking"
    SLEEPING = "sleeping"


class Emotion(StrEnum):
    NEUTRAL = "neutral"
    HAPPY = "happy"
    EXCITED = "excited"
    CURIOUS = "curious"
    CONFUSED = "confused"
    SAD = "sad"
    SLEEPY = "sleepy"
    SURPRISED = "surprised"

    @classmethod
    def parse(cls, value: str | None, default: Emotion | None = None) -> Emotion:
        """Lenient parse ("Happy", " happy ") falling back to ``default`` or NEUTRAL."""
        try:
            return cls((value or "").strip().lower())
        except ValueError:
            return default or cls.NEUTRAL


class EventBus:
    """Thread-safe publish/subscribe. Subscribe to ``"*"`` for every (topic, payload)."""

    def __init__(self) -> None:
        self._handlers: dict[str, list[Handler]] = defaultdict(list)
        self._lock = threading.RLock()

    def subscribe(self, topic: str, handler: Handler) -> Callable[[], None]:
        """Register ``handler`` for ``topic``. Returns a function that unsubscribes it."""
        with self._lock:
            self._handlers[str(topic)].append(handler)

        def unsubscribe() -> None:
            with self._lock:
                handlers = self._handlers.get(str(topic), [])
                if handler in handlers:
                    handlers.remove(handler)

        return unsubscribe

    def publish(self, topic: str, payload: Any = None) -> None:
        with self._lock:
            handlers = list(self._handlers.get(str(topic), ()))
            wildcard = list(self._handlers.get(WILDCARD, ()))
        for handler in handlers:
            self._dispatch(handler, payload, topic)
        for handler in wildcard:
            self._dispatch(handler, (str(topic), payload), topic)

    @staticmethod
    def _dispatch(handler: Handler, payload: Any, topic: str) -> None:
        try:
            handler(payload)
        except Exception:
            log.exception("Bus handler %r failed on topic %r", handler, str(topic))

    # Typed helpers so callers don't hand-roll topic strings.
    def set_state(self, state: State | str) -> None:
        self.publish(Topic.STATE, State(state))

    def set_emotion(self, emotion: Emotion | str) -> None:
        self.publish(Topic.EMOTION, Emotion.parse(str(emotion)))

    def set_mouth(self, amplitude: float) -> None:
        self.publish(Topic.MOUTH, min(1.0, max(0.0, float(amplitude))))

    def announce(self, text: str) -> None:
        if text and text.strip():
            self.publish(Topic.ANNOUNCE, text.strip())
