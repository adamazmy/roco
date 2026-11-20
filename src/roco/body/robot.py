"""RobotBody: maps bus events onto the face and the head's gestures."""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Any

from roco.body.base import Body, Face, ServoBackend
from roco.body.head import HeadController
from roco.bus import Emotion, State

log = logging.getLogger(__name__)

# Emotion -> head gesture (None = leave the head alone).
EMOTION_GESTURES: dict[Emotion, str | None] = {
    Emotion.NEUTRAL: None,
    Emotion.HAPPY: "nod",
    Emotion.EXCITED: "nod",
    Emotion.CURIOUS: "tilt",
    Emotion.CONFUSED: "tilt",
    Emotion.SAD: "look_down",
    Emotion.SLEEPY: "look_down",
    Emotion.SURPRISED: "startle",
}

# State -> head gesture on entering that state.
STATE_GESTURES: dict[State, str | None] = {
    State.IDLE: None,  # handled by HeadController.set_idle (look-around)
    State.LISTENING: "center",  # face the user
    State.THINKING: None,
    State.SPEAKING: None,
    State.SLEEPING: "sleep",  # head down
}


class RobotBody(Body):
    """Face + optional pan/tilt head.

    Handlers only store values or queue a gesture, so they return immediately.
    """

    def __init__(
        self,
        face: Face,
        servos: ServoBackend | None = None,
        head: HeadController | None = None,
        cfg: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(face, servos)
        if head is None and servos is not None:
            head = HeadController.from_config(servos, cfg)
        self.head = head
        self.state = State.IDLE

    def on_state(self, state: State) -> None:
        super().on_state(state)
        previous, self.state = self.state, state
        if self.head is None or state is previous:
            return
        self.head.set_idle(state is State.IDLE)
        if gesture := STATE_GESTURES.get(state):
            self.head.play(gesture)

    def on_emotion(self, emotion: Emotion) -> None:
        super().on_emotion(emotion)
        if self.head is None or self.state is State.SLEEPING:
            return
        if gesture := EMOTION_GESTURES.get(emotion):
            self.head.play(gesture)

    def close(self) -> None:
        self.detach()
        if self.head is not None:
            self.head.close()
        super().close()


def create_body(face: Face, cfg: Mapping[str, Any] | None = None) -> RobotBody:
    """Face + the servo backend named in ``servos.backend``, ready to ``attach(bus)``."""
    from roco.body.servos import create_servo_backend

    servos = create_servo_backend(cfg)
    body = RobotBody(face, servos, cfg=cfg)
    if body.head is not None:
        body.head.set_idle(True)
    return body
