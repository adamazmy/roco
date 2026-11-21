"""``roco demo-face``: cycle the face through every emotion and state.

Keys: Space = pause/resume, Right/Left = next/previous step, Esc/Q = quit.
Servos are simulated (unless ``servos.backend`` says otherwise) and gestures are
logged and shown in the caption.

``python -m roco.body.demo --save DIR`` renders every emotion/state to PNG headlessly.
"""

from __future__ import annotations

import argparse
import logging
import math
import os
import random
import threading
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from roco.bus import Emotion, EventBus, State

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Step:
    state: State
    emotion: Emotion
    seconds: float = 3.5

    @property
    def label(self) -> str:
        return f"{self.emotion.value} · {self.state.value}"


SEQUENCE: tuple[Step, ...] = (
    *(Step(State.IDLE, e) for e in Emotion),
    Step(State.IDLE, Emotion.NEUTRAL, 5.0),
    Step(State.LISTENING, Emotion.NEUTRAL),
    Step(State.THINKING, Emotion.NEUTRAL),
    Step(State.SPEAKING, Emotion.NEUTRAL, 4.5),
    Step(State.SPEAKING, Emotion.HAPPY, 4.5),
    Step(State.SLEEPING, Emotion.NEUTRAL, 6.0),
)


def speech_envelope(t: float, rng: random.Random) -> float:
    """Fake speech loudness: syllable-rate wobble with word gaps, 0..1."""
    syllables = 0.5 + 0.5 * math.sin(t * 2 * math.pi * 4.3)
    words = 1.0 if math.sin(t * 2 * math.pi * 0.7) > -0.55 else 0.0
    return max(0.0, min(1.0, words * (0.25 + 0.65 * syllables) + rng.uniform(-0.12, 0.12)))


class DemoController:
    """Steps through :data:`SEQUENCE` on a worker thread, publishing on a bus."""

    def __init__(
        self, bus: EventBus, face: Any, head: Any = None, sequence: tuple[Step, ...] = SEQUENCE
    ) -> None:
        self.bus = bus
        self.face = face
        self.head = head
        self.sequence = sequence
        self.index = 0
        self.paused = False
        self._jump = threading.Event()
        self._rng = random.Random()

    def next(self, delta: int = 1) -> None:
        self.index = (self.index + delta) % len(self.sequence)
        self._jump.set()

    def toggle_pause(self) -> None:
        self.paused = not self.paused

    def run(self, stop: threading.Event) -> None:
        tick = 1 / 30
        while not stop.is_set():
            step = self.sequence[self.index]
            self._jump.clear()
            self.bus.set_emotion(step.emotion)
            self.bus.set_state(step.state)
            elapsed = 0.0
            while not stop.is_set() and not self._jump.is_set():
                if step.state is State.SPEAKING:
                    self.bus.set_mouth(speech_envelope(elapsed, self._rng))
                self._caption(step)
                stop.wait(tick)
                if not self.paused:
                    elapsed += tick
                if elapsed >= step.seconds:
                    self.index = (self.index + 1) % len(self.sequence)
                    break
            self.bus.set_mouth(0.0)

    def _caption(self, step: Step) -> None:
        parts = [f"{self.index + 1}/{len(self.sequence)}  {step.label}"]
        if self.head is not None:
            pan, tilt = self.head.angles
            gesture = self.head.current_gesture or "-"
            parts.append(f"pan {pan:5.1f}°  tilt {tilt:5.1f}°  gesture: {gesture}")
        if self.paused:
            parts.append("PAUSED")
        parts.append("space pause · ←/→ step · esc quit")
        self.face.set_label("   |   ".join(parts))


def run_demo(cfg: Mapping[str, Any] | None = None) -> None:
    import pygame

    from roco.body.face import PygameFace
    from roco.body.robot import create_body
    from roco.runtime import run_with_face

    controller: DemoController | None = None

    class DemoFace(PygameFace):
        def on_key(self, key: int) -> None:
            if controller is None:
                return
            if key == pygame.K_SPACE:
                controller.toggle_pause()
            elif key == pygame.K_RIGHT:
                controller.next(1)
            elif key == pygame.K_LEFT:
                controller.next(-1)

    face = DemoFace(cfg)
    bus = EventBus()
    body = create_body(face, cfg)
    body.attach(bus)
    controller = DemoController(bus, face, body.head)
    try:
        run_with_face(face, controller.run)
    finally:
        body.close()


# --------------------------------------------------------------------------- gallery


GALLERY: tuple[Step, ...] = (
    *(Step(State.IDLE, e) for e in Emotion),
    Step(State.LISTENING, Emotion.NEUTRAL),
    Step(State.THINKING, Emotion.NEUTRAL),
    Step(State.SPEAKING, Emotion.HAPPY),
    Step(State.SLEEPING, Emotion.NEUTRAL),
)


def save_gallery(
    out_dir: str | os.PathLike[str],
    cfg: Mapping[str, Any] | None = None,
    size: tuple[int, int] | None = None,
    labels: bool = True,
) -> list[Path]:
    """Render each gallery step, fully settled and mid-blink-free, to ``out_dir/*.png``."""
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    import pygame

    from roco.body.expression import FaceAnimator
    from roco.body.face import FaceSettings, PygameFace

    pygame.init()
    try:
        settings = FaceSettings.from_config(cfg)
        w, h = size or (settings.width, settings.height)
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        paths = []
        for step in GALLERY:
            clock = _FakeClock()
            animator = FaceAnimator(clock=clock, rng=random.Random(7))
            animator.blinker.next_blink = 1e9  # no blink in screenshots
            face = PygameFace(settings=settings, animator=animator)
            animator.set_emotion(step.emotion)
            animator.set_state(step.state)
            animator.set_mouth(0.75)
            frame = None
            for _ in range(60):  # 2 s at 30 fps: transitions settled
                clock.t += 1 / 30
                frame = animator.update()
            assert frame is not None
            if labels:
                face.set_label(step.label)
            surface = pygame.Surface((w, h))
            face.render(surface, frame)
            path = out / f"{step.emotion.value}-{step.state.value}.png"
            pygame.image.save(surface, str(path))
            paths.append(path)
        return paths
    finally:
        pygame.quit()


class _FakeClock:
    def __init__(self) -> None:
        self.t = 0.0

    def __call__(self) -> float:
        return self.t


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--save", metavar="DIR", help="render PNGs instead of opening a window")
    parser.add_argument("--size", default=None, help="WxH for --save, e.g. 800x480")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s: %(message)s")
    from roco.config import load_config

    cfg = load_config()
    if args.save:
        size = tuple(int(v) for v in args.size.lower().split("x")) if args.size else None
        for path in save_gallery(args.save, cfg, size):  # type: ignore[arg-type]
            print(path)
    else:
        run_demo(cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
