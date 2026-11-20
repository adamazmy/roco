"""Rendering smoke tests: only run where pygame is installed (not in CI)."""

from __future__ import annotations

import os
import random
import threading

import pytest

pygame = pytest.importorskip("pygame")

from roco.body.demo import SEQUENCE, DemoController, save_gallery  # noqa: E402
from roco.body.expression import FaceAnimator  # noqa: E402
from roco.body.face import FaceSettings, PygameFace  # noqa: E402
from roco.bus import Emotion, EventBus, State  # noqa: E402


@pytest.fixture(autouse=True)
def dummy_video(monkeypatch):
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    pygame.init()
    yield
    pygame.quit()


def count_eye_pixels(surface) -> int:
    """Bright cyan-ish samples (brightness and glow shift the exact colour)."""
    w, h = surface.get_size()
    count = 0
    for x in range(0, w, 4):
        for y in range(0, h, 4):
            r, g, b = surface.get_at((x, y))[:3]
            count += r < 120 and g > 150 and b > 150
    return count


@pytest.mark.parametrize("state", list(State))
@pytest.mark.parametrize("emotion", list(Emotion))
def test_render_every_emotion_and_state(emotion, state):
    t = [0.0]
    animator = FaceAnimator(clock=lambda: t[0], rng=random.Random(1))
    animator.blinker.next_blink = 1e9
    face = PygameFace(settings=FaceSettings(), animator=animator)
    animator.set_emotion(emotion)
    animator.set_state(state)
    animator.set_mouth(0.8)
    for _ in range(45):
        t[0] += 1 / 30
        frame = animator.update()
    surface = pygame.Surface((800, 480))
    face.render(surface, frame)
    if state is not State.SLEEPING:
        assert count_eye_pixels(surface) > 500


def test_render_scales_to_any_size():
    face = PygameFace({"face": {"width": 320, "height": 240}})
    frame = face.animator.update()
    for size in [(320, 240), (1920, 1080), (50, 50)]:
        face.render(pygame.Surface(size), frame)


def test_run_loop_exits_on_stop():
    face = PygameFace({"face": {"fps": 120}})
    threading.Timer(0.3, face.stop).start()
    face.set_emotion(Emotion.HAPPY)
    face.set_label("smoke")
    face.run()  # returns once stop() is called


def test_gallery(tmp_path):
    paths = save_gallery(tmp_path, size=(400, 240))
    assert len(paths) >= len(Emotion) and all(p.stat().st_size > 0 for p in paths)
    assert os.environ["SDL_VIDEODRIVER"] == "dummy"


def test_demo_controller_publishes_and_stops():
    bus = EventBus()
    seen = []
    bus.subscribe("*", seen.append)
    face = PygameFace()
    stop = threading.Event()
    controller = DemoController(bus, face, sequence=SEQUENCE[-3:])  # includes speaking
    worker = threading.Thread(target=controller.run, args=(stop,))
    worker.start()
    threading.Event().wait(0.2)
    controller.next()
    threading.Event().wait(0.2)
    stop.set()
    worker.join(2)
    topics = {topic for topic, _ in seen}
    assert {"state", "emotion", "mouth"} <= topics
