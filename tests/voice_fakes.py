"""Fakes for the voice tests: no numpy, sounddevice or models needed."""

from __future__ import annotations

import threading
from array import array
from collections import deque

from roco.voice.base import Listener, Speaker

SILENCE = 100.0  # int16 RMS of a quiet room
SPEECH = 3000.0


def frame(level: float, n: int = 1280) -> array:
    """A square-wave frame whose RMS is exactly ``level``."""
    v = int(level)
    return array("h", [v if i % 2 else -v for i in range(n)])


class FakeSource:
    """Frame source fed from a script of (level, count) pairs; then returns None."""

    def __init__(self, script=()):
        self.frames: deque = deque()
        self.flushes = 0
        self.closed = False
        self.reads = 0
        self.on_read = None  # callback(reads) for injecting events mid-stream
        for level, count in script:
            self.add(level, count)

    def add(self, level: float, count: int = 1):
        for _ in range(count):
            self.frames.append(frame(level))
        return self

    def read(self, timeout: float):
        self.reads += 1
        if self.on_read:
            self.on_read(self.reads)
        if self.closed or not self.frames:
            return None
        return self.frames.popleft()

    def flush(self):
        self.flushes += 1

    def close(self):
        self.closed = True


class FakeWake:
    """Fires (score 1.0) on frames whose RMS is exactly ``trigger_level``."""

    def __init__(self, trigger_level: float | None = None):
        self.trigger_level = trigger_level
        self.resets = 0
        self.seen = 0

    def score(self, f) -> float:
        self.seen += 1
        if self.trigger_level is not None and abs(f[1]) == int(self.trigger_level):
            return 1.0
        return 0.0

    def reset(self):
        self.resets += 1


class FakeTranscriber:
    def __init__(self, text: str = "what time is it"):
        self.text = text
        self.calls: list[int] = []

    def __call__(self, frames) -> str:
        self.calls.append(len(frames))
        return self.text


class FakeTTS:
    """Each sentence -> one chunk of ``samples`` int16 samples at ``level``."""

    def __init__(self, rate: int = 1000, samples: int = 200, level: int = 3000):
        self.rate, self.samples, self.level = rate, samples, level
        self.calls: list[str] = []
        self.fail_on: set[str] = set()

    def __call__(self, text: str):
        self.calls.append(text)
        if text in self.fail_on:
            raise RuntimeError("boom")
        yield frame(self.level, self.samples).tobytes(), self.rate


class FakeOutput:
    def __init__(self, gate: threading.Event | None = None):
        self.blocks: list[tuple[bytes, int]] = []
        self.events: list[str] = []
        self.gate = gate  # when set, write() blocks until the gate opens
        self.closed = False

    def write(self, block: bytes, sample_rate: int) -> None:
        if self.gate is not None:
            self.gate.wait(5)
        self.blocks.append((block, sample_rate))
        self.events.append("write")

    def drain(self):
        self.events.append("drain")

    def abort(self):
        self.events.append("abort")

    def close(self):
        self.closed = True


class ScriptedListener(Listener):
    """Returns queued results from listen()/listen_followup(); records mute calls."""

    def __init__(self, wakes=(), followups=()):
        self.wakes = deque(wakes)
        self.followups = deque(followups)
        self.log: list[tuple] = []
        self.closed = threading.Event()
        self.last_timing = None

    def listen(self):
        self.log.append(("listen",))
        if self.wakes:
            return self.wakes.popleft()
        self.closed.wait(5)
        return None

    def listen_followup(self, timeout):
        self.log.append(("followup", timeout))
        return self.followups.popleft() if self.followups else None

    def set_muted(self, muted):
        self.log.append(("muted", muted))

    def close(self):
        self.closed.set()


class RecordingSpeaker(Speaker):
    def __init__(self, log: list | None = None):
        self.spoken: list[str] = []
        self.log = log if log is not None else []

    def say(self, text):
        if text and text.strip():
            self.spoken.append(text)
            self.log.append(("say", text))

    def wait_done(self, timeout=None):
        self.log.append(("wait_done",))
        return True

    @property
    def is_speaking(self):
        return False

    def stop(self):
        self.log.append(("stop",))
