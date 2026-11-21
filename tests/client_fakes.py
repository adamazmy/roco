"""Fakes for the body-client tests: no numpy, audio devices or models."""

from __future__ import annotations

import time
from array import array

from roco.protocol import (
    AudioFlag,
    AudioFrame,
    AudioKind,
    ConnectionClosed,
    Hello,
    Welcome,
    memory_pipe,
)

SPEECH = 3000
QUIET = 100


def frame(level: int, n: int = 1280) -> array:
    return array("h", [level if i % 2 else -level for i in range(n)])


class FakeVad:
    """Probability 1.0 for loud frames, 0.0 otherwise; one 80 ms score per frame."""

    name = "fake"

    def __init__(self, threshold: int = 1000) -> None:
        self.threshold = threshold

    def scores(self, f) -> list[tuple[float, float]]:
        return [(1.0 if abs(f[1]) >= self.threshold else 0.0, 80.0)]

    def reset(self) -> None:
        pass


class FakeWake:
    """Fires on frames whose level is exactly ``trigger``."""

    def __init__(self, trigger: int = 7777) -> None:
        self.trigger = trigger
        self.resets = 0

    def score(self, f) -> float:
        return 1.0 if abs(f[1]) == self.trigger else 0.0

    def reset(self) -> None:
        self.resets += 1


class FakeClock:
    def __init__(self, t: float = 1000.0) -> None:
        self.t = t

    def __call__(self) -> float:
        return self.t

    def advance(self, s: float) -> None:
        self.t += s


class FakeOutput:
    """Stands in for CallbackOutput: records flushes, can advance a fake clock."""

    def __init__(self, clock: FakeClock | None = None, flush_s: float = 0.0) -> None:
        self.flushes = 0
        self.clock = clock
        self.flush_s = flush_s

    def flush(self) -> None:
        self.flushes += 1
        if self.clock is not None:
            self.clock.advance(self.flush_s)


class Brain:
    """The test plays the brain on the other end of a memory pipe."""

    def __init__(self) -> None:
        self.ends = []  # brain-side ends, newest last
        self.refuse = None  # an Error to answer hello with
        self.fail_connect = False

    def connect(self):
        if self.fail_connect:
            raise ConnectionRefusedError("brain is down")
        body_end, brain_end = memory_pipe()
        self.ends.append(brain_end)
        return body_end

    @property
    def end(self):
        return self.ends[-1]

    def recv(self, timeout: float = 2.0, skip=()):
        """Next frame the body sent (skipping types in ``skip``)."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            try:
                f = self.end.recv(timeout=0.05)
            except ConnectionClosed:
                return None
            if f is not None and not isinstance(f, skip):
                return f
        return None

    def drain(self, timeout: float = 0.2, skip=()) -> list:
        out = []
        while (f := self.recv(timeout, skip)) is not None:
            out.append(f)
        return out

    def accept(self, tts_rate: int = 16000) -> Hello:
        hello = self.recv()
        assert isinstance(hello, Hello), hello
        self.end.send(self.refuse or Welcome(protocol=1, brain="fake", tts_rate=tts_rate))
        return hello

    def say(self, turn: int, index: int, pcm: bytes, rate: int = 16000, end: bool = True,
            seq: int = 0):
        self.end.send(AudioFrame(kind=AudioKind.TTS, turn=turn, seq=seq, rate=rate, pcm=pcm,
                                 sentence=index, flags=AudioFlag.END if end else AudioFlag.NONE))


def wait_for(cond, timeout: float = 2.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if cond():
            return True
        time.sleep(0.005)
    return cond()
