"""Fakes for the brain server tests: STT engine, TTS, scripted brain, a body on a memory pipe.

Mic audio convention for :class:`FakeEngine`: every 80 ms frame is one constant sample
value. A value >= 1000 is the spoken word ``w<value - 1000>``; 0 is silence.
"""

from __future__ import annotations

import struct
import threading
import time
from collections.abc import Callable
from concurrent.futures import Executor, Future
from typing import Any

from roco.brain.base import Brain, TurnHooks
from roco.bus import EventBus
from roco.config import load_config
from roco.protocol import (
    PROTOCOL_VERSION,
    AudioFrame,
    AudioKind,
    Connection,
    Hello,
    TurnDone,
    memory_pipe,
)
from roco.server.context import BrainContext, FillerLine
from roco.server.session import serve_connection
from roco.voice.stt import Segment

MIC_RATE = 16000
FRAME_SAMPLES = 1280  # 80 ms
TTS_RATE = 22050
TOKEN = "test-token"


def word_frame(i: int) -> bytes:
    return struct.pack("<h", 1000 + i) * FRAME_SAMPLES


def silence_frame() -> bytes:
    return b"\x00\x00" * FRAME_SAMPLES


class FakeEngine:
    """``segments(pcm, rate)``: one word per loud frame, grouped ``per_segment`` at a time."""

    def __init__(self, per_segment: int = 5, delay: float = 0.0) -> None:
        self.per_segment = per_segment
        self.delay = delay
        self.calls: list[int] = []  # length in bytes of each decoded buffer

    def segments(self, pcm: bytes, rate: int) -> list[Segment]:
        self.calls.append(len(pcm))
        if self.delay:
            time.sleep(self.delay)
        step = FRAME_SAMPLES * 2
        words: list[tuple[float, float, str]] = []
        for k in range(0, len(pcm) - step + 1, step):
            (value,) = struct.unpack_from("<h", pcm, k)
            if value >= 1000:
                t = k / 2 / rate
                words.append((t, t + FRAME_SAMPLES / rate, f"w{value - 1000}"))
        out = []
        for i in range(0, len(words), self.per_segment):
            group = words[i : i + self.per_segment]
            out.append(Segment(group[0][0], group[-1][1], " ".join(w[2] for w in group)))
        return out


class InlineExecutor(Executor):
    """Runs every job at once, on the caller's thread (deterministic tests)."""

    def submit(self, fn: Callable[..., Any], /, *args: Any, **kwargs: Any) -> Future[Any]:
        future: Future[Any] = Future()
        try:
            future.set_result(fn(*args, **kwargs))
        except Exception as exc:  # pragma: no cover
            future.set_exception(exc)
        return future


class FakeTTS:
    """200 ms of PCM per sentence (3 frames of 80 ms at 22050 Hz, the last one short)."""

    def __init__(self, seconds: float = 0.2, delay: float = 0.0, chunks: int = 1) -> None:
        self.seconds = seconds
        self.delay = delay  # before each chunk
        self.chunks = chunks
        self.said: list[str] = []

    def __call__(self, text: str):
        self.said.append(text)
        for _ in range(self.chunks):
            if self.delay:
                time.sleep(self.delay)
            yield b"\x01\x00" * int(TTS_RATE * self.seconds / self.chunks), TTS_RATE


class ScriptBrain(Brain):
    """Plays a script per turn. Steps: ("emotion", e), ("say", text), ("tools", [names]),
    ("sleep", s). Stops at the next step once ``cancel`` is set."""

    def __init__(self, bus: EventBus, script: list[tuple[str, Any]] | None = None) -> None:
        self.bus = bus
        self.script = script or [("emotion", "happy"), ("say", "Hello Adam."),
                                 ("say", "Nice to see you.")]
        self.heard: list[str] = []
        self.amended: list[str] = []
        self.forgotten: list[str] = []
        self.cancelled_at: float | None = None
        self.returned_at: float | None = None

    def respond(self, text: str, on_sentence=None) -> str:
        return self.respond_turn(text, on_sentence)

    def respond_turn(self, text, on_sentence=None, *, cancel=None, hooks=None) -> str:
        hooks = hooks or TurnHooks()
        self.heard.append(text)
        said: list[str] = []
        for kind, arg in self.script:
            if cancel is not None and cancel.is_set():
                break
            if kind == "emotion":
                self.bus.set_emotion(arg)
            elif kind == "say":
                if hooks.on_token:
                    hooks.on_token()
                said.append(arg)
                if on_sentence:
                    on_sentence(arg)
            elif kind == "tools":
                if hooks.on_tools:
                    hooks.on_tools(list(arg))
            elif kind == "sleep":
                end = time.monotonic() + arg
                while time.monotonic() < end and not (cancel and cancel.is_set()):
                    time.sleep(0.005)
        self.returned_at = time.monotonic()
        return " ".join(said)

    def amend_last_reply(self, heard: str) -> None:
        self.amended.append(heard)

    def forget_turn(self, user_text: str) -> None:
        self.forgotten.append(user_text)

    def reset(self) -> None:
        pass


def make_ctx(tmp_home, brain: Brain | None = None, bus: EventBus | None = None,
             engine: Any = None, tts: Any = None, executor: Executor | None = None,
             wake_word: str | None = None, **server_cfg: Any) -> BrainContext:
    bus = bus or EventBus()
    overrides: dict[str, Any] = {"server": server_cfg} if server_cfg else {}
    if wake_word:
        overrides["voice"] = {"wake_word": {"model": wake_word}}
    cfg = load_config(overrides=overrides or None)
    ctx = BrainContext(
        cfg=cfg,
        bus=bus,
        brain=brain or ScriptBrain(bus),
        stt=engine or FakeEngine(),
        tts=tts or FakeTTS(),
        tts_rate=TTS_RATE,
        token=TOKEN,
        brain_name="fake:brain",
        fillers=[FillerLine("Let me check.", ((b"\x02\x00" * 800, TTS_RATE),))],
        latency_path=tmp_home / "latency.jsonl",
    )
    if executor is not None:
        ctx.stt_executor = executor  # type: ignore[assignment]
    return ctx


class FakeBody:
    """The body end of a memory pipe, with the server running on a thread."""

    def __init__(self, ctx: BrainContext) -> None:
        self.conn, server_end = memory_pipe()
        self.stop = threading.Event()
        self.thread = threading.Thread(target=serve_connection, args=(server_end, ctx, self.stop),
                                       daemon=True)
        self.thread.start()

    def send(self, frame: Any) -> None:
        self.conn.send(frame)

    def hello(self, token: str = TOKEN, protocol: int = PROTOCOL_VERSION) -> Any:
        self.send(Hello(protocol=protocol, body_id="test-body", token=token))
        return self.conn.recv(timeout=5)

    def recv_until(self, done: Callable[[Any], bool], timeout: float = 5.0) -> list[Any]:
        out: list[Any] = []
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            frame = self.conn.recv(timeout=0.05)
            if frame is None:
                continue
            out.append(frame)
            if done(frame):
                return out
        raise AssertionError(f"timed out; got {[describe(f) for f in out]}")

    def until_turn_done(self, turn: int, timeout: float = 5.0) -> list[Any]:
        return self.recv_until(lambda f: isinstance(f, TurnDone) and f.turn == turn, timeout)

    def speak(self, turn: int, words: int, quiet_frames: int = 6) -> None:
        seq = 0
        for i in range(words):
            self.send(AudioFrame(AudioKind.MIC, turn, seq, MIC_RATE, word_frame(i)))
            seq += 1
        for _ in range(quiet_frames):
            self.send(AudioFrame(AudioKind.MIC, turn, seq, MIC_RATE, silence_frame()))
            seq += 1

    def close(self) -> None:
        self.stop.set()
        self.conn.close()
        self.thread.join(timeout=5)


def describe(frame: Any) -> str:
    if isinstance(frame, AudioFrame):
        return f"audio(s{frame.sentence}{' END' if frame.end else ''})"
    name = frame.TYPE
    text = getattr(frame, "text", None)
    return f"{name}({text!r})" if text is not None else name


def drain(conn: Connection, seconds: float) -> list[Any]:
    out = []
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        frame = conn.recv(timeout=0.02)
        if frame is not None:
            out.append(frame)
    return out
