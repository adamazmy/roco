"""Playback of streamed TTS audio, with an instant local cut for barge-in.

The audio device *pulls*: :class:`CallbackOutput` runs a PortAudio callback stream
that asks :meth:`Player.pull` for the next block (silence when nothing is queued).
That design gives the body three things a blocking ``write()`` loop cannot:

- **cut in milliseconds**: :meth:`Player.cut` drops the queued audio under a lock,
  then the output's ``flush`` aborts the host buffer; the next callback plays silence;
- **an exact AEC reference**: every block handed to the device is also pushed to
  the :class:`roco.voice.aec.EchoReference`, in the order and at the time it plays;
- **truthful bookkeeping**: "first audio played", "which sentences were heard" and
  the mouth amplitude all come from what was actually pulled, not what was queued.

The callback thread only copies samples and appends small events to a queue; a
monitor thread turns those into :class:`PlayerListener` calls and mouth updates, so
nothing slow (bus handlers, network sends) ever runs on the audio thread.
"""

from __future__ import annotations

import contextlib
import logging
import queue
import threading
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Protocol

from roco.voice.audio import frame_rms, mouth_from_rms

log = logging.getLogger(__name__)

Clock = Callable[[], float]


class PlayerListener(Protocol):
    def on_audio_started(self, turn: int, t: float) -> None:
        """First sample of ``turn`` handed to the device at monotonic time ``t``."""

    def on_turn_drained(self, turn: int) -> None:
        """Every sample of a finished ``turn`` has been handed to the device."""

    def on_mouth(self, amplitude: float) -> None: ...


@dataclass
class _Segment:
    turn: int
    sentence: int
    data: bytes  # PCM16 mono at the player's rate
    end: bool  # last segment of this sentence
    pos: int = 0  # bytes already played


@dataclass
class _TurnState:
    texts: dict[int, str] = field(default_factory=dict)
    heard: list[int] = field(default_factory=list)
    queued: int = 0  # segments still queued
    started: bool = False
    finished: bool = False  # no more audio will come
    drained: bool = False


class Player:
    """Queue of TTS segments per turn, pulled by the audio device.

    Pure Python on bytes (numpy only to convert a foreign sample rate), so the
    turn logic is tested in text-mode CI.
    """

    def __init__(
        self,
        rate: int = 22050,
        listener: PlayerListener | None = None,
        clock: Clock = time.monotonic,
        mouth_interval_s: float = 0.04,
    ) -> None:
        self.rate = rate
        self.listener = listener
        self.clock = clock
        self.mouth_interval_s = mouth_interval_s
        self.output: Any = None  # CallbackOutput (or a fake); flush() on cut
        self._lock = threading.Lock()
        self._segments: deque[_Segment] = deque()
        self._turns: dict[int, _TurnState] = {}
        self._cancelled: set[int] = set()
        self._events: queue.Queue[tuple[str, int, float] | None] = queue.Queue()
        self._mouth = 0.0
        self._mouth_dirty = False
        self._last_block = b""
        self._thread: threading.Thread | None = None
        self._closed = False

    # ------------------------------------------------------------- receiving side
    def set_text(self, turn: int, index: int, text: str) -> None:
        with self._lock:
            if turn in self._cancelled:
                return
            self._turn(turn).texts[index] = text

    def feed(self, turn: int, sentence: int, pcm: bytes, rate: int, end: bool = False) -> bool:
        """Queue TTS audio. Returns False when the turn was cancelled (late frame)."""
        data = bytes(pcm)
        if rate != self.rate and data:
            import numpy as np

            from roco.voice.audio import resample_int16

            data = resample_int16(np.frombuffer(data, dtype=np.int16), rate, self.rate).tobytes()
        with self._lock:
            if turn in self._cancelled:
                return False
            state = self._turn(turn)
            if state.finished:
                log.debug("audio for finished turn %d ignored", turn)
                return False
            state.queued += 1
            self._segments.append(_Segment(turn, sentence, data, end))
        return True

    def finish(self, turn: int) -> None:
        """No more audio will come for ``turn`` (turn_done / local reply complete)."""
        with self._lock:
            if turn in self._cancelled:
                return
            state = self._turn(turn)
            state.finished = True
            if state.queued == 0 and not state.drained:
                state.drained = True
                self._events.put(("drained", turn, self.clock()))

    def cut(self, turn: int | None = None) -> str:
        """Stop ``turn`` (None = everything) now. Returns the text Adam heard.

        Later frames for a cut turn are dropped by :meth:`feed`. The output's host
        buffer is flushed too, so the speaker goes quiet within one callback.
        """
        with self._lock:
            victims = {s.turn for s in self._segments} | set(self._turns)
            if turn is not None:
                victims = {turn}
            self._segments = deque(s for s in self._segments if s.turn not in victims)
            heard = []
            for t in victims:
                state = self._turns.pop(t, None)
                self._cancelled.add(t)
                if state and t == turn:
                    heard = [state.texts.get(i, "") for i in state.heard]
            self._last_block, self._mouth, self._mouth_dirty = b"", 0.0, True
        output = self.output
        if output is not None:
            try:
                output.flush()
            except Exception:
                log.exception("flushing the audio output failed")
        return " ".join(h for h in heard if h).strip()

    def heard(self, turn: int) -> str:
        with self._lock:
            state = self._turns.get(turn)
            if state is None:
                return ""
            return " ".join(state.texts.get(i, "") for i in state.heard if state.texts.get(i))

    def forget(self, turn: int) -> None:
        """Drop the bookkeeping of a completed turn."""
        with self._lock:
            self._turns.pop(turn, None)

    def is_cancelled(self, turn: int) -> bool:
        with self._lock:
            return turn in self._cancelled

    @property
    def active(self) -> bool:
        """Audio queued or a turn still expected to produce audio."""
        with self._lock:
            return bool(self._segments) or any(
                not s.drained for s in self._turns.values()
            )

    @property
    def playing(self) -> bool:
        """Samples are queued right now (the speaker is producing ROCO's voice)."""
        with self._lock:
            return bool(self._segments)

    # ---------------------------------------------------------------- device side
    def pull(self, n: int) -> bytes:
        """Next ``n`` samples (PCM16 bytes) for the device, silence when idle.

        Runs on the audio thread: only copies bytes and queues events.
        """
        want = 2 * n
        parts: list[bytes] = []
        filled = 0
        now = None
        with self._lock:
            while filled < want and self._segments:
                seg = self._segments[0]
                state = self._turns.get(seg.turn)
                if state is not None and not state.started:
                    state.started = True
                    now = now if now is not None else self.clock()
                    self._events.put(("started", seg.turn, now))
                k = min(want - filled, len(seg.data) - seg.pos)
                parts.append(seg.data[seg.pos : seg.pos + k])
                seg.pos += k
                filled += k
                if seg.pos >= len(seg.data):
                    self._segments.popleft()
                    if state is not None:
                        state.queued -= 1
                        if seg.end and seg.sentence not in state.heard:
                            state.heard.append(seg.sentence)
                        if state.finished and state.queued == 0 and not state.drained:
                            state.drained = True
                            self._events.put(("drained", seg.turn, self.clock()))
            if filled:
                self._last_block = parts[0] if len(parts) == 1 else b"".join(parts)
                self._mouth_dirty = True
            elif self._last_block:
                self._last_block, self._mouth_dirty = b"", True
        if filled < want:
            parts.append(bytes(want - filled))
        return parts[0] if len(parts) == 1 else b"".join(parts)

    # ------------------------------------------------------------------ monitor
    def start(self) -> None:
        if self._thread is None:
            self._thread = threading.Thread(target=self._monitor, name="roco-player",
                                            daemon=True)
            self._thread.start()

    def close(self) -> None:
        self._closed = True
        self._events.put(None)
        if self._thread is not None:
            self._thread.join(timeout=2)

    def dispatch_pending(self) -> None:
        """Deliver queued events now (tests drive the player without its thread)."""
        while True:
            try:
                event = self._events.get_nowait()
            except queue.Empty:
                break
            if event is not None:
                self._deliver(event)
        self._publish_mouth()

    def _monitor(self) -> None:
        while not self._closed:
            try:
                event = self._events.get(timeout=self.mouth_interval_s)
            except queue.Empty:
                event = None
            if event is not None:
                self._deliver(event)
            self._publish_mouth()

    def _deliver(self, event: tuple[str, int, float]) -> None:
        kind, turn, t = event
        if self.listener is None:
            return
        try:
            if kind == "started":
                self.listener.on_audio_started(turn, t)
            elif kind == "drained":
                self.listener.on_turn_drained(turn)
        except Exception:
            log.exception("player listener failed on %s", kind)

    def _publish_mouth(self) -> None:
        with self._lock:
            if not self._mouth_dirty:
                return
            block, self._mouth_dirty = self._last_block, False
        value = mouth_from_rms(frame_rms(block)) if block else 0.0  # off the audio thread
        if self.listener is not None:
            try:
                self.listener.on_mouth(value)
            except Exception:
                log.exception("mouth update failed")

    def _turn(self, turn: int) -> _TurnState:
        state = self._turns.get(turn)
        if state is None:
            state = self._turns[turn] = _TurnState()
        return state


class CallbackOutput:
    """sounddevice callback stream that pulls from a :class:`Player`.

    Runs for the life of the body (silence when idle) so the AEC reference stays a
    continuous, sample-aligned stream. ``flush()`` aborts and restarts the stream,
    dropping whatever the host API had buffered: that is the barge-in cut.
    """

    def __init__(
        self,
        player: Player,
        device: int | None = None,
        reference: Any = None,
        latency: str | float = "low",
        block_ms: int = 20,
    ) -> None:
        self.player = player
        self.device = device
        self.reference = reference
        self.latency_setting = latency
        self.block_ms = block_ms
        self.underflows = 0
        self._stream: Any = None
        self._lock = threading.Lock()
        player.output = self

    def start(self) -> None:
        import sounddevice as sd

        with self._lock:
            if self._stream is None:
                rate = self.player.rate
                self._stream = sd.RawOutputStream(
                    samplerate=rate, channels=1, dtype="int16", device=self.device,
                    blocksize=rate * self.block_ms // 1000, latency=self.latency_setting,
                    callback=self._callback,
                )
            if not self._stream.active:
                self._stream.start()

    @property
    def latency_s(self) -> float:
        stream = self._stream
        try:
            return float(stream.latency) if stream is not None else 0.0
        except Exception:
            return 0.0

    def _callback(self, outdata: Any, frames: int, time_info: Any, status: Any) -> None:
        if status and status.output_underflow:
            self.underflows += 1
        block = self.player.pull(frames)
        outdata[:] = block
        ref = self.reference
        if ref is not None:
            ref.push(block, self.player.rate)

    def flush(self) -> None:
        with self._lock:
            stream = self._stream
            if stream is None:
                return
            with contextlib.suppress(Exception):
                stream.abort()
            stream.start()

    def reopen(self, rate: int) -> None:
        """Switch to another sample rate (a brain with a different TTS)."""
        self.close()
        self.player.rate = rate
        self.start()

    def close(self) -> None:
        with self._lock:
            stream, self._stream = self._stream, None
        if stream is not None:
            with contextlib.suppress(Exception):
                stream.abort()
            with contextlib.suppress(Exception):
                stream.close()
