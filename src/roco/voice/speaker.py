"""Speaker: a sentence queue -> TTS -> audio output pipeline on two threads.

``say()`` only queues. A synthesis thread turns sentences into audio (so the next
sentence is ready while the current one plays) and a playback thread writes it to
the output in small blocks, publishing a mouth amplitude per block.

Bus contract (see :class:`roco.voice.base.Speaker`): ``State.SPEAKING`` once when
audio starts, ``mouth`` per block, then ``mouth`` 0.0 and ``State.IDLE`` when
everything has been heard, unless someone else changed the state meanwhile.

The TTS and the output are injected so tests use fakes; :func:`create_speaker` in
:mod:`roco.voice.factory` wires Piper and sounddevice.
"""

from __future__ import annotations

import contextlib
import logging
import queue
import threading
import time
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass, field
from typing import Any, Protocol

from roco.bus import EventBus, State, Topic
from roco.voice.audio import frame_rms, iter_blocks, mouth_from_rms
from roco.voice.base import Speaker

log = logging.getLogger(__name__)

Audio = tuple[bytes, int]
TTS = Callable[[str], Iterable[Audio]]


class AudioOutput(Protocol):
    def write(self, block: bytes, sample_rate: int) -> None:
        """Play int16 mono PCM; blocks roughly in real time once the buffer is full."""

    def drain(self) -> None:
        """Return once everything written so far has actually been played."""

    def abort(self) -> None:
        """Drop buffered audio immediately."""

    def close(self) -> None: ...


class SoundDeviceOutput:
    """sounddevice output; the stream is (re)opened lazily at the voice's sample rate."""

    def __init__(self, device: int | None = None, tail_s: float = 0.1) -> None:
        self.device = device
        self.tail_s = tail_s  # silence written before stopping, so no hostapi clips the end
        self._stream: Any = None
        self._rate: int | None = None

    def _ensure(self, rate: int) -> Any:
        import sounddevice as sd

        if self._stream is not None and self._rate != rate:
            self.close()
        if self._stream is None:
            self._stream = sd.RawOutputStream(
                samplerate=rate, channels=1, dtype="int16", device=self.device
            )
            self._rate = rate
        if not self._stream.active:
            self._stream.start()
        return self._stream

    def write(self, block: bytes, sample_rate: int) -> None:
        self._ensure(sample_rate).write(block)

    def drain(self) -> None:
        stream = self._stream
        if stream is not None and stream.active:
            if self.tail_s and self._rate:
                stream.write(b"\x00\x00" * int(self._rate * self.tail_s))
            stream.stop()  # PortAudio: waits until pending buffers have played

    def abort(self) -> None:
        stream = self._stream
        if stream is not None and stream.active:
            stream.abort()

    def close(self) -> None:
        stream, self._stream = self._stream, None
        if stream is not None:
            with contextlib.suppress(Exception):
                stream.close()


@dataclass
class _Item:
    text: str | None  # None = flush marker (end of a hold)
    gen: int
    audio: list[Audio] = field(default_factory=list)


_SHUTDOWN = None


class PiperSpeaker(Speaker):
    """Queue-based speaker. Despite the name it works with any ``TTS`` callable."""

    def __init__(
        self,
        tts: TTS,
        output: AudioOutput,
        bus: EventBus | None = None,
        block_ms: int = 50,
        synth_ahead: int = 3,
    ) -> None:
        self._tts = tts
        self._output = output
        self._bus = bus
        self._block_ms = block_ms
        self._text_q: queue.Queue[_Item | None] = queue.Queue()
        self._audio_q: queue.Queue[_Item | None] = queue.Queue(maxsize=synth_ahead)
        self._cond = threading.Condition()
        self._pending = 0  # items queued and not yet fully played
        self._holds = 0
        self._finalizing = False  # publishing mouth 0 / IDLE: not "done" yet
        self._gen = 0  # bumped by stop(): older items are dropped
        self._announced = False  # SPEAKING published for the current run
        self._bus_state: State | None = None
        self._audio_start_hooks: list[Callable[[float], None]] = []
        self._closed = False
        self._unsubscribe = bus.subscribe(Topic.STATE, self._on_state) if bus else None
        self._threads = [
            threading.Thread(target=self._synth_loop, name="roco-tts", daemon=True),
            threading.Thread(target=self._play_loop, name="roco-playback", daemon=True),
        ]
        for t in self._threads:
            t.start()

    # ------------------------------------------------------------------ public API
    def say(self, text: str) -> None:
        if not text or not text.strip() or self._closed:
            return
        with self._cond:
            self._pending += 1
            gen = self._gen
        self._text_q.put(_Item(text.strip(), gen))

    def wait_done(self, timeout: float | None = None) -> bool:
        with self._cond:
            return self._cond.wait_for(self._idle_locked, timeout)

    @property
    def is_speaking(self) -> bool:
        with self._cond:
            return not self._idle_locked()

    def stop(self) -> None:
        with self._cond:
            self._gen += 1

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self.stop()
        self._text_q.put(_SHUTDOWN)
        for t in self._threads:
            t.join(timeout=3)
        if self._unsubscribe:
            self._unsubscribe()
        with contextlib.suppress(Exception):
            self._output.close()

    # ------------------------------------------------------------- extras for the loop
    @contextlib.contextmanager
    def hold(self) -> Iterator[None]:
        """Keep ``is_speaking`` true (and no IDLE) while a reply is still being generated,
        so the face does not flicker back to idle between two slow sentences."""
        with self._cond:
            self._holds += 1
        try:
            yield
        finally:
            with self._cond:
                self._holds -= 1
                self._pending += 1
                gen = self._gen
            self._text_q.put(_Item(None, gen))  # flows through both queues, in order

    def on_audio_start(self, hook: Callable[[float], None]) -> None:
        """Call ``hook(time.monotonic())`` when the first audio block of a run plays."""
        self._audio_start_hooks.append(hook)

    # -------------------------------------------------------------------- internals
    def _idle_locked(self) -> bool:
        return self._pending == 0 and self._holds == 0 and not self._finalizing

    def _on_state(self, state: Any) -> None:
        self._bus_state = state

    def _stale(self, item: _Item) -> bool:
        with self._cond:
            return item.gen != self._gen

    def _synth_loop(self) -> None:
        while True:
            item = self._text_q.get()
            if item is _SHUTDOWN:
                self._audio_q.put(_SHUTDOWN)
                return
            if item.text is not None and not self._stale(item):
                try:
                    item.audio = list(self._tts(item.text))
                except Exception:
                    log.exception("TTS failed for %r", item.text)
            self._audio_q.put(item)

    def _play_loop(self) -> None:
        while True:
            item = self._audio_q.get()
            if item is _SHUTDOWN:
                return
            try:
                self._play(item)
            except Exception:
                log.exception("Audio playback failed")
                with contextlib.suppress(Exception):
                    self._output.abort()
            self._finish_item()

    def _play(self, item: _Item) -> None:
        for pcm, rate in item.audio:
            block_samples = max(1, rate * self._block_ms // 1000)
            for block in iter_blocks(pcm, block_samples):
                if self._stale(item):
                    self._output.abort()
                    return
                if not self._announced:
                    self._announce_start()
                if self._bus:
                    self._bus.set_mouth(mouth_from_rms(frame_rms(block)))
                self._output.write(block, rate)

    def _announce_start(self) -> None:
        self._announced = True
        now = time.monotonic()
        if self._bus:
            self._bus.set_state(State.SPEAKING)
        for hook in list(self._audio_start_hooks):
            try:
                hook(now)
            except Exception:
                log.exception("audio-start hook failed")

    def _finish_item(self) -> None:
        with self._cond:
            last = self._pending == 1 and self._holds == 0
        if last and self._announced:
            try:
                self._output.drain()
            except Exception:
                log.exception("Audio drain failed")
        finalize = announced = False
        with self._cond:
            self._pending -= 1
            if self._pending == 0 and self._holds == 0:
                finalize, announced = True, self._announced
                self._announced = False
                self._finalizing = True
        if not finalize:
            return
        try:
            if self._bus and announced:
                self._bus.set_mouth(0.0)
                if self._bus_state in (None, State.SPEAKING):
                    self._bus.set_state(State.IDLE)
        finally:
            with self._cond:
                self._finalizing = False
                self._cond.notify_all()
