"""Listener: wake word -> record until silence -> transcribe.

The microphone, wake-word model and speech-to-text engine are injected (see
:mod:`roco.voice.factory` for the real ones), so this module has no heavy imports
and its logic is tested with fakes. Frames can be numpy int16 arrays or any int16
sequence.
"""

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from typing import Any, Protocol

from roco.bus import EventBus, State
from roco.voice.audio import frame_rms
from roco.voice.base import Listener
from roco.voice.endpoint import Endpoint, NoiseFloor, RecordingSettings, SilenceDetector

log = logging.getLogger(__name__)


class FrameSource(Protocol):
    def read(self, timeout: float) -> Any | None:
        """Next 80 ms frame, or None if none arrived within ``timeout`` seconds."""

    def flush(self) -> None:
        """Drop frames captured so far (stale audio from before ``listen``)."""

    def close(self) -> None: ...


class WakeDetector(Protocol):
    def score(self, frame: Any) -> float: ...

    def reset(self) -> None: ...


Transcriber = Callable[[Sequence[Any]], str]


@dataclass
class TurnTiming:
    """monotonic() timestamps of one utterance, for latency logging."""

    speech_end: float  # last frame that contained speech
    stt_start: float
    stt_done: float
    audio_s: float


class _Closed(Exception):
    pass


class VoiceListener(Listener):
    def __init__(
        self,
        source: FrameSource,
        wake: WakeDetector,
        transcriber: Transcriber,
        bus: EventBus | None = None,
        *,
        recording: RecordingSettings | None = None,
        wake_threshold: float = 0.5,
        is_speaking: Callable[[], bool] | None = None,
        frame_s: float = 0.08,
        unmute_guard_s: float = 0.3,
        wake_tail_s: float = 0.24,
        preroll_s: float = 0.32,
        read_timeout_s: float = 0.1,
    ) -> None:
        self.source = source
        self.wake = wake
        self.transcriber = transcriber
        self.bus = bus
        self.recording = recording or RecordingSettings()
        self.wake_threshold = wake_threshold
        self.is_speaking = is_speaking
        self.frame_s = frame_s
        self.guard_frames = round(unmute_guard_s / frame_s)
        self.wake_tail_frames = round(wake_tail_s / frame_s)
        self.preroll_frames = max(0, round(preroll_s / frame_s))
        self.read_timeout_s = read_timeout_s
        self.noise_floor = NoiseFloor()
        self.last_timing: TurnTiming | None = None
        self.on_wake: Callable[[float], None] | None = None  # hook(score) for diagnostics
        self._lock = threading.Lock()
        self._muted = False
        self._was_deaf = True  # first frame resets the wake detector
        self._guard = 0
        self._closed = threading.Event()

    # ------------------------------------------------------------------ interface
    def set_muted(self, muted: bool) -> None:
        with self._lock:
            self._muted = bool(muted)

    @property
    def muted(self) -> bool:
        return self._muted

    def listen(self) -> str | None:
        try:
            self._start()
            while True:
                frame = self._next_frame(deadline=None)
                score = float(self.wake.score(frame))
                if score >= self.wake_threshold:
                    log.info("Wake word detected (score %.2f)", score)
                    self.wake.reset()
                    if self.on_wake:
                        self.on_wake(score)
                    return self._record(self.recording, skip_wake_tail=True)
                self.noise_floor.update(frame_rms(frame))  # idle: learn the room
        except _Closed:
            return None

    def listen_followup(self, timeout: float) -> str | None:
        try:
            self._start()
            settings = replace(self.recording, start_timeout_s=timeout)
            return self._record(settings, skip_wake_tail=False)
        except _Closed:
            return None

    def close(self) -> None:
        self._closed.set()
        try:
            self.source.close()
        except Exception:
            log.exception("Closing the audio source failed")

    # ------------------------------------------------------------------ internals
    def _start(self) -> None:
        if self._closed.is_set():
            raise _Closed
        self.source.flush()
        self._was_deaf = True  # stale audio dropped: wake-word context is gone too

    def _deaf(self) -> bool:
        if self._muted:
            return True
        return bool(self.is_speaking and self.is_speaking())

    def _next_frame(self, deadline: float | None) -> Any | None:
        """Next usable frame. Frames heard while muted/speaking are dropped; after
        unmuting the wake detector is reset and a short guard (room echo) skipped.
        Returns None once ``deadline`` (monotonic) passes without a frame."""
        while True:
            if self._closed.is_set():
                raise _Closed
            frame = self.source.read(self.read_timeout_s)
            if frame is None:
                if self._closed.is_set():
                    raise _Closed
                if deadline is not None and time.monotonic() > deadline:
                    return None
                continue
            if self._deaf():
                self._was_deaf = True
                continue
            if self._was_deaf:
                self._was_deaf = False
                self.wake.reset()
                self._guard = self.guard_frames
            if self._guard > 0:
                self._guard -= 1
                continue
            return frame

    def _publish(self, state: State) -> None:
        if self.bus:
            self.bus.set_state(state)

    def _record(
        self,
        settings: RecordingSettings,
        skip_wake_tail: bool,
    ) -> str | None:
        """Record one utterance, keeping a short pre-roll from before speech was
        detected (detection needs a couple of loud frames: the first syllable)."""
        self._publish(State.LISTENING)
        det = SilenceDetector(settings, self.noise_floor, self.frame_s)
        t0 = time.monotonic()
        # Wall-clock safety net in case the mic stops delivering frames.
        deadline = t0 + max(settings.start_timeout_s, settings.max_s) + 2.0

        if skip_wake_tail:  # the end of "...Jarvis" can still be in the next frames
            for _ in range(self.wake_tail_frames):
                frame = self._next_frame(deadline)
                if frame is None or frame_rms(frame) < det.threshold:
                    break

        before: deque[Any] = deque(maxlen=max(1, self.preroll_frames))
        frames: list[Any] = []
        speech_end_t = t0
        while True:
            frame = self._next_frame(deadline)
            if frame is None:
                event = Endpoint.MAX_LENGTH if det.speech_started else Endpoint.NO_SPEECH
            else:
                event = det.feed(frame_rms(frame))
                if det.last_speech_frame == det.frames - 1:
                    speech_end_t = time.monotonic()
                if det.speech_started:
                    if event is Endpoint.SPEECH_START:
                        frames.extend(before)
                    frames.append(frame)
                else:
                    before.append(frame)
            if event is Endpoint.SPEECH_START:
                log.debug("Speech started (threshold %.0f, floor %.0f)",
                          det.threshold, self.noise_floor.level)
            elif event is Endpoint.NO_SPEECH:
                log.info("No speech heard")
                self._publish(State.IDLE)
                return None
            elif event in (Endpoint.SILENCE, Endpoint.MAX_LENGTH):
                break

        # Drop most of the trailing silence: whisper is faster and hallucinates less.
        keep = len(frames) - max(0, det.frames - 1 - det.last_speech_frame) + 2
        frames = frames[: max(1, keep)]
        audio_s = len(frames) * self.frame_s
        stt_start = time.monotonic()
        try:
            text = (self.transcriber(frames) or "").strip()
        except Exception:
            log.exception("Speech-to-text failed")
            text = ""
        stt_done = time.monotonic()
        self.last_timing = TurnTiming(speech_end_t, stt_start, stt_done, audio_s)
        log.info("Heard (%.1fs audio, STT %d ms): %r", audio_s,
                 (stt_done - stt_start) * 1000, text)
        if not _intelligible(text):
            self._publish(State.IDLE)
            return None
        self._publish(State.THINKING)
        return text


def _intelligible(text: str) -> bool:
    """False for empty transcripts and whisper's punctuation-only output."""
    return any(ch.isalnum() for ch in text)
