"""Streaming speech-to-text for one user turn: decode while Adam is still talking.

faster-whisper is not a streaming model, so the server gets its head start two ways,
both run on one background decode thread (``executor``) while mic frames arrive:

- **Speculative final decode.** When the incoming audio has been quiet for
  ``speculative_quiet_ms`` (well before the body's VAD closes the turn, which waits
  ``trailing_ms`` ≈ 450 ms), the uncommitted audio is decoded as if the turn had
  ended. If ``turn_end`` then confirms that speech ended inside the decoded audio, the
  result is used as is: STT costs nothing after end of speech. If Adam started
  talking again, the guess is thrown away and the tail decoded normally.
- **Committed prefix (long utterances).** Every ``partial_every_s`` of new audio once
  the uncommitted part is ``partial_min_s`` long, the uncommitted audio is decoded.
  Leading segments that two consecutive decodes agree on and that end at least
  ``keep_tail_s`` before the newest audio are committed: their text is kept and their
  audio is never decoded again, so the final decode only covers the tail.
  Partial transcripts go to ``on_partial`` (sent as ``transcript{final: false}``).

The engine is anything with ``segments(pcm: bytes, rate: int) -> list[Segment]``
(:class:`roco.voice.stt.WhisperTranscriber`), or a plain callable
``(pcm: bytes, rate: int) -> str``. Plain Python only here (no numpy), so the logic is
tested in CI without the voice extras.
"""

from __future__ import annotations

import logging
import math
import re
import sys
import threading
from array import array
from collections.abc import Callable
from concurrent.futures import Executor, Future
from dataclasses import dataclass
from typing import Any

from roco.voice.stt import Segment

log = logging.getLogger(__name__)

BYTES_PER_SAMPLE = 2


@dataclass(frozen=True)
class StreamingSettings:
    partial_every_s: float = 1.0
    partial_min_s: float = 3.0
    keep_tail_s: float = 1.0
    speculative_quiet_ms: int = 240
    # A frame is "speech" above max(min_speech_rms, speech_ratio * loudest frame so far).
    min_speech_rms: float = 300.0
    speech_ratio: float = 0.2
    # Anything above max(soft_rms, soft_ratio * loudest) may still be a word (a quiet
    # "it?" at the end); a speculative decode is only used if it covered all of that.
    soft_rms: float = 120.0
    soft_ratio: float = 0.05

    @classmethod
    def from_config(cls, section: dict[str, Any] | None) -> StreamingSettings:
        section = section or {}
        names = {f for f in cls.__dataclass_fields__}
        return cls(**{k: type(getattr(cls, k))(v) for k, v in section.items() if k in names})


def frame_rms(pcm: bytes) -> float:
    """Root mean square of PCM16 little-endian samples."""
    usable = len(pcm) - len(pcm) % 2
    if usable <= 0:
        return 0.0
    samples = array("h")
    samples.frombytes(pcm[:usable])
    if samples.itemsize != 2:  # pragma: no cover - exotic platforms
        return 0.0
    if sys.byteorder == "big":  # pragma: no cover
        samples.byteswap()
    return math.sqrt(sum(s * s for s in samples) / len(samples))


def engine_segments(engine: Any, pcm: bytes, rate: int) -> list[Segment]:
    if hasattr(engine, "segments"):
        return [Segment(*s) for s in engine.segments(pcm, rate)]
    text = str(engine(pcm, rate) or "").strip()
    duration = len(pcm) / BYTES_PER_SAMPLE / rate
    return [Segment(0.0, duration, text)] if text else []


def _norm(text: str) -> str:
    return re.sub(r"[^a-z0-9']+", " ", text.lower()).strip()


def join_text(parts: list[str]) -> str:
    return " ".join(p.strip() for p in parts if p and p.strip()).strip()


@dataclass
class _Speculation:
    offset: int  # committed offset (bytes) the decode started from
    end: int  # bytes decoded up to
    text: str


class StreamingSTT:
    """One turn's audio. ``feed`` from the receiver thread, ``finish`` from the worker."""

    def __init__(
        self,
        engine: Any,
        executor: Executor,
        rate: int = 16000,
        settings: StreamingSettings | None = None,
        on_partial: Callable[[str], None] | None = None,
    ) -> None:
        self.engine = engine
        self.executor = executor
        self.rate = rate
        self.settings = settings or StreamingSettings()
        self.on_partial = on_partial
        self._lock = threading.RLock()  # re-entrant: an inline executor decodes inside feed()
        self._pcm = bytearray()
        self._peak = 0.0
        self._last_speech_end = 0  # byte offset just after the last speech frame
        self._last_soft_end = 0  # ... just after the last frame that is not near-silence
        self._committed = 0  # byte offset: audio before it is transcribed for good
        self._committed_text: list[str] = []
        self._previous: list[tuple[int, str]] = []  # last partial: (abs end byte, norm text)
        self._last_partial_at = 0
        self._job: Future[Any] | None = None
        self._job_spec: tuple[int, int] | None = None  # (offset, end) if _job speculates
        self._spec: _Speculation | None = None
        self._finished = False
        self.stats = {"partials": 0, "speculative": 0, "speculative_used": 0, "final": 0}

    # ------------------------------------------------------------------ input
    @property
    def total_bytes(self) -> int:
        return len(self._pcm)

    def bytes_for_ms(self, ms: float) -> int:
        return int(self.rate * ms / 1000) * BYTES_PER_SAMPLE

    def feed(self, pcm: bytes) -> None:
        if not pcm:
            return
        rms = frame_rms(pcm)
        with self._lock:
            if self._finished:
                return
            self._pcm += pcm
            self._peak = max(self._peak, rms)
            threshold = max(self.settings.min_speech_rms, self.settings.speech_ratio * self._peak)
            if rms >= threshold:
                self._last_speech_end = len(self._pcm)
            if rms >= max(self.settings.soft_rms, self.settings.soft_ratio * self._peak):
                self._last_soft_end = len(self._pcm)
            self._schedule_locked()

    # ------------------------------------------------------------------ result
    def finish(self, speech_end: int | None = None, timeout: float = 30.0) -> str:
        """Final transcript. ``speech_end``: byte offset where the body's VAD says speech
        ended (total minus ``trailing_ms``); a speculative decode that reached it is used."""
        with self._lock:
            self._finished = True
            end = len(self._pcm)
            speech_end = end if speech_end is None else max(0, min(speech_end, end))
            # Audio after the last non-silent frame cannot hold a word.
            needed = min(speech_end, self._last_soft_end)
            job, job_spec, committed = self._job, self._job_spec, self._committed
        # Wait for a running decode only if its result can be used; a partial or a
        # speculation that stopped short would just delay the final decode.
        usable = job_spec is not None and job_spec[0] == committed and job_spec[1] >= needed
        if job is not None and usable:
            try:
                job.result(timeout=timeout)
            except Exception:
                log.exception("background STT decode failed")
        with self._lock:
            spec = self._spec
            if spec and spec.offset == self._committed and spec.end >= needed:
                self.stats["speculative_used"] += 1
                return join_text([*self._committed_text, spec.text])
            offset = self._committed
            head = list(self._committed_text)
            tail_pcm = bytes(self._pcm[offset:end])
        self.stats["final"] += 1
        tail = join_text([s.text for s in engine_segments(self.engine, tail_pcm, self.rate)])
        return join_text([*head, tail])

    # ------------------------------------------------------------------ background decodes
    def _schedule_locked(self) -> None:
        if self._job is not None and not self._job.done():
            return
        s = self.settings
        end = len(self._pcm)
        quiet = end - self._last_speech_end
        has_speech = self._last_speech_end > self._committed
        if (
            has_speech
            and quiet >= self.bytes_for_ms(s.speculative_quiet_ms)
            # again if anything that may be a word arrived after the last guess
            and (self._spec is None or self._spec.end < self._last_soft_end
                 or self._spec.offset != self._committed)
        ):
            self._submit(self._speculate, self._committed, end)
            return
        uncommitted = end - self._committed
        if (
            s.partial_min_s > 0
            and uncommitted >= self.bytes_for_ms(s.partial_min_s * 1000)
            and end - self._last_partial_at >= self.bytes_for_ms(s.partial_every_s * 1000)
        ):
            self._last_partial_at = end
            self._submit(self._partial, self._committed, end)

    def _submit(self, fn: Callable[[int, int, bytes], None], offset: int, end: int) -> None:
        pcm = bytes(self._pcm[offset:end])
        self._job_spec = (offset, end) if fn == self._speculate else None
        try:
            self._job = self.executor.submit(fn, offset, end, pcm)
        except RuntimeError:  # executor shut down
            self._job = self._job_spec = None

    def _speculate(self, offset: int, end: int, pcm: bytes) -> None:
        text = join_text([s.text for s in engine_segments(self.engine, pcm, self.rate)])
        with self._lock:
            self.stats["speculative"] += 1
            if offset == self._committed:
                self._spec = _Speculation(offset, end, text)

    def _partial(self, offset: int, end: int, pcm: bytes) -> None:
        segments = engine_segments(self.engine, pcm, self.rate)
        bps = self.rate * BYTES_PER_SAMPLE
        keep_from = end - self.bytes_for_ms(self.settings.keep_tail_s * 1000)
        current = [(offset + round(seg.end * self.rate) * BYTES_PER_SAMPLE, _norm(seg.text))
                   for seg in segments]
        with self._lock:
            self.stats["partials"] += 1
            if offset != self._committed or self._finished:
                return  # stale, or the final transcript is already being made
            commit = 0
            for i, (abs_end, norm) in enumerate(current[:-1]):  # never the last segment
                agreed = i < len(self._previous) and self._previous[i][1] == norm
                if not (agreed and norm and abs_end <= keep_from):
                    break
                commit = i + 1
            if commit:
                self._committed_text += [segments[i].text for i in range(commit)]
                self._committed = min(current[commit - 1][0], end)
                self._spec = None
                log.debug("STT committed %.1f s: %r", self._committed / bps,
                          join_text(self._committed_text))
            # After a commit the next decode starts at the new offset: agreement restarts.
            self._previous = [] if commit else current
            partial = join_text([*self._committed_text, *(s.text for s in segments[commit:])])
        if self.on_partial is not None and partial:
            try:
                self.on_partial(partial)
            except Exception:
                log.exception("on_partial failed")
