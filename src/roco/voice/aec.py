"""Software acoustic echo cancellation (AEC) for barge-in.

The body keeps the mic open while ROCO speaks. The mic then hears ROCO's own voice
from the speaker (the *echo*); the echo canceller subtracts it using a copy of what
was played (the *far end* / reference), so the VAD only reacts to Adam.

Pieces:

- :class:`EchoCanceller` protocol: ``process(near, far) -> cleaned``, all int16 mono
  16 kHz numpy arrays of the same length (a multiple of 10 ms). Implementations:
  :class:`WebRtcAec` (WebRTC AudioProcessingModule AEC3 through the ``livekit``
  wheel, the chosen default, see docs/AEC.md), :class:`SpeexAec` (SpeexDSP through
  ``pyaec``) and :class:`NullAec` (pass-through: CI and "no AEC installed").
- :class:`EchoReference`: a FIFO the playback callback pushes every played block
  into (at the output rate); the mic side takes exactly as many 16 kHz samples as
  it captured. Both streams run in real time, so the FIFO keeps them aligned up to
  a constant delay, which the canceller estimates itself.
- :class:`AecFrameSource`: wraps the mic frame source; every frame it returns has
  been cleaned. It also feeds :class:`EchoMonitor`, which measures the echo return
  loss enhancement (ERLE) while ROCO speaks, so the body can fall back to
  wake-word-only barge-in when the AEC is not doing its job.

numpy is imported lazily: the module imports fine in text-mode CI.
"""

from __future__ import annotations

import contextlib
import logging
import math
import statistics
import threading
from collections import deque
from typing import Any, Protocol

log = logging.getLogger(__name__)

AEC_RATE = 16_000
CHUNK = AEC_RATE // 100  # 10 ms: what both WebRTC and our Speex setup process


class EchoCanceller(Protocol):
    name: str

    def process(self, near: Any, far: Any) -> Any:
        """Return ``near`` (mic) with the echo of ``far`` (played audio) removed."""

    def close(self) -> None: ...


class AecUnavailable(RuntimeError):
    """The requested AEC engine is not installed (the message says how to get it)."""


# --------------------------------------------------------------------------- engines


class NullAec:
    """No echo cancellation: the mic frame is returned unchanged."""

    name = "none"

    def process(self, near: Any, far: Any) -> Any:
        return near

    def close(self) -> None:
        pass


class WebRtcAec:
    """WebRTC AudioProcessingModule (AEC3) from the ``livekit`` package.

    AEC3 estimates the echo path delay itself (it searches up to ~0.5 s), adapts to
    a laptop's changing echo path and suppresses the non-linear residual, which is
    what makes "any speech" barge-in safe on cheap speakers. ``delay_ms`` is only a
    hint. Noise suppression and AGC stay off by default so the wake word and VAD
    see the same levels as without AEC.
    """

    name = "webrtc"

    def __init__(
        self,
        delay_ms: int = 0,
        noise_suppression: bool = False,
        high_pass_filter: bool = True,
        auto_gain_control: bool = False,
    ) -> None:
        try:
            from livekit.rtc import AudioFrame, AudioProcessingModule
        except ImportError as exc:
            raise AecUnavailable(
                'WebRTC AEC needs the livekit package: pip install -e ".[body]"'
            ) from exc
        import numpy as np

        self._np = np
        self._frame_cls = AudioFrame
        self._apm = AudioProcessingModule(
            echo_cancellation=True,
            noise_suppression=noise_suppression,
            high_pass_filter=high_pass_filter,
            auto_gain_control=auto_gain_control,
        )
        self.delay_ms = int(delay_ms)

    def process(self, near: Any, far: Any) -> Any:
        np = self._np
        near = np.ascontiguousarray(near, dtype=np.int16)
        far = np.ascontiguousarray(far, dtype=np.int16)
        out = np.empty_like(near)
        frame_cls = self._frame_cls
        for i in range(0, len(near) - CHUNK + 1, CHUNK):
            rev = frame_cls(bytearray(far[i : i + CHUNK].tobytes()), AEC_RATE, 1, CHUNK)
            self._apm.process_reverse_stream(rev)
            self._apm.set_stream_delay_ms(self.delay_ms)
            buf = bytearray(near[i : i + CHUNK].tobytes())
            self._apm.process_stream(frame_cls(buf, AEC_RATE, 1, CHUNK))
            out[i : i + CHUNK] = np.frombuffer(buf, dtype=np.int16)
        rest = len(near) % CHUNK
        if rest:  # never happens with 80 ms frames; keep the tail as is
            out[-rest:] = near[-rest:]
        return out

    def close(self) -> None:
        self._apm = None


class SpeexAec:
    """SpeexDSP echo canceller (+ its residual-echo preprocessor) via ``pyaec``.

    A plain NLMS adaptive filter: ``filter_ms`` must cover the whole echo path
    *including* the output+input buffering delay (100-300 ms on Windows MME),
    and it struggles with the non-linear distortion of small laptop speakers.
    """

    name = "speex"

    def __init__(self, filter_ms: int = 300, preprocess: bool = True) -> None:
        try:
            import pyaec
        except (ImportError, OSError) as exc:
            raise AecUnavailable("Speex AEC needs the pyaec package: pip install pyaec") from exc
        if getattr(pyaec, "lib", None) is None:
            raise AecUnavailable("pyaec is installed but its native library did not load")
        import ctypes

        import numpy as np

        self._np = np
        self._ct = ctypes
        self._lib = pyaec.lib
        filter_len = int(AEC_RATE * filter_ms / 1000)
        self._handle = self._lib.AecNew(CHUNK, filter_len, AEC_RATE, bool(preprocess))
        if not self._handle:
            raise AecUnavailable("Speex AEC could not be created")

    def process(self, near: Any, far: Any) -> Any:
        np, ct = self._np, self._ct
        near = np.ascontiguousarray(near, dtype=np.int16)
        far = np.ascontiguousarray(far, dtype=np.int16)
        out = np.empty_like(near)
        ptr = ct.POINTER(ct.c_int16)
        for i in range(0, len(near) - CHUNK + 1, CHUNK):
            n, f, o = near[i : i + CHUNK], far[i : i + CHUNK], out[i : i + CHUNK]
            self._lib.AecCancelEcho(
                self._handle, n.ctypes.data_as(ptr), f.ctypes.data_as(ptr),
                o.ctypes.data_as(ptr), CHUNK,
            )
        rest = len(near) % CHUNK
        if rest:
            out[-rest:] = near[-rest:]
        return out

    def close(self) -> None:
        handle, self._handle = self._handle, None
        if handle:
            with contextlib.suppress(Exception):
                self._lib.AecDestroy(handle)


ENGINES = ("webrtc", "speex", "none")


def create_aec(engine: str = "auto", **options: Any) -> EchoCanceller:
    """Build an echo canceller. ``auto`` tries WebRTC, then Speex, then none.

    An explicit engine that is not installed raises :class:`AecUnavailable`.
    """
    engine = (engine or "auto").lower()
    if engine in ("none", "off", "false"):
        return NullAec()
    if engine == "webrtc":
        return WebRtcAec(delay_ms=int(options.get("delay_ms", 0)),
                         noise_suppression=bool(options.get("noise_suppression", False)))
    if engine == "speex":
        return SpeexAec(filter_ms=int(options.get("filter_ms", 300)))
    if engine != "auto":
        raise ValueError(f"Unknown AEC engine {engine!r} (expected auto, {', '.join(ENGINES)})")
    for name in ("webrtc", "speex"):
        try:
            return create_aec(name, **options)
        except AecUnavailable as exc:
            log.info("AEC %s unavailable: %s", name, exc)
    log.warning("No echo canceller installed: barge-in falls back to the wake word")
    return NullAec()


# --------------------------------------------------------------------------- reference


class StreamResampler:
    """Linear-interpolation resampler that keeps its phase across blocks, so a
    stream resampled block by block has no seams (good enough for an AEC reference)."""

    def __init__(self, from_rate: int, to_rate: int) -> None:
        self.from_rate = from_rate
        self.to_rate = to_rate
        self._step = from_rate / to_rate
        # Position of the next output sample in the coordinates of ``x`` below, where
        # x = [last sample of the previous block] + this block (index 0 = that sample).
        self._pos = 0.0
        self._last: float | None = None

    def __call__(self, block: Any) -> Any:
        import numpy as np

        if self.from_rate == self.to_rate:
            return np.asarray(block, dtype=np.int16)
        x = np.asarray(block, dtype=np.float64)
        if len(x) == 0:
            return np.zeros(0, dtype=np.int16)
        if self._last is not None:
            x = np.concatenate([[self._last], x])
        end = len(x) - 1
        n = int(math.floor((end - self._pos) / self._step)) + 1 if self._pos <= end else 0
        idx = self._pos + self._step * np.arange(n)
        y = np.interp(idx, np.arange(len(x)), x) if n else np.zeros(0)
        self._pos = self._pos + self._step * n - end  # next block's x starts at our x[end]
        self._last = float(x[-1])
        return np.clip(np.round(y), -32768, 32767).astype(np.int16)


class EchoReference:
    """What the speaker played, resampled to 16 kHz, waiting for the mic side.

    ``push`` runs on the audio output callback (cheap: resample + append);
    ``take(n)`` runs on the mic thread. If the mic side stalls, the FIFO is capped at
    ``max_s`` (oldest dropped); if playback produced less than asked, zeros pad.
    """

    def __init__(self, max_s: float = 1.0) -> None:
        self._lock = threading.Lock()
        self._chunks: deque[Any] = deque()
        self._size = 0
        self._max = int(max_s * AEC_RATE)
        self._resampler: StreamResampler | None = None
        self.dropped = 0
        self.underruns = 0

    def push(self, pcm: Any, rate: int) -> None:
        import numpy as np

        samples = np.frombuffer(pcm, dtype=np.int16) if isinstance(pcm, bytes | bytearray) \
            else np.asarray(pcm, dtype=np.int16)
        if self._resampler is None or self._resampler.from_rate != rate:
            self._resampler = StreamResampler(rate, AEC_RATE)
        block = self._resampler(samples)
        if not len(block):
            return
        with self._lock:
            self._chunks.append(block)
            self._size += len(block)
            while self._size > self._max and self._chunks:
                old = self._chunks.popleft()
                self._size -= len(old)
                self.dropped += len(old)

    def take(self, n: int) -> Any:
        import numpy as np

        out = np.zeros(n, dtype=np.int16)
        filled = 0
        with self._lock:
            while filled < n and self._chunks:
                chunk = self._chunks[0]
                k = min(n - filled, len(chunk))
                out[filled : filled + k] = chunk[:k]
                filled += k
                if k == len(chunk):
                    self._chunks.popleft()
                else:
                    self._chunks[0] = chunk[k:]
                self._size -= k
        if filled < n:
            self.underruns += 1
        return out

    def clear(self) -> None:
        with self._lock:
            self._chunks.clear()
            self._size = 0

    @property
    def buffered_ms(self) -> float:
        return 1000.0 * self._size / AEC_RATE


# --------------------------------------------------------------------------- monitor


def _power(x: Any) -> float:
    import numpy as np

    if len(x) == 0:
        return 0.0
    return float(np.mean(np.square(np.asarray(x, dtype=np.float64))))


class EchoMonitor:
    """Watches the echo while ROCO speaks: ERLE, and a double-talk gate for barge-in.

    **ERLE** = 10*log10(P_mic / P_cleaned) over frames where the far end is clearly
    playing (``far_rms_min``). Near-end speech lowers a frame's ERLE, so the verdict
    uses the median over the last ``window`` such frames: AEC3 shows 15-40 dB.

    **Double-talk gate** (:meth:`near_end_db`): what survives the AEC while ROCO talks
    is mostly residual echo, and Silero happily calls ROCO's own residual voice
    "speech". The monitor learns that residual level (a high percentile of the
    cleaned power over the last ``residual_window`` playback frames that did not look
    like Adam) and reports how far the current frame is above it. The body only lets
    a frame count towards barge-in when it is ``margin_db`` above the residual. The
    far end stays "active" for ``hangover`` frames after playback stops (output
    buffer + room echo still arrive).
    """

    def __init__(
        self,
        far_rms_min: float = 300.0,
        window: int = 50,
        residual_window: int = 40,
        residual_percentile: float = 0.8,
        min_residual_frames: int = 5,
        hangover: int = 4,
        residual_rms_min: float = 30.0,
        learn_margin_db: float = 6.0,
    ) -> None:
        self.far_power_min = far_rms_min**2
        self._recent: deque[float] = deque(maxlen=window)
        self._residual: deque[float] = deque(maxlen=residual_window)
        self.residual_percentile = residual_percentile
        self.min_residual_frames = min_residual_frames
        self.hangover = hangover
        self.residual_power_min = residual_rms_min**2
        self.learn_margin = 10 ** (learn_margin_db / 10)
        self.frames = 0
        self._since_far = hangover + 1  # frames since the far end was last playing
        self.last_clean_power = 0.0

    @property
    def far_active(self) -> bool:
        """ROCO's voice is (or may still be) in the mic."""
        return self._since_far <= self.hangover

    def update(self, raw: Any, clean: Any, far: Any) -> None:
        p_far = _power(far)
        p_clean = _power(clean)
        self.last_clean_power = p_clean
        if p_far >= self.far_power_min:
            self._since_far = 0
        else:
            self._since_far += 1
        if not self.far_active:
            return
        floor = self.residual_floor
        if floor is None or p_clean <= floor * self.learn_margin:
            self._residual.append(p_clean)  # residual echo, not Adam: learn it
        if p_far < self.far_power_min:
            return
        p_raw = _power(raw)
        if p_raw <= 1.0:  # mic muted / digital silence: no information
            return
        self.frames += 1
        self._recent.append(10.0 * math.log10(p_raw / max(p_clean, 1.0)))

    @property
    def residual_floor(self) -> float | None:
        """Learnt residual-echo power after AEC, or None until enough playback frames."""
        if len(self._residual) < self.min_residual_frames:
            return None
        ordered = sorted(self._residual)
        idx = min(len(ordered) - 1, int(self.residual_percentile * len(ordered)))
        return max(ordered[idx], self.residual_power_min)

    def near_end_db(self) -> float | None:
        """How far the last frame is above the residual echo (dB); None = no echo now."""
        if not self.far_active:
            return None
        floor = self.residual_floor
        if floor is None:
            return float("-inf")  # just started playing: do not trust anything yet
        return 10.0 * math.log10(max(self.last_clean_power, 1.0) / floor)

    def near_end(self, margin_db: float) -> bool:
        """True when the last frame can be Adam: no echo now, or clearly above it."""
        db = self.near_end_db()
        return db is None or db >= margin_db

    @property
    def erle_db(self) -> float | None:
        return statistics.median(self._recent) if self._recent else None

    @property
    def samples(self) -> int:
        return len(self._recent)

    def reset(self) -> None:
        """Forget the ERLE of the last reply (the residual level is kept: same room)."""
        self._recent.clear()


class AecFrameSource:
    """A mic frame source whose frames come out echo-cancelled.

    Same interface as :class:`roco.voice.mic.MicStream` (``read``/``flush``/``close``).
    ``last_raw`` / ``last_far`` keep the latest unprocessed frame and reference for
    diagnostics.
    """

    def __init__(
        self,
        source: Any,
        aec: EchoCanceller,
        reference: EchoReference,
        monitor: EchoMonitor | None = None,
    ) -> None:
        self.source = source
        self.aec = aec
        self.reference = reference
        self.monitor = monitor or EchoMonitor()
        self.last_raw: Any = None
        self.last_far: Any = None

    def read(self, timeout: float) -> Any | None:
        frame = self.source.read(timeout)
        if frame is None:
            return None
        far = self.reference.take(len(frame))
        self.last_raw, self.last_far = frame, far
        if isinstance(self.aec, NullAec):
            clean = frame
        else:
            try:
                clean = self.aec.process(frame, far)
            except Exception:
                log.exception("AEC failed; passing the mic frame through")
                clean = frame
        self.monitor.update(frame, clean, far)
        return clean

    def flush(self) -> None:
        self.source.flush()

    def close(self) -> None:
        try:
            self.source.close()
        finally:
            with contextlib.suppress(Exception):
                self.aec.close()
