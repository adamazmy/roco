"""Light post-processing for TTS audio (spec §3): gentle EQ, soft compression, small room.

Everything is vectorised numpy on one sentence at a time (a few ms per sentence), so it
adds next to nothing to the time to first audio:

- **EQ** (zero-phase, in the frequency domain): high-pass under ~80 Hz, a small cut of
  low-mid "mud" around 300 Hz, a small presence lift around 3 kHz and a gentle shelf
  cut above ~7 kHz, where Piper's hiss and jitter live.
- **Compression**: a soft-knee RMS compressor on 10 ms frames with attack/release
  smoothing, then a fixed make-up gain and a soft limiter (no per-sentence
  normalisation, so a short "Hi." is not louder than a long sentence).
- **Room**: convolution with a short synthetic small-room impulse response (about
  0.3 s, low-passed), mixed in quietly so the voice sounds like it comes from a little
  robot on a desk rather than from inside your head.

:class:`ProcessedTTS` wraps any TTS callable (``text -> Iterable[(pcm, rate)]``) so the
speaker and the brain server can use any backend with or without it.
"""

from __future__ import annotations

import threading
from collections.abc import Callable, Iterable, Iterator, Mapping
from dataclasses import dataclass
from typing import Any

Audio = tuple[bytes, int]


@dataclass(frozen=True)
class PostFxSettings:
    eq: bool = True
    compress: bool = True
    reverb: float = 0.12  # wet level 0..1; 0 = off
    room_s: float = 0.3  # impulse response length
    threshold_db: float = -22.0
    ratio: float = 2.5
    makeup_db: float = 3.0

    @classmethod
    def from_config(cls, data: Mapping[str, Any] | bool | None) -> PostFxSettings | None:
        """``voice.tts.postfx``: false/None = off, true = defaults, or a mapping of fields."""
        if not data:
            return None
        if data is True:
            return cls()
        if not isinstance(data, Mapping) or data.get("enabled") is False:
            return None
        known = {k: v for k, v in data.items() if k in cls.__dataclass_fields__}
        return cls(**known)


def _bell(f: Any, centre: float, gain_db: float, octaves: float) -> Any:
    import numpy as np

    with np.errstate(divide="ignore"):
        distance = np.log2(np.maximum(f, 1.0) / centre) / octaves
    return gain_db * np.exp(-0.5 * distance**2)


def eq_curve_db(freqs: Any) -> Any:
    """The EQ's gain in dB at each frequency (Hz)."""
    import numpy as np

    f = np.maximum(freqs, 1.0)
    gain = np.zeros_like(f, dtype=np.float64)
    gain += -12.0 * np.log2(1 + (80.0 / f) ** 2)  # ~12 dB/octave high-pass under 80 Hz
    gain += _bell(f, 300.0, -2.0, 0.7)  # less boxy
    gain += _bell(f, 3000.0, 2.0, 0.8)  # presence
    gain += -2.5 / (1 + (7000.0 / f) ** 4)  # soft shelf cut above ~7 kHz
    return gain


class PostFx:
    """Applies :class:`PostFxSettings` to int16 mono PCM. Thread-safe."""

    def __init__(self, settings: PostFxSettings | None = None, seed: int = 7) -> None:
        self.settings = settings or PostFxSettings()
        self._seed = seed
        self._ir: dict[int, Any] = {}
        self._lock = threading.Lock()

    def process(self, pcm: bytes, rate: int) -> bytes:
        import numpy as np

        if not pcm:
            return pcm
        x = np.frombuffer(pcm, dtype=np.int16).astype(np.float64) / 32768.0
        s = self.settings
        if s.eq:
            x = self._eq(x, rate)
        if s.compress:
            x = self._compress(x, rate)
        if s.reverb > 0:
            x = self._room(x, rate)
        x = np.tanh(x * 1.1) / np.tanh(1.1)  # soft limiter, transparent at normal levels
        return (np.clip(x, -1.0, 1.0) * 32767.0).astype(np.int16).tobytes()

    # ------------------------------------------------------------------ stages
    @staticmethod
    def _eq(x: Any, rate: int) -> Any:
        import numpy as np

        pad = 2048
        n = len(x) + 2 * pad
        spectrum = np.fft.rfft(np.pad(x, pad), n)
        gain = 10 ** (eq_curve_db(np.fft.rfftfreq(n, 1.0 / rate)) / 20)
        return np.fft.irfft(spectrum * gain, n)[pad : pad + len(x)]

    def _compress(self, x: Any, rate: int) -> Any:
        import numpy as np

        s = self.settings
        hop = max(1, rate // 100)  # 10 ms frames
        frames = -(-len(x) // hop)
        padded = np.pad(x, (0, frames * hop - len(x)))
        rms = np.sqrt(np.mean(padded.reshape(frames, hop) ** 2, axis=1) + 1e-12)
        level = 20 * np.log10(rms)
        knee = 6.0
        over = level - s.threshold_db
        # Soft knee: quadratic within +-knee/2 of the threshold.
        reduction = np.where(
            over <= -knee / 2,
            0.0,
            np.where(
                over >= knee / 2,
                over * (1 - 1 / s.ratio),
                (1 - 1 / s.ratio) * (over + knee / 2) ** 2 / (2 * knee),
            ),
        )
        # Attack ~5 ms (one frame: instant), release ~80 ms.
        smooth = np.empty_like(reduction)
        release = np.exp(-1.0 / 8.0)
        current = 0.0
        for i, target in enumerate(reduction):
            current = target if target > current else release * current + (1 - release) * target
            smooth[i] = current
        gain_db = s.makeup_db - smooth
        centres = (np.arange(frames) + 0.5) * hop
        gain = 10 ** (np.interp(np.arange(len(x)), centres, gain_db) / 20)
        return x * gain

    def _impulse(self, rate: int) -> Any:
        import numpy as np

        with self._lock:
            ir = self._ir.get(rate)
            if ir is not None:
                return ir
            rng = np.random.default_rng(self._seed)
            n = max(1, int(self.settings.room_s * rate))
            t = np.arange(n) / rate
            rt60 = self.settings.room_s * 1.2
            tail = rng.standard_normal(n) * 10 ** (-3 * t / rt60)
            tail[: int(0.004 * rate)] = 0.0  # pre-delay: direct sound comes first
            for delay_ms, gain in ((7, 0.5), (11, -0.35), (17, 0.25), (23, -0.2)):
                i = int(delay_ms / 1000 * rate)
                if i < n:
                    tail[i] += gain * 4
            # Low-pass the reflections (walls absorb the highs).
            spec = np.fft.rfft(tail)
            freqs = np.fft.rfftfreq(n, 1.0 / rate)
            tail = np.fft.irfft(spec / np.sqrt(1 + (freqs / 4000.0) ** 2), n)
            tail /= np.sqrt(np.sum(tail**2)) + 1e-12
            self._ir[rate] = tail
            return tail

    def _room(self, x: Any, rate: int) -> Any:
        import numpy as np

        ir = self._impulse(rate)
        n = len(x) + len(ir) - 1
        size = 1 << (n - 1).bit_length()
        wet = np.fft.irfft(np.fft.rfft(x, size) * np.fft.rfft(ir, size), size)[:n]
        dry = np.pad(x, (0, len(ir) - 1))
        out = dry + self.settings.reverb * wet
        # Trim the reverb tail where it has died away (keeps sentence gaps natural).
        quiet = np.flatnonzero(np.abs(out) > 1e-3)
        end = max(len(x), int(quiet[-1]) + 1) if len(quiet) else len(x)
        return out[:end]


class ProcessedTTS:
    """A TTS callable whose audio goes through :class:`PostFx` one sentence at a time."""

    def __init__(self, inner: Callable[[str], Iterable[Audio]], fx: PostFx) -> None:
        self.inner = inner
        self.fx = fx

    def load(self) -> Any:
        load = getattr(self.inner, "load", None)
        return load() if callable(load) else None

    def __call__(self, text: str) -> Iterator[Audio]:
        chunks = list(self.inner(text))
        if not chunks:
            return
        rate = chunks[0][1]
        if any(r != rate for _, r in chunks):  # pragma: no cover - no backend does this
            for pcm, r in chunks:
                yield self.fx.process(pcm, r), r
            return
        yield self.fx.process(b"".join(pcm for pcm, _ in chunks), rate), rate
