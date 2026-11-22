"""Voice activity detection and end-of-turn decisions on the body.

- :class:`SileroVad`: Silero VAD (v5, or the v4 file openWakeWord ships) run with
  onnxruntime, which the body already needs for the wake word: no torch. It scores
  32 ms windows (512 samples at 16 kHz); a 80 ms mic frame gives 2-3 scores.
- :class:`EnergyVad`: the Phase 2 adaptive-noise-floor energy detector, as a
  fallback when the Silero model is missing (and for CI).
- :class:`VadEndpointer`: turns those scores into turn events: speech start, end of
  speech after ``trailing_ms`` of silence (fast: 450 ms by default), length cap,
  nobody spoke. It counts time in milliseconds of audio, not wall time, so it is
  deterministic under test.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from roco.voice.audio import frame_rms
from roco.voice.endpoint import Endpoint, NoiseFloor

log = logging.getLogger(__name__)

SAMPLE_RATE = 16_000
WINDOW = 512  # Silero at 16 kHz
WINDOW_MS = 1000 * WINDOW / SAMPLE_RATE  # 32 ms


class Vad(Protocol):
    name: str

    def scores(self, frame: Any) -> list[tuple[float, float]]:
        """Speech probabilities 0..1 for the audio in ``frame``, as (prob, ms) pairs."""

    def reset(self) -> None: ...


class SileroVad:
    """Silero VAD through onnxruntime, streaming over arbitrary frame sizes."""

    name = "silero"

    def __init__(self, model_path: Path, threads: int = 1) -> None:
        import numpy as np
        import onnxruntime as ort

        if not Path(model_path).is_file():
            raise FileNotFoundError(f"Silero VAD model missing: {model_path}. "
                                    "Run: roco download-models")
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = threads
        opts.inter_op_num_threads = 1
        opts.log_severity_level = 3
        self._np = np
        self._session = ort.InferenceSession(
            str(model_path), sess_options=opts, providers=["CPUExecutionProvider"]
        )
        inputs = {i.name for i in self._session.get_inputs()}
        self.version = 5 if "state" in inputs else 4
        self._sr = np.array(SAMPLE_RATE, dtype=np.int64)
        self.reset()

    def reset(self) -> None:
        np = self._np
        self._pending = np.zeros(0, dtype=np.float32)
        self._context = np.zeros(64, dtype=np.float32)  # v5 wants the previous 64 samples
        if self.version == 5:
            self._state = np.zeros((2, 1, 128), dtype=np.float32)
        else:
            self._h = np.zeros((2, 1, 64), dtype=np.float32)
            self._c = np.zeros((2, 1, 64), dtype=np.float32)

    def _window(self, x: Any) -> float:
        np = self._np
        if self.version == 5:
            inp = np.concatenate([self._context, x])[None, :]
            out, self._state = self._session.run(
                None, {"input": inp, "state": self._state, "sr": self._sr})
            self._context = x[-64:]
        else:
            out, self._h, self._c = self._session.run(
                None, {"input": x[None, :], "sr": self._sr, "h": self._h, "c": self._c})
        return float(out.reshape(-1)[0])

    def scores(self, frame: Any) -> list[tuple[float, float]]:
        np = self._np
        x = np.asarray(frame, dtype=np.int16).astype(np.float32) / 32768.0
        buf = np.concatenate([self._pending, x]) if len(self._pending) else x
        out = []
        i = 0
        while i + WINDOW <= len(buf):
            out.append((self._window(buf[i : i + WINDOW]), WINDOW_MS))
            i += WINDOW
        self._pending = buf[i:].copy()
        return out


class EnergyVad:
    """Energy detector with an adaptive noise floor. Probability is 1.0 or 0.0 per frame."""

    name = "energy"

    def __init__(self, margin: float = 2.5, min_threshold: float = 300.0,
                 frame_ms: float = 80.0, noise_floor: NoiseFloor | None = None) -> None:
        self.floor = noise_floor or NoiseFloor()
        self.margin = margin
        self.min_threshold = min_threshold
        self.frame_ms = frame_ms

    @property
    def threshold(self) -> float:
        return max(self.min_threshold, self.floor.level * self.margin)

    def scores(self, frame: Any) -> list[tuple[float, float]]:
        rms = frame_rms(frame)
        loud = rms >= self.threshold
        if not loud:
            self.floor.update(rms)
        return [(1.0 if loud else 0.0, self.frame_ms)]

    def reset(self) -> None:
        pass


def create_vad(engine: str, model_path: Path | None) -> Vad:
    """``silero`` (falls back to energy with a warning if the model/onnxruntime is
    missing) or ``energy``."""
    if engine == "silero" and model_path is not None:
        try:
            return SileroVad(model_path)
        except (ImportError, FileNotFoundError) as exc:
            log.warning("Silero VAD unavailable (%s); using the energy detector", exc)
    elif engine not in ("silero", "energy"):
        raise ValueError(f"Unknown VAD engine {engine!r} (silero | energy)")
    return EnergyVad()


# --------------------------------------------------------------------------- endpointer


@dataclass
class EndpointSettings:
    threshold: float = 0.5  # a window is speech at or above this probability
    neg_threshold: float = 0.35  # ...and silence only below this (hysteresis)
    start_ms: float = 96.0  # speech needed before a turn counts as started
    trailing_ms: float = 450.0  # silence after speech that ends the turn
    max_s: float = 15.0  # hard cap on a turn
    start_timeout_s: float = 4.0  # nobody spoke: give up

    @classmethod
    def from_config(cls, vad: dict, **overrides: Any) -> EndpointSettings:
        s = cls()
        for key in ("threshold", "neg_threshold", "start_ms", "trailing_ms", "max_s",
                    "start_timeout_s"):
            if key in vad and vad[key] is not None:
                setattr(s, key, float(vad[key]))
        for key, value in overrides.items():
            setattr(s, key, float(value))
        return s


class VadEndpointer:
    """Frame-by-frame turn decisions from VAD scores. Feed every mic frame once.

    Events (:class:`roco.voice.endpoint.Endpoint`): ``SPEECH_START`` once, then
    ``SILENCE`` (end of speech), ``MAX_LENGTH`` or, before any speech,
    ``NO_SPEECH``; ``CONTINUE`` otherwise.
    """

    def __init__(self, settings: EndpointSettings | None = None) -> None:
        self.s = settings or EndpointSettings()
        self.elapsed_ms = 0.0
        self.speech_started = False
        self.speech_ms = 0.0  # total speech heard
        self.silence_ms = 0.0  # silence since the last speech window
        self._run_ms = 0.0  # current run of speech windows (for the start decision)
        self._in_speech = False
        self.onset_ms: float | None = None  # where the run that started the turn began

    @property
    def trailing_ms(self) -> int:
        return int(round(self.silence_ms))

    def feed(self, scores: list[tuple[float, float]]) -> Endpoint:
        s = self.s
        started_now = False
        for prob, ms in scores:
            self.elapsed_ms += ms
            speech = prob >= s.threshold or (self._in_speech and prob >= s.neg_threshold)
            self._in_speech = speech
            if speech:
                if self._run_ms == 0.0:
                    run_start = self.elapsed_ms - ms
                    if not self.speech_started:
                        self.onset_ms = run_start
                self._run_ms += ms
                self.speech_ms += ms
                self.silence_ms = 0.0
            else:
                self._run_ms = 0.0
                self.silence_ms += ms
            if not self.speech_started and self._run_ms >= s.start_ms:
                self.speech_started = started_now = True
        if started_now:
            return Endpoint.SPEECH_START
        if not self.speech_started:
            if self.elapsed_ms >= s.start_timeout_s * 1000 or self.elapsed_ms >= s.max_s * 1000:
                return Endpoint.NO_SPEECH
            return Endpoint.CONTINUE
        if self.elapsed_ms >= s.max_s * 1000:
            return Endpoint.MAX_LENGTH
        if self.silence_ms >= s.trailing_ms:
            return Endpoint.SILENCE
        return Endpoint.CONTINUE
