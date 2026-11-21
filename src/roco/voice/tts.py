"""Piper text-to-speech, with an adapter over the two piper-tts Python APIs.

- piper-tts <= 1.2: ``PiperVoice.synthesize_stream_raw(text, length_scale=...)``
  yields raw int16 bytes per sentence (``voice.config.sample_rate``), and
  ``PiperVoice.synthesize(text, wav_file, ...)`` writes a WAV file.
- piper-tts >= 1.3: ``PiperVoice.synthesize(text, syn_config=SynthesisConfig(...))``
  yields ``AudioChunk`` objects (``audio_int16_bytes``, ``sample_rate``).

The API is detected on the loaded voice object at runtime, so either release works.
"""

from __future__ import annotations

import inspect
import io
import logging
import threading
import wave
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

Audio = tuple[bytes, int]  # (int16 mono PCM, sample rate)

# Smoother than the voices' own defaults (noise 0.667, noise_w 0.8), which sound choppy.
# Chosen by ear in A/B listening tests: 0.5 / 0.333 in Phase 2, then 0.4 / 0.25 with
# length_scale 1.08 and roco.voice.postfx in the Phase 3 A/B (scripts/voice_ab.py).
DEFAULT_NOISE_SCALE = 0.4
DEFAULT_NOISE_W = 0.25


@dataclass(frozen=True)
class Synthesis:
    """Voice settings; None means "use the voice's own default"."""

    length_scale: float | None = None
    noise_scale: float | None = None
    noise_w: float | None = None

    def old_kwargs(self) -> dict[str, float]:
        """Keyword arguments for piper-tts <= 1.2."""
        pairs = {"length_scale": self.length_scale, "noise_scale": self.noise_scale,
                 "noise_w": self.noise_w}
        return {k: v for k, v in pairs.items() if v is not None}

    def is_default(self) -> bool:
        return not self.old_kwargs()


def piper_api(voice: Any) -> str:
    """Return ``"stream_raw"``, ``"wav_file"`` (old APIs) or ``"chunks"`` (>= 1.3)."""
    if hasattr(voice, "synthesize_stream_raw"):
        return "stream_raw"
    try:
        params = inspect.signature(voice.synthesize).parameters
    except (TypeError, ValueError):
        params = {}
    if "wav_file" in params:
        return "wav_file"
    return "chunks"


def _make_syn_config(settings: Synthesis) -> Any:
    if settings.is_default():
        return None
    try:
        from piper import SynthesisConfig
    except ImportError:  # pragma: no cover - only very unusual builds
        from piper.config import SynthesisConfig
    return SynthesisConfig(
        length_scale=settings.length_scale,
        noise_scale=settings.noise_scale,
        noise_w_scale=settings.noise_w,
    )


def iter_piper_audio(
    voice: Any,
    text: str,
    settings: Synthesis | None = None,
    syn_config_factory: Any = _make_syn_config,
) -> Iterator[Audio]:
    """Synthesize ``text`` with any piper-tts version, yielding (pcm_bytes, sample_rate)."""
    settings = settings or Synthesis()
    api = piper_api(voice)
    if api == "stream_raw":
        rate = int(voice.config.sample_rate)
        for raw in voice.synthesize_stream_raw(text, **settings.old_kwargs()):
            if raw:
                yield bytes(raw), rate
    elif api == "wav_file":
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wav:
            voice.synthesize(text, wav, **settings.old_kwargs())
        buf.seek(0)
        with wave.open(buf, "rb") as wav:
            rate = wav.getframerate()
            pcm = wav.readframes(wav.getnframes())
        if pcm:
            yield pcm, rate
    else:
        syn_config = syn_config_factory(settings)
        for chunk in voice.synthesize(text, syn_config=syn_config):
            pcm = chunk.audio_int16_bytes
            if pcm:
                yield bytes(pcm), int(chunk.sample_rate)


class PiperTTS:
    """Loads a Piper voice once (thread-safe, lazily) and synthesizes sentences."""

    def __init__(self, model_path: Path, config_path: Path | None = None,
                 settings: Synthesis | None = None) -> None:
        self.model_path = Path(model_path)
        self.config_path = Path(config_path) if config_path else None
        self.settings = settings or Synthesis()
        self._voice: Any = None
        self._lock = threading.Lock()

    def load(self) -> Any:
        with self._lock:
            if self._voice is None:
                from piper import PiperVoice

                if not self.model_path.is_file():
                    raise FileNotFoundError(
                        f"Piper voice not found: {self.model_path}. Run: roco download-models"
                    )
                log.info("Loading Piper voice %s", self.model_path.name)
                self._voice = PiperVoice.load(
                    str(self.model_path),
                    config_path=str(self.config_path) if self.config_path else None,
                )
                log.debug("Piper API: %s", piper_api(self._voice))
            return self._voice

    def __call__(self, text: str) -> Iterator[Audio]:
        voice = self.load()
        yield from iter_piper_audio(voice, text, self.settings)
