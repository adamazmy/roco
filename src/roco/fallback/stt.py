"""Speech-to-text on the body for the fallback: faster-whisper ``tiny.en``.

Nothing is imported or loaded until the first :meth:`LazyStt.transcribe`, so the body
client pays no RAM for it while the PC brain is up. ``tiny.en`` int8 needs roughly
100-150 MB once loaded, which fits next to the body client on a Pi 5 (2 GB).
The model files live in ``~/.roco/models/whisper`` (same cache as ``roco download-models``);
``scripts/pi_setup.sh`` downloads them with :func:`download_stt_model`.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable
from typing import Any

from roco.config import Config, deep_merge

log = logging.getLogger(__name__)

DEFAULT_STT_MODEL = "tiny.en"
WHISPER_RATE = 16000

# (float32 samples at 16 kHz) -> text
TranscribeFn = Callable[[Any], str]


def pcm16_to_float32(pcm: bytes, rate: int) -> Any:
    """PCM16 little-endian mono at ``rate`` -> float32 numpy array at 16 kHz in [-1, 1]."""
    import numpy as np

    if len(pcm) % 2:
        pcm = pcm[:-1]
    audio = np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768.0
    if rate != WHISPER_RATE and audio.size:
        n_out = max(1, int(round(audio.size * WHISPER_RATE / rate)))
        x_old = np.arange(audio.size, dtype=np.float64) / rate
        x_new = np.arange(n_out, dtype=np.float64) / WHISPER_RATE
        audio = np.interp(x_new, x_old, audio).astype(np.float32)
    return audio


class LazyStt:
    """Loads the whisper model on first use and keeps it for later turns."""

    def __init__(self, model: str = DEFAULT_STT_MODEL, transcribe: TranscribeFn | None = None,
                 download_root: str | None = None) -> None:
        self.model = model
        self.download_root = download_root
        self._transcribe = transcribe
        self._lock = threading.Lock()

    @classmethod
    def from_config(cls, cfg: Config) -> LazyStt:
        return cls(str(cfg.get_path("cloud.stt_model", DEFAULT_STT_MODEL) or DEFAULT_STT_MODEL))

    @property
    def loaded(self) -> bool:
        return self._transcribe is not None

    def _load(self) -> TranscribeFn:
        with self._lock:
            if self._transcribe is None:
                from roco.voice.models import whisper_dir
                from roco.voice.stt import WhisperTranscriber

                root = self.download_root or str(whisper_dir())
                transcriber = WhisperTranscriber(model=self.model, device="cpu",
                                                 compute_type="int8", download_root=root)
                transcriber.load()
                self._transcribe = transcriber.transcribe_array
            return self._transcribe

    def transcribe(self, pcm: bytes, rate: int) -> str:
        """Raises ImportError when faster-whisper/numpy are missing; the chain handles it."""
        if not pcm:
            return ""
        fn = self._load()
        return (fn(pcm16_to_float32(pcm, rate)) or "").strip()


def download_stt_model(cfg: Config) -> bool:
    """Fetch the fallback whisper model into ~/.roco/models/whisper (idempotent)."""
    from roco.voice.models import download_whisper

    model = str(cfg.get_path("cloud.stt_model", DEFAULT_STT_MODEL) or DEFAULT_STT_MODEL)
    sub = Config(deep_merge(cfg, {"voice": {"stt": {"model": model}}}))
    result = download_whisper(sub)
    print(f"[{'OK  ' if result.ok else 'FAIL'}] {result.name}: {result.status}"
          + (f" ({result.detail})" if result.detail else ""))
    return result.ok
