"""Kokoro text-to-speech (kokoro-onnx), a smoother alternative to Piper (spec §3).

Same callable interface as :class:`roco.voice.tts.PiperTTS`: ``tts(text)`` yields
``(int16 mono PCM bytes, sample_rate)`` and ``tts.load()`` loads the model, so the
speaker, the brain server and the A/B script can use either backend.

Runs on the CPU through onnxruntime: the 6 GB GPU is already full with qwen2.5:7b.
Model files live in ``~/.roco/models/kokoro`` (``roco download-models`` with
``voice.tts.engine: kokoro``, or by hand from the kokoro-onnx releases page).
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Iterator
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

Audio = tuple[bytes, int]

KOKORO_RELEASE = (
    "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0"
)
KOKORO_MODEL = "kokoro-v1.0.onnx"  # fp32, 310 MB; kokoro-v1.0.int8.onnx is 88 MB
KOKORO_VOICES = "voices-v1.0.bin"


class KokoroTTS:
    """Loads Kokoro once (thread-safe, lazily) and synthesizes sentences."""

    def __init__(
        self,
        model_path: Path,
        voices_path: Path,
        voice: str = "bf_emma",
        speed: float = 1.0,
        lang: str = "en-gb",
        threads: int | None = None,
    ) -> None:
        self.model_path = Path(model_path)
        self.voices_path = Path(voices_path)
        self.voice = voice
        self.speed = speed
        self.lang = lang
        self.threads = threads
        self._kokoro: Any = None
        self._lock = threading.Lock()

    def load(self) -> Any:
        with self._lock:
            if self._kokoro is None:
                for path in (self.model_path, self.voices_path):
                    if not path.is_file():
                        raise FileNotFoundError(
                            f"Kokoro file not found: {path}. Run: roco download-models"
                        )
                import onnxruntime as ort
                from kokoro_onnx import Kokoro

                options = ort.SessionOptions()
                if self.threads:
                    options.intra_op_num_threads = int(self.threads)
                session = ort.InferenceSession(
                    str(self.model_path), options, providers=["CPUExecutionProvider"]
                )
                log.info("Loading Kokoro %s (voice %s)", self.model_path.name, self.voice)
                self._kokoro = Kokoro.from_session(session, str(self.voices_path))
            return self._kokoro

    def __call__(self, text: str) -> Iterator[Audio]:
        import numpy as np

        kokoro = self.load()
        samples, rate = kokoro.create(text, voice=self.voice, speed=self.speed, lang=self.lang)
        if samples is None or len(samples) == 0:
            return
        pcm = (np.clip(samples, -1.0, 1.0) * 32767.0).astype(np.int16).tobytes()
        yield pcm, int(rate)
