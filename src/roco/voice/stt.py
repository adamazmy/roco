"""faster-whisper speech-to-text. The model is loaded once and reused.

``WhisperTranscriber(frames)`` transcribes a whole utterance (Phase 2 voice loop).
:meth:`WhisperTranscriber.segments` returns timed segments for raw PCM16 bytes; the
brain server's streaming decoder (``roco.server.stt``) uses them to commit the start
of a long utterance while Adam is still talking.
"""

from __future__ import annotations

import logging
import os
import threading
from collections.abc import Sequence
from typing import Any, NamedTuple

log = logging.getLogger(__name__)


class Segment(NamedTuple):
    """A piece of a transcript, with times in seconds from the start of the audio."""

    start: float
    end: float
    text: str


class WhisperTranscriber:
    def __init__(
        self,
        model: str = "base.en",
        device: str = "cpu",
        compute_type: str = "int8",
        language: str | None = "en",
        vad_filter: bool = True,
        beam_size: int = 1,
        download_root: str | None = None,
        num_workers: int = 2,
    ) -> None:
        self.model_name = model
        self.device = device
        self.compute_type = compute_type
        self.language = language
        self.vad_filter = vad_filter
        self.beam_size = beam_size
        self.download_root = download_root
        # Two workers: the brain server's final decode can run while a background
        # (speculative/partial) decode is still finishing, instead of queueing behind it.
        self.num_workers = num_workers
        self._model: Any = None
        self._lock = threading.Lock()

    def load(self) -> Any:
        with self._lock:
            if self._model is None:
                os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
                from faster_whisper import WhisperModel

                log.info("Loading faster-whisper %s (%s, %s)", self.model_name, self.device,
                         self.compute_type)
                self._model = WhisperModel(
                    self.model_name, device=self.device, compute_type=self.compute_type,
                    download_root=self.download_root, num_workers=self.num_workers,
                )
            return self._model

    def warm_up(self) -> None:
        """Run one tiny transcription so the first real one is not slower."""
        import numpy as np

        self.transcribe_array(np.zeros(16000, dtype=np.float32))

    def transcribe_array(self, audio: Any) -> str:
        model = self.load()
        segments, _info = model.transcribe(
            audio,
            language=self.language,
            beam_size=self.beam_size,
            vad_filter=self.vad_filter,
            condition_on_previous_text=False,
        )
        return " ".join(seg.text.strip() for seg in segments).strip()

    def segments(self, pcm: bytes, rate: int = 16000) -> list[Segment]:
        """Timed segments for PCM16 little-endian mono ``pcm`` (must be 16 kHz)."""
        import numpy as np

        if rate != 16000:
            raise ValueError(f"faster-whisper needs 16 kHz audio, got {rate} Hz")
        if not pcm:
            return []
        audio = np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768.0
        model = self.load()
        found, _info = model.transcribe(
            audio,
            language=self.language,
            beam_size=self.beam_size,
            vad_filter=self.vad_filter,
            condition_on_previous_text=False,
        )
        return [Segment(float(s.start), float(s.end), s.text.strip()) for s in found]

    def __call__(self, frames: Sequence[Any]) -> str:
        import numpy as np

        if not frames:
            return ""
        audio = np.concatenate([np.asarray(f, dtype=np.int16) for f in frames])
        return self.transcribe_array(audio.astype(np.float32) / 32768.0)
