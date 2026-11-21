"""Microphone capture with sounddevice: 16 kHz mono int16, 80 ms frames.

If the device cannot open at 16 kHz (some ALSA hardware devices on the Pi), it is
opened at its default rate and resampled. On Windows the MME host API resamples
for us, which is why :func:`roco.voice.audio.resolve_device` prefers it.
"""

from __future__ import annotations

import contextlib
import logging
import queue
from typing import Any

from roco.voice.audio import FRAME_SAMPLES, SAMPLE_RATE, resample_int16, resolve_device

log = logging.getLogger(__name__)


def find_device(spec: int | str | None, kind: str) -> int | None:
    import sounddevice as sd

    return resolve_device(spec, kind, list(sd.query_devices()))


def device_name(index: int | None, kind: str) -> str:
    import sounddevice as sd

    info = sd.query_devices(index, kind=kind) if index is None else sd.query_devices(index)
    return str(info["name"])


class MicStream:
    """Captures frames on PortAudio's thread into a bounded queue (oldest dropped)."""

    def __init__(
        self,
        device: int | None = None,
        sample_rate: int = SAMPLE_RATE,
        frame_samples: int = FRAME_SAMPLES,
        max_queue_s: float = 20.0,
    ) -> None:
        import numpy as np
        import sounddevice as sd

        self._np = np
        self.device = device
        self.sample_rate = sample_rate
        self.frame_samples = frame_samples
        max_frames = max(1, int(max_queue_s * sample_rate / frame_samples))
        self._q: queue.Queue[Any] = queue.Queue(maxsize=max_frames)
        self._closed = False
        self.overflows = 0

        self.capture_rate = sample_rate
        try:
            sd.check_input_settings(device=device, channels=1, dtype="int16",
                                    samplerate=sample_rate)
        except Exception:
            info = sd.query_devices(device, kind="input")
            self.capture_rate = int(info["default_samplerate"])
            log.warning("Mic cannot capture at %d Hz; capturing at %d Hz and resampling",
                        sample_rate, self.capture_rate)
        blocksize = int(round(frame_samples * self.capture_rate / sample_rate))
        self._pending = np.zeros(0, dtype=np.int16)
        self._stream = sd.InputStream(
            samplerate=self.capture_rate, channels=1, dtype="int16", device=device,
            blocksize=blocksize, callback=self._callback,
        )
        self._stream.start()

    def _callback(self, indata: Any, frames: int, time_info: Any, status: Any) -> None:
        if status:
            self.overflows += 1
        mono = indata[:, 0].copy()
        if self.capture_rate != self.sample_rate:
            mono = resample_int16(mono, self.capture_rate, self.sample_rate)
            self._pending = self._np.concatenate([self._pending, mono])
            while len(self._pending) >= self.frame_samples:
                self._put(self._pending[: self.frame_samples])
                self._pending = self._pending[self.frame_samples :]
        else:
            self._put(mono)

    def _put(self, frame: Any) -> None:
        try:
            self._q.put_nowait(frame)
        except queue.Full:  # nobody is reading (e.g. while thinking): drop the oldest
            with contextlib.suppress(queue.Empty):
                self._q.get_nowait()
            self._q.put_nowait(frame)

    def read(self, timeout: float) -> Any | None:
        if self._closed:
            return None
        try:
            return self._q.get(timeout=timeout)
        except queue.Empty:
            return None

    def flush(self) -> None:
        while True:
            try:
                self._q.get_nowait()
            except queue.Empty:
                return

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            self._stream.stop()
            self._stream.close()
        except Exception:
            log.debug("closing mic stream failed", exc_info=True)
