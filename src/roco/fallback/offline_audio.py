"""The canned "my brain's offline" line, pre-rendered to a WAV at setup time.

Spec §11 says ROCO is never fully unresponsive: if the body's local TTS cannot load on
the Pi (missing voice, out of memory), the body client can still play this file.
``scripts/pi_setup.sh`` renders it once via ``roco fallback-prepare``; it is not committed.

The file name carries a hash of the line and the ``voice.tts`` settings, so changing the
voice or the wording makes :func:`find_offline_wav` return ``None`` until it is rendered
again, instead of playing a line in the old voice.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import logging
import os
import wave
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

from roco.config import Config
from roco.paths import models_dir

log = logging.getLogger(__name__)

# text -> iterable of (pcm16 mono little-endian bytes, sample_rate), like PiperTTS
TtsFn = Callable[[str], Iterable[tuple[bytes, int]]]


def _voice_fingerprint(cfg: Config, text: str) -> str:
    tts = cfg.get_path("voice.tts", {}) or {}
    blob = json.dumps({"text": text, "tts": tts}, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:12]


def offline_wav_dir() -> Path:
    return models_dir() / "fallback"


def offline_wav_path(cfg: Config, text: str) -> Path:
    """Where the WAV for ``text`` in the configured voice lives (it may not exist)."""
    return offline_wav_dir() / f"offline-{_voice_fingerprint(cfg, text)}.wav"


def find_offline_wav(cfg: Config, text: str) -> Path | None:
    """The rendered WAV if it matches the current line and voice, else ``None``. Cheap."""
    path = offline_wav_path(cfg, text)
    return path if path.is_file() and path.stat().st_size > 44 else None


def render_offline_wav(cfg: Config, text: str, tts: TtsFn | None = None,
                       force: bool = False) -> Path:
    """Synthesize ``text`` once into a mono 16-bit WAV (idempotent). Raises on TTS failure."""
    path = offline_wav_path(cfg, text)
    if path.is_file() and not force:
        return path
    if tts is None:
        from roco.voice.factory import create_tts

        tts = create_tts(cfg)
    pcm = bytearray()
    rate: int | None = None
    for chunk, chunk_rate in tts(text):
        if rate is None:
            rate = int(chunk_rate)
        elif int(chunk_rate) != rate:
            raise ValueError(f"TTS changed sample rate mid-line ({rate} -> {chunk_rate})")
        pcm += chunk
    if rate is None or not pcm:
        raise ValueError("TTS produced no audio for the offline line")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".wav.tmp")
    with wave.open(str(tmp), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(rate)
        wav.writeframes(bytes(pcm[: len(pcm) - len(pcm) % 2]))
    os.replace(tmp, path)
    for old in path.parent.glob("offline-*.wav"):  # renders for an older voice/line
        if old != path:
            with contextlib.suppress(OSError):
                old.unlink()
    return path


def prepare(cfg: Config, *, tts: Any = None, printer: Callable[[str], None] = print) -> bool:
    """``roco fallback-prepare``: fetch the fallback STT model and render the offline WAV."""
    from roco.fallback.chain import OFFLINE_LINE
    from roco.fallback.stt import download_stt_model

    ok = download_stt_model(cfg)
    try:
        path = render_offline_wav(cfg, OFFLINE_LINE, tts=tts)
        printer(f"[OK  ] offline line: {path}")
    except Exception as exc:  # missing piper voice / extra: say how to fix, keep going
        printer(f"[FAIL] offline line: {exc} (run `roco download-models` first)")
        ok = False
    return ok
