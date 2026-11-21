"""``roco run``: wires brain, voice and face together and runs the loop."""

from __future__ import annotations

import contextlib
import logging
import sys
import threading
import time
from typing import Any

from roco.bus import EventBus
from roco.config import Config

log = logging.getLogger(__name__)


def run_voice(cfg: Config) -> int:
    """Returns a process exit code (0 ok, 1 failure)."""
    from roco.voice.factory import (
        ComponentUnavailable,
        VoiceUnavailable,
        create_body,
        create_brain,
        create_listener,
        create_speaker,
        require_models,
        require_voice_extra,
    )

    try:
        require_voice_extra()
        require_models(cfg)
    except VoiceUnavailable as exc:
        print(exc, file=sys.stderr)
        return 1

    bus = EventBus()
    try:
        brain = create_brain(cfg, bus)
    except ComponentUnavailable as exc:
        print(exc, file=sys.stderr)
        return 1

    _preload_llm(cfg)  # loads in parallel with the voice models below

    face: Any = None
    body: Any = None
    if cfg.get_path("face.enabled", True):
        try:
            face, body = create_body(cfg, bus)
        except (ComponentUnavailable, ImportError) as exc:
            log.warning("Face unavailable, running headless: %s", exc)

    speaker = listener = None
    try:
        print("Loading voice models (first start can take a few seconds)...")
        speaker = create_speaker(cfg, bus)
        listener = create_listener(cfg, bus, speaker)
    except (VoiceUnavailable, FileNotFoundError, ValueError, OSError) as exc:
        # ValueError: bad device spec; OSError covers sounddevice.PortAudioError.
        print(f"Could not start voice: {exc}", file=sys.stderr)
        _close(listener, speaker, body)
        return 1
    except Exception as exc:
        if exc.__class__.__name__ != "PortAudioError":
            _close(listener, speaker, body)
            raise
        print(f"Audio device error: {exc}", file=sys.stderr)
        _close(listener, speaker, body)
        return 1

    from roco.runtime import run_with_face
    from roco.voice.loop import VoiceLoop

    loop = VoiceLoop(listener, speaker, brain, bus,
                     followup_s=float(cfg.get_path("voice.followup_s", 5.0)))
    from roco.voice.models import wake_phrase

    print(f'ROCO is listening. Say "{wake_phrase(cfg)}". Press Ctrl+C to quit.')
    try:
        run_with_face(face, loop.run)
    finally:
        _close(listener, speaker, body, brain)
    return 0


def _preload_llm(cfg: Config) -> threading.Thread:
    """Ask Ollama to load the model now, so the first answer is not 10+ s late.

    An empty /api/generate request just loads the model (and keeps it loaded for
    ``llm.keep_alive``). Best effort, in the background: errors are only logged, the
    brain reports real problems itself.
    """

    def load() -> None:
        import httpx

        host = str(cfg.get_path("llm.host", "http://localhost:11434")).rstrip("/")
        body = {"model": cfg.get_path("llm.model"),
                "keep_alive": cfg.get_path("llm.keep_alive", "30m")}
        try:
            t0 = time.monotonic()
            httpx.post(f"{host}/api/generate", json=body, timeout=120).raise_for_status()
            log.info("LLM %s loaded in %d ms", body["model"], (time.monotonic() - t0) * 1000)
        except Exception as exc:
            log.warning("Could not preload the LLM: %s", exc)

    thread = threading.Thread(target=load, name="roco-llm-preload", daemon=True)
    thread.start()
    return thread


def _close(*things: Any) -> None:
    for thing in things:
        if thing is not None:
            with contextlib.suppress(Exception):
                thing.close()
