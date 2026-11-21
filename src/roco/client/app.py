"""``roco body`` (connect to a brain server) and ``roco run`` (brain + body in one process)."""

from __future__ import annotations

import importlib
import logging
import sys
import threading
from collections.abc import Callable
from typing import Any

from roco.bus import EventBus
from roco.config import Config
from roco.protocol import Connection, memory_pipe

log = logging.getLogger(__name__)


def preflight(cfg: Config) -> str | None:
    """Fail fast (before loading a brain) when the body cannot run. Returns the error."""
    from roco.client.factory import require_body_extra
    from roco.voice.factory import VoiceUnavailable
    from roco.voice.models import wakeword_files

    try:
        require_body_extra()
    except VoiceUnavailable as exc:
        return str(exc)
    missing = [p.name for p in wakeword_files(cfg) if not p.is_file()]
    if missing:
        return ("Model files missing: " + ", ".join(missing)
                + ". Download them with: roco download-models")
    return None


def _start(cfg: Config, connect: Callable[[], Connection], face: bool, banner: str) -> int:
    from roco.client.factory import build_body
    from roco.runtime import run_with_face
    from roco.voice.factory import VoiceUnavailable
    from roco.voice.models import wake_phrase

    bus = EventBus()
    try:
        parts = build_body(cfg, bus, connect, face=face)
    except (VoiceUnavailable, FileNotFoundError, ValueError, OSError) as exc:
        print(f"Could not start the body: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        if exc.__class__.__name__ != "PortAudioError":
            raise
        print(f"Audio device error: {exc}", file=sys.stderr)
        return 1
    print(banner)
    print(f'Say "{wake_phrase(cfg)}". Press Ctrl+C to quit.')
    try:
        run_with_face(parts.face, parts.client.run)
    finally:
        parts.close()
    return 0


def run_body(cfg: Config, url: str | None = None, face: bool = True) -> int:
    """``roco body``: the robot side, talking to ``roco brain`` over a WebSocket."""
    from roco.client.link import websocket_connector

    if error := preflight(cfg):
        print(error, file=sys.stderr)
        return 1
    url = url or str(cfg.get_path("link.brain_url", "ws://127.0.0.1:8765"))
    return _start(cfg, websocket_connector(url), face, f"ROCO body: brain at {url}")


# --------------------------------------------------------------------------- roco run


def load_server() -> Any:
    """Workstream A's in-process server API (``create_brain_context``, ``serve_connection``)."""
    from roco.voice.factory import ComponentUnavailable

    try:
        server = importlib.import_module("roco.server")
    except ModuleNotFoundError as exc:
        if exc.name and "roco.server".startswith(exc.name):
            raise ComponentUnavailable(
                "roco run needs the brain server (roco.server, workstream A)") from exc
        raise
    for name in ("create_brain_context", "serve_connection"):
        if not callable(getattr(server, name, None)):
            raise ComponentUnavailable(f"roco.server.{name} is missing")
    return server


class LocalBrain:
    """The brain half of ``roco run``: every body connection gets a fresh
    :func:`memory_pipe` with ``serve_connection`` running on the other end."""

    def __init__(self, server: Any, ctx: Any) -> None:
        self.server = server
        self.ctx = ctx
        self.stop = threading.Event()
        self.threads: list[threading.Thread] = []

    def connect(self) -> Connection:
        if self.stop.is_set():
            raise ConnectionError("brain stopped")
        body_end, brain_end = memory_pipe()

        def serve() -> None:
            try:
                self.server.serve_connection(brain_end, self.ctx, self.stop)
            except Exception:
                log.exception("Brain session crashed")
            finally:
                brain_end.close()

        thread = threading.Thread(target=serve, name="roco-brain-session", daemon=True)
        thread.start()
        self.threads = [t for t in self.threads if t.is_alive()] + [thread]
        return body_end

    def close(self) -> None:
        self.stop.set()
        for t in self.threads:
            t.join(timeout=3)
        close = getattr(self.ctx, "close", None)
        if callable(close):
            close()


def run_local(cfg: Config, face: bool = True, server: Any = None) -> int:
    """``roco run``: brain and body in one process, joined by ``memory_pipe()``."""
    from roco.voice.factory import ComponentUnavailable

    if error := preflight(cfg):
        print(error, file=sys.stderr)
        return 1
    try:
        server = server or load_server()
        print("Loading the brain (STT, LLM, TTS)...")
        ctx = server.create_brain_context(cfg)
    except ComponentUnavailable as exc:
        print(exc, file=sys.stderr)
        return 1
    except RuntimeError as exc:  # missing models / deps, reported by the server
        print(f"Could not start the brain: {exc}", file=sys.stderr)
        return 1
    brain = LocalBrain(server, ctx)
    try:
        return _start(cfg, brain.connect, face, "ROCO is running (brain + body in one process).")
    finally:
        brain.close()
