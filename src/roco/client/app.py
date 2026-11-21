"""``roco body`` (connect to a brain server) and ``roco run`` (brain + body in one process)."""

from __future__ import annotations

import importlib
import logging
import sys
import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any

from roco.bus import EventBus
from roco.config import Config
from roco.protocol import Connection, memory_pipe

log = logging.getLogger(__name__)


def process_rss_mb() -> tuple[float, float] | None:
    """(current, peak) resident memory of this process in MB, or None if unknown."""
    try:
        import psutil

        info = psutil.Process().memory_info()
        peak = getattr(info, "peak_wset", 0) or info.rss
        return info.rss / 2**20, peak / 2**20
    except ImportError:
        pass
    if sys.platform.startswith("linux"):
        try:
            fields = dict(line.split(":", 1) for line in
                          Path("/proc/self/status").read_text(encoding="ascii").splitlines())
            kb = lambda k: float(fields[k].split()[0]) / 1024  # noqa: E731
            return kb("VmRSS"), kb("VmHWM")
        except (OSError, KeyError, ValueError):
            return None
    if sys.platform == "win32":
        import ctypes
        from ctypes import wintypes

        class Counters(ctypes.Structure):
            _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
                        ("PeakWorkingSetSize", ctypes.c_size_t),
                        ("WorkingSetSize", ctypes.c_size_t)] + [
                (n, ctypes.c_size_t) for n in ("a", "b", "c", "d", "e", "f")]

        c = Counters()
        c.cb = ctypes.sizeof(c)
        proc = ctypes.windll.kernel32.GetCurrentProcess()
        if ctypes.windll.psapi.GetProcessMemoryInfo(proc, ctypes.byref(c), c.cb):
            return c.WorkingSetSize / 2**20, c.PeakWorkingSetSize / 2**20
    return None


def _log_rss(when: str) -> None:
    rss = process_rss_mb()
    if rss:
        log.info("Memory %s: %.0f MB resident (peak %.0f MB)", when, *rss)


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
    _log_rss("at start")
    try:
        run_with_face(parts.face, parts.client.run)
    finally:
        _log_rss("at exit")
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
