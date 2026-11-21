"""``roco brain``: the WebSocket server the body connects to (docs/PROTOCOL.md)."""

from __future__ import annotations

import logging
import threading
from typing import Any

from roco.config import Config
from roco.protocol import WebSocketConnection
from roco.server.context import BrainContext, BrainUnavailable, create_brain_context
from roco.server.session import serve_connection

log = logging.getLogger(__name__)

MAX_FRAME_BYTES = 4 * 1024 * 1024


def make_server(ctx: BrainContext, host: str, port: int, stop: threading.Event) -> Any:
    """A ``websockets.sync`` server that runs :func:`serve_connection` per client.

    Each client gets its own thread; a second body is refused with ``error{busy}``.
    Compression is off: PCM barely compresses and deflate adds CPU and latency.
    """
    from websockets.sync.server import serve

    def handler(ws: Any) -> None:
        log.info("Connection from %s", getattr(ws, "remote_address", "?"))
        serve_connection(WebSocketConnection(ws), ctx, stop)

    return serve(handler, host, port, compression=None, max_size=MAX_FRAME_BYTES)


def run_server(cfg: Config, stop: threading.Event | None = None,
               ctx: BrainContext | None = None) -> int:
    """Serve until ``stop`` is set (or Ctrl+C). Returns a process exit code."""
    stop = stop or threading.Event()
    host = str(cfg.get_path("link.host", "0.0.0.0"))
    port = int(cfg.get_path("link.port", 8765))
    own_ctx = ctx is None
    if ctx is None:
        print("Loading the brain, speech-to-text and voice (first start takes a while)...")
        try:
            ctx = create_brain_context(cfg)
        except BrainUnavailable as exc:
            print(exc)
            return 1
        print("Waiting for Ollama to load the model...")
        if not ctx.wait_until_ready():
            print("The model is still loading; the first answer may be slow.")
    try:
        server = make_server(ctx, host, port, stop)
    except OSError as exc:
        print(f"Could not listen on {host}:{port}: {exc}")
        if own_ctx:
            ctx.close()
        return 1

    def shutdown_on_stop() -> None:
        stop.wait()
        server.shutdown()

    threading.Thread(target=shutdown_on_stop, name="roco-brain-stop", daemon=True).start()
    print(f"ROCO brain listening on ws://{host}:{port} ({ctx.brain_name}). Ctrl+C to quit.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        stop.set()
        server.shutdown()
        if own_ctx:
            ctx.close()
    return 0
