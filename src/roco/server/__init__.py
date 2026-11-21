"""The brain server: one body session at a time, speech in, speech out (Phase 3, workstream A).

Public API (``roco brain`` and ``roco run`` are both built on it; keep these names):

- ``ctx = create_brain_context(cfg)``: loads everything once (brain + tools + reminder
  scheduler, Ollama preload, faster-whisper STT, TTS, pre-synthesized filler lines,
  link token). Pass ``brain=``, ``stt=``, ``tts=``, ``bus=``, ``token=`` to inject your
  own (tests, ``roco run`` sharing a bus). ``ctx.close()`` at shutdown. Raises
  :class:`BrainUnavailable` (a ``RuntimeError``) when speech deps/models are missing.
  The Ollama model keeps loading in the background; ``ctx.wait_until_ready()`` waits.
- ``serve_connection(conn, ctx, stop)``: runs one body session on any
  :class:`roco.protocol.Connection` (a ``WebSocketConnection`` for ``roco brain``, a
  ``memory_pipe()`` end for ``roco run``) until the connection closes or ``stop`` is
  set. Blocking; call it on its own thread. It does the full handshake of
  docs/PROTOCOL.md (version, token, one body at a time) even in-process.
- ``run_server(cfg, stop=None, ctx=None)``: ``roco brain``, a WebSocket server on
  ``link.host:link.port`` that calls ``serve_connection`` per client.
- ``ctx.on_settings(fn)`` / ``ctx.apply_settings(values)``: the settings the body may
  change; ``fn(full_settings)`` runs after every change (personality hooks in here).

What a session does (details in :mod:`roco.server.session`):

- Mic frames are decoded while Adam talks (:mod:`roco.server.stt`); ``turn_end``
  finishes the transcript, usually from a decode that already ran.
- Brain tokens -> sentences -> TTS per sentence -> ``sentence`` then its audio frames
  (``server.tts_frame_ms`` blocks, then an empty END frame), sent as soon as they exist.
- A slow tool about to run with nothing said yet gets an instant filler line.
- ``interrupt{turn}`` stops generation and TTS within one chunk/frame; the history then
  keeps only ``playback_done.heard``.
- Reminders (bus ``announce``) are spoken as ``announce`` + sentences + audio with a
  :func:`roco.protocol.brain_turn_id`, once no turn is recording or playing.
- One latency line per turn (:mod:`roco.server.latency`), also appended to
  ``~/.roco/latency.jsonl``.
"""

from roco.server.context import (
    BrainContext,
    BrainUnavailable,
    FillerLine,
    create_brain_context,
)
from roco.server.latency import LatencyLog, TurnLatency
from roco.server.session import Session, serve_connection
from roco.server.ws import run_server

__all__ = [
    "BrainContext",
    "BrainUnavailable",
    "FillerLine",
    "LatencyLog",
    "Session",
    "TurnLatency",
    "create_brain_context",
    "run_server",
    "serve_connection",
]
