"""Transports for the link protocol: one interface, two implementations.

- :class:`WebSocketConnection` wraps a ``websockets`` *sync* connection (client or
  server side; both have the same API). Used by ``roco brain`` / ``roco body``.
- :func:`memory_pipe` returns two connected in-process ends. Used by ``roco run``
  (brain and body in one process) and by tests. Every message is encoded and
  decoded on the way through, exactly like on the wire, so the in-process path
  exercises the same codec.

The API is blocking and thread-safe for one sender thread plus one receiver thread
(the codebase is threaded, not asyncio). ``recv`` returns ``None`` on timeout and
raises :class:`ConnectionClosed` once the link is gone.
"""

from __future__ import annotations

import contextlib
import hmac
import queue
import secrets
import threading
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from roco.protocol.audio import AudioFrame, decode_audio, encode_audio
from roco.protocol.messages import Message, ProtocolError, decode, encode

Frame = Message | AudioFrame


class ConnectionClosed(Exception):
    """The peer went away or :meth:`Connection.close` was called."""


def encode_frame(frame: Frame) -> str | bytes:
    return encode_audio(frame) if isinstance(frame, AudioFrame) else encode(frame)


def decode_frame(data: str | bytes) -> Frame:
    """Text frames are control messages, binary frames are audio."""
    if isinstance(data, str):
        return decode(data)
    if isinstance(data, bytes | bytearray | memoryview):
        return decode_audio(bytes(data))
    raise ProtocolError(f"unexpected frame type {type(data).__name__}")


class Connection(ABC):
    @abstractmethod
    def send(self, frame: Frame) -> None:
        """Send one message or audio frame. Raises ConnectionClosed if closed."""

    @abstractmethod
    def recv(self, timeout: float | None = None) -> Frame | None:
        """Next frame, or None after ``timeout`` seconds (None = wait forever).

        Raises ConnectionClosed when the link is closed, ProtocolError for a frame
        that cannot be decoded (the connection stays usable).
        """

    @abstractmethod
    def close(self) -> None:
        """Close the link. Safe to call twice."""

    @property
    @abstractmethod
    def closed(self) -> bool: ...


# --------------------------------------------------------------------------- in-memory
_CLOSE = object()


class _MemoryEnd(Connection):
    def __init__(self, inbox: queue.Queue[Any], outbox: queue.Queue[Any]) -> None:
        self._inbox = inbox
        self._outbox = outbox
        self._closed = threading.Event()
        self.peer: _MemoryEnd | None = None

    def send(self, frame: Frame) -> None:
        if self._closed.is_set():
            raise ConnectionClosed("connection closed")
        self._outbox.put(encode_frame(frame))

    def recv(self, timeout: float | None = None) -> Frame | None:
        if self._closed.is_set() and self._inbox.empty():
            raise ConnectionClosed("connection closed")
        try:
            data = self._inbox.get(timeout=timeout)
        except queue.Empty:
            return None
        if data is _CLOSE:
            self._closed.set()
            raise ConnectionClosed("peer closed the connection")
        return decode_frame(data)

    def close(self) -> None:
        if self._closed.is_set():
            return
        self._closed.set()
        self._outbox.put(_CLOSE)  # wakes the peer's recv
        self._inbox.put(_CLOSE)  # wakes our own blocked recv

    @property
    def closed(self) -> bool:
        return self._closed.is_set()


def memory_pipe() -> tuple[Connection, Connection]:
    """Two connected ends: whatever one sends, the other receives."""
    a_to_b: queue.Queue[Any] = queue.Queue()
    b_to_a: queue.Queue[Any] = queue.Queue()
    a = _MemoryEnd(inbox=b_to_a, outbox=a_to_b)
    b = _MemoryEnd(inbox=a_to_b, outbox=b_to_a)
    a.peer, b.peer = b, a
    return a, b


# --------------------------------------------------------------------------- websocket
class WebSocketConnection(Connection):
    """Adapter around ``websockets.sync`` ``ClientConnection``/``ServerConnection``."""

    def __init__(self, ws: Any) -> None:
        self._ws = ws
        self._send_lock = threading.Lock()
        self._closed = False

    def send(self, frame: Frame) -> None:
        from websockets.exceptions import ConnectionClosed as WsClosed

        if self._closed:
            raise ConnectionClosed("connection closed")
        data = encode_frame(frame)
        try:
            with self._send_lock:
                self._ws.send(data)
        except WsClosed as exc:
            self._closed = True
            raise ConnectionClosed(str(exc)) from exc

    def recv(self, timeout: float | None = None) -> Frame | None:
        from websockets.exceptions import ConnectionClosed as WsClosed

        if self._closed:
            raise ConnectionClosed("connection closed")
        try:
            data = self._ws.recv(timeout=timeout)
        except TimeoutError:
            return None
        except WsClosed as exc:
            self._closed = True
            raise ConnectionClosed(str(exc)) from exc
        return decode_frame(data)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        with contextlib.suppress(Exception):  # closing must never raise
            self._ws.close()

    @property
    def closed(self) -> bool:
        return self._closed


# --------------------------------------------------------------------------- auth
def load_or_create_token(path: Path) -> str:
    """The shared secret both sides send/check in ``hello``.

    Created on first use with restrictive permissions where the OS supports them.
    Copy the same file to the Pi (``~/.roco/link_token``); never commit it.
    """
    path = Path(path)
    if path.exists():
        token = path.read_text(encoding="utf-8").strip()
        if token:
            return token
    path.parent.mkdir(parents=True, exist_ok=True)
    token = secrets.token_urlsafe(32)
    path.write_text(token + "\n", encoding="utf-8")
    with contextlib.suppress(OSError):  # Windows ignores most modes
        path.chmod(0o600)
    return token


def token_matches(expected: str, given: str | None) -> bool:
    return bool(expected) and hmac.compare_digest(expected.encode(), (given or "").encode())
