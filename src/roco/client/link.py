"""The body's side of the brain link: connect, handshake, reconnect, ping.

:class:`LinkClient` owns one background thread that keeps a
:class:`roco.protocol.Connection` alive:

    connect() -> send hello -> wait for welcome -> receive loop (+ ping every 5 s)
         ^                                                      |
         +--------- backoff 0.5 s, 1 s, 2 s ... 10 s <----------+ (closed, error, no pong)

``connect`` is injected: ``roco body`` dials a WebSocket, ``roco run`` makes a
:func:`roco.protocol.memory_pipe` and starts the in-process server on the other end,
tests pass fakes. Incoming frames other than ping/pong go to ``on_frame`` on the
link thread; handlers must be quick (queue work elsewhere).
"""

from __future__ import annotations

import logging
import random
import threading
import time
from collections.abc import Callable
from typing import Any

from roco.protocol import (
    PROTOCOL_VERSION,
    Connection,
    ConnectionClosed,
    Error,
    Frame,
    Hello,
    Ping,
    Pong,
    ProtocolError,
    Welcome,
)

log = logging.getLogger(__name__)

Connect = Callable[[], Connection]

FATAL_CODES = ("auth", "version")


class HandshakeError(RuntimeError):
    def __init__(self, code: str, message: str = "") -> None:
        super().__init__(f"{code}: {message}" if message else code)
        self.code = code


class LinkClient:
    def __init__(
        self,
        connect: Connect,
        hello: Callable[[], Hello],
        on_frame: Callable[[Frame], None],
        on_up: Callable[[Welcome], None] | None = None,
        on_down: Callable[[str], None] | None = None,
        *,
        ping_interval_s: float = 5.0,
        pong_timeout_s: float = 15.0,
        handshake_timeout_s: float = 5.0,
        backoff_min_s: float = 0.5,
        backoff_max_s: float = 10.0,
        recv_timeout_s: float = 0.25,
        clock: Callable[[], float] = time.monotonic,
        rng: random.Random | None = None,
    ) -> None:
        self._connect = connect
        self._hello = hello
        self.on_frame = on_frame
        self.on_up = on_up
        self.on_down = on_down
        self.ping_interval_s = ping_interval_s
        self.pong_timeout_s = pong_timeout_s
        self.handshake_timeout_s = handshake_timeout_s
        self.backoff_min_s = backoff_min_s
        self.backoff_max_s = backoff_max_s
        self.recv_timeout_s = recv_timeout_s
        self.clock = clock
        self._rng = rng or random.Random()
        self._stop = threading.Event()
        self._send_lock = threading.Lock()
        self._conn: Connection | None = None
        self._up = threading.Event()
        self._thread: threading.Thread | None = None
        self.welcome: Welcome | None = None
        self.rtt_ms: float | None = None
        self.connects = 0  # successful handshakes
        self.attempts = 0
        self.last_error: str = ""
        self._ping_id = 0
        self._last_ping = 0.0
        self._last_pong = 0.0

    # ------------------------------------------------------------------ public
    def start(self) -> None:
        if self._thread is None:
            self._thread = threading.Thread(target=self.run, name="roco-link", daemon=True)
            self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        conn = self._conn
        if conn is not None:
            conn.close()
        if self._thread is not None and self._thread is not threading.current_thread():
            self._thread.join(timeout=3)

    @property
    def is_up(self) -> bool:
        return self._up.is_set()

    def wait_up(self, timeout: float | None = None) -> bool:
        return self._up.wait(timeout)

    def send(self, frame: Frame) -> bool:
        """Send if the link is up. Returns False (never raises) when it is down."""
        conn = self._conn
        if conn is None or not self._up.is_set():
            return False
        try:
            with self._send_lock:
                conn.send(frame)
            return True
        except (ConnectionClosed, OSError) as exc:
            log.debug("send failed: %s", exc)
            self._mark_down(conn, f"send failed: {exc}")
            return False

    # ------------------------------------------------------------------ thread
    def run(self) -> None:
        delay = self.backoff_min_s
        while not self._stop.is_set():
            self.attempts += 1
            try:
                conn = self._connect()
            except Exception as exc:  # DNS, refused, timeout, server not started
                self._note_failure(f"connect failed: {exc}")
            else:
                try:
                    self._handshake(conn)
                except HandshakeError as exc:
                    conn.close()
                    self._note_failure(f"brain refused us: {exc}")
                    if exc.code in FATAL_CODES:
                        log.error("Brain link refused (%s). Check ~/.roco/link_token and "
                                  "that both sides run the same ROCO version.", exc)
                        delay = self.backoff_max_s
                except (ConnectionClosed, OSError, ProtocolError, TimeoutError) as exc:
                    conn.close()
                    self._note_failure(f"handshake failed: {exc}")
                else:
                    delay = self.backoff_min_s
                    self._serve(conn)
            if self._stop.is_set():
                break
            wait = delay * (0.8 + 0.4 * self._rng.random())
            log.debug("reconnecting in %.1f s", wait)
            self._stop.wait(wait)
            delay = min(self.backoff_max_s, delay * 2)

    def _note_failure(self, reason: str) -> None:
        if reason != self.last_error:
            log.warning("Brain link: %s", reason)
        self.last_error = reason

    def _handshake(self, conn: Connection) -> None:
        conn.send(self._hello())
        deadline = self.clock() + self.handshake_timeout_s
        while True:
            remaining = deadline - self.clock()
            if remaining <= 0 or self._stop.is_set():
                raise TimeoutError("no welcome from the brain")
            frame = conn.recv(timeout=min(remaining, self.recv_timeout_s))
            if frame is None:
                continue
            if isinstance(frame, Welcome):
                if frame.protocol != PROTOCOL_VERSION:
                    raise HandshakeError("version", f"brain speaks v{frame.protocol}")
                self._conn = conn
                self.welcome = frame
                self.connects += 1
                self.last_error = ""
                now = self.clock()
                self._last_ping = self._last_pong = now
                self._up.set()
                log.info("Brain link up: %s (session %s)", frame.brain, frame.session or "-")
                if self.on_up:
                    self.on_up(frame)
                return
            if isinstance(frame, Error):
                raise HandshakeError(frame.code, frame.message)
            if isinstance(frame, Ping):
                conn.send(Pong(id=frame.id, ts=frame.ts))
                continue
            log.debug("ignoring %s before welcome", type(frame).__name__)

    def _serve(self, conn: Connection) -> None:
        reason = "closed"
        try:
            while not self._stop.is_set():
                now = self.clock()
                if now - self._last_pong > self.pong_timeout_s:
                    reason = "no pong from the brain"
                    break
                if now - self._last_ping >= self.ping_interval_s:
                    self._ping_id += 1
                    self._last_ping = now
                    with self._send_lock:
                        conn.send(Ping(id=self._ping_id, ts=now))
                try:
                    frame = conn.recv(timeout=self.recv_timeout_s)
                except ProtocolError as exc:
                    log.warning("bad frame from the brain: %s", exc)
                    continue
                if frame is None:
                    continue
                self._last_pong = self.clock()  # anything from the brain proves it is alive
                if isinstance(frame, Pong):
                    self.rtt_ms = (self._last_pong - float(frame.ts)) * 1000.0
                    continue
                if isinstance(frame, Ping):
                    with self._send_lock:
                        conn.send(Pong(id=frame.id, ts=frame.ts))
                    continue
                if isinstance(frame, Error) and frame.fatal:
                    reason = f"brain error {frame.code}: {frame.message}"
                    self._dispatch(frame)
                    break
                self._dispatch(frame)
        except (ConnectionClosed, OSError) as exc:
            reason = str(exc) or "closed"
        self._mark_down(conn, reason)

    def _dispatch(self, frame: Frame) -> None:
        try:
            self.on_frame(frame)
        except Exception:
            log.exception("handling %s failed", type(frame).__name__)

    def _mark_down(self, conn: Connection, reason: str) -> None:
        with self._send_lock:
            if self._conn is not conn:
                return
            self._conn = None
            was_up = self._up.is_set()
            self._up.clear()
        conn.close()
        if was_up:
            if not self._stop.is_set():
                log.warning("Brain link down: %s", reason)
            if self.on_down:
                try:
                    self.on_down(reason)
                except Exception:
                    log.exception("on_down handler failed")


def websocket_connector(url: str, open_timeout_s: float = 5.0) -> Connect:
    """``connect`` for ``roco body``: a sync websockets client, no compression (raw
    PCM barely compresses and deflate costs Pi CPU)."""

    def connect() -> Connection:
        from websockets.sync.client import connect as ws_connect

        from roco.protocol import WebSocketConnection

        ws: Any = ws_connect(url, open_timeout=open_timeout_s, compression=None,
                             max_size=None, close_timeout=1)
        return WebSocketConnection(ws)

    return connect
