"""`roco run` / `roco body` wiring with fakes (no audio, no real server)."""

from __future__ import annotations

import threading
import types

from roco import cli
from roco.client import app
from roco.protocol import ConnectionClosed, Hello, Welcome


class FakeServer:
    """Stands in for workstream A's roco.server: create_brain_context + serve_connection."""

    def __init__(self):
        self.contexts = []
        self.sessions = 0
        self.hellos = []

    def create_brain_context(self, cfg):
        ctx = types.SimpleNamespace(closed=False)
        ctx.close = lambda: setattr(ctx, "closed", True)
        self.contexts.append(ctx)
        return ctx

    def serve_connection(self, conn, ctx, stop: threading.Event):
        self.sessions += 1
        try:
            while not stop.is_set():
                frame = conn.recv(timeout=0.05)
                if isinstance(frame, Hello):
                    self.hellos.append(frame)
                    conn.send(Welcome(protocol=1, brain="fake", tts_rate=22050))
        except ConnectionClosed:
            pass


def test_local_brain_serves_each_connection_over_a_memory_pipe():
    server = FakeServer()
    ctx = server.create_brain_context({})
    brain = app.LocalBrain(server, ctx)
    for _ in range(2):  # a reconnect gets a fresh pipe and session
        conn = brain.connect()
        conn.send(Hello(protocol=1, body_id="b", token="t"))
        assert isinstance(conn.recv(timeout=2), Welcome)
        conn.close()
    brain.close()
    assert server.sessions == 2
    assert ctx.closed


def test_run_local_wires_the_server_into_the_body(monkeypatch, config):
    server = FakeServer()
    seen = {}

    def fake_start(cfg, connect, face, banner):
        conn = connect()
        conn.send(Hello(protocol=1, body_id="b", token="t"))
        seen["welcome"] = conn.recv(timeout=2)
        seen["face"] = face
        return 0

    monkeypatch.setattr(app, "_start", fake_start)
    monkeypatch.setattr(app, "preflight", lambda cfg: None)
    assert app.run_local(config, face=False, server=server) == 0
    assert isinstance(seen["welcome"], Welcome)
    assert seen["face"] is False
    assert server.contexts[0].closed  # brain context closed on exit


def test_run_local_without_the_server_package_fails_cleanly(monkeypatch, config, capsys):
    import importlib

    real = importlib.import_module

    def fake_import(name, *a, **kw):
        if name == "roco.server":
            raise ModuleNotFoundError("No module named 'roco.server'", name="roco.server")
        return real(name, *a, **kw)

    monkeypatch.setattr(app.importlib, "import_module", fake_import)
    monkeypatch.setattr(app, "preflight", lambda cfg: None)
    assert app.run_local(config) == 1
    assert "brain server" in capsys.readouterr().err


def test_cli_run_and_body_dispatch(monkeypatch):
    calls = []
    monkeypatch.setattr(app, "run_local", lambda cfg, face=True: calls.append(("run", face)) or 0)
    monkeypatch.setattr(app, "run_body",
                        lambda cfg, url=None, face=True: calls.append(("body", url, face)) or 0)
    assert cli.main(["run", "--no-face"]) == 0
    assert cli.main(["body", "--url", "ws://pc:8765"]) == 0
    assert calls == [("run", False), ("body", "ws://pc:8765", True)]
