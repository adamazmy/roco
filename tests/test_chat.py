"""``roco chat``: the session loop with a fake brain, and the CLI wiring."""

from __future__ import annotations

import io

from mock_ollama import MockOllama, content
from roco import cli
from roco.brain.chat import ChatSession
from roco.bus import State
from roco.voice.base import NullSpeaker


class FakeBrain:
    def __init__(self) -> None:
        self.resets = 0
        self.heard: list[str] = []

    def respond(self, text, on_sentence=None):
        self.heard.append(text)
        for s in ("First.", "Second."):
            on_sentence(s)
        return "First. Second."

    def reset(self):
        self.resets += 1


def lines(*items):
    it = iter(items)

    def read(_prompt):
        try:
            return next(it)
        except StopIteration:
            raise EOFError from None

    return read


def test_session_prints_sentences_and_handles_commands(bus):
    out = io.StringIO()
    brain = FakeBrain()
    states = []
    bus.subscribe("state", states.append)
    session = ChatSession(brain, bus, out=out)
    session.run(lines("hello", "  ", "/reset", "/quit", "never read"))

    text = out.getvalue()
    assert "ROCO> First. Second.\n" in text
    assert "(conversation forgotten)" in text
    assert brain.heard == ["hello"] and brain.resets == 1
    assert states == [State.IDLE]
    assert text.endswith("Bye!\n")


def test_session_speaks_and_prints_announcements(bus):
    out = io.StringIO()
    speaker = NullSpeaker()
    session = ChatSession(FakeBrain(), bus, speaker=speaker, out=out)
    bus.announce("Reminder: kettle.")
    session.handle("hi")
    assert "[ROCO] Reminder: kettle." in out.getvalue()
    assert speaker.spoken == ["Reminder: kettle.", "First.", "Second."]
    session.close()
    bus.announce("ignored after close")
    assert "ignored" not in out.getvalue()


def test_ctrl_c_ends_session(bus):
    out = io.StringIO()

    def interrupt(_prompt):
        raise KeyboardInterrupt

    ChatSession(FakeBrain(), bus, out=out).run(interrupt)
    assert out.getvalue().endswith("Bye!\n")


def test_cli_chat_end_to_end_with_mock_ollama(monkeypatch, capsys, tmp_path):
    server = MockOllama()
    try:
        server.script = [[content("[happy] Hello Adam. Nice to see you!")]]
        cfg_file = tmp_path / "config.yaml"
        cfg_file.write_text(f"llm:\n  host: {server.url}\n", encoding="utf-8")
        monkeypatch.setattr("builtins.input", lines("hi there", "/quit"))
        assert cli.main(["-c", str(cfg_file), "chat", "--reset", "--speak", "--face"]) == 0
    finally:
        server.close()
    captured = capsys.readouterr()
    assert "ROCO> Hello Adam. Nice to see you!" in captured.out
    # B and C have not shipped their factories on this branch: clear message, text continues.
    assert "Speech is not available" in captured.err
    assert "The face is not available" in captured.err
    assert server.requests[0]["messages"][-1]["content"].startswith("hi there")


def test_create_brain_builds_standard_tools(config, bus):
    from roco.brain.factory import create_brain
    from roco.tools import reminders as rem

    brain = create_brain(config, bus)
    try:
        assert {"get_current_time", "set_reminder"} <= set(brain.registry.names())
    finally:
        brain.close()
        rem.stop_all()


def test_start_face_prefers_create_body(monkeypatch, config, bus):
    import types

    from roco.body.base import NullFace
    from roco.brain import chat

    made = {}

    def create_body(cfg, bus_):
        made["body"] = object()
        return NullFace(), made["body"]

    fake = types.ModuleType("roco.body.factory")
    fake.create_body = create_body
    monkeypatch.setitem(__import__("sys").modules, "roco.body.factory", fake)
    face, body, why = chat._start_face(config, bus)
    assert isinstance(face, NullFace) and body is made["body"] and why == ""


def test_start_face_falls_back_to_create_face(monkeypatch, config, bus):
    import types

    from roco.body.base import NullFace
    from roco.brain import chat
    from roco.bus import Emotion

    fake = types.ModuleType("roco.body.factory")
    fake.create_face = lambda cfg: NullFace()
    monkeypatch.setitem(__import__("sys").modules, "roco.body.factory", fake)
    face, body, why = chat._start_face(config, bus)
    bus.set_emotion(Emotion.HAPPY)
    assert face.emotion == Emotion.HAPPY  # base Body attached to the bus
    body.close()
