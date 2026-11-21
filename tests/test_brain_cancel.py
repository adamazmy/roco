"""OllamaBrain.respond_turn: cancel (barge-in), hooks, amend_last_reply; Brain defaults."""

from __future__ import annotations

import threading
import time

import pytest

from mock_ollama import MockOllama, content, tokens, tool_calls
from roco.brain.base import Brain, TurnHooks
from roco.brain.history import History
from roco.brain.ollama_brain import OllamaBrain
from roco.config import load_config
from roco.tools.registry import ToolRegistry


@pytest.fixture
def mock():
    server = MockOllama()
    yield server
    server.close()


@pytest.fixture
def registry():
    reg = ToolRegistry()
    calls: list[str] = []

    @reg.tool(description="Read the calendar.")
    def get_calendar_events() -> str:
        calls.append("get_calendar_events")
        return "Dentist at 10."

    @reg.tool(description="Read email.")
    def list_unread_emails() -> str:
        calls.append("list_unread_emails")
        return "No unread email."

    reg.calls = calls  # type: ignore[attr-defined]
    return reg


def make_brain(url, bus, registry=None):
    cfg = load_config(overrides={"llm": {"host": url, "timeout_s": 5},
                                 "history": {"persist": False}})
    return OllamaBrain(cfg, bus, registry or ToolRegistry())


def slow(text: str, delay: float = 0.05):
    out = []
    for chunk in tokens(text, 4):
        out += [chunk, delay]
    return out


def test_hooks_report_tokens_and_tools(mock, bus, registry):
    mock.script = [
        [tool_calls(("get_calendar_events", {}))],
        tokens("[happy] You have the dentist at ten."),
    ]
    brain = make_brain(mock.url, bus, registry)
    seen: list[object] = []
    hooks = TurnHooks(on_token=lambda: seen.append("token"),
                      on_tools=lambda names: seen.append(("tools", names)))
    reply = brain.respond_turn("What's on today?", hooks=hooks)
    assert reply == "You have the dentist at ten."
    assert seen[0] == ("tools", ["get_calendar_events"])
    assert "token" in seen[1:]


def test_cancel_mid_stream_stops_quickly_and_keeps_what_was_said(mock, bus):
    mock.script = [
        [content("[happy] First sentence. "), 0.05, *slow("Second sentence is long. " * 20)]
    ]
    brain = make_brain(mock.url, bus)
    cancel = threading.Event()
    sentences: list[str] = []
    cancelled_at: list[float] = []

    def on_sentence(s):
        sentences.append(s)
        if len(sentences) == 1:
            threading.Timer(0.1, lambda: (cancelled_at.append(time.monotonic()),
                                          cancel.set())).start()

    reply = brain.respond_turn("Talk to me.", on_sentence, cancel=cancel)
    returned = time.monotonic()
    assert sentences[0] == "First sentence."
    assert returned - cancelled_at[0] < 0.3  # stops within a chunk or two
    assert len(sentences) < 5
    assert reply == " ".join(sentences)
    assert brain.history.messages[-1] == {"role": "assistant", "content": reply}
    brain.amend_last_reply("First sentence.")
    assert brain.history.messages[-1]["content"] == "First sentence."
    assert brain.history.messages[-2] == {"role": "user", "content": "Talk to me."}


def test_cancel_during_tools_skips_the_rest_but_answers_every_call(mock, bus, registry):
    mock.script = [
        [tool_calls(("get_calendar_events", {}), ("list_unread_emails", {}))],
        tokens("[happy] Never said."),
    ]
    brain = make_brain(mock.url, bus, registry)
    cancel = threading.Event()
    hooks = TurnHooks(on_tools=lambda names: cancel.set())
    reply = brain.respond_turn("Calendar and email?", cancel=cancel, hooks=hooks)
    assert reply == ""
    assert registry.calls == []
    assert len(mock.requests) == 1  # no answer round after cancelling
    tools = [m for m in brain.history.for_model(now=None) if m["role"] == "tool"]
    assert [t["tool_name"] for t in tools] == ["get_calendar_events", "list_unread_emails"]
    assert all("Cancelled" in t["content"] for t in tools)


def test_cancel_before_start_returns_empty_without_calling_ollama(mock, bus):
    brain = make_brain(mock.url, bus)
    cancel = threading.Event()
    cancel.set()
    assert brain.respond_turn("Hello?", cancel=cancel) == ""
    assert mock.requests == []


def test_respond_still_works_as_before(mock, bus):
    mock.script = [tokens("[happy] Hello Adam. Nice day.")]
    brain = make_brain(mock.url, bus)
    out: list[str] = []
    assert brain.respond("Hi", out.append) == "Hello Adam. Nice day."
    assert out == ["Hello Adam.", "Nice day."]


def test_history_amend_last_reply_keeps_emotion_and_tools():
    history = History(4)
    assert history.amend_last_reply("x") is False
    history.add_turn("Hi", "Hello there, how are you?", "happy",
                     [{"role": "tool", "content": "r", "tool_name": "t"}])
    assert history.amend_last_reply("Hello there,") is True
    replay = history.for_model(now=None)
    assert replay[-1] == {"role": "assistant", "content": "[happy] Hello there,"}
    assert replay[-2]["role"] == "tool"


class _PlainBrain(Brain):
    def __init__(self) -> None:
        self.cancel: threading.Event | None = None

    def respond(self, text, on_sentence=None):
        for s in ("One.", "Two.", "Three."):
            if s == "Two." and self.cancel is not None:
                self.cancel.set()
            if on_sentence:
                on_sentence(s)
        return "One. Two. Three."

    def reset(self) -> None:
        pass


def test_default_respond_turn_drops_sentences_after_cancel():
    brain = _PlainBrain()
    brain.cancel = threading.Event()
    got: list[str] = []
    brain.respond_turn("x", got.append, cancel=brain.cancel, hooks=TurnHooks())
    assert got == ["One."]
    brain.amend_last_reply("One.")  # default: no-op, no error
