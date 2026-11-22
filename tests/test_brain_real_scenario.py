"""Long conversation regression test against the real model.

Guards the behaviour a prompt, history or model change can quietly break:
- every action is a real tool call (no "I've cancelled it" without cancel_reminder),
- calendar data is fetched again instead of answered from an old result,
- replies keep their emotion tag and never leak it into the text,
- the reminder file ends in the state the conversation asked for.

Runs against both brains: local Ollama (``pytest -m ollama -s``, skipped unless Ollama runs
with the configured model) and Claude (``pytest -m cloud -s``, skipped without an
Anthropic key; costs well under a penny per run on Haiku). The calendar tool is a
stand-in shaped like workstream D's.
"""

from __future__ import annotations

import os
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import httpx
import pytest

from roco.brain.claude_brain import ClaudeBrain, create_claude_client
from roco.brain.ollama_brain import OllamaBrain
from roco.bus import EventBus
from roco.config import DEFAULTS, load_config
from roco.paths import resolve
from roco.tools import ToolContext, build_registry
from roco.tools import reminders as rem
from roco.tools.clock import spoken_time

pytestmark = [pytest.mark.timeout(900)]
BRAINS = [pytest.param("ollama", marks=pytest.mark.ollama),
          pytest.param("claude", marks=pytest.mark.cloud)]

# (what Adam says, tool that must be called or None, hook run before the turn)
SCENARIO: list[tuple[str, str | None, str | None]] = [
    ("Morning ROCO! How are you today?", None, None),
    ("What time is it?", "get_current_time", None),
    ("Remind me to call Mum at 9pm.", "set_reminder", None),
    ("What's on my calendar tomorrow?", "get_events", None),
    ("Tell me a quick joke.", None, None),
    ("Set a reminder to feed the cat in 15 minutes.", "set_reminder", None),
    ("What reminders do I have?", "list_reminders", None),
    ("Cancel the cat one.", "cancel_reminder", None),
    ("Is there anything on tomorrow now?", "get_events", "book_dentist"),
    ("Why do owls fly so quietly? One sentence, please.", None, None),
    ("Remind me to take the bins out in 30 minutes.", "set_reminder", None),
    # Phrasing from a real voice-loop transcript (speech-to-text drops the "to").
    ("can you remind me in twenty minutes stretch?", "set_reminder", None),
    ("And what time is it now?", "get_current_time", None),
    ("Cancel the reminder about Mum.", "cancel_reminder", None),
    ("Which reminders are left?", "list_reminders", None),
    ("Lovely, thanks. That's all for now.", None, None),
]


def _states_current_time(reply: str) -> bool:
    """True if the reply gives a time within a few minutes of now.

    Accepts the spoken form ROCO is given ("ten minutes to ten") or HH:MM / H:MM,
    for any minute from 4 minutes ago to 1 minute ahead. "Half past nine" at
    21:46 fails, which is the invented-time bug this guards against.
    """
    now = datetime.now(ZoneInfo(DEFAULTS["assistant"]["timezone"]))
    text = reply.lower()
    for delta in range(-4, 2):
        t = now + timedelta(minutes=delta)
        phrase = spoken_time(t)
        for suffix in (" in the morning", " in the afternoon", " in the evening", " at night"):
            phrase = phrase.removesuffix(suffix)
        h12 = t.hour % 12 or 12
        if phrase in text or f"{t:%H:%M}" in text or f"{h12}:{t:%M}" in text:
            return True
    return False


def _real_key() -> str | None:
    """ANTHROPIC_API_KEY, else the real ~/.roco/anthropic_key (tests isolate ROCO_HOME)."""
    if key := os.environ.get("ANTHROPIC_API_KEY", "").strip():
        return key
    path = Path(os.path.expanduser("~")) / ".roco" / "anthropic_key"
    try:
        return path.read_text(encoding="utf-8").split()[0]
    except (OSError, IndexError):
        return None


def _require(kind: str) -> str | None:
    if kind == "claude":
        pytest.importorskip("anthropic")
        key = _real_key()
        if not key:
            pytest.skip("no Anthropic API key")
        return key
    host, model = DEFAULTS["llm"]["host"], DEFAULTS["llm"]["model"]
    try:
        names = {m["name"] for m in httpx.get(f"{host}/api/tags", timeout=3).json()["models"]}
    except (httpx.HTTPError, ValueError, KeyError):
        pytest.skip(f"Ollama is not reachable at {host}")
    if model not in names:
        pytest.skip(f"model {model} is not pulled")
    return None


@pytest.mark.parametrize("kind", BRAINS)
def test_long_conversation_uses_tools_for_every_action(kind):
    key = _require(kind)
    cfg = load_config(overrides={"history": {"persist": False}})
    bus = EventBus()
    registry = build_registry(ToolContext(config=cfg, bus=bus))
    events = {"tomorrow": ["nothing booked"]}

    @registry.tool(
        description="List calendar events for a day: 'today', 'tomorrow' or YYYY-MM-DD.",
        parameters={
            "type": "object",
            "properties": {"day": {"type": "string", "description": "today, tomorrow or a date"}},
            "required": ["day"],
        },
    )
    def get_events(day: str) -> str:
        items = events.get(str(day).lower().strip(), ["nothing booked"])
        return f"{day}: " + "; ".join(items) + "."

    hooks = {"book_dentist": lambda: events.update(tomorrow=["dentist at 11:00"])}

    calls: list[str] = []
    original_call = registry.call

    def spy(name, arguments=None):
        calls.append(name)
        return original_call(name, arguments)

    registry.call = spy
    first_chunks: list[str] = []
    if kind == "claude":
        brain = ClaudeBrain(cfg, bus, registry, client=create_claude_client(cfg, key))
        original_message = brain.client.stream_message

        def stream_message(params, on_text, cancel=None, **kw):
            seen: list[bool] = []

            def spy_text(text: str) -> None:
                if text.strip() and not seen:
                    first_chunks.append(text.lstrip())
                    seen.append(True)
                on_text(text)

            return original_message(params, spy_text, cancel, **kw)

        brain.client.stream_message = stream_message
    else:
        brain = OllamaBrain(cfg, bus, registry)
        original_stream = brain.client.stream_chat

        def stream(messages, tools=None):
            seen = False
            for chunk in original_stream(messages, tools):
                text = (chunk.get("message") or {}).get("content") or ""
                if text.strip() and not seen:
                    first_chunks.append(text.lstrip())
                    seen = True
                yield chunk

        brain.client.stream_chat = stream

    transcript: list[str] = []
    failures: list[str] = []
    tagged = 0
    try:
        for said, required, hook in SCENARIO:
            if hook:
                hooks[hook]()
            calls.clear()
            first_chunks.clear()
            reply = brain.respond(said)
            # The tag is checked on the reply's own text: the first chunk that has text.
            has_tag = any(c.startswith("[") for c in first_chunks[-1:])
            tagged += has_tag
            line = f"you: {said}\n  tools={calls} tag={has_tag}\n  roco: {reply}"
            transcript.append(line)
            if not reply.strip():
                failures.append(f"empty reply to {said!r}")
            if "[" in reply or "]" in reply:
                failures.append(f"tag leaked into {reply!r}")
            if required == "get_current_time":
                # The brain gives the model the time with every turn, so a correct
                # answer without the tool is fine; an invented time is not.
                if required not in calls and not _states_current_time(reply):
                    failures.append(f"{said!r}: wrong or invented time: {reply!r}")
            elif required and required not in calls:
                failures.append(f"{said!r}: expected a {required} call, got {calls}")
            if said.startswith("Is there anything on tomorrow") and "dentist" not in reply.lower():
                failures.append(f"stale calendar answer: {reply!r}")

        service = rem.get_service(resolve(cfg.get_path("reminders.file")))
        left = [r.text.lower() for r in service.pending()]
        expected_left = len(left) == 2 and any("bin" in t for t in left) and any(
            "stretch" in t for t in left
        )
        if not expected_left:
            failures.append(f"reminders left should be bins + stretch, got {left}")
        tag_rate = tagged / len(SCENARIO)
        if tag_rate < 0.7:
            failures.append(f"only {tag_rate:.0%} of replies started with an emotion tag")
    finally:
        rem.stop_all()
        brain.close()

    report = "\n".join(transcript)
    print(f"\n[{kind}]\n" + report + f"\n  tag rate: {tagged}/{len(SCENARIO)}")
    assert not failures, "\n".join(failures) + "\n\n" + report


# Adam's spoken retest (2026-11-22): echo transcripts of ROCO's own voice filled the
# history with turns that used no tools, and qwen2.5 then answered "Do I have any unread
# emails?" from nothing. These turns rebuild that history before the real questions.
CHATTER: list[tuple[str, str]] = [
    ("Hello? Hello?", "Hello, Adam. I'm right here."),
    ("I'm right here.", "Yes, you are. Lovely to see you."),
    ("Lovely to see you.", "Are you repeating me now?"),
    ("Are you repeating me now?", "Apparently so. Very flattering."),
    ("Very flattering.", "Mm, flattery gets you everywhere."),
    ("Everywhere.", "Everywhere indeed."),
    ("Okay.", "Okay."),
    ("Yeah.", "Right then."),
    ("Right then.", "Anything you need?"),
    ("Anything you need?", "That's my line, Adam."),
]
AFTER_CHATTER: list[tuple[str, str]] = [
    ("Do I have any unread emails?", "get_unread_emails"),
    ("What's on my calendar tomorrow?", "get_events"),
    ("Remind me to ring the garage in ten minutes.", "set_reminder"),
    ("What reminders have I got?", "list_reminders"),
    ("Cancel the garage one.", "cancel_reminder"),
]


@pytest.mark.parametrize("kind", BRAINS)
def test_tools_are_still_called_after_a_history_of_chatter(kind):
    key = _require(kind)
    cfg = load_config(overrides={"history": {"persist": False}})
    bus = EventBus()
    registry = build_registry(ToolContext(config=cfg, bus=bus))

    @registry.tool(
        description="List calendar events for a day: 'today', 'tomorrow' or YYYY-MM-DD.",
        parameters={"type": "object", "properties": {"day": {"type": "string"}},
                    "required": ["day"]},
    )
    def get_events(day: str) -> str:
        return f"{day}: MOT at the garage at 09:30."

    if "get_unread_emails" not in registry:
        @registry.tool(
            description="Read Adam's unread emails (sender and subject), newest first.",
            parameters={"type": "object", "properties": {"max": {"type": "integer"}},
                        "required": []},
        )
        def get_unread_emails(max: int = 5) -> str:
            return "2 unread: Halfords (Your MOT is booked); Mum (Sunday lunch?)."

    calls: list[str] = []
    original_call = registry.call

    def spy(name, arguments=None):
        calls.append(name)
        return original_call(name, arguments)

    registry.call = spy
    if kind == "claude":
        brain = ClaudeBrain(cfg, bus, registry, client=create_claude_client(cfg, key))
    else:
        brain = OllamaBrain(cfg, bus, registry)
    for user, reply in CHATTER:
        brain.history.add_turn(user, reply, "neutral", [])
    failures: list[str] = []
    transcript: list[str] = []
    try:
        for said, required in AFTER_CHATTER:
            calls.clear()
            reply = brain.respond(said)
            transcript.append(f"you: {said}\n  tools={calls}\n  roco: {reply}")
            if required not in calls:
                failures.append(f"{said!r}: expected {required}, got {calls}: {reply!r}")
    finally:
        rem.stop_all()
        brain.close()
    report = "\n".join(transcript)
    print(f"\n[{kind} after chatter]\n" + report)
    assert not failures, "\n".join(failures) + "\n\n" + report
