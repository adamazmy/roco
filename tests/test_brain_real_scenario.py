"""Long conversation regression test against the real model.

Guards the behaviour a prompt, history or model change can quietly break:
- every action is a real tool call (no "I've cancelled it" without cancel_reminder),
- calendar data is fetched again instead of answered from an old result,
- replies keep their emotion tag and never leak it into the text,
- the reminder file ends in the state the conversation asked for.

Skipped unless Ollama runs with the configured model (``pytest -m ollama -s``).
The calendar tool is a stand-in shaped like workstream D's.
"""

from __future__ import annotations

import httpx
import pytest

from roco.brain.ollama_brain import OllamaBrain
from roco.bus import EventBus
from roco.config import DEFAULTS, load_config
from roco.paths import resolve
from roco.tools import ToolContext, build_registry
from roco.tools import reminders as rem

pytestmark = [pytest.mark.ollama, pytest.mark.timeout(900)]

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
    ("And what time is it now?", "get_current_time", None),
    ("Cancel the reminder about Mum.", "cancel_reminder", None),
    ("Which reminders are left?", "list_reminders", None),
    ("Lovely, thanks. That's all for now.", None, None),
]


@pytest.fixture(scope="module", autouse=True)
def require_model():
    host, model = DEFAULTS["llm"]["host"], DEFAULTS["llm"]["model"]
    try:
        names = {m["name"] for m in httpx.get(f"{host}/api/tags", timeout=3).json()["models"]}
    except (httpx.HTTPError, ValueError, KeyError):
        pytest.skip(f"Ollama is not reachable at {host}")
    if model not in names:
        pytest.skip(f"model {model} is not pulled")


def test_long_conversation_uses_tools_for_every_action():
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
    brain = OllamaBrain(cfg, bus, registry)
    first_chunks: list[str] = []
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
            if required and required not in calls:
                failures.append(f"{said!r}: expected a {required} call, got {calls}")
            if said.startswith("Is there anything on tomorrow") and "dentist" not in reply.lower():
                failures.append(f"stale calendar answer: {reply!r}")

        service = rem.get_service(resolve(cfg.get_path("reminders.file")))
        left = [r.text.lower() for r in service.pending()]
        if not (len(left) == 1 and "bin" in left[0]):
            failures.append(f"reminders left should be just the bins one, got {left}")
        tag_rate = tagged / len(SCENARIO)
        if tag_rate < 0.7:
            failures.append(f"only {tag_rate:.0%} of replies started with an emotion tag")
    finally:
        rem.stop_all()
        brain.close()

    report = "\n".join(transcript)
    print("\n" + report + f"\n  tag rate: {tagged}/{len(SCENARIO)}")
    assert not failures, "\n".join(failures) + "\n\n" + report
