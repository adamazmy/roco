"""OllamaBrain against a real local HTTP server that imitates Ollama."""

from __future__ import annotations

import json

import pytest

from mock_ollama import MockOllama, Status, closed_port_url, content, tokens, tool_calls
from roco.brain import ollama_brain
from roco.brain.ollama_brain import OllamaBrain
from roco.bus import Emotion, EventBus, State
from roco.config import load_config
from roco.tools.registry import ToolRegistry


@pytest.fixture
def mock():
    server = MockOllama()
    yield server
    server.close()


@pytest.fixture
def events(bus):
    """Everything published on the bus, plus sentences, in one ordered list."""
    log: list[tuple[str, object]] = []
    bus.subscribe("*", lambda item: log.append(item))
    return log


@pytest.fixture
def registry():
    reg = ToolRegistry()

    @reg.tool(description="Get the time.")
    def get_current_time() -> str:
        return "Friday 20 November 2026, 14:05"

    @reg.tool(
        description="Add two numbers.",
        parameters={
            "type": "object",
            "properties": {"a": {"type": "number"}, "b": {"type": "number"}},
            "required": ["a", "b"],
        },
    )
    def add(a: float, b: float) -> str:
        return str(a + b)

    return reg


def make_brain(url, bus, registry=None, **llm):
    cfg = load_config(
        overrides={
            "llm": {"host": url, "timeout_s": 5, **llm},
            "history": {"persist": False},
        }
    )
    return OllamaBrain(cfg, bus, registry)


def run(brain, text, events=None):
    sentences: list[str] = []

    def on_sentence(s):
        sentences.append(s)
        if events is not None:
            events.append(("sentence", s))

    return brain.respond(text, on_sentence), sentences


# ---------------------------------------------------------------------- basic replies


def test_plain_reply_publishes_thinking_then_emotion_then_sentence(mock, bus, events):
    mock.script = [[content("[happy] Hello, Adam!")]]
    brain = make_brain(mock.url, bus)
    reply, sentences = run(brain, "hi", events)

    assert reply == "Hello, Adam!"
    assert sentences == ["Hello, Adam!"]
    assert events == [
        ("state", State.THINKING),
        ("emotion", Emotion.HAPPY),
        ("sentence", "Hello, Adam!"),
    ]


def test_request_payload(mock, bus, registry):
    mock.script = [[content("[neutral] Fine.")]]
    brain = make_brain(mock.url, bus, registry, temperature=0.2, num_ctx=2048, keep_alive="5m")
    run(brain, "how are you")

    req = mock.requests[0]
    assert req["model"] == "qwen2.5:7b" and req["stream"] is True
    assert req["options"] == {"temperature": 0.2, "num_ctx": 2048}
    assert req["keep_alive"] == "5m"
    assert {t["function"]["name"] for t in req["tools"]} == {"get_current_time", "add"}
    assert req["messages"][0]["role"] == "system" and "[happy]" in req["messages"][0]["content"]
    user = req["messages"][-1]
    assert user == {"role": "user", "content": "how are you"}


def test_extra_ollama_options_are_sent_and_cannot_override_named_ones():
    from roco.brain.ollama import OllamaClient

    client = OllamaClient(
        "http://localhost:11434", "m", temperature=0.3, num_ctx=1024,
        extra_options={"num_gpu": 99, "temperature": 1.5},
    )
    try:
        options = client.build_payload([{"role": "user", "content": "hi"}])["options"]
    finally:
        client.close()
    assert options == {"num_gpu": 99, "temperature": 0.3, "num_ctx": 1024}


def test_no_tools_key_when_registry_empty(mock, bus):
    mock.script = [[content("[neutral] Fine.")]]
    run(make_brain(mock.url, bus), "hi")
    assert "tools" not in mock.requests[0]


def test_streamed_reply_is_split_into_sentences_as_they_complete(mock, bus):
    text = (
        "[curious] It's 3.5 degrees outside. Mr. Smith rang at 10.30 this morning! "
        "Well... maybe not, e.g. it was Mrs. Jones. Shall I check?"
    )
    # Pause before the tail so the first sentences must arrive before the stream ends.
    chunks = tokens(text[:60]) + [0.3] + tokens(text[60:])
    mock.script = [chunks]
    brain = make_brain(mock.url, bus)
    seen_while_streaming: list[bool] = []
    sentences: list[str] = []

    def on_sentence(s):
        sentences.append(s)
        seen_while_streaming.append(not mock.finished.is_set())

    reply = brain.respond("weather?", on_sentence)
    assert sentences == [
        "It's 3.5 degrees outside.",
        "Mr. Smith rang at 10.30 this morning!",
        "Well... maybe not, e.g. it was Mrs. Jones.",
        "Shall I check?",
    ]
    assert seen_while_streaming[0] is True  # delivered before generation finished
    assert reply == " ".join(sentences)
    assert "[" not in reply


@pytest.mark.parametrize(
    "chunks, emotion",
    [
        (["[", "hap", "py", "]", " Hi there."], Emotion.HAPPY),
        (["  [Exci", "TED]Hi there."], Emotion.EXCITED),
        (["[sad] ", "Hi there."], Emotion.SAD),
        (["Hi there."], Emotion.NEUTRAL),  # missing tag
        (["[grumpy] Hi there."], Emotion.NEUTRAL),  # unknown tag word: stripped anyway
    ],
)
def test_emotion_tag_split_across_chunks(mock, bus, events, chunks, emotion):
    mock.script = [[content(c) for c in chunks]]
    reply, sentences = run(make_brain(mock.url, bus), "hi", events)
    assert reply == "Hi there." and sentences == ["Hi there."]
    emotions = [p for t, p in events if t == "emotion"]
    assert emotions == [emotion]
    assert events.index(("emotion", emotion)) < events.index(("sentence", "Hi there."))


def test_inline_tags_never_reach_sentences(mock, bus):
    mock.script = [[content("[happy] Hello! [excited] Big news today.")]]
    reply, sentences = run(make_brain(mock.url, bus), "hi")
    assert sentences == ["Hello!", "Big news today."]


# ---------------------------------------------------------------------- tool calls


def test_one_tool_call(mock, bus, registry):
    mock.script = [
        [tool_calls(("get_current_time", {}))],
        [content("[neutral] It's five past two.")],
    ]
    # Worded so the "what time is it" prefetch does not apply: the model calls the tool.
    reply, sentences = run(make_brain(mock.url, bus, registry), "how late is it getting?")

    assert reply == "It's five past two."
    second = mock.requests[1]["messages"]
    assert second[-2]["role"] == "assistant"
    assert second[-2]["tool_calls"][0]["function"]["name"] == "get_current_time"
    assert second[-1] == {
        "role": "tool",
        "content": "Friday 20 November 2026, 14:05",
        "tool_name": "get_current_time",
    }


def test_multiple_tool_calls_and_rounds(mock, bus, registry):
    mock.script = [
        [tool_calls(("add", {"a": 1, "b": 2}), ("get_current_time", {}))],
        [tool_calls(("add", {"a": 3, "b": 4}))],
        [content("[happy] Three, then seven.")],
    ]
    brain = make_brain(mock.url, bus, registry)
    reply, _ = run(brain, "sums")

    assert reply == "Three, then seven."
    assert len(mock.requests) == 3
    tool_msgs = [m for m in mock.requests[2]["messages"] if m["role"] == "tool"]
    assert [m["content"] for m in tool_msgs] == ["3", "Friday 20 November 2026, 14:05", "7"]
    # Tool-call messages are not kept in the history.
    assert [m["role"] for m in brain.history.messages] == ["user", "assistant"]


def test_history_text_is_user_and_tag_free_final_text(mock, bus, registry):
    mock.script = [
        [tool_calls(("get_current_time", {}))],
        [content("[neutral] Two o'clock.")],
        [content("[neutral] You're welcome.")],
    ]
    brain = make_brain(mock.url, bus, registry)
    run(brain, "time?")
    run(brain, "thanks")
    assert brain.history.messages == [
        {"role": "user", "content": "time?"},
        {"role": "assistant", "content": "Two o'clock."},
        {"role": "user", "content": "thanks"},
        {"role": "assistant", "content": "You're welcome."},
    ]


def test_malformed_tool_calls_return_errors_and_conversation_continues(mock, bus, registry):
    mock.script = [
        [
            tool_calls(
                ("does_not_exist", {}),
                ("add", "{not json"),
                ("add", {"a": 1}),
            ),
            {"message": {"role": "assistant", "content": "", "tool_calls": [{"bogus": 1}]}},
        ],
        [content("[confused] Sorry, I fumbled that.")],
    ]
    reply, _ = run(make_brain(mock.url, bus, registry), "do things")

    assert reply == "Sorry, I fumbled that."
    messages = mock.requests[1]["messages"]
    results = [m["content"] for m in messages if m["role"] == "tool"]
    assert results[0].startswith("Error: unknown tool 'does_not_exist'")
    assert results[1].startswith("Error: invalid arguments for add")
    assert results[2].startswith("Error: add is missing required argument(s): b")
    assert results[3].startswith("Error: malformed tool call")
    # Arguments are always echoed back as objects so Ollama accepts the next request.
    echoed = next(m for m in messages if m.get("tool_calls"))["tool_calls"]
    assert all(isinstance(c["function"]["arguments"], dict) for c in echoed)


def test_max_tool_rounds_then_forced_answer_without_tools(mock, bus, registry):
    mock.script = [
        [tool_calls(("get_current_time", {}))],
        [tool_calls(("get_current_time", {}))],
        [content("[neutral] It's two.")],
    ]
    reply, _ = run(make_brain(mock.url, bus, registry, max_tool_rounds=2), "time")
    assert reply == "It's two."
    assert len(mock.requests) == 3
    assert "tools" in mock.requests[1] and "tools" not in mock.requests[2]


def test_max_tool_rounds_and_still_no_answer_apologises(mock, bus, registry):
    mock.script = [[tool_calls(("get_current_time", {}))] for _ in range(3)]
    reply, sentences = run(make_brain(mock.url, bus, registry, max_tool_rounds=2), "time")
    assert reply == ollama_brain.APOLOGY_TANGLED == sentences[0]
    assert len(mock.requests) == 3


def test_preamble_before_tool_call_is_spoken_in_order(mock, bus, registry):
    mock.script = [
        [content("[curious] Let me check."), tool_calls(("get_current_time", {}))],
        [content("[happy] It's five past two.")],
    ]
    reply, sentences = run(make_brain(mock.url, bus, registry), "time")
    assert sentences == ["Let me check.", "It's five past two."]
    assert reply == "Let me check. It's five past two."


# ---------------------------------------------------------------------- failures


def test_server_down_apologises_without_raising(bus, events):
    brain = make_brain(closed_port_url(), bus)
    reply, sentences = run(brain, "hello", events)
    assert reply == ollama_brain.APOLOGY_UNREACHABLE
    assert sentences == [reply]
    assert "Ollama" in reply
    assert ("emotion", Emotion.SAD) in events
    assert len(brain.history) == 0  # failed turns are not remembered


def test_model_missing(mock, bus):
    mock.script = [Status(404, {"error": 'model "qwen2.5:7b" not found, try pulling it first'})]
    reply, _ = run(make_brain(mock.url, bus), "hi")
    assert "qwen2.5 7b" in reply and "pulling" in reply


@pytest.mark.parametrize(
    "script",
    [
        [Status(500, {"error": "out of memory"})],
        [[content("[happy] Hel"), {"error": "model crashed"}]],
        [[b"this is not json\n"]],
    ],
)
def test_server_errors_apologise(mock, bus, script):
    mock.script = script
    reply, _ = run(make_brain(mock.url, bus), "hi")
    assert reply == ollama_brain.APOLOGY_GENERIC


def test_stream_breaking_mid_sentence_drops_the_partial_sentence(mock, bus):
    mock.script = [[content("[happy] First one. Second is cut"), {"error": "boom"}]]
    reply, sentences = run(make_brain(mock.url, bus), "hi")
    assert sentences == ["First one.", ollama_brain.APOLOGY_GENERIC]
    assert reply == "First one. " + ollama_brain.APOLOGY_GENERIC


def test_timeout_apologises(mock, bus):
    mock.script = [[0.8, content("[happy] Too late.")]]
    reply, _ = run(make_brain(mock.url, bus, timeout_s=0.3), "hi")
    assert reply == ollama_brain.APOLOGY_TIMEOUT


def test_empty_input_does_not_call_the_model(mock, bus):
    assert make_brain(mock.url, bus).respond("   ") == ""
    assert mock.requests == []


def test_failing_on_sentence_callback_does_not_break_the_reply(mock, bus):
    mock.script = [[content("[happy] One. Two.")]]

    def explode(_s):
        raise RuntimeError("speaker died")

    assert make_brain(mock.url, bus).respond("hi", explode) == "One. Two."


# ---------------------------------------------------------------------- history


def test_history_is_persisted_and_reset(mock, isolated_home):
    mock.script = [[content("[happy] Hello!")]]
    cfg = load_config(overrides={"llm": {"host": mock.url}, "history": {"max_turns": 3}})
    bus = EventBus()
    brain = OllamaBrain(cfg, bus)
    brain.respond("hi")
    path = isolated_home / "history.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    user, reply = data["messages"]
    assert user == {"role": "user", "content": "hi"}
    assert reply["content"] == "Hello!" and reply["emotion"] == "happy"
    assert reply["at"]  # when the turn happened, for stamping replayed tool results

    again = OllamaBrain(cfg, bus)
    assert len(again.history) == 2
    again.reset()
    assert len(again.history) == 0 and not path.exists()


def test_next_turn_replays_tag_and_tool_trace(mock, bus, registry):
    # Small models copy their past turns: without these they stop tagging and bluff tool use.
    mock.script = [
        [tool_calls(("add", {"a": 1, "b": 2}))],
        [content("[happy] Three.")],
        [content("[neutral] Sure.")],
    ]
    brain = make_brain(mock.url, bus, registry)
    run(brain, "1+2?")
    run(brain, "thanks")
    sent = mock.requests[2]["messages"][1:]
    assert [m["role"] for m in sent] == ["user", "assistant", "tool", "assistant", "user"]
    assert sent[1]["tool_calls"][0]["function"] == {"name": "add", "arguments": {"a": 1, "b": 2}}
    assert sent[2]["content"].startswith("(Old result, fetched a few minutes ago.")
    assert sent[2]["content"].endswith(") 3")
    assert sent[3]["content"] == "[happy] Three."
    assert brain.history.messages[1] == {"role": "assistant", "content": "Three."}


def test_empty_answer_is_retried_once(mock, bus):
    mock.script = [[content("[happy]")], [content("[happy] Here you go.")]]
    reply, _ = run(make_brain(mock.url, bus), "hi")
    assert reply == "Here you go."
    assert len(mock.requests) == 2


@pytest.mark.parametrize(
    "written",
    [
        'Let me check. {"name": "add", "arguments": {"a": 1, "b": 2}}',
        '<tool_call>\n{"name": "add", "arguments": {"a": 1, "b": 2}}\n</tool_call>',
    ],
)
def test_tool_call_written_as_text_is_run_not_spoken(mock, bus, registry, written):
    mock.script = [tokens("[curious] " + written, 5), [content("[happy] It's three.")]]
    reply, sentences = run(make_brain(mock.url, bus, registry), "1+2?")
    assert all("{" not in s and "tool_call" not in s for s in sentences)
    assert sentences[-1] == "It's three."
    tool_msgs = [m for m in mock.requests[1]["messages"] if m["role"] == "tool"]
    assert tool_msgs[0]["content"] == "3" and tool_msgs[0]["tool_name"] == "add"


def test_braces_that_are_not_a_tool_call_are_still_spoken(mock, bus, registry):
    mock.script = [[content("[happy] Sets use {curly} braces.")]]
    reply, _ = run(make_brain(mock.url, bus, registry), "braces?")
    assert reply == "Sets use {curly} braces."
    assert len(mock.requests) == 1


def test_localhost_is_rewritten_to_ipv4_loopback():
    from roco.brain.ollama import OllamaClient

    assert OllamaClient("http://localhost:11434/").host == "http://127.0.0.1:11434"
    assert OllamaClient("http://192.168.1.20:11434").host == "http://192.168.1.20:11434"


def test_connection_is_reused_between_turns(mock, bus):
    mock.script = [[content("[happy] One.")], [content("[happy] Two.")]]
    brain = make_brain(mock.url, bus)
    ports = []
    original = mock.server.RequestHandlerClass.do_POST

    def recording(handler):
        ports.append(handler.client_address[1])
        original(handler)

    mock.server.RequestHandlerClass.do_POST = recording
    run(brain, "a")
    run(brain, "b")
    assert len(ports) == 2 and ports[0] == ports[1]  # same TCP connection


@pytest.mark.parametrize(
    "said", ["What time is it?", "and what time is it now", "What's the time?", "tell me the time"]
)
def test_time_question_is_answered_from_a_prefetched_clock(mock, bus, registry, said):
    mock.script = [[content("[neutral] It's five past two.")]]
    brain = make_brain(mock.url, bus, registry)
    reply, _ = run(brain, said)

    assert reply == "It's five past two."
    assert len(mock.requests) == 1  # no extra round for the tool call
    sent = mock.requests[0]["messages"]
    assert sent[-3] == {"role": "user", "content": said}
    assert sent[-2]["tool_calls"][0]["function"]["name"] == "get_current_time"
    assert sent[-1] == {
        "role": "tool",
        "content": "Friday 20 November 2026, 14:05",
        "tool_name": "get_current_time",
    }
    # Kept in history like a real call, so later turns still see the tool being used.
    run(brain, "thanks")
    later = mock.requests[1]["messages"]
    assert any(m.get("tool_name") == "get_current_time" for m in later)


@pytest.mark.parametrize("said", ["Remind me at what time the bins go out", "Time for tea?"])
def test_other_questions_are_not_prefetched(mock, bus, registry, said):
    mock.script = [[content("[neutral] Righto.")]]
    run(make_brain(mock.url, bus, registry), said)
    assert not any(m["role"] == "tool" for m in mock.requests[0]["messages"])


# ---------------------------------------------------------------------- reminder repair
@pytest.fixture
def reminder_registry():
    reg = ToolRegistry()
    calls: list[dict] = []

    @reg.tool(
        description="Set a reminder.",
        parameters={
            "type": "object",
            "properties": {"text": {"type": "string"}, "minutes_from_now": {"type": "number"}},
            "required": ["text"],
        },
    )
    def set_reminder(text: str, minutes_from_now: float | None = None) -> str:
        calls.append({"text": text, "minutes_from_now": minutes_from_now})
        return f"Reminder 1 set: {text}."

    reg.calls = calls
    return reg


def test_claimed_but_unset_reminder_is_set_silently(mock, bus, reminder_registry):
    mock.script = [
        [content("[smug] Water? Groundbreaking. I'll set it.")],  # bluff: no tool call
        [tool_calls(("set_reminder", {"text": "drink water", "minutes_from_now": 10}))],
        [content("[happy] Done.")],  # never spoken
    ]
    brain = make_brain(mock.url, bus, reminder_registry)
    reply, sentences = run(brain, "Remind me to drink water in ten minutes.")
    assert reminder_registry.calls == [{"text": "drink water", "minutes_from_now": 10}]
    assert sentences == ["Water?", "Groundbreaking.", "I'll set it."]
    repair_note = mock.requests[1]["messages"][-1]
    assert repair_note["role"] == "user" and "didn't call set_reminder" in repair_note["content"]
    # History keeps the real tool exchange, not the system note.
    replay = json.dumps(brain.history.for_model(tools=True))
    assert "set_reminder" in replay and "didn't call set_reminder" not in replay


def test_reminder_repair_owns_up_when_the_model_still_does_not_set_it(
    mock, bus, reminder_registry
):
    mock.script = [
        [content("[smug] Oven, five minutes. Consider it done.")],
        [content("[neutral] Sure.")],  # repair round, still no tool call
    ]
    brain = make_brain(mock.url, bus, reminder_registry)
    reply, sentences = run(brain, "Set a reminder to check the oven in five minutes.")
    assert reminder_registry.calls == []
    assert sentences[-1] == ollama_brain.REMINDER_REPAIR_FAILED


def test_no_repair_when_reminder_was_set_or_not_claimed(mock, bus, reminder_registry):
    mock.script = [
        [tool_calls(("set_reminder", {"text": "call mum"}))],
        [content("[happy] Done, mum at six.")],
        [content("[curious] What time should I remind you?")],
    ]
    brain = make_brain(mock.url, bus, reminder_registry)
    run(brain, "Remind me to call mum at six.")
    run(brain, "Remind me to stretch.")
    assert len(mock.requests) == 3
    assert len(reminder_registry.calls) == 1


def test_warm_up_sends_the_real_prompt_once_without_streaming(mock, bus, registry):
    mock.script = [content("[neutral] H")]  # a single chunk; Ollama answers non-streamed
    brain = make_brain(mock.url, bus, registry, options={"num_gpu": 99})
    brain.warm_up()
    req = mock.requests[0]
    assert req["stream"] is False
    assert req["options"]["num_predict"] == 1 and req["options"]["num_gpu"] == 99
    assert req["messages"][0] == brain._system_message()
    assert {t["function"]["name"] for t in req["tools"]} == {"get_current_time", "add"}


def test_warm_up_failure_is_only_logged(bus):
    from mock_ollama import closed_port_url

    brain = make_brain(closed_port_url(), bus)
    assert brain.warm_up() is False


def test_server_context_warms_the_brain_after_the_model_load():
    import threading

    from roco.server.context import _then_warm_up

    order = []
    preload = threading.Thread(target=lambda: order.append("load"))

    class Brain:
        def warm_up(self):
            order.append("warm")

    preload.start()
    thread = _then_warm_up(preload, Brain())
    thread.join(2)
    assert order == ["load", "warm"]
    assert _then_warm_up(None, Brain()) is None
