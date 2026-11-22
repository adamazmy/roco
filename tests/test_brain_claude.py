"""Claude as the default brain: ClaudeBrain, the router, budgets and privacy mode.

Claude is a scripted fake SDK (tests/claude_fakes.py), plus one test of the real
``anthropic`` SDK over a mocked HTTP transport; local is OllamaBrain on MockOllama.
Nothing here touches the network.
"""

from __future__ import annotations

import json
import threading
from datetime import UTC, datetime, timedelta
from zoneinfo import ZoneInfo

import pytest

from claude_fakes import FakeClaude, Round, call, say, sse_body
from mock_ollama import MockOllama, content
from roco.brain import router as rt
from roco.brain.budget import CloudBudget, Spend, format_budget
from roco.brain.claude_brain import REFUSAL_LINE, ClaudeBrain, to_anthropic
from roco.brain.core import REMINDER_REPAIR_FAILED
from roco.brain.router import BrainRouter, create_brain
from roco.bus import Emotion, EventBus
from roco.config import load_config
from roco.fallback.chain import OFFLINE_LINE
from roco.fallback.claude import CloudPermanentError
from roco.fallback.usage import read_records
from roco.paths import resolve
from roco.tools.registry import ToolRegistry

LONDON = ZoneInfo("Europe/London")
DEAD_OLLAMA = "http://127.0.0.1:9"  # nothing listens: OllamaError("unreachable")


@pytest.fixture
def ollama():
    server = MockOllama()
    yield server
    server.close()


@pytest.fixture
def registry():
    reg = ToolRegistry()
    reg.calls = []  # type: ignore[attr-defined]

    @reg.tool(description="The current local time.")
    def get_current_time() -> str:
        reg.calls.append("get_current_time")
        return "It's twenty past four in the afternoon."

    @reg.tool(description="Read the calendar.",
              parameters={"type": "object", "properties": {"day": {"type": "string"}},
                          "required": ["day"]})
    def get_events(day: str) -> str:
        reg.calls.append(f"get_events:{day}")
        return f"{day}: dentist at 10."

    @reg.tool(description="Set a reminder.",
              parameters={"type": "object", "properties": {"text": {"type": "string"},
                          "at": {"type": "string"}}, "required": ["text"]})
    def set_reminder(text: str, at: str = "") -> str:
        reg.calls.append(f"set_reminder:{text}")
        return f"Reminder 1 set: {text} at {at}."

    return reg


def make(ollama_url, registry, *rounds: Round, bus=None, **cfg_over):
    over = {"llm": {"host": ollama_url}, "history": {"persist": False}}
    for key, value in cfg_over.items():
        over.setdefault(key, {}).update(value)
    cfg = load_config(overrides=over)
    fake = FakeClaude(*rounds)
    bus = bus or EventBus()
    router = create_brain(cfg, bus, registry, sdk=fake)
    return router, fake


def run(router, text, cancel=None):
    said: list[str] = []
    reply = router.respond_turn(text, said.append, cancel=cancel)
    return said, reply


# =========================================================================== ClaudeBrain


def test_claude_turn_with_a_tool_streams_sentences_and_stores_the_trace(ollama, registry):
    bus = EventBus()
    emotions: list[Emotion] = []
    bus.subscribe("emotion", emotions.append)
    router, fake = make(ollama.url, registry,
                        call(("get_events", {"day": "tomorrow"}), text="[curious] Let me look."),
                        say("[happy] You've got", " the dentist at ten. ", "Lucky you."),
                        bus=bus)
    said, reply = run(router, "What's on tomorrow?")
    assert said == ["Let me look.", "You've got the dentist at ten.", "Lucky you."]
    assert reply == " ".join(said)
    assert registry.calls == ["get_events:tomorrow"]
    assert emotions[:1] == [Emotion.CURIOUS] and Emotion.HAPPY in emotions
    assert router.active_name == "claude:claude-haiku-5-5"
    assert ollama.requests == []  # local untouched
    # Round 2 sends round 1 back unchanged, then the tool result with the same id.
    second = fake.calls[1]["messages"]
    assistant, results = second[-2], second[-1]
    tool_use = next(b for b in assistant["content"] if b["type"] == "tool_use")
    assert tool_use == {"type": "tool_use", "id": "toolu_live1_0", "name": "get_events",
                        "input": {"day": "tomorrow"}}
    assert results["role"] == "user"
    assert results["content"][0]["tool_use_id"] == "toolu_live1_0"
    assert results["content"][0]["content"] == "tomorrow: dentist at 10."
    # The history keeps the neutral trace, so the local brain can replay it too.
    stored = router.history.for_model()
    assert stored[1]["tool_calls"][0]["function"] == {"name": "get_events",
                                                      "arguments": {"day": "tomorrow"}}
    assert stored[2]["role"] == "tool" and "dentist" in stored[2]["content"]
    assert stored[-1]["content"].startswith("[curious] Let me look.")


def test_requests_are_cache_friendly(ollama, registry):
    router, fake = make(ollama.url, registry, say("[happy] Hi."), say("[happy] Again."))
    run(router, "Hello.")
    run(router, "Hello again.")
    first, second = fake.calls
    # Breakpoints: last tool, the system prompt, the last message block. At most 4.
    assert first["tools"][-1]["cache_control"] == {"type": "ephemeral"}
    assert all("cache_control" not in t for t in first["tools"][:-1])
    assert first["system"][0]["cache_control"] == {"type": "ephemeral"}
    assert first["messages"][-1]["content"][-1]["cache_control"] == {"type": "ephemeral"}
    marks = json.dumps(first).count("cache_control")
    assert marks == 3
    # The prefix is byte-identical from turn to turn (tools -> system -> old messages).
    assert second["tools"] == first["tools"]
    assert second["system"] == first["system"]
    blocks = first["messages"][0]["content"]
    unmarked = {**first["messages"][0],
                "content": [{k: v for k, v in b.items() if k != "cache_control"} for b in blocks]}
    assert second["messages"][0] == unmarked
    assert first["model"] == "claude-haiku-5-5"
    assert first["output_config"] == {"effort": "low"}
    assert first["thinking"] == {"type": "adaptive"}
    # The persona is roco.personality's prompt (plus the date line).
    assert "get_current_time" in first["system"][0]["text"]


def test_history_tool_calls_replay_with_stable_ids(ollama, registry):
    router, fake = make(ollama.url, registry,
                        call(("set_reminder", {"text": "call Mum", "at": "21:00"})),
                        say("[happy] Done, nine tonight."),
                        say("[neutral] Yes."))
    run(router, "Remind me to call Mum at 9pm.")
    run(router, "Did you set it?")
    replay = fake.calls[2]["messages"]
    blocks = [b for m in replay for b in m["content"]]
    use = next(b for b in blocks if b["type"] == "tool_use")
    result = next(b for b in blocks if b["type"] == "tool_result")
    assert use["id"] == result["tool_use_id"] == "toolu_h0000"
    # Converting the same history twice gives the same bytes (cache hits).
    hist = router.history.for_model()
    assert to_anthropic(hist) == to_anthropic(hist)


def test_thinking_blocks_go_back_unchanged_within_the_turn(ollama, registry):
    router, fake = make(ollama.url, registry,
                        call(("get_current_time", {}), thinking=True),
                        say("[neutral] It's twenty past four."))
    run(router, "Is it late?")
    assistant = fake.calls[1]["messages"][-2]
    assert assistant["content"][0] == {"type": "thinking", "thinking": "", "signature": "sig1"}


def test_conversion_closes_unanswered_calls_and_folds_tools_without_schemas():
    msgs = [
        {"role": "system", "content": "persona"},
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "", "tool_calls": [
            {"function": {"name": "get_events", "arguments": {}}},
            {"function": {"name": "get_current_time", "arguments": {}}}]},
        {"role": "tool", "content": "10:00", "tool_name": "get_current_time"},
        {"role": "assistant", "content": "[happy] ok"},
        {"role": "user", "content": "and?"},
    ]
    system, out = to_anthropic(msgs)
    assert system == "persona"
    results = [b for m in out for b in m["content"] if b["type"] == "tool_result"]
    by_id = {r["tool_use_id"]: r["content"] for r in results}
    assert by_id == {"toolu_h0001": "10:00", "toolu_h0000": "Cancelled: not run."}
    _, folded = to_anthropic(msgs, with_tools=False)
    assert all(b["type"] == "text" for m in folded for b in m["content"])
    assert out[0]["role"] == "user" and out[-1]["role"] == "user"


def test_reminder_claim_without_the_tool_is_repaired_on_claude(ollama, registry):
    router, fake = make(ollama.url, registry,
                        say("[happy] Done, I'll remind you."),
                        call(("set_reminder", {"text": "bins", "at": "18:00"})),
                        say("[happy] Set."))
    said, _ = run(router, "Remind me about the bins at six.")
    assert registry.calls == ["set_reminder:bins"]
    assert said == ["Done, I'll remind you."]  # the repair round is silent


def test_reminder_repair_owns_up_when_it_still_fails(ollama, registry):
    router, _ = make(ollama.url, registry, say("[happy] Done, I'll remind you."),
                     say("[neutral] Sorted."))
    said, _ = run(router, "Remind me about the bins at six.")
    assert said[-1] == REMINDER_REPAIR_FAILED


def test_time_question_is_prefetched_for_claude_too(ollama, registry):
    router, fake = make(ollama.url, registry, say("[neutral] Twenty past four."))
    run(router, "What time is it?")
    assert registry.calls == ["get_current_time"]
    last = fake.calls[0]["messages"][-1]["content"][0]
    assert last["type"] == "tool_result" and "twenty past four" in last["content"]


def test_cancel_stops_claude_and_keeps_the_turn(ollama, registry):
    cancel = threading.Event()
    router, fake = make(ollama.url, registry,
                        say("[happy] One. ", "Two. ", "Three. ", "Four.", delay=0.05))
    said: list[str] = []

    def on_sentence(s):
        said.append(s)
        cancel.set()

    router.respond_turn("Count.", on_sentence, cancel=cancel)
    assert said == ["One."]
    router.amend_last_reply("One.")
    assert router.history.messages[-1] == {"role": "assistant", "content": "One."}


def test_refusal_is_answered_in_character(ollama, registry):
    router, _ = make(ollama.url, registry, Round([], stop_reason="refusal"))
    said, _ = run(router, "Something dodgy.")
    assert said == [REFUSAL_LINE]


def test_usage_is_logged_with_cache_tokens_and_purpose(ollama, registry):
    router, _ = make(ollama.url, registry, say("[happy] Hi."))
    run(router, "Hello.")
    rec = read_records(resolve("cloud_usage.jsonl"))[-1]
    assert rec["purpose"] == "brain" and rec["model"] == "claude-haiku-5-5"
    assert rec["cache_read_input_tokens"] == 2000
    # 50 * 0.10 + 20 * 0.50 + 2000 * 0.01 per million
    assert rec["cost_usd"] == pytest.approx(35e-6)


def test_real_sdk_over_mock_http_sends_cache_marks_and_runs_tools(ollama, registry):
    anthropic = pytest.importorskip("anthropic")
    httpx2 = pytest.importorskip("httpx2")
    bodies: list[dict] = []
    responses = [
        sse_body([{"type": "tool_use", "id": "toolu_01A", "name": "get_events",
                   "input": {"day": "today"}}], stop="tool_use"),
        sse_body([{"type": "text", "text": "[happy] Dentist at ten. Brave you."}]),
    ]

    def handler(request):
        bodies.append(json.loads(request.content))
        return httpx2.Response(200, headers={"content-type": "text/event-stream"},
                               content=responses.pop(0))

    sdk = anthropic.Anthropic(
        api_key="sk-ant-test", max_retries=0,
        http_client=anthropic.DefaultHttpxClient(transport=httpx2.MockTransport(handler)))
    cfg = load_config(overrides={"llm": {"host": ollama.url}, "history": {"persist": False}})
    router = create_brain(cfg, EventBus(), registry, sdk=sdk)
    said, _ = run(router, "What's on today?")
    assert said == ["Dentist at ten.", "Brave you."]
    assert registry.calls == ["get_events:today"]
    first, second = bodies
    assert first["stream"] is True
    assert first["tools"][-1]["cache_control"] == {"type": "ephemeral"}
    assert first["system"][0]["cache_control"] == {"type": "ephemeral"}
    assert second["messages"][-2]["content"][0]["id"] == "toolu_01A"
    assert second["messages"][-1]["content"][0]["tool_use_id"] == "toolu_01A"
    assert second["messages"][-1]["content"][-1]["cache_control"] == {"type": "ephemeral"}


# =========================================================================== router


def test_no_key_runs_local_without_announcing(ollama, registry, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    cfg = load_config(overrides={"llm": {"host": ollama.url}, "history": {"persist": False}})
    router = create_brain(cfg, EventBus(), registry)
    assert router.claude is None
    ollama.script = [[content("[happy] Hello from the PC.")]]
    said, _ = run(router, "Hi.")
    assert said == ["Hello from the PC."]
    assert router.active_name == "ollama:qwen2.5:7b"


def test_cloud_error_answers_locally_then_returns_to_cloud(ollama, registry):
    clock = [1000.0]
    router, fake = make(ollama.url, registry, Round(error=RuntimeError("network down")),
                        say("[happy] Cloud again."))
    router._monotonic = lambda: clock[0]
    ollama.script = [[content("[neutral] Local here.")], [content("[neutral] Still local.")]]
    said, _ = run(router, "Hi.")
    assert said == [rt.LOCAL_CLOUD_DOWN, "Local here."]
    said, _ = run(router, "And now?")
    assert said == ["Still local."]  # inside retry_after_s: no second call, no line
    assert len(fake.calls) == 1
    clock[0] += 61
    said, _ = run(router, "Better?")
    assert said == [rt.BACK_ON_CLOUD, "Cloud again."]
    assert router.active_name.startswith("claude:")
    # The history is shared: the cloud turn sees what local said.
    texts = json.dumps(fake.calls[-1]["messages"])
    assert "Local here." in texts and "Still local." in texts


def test_failure_mid_reply_says_it_lost_the_cloud(ollama, registry):
    router, _ = make(ollama.url, registry,
                     Round(["[happy] Starting well. ", "and then"], delay=0.0))
    router.claude.client.stream_message = _fail_after_text(router.claude.client)
    ollama.script = [[content("[neutral] Local answer.")]]
    said, _ = run(router, "Tell me a story.")
    assert said == ["Starting well.", rt.LOCAL_CLOUD_LOST, "Local answer."]


def _fail_after_text(client):
    from roco.fallback.claude import CloudUnavailableError

    def stream_message(params, on_text, cancel=None, **kw):
        on_text("[happy] Starting well. And th")
        raise CloudUnavailableError("reply took longer than 30 s")

    return stream_message


def test_bad_key_disables_cloud_until_restart(ollama, registry):
    router, fake = make(ollama.url, registry, Round(error=CloudPermanentError("401 bad key")))
    ollama.script = [[content("[neutral] Local.")], [content("[neutral] Local again.")]]
    said, _ = run(router, "Hi.")
    assert said == [rt.LOCAL_BAD_KEY, "Local."]
    said, _ = run(router, "Hi again.")
    assert said == ["Local again."]
    assert len(fake.calls) == 1


def test_both_down_plays_the_offline_line(registry):
    router, _ = make(DEAD_OLLAMA, registry, Round(error=RuntimeError("no internet")))
    bus_emotions: list[Emotion] = []
    router.bus.subscribe("emotion", bus_emotions.append)
    said, reply = run(router, "Hello?")
    assert said == [rt.LOCAL_CLOUD_DOWN, OFFLINE_LINE]
    assert bus_emotions[-1] == Emotion.SAD
    assert router.history.messages == []  # nothing answered, nothing stored


def test_private_mode_and_local_down_says_the_offline_line(registry):
    router, fake = make(DEAD_OLLAMA, registry)
    run(router, "ROCO, local only.")
    said, _ = run(router, "What's the weather like?")
    assert said == [OFFLINE_LINE]
    assert fake.calls == []


def test_local_only_by_voice_keeps_everything_off_the_cloud(ollama, registry):
    router, fake = make(ollama.url, registry, say("[happy] Cloud's back."))
    said, _ = run(router, "ROCO, local only.")
    assert said == [rt.PRIVATE_ON]
    ollama.script = [[content("[neutral] Private answer.")]]
    said, _ = run(router, "What's on my mind?")
    assert said == ["Private answer."] and fake.calls == []
    assert json.loads(resolve("brain_state.json").read_text())["local_only"] is True
    # Survives a restart.
    again, fake2 = make(ollama.url, registry)
    assert again.local_only
    said, _ = run(router, "Hey ROCO, cloud back on please.")
    assert said == [rt.PRIVATE_OFF]
    said, _ = run(router, "Hello.")
    assert said == ["Cloud's back."]  # no "back on my cloud brain" twice


@pytest.mark.parametrize("phrase", ["go private", "Roco, private mode on.", "stay local",
                                    "privacy mode", "turn the cloud off"])
def test_private_phrases(ollama, registry, phrase):
    router, _ = make(ollama.url, registry)
    assert run(router, phrase)[0] == [rt.PRIVATE_ON]


def test_private_words_inside_a_question_are_not_a_command(ollama, registry):
    router, fake = make(ollama.url, registry, say("[neutral] It means offline."))
    said, _ = run(router, "What does local only mean?")
    assert said == ["It means offline."] and not router.local_only


def test_local_only_in_config_cannot_be_switched_off(ollama, registry):
    router, fake = make(ollama.url, registry, brain={"local_only": True})
    assert run(router, "cloud back on")[0] == [rt.PRIVATE_OFF_FORCED]
    ollama.script = [[content("[neutral] Local.")]]
    assert run(router, "Hi.")[0] == ["Local."] and fake.calls == []


def test_which_brain(ollama, registry):
    router, _ = make(ollama.url, registry)
    assert "cloud brain" in run(router, "Which brain are you on?")[0][0]
    run(router, "go private")
    assert "private mode is on" in run(router, "which brain are you using")[0][0]


def test_hard_question_escalates_to_the_bigger_model_for_one_turn(ollama, registry):
    router, fake = make(ollama.url, registry, say("[curious] Here's the long answer."),
                        say("[happy] Short one."))
    said, _ = run(router, "Think harder: why is the sky blue?")
    assert said == [rt.ESCALATE_LINE, "Here's the long answer."]
    assert fake.calls[0]["model"] == "claude-sonnet-5-5"
    assert fake.calls[0]["output_config"] == {"effort": "medium"}
    assert fake.calls[0]["max_tokens"] == 4096
    run(router, "And a joke?")
    assert fake.calls[1]["model"] == "claude-haiku-5-5"


def test_amend_drops_the_routers_own_lines(ollama, registry):
    router, _ = make(ollama.url, registry, Round(error=RuntimeError("down")))
    ollama.script = [[content("[neutral] One. Two. Three.")]]
    run(router, "Count.")
    router.amend_last_reply(f"{rt.LOCAL_CLOUD_DOWN} One.")
    assert router.history.messages[-1]["content"] == "One."


# =========================================================================== budget


def _log_spend(usd: float, when: datetime, model: str = "claude-haiku-5-5") -> None:
    path = resolve("cloud_usage.jsonl")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps({"ts": when.astimezone(UTC).isoformat(timespec="seconds"),
                             "model": model, "ok": True, "cost_usd": usd}) + "\n")


def _budgeted(router: BrainRouter, now: list[datetime]) -> None:
    router.budget = CloudBudget.from_config(router.config, now=lambda: now[0])


def test_budget_warns_once_at_80_percent(ollama, registry):
    now = [datetime(2026, 11, 22, 15, 0, tzinfo=LONDON)]
    router, _ = make(ollama.url, registry, say("[happy] A."), say("[happy] B."))
    _budgeted(router, now)
    _log_spend(0.17 / 0.79, now[0])  # £0.17 of £0.20
    said, _ = run(router, "One?")
    assert said == ["A.", rt.BUDGET_WARNING]
    said, _ = run(router, "Two?")
    assert said == ["B."]  # once a day
    assert json.loads(resolve("brain_state.json").read_text())["budget_warned"] == "2026-11-22"


def test_daily_cap_goes_local_until_midnight(ollama, registry):
    now = [datetime(2026, 11, 22, 23, 50, tzinfo=LONDON)]
    router, fake = make(ollama.url, registry, say("[happy] New day, cloud again."))
    _budgeted(router, now)
    _log_spend(0.21 / 0.79, now[0])
    ollama.script = [[content("[neutral] Local tonight.")]]
    said, _ = run(router, "Hi.")
    assert said == [rt.LOCAL_DAILY_CAP, "Local tonight."] and fake.calls == []
    assert router.budget.resets_at("daily") == datetime(2026, 11, 23, 0, 0, tzinfo=LONDON)
    now[0] += timedelta(minutes=15)  # past local midnight
    said, _ = run(router, "Morning.")
    assert said == [rt.BACK_ON_CLOUD, "New day, cloud again."]


def test_monthly_cap_wins_over_a_fresh_day(ollama, registry):
    now = [datetime(2026, 11, 22, 9, 0, tzinfo=LONDON)]
    router, fake = make(ollama.url, registry)
    _budgeted(router, now)
    for day in range(1, 22):  # £0.24 a day for 21 days = £5.04
        _log_spend(0.24 / 0.79, datetime(2026, 11, day, 12, tzinfo=LONDON))
    ollama.script = [[content("[neutral] Local.")]]
    said, _ = run(router, "Hi.")
    assert said == [rt.LOCAL_MONTHLY_CAP, "Local."] and fake.calls == []
    assert router.budget.resets_at("monthly") == datetime(2026, 12, 1, tzinfo=LONDON)


def test_budget_days_are_local_and_reads_are_incremental():
    path = resolve("cloud_usage.jsonl")
    now = [datetime(2026, 7, 1, 0, 30, tzinfo=LONDON)]  # BST: 23:30 UTC the day before
    budget = CloudBudget(path, usd_to_gbp=1.0, tz=LONDON, now=lambda: now[0],
                         pricing={"m": {"input": 1.0, "output": 1.0}})
    _log_spend(0.05, datetime(2026, 6, 30, 23, 40, tzinfo=UTC))  # 00:40 local, 1 July
    _log_spend(0.07, datetime(2026, 6, 30, 22, 0, tzinfo=UTC))  # 23:00 local, 30 June
    spend = budget.spend()
    assert spend.today_gbp == pytest.approx(0.05)
    assert spend.month_gbp == pytest.approx(0.05)
    with path.open("a", encoding="utf-8") as fh:  # unpriced record: estimated from tokens
        fh.write(json.dumps({"ts": "2026-06-30T23:50:00+00:00", "model": "m",
                             "input_tokens": 10000, "output_tokens": 0}) + "\n")
        fh.write("not json\n")
    assert budget.spend().today_gbp == pytest.approx(0.06)
    path.unlink()
    assert budget.spend().today_gbp == 0


def test_spend_status_and_format():
    s = Spend(today_gbp=0.1, month_gbp=1.0, daily_gbp=0.2, monthly_gbp=5.0, warn_at=0.8)
    assert s.status == "ok"
    assert Spend(0.17, 1.0, 0.2, 5.0, 0.8).status == "warn"
    assert Spend(0.2, 1.0, 0.2, 5.0, 0.8).status == "daily"
    assert Spend(0.0, 5.0, 0.2, 5.0, 0.8).status == "monthly"
    text = format_budget(s)
    assert "Today       £0.1000 of £0.20 (50%)" in text
    assert "This month  £1.0000 of £5.00 (20%)" in text


def test_cloud_usage_cli_shows_spend_against_the_caps(capsys):
    from roco.fallback.cli import run_cloud_usage

    _log_spend(0.079, datetime.now(LONDON))
    assert run_cloud_usage(load_config()) == 0
    out = capsys.readouterr().out
    assert "Today       £0.0624 of £0.20" in out
    assert "Budget: under budget" in out


# =========================================================================== server


def test_server_context_reports_the_brain_that_answered(ollama, registry, tmp_path):
    from server_fakes import FakeEngine, FakeTTS, make_ctx

    router, _ = make(ollama.url, registry, Round(error=RuntimeError("down")))
    ollama.script = [[content("[neutral] Local.")]]
    ctx = make_ctx(tmp_path, brain=router, engine=FakeEngine(), tts=FakeTTS())
    assert ctx.cloud_ready()
    assert ctx.current_brain_name() == "claude:claude-haiku-5-5"
    run(router, "Hi.")
    assert ctx.current_brain_name() == "ollama:qwen2.5:7b"
    assert not ctx.cloud_ready()  # inside the retry window


def test_claude_brain_alone_apologises_in_character(ollama, registry):
    from roco.brain.claude_brain import APOLOGY_CLOUD, create_claude_client

    cfg = load_config(overrides={"history": {"persist": False}})
    fake = FakeClaude(Round(error=RuntimeError("down")))
    brain = ClaudeBrain(cfg, EventBus(), registry, client=create_claude_client(cfg, "k", fake))
    said: list[str] = []
    brain.respond("Hi.", said.append)
    assert said == [APOLOGY_CLOUD]


# =========================================================================== fact guard


@pytest.fixture
def data_registry(registry):
    @registry.tool(description="Unread emails.")
    def get_unread_emails() -> str:
        registry.calls.append("get_unread_emails")
        return "2 unread: Halfords (MOT booked); Mum (Sunday lunch?)."

    @registry.tool(description="List reminders.")
    def list_reminders() -> str:
        registry.calls.append("list_reminders")
        return "1: ring the garage at 14:10."

    @registry.tool(description="Cancel a reminder.",
                   parameters={"type": "object", "properties": {"number": {"type": "integer"}},
                               "required": ["number"]})
    def cancel_reminder(number: int) -> str:
        registry.calls.append(f"cancel_reminder:{number}")
        return f"Reminder {number} cancelled."

    return registry


def test_invented_data_answer_is_never_spoken_and_is_fetched(ollama, data_registry):
    router, fake = make(ollama.url, data_registry,
                        say("[neutral] No new emails since you last looked."),
                        call(("get_unread_emails", {})),
                        say("[happy] Two unread: Halfords and your mum."))
    said, _ = run(router, "Do I have any unread emails?")
    assert said == ["Two unread: Halfords and your mum."]
    assert data_registry.calls == ["get_unread_emails"]
    note = fake.calls[1]["messages"][-1]["content"][-1]["text"]
    assert "get_unread_emails" in note and "made up" in note
    assert "No new emails" not in router.history.messages[-1]["content"]


def test_words_before_the_tool_are_released_when_it_runs(ollama, data_registry):
    router, _ = make(ollama.url, data_registry,
                     call(("get_unread_emails", {}), text="[curious] Let me look."),
                     say("[happy] Two unread."))
    said, _ = run(router, "Any emails?")
    assert said == ["Let me look.", "Two unread."]


def test_fetch_guard_owns_up_when_the_model_still_will_not_check(ollama, data_registry):
    from roco.brain.core import FETCH_FAILED

    router, _ = make(ollama.url, data_registry, say("[neutral] Nothing new."),
                     say("[neutral] Still nothing new."))
    said, _ = run(router, "Do I have any unread emails?")
    assert said == [FETCH_FAILED]


def test_fetch_guard_on_the_local_brain(data_registry):
    from mock_ollama import tool_calls

    server = MockOllama()
    try:
        server.script = [[content("[happy] You've got a coding meet-up at six.")],
                         [tool_calls(("get_events", {"day": "tomorrow"}))],
                         [content("[neutral] Tomorrow you've got the dentist at ten.")]]
        router, _ = make(server.url, data_registry, brain={"default": "local"})
        said, _ = run(router, "What's on my calendar tomorrow?")
    finally:
        server.close()
    assert " ".join(said) == "Tomorrow you've got the dentist at ten."
    assert data_registry.calls == ["get_events:tomorrow"]


def test_change_requests_and_chat_are_not_held(ollama, data_registry):
    router, _ = make(ollama.url, data_registry, say("[happy] Shall I add it?"),
                     say("[happy] Lovely weather."))
    assert run(router, "Add the dentist to my calendar for Friday.")[0] == ["Shall I add it?"]
    assert run(router, "Nice day, isn't it?")[0] == ["Lovely weather."]


def test_cancel_claim_without_the_tool_is_repaired(ollama, data_registry):
    router, _ = make(ollama.url, data_registry,
                     say("[happy] Cancelled, all done."),
                     call(("list_reminders", {})),
                     call(("cancel_reminder", {"number": 1})),
                     say("[happy] Gone."))
    said, _ = run(router, "Cancel the garage one.")
    assert said == ["Cancelled, all done."]
    assert data_registry.calls == ["list_reminders", "cancel_reminder:1"]


def test_cancelling_a_reminder_never_triggers_the_set_reminder_repair(ollama, data_registry):
    router, fake = make(ollama.url, data_registry,
                        call(("cancel_reminder", {"number": 1})),
                        say("[happy] Done, the Mum reminder's gone."))
    said, _ = run(router, "Cancel the reminder about Mum.")
    assert said == ["Done, the Mum reminder's gone."]
    assert len(fake.calls) == 2  # no repair round


def test_warm_up_writes_the_cache_without_streaming(ollama, registry):
    router, fake = make(ollama.url, registry, say("[happy] Hi."))
    created: list[dict] = []

    def create(**params):
        created.append(params)
        usage = {"input_tokens": 3, "output_tokens": 0, "cache_read_input_tokens": 0,
                 "cache_creation_input_tokens": 2050}
        return type("Msg", (), {"content": [], "usage": usage, "stop_reason": "max_tokens"})()

    fake.create = create
    assert router.warm_cloud() is True
    params = created[0]
    assert params["max_tokens"] == 0 and "stream" not in params
    assert params["tools"][-1]["cache_control"] == {"type": "ephemeral"}
    run(router, "Hello.")
    assert fake.calls[0]["system"] == params["system"]  # same prefix: the real turn hits it
    assert fake.calls[0]["tools"] == params["tools"]
    rec = read_records(resolve("cloud_usage.jsonl"))[0]
    assert rec["purpose"] == "warm-up" and rec["cache_creation_input_tokens"] == 2050
    router.local_only = True
    assert router.warm_cloud() is False



def test_forget_turn_drops_only_that_exchange_for_every_brain(ollama, registry):
    router, _ = make(ollama.url, registry, say("[happy] Real answer."),
                     say("[neutral] Echo answer."))
    run(router, "What's up?")
    run(router, "House three.")
    router.forget_turn("House three.")
    assert [m["content"] for m in router.history.messages] == ["What's up?", "Real answer."]
    router.forget_turn("never said")  # a turn that never reached the history
    assert len(router.history.messages) == 2
    router.local.forget_turn("What's up?")  # the backends share it
    assert router.history.messages == []
