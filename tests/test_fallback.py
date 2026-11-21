"""Fallback brain (workstream D): chain switching, Claude client, cost log, key loading.

No network: the Claude client is tested against the real ``anthropic`` SDK over an
``httpx2.MockTransport`` (skipped if the SDK is missing) and against a fake SDK object.
"""

from __future__ import annotations

import json
import os
import sys
import threading
import time
from datetime import UTC, datetime
from pathlib import Path

import pytest

from roco.bus import Emotion
from roco.cli import main
from roco.config import load_config
from roco.fallback import (
    BRAIN_RESTORED_LINE,
    CLOUD_SWITCH_LINE,
    OFFLINE_LINE,
    create_fallback,
)
from roco.fallback.chain import CLOUD_LOST_LINE, FallbackChain
from roco.fallback.claude import (
    ClaudeClient,
    CloudPermanentError,
    CloudUnavailableError,
)
from roco.fallback.keys import key_file, load_api_key, mask, save_api_key
from roco.fallback.persona import fallback_system_prompt
from roco.fallback.stt import LazyStt
from roco.fallback.usage import (
    Usage,
    UsageLog,
    estimate_cost,
    format_summary,
    read_records,
    summarize,
)

HAIKU_PRICE = {"claude-haiku-5-5": {"input": 0.10, "output": 0.50}}


@pytest.fixture(autouse=True)
def no_env_key(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)


# =========================================================================== fakes


class FakeStream:
    def __init__(self, chunks, usage, stop_reason="end_turn", error=None, error_after=None):
        self.chunks = chunks
        self.usage = usage
        self.stop_reason = stop_reason
        self.error = error
        self.error_after = error_after
        self.closed = False

    def __enter__(self):
        if self.error is not None and self.error_after is None:
            raise self.error
        return self

    def __exit__(self, *exc):
        self.closed = True
        return False

    @property
    def text_stream(self):
        for i, chunk in enumerate(self.chunks):
            if self.error is not None and self.error_after == i:
                raise self.error
            yield chunk

    def get_final_message(self):
        return type("Msg", (), {"usage": self.usage, "stop_reason": self.stop_reason})()

    @property
    def current_message_snapshot(self):
        return type("Snap", (), {"usage": {"input_tokens": self.usage["input_tokens"]}})()


class FakeSDK:
    """Stands in for ``anthropic.Anthropic``: ``.messages.stream(**params)``."""

    def __init__(self, *streams):
        self.streams = list(streams)
        self.calls: list[dict] = []
        self.messages = self

    def stream(self, **params):
        self.calls.append(params)
        return self.streams.pop(0)


def ok_stream(*chunks, usage=None):
    return FakeStream(list(chunks), usage or {"input_tokens": 400, "output_tokens": 30})


def make_client(tmp_path, sdk, **kw) -> ClaudeClient:
    return ClaudeClient("sk-test", usage_log=UsageLog(tmp_path / "usage.jsonl", HAIKU_PRICE),
                        client=sdk, **kw)


class FakeStt(LazyStt):
    def __init__(self, text="what time is it", error=None):
        super().__init__(transcribe=lambda audio: text)
        self.text = text
        self.error = error
        self.calls = 0

    def transcribe(self, pcm, rate):
        self.calls += 1
        if self.error:
            raise self.error
        return self.text


def make_chain(tmp_path, sdk, stt=None, **kw) -> FallbackChain:
    claude = make_client(tmp_path, sdk) if sdk is not None else None
    return FallbackChain(claude, stt or FakeStt(),
                         now=lambda: datetime(2026, 11, 21, 16, 20), **kw)


# =========================================================================== chain


def test_create_fallback_without_key_is_canned_and_light(config):
    fb = create_fallback(config)
    assert fb.name == "canned"
    assert fb.offline_line == OFFLINE_LINE
    said: list[str] = []
    assert fb.respond_audio(b"\x00\x00" * 1600, 16000, said.append) == OFFLINE_LINE
    assert said == [OFFLINE_LINE]
    assert fb.last_emotion is Emotion.SAD


def test_create_fallback_with_key_file_uses_claude(config):
    save_api_key(config, "sk-ant-abcdefghijklmnop")
    fb = create_fallback(config)
    assert fb.name == "claude:claude-haiku-5-5"


def test_cloud_disabled_in_config_is_canned(config):
    save_api_key(config, "sk-ant-abcdefghijklmnop")
    cfg = load_config(overrides={"cloud": {"enabled": False}})
    assert create_fallback(cfg).name == "canned"


def test_first_reply_announces_cloud_then_streams_clean_sentences(tmp_path):
    sdk = FakeSDK(ok_stream("[hap", "py] It's twenty past ", "four. Tea ", "time? Go on then."),
                  ok_stream("[curious] Still here."))
    fb = make_chain(tmp_path, sdk)
    said: list[str] = []
    emotions: list[Emotion | None] = []

    def on_sentence(s):
        said.append(s)
        emotions.append(fb.last_emotion)

    text = fb.respond_audio(b"\x01\x00" * 800, 16000, on_sentence)
    assert said == [CLOUD_SWITCH_LINE, "It's twenty past four.", "Tea time?", "Go on then."]
    assert text.startswith(CLOUD_SWITCH_LINE)
    assert fb.last_emotion is Emotion.HAPPY
    assert emotions[1] is Emotion.HAPPY  # known before the first Claude sentence
    # Second turn: no announcement again, history carried.
    said.clear()
    fb.respond_text("you there?", said.append)
    assert said == ["Still here."]
    msgs = sdk.calls[1]["messages"]
    assert [m["role"] for m in msgs] == ["user", "assistant", "user"]
    # History keeps only Claude's own words, not the switch announcement.
    assert msgs[1]["content"] == "It's twenty past four. Tea time? Go on then."


def test_system_prompt_carries_clock_and_limits(tmp_path):
    sdk = FakeSDK(ok_stream("[happy] Hi."))
    fb = make_chain(tmp_path, sdk, describe_now=lambda now: "twenty past four in the afternoon")
    fb.respond_text("hello", lambda s: None)
    params = sdk.calls[0]
    assert params["model"] == "claude-haiku-5-5"
    assert params["output_config"] == {"effort": "low"}
    assert "twenty past four in the afternoon" in params["system"]
    assert "calendar" in params["system"] and "never pretend" in params["system"]
    assert "temperature" not in params


def test_empty_transcript_returns_empty_without_calling_claude(tmp_path):
    sdk = FakeSDK()
    fb = make_chain(tmp_path, sdk, stt=FakeStt(text=""))
    said: list[str] = []
    assert fb.respond_audio(b"\x00\x00" * 10, 16000, said.append) == ""
    assert said == [] and sdk.calls == []


def test_stt_failure_says_offline_line(tmp_path):
    fb = make_chain(tmp_path, FakeSDK(), stt=FakeStt(error=ImportError("no faster_whisper")))
    said: list[str] = []
    assert fb.respond_audio(b"\x00\x00", 16000, said.append) == OFFLINE_LINE
    assert said == [OFFLINE_LINE]


def test_temporary_cloud_failure_says_offline_line_then_recovers(tmp_path):
    sdk = FakeSDK(FakeStream([], {}, error=_unavailable()), ok_stream("[happy] Back."))
    fb = make_chain(tmp_path, sdk)
    said: list[str] = []
    fb.respond_text("hi", said.append)
    assert said == [CLOUD_SWITCH_LINE, OFFLINE_LINE]
    assert fb.name == "claude:claude-haiku-5-5"  # still tries next turn
    said.clear()
    fb.respond_text("hi again", said.append)
    assert said == ["Back."]


def test_failure_mid_reply_says_lost_line(tmp_path):
    stream = FakeStream(["[happy] First bit. ", "Second", " half"], {"input_tokens": 1},
                        error=_unavailable(), error_after=2)
    fb = make_chain(tmp_path, FakeSDK(stream))
    said: list[str] = []
    fb.respond_text("hi", said.append)
    assert said == [CLOUD_SWITCH_LINE, "First bit.", CLOUD_LOST_LINE]


def test_bad_key_switches_to_canned_until_brain_restored(tmp_path):
    sdk = FakeSDK(FakeStream([], {}, error=CloudPermanentError("401 invalid x-api-key")),
                  ok_stream("[happy] Fixed."))
    fb = make_chain(tmp_path, sdk)
    said: list[str] = []
    fb.respond_text("hi", said.append)
    assert said[-1] == OFFLINE_LINE
    assert fb.name == "canned"
    said.clear()
    fb.respond_audio(b"\x00\x00", 16000, said.append)
    assert said == [OFFLINE_LINE] and len(sdk.calls) == 1  # no more cloud calls
    assert fb.brain_restored_line() == BRAIN_RESTORED_LINE
    assert fb.name == "claude:claude-haiku-5-5"  # another chance after the next outage


def test_brain_restored_line_only_after_fallback_spoke(tmp_path):
    fb = make_chain(tmp_path, FakeSDK(ok_stream("[happy] Hi."), ok_stream("[happy] Hi.")))
    assert fb.brain_restored_line() is None  # brain blipped, fallback never used
    said: list[str] = []
    fb.respond_text("hi", said.append)
    assert fb.brain_restored_line() == BRAIN_RESTORED_LINE
    assert fb.brain_restored_line() is None
    # Down again: announces again, history was cleared.
    said.clear()
    fb.respond_text("hi", said.append)
    assert said[0] == CLOUD_SWITCH_LINE
    assert len(fb._claude._client.calls[1]["messages"]) == 1


def test_cancel_stops_further_sentences(tmp_path):
    fb = make_chain(tmp_path, FakeSDK(ok_stream("[happy] One. ", "Two. ", "Three. ", "Four.")))
    said: list[str] = []

    def on_sentence(s):
        said.append(s)
        if s == "One.":
            fb.cancel()

    fb.respond_text("count", on_sentence)
    assert said == [CLOUD_SWITCH_LINE, "One."]
    records = read_records(tmp_path / "usage.jsonl")
    assert records[-1]["error"] == "cancelled" and records[-1]["input_tokens"] == 400


def test_refusal_without_text_says_something(tmp_path):
    stream = FakeStream([], {"input_tokens": 10, "output_tokens": 0}, stop_reason="refusal")
    fb = make_chain(tmp_path, FakeSDK(stream))
    said: list[str] = []
    fb.respond_text("hm", said.append)
    assert said == [CLOUD_SWITCH_LINE, "Sorry, I didn't catch that."]


def test_history_is_bounded(tmp_path):
    sdk = FakeSDK(*[ok_stream(f"[happy] Reply {i}.") for i in range(5)])
    fb = make_chain(tmp_path, sdk, history_turns=2)
    for i in range(5):
        fb.respond_text(f"q{i}", lambda s: None)
    assert len(sdk.calls[-1]["messages"]) == 5  # 2 pairs + the new question


# =========================================================================== Claude client


def test_client_streams_and_logs_cost(tmp_path):
    seen: list[str] = []
    client = make_client(tmp_path, FakeSDK(ok_stream("Hello ", "there.",
                                                     usage={"input_tokens": 1000,
                                                            "output_tokens": 100})))
    reply = client.stream_reply("sys", [{"role": "user", "content": "hi"}], seen.append)
    assert reply.text == "Hello there." and seen == ["Hello ", "there."]
    assert reply.usage == Usage(1000, 100)
    rec = read_records(tmp_path / "usage.jsonl")[0]
    assert rec["ok"] is True and rec["model"] == "claude-haiku-5-5"
    assert rec["cost_usd"] == pytest.approx((1000 * 0.10 + 100 * 0.50) / 1e6)
    assert rec["first_text_ms"] is not None


def test_client_total_timeout(tmp_path):
    ticks = iter([0.0, 0.1, 0.2, 30.0, 30.1, 30.2, 30.3])
    client = make_client(tmp_path, FakeSDK(ok_stream("a", "b", "c", "d")),
                         total_timeout_s=20.0, clock=lambda: next(ticks))
    with pytest.raises(CloudUnavailableError, match="longer than 20"):
        client.stream_reply("s", [{"role": "user", "content": "x"}], lambda t: None)
    assert read_records(tmp_path / "usage.jsonl")[0]["ok"] is False


class StalledStream(FakeStream):
    """Connects, then never sends a word (what a stalled connection looks like)."""

    def __init__(self, release: threading.Event):
        super().__init__([], {"input_tokens": 0, "output_tokens": 0})
        self.release = release

    @property
    def text_stream(self):
        self.release.wait(5)
        return iter(())


def test_client_gives_up_when_no_text_arrives(tmp_path):
    release = threading.Event()
    client = make_client(tmp_path, FakeSDK(StalledStream(release)), first_text_timeout_s=0.2)
    t = time.monotonic()
    try:
        with pytest.raises(CloudUnavailableError, match="no reply within"):
            client.stream_reply("s", [{"role": "user", "content": "x"}], lambda t: None)
    finally:
        release.set()
    assert time.monotonic() - t < 2
    rec = read_records(tmp_path / "usage.jsonl")[0]
    assert rec["ok"] is False and "no reply" in rec["error"]


def test_client_cancel_while_waiting(tmp_path):
    release = threading.Event()
    cancel = threading.Event()
    client = make_client(tmp_path, FakeSDK(StalledStream(release)), first_text_timeout_s=5)
    threading.Timer(0.1, cancel.set).start()
    try:
        reply = client.stream_reply("s", [{"role": "user", "content": "x"}], lambda t: None,
                                    cancel=cancel)
    finally:
        release.set()
    assert reply.cancelled and reply.text == ""


def test_client_without_key_is_permanent_error(tmp_path):
    client = ClaudeClient(None, usage_log=UsageLog(tmp_path / "u.jsonl"))
    with pytest.raises(CloudPermanentError):
        client.stream_reply("s", [{"role": "user", "content": "x"}], lambda t: None)


def test_client_from_config_builds_sdk_with_timeouts(config):
    anthropic = pytest.importorskip("anthropic")
    client = ClaudeClient.from_config(config, "sk-ant-test")
    sdk = client._sdk()
    assert isinstance(sdk, anthropic.Anthropic)
    assert sdk.max_retries == 1
    assert sdk.timeout.connect == 3.0 and sdk.timeout.read == 8.0


# ---- real SDK over a mocked HTTP transport


def _sse(*events: dict) -> bytes:
    return "".join(f"event: {e['type']}\ndata: {json.dumps(e)}\n\n" for e in events).encode()


def _stream_body(texts, input_tokens=420, output_tokens=25, stop="end_turn") -> bytes:
    events = [
        {"type": "message_start", "message": {
            "id": "msg_1", "type": "message", "role": "assistant", "model": "claude-haiku-5-5",
            "content": [], "stop_reason": None, "stop_sequence": None,
            "usage": {"input_tokens": input_tokens, "output_tokens": 1}}},
        {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
    ]
    events += [{"type": "content_block_delta", "index": 0,
                "delta": {"type": "text_delta", "text": t}} for t in texts]
    events += [
        {"type": "content_block_stop", "index": 0},
        {"type": "message_delta", "delta": {"stop_reason": stop, "stop_sequence": None},
         "usage": {"output_tokens": output_tokens}},
        {"type": "message_stop"},
    ]
    return _sse(*events)


def _sdk_with(handler):
    anthropic = pytest.importorskip("anthropic")
    httpx2 = pytest.importorskip("httpx2")
    return anthropic.Anthropic(
        api_key="sk-ant-test", max_retries=0,
        http_client=anthropic.DefaultHttpxClient(transport=httpx2.MockTransport(handler)),
    )


def test_real_sdk_streaming_over_mock_http(tmp_path):
    httpx2 = pytest.importorskip("httpx2")
    requests = []

    def handler(request):
        requests.append(json.loads(request.content))
        return httpx2.Response(200, headers={"content-type": "text/event-stream"},
                               content=_stream_body(["[happy] Morning", ", you. ", "Coffee?"]))

    fb = make_chain(tmp_path, _sdk_with(handler))
    said: list[str] = []
    fb.respond_text("morning", said.append)
    assert said == [CLOUD_SWITCH_LINE, "Morning, you.", "Coffee?"]
    body = requests[0]
    assert body["stream"] is True and body["model"] == "claude-haiku-5-5"
    assert body["output_config"] == {"effort": "low"}
    rec = read_records(tmp_path / "usage.jsonl")[0]
    assert (rec["input_tokens"], rec["output_tokens"]) == (420, 25)


@pytest.mark.parametrize(
    ("status", "expected"),
    [(401, CloudPermanentError), (404, CloudPermanentError), (429, CloudUnavailableError),
     (529, CloudUnavailableError), (500, CloudUnavailableError)],
)
def test_real_sdk_http_errors_are_classified(tmp_path, status, expected):
    httpx2 = pytest.importorskip("httpx2")

    def handler(request):
        return httpx2.Response(status, json={"type": "error", "error": {
            "type": "some_error", "message": f"status {status}"}})

    client = make_client(tmp_path, _sdk_with(handler))
    with pytest.raises(expected):
        client.stream_reply("s", [{"role": "user", "content": "x"}], lambda t: None)
    rec = read_records(tmp_path / "usage.jsonl")[0]
    assert rec["ok"] is False and rec["error"]


@pytest.mark.parametrize("exc_name", ["ReadTimeout", "ConnectError", "ConnectTimeout"])
def test_real_sdk_network_failures_are_temporary(tmp_path, exc_name):
    httpx2 = pytest.importorskip("httpx2")

    def handler(request):
        raise getattr(httpx2, exc_name)("boom", request=request)

    client = make_client(tmp_path, _sdk_with(handler))
    with pytest.raises(CloudUnavailableError):
        client.stream_reply("s", [{"role": "user", "content": "x"}], lambda t: None)


def test_real_sdk_permanent_error_drops_chain_to_canned(tmp_path):
    httpx2 = pytest.importorskip("httpx2")

    def handler(request):
        return httpx2.Response(401, json={"type": "error", "error": {
            "type": "authentication_error", "message": "invalid x-api-key"}})

    fb = make_chain(tmp_path, _sdk_with(handler))
    said: list[str] = []
    fb.respond_text("hi", said.append)
    assert said == [CLOUD_SWITCH_LINE, OFFLINE_LINE] and fb.name == "canned"


# =========================================================================== cost log


def test_estimate_cost_and_unknown_model():
    usage = Usage(input_tokens=2_000_000, output_tokens=1_000_000,
                  cache_read_input_tokens=1_000_000, cache_creation_input_tokens=0)
    price = {"m": {"input": 1.0, "output": 5.0, "cache_read": 0.1}}
    assert estimate_cost("m", usage, price) == pytest.approx(2.0 + 5.0 + 0.1)
    assert estimate_cost("other", usage, price) is None


def test_usage_log_summary_by_month_and_day(tmp_path):
    log = UsageLog(tmp_path / "u.jsonl", HAIKU_PRICE)
    log.record("claude-haiku-5-5", Usage(1000, 100), ok=True,
               now=datetime(2026, 11, 20, 9, tzinfo=UTC))
    log.record("claude-haiku-5-5", Usage(500, 0), ok=False, error="timeout",
               now=datetime(2026, 11, 21, 9, tzinfo=UTC))
    log.record("mystery", Usage(10, 10), ok=True, now=datetime(2026, 12, 1, tzinfo=UTC))
    (tmp_path / "u.jsonl").open("a").write("not json\n")
    records = read_records(tmp_path / "u.jsonl")
    assert len(records) == 3
    months = summarize(records)
    assert months["2026-11"].calls == 2 and months["2026-11"].failed == 1
    assert months["total"].unpriced == 1
    assert summarize(records, by="day")["2026-11-21"].input_tokens == 500
    text = format_summary(months, tmp_path / "u.jsonl")
    assert "2026-11" in text and "mystery" in text and "*" in text


def test_usage_log_never_raises_on_unwritable_path(tmp_path):
    blocker = tmp_path / "file"
    blocker.write_text("x")
    entry = UsageLog(blocker / "sub" / "u.jsonl").record("m", Usage(), ok=True)
    assert entry["cost_usd"] is None


def test_cloud_usage_command(capsys, isolated_home):
    assert main(["cloud-usage"]) == 0
    assert "No cloud calls" in capsys.readouterr().out
    UsageLog(isolated_home / "cloud_usage.jsonl", HAIKU_PRICE).record(
        "claude-haiku-5-5", Usage(1000, 100), ok=True)
    assert main(["cloud-usage", "--by", "day"]) == 0
    assert "claude-haiku-5-5" in capsys.readouterr().out


# =========================================================================== keys


def test_key_from_env_wins(config, monkeypatch):
    save_api_key(config, "sk-ant-file-key-123456")
    monkeypatch.setenv("ANTHROPIC_API_KEY", " sk-ant-env-key-123456 ")
    assert load_api_key(config) == "sk-ant-env-key-123456"


def test_key_file_roundtrip_and_permissions(config, isolated_home):
    assert load_api_key(config) is None
    path = save_api_key(config, "  sk-ant-abcdefghijklmnop\n")
    assert path == isolated_home / "anthropic_key" == key_file(config)
    assert load_api_key(config) == "sk-ant-abcdefghijklmnop"
    if os.name == "posix":
        assert path.stat().st_mode & 0o777 == 0o600
    with pytest.raises(ValueError):
        save_api_key(config, "   ")
    with pytest.raises(ValueError):
        save_api_key(config, "two words")
    assert mask("sk-ant-abcdefghijklmnop") == "sk-ant-...mnop"
    assert "abcdefgh" not in mask("sk-ant-abcdefghijklmnop")


def test_empty_key_file_means_no_key(config, isolated_home):
    (isolated_home / "anthropic_key").write_text("\n\n")
    assert load_api_key(config) is None


def test_cloud_key_command_from_stdin(monkeypatch, capsys, isolated_home):
    monkeypatch.setattr(sys, "stdin", __import__("io").StringIO("sk-ant-fromstdin-0001\n"))
    assert main(["cloud-key", "--stdin"]) == 0
    out = capsys.readouterr().out
    assert "sk-ant-fromstdin-0001" not in out and "...0001" in out
    assert (isolated_home / "anthropic_key").read_text().strip() == "sk-ant-fromstdin-0001"
    assert main(["cloud-key", "--show"]) == 0


def test_key_file_is_gitignored():
    gitignore = Path(__file__).resolve().parents[1] / ".gitignore"
    text = gitignore.read_text(encoding="utf-8")
    assert "anthropic_key" in text and ".roco/" in text


# =========================================================================== misc


def test_persona_is_compact_and_spoken():
    prompt = fallback_system_prompt(datetime(2026, 11, 21, 16, 20), describe="tea time")
    assert "tea time" in prompt and "[happy]" in prompt
    assert len(prompt) < 2500  # paid for on every cloud turn


def test_pcm_resampling():
    np = pytest.importorskip("numpy")
    from roco.fallback.stt import pcm16_to_float32

    pcm = (np.ones(48000, dtype="<i2") * 16384).tobytes()
    audio = pcm16_to_float32(pcm, 48000)
    assert audio.dtype == np.float32 and len(audio) == 16000
    assert audio[100] == pytest.approx(0.5)
    assert len(pcm16_to_float32(pcm + b"\x00", 16000)) == 48000  # odd byte dropped


def test_lazy_stt_loads_once():
    loads = []

    class Probe(LazyStt):
        def _load(self):
            loads.append(1)
            return lambda audio: " hello "

    pytest.importorskip("numpy")
    stt = Probe()
    assert stt.transcribe(b"", 16000) == "" and loads == []
    assert stt.transcribe(b"\x00\x00" * 100, 16000) == "hello"
    assert stt.transcribe(b"\x00\x00" * 100, 16000) == "hello"


def test_cancel_is_thread_safe(tmp_path):
    fb = make_chain(tmp_path, FakeSDK(ok_stream("[happy] Hi.")))
    t = threading.Thread(target=fb.cancel)
    t.start()
    t.join()
    fb.respond_text("hi", lambda s: None)  # a new turn clears the flag
    assert not fb._cancel.is_set()


def _unavailable():
    return CloudUnavailableError("connection reset")
