"""The body fallback never calls Claude in private mode or past the budget caps."""

from __future__ import annotations

import json
from datetime import datetime

import pytest

from roco.config import Config, deep_merge, load_config
from roco.fallback import create_fallback, note_brain_settings
from roco.fallback.chain import (
    BUDGET_DAILY_LINE,
    BUDGET_MONTHLY_LINE,
    CLOUD_SWITCH_LINE,
    OFFLINE_LINE,
    PRIVATE_LINE,
    FallbackChain,
)
from roco.fallback.claude import ClaudeClient
from roco.fallback.guards import CloudGuard, local_only_from_settings, state_file
from roco.fallback.stt import LazyStt


class FakeStream:
    def __init__(self, *chunks):
        self.chunks = list(chunks)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    @property
    def text_stream(self):
        yield from self.chunks

    def get_final_message(self):
        return type("Msg", (), {"usage": {"input_tokens": 500, "output_tokens": 20},
                                "stop_reason": "end_turn"})()


class FakeSDK:
    def __init__(self):
        self.calls: list[dict] = []
        self.messages = self

    def stream(self, **params):
        self.calls.append(params)
        return FakeStream("[happy] Hello there. ")


class FakeStt(LazyStt):
    def __init__(self):
        super().__init__(transcribe=lambda audio: "hello")
        self.calls = 0

    def transcribe(self, pcm, rate):
        self.calls += 1
        return "hello"


class FakeBudget:
    def __init__(self, status="ok"):
        self.value = status

    def status(self):
        return self.value


@pytest.fixture(autouse=True)
def no_env_key(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)


def make(tmp_path, *, budget=None, forced=False):
    sdk = FakeSDK()
    guard = CloudGuard(tmp_path / "fallback_state.json", forced_local_only=forced,
                       budget=budget or FakeBudget())
    stt = FakeStt()
    chain = FallbackChain(ClaudeClient("sk-test", client=sdk), stt, guard=guard,
                          now=lambda: datetime(2026, 11, 22, 19, 0))
    return chain, sdk, stt, guard


def say(chain, text="hello"):
    out: list[str] = []
    chain.respond_text(text, out.append)
    return out


def hear(chain):
    out: list[str] = []
    chain.respond_audio(b"\0\0" * 1600, 16000, out.append)
    return out


# ---------------------------------------------------------------- private mode


def test_settings_parsing():
    assert local_only_from_settings({"brain": {"local_only": True}}) is True
    assert local_only_from_settings({"brain": {"local_only": False}}) is False
    for values in ({}, {"muted": True}, {"brain": {}}, {"brain": {"local_only": "yes"}},
                   {"brain": "x"}, None, []):
        assert local_only_from_settings(values) is None


def test_private_mode_blocks_the_call_and_the_stt(tmp_path):
    chain, sdk, stt, guard = make(tmp_path)
    guard.set_local_only(True)
    assert hear(chain) == [PRIVATE_LINE]
    assert sdk.calls == [] and stt.calls == 0
    # said once per outage, then just the offline line
    assert say(chain) == [OFFLINE_LINE]
    assert sdk.calls == []


def test_settings_message_turns_private_mode_on_and_off(tmp_path):
    chain, sdk, _, _ = make(tmp_path)
    assert chain.apply_settings({"brain": {"local_only": True}, "muted": False}) is True
    assert chain.local_only
    assert say(chain) == [PRIVATE_LINE]
    assert sdk.calls == []

    assert chain.apply_settings({"personality": {"roast": 50}}) is None  # no say: unchanged
    assert chain.local_only

    assert chain.apply_settings({"brain": {"local_only": False}}) is False
    assert not chain.local_only
    assert say(chain) == [CLOUD_SWITCH_LINE, "Hello there."]
    assert len(sdk.calls) == 1


def test_private_mode_persists_for_the_next_boot(tmp_path):
    cfg = load_config()
    assert note_brain_settings(cfg, {"brain": {"local_only": True}}) is True
    assert json.loads(state_file(cfg).read_text())["local_only"] is True
    # a new chain (e.g. the Pi rebooted with the PC off) still refuses the cloud
    sdk = FakeSDK()
    chain = create_fallback(cfg, client=sdk)
    assert chain.local_only
    out: list[str] = []
    chain.respond_text("hello", out.append)
    assert out == [PRIVATE_LINE] and sdk.calls == []

    assert note_brain_settings(cfg, {"brain": {"local_only": False}}) is False
    assert not create_fallback(cfg, client=sdk).local_only
    assert note_brain_settings(cfg, {"muted": True}) is None


def test_config_forces_private_mode(tmp_path):
    chain, sdk, _, guard = make(tmp_path, forced=True)
    chain.apply_settings({"brain": {"local_only": False}})  # the brain can't undo it
    assert chain.local_only
    assert say(chain) == [PRIVATE_LINE] and sdk.calls == []

    cfg = Config(deep_merge(load_config(), {"brain": {"local_only": True}}))
    assert create_fallback(cfg, client=FakeSDK()).local_only


def test_corrupt_state_file_means_not_private(tmp_path):
    chain, sdk, _, guard = make(tmp_path)
    guard.state_path.write_text("{not json", encoding="utf-8")
    assert not chain.local_only
    assert say(chain)[0] == CLOUD_SWITCH_LINE


# ---------------------------------------------------------------- budget


@pytest.mark.parametrize("status,line", [("daily", BUDGET_DAILY_LINE),
                                         ("monthly", BUDGET_MONTHLY_LINE)])
def test_over_budget_blocks_the_call(tmp_path, status, line):
    chain, sdk, stt, _ = make(tmp_path, budget=FakeBudget(status))
    assert hear(chain) == [line]
    assert say(chain) == [OFFLINE_LINE]  # explained once
    assert sdk.calls == [] and stt.calls == 0


@pytest.mark.parametrize("status", ["ok", "warn"])
def test_under_budget_passes(tmp_path, status):
    chain, sdk, stt, _ = make(tmp_path, budget=FakeBudget(status))
    assert hear(chain) == [CLOUD_SWITCH_LINE, "Hello there."]
    assert len(sdk.calls) == 1 and stt.calls == 1


def test_budget_back_under_cap_uses_the_cloud_and_announces_it(tmp_path):
    budget = FakeBudget("daily")
    chain, sdk, _, _ = make(tmp_path, budget=budget)
    assert say(chain) == [BUDGET_DAILY_LINE]
    budget.value = "ok"  # midnight
    assert say(chain) == [CLOUD_SWITCH_LINE, "Hello there."]
    budget.value = "daily"
    assert say(chain) == [BUDGET_DAILY_LINE]  # a new block is explained again


def test_brain_restored_resets_the_explanation(tmp_path):
    chain, _, _, _ = make(tmp_path, budget=FakeBudget("monthly"))
    assert say(chain) == [BUDGET_MONTHLY_LINE]
    assert chain.brain_restored_line() is not None
    assert say(chain) == [BUDGET_MONTHLY_LINE]


def test_broken_budget_does_not_block(tmp_path):
    class Broken:
        def status(self):
            raise RuntimeError("disk")

    chain, sdk, _, _ = make(tmp_path, budget=Broken())
    assert say(chain)[0] == CLOUD_SWITCH_LINE and len(sdk.calls) == 1


def test_real_budget_reads_this_machines_usage_log(tmp_path):
    """The Pi's own cloud_usage.jsonl counts against the Pi's caps."""
    from roco.fallback.usage import usage_file

    cfg = Config(deep_merge(load_config(), {"cloud": {"budget": {"daily_gbp": 0.01}}}))
    path = usage_file(cfg)
    path.parent.mkdir(parents=True, exist_ok=True)
    now = datetime.now().astimezone().isoformat(timespec="seconds")
    path.write_text(json.dumps({"ts": now, "model": "claude-haiku-5-5", "ok": True,
                                "cost_usd": 0.02}) + "\n", encoding="utf-8")
    sdk = FakeSDK()
    chain = create_fallback(cfg, client=sdk)
    out: list[str] = []
    chain.respond_text("hello", out.append)
    assert out == [BUDGET_DAILY_LINE] and sdk.calls == []


def test_import_stays_light():
    import subprocess
    import sys

    code = ("import sys; from roco.config import load_config; import roco.fallback as f; "
            "f.note_brain_settings(load_config(), {'brain': {'local_only': False}}); "
            "bad = [m for m in ('anthropic', 'faster_whisper', 'numpy') if m in sys.modules]; "
            "print(bad); raise SystemExit(1 if bad else 0)")
    assert subprocess.run([sys.executable, "-c", code], capture_output=True).returncode == 0
