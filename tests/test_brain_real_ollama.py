"""Against a real Ollama with the configured model. Skipped in CI (``-m "not ollama"``).

Run locally with ``pytest -m ollama -s`` once ``ollama list`` shows the model.
"""

from __future__ import annotations

import httpx
import pytest

from roco.brain.ollama_brain import OllamaBrain
from roco.bus import Topic
from roco.config import DEFAULTS, load_config
from roco.paths import resolve
from roco.tools import ToolContext, build_registry
from roco.tools import reminders as rem

pytestmark = [pytest.mark.ollama, pytest.mark.timeout(300)]


@pytest.fixture(scope="module", autouse=True)
def require_model():
    host, model = DEFAULTS["llm"]["host"], DEFAULTS["llm"]["model"]
    try:
        names = {m["name"] for m in httpx.get(f"{host}/api/tags", timeout=3).json()["models"]}
    except (httpx.HTTPError, ValueError, KeyError):
        pytest.skip(f"Ollama is not reachable at {host}")
    if model not in names:
        pytest.skip(f"model {model} is not pulled")


@pytest.fixture
def brain(bus):
    cfg = load_config(overrides={"history": {"persist": False}})
    registry = build_registry(ToolContext(config=cfg, bus=bus))
    yield OllamaBrain(cfg, bus, registry)
    rem.stop_all()


def test_real_reply_has_emotion_and_sentences(brain, bus):
    emotions = []
    bus.subscribe(Topic.EMOTION, emotions.append)
    sentences: list[str] = []
    reply = brain.respond("Hello! Say hi back in one sentence.", sentences.append)
    print("\nemotion:", emotions, "\nreply:", reply)
    assert sentences and reply
    assert "[" not in reply
    assert emotions


def test_real_time_tool(brain):
    reply = brain.respond("What time is it right now?")
    print("\nreply:", reply)
    assert reply


def test_real_reminder_tool(brain):
    reply = brain.respond("Remind me to stretch in 20 minutes.")
    print("\nreply:", reply)
    pending = rem.get_service(resolve("reminders.json")).pending()
    assert any("stretch" in r.text.lower() for r in pending)
