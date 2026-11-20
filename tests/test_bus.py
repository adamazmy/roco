from __future__ import annotations

import threading

import pytest

from roco.bus import Emotion, State, Topic


def test_publish_reaches_subscribers_in_order(bus):
    seen = []
    bus.subscribe(Topic.STATE, lambda p: seen.append(("a", p)))
    bus.subscribe("state", lambda p: seen.append(("b", p)))
    bus.set_state(State.THINKING)
    assert seen == [("a", State.THINKING), ("b", State.THINKING)]


def test_unsubscribe(bus):
    seen = []
    unsubscribe = bus.subscribe(Topic.ANNOUNCE, seen.append)
    bus.announce("tea time")
    unsubscribe()
    unsubscribe()  # idempotent
    bus.announce("again")
    assert seen == ["tea time"]


def test_failing_handler_does_not_break_others(bus, caplog):
    seen = []

    def boom(_):
        raise RuntimeError("boom")

    bus.subscribe(Topic.MOUTH, boom)
    bus.subscribe(Topic.MOUTH, seen.append)
    bus.set_mouth(0.5)
    assert seen == [0.5]
    assert "boom" in caplog.text


def test_wildcard_sees_topic_and_payload(bus):
    seen = []
    bus.subscribe("*", seen.append)
    bus.set_emotion("happy")
    assert seen == [("emotion", Emotion.HAPPY)]


def test_helpers_normalise_payloads(bus):
    seen = []
    bus.subscribe("*", seen.append)
    bus.set_mouth(3.0)
    bus.set_mouth(-1)
    bus.set_emotion("Not An Emotion")
    bus.announce("   ")  # ignored
    assert seen == [("mouth", 1.0), ("mouth", 0.0), ("emotion", Emotion.NEUTRAL)]
    with pytest.raises(ValueError):
        bus.set_state("dancing")


def test_emotion_parse():
    assert Emotion.parse(" Curious ") is Emotion.CURIOUS
    assert Emotion.parse(None) is Emotion.NEUTRAL
    assert Emotion.parse("x", Emotion.SAD) is Emotion.SAD


def test_subscribing_from_a_handler_does_not_deadlock(bus):
    seen = []

    def handler(_):
        bus.subscribe(Topic.ANNOUNCE, seen.append)

    bus.subscribe(Topic.STATE, handler)
    bus.set_state(State.IDLE)
    bus.announce("hi")
    assert seen == ["hi"]


def test_concurrent_publish(bus):
    count = 0
    lock = threading.Lock()

    def inc(_):
        nonlocal count
        with lock:
            count += 1

    bus.subscribe(Topic.MOUTH, inc)
    threads = [threading.Thread(target=lambda: [bus.set_mouth(0.1) for _ in range(200)])
               for _ in range(5)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert count == 1000
