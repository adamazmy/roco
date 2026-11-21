"""Speaker queue ordering, mouth/state events and stop, with a fake TTS and output."""

from __future__ import annotations

import threading

import pytest

from roco.bus import State
from roco.voice.speaker import PiperSpeaker
from voice_fakes import FakeOutput, FakeTTS


@pytest.fixture
def events(bus):
    seen: list = []
    bus.subscribe("*", seen.append)
    return seen


def make(bus, tts=None, output=None, block_ms=100):
    return PiperSpeaker(tts or FakeTTS(), output or FakeOutput(), bus, block_ms=block_ms)


def states(events):
    return [p for t, p in events if t == "state"]


def test_sentences_play_in_fifo_order(bus):
    tts, out = FakeTTS(), FakeOutput()
    sp = make(bus, tts, out)
    for s in ["One.", "Two.", "Three.", "  ", ""]:
        sp.say(s)
    assert sp.wait_done(5)
    assert tts.calls == ["One.", "Two.", "Three."]
    # 200 samples at 1 kHz in 100 ms blocks -> 2 blocks per sentence
    assert len(out.blocks) == 6
    sp.close()


def test_state_and_mouth_events(bus, events):
    sp = make(bus)
    sp.say("Hello there.")
    sp.say("Second.")
    assert sp.wait_done(5)
    sp.close()
    assert states(events) == [State.SPEAKING, State.IDLE]  # SPEAKING once, IDLE once
    mouths = [p for t, p in events if t == "mouth"]
    assert len(mouths) == 5  # one per block (4) + final close
    assert all(m > 0.3 for m in mouths[:-1]) and mouths[-1] == 0.0
    # Order: SPEAKING before the first mouth, IDLE after the final 0.
    topics = [(t, p) for t, p in events if t in ("state", "mouth")]
    assert topics[0] == ("state", State.SPEAKING)
    assert topics[-2:] == [("mouth", 0.0), ("state", State.IDLE)]


def test_drains_output_before_reporting_done(bus):
    out = FakeOutput()
    sp = make(bus, output=out)
    sp.say("Hi.")
    assert sp.wait_done(5)
    assert out.events[-1] == "drain"
    sp.close()


def test_is_speaking_from_queue_until_played(bus):
    gate = threading.Event()
    sp = make(bus, output=FakeOutput(gate))
    assert not sp.is_speaking
    sp.say("Wait for it.")
    assert sp.is_speaking  # immediately after say(), before any audio
    assert not sp.wait_done(0.2)
    gate.set()
    assert sp.wait_done(5)
    assert not sp.is_speaking
    sp.close()


def test_no_idle_if_someone_else_changed_state(bus, events):
    gate = threading.Event()
    sp = make(bus, output=FakeOutput(gate))
    sp.say("Hello.")
    # wait until SPEAKING has been published, then the listener takes over
    for _ in range(100):
        if State.SPEAKING in states(events):
            break
        threading.Event().wait(0.01)
    bus.set_state(State.LISTENING)
    gate.set()
    assert sp.wait_done(5)
    sp.close()
    assert states(events)[-1] == State.LISTENING


def test_hold_keeps_speaking_between_slow_sentences(bus, events):
    sp = make(bus)
    with sp.hold():
        sp.say("First.")
        threading.Event().wait(0.2)  # the brain is slow producing the next sentence
        assert sp.is_speaking
        assert State.IDLE not in states(events)
        sp.say("Second.")
    assert sp.wait_done(5)
    sp.close()
    assert states(events) == [State.SPEAKING, State.IDLE]


def test_hold_without_sentences_publishes_nothing(bus, events):
    sp = make(bus)
    with sp.hold():
        pass
    assert sp.wait_done(5)
    sp.close()
    assert states(events) == []


def test_stop_drops_queue_and_aborts(bus):
    gate = threading.Event()
    tts, out = FakeTTS(samples=2000), FakeOutput(gate)
    sp = make(bus, tts, out)
    for s in ["A.", "B.", "C.", "D."]:
        sp.say(s)
    sp.stop()
    gate.set()
    assert sp.wait_done(5)
    assert len(out.blocks) <= 1  # at most the block already in flight
    sp.say("After.")
    assert sp.wait_done(5)
    assert tts.calls[-1] == "After."
    sp.close()


def test_tts_failure_skips_sentence_and_continues(bus):
    tts = FakeTTS()
    tts.fail_on = {"Bad."}
    sp = make(bus, tts)
    for s in ["Good.", "Bad.", "Also good."]:
        sp.say(s)
    assert sp.wait_done(5)
    assert tts.calls == ["Good.", "Bad.", "Also good."]
    sp.close()


def test_audio_start_hook_and_close_twice(bus):
    sp = make(bus)
    starts: list[float] = []
    sp.on_audio_start(starts.append)
    sp.say("One.")
    sp.say("Two.")
    assert sp.wait_done(5)
    sp.say("Three.")
    assert sp.wait_done(5)
    assert len(starts) == 2  # once per run, not per sentence
    sp.close()
    sp.close()
    sp.say("ignored after close")


def test_works_without_bus():
    sp = PiperSpeaker(FakeTTS(), FakeOutput(), None)
    sp.say("Quiet.")
    assert sp.wait_done(5)
    sp.close()

