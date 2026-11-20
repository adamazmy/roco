"""Conversation loop: turns, muting, follow-up chaining, announcements."""

from __future__ import annotations

import threading

from roco.brain.base import Brain
from roco.bus import State
from roco.voice.loop import APOLOGY, VoiceLoop
from voice_fakes import RecordingSpeaker, ScriptedListener


class FakeBrain(Brain):
    def __init__(self, replies=None, fail=False, on_respond=None):
        self.replies = replies or {}
        self.fail = fail
        self.heard: list[str] = []
        self.on_respond = on_respond

    def respond(self, text, on_sentence=None):
        self.heard.append(text)
        if self.on_respond:
            self.on_respond(text)
        if self.fail:
            raise RuntimeError("ollama down")
        reply = self.replies.get(text, f"You said {text}.")
        for sentence in reply.split("|"):
            if on_sentence:
                on_sentence(sentence)
        return reply.replace("|", " ")

    def reset(self):
        pass


def run_loop(listener, speaker, brain, bus, followup_s=5.0):
    loop = VoiceLoop(listener, speaker, brain, bus, followup_s=followup_s)
    stop = threading.Event()

    def stop_when_idle():
        # ScriptedListener.listen blocks once its wakes are used up; stop then.
        while not stop.is_set():
            if not listener.wakes and listener.log and listener.log[-1] == ("listen",):
                stop.set()
            threading.Event().wait(0.01)

    threading.Thread(target=stop_when_idle, daemon=True).start()
    loop.run(stop)
    return loop


def test_turn_mutes_mic_while_answering(bus):
    log: list = []
    listener = ScriptedListener(wakes=["hello"])
    listener.log = log
    speaker = RecordingSpeaker(log)
    brain = FakeBrain({"hello": "Hi Adam.|How are you?"})
    run_loop(listener, speaker, brain, bus)
    assert speaker.spoken == ["Hi Adam.", "How are you?"]
    i_mute = log.index(("muted", True))
    i_unmute = log.index(("muted", False))
    says = [i for i, e in enumerate(log) if e[0] == "say"]
    assert i_mute < says[0] and says[-1] < i_unmute
    assert log.index(("wait_done",)) < i_unmute  # unmuted only after playback finished
    assert ("followup", 5.0) in log


def test_followup_chain_without_wake_word(bus):
    listener = ScriptedListener(wakes=["what's the time"], followups=["and the date", None])
    speaker = RecordingSpeaker()
    brain = FakeBrain()
    loop = run_loop(listener, speaker, brain, bus, followup_s=4.0)
    assert brain.heard == ["what's the time", "and the date"]
    assert [e for e in listener.log if e[0] == "followup"] == [("followup", 4.0)] * 2
    assert loop.turns == 2


def test_followup_window_opens_only_after_reply_is_spoken(bus):
    log: list = []
    listener = ScriptedListener(wakes=["hi"])
    listener.log = log
    run_loop(listener, RecordingSpeaker(log), FakeBrain(), bus)
    assert log.index(("wait_done",)) < log.index(("followup", 5.0))


def test_brain_exception_is_spoken_as_apology(bus):
    listener = ScriptedListener(wakes=["hello"])
    speaker = RecordingSpeaker()
    run_loop(listener, speaker, FakeBrain(fail=True), bus)
    assert speaker.spoken == [APOLOGY]
    assert ("muted", False) in listener.log


def test_announcement_while_idle_is_spoken_immediately(bus):
    speaker = RecordingSpeaker()
    loop = VoiceLoop(ScriptedListener(), speaker, FakeBrain(), bus)
    stop = threading.Event()
    t = threading.Thread(target=loop.run, args=(stop,), daemon=True)
    t.start()
    for _ in range(100):
        if loop.listener.log:
            break
        threading.Event().wait(0.01)
    bus.announce("Time to stretch.")
    assert speaker.spoken == ["Time to stretch."]
    stop.set()
    t.join(5)
    assert not t.is_alive()


def test_announcement_during_turn_waits_until_reply_spoken(bus):
    log: list = []
    listener = ScriptedListener(wakes=["hello"])
    listener.log = log
    speaker = RecordingSpeaker(log)
    brain = FakeBrain({"hello": "First.|Second."},
                      on_respond=lambda text: bus.announce("Reminder: call mum."))
    run_loop(listener, speaker, brain, bus)
    assert speaker.spoken == ["First.", "Second.", "Reminder: call mum."]
    # spoken before the follow-up window, after the reply
    assert log.index(("say", "Reminder: call mum.")) < log.index(("followup", 5.0))


def test_stop_closes_listener_and_speaker(bus):
    listener = ScriptedListener()
    speaker = RecordingSpeaker()
    loop = VoiceLoop(listener, speaker, FakeBrain(), bus)
    stop = threading.Event()
    t = threading.Thread(target=loop.run, args=(stop,), daemon=True)
    t.start()
    stop.set()
    t.join(5)
    assert not t.is_alive()
    assert listener.closed.is_set() and ("stop",) in speaker.log


def test_loop_starts_idle_and_logs_latency(bus):
    seen: list = []
    bus.subscribe("state", seen.append)
    listener = ScriptedListener(wakes=["hi"])
    loop = run_loop(listener, RecordingSpeaker(), FakeBrain(), bus)
    assert seen[0] == State.IDLE
    assert "first_sentence" in loop.latency.summary()
