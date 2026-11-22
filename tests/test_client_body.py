"""BodyClient: turn message order, playback reports, barge-in, late frames, fallback.

The test plays the brain on the far end of a memory pipe (the real LinkClient runs in
between) and drives the mic and the audio device by hand: ``step(frame)`` for every
80 ms mic frame and ``player.pull`` + ``dispatch_pending`` for the speaker.
"""

from __future__ import annotations

import random
import threading

import pytest

from client_fakes import (
    QUIET,
    SPEECH,
    Brain,
    FakeClock,
    FakeOutput,
    FakeVad,
    FakeWake,
    frame,
    wait_for,
)
from roco.bus import EventBus, State, Topic
from roco.client.body import BargeInMode, BodyClient, BodySettings, Mode
from roco.client.playback import Player
from roco.protocol import (
    AudioFrame,
    Hello,
    Interrupt,
    Ping,
    PlaybackDone,
    Sentence,
    TurnDone,
    TurnEnd,
    TurnStart,
)

WAKE = 7777
TTS = b"\x10\x00" * 1600  # 100 ms at 16 kHz


class Rig:
    def __init__(self, aec_name="webrtc", connect=True, monitor=None, fallback=None,
                 tts=None, settings=None):
        self.brain = Brain()
        self.clock = FakeClock()
        self.bus = EventBus()
        self.states = []
        self.bus.subscribe(Topic.STATE, self.states.append)
        self.player = Player(rate=16000, clock=self.clock)
        self.output = FakeOutput(self.clock, flush_s=0.004)
        self.player.output = self.output
        self.body = BodyClient(
            source=None, vad=FakeVad(), player=self.player, bus=self.bus,
            connect=self.brain.connect if connect else None, token="tok",
            wake=FakeWake(WAKE), settings=settings or BodySettings(),
            aec_name=aec_name, echo_monitor=monitor, fallback_factory=fallback,
            local_tts_factory=tts, output_latency=lambda: 0.02, clock=self.clock,
            link_options=dict(backoff_min_s=0.01, backoff_max_s=0.02, recv_timeout_s=0.02,
                              rng=random.Random(0)),
        )
        if connect:
            self.body.link.start()
            assert wait_for(lambda: self.brain.ends)
            self.hello = self.brain.accept()
            assert self.body.link.wait_up(2)

    def close(self):
        if self.body.link:
            self.body.link.stop()

    def mic(self, level, n=1):
        for _ in range(n):  # a mic delivers each frame at the end of its 80 ms
            self.clock.advance(0.08)
            self.body.step(frame(level))

    def sent(self):
        if not self.brain.ends:
            return []
        return self.brain.drain(timeout=0.15, skip=(Ping,))

    def play(self, seconds=1.0):
        for _ in range(int(seconds / 0.02)):
            self.player.pull(320)
            self.clock.advance(0.02)
        self.player.dispatch_pending()

    def user_turn(self):
        """Wake word, 0.4 s of speech, 0.48 s of silence; returns the frames sent."""
        self.mic(WAKE)
        self.mic(QUIET)  # wake tail ends
        self.mic(SPEECH, 5)
        self.mic(QUIET, 6)
        return self.sent()

    def reply(self, turn, texts=("Hello there,", "Adam.")):
        for i, text in enumerate(texts):
            self.brain.end.send(Sentence(turn=turn, index=i, text=text))
            self.brain.say(turn, i, TTS)
        self.brain.end.send(TurnDone(turn=turn, text=" ".join(texts), brain="fake"))
        assert wait_for(lambda: self.body._reply is None or self.body._reply.done
                        or self.player.playing)


@pytest.fixture
def rig():
    r = Rig()
    yield r
    r.close()


def kinds(frames):
    return [type(f).__name__ for f in frames]


# --------------------------------------------------------------------------- turns


def test_hello_advertises_aec_and_barge_in(rig):
    assert isinstance(rig.hello, Hello)
    assert rig.hello.token == "tok"
    assert rig.hello.caps["aec"] == "webrtc"
    assert rig.hello.caps["barge_in"] == "speech"


def test_turn_message_order(rig):
    sent = rig.user_turn()
    assert isinstance(sent[0], TurnStart) and sent[0].trigger == "wake" and sent[0].turn == 1
    audio = [f for f in sent if isinstance(f, AudioFrame)]
    assert [a.seq for a in audio] == list(range(len(audio)))
    assert all(a.turn == 1 and a.rate == 16000 for a in audio)
    assert len(audio) == 1 + 1 + 5 + 6  # wake frame; wake tail; speech; silence
    end = sent[-1]
    assert isinstance(end, TurnEnd) and end.reason == "silence"
    assert end.trailing_ms == 480  # 6 x 80 ms >= 450 ms
    assert kinds(sent[1:-1]) == ["AudioFrame"] * len(audio)
    assert rig.body.mode is Mode.REPLY
    assert rig.states[:2] == [State.LISTENING, State.THINKING]


def test_no_speech_after_wake_word_ends_with_nothing(rig):
    rig.mic(WAKE)
    rig.mic(QUIET, 60)  # 4 s start timeout
    sent = rig.sent()
    assert isinstance(sent[-1], TurnEnd) and sent[-1].reason == "nothing"
    assert rig.body.mode is Mode.IDLE


def test_reply_plays_then_reports_heard_and_metrics(rig):
    rig.user_turn()
    rig.clock.advance(0.5)  # brain thinking
    rig.reply(1)
    rig.play(0.5)
    assert rig.states[-1] == State.LISTENING  # follow-up window
    sent = rig.sent()
    assert kinds(sent) == ["PlaybackDone", "TurnMetrics"]
    done, metrics = sent
    assert done == PlaybackDone(turn=1, interrupted=False, heard="Hello there, Adam.")
    assert metrics.stages["vad_trailing"] == 480
    # speech ended at the last loud frame; 6 quiet frames + 0.5 s thinking + 20 ms latency
    assert metrics.stages["first_audio_played"] == pytest.approx(480 + 500 + 20, abs=5)
    assert rig.body.mode is Mode.FOLLOWUP
    assert State.SPEAKING in rig.states


def test_followup_speech_starts_a_new_turn_without_wake_word(rig):
    rig.user_turn()
    rig.reply(1)
    rig.play(0.5)
    rig.sent()
    rig.mic(SPEECH, 2)
    sent = rig.sent()
    assert isinstance(sent[0], TurnStart) and sent[0] == TurnStart(turn=2, trigger="followup")
    assert sum(isinstance(f, AudioFrame) for f in sent) >= 2  # pre-roll included


def test_followup_window_times_out(rig):
    rig.user_turn()
    rig.reply(1)
    rig.play(0.5)
    rig.mic(QUIET, 70)  # > 5 s
    assert rig.body.mode is Mode.IDLE
    assert rig.states[-1] == State.IDLE


# --------------------------------------------------------------------------- barge-in


def test_barge_in_cuts_locally_then_tells_the_brain(rig):
    rig.user_turn()
    rig.reply(1)
    rig.play(0.12)  # first sentence (100 ms) fully played, second started
    assert rig.player.playing
    rig.mic(SPEECH)  # 80 ms of speech: not yet
    assert rig.player.playing
    rig.mic(SPEECH)  # 160 ms: barge-in
    assert not rig.player.playing
    assert rig.output.flushes == 1
    sent = rig.sent()
    assert kinds(sent[:4]) == ["Interrupt", "PlaybackDone", "TurnMetrics", "TurnStart"]
    assert sent[0] == Interrupt(turn=1, reason="barge_in")
    assert sent[1] == PlaybackDone(turn=1, interrupted=True, heard="Hello there,")
    assert sent[3] == TurnStart(turn=2, trigger="barge_in")
    preroll = [f for f in sent[4:] if isinstance(f, AudioFrame)]
    assert preroll and all(f.turn == 2 for f in preroll)
    stats = rig.body.stats["barge_ins"][-1]
    # fake clock: the output flush "takes" 4 ms; speech needed 160 ms; device latency 20 ms
    assert stats["cut_ms"] == pytest.approx(4.0)
    assert stats["detect_ms"] == 160
    assert stats["stop_ms"] == pytest.approx(184, abs=1)
    assert sent[2].stages["barge_in_cut"] == 4


def test_late_frames_of_a_cancelled_turn_are_dropped(rig):
    rig.user_turn()
    rig.reply(1, texts=("One.",))
    rig.brain.say(1, 1, TTS, end=False)  # more audio queued...
    rig.play(0.05)
    rig.mic(SPEECH, 2)  # barge-in
    rig.sent()
    before = rig.body.stats["late_dropped"]
    rig.brain.end.send(Sentence(turn=1, index=2, text="Too late."))
    rig.brain.say(1, 2, TTS)
    rig.brain.end.send(TurnDone(turn=1))
    assert wait_for(lambda: rig.body.stats["late_dropped"] >= before + 3)
    assert not rig.player.playing


def test_barge_in_needs_the_wake_word_without_aec():
    r = Rig(aec_name="none")
    try:
        assert r.body.barge_mode is BargeInMode.NAME
        assert r.hello.caps["barge_in"] == "name"
        r.user_turn()
        r.reply(1)
        r.play(0.05)
        r.mic(SPEECH, 5)  # ROCO's own echo would look like this: ignored
        assert r.player.playing
        r.mic(WAKE)
        assert not r.player.playing
        assert TurnStart(turn=2, trigger="barge_in") in r.sent()
    finally:
        r.close()


class Monitor:
    def __init__(self, erle):
        self.erle_db = erle
        self.samples = 40

    def reset(self):
        pass


def test_weak_echo_cancellation_falls_back_to_wake_word(capsys):
    r = Rig(monitor=Monitor(3.0))
    try:
        r.user_turn()
        r.reply(1)
        r.play(0.5)
        assert r.body.barge_mode is BargeInMode.NAME
        assert "wake word" in capsys.readouterr().out
    finally:
        r.close()


def test_barge_ins_without_words_fall_back_to_wake_word(rig):
    from roco.protocol import Transcript

    rig.user_turn()
    for turn in (1, 2):
        rig.reply(turn)
        rig.play(0.05)
        rig.mic(SPEECH, 2)  # an echo burst the AEC let through
        rig.mic(QUIET, 6)
        rig.sent()
        assert rig.body.barge_mode is BargeInMode.SPEECH
        rig.brain.end.send(Transcript(turn=turn + 1, text=""))  # STT: no words
        if turn == 1:
            assert wait_for(lambda: rig.body._false_barges == 1)
    assert wait_for(lambda: rig.body.barge_mode is BargeInMode.NAME)


def test_a_real_barge_in_resets_the_false_count(rig):
    from roco.protocol import Transcript

    rig.user_turn()
    rig.body._false_barges = 1
    rig.reply(1)
    rig.play(0.05)
    rig.mic(SPEECH, 4)
    rig.mic(QUIET, 6)
    rig.brain.end.send(Transcript(turn=2, text="Stop."))
    assert wait_for(lambda: rig.body._false_barges == 0)
    assert rig.body.barge_mode is BargeInMode.SPEECH


def test_announcement_plays_and_can_be_interrupted(rig):
    from roco.protocol import Announce, brain_turn_id

    t = brain_turn_id(1)
    rig.brain.end.send(Announce(turn=t, text="Reminder: call mum."))
    rig.brain.end.send(Sentence(turn=t, index=0, text="Reminder: call mum."))
    rig.brain.say(t, 0, TTS * 3)
    rig.brain.end.send(TurnDone(turn=t))
    assert wait_for(lambda: rig.player.playing)
    rig.play(0.05)
    rig.mic(SPEECH, 2)
    sent = rig.sent()
    assert Interrupt(turn=t, reason="barge_in") in sent
    assert TurnStart(turn=1, trigger="barge_in") in sent


# --------------------------------------------------------------------------- fallback


class FakeFallback:
    name = "canned"
    offline_line = "My brain's offline."
    last_emotion = None

    def __init__(self):
        self.calls = []
        self.cancelled = False

    def respond_audio(self, pcm, rate, on_sentence):
        self.calls.append((len(pcm), rate))
        on_sentence("Sorry, I'm on my backup brain.")
        on_sentence("It's ten past six.")
        return "..."

    def cancel(self):
        self.cancelled = True

    def brain_restored_line(self):
        return "My proper brain's back."


def fake_tts(text):
    yield (b"\x01\x00" * 800, 16000)
    yield (b"\x01\x00" * 800, 16000)


def test_link_down_turn_goes_to_the_fallback_and_is_spoken_locally():
    fb = FakeFallback()
    r = Rig(connect=False, fallback=lambda: fb, tts=lambda: fake_tts)
    try:
        r.user_turn()
        assert wait_for(lambda: fb.calls)
        assert fb.calls[0] == (13 * 2560, 16000)  # every frame of the utterance
        assert wait_for(lambda: r.player.playing)
        # The fallback thread may still be finishing the reply when playback starts: keep
        # playing (in real time, so that thread can run) until the reply has all been heard.
        # A single 0.5 s step raced, ~50 % of runs on Windows.

        def heard_all():
            r.play(0.1)
            return r.body.mode is Mode.FOLLOWUP

        assert wait_for(heard_all)
        assert r.body.stats["turns"][-1]["local"] is True
    finally:
        r.close()


def test_link_drop_before_the_answer_asks_the_fallback(rig):
    fb = FakeFallback()
    rig.body.fallback_factory = lambda: fb
    rig.body.local_tts_factory = lambda: fake_tts
    rig.user_turn()
    rig.brain.fail_connect = True
    rig.brain.end.close()
    assert wait_for(lambda: fb.calls)
    assert wait_for(lambda: rig.player.playing)


def test_barge_in_over_a_fallback_reply_cancels_it():
    fb = FakeFallback()
    gate = threading.Event()

    def slow_tts(text):
        yield (b"\x01\x00" * 16000, 16000)
        gate.wait(1)

    r = Rig(connect=False, fallback=lambda: fb, tts=lambda: slow_tts)
    try:
        r.user_turn()
        assert wait_for(lambda: r.player.playing)
        r.play(0.05)
        r.mic(SPEECH, 2)
        assert fb.cancelled
        assert not r.player.playing
        gate.set()
    finally:
        r.close()


def test_question_said_in_one_breath_with_the_wake_word_is_not_clipped(rig):
    # Hardware test 2026-11-21: "Hey Jarvis what's on today" was heard as "on today".
    rig.mic(WAKE)
    rig.mic(SPEECH, 5)  # no pause after the wake word
    rig.mic(QUIET, 6)
    sent = rig.sent()
    audio = [f for f in sent if isinstance(f, AudioFrame)]
    assert len(audio) == 1 + 5 + 6  # every frame from the wake word on goes up
    assert isinstance(sent[-1], TurnEnd) and sent[-1].reason == "silence"


def test_pause_after_the_wake_word_does_not_end_the_turn(rig):
    rig.mic(WAKE)
    rig.mic(QUIET, 10)  # 0.8 s pause, longer than trailing_ms
    assert rig.body.mode is Mode.RECORDING
    rig.mic(SPEECH, 5)
    rig.mic(QUIET, 6)
    assert rig.sent()[-1].reason == "silence"


def test_audio_just_before_the_wake_decision_is_sent(rig):
    # openWakeWord fires late: the frames before the decision may hold the question.
    rig.mic(SPEECH, 3)  # "...Jarvis, tell me" before the detector fires
    rig.mic(WAKE)
    rig.mic(QUIET, 60)
    audio = [f for f in rig.sent() if isinstance(f, AudioFrame)]
    assert [abs(array_level(f.pcm)) for f in audio[:4]] == [SPEECH, SPEECH, SPEECH, WAKE]


def array_level(pcm: bytes) -> int:
    from array import array

    return array("h", pcm)[1]


def _write_wav(path, rate=22050, n=2205):
    import wave

    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(bytes([5, 0]) * n)
    return path


@pytest.mark.parametrize("tts_factory", ["missing", "broken"])
def test_offline_wav_plays_when_the_local_tts_cannot_speak(tmp_path, tts_factory):
    fb = FakeFallback()
    fb.offline_wav = _write_wav(tmp_path / "offline.wav", rate=16000)

    def broken(text):
        raise RuntimeError("piper crashed")
        yield  # pragma: no cover

    tts = None if tts_factory == "missing" else (lambda: broken)
    r = Rig(connect=False, fallback=lambda: fb, tts=tts)
    try:
        r.user_turn()
        assert wait_for(lambda: r.player.playing)
        assert wait_for(lambda: r.body._reply is None or r.body._reply.done)
        r.play(0.5)
        assert r.body._offline_wav_played == {1}  # once, not once per sentence
        assert r.body.mode is Mode.FOLLOWUP
    finally:
        r.close()


def test_no_offline_wav_keeps_the_old_silent_behaviour():
    fb = FakeFallback()
    fb.offline_wav = None
    r = Rig(connect=False, fallback=lambda: fb, tts=None)
    try:
        r.user_turn()
        assert wait_for(lambda: r.body.mode is Mode.FOLLOWUP or
                        (r.body._reply is not None and r.body._reply.done))
        r.player.dispatch_pending()
        assert not r.player.playing
        assert r.body._offline_wav_played == set()
    finally:
        r.close()


# --------------------------------------------------------------------------- echo guard
TIME_REPLY = ("It's just gone half past three", "on Sunday afternoon.")


def _barge_then_transcript(rig, reply_turn, text):
    from roco.protocol import Transcript

    rig.reply(reply_turn, texts=TIME_REPLY)
    rig.play(0.12)  # first sentence heard, second playing
    rig.mic(SPEECH, 2)  # barge-in
    rig.mic(QUIET, 6)  # turn ends
    rig.sent()
    rig.brain.end.send(Transcript(turn=reply_turn + 1, text=text))


def test_a_barge_in_that_was_rocos_echo_is_cancelled_before_it_is_answered(rig):
    rig.user_turn()
    _barge_then_transcript(rig, 1, "House three.")
    assert wait_for(lambda: 2 in rig.body._cancelled)
    sent = rig.sent()
    assert Interrupt(turn=2, reason="echo") in sent
    # the brain's answer to the echo may arrive anyway: dropped, never played
    for i, text in enumerate(("You've passed House three.",)):
        rig.brain.end.send(Sentence(turn=2, index=i, text=text))
        rig.brain.say(2, i, TTS)
    assert wait_for(lambda: rig.body.stats["late_dropped"] >= 2)
    assert not rig.player.playing
    assert rig.body.stats["echo_barge_ins"][0]["heard"] == "House three."
    assert rig.body.mode is Mode.FOLLOWUP
    assert rig.body.barge_mode is BargeInMode.SPEECH  # one echo is not enough


def test_a_real_barge_in_is_answered(rig):
    rig.user_turn()
    _barge_then_transcript(rig, 1, "Stop, do I have any unread emails?")
    assert wait_for(lambda: rig.body._false_barges == 0 and not rig.body._barge_turns)
    assert 2 not in rig.body._cancelled
    assert Interrupt(turn=2, reason="echo") not in rig.sent()


def test_repeated_echo_barge_ins_switch_to_name_only(rig, capsys):
    rig.user_turn()
    _barge_then_transcript(rig, 1, "on the song.")
    assert wait_for(lambda: 2 in rig.body._cancelled)
    rig.clock.advance(10)
    rig.mic(SPEECH, 3)  # a follow-up turn (3)...
    rig.mic(QUIET, 6)
    _barge_then_transcript(rig, 3, "Just go.")  # ...interrupted by echo again (4)
    assert wait_for(lambda: rig.body.barge_mode is BargeInMode.NAME)
    assert "ROCO's own echo" in capsys.readouterr().out


def test_echo_barge_ins_far_apart_do_not_switch(rig):
    rig.user_turn()
    _barge_then_transcript(rig, 1, "on the song.")
    assert wait_for(lambda: 2 in rig.body._cancelled)
    rig.clock.advance(60)  # outside the 30 s window
    rig.mic(SPEECH, 3)
    rig.mic(QUIET, 6)
    _barge_then_transcript(rig, 3, "Just go.")
    assert wait_for(lambda: 4 in rig.body._cancelled)
    assert rig.body.barge_mode is BargeInMode.SPEECH


class GateMonitor:
    """Echo monitor stand-in: says whether the current frame is above the echo."""

    def __init__(self):
        self.above = False
        self.erle_db = None
        self.samples = 0

    def near_end(self, margin_db):
        return self.above

    def reset(self):
        pass


def test_speech_below_the_residual_echo_does_not_barge_in():
    mon = GateMonitor()
    r = Rig(monitor=mon)
    try:
        r.user_turn()
        r.reply(1)
        r.play(0.05)
        r.mic(SPEECH, 6)  # the VAD says speech, but it is not above ROCO's echo
        assert r.player.playing
        mon.above = True
        r.mic(SPEECH, 2)
        assert not r.player.playing
    finally:
        r.close()


def test_barge_in_turn_carries_the_speech_before_the_decision(rig):
    rig.user_turn()
    rig.reply(1)
    rig.play(0.05)
    rig.mic(SPEECH, 2)
    audio = [f for f in rig.sent() if isinstance(f, AudioFrame) and f.turn == 2]
    assert len(audio) == 10  # 800 ms pre-roll: the decision plus a late VAD start


def test_wake_turn_carries_the_whole_wake_phrase(rig):
    rig.mic(SPEECH, 20)  # "hey Jarvis" and earlier noise
    rig.mic(WAKE)
    rig.mic(QUIET, 60)
    audio = [f for f in rig.sent() if isinstance(f, AudioFrame)]
    first = [abs(array_level(f.pcm)) for f in audio[:16]]
    assert first == [SPEECH] * 14 + [WAKE, QUIET]  # 1.2 s = 15 frames incl. the wake frame


def test_barge_in_mode_accepts_the_old_name():
    assert BargeInMode.parse("wake_word") is BargeInMode.NAME
    assert BargeInMode.parse("name") is BargeInMode.NAME
    assert BargeInMode.parse(None) is BargeInMode.SPEECH
