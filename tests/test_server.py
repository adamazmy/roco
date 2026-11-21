"""The brain server over a memory pipe with fake STT/TTS/brain, plus one real WebSocket test."""

from __future__ import annotations

import json
import threading
import time

import pytest

from roco.bus import EventBus
from roco.protocol import (
    Announce,
    AudioFrame,
    EmotionChange,
    Error,
    Interrupt,
    Ping,
    PlaybackDone,
    Pong,
    Sentence,
    Settings,
    StateChange,
    TextInput,
    Transcript,
    TurnDone,
    TurnEnd,
    TurnMetrics,
    TurnStart,
    Welcome,
    is_brain_turn,
)
from server_fakes import (
    TOKEN,
    TTS_RATE,
    FakeBody,
    FakeTTS,
    InlineExecutor,
    ScriptBrain,
    describe,
    drain,
    make_ctx,
)


@pytest.fixture
def bus():
    return EventBus()


@pytest.fixture
def body_factory(isolated_home):
    bodies: list[FakeBody] = []

    def make(ctx):
        body = FakeBody(ctx)
        bodies.append(body)
        return body

    yield make
    for body in bodies:
        body.close()


def voice_turn(body, turn=1, words=4, trailing_ms=480):
    body.send(TurnStart(turn=turn))
    body.speak(turn, words)
    body.send(TurnEnd(turn=turn, reason="silence", trailing_ms=trailing_ms))


def control(frames):
    return [f for f in frames if not isinstance(f, AudioFrame)]


def last_latency_record(home, timeout=5.0):
    path = home / "latency.jsonl"
    deadline = time.monotonic() + timeout
    while not path.exists() and time.monotonic() < deadline:
        time.sleep(0.02)
    return json.loads(path.read_text(encoding="utf-8").splitlines()[-1])


# ---------------------------------------------------------------------- handshake
def test_welcome(isolated_home, body_factory):
    ctx = make_ctx(isolated_home)
    ctx.apply_settings({"personality": {"roast": 40}})
    welcome = body_factory(ctx).hello()
    assert isinstance(welcome, Welcome)
    assert welcome.brain == "fake:brain"
    assert welcome.tts_rate == TTS_RATE
    assert welcome.session
    assert welcome.settings == {"personality": {"roast": 40}}


@pytest.mark.parametrize(
    ("kwargs", "code"),
    [({"token": "wrong"}, "auth"), ({"protocol": 99}, "version")],
)
def test_handshake_refusals(isolated_home, body_factory, kwargs, code):
    body = body_factory(make_ctx(isolated_home))
    error = body.hello(**kwargs)
    assert isinstance(error, Error)
    assert (error.code, error.fatal) == (code, True)
    body.thread.join(timeout=2)
    assert not body.thread.is_alive()


def test_first_message_must_be_hello(isolated_home, body_factory):
    body = body_factory(make_ctx(isolated_home))
    body.send(Ping(id=1, ts=0.0))
    error = body.conn.recv(timeout=5)
    assert isinstance(error, Error) and error.code == "bad_message" and error.fatal


def test_second_body_is_busy_until_the_first_leaves(isolated_home, body_factory):
    ctx = make_ctx(isolated_home)
    first = body_factory(ctx)
    assert isinstance(first.hello(), Welcome)
    second = body_factory(ctx)
    error = second.hello()
    assert isinstance(error, Error) and error.code == "busy" and error.fatal
    first.close()
    third = body_factory(ctx)
    assert isinstance(third.hello(), Welcome)


def test_ping_and_settings_broadcast(isolated_home, body_factory):
    ctx = make_ctx(isolated_home)
    seen = []
    ctx.on_settings(seen.append)
    body = body_factory(ctx)
    body.hello()
    body.send(Ping(id=7, ts=12.5))
    assert body.conn.recv(timeout=5) == Pong(id=7, ts=12.5)
    body.send(Settings(values={"personality": {"roast": 80}}))
    body.send(Settings(values={"muted": True}))
    first = body.conn.recv(timeout=5)
    second = body.conn.recv(timeout=5)
    assert first.values == {"personality": {"roast": 80}}
    assert second.values == {"personality": {"roast": 80}, "muted": True}
    assert seen[-1] == second.values


# ---------------------------------------------------------------------- a full turn
def test_full_voice_turn_message_order_and_latency(isolated_home, body_factory, bus):
    brain = ScriptBrain(bus)
    ctx = make_ctx(isolated_home, brain=brain, bus=bus)
    body = body_factory(ctx)
    body.hello()
    voice_turn(body, turn=1, words=4)
    frames = body.until_turn_done(1)
    names = [describe(f) for f in frames]
    assert names == [
        "state",
        "transcript('w0 w1 w2 w3')",
        "emotion",
        "sentence('Hello Adam.')",
        "audio(s0)", "audio(s0)", "audio(s0)", "audio(s0 END)",
        "sentence('Nice to see you.')",
        "audio(s1)", "audio(s1)", "audio(s1)", "audio(s1 END)",
        "turn_done('Hello Adam. Nice to see you.')",
    ], names
    assert brain.heard == ["w0 w1 w2 w3"]
    state, _, emotion = frames[:3]
    assert state == StateChange(state="thinking", turn=1)
    assert emotion == EmotionChange(emotion="happy", turn=1)
    audio = [f for f in frames if isinstance(f, AudioFrame)]
    assert [f.seq for f in audio] == list(range(len(audio)))
    assert all(f.turn == 1 and f.rate == TTS_RATE for f in audio)
    assert len(audio[0].pcm) == TTS_RATE * 80 // 1000 * 2  # 80 ms blocks
    done = frames[-1]
    assert set(done.stages) >= {"stt_done", "first_token", "first_sentence",
                                "first_tts_done", "first_audio_sent"}
    stages = done.stages
    assert stages["stt_done"] <= stages["first_token"] <= stages["first_audio_sent"]
    assert stages["stt_done"] >= 470  # measured from end of speech, not from turn_end

    body.send(PlaybackDone(turn=1, heard="Hello Adam. Nice to see you."))
    body.send(TurnMetrics(turn=1, stages={"first_audio_played": 900, "vad_trailing": 480}))
    record = last_latency_record(isolated_home)
    assert record["turn"] == 1
    assert record["body"] == {"first_audio_played": 900, "vad_trailing": 480}
    assert record["stages"]["first_audio_sent"] == stages["first_audio_sent"]
    assert record["transcript"] == "w0 w1 w2 w3"
    assert brain.amended == []  # not interrupted: the history keeps the full reply


def test_streaming_stt_finishes_from_the_speculative_decode(isolated_home, body_factory):
    ctx = make_ctx(isolated_home, executor=InlineExecutor())
    body = body_factory(ctx)
    body.hello()
    voice_turn(body, words=6)
    done = body.until_turn_done(1)[-1]
    assert done.text
    body.send(TurnMetrics(turn=1, stages={}))
    record = last_latency_record(isolated_home)
    assert record["stt"]["speculative_used"] == 1
    assert record["stt"]["final"] == 0


def test_turn_with_nothing_said_is_dropped(isolated_home, body_factory, bus):
    brain = ScriptBrain(bus)
    body = body_factory(make_ctx(isolated_home, brain=brain, bus=bus))
    body.hello()
    body.send(TurnStart(turn=1))
    body.send(TurnEnd(turn=1, reason="nothing"))
    assert drain(body.conn, 0.3) == []
    assert brain.heard == []


def test_empty_transcript_ends_the_turn_without_the_brain(isolated_home, body_factory, bus):
    brain = ScriptBrain(bus)
    body = body_factory(make_ctx(isolated_home, brain=brain, bus=bus))
    body.hello()
    voice_turn(body, words=0)
    frames = body.until_turn_done(1)
    assert [describe(f) for f in frames] == ["state", "transcript('')", "turn_done('')"]
    assert body.conn.recv(timeout=2) == StateChange(state="idle", turn=1)
    assert brain.heard == []


def test_text_input_turn(isolated_home, body_factory, bus):
    brain = ScriptBrain(bus, [("say", "Four.")])
    body = body_factory(make_ctx(isolated_home, brain=brain, bus=bus))
    body.hello()
    body.send(TextInput(turn=3, text="What is two plus two?"))
    frames = control(body.until_turn_done(3))
    assert [describe(f) for f in frames] == ["state", "sentence('Four.')", "turn_done('Four.')"]
    assert brain.heard == ["What is two plus two?"]


# ---------------------------------------------------------------------- filler
def test_filler_when_a_slow_tool_runs_before_anything_was_said(isolated_home, body_factory, bus):
    brain = ScriptBrain(bus, [("emotion", "curious"), ("tools", ["get_calendar_events"]),
                              ("sleep", 0.2), ("say", "You have the dentist at ten.")])
    tts = FakeTTS()
    body = body_factory(make_ctx(isolated_home, brain=brain, bus=bus, tts=tts))
    body.hello()
    body.send(TextInput(turn=1, text="What's on today?"))
    frames = body.until_turn_done(1)
    sentences = [f for f in frames if isinstance(f, Sentence)]
    assert [(s.index, s.text, s.filler) for s in sentences] == [
        (0, "Let me check.", True),
        (1, "You have the dentist at ten.", False),
    ]
    first_audio = next(f for f in frames if isinstance(f, AudioFrame))
    assert first_audio.sentence == 0 and first_audio.pcm[:2] == b"\x02\x00"  # pre-synthesized
    assert tts.said == ["You have the dentist at ten."]
    assert frames[-1].text == "You have the dentist at ten."  # the filler is not the reply


@pytest.mark.parametrize(
    "script",
    [
        [("tools", ["get_current_time"]), ("say", "It's ten past two.")],  # fast tool
        [("say", "Let me look."), ("tools", ["get_calendar_events"]), ("say", "Nothing.")],
    ],
    ids=["fast-tool", "already-talking"],
)
def test_no_filler(isolated_home, body_factory, bus, script):
    body = body_factory(make_ctx(isolated_home, brain=ScriptBrain(bus, script), bus=bus))
    body.hello()
    body.send(TextInput(turn=1, text="?"))
    frames = body.until_turn_done(1)
    assert not any(isinstance(f, Sentence) and f.filler for f in frames)


# ---------------------------------------------------------------------- barge-in
def test_interrupt_mid_reply_cancels_fast_and_keeps_what_was_heard(
    isolated_home, body_factory, bus
):
    brain = ScriptBrain(bus, [("say", "Tomorrow you have the dentist."), ("sleep", 5.0),
                              ("say", "And lunch with Sam.")])
    body = body_factory(make_ctx(isolated_home, brain=brain, bus=bus))
    body.hello()
    body.send(TextInput(turn=1, text="What's on tomorrow?"))
    body.recv_until(lambda f: isinstance(f, AudioFrame) and f.end)
    sent_at = time.monotonic()
    body.send(Interrupt(turn=1))
    body.send(PlaybackDone(turn=1, interrupted=True, heard="Tomorrow you have the dentist."))
    deadline = time.monotonic() + 3
    while brain.returned_at is None and time.monotonic() < deadline:
        time.sleep(0.005)
    assert brain.returned_at is not None
    assert brain.returned_at - sent_at < 0.1  # generation stopped within tens of ms
    # Nothing more for turn 1, then the next turn works normally.
    brain.script = [("say", "Sure.")]
    body.send(TurnStart(turn=2, trigger="barge_in"))
    body.speak(2, 2)
    body.send(TurnEnd(turn=2, trailing_ms=450))
    frames = body.until_turn_done(2)
    assert all(getattr(f, "turn", 2) == 2 for f in frames), [describe(f) for f in frames]
    assert brain.amended == ["Tomorrow you have the dentist."]
    assert brain.heard[-1] == "w0 w1"


def test_interrupt_stops_tts_audio_and_strips_filler_from_heard(isolated_home, body_factory, bus):
    brain = ScriptBrain(bus, [("tools", ["list_unread_emails"]), ("sleep", 0.1),
                              ("say", "You have three emails."), ("say", "One from Sam.")])
    tts = FakeTTS(seconds=3.0, delay=0.03, chunks=30)  # streamed 100 ms at a time
    body = body_factory(make_ctx(isolated_home, brain=brain, bus=bus, tts=tts))
    body.hello()
    body.send(TextInput(turn=1, text="Any email?"))
    body.recv_until(lambda f: isinstance(f, AudioFrame) and f.sentence == 1)
    body.send(Interrupt(turn=1))
    body.send(PlaybackDone(turn=1, interrupted=True, heard="Let me check. You have three"))
    rest = drain(body.conn, 0.5)
    assert not any(isinstance(f, TurnDone) for f in rest)
    late_audio = [f for f in rest if isinstance(f, AudioFrame)]
    assert len(late_audio) < 5  # stopped within a frame or two, mid-sentence
    assert not any(f.end for f in late_audio)
    assert not any(isinstance(f, Sentence) for f in rest)  # "One from Sam." never sent
    body.send(TextInput(turn=2, text="ok"))
    body.until_turn_done(2)
    assert brain.amended == ["You have three"]


def test_new_turn_supersedes_a_running_one(isolated_home, body_factory, bus):
    brain = ScriptBrain(bus, [("sleep", 5.0), ("say", "Too late.")])
    body = body_factory(make_ctx(isolated_home, brain=brain, bus=bus))
    body.hello()
    body.send(TextInput(turn=1, text="first"))
    body.recv_until(lambda f: isinstance(f, StateChange))
    brain.script = [("say", "Second answer.")]
    body.send(TextInput(turn=2, text="second"))
    frames = body.until_turn_done(2, timeout=3)
    assert not any(isinstance(f, TurnDone) and f.turn == 1 for f in frames)
    assert frames[-1].text == "Second answer."


# ---------------------------------------------------------------------- announcements
def test_announcement(isolated_home, body_factory, bus):
    body = body_factory(make_ctx(isolated_home, bus=bus))
    body.hello()
    bus.announce("Time to call Mum.")
    frames = body.recv_until(lambda f: isinstance(f, TurnDone))
    names = [describe(f) for f in frames]
    assert names == ["announce('Time to call Mum.')", "sentence('Time to call Mum.')",
                     "audio(s0)", "audio(s0)", "audio(s0)", "audio(s0 END)",
                     "turn_done('Time to call Mum.')"], names
    announce = frames[0]
    assert isinstance(announce, Announce) and is_brain_turn(announce.turn)
    assert all(f.turn == announce.turn for f in frames)


def test_announcement_waits_for_the_reply_to_be_played(isolated_home, body_factory, bus):
    brain = ScriptBrain(bus, [("say", "Hello.")])
    body = body_factory(make_ctx(isolated_home, brain=brain, bus=bus))
    body.hello()
    body.send(TurnStart(turn=1))
    bus.announce("Reminder: stretch.")  # Adam is talking: wait
    body.speak(1, 2)
    body.send(TurnEnd(turn=1, trailing_ms=450))
    body.until_turn_done(1)
    assert not any(isinstance(f, Announce) for f in drain(body.conn, 0.3))  # still playing
    body.send(PlaybackDone(turn=1, heard="Hello."))
    frames = body.recv_until(lambda f: isinstance(f, TurnDone))
    assert isinstance(frames[0], Announce)


def test_transcript_partials_are_marked_not_final(isolated_home, body_factory):
    ctx = make_ctx(isolated_home, executor=InlineExecutor(),
                   stt={"partial_min_s": 0.5, "partial_every_s": 0.3})
    body = body_factory(ctx)
    body.hello()
    voice_turn(body, words=12)
    frames = body.until_turn_done(1)
    transcripts = [f for f in frames if isinstance(f, Transcript)]
    assert any(not t.final for t in transcripts)
    assert transcripts[-1].final and transcripts[-1].text.startswith("w0 w1")


# ---------------------------------------------------------------------- websocket
def test_websocket_localhost_turn(isolated_home, bus):
    from websockets.sync.client import connect

    from roco.protocol import Hello, WebSocketConnection
    from roco.server.ws import make_server

    ctx = make_ctx(isolated_home, brain=ScriptBrain(bus, [("say", "Hi over the wire.")]), bus=bus)
    stop = threading.Event()
    server = make_server(ctx, "127.0.0.1", 0, stop)
    port = server.socket.getsockname()[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        with connect(f"ws://127.0.0.1:{port}", compression=None) as ws:
            conn = WebSocketConnection(ws)
            conn.send(Hello(protocol=1, body_id="ws-body", token=TOKEN))
            assert isinstance(conn.recv(timeout=5), Welcome)
            conn.send(TextInput(turn=1, text="hello"))
            frames = []
            while not (frames and isinstance(frames[-1], TurnDone)):
                frame = conn.recv(timeout=5)
                assert frame is not None
                frames.append(frame)
            assert frames[-1].text == "Hi over the wire."
            assert any(isinstance(f, AudioFrame) and f.end for f in frames)
            conn.close()
    finally:
        stop.set()
        server.shutdown()
        thread.join(timeout=5)


# ---------------------------------------------------------------------- personality settings
class FakePersonality:
    def __init__(self):
        self.values = {"personality": {"roast": 50, "preset": "default"}}
        self.callbacks = []

    def apply_settings(self, values):
        if values["personality"].get("roast", 0) > 100:
            raise ValueError("roast must be 0-100")
        self.values = {"personality": {**self.values["personality"], **values["personality"]}}
        return self.values

    def snapshot(self):
        return {"personality": dict(self.values["personality"])}

    def on_change(self, cb):
        self.callbacks.append(cb)
        return lambda: self.callbacks.remove(cb)

    def voice_command(self, **change):
        self.apply_settings({"personality": change})
        for cb in list(self.callbacks):
            cb(self.snapshot())


def test_personality_settings_go_to_the_brain(isolated_home, body_factory, bus):
    brain = ScriptBrain(bus)
    brain.personality = FakePersonality()
    body = body_factory(make_ctx(isolated_home, brain=brain, bus=bus))
    welcome = body.hello()
    assert welcome.settings == {"personality": {"roast": 50, "preset": "default"}}
    body.send(Settings(values={"personality": {"roast": 80}, "muted": True}))
    reply = body.conn.recv(timeout=5)
    assert reply.values == {"personality": {"roast": 80, "preset": "default"}, "muted": True}
    body.send(Settings(values={"personality": {"roast": 500}}))
    error = body.conn.recv(timeout=5)
    assert isinstance(error, Error) and error.code == "bad_message" and not error.fatal
    brain.personality.voice_command(preset="unhinged")  # "ROCO, go unhinged"
    pushed = body.conn.recv(timeout=5)
    assert pushed.values["personality"]["preset"] == "unhinged"
    body.close()
    assert brain.personality.callbacks == []  # unsubscribed on disconnect


def test_missing_speech_models_raise_brain_unavailable(isolated_home, bus):
    from roco.config import load_config
    from roco.server import BrainUnavailable, create_brain_context

    def broken_tts(text):
        raise FileNotFoundError("Piper voice not found: x.onnx")
        yield  # pragma: no cover

    cfg = load_config()
    with pytest.raises(BrainUnavailable, match="roco download-models"):
        create_brain_context(cfg, bus=bus, brain=ScriptBrain(bus), stt=object(), tts=broken_tts,
                             token="t")
