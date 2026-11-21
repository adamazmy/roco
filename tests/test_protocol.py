"""Link protocol v1: codec round trips, versioning, transports and auth."""

from __future__ import annotations

import contextlib
import json
import threading

import pytest

from roco.protocol import (
    PROTOCOL_VERSION,
    AudioFlag,
    AudioFrame,
    AudioKind,
    ConnectionClosed,
    Hello,
    Interrupt,
    ProtocolError,
    Sentence,
    TurnDone,
    TurnStart,
    Unknown,
    WebSocketConnection,
    brain_turn_id,
    decode,
    decode_audio,
    encode,
    encode_audio,
    is_brain_turn,
    load_or_create_token,
    memory_pipe,
    token_matches,
)
from roco.protocol.messages import message_types

# A sample instance of every message type, so new types can't skip the round trip.
REQUIRED = {
    "ping": {"id": 1, "ts": 12.5},
    "pong": {"id": 1, "ts": 12.5},
    "error": {"code": "auth", "message": "bad token", "fatal": True},
    "settings": {"values": {"personality": {"roast": 80}}},
    "hello": {"protocol": 1, "body_id": "pi", "token": "t", "caps": {"face": True}},
    "turn_start": {"turn": 3, "trigger": "barge_in"},
    "turn_end": {"turn": 3, "reason": "silence", "trailing_ms": 450},
    "text_input": {"turn": 4, "text": "what's the time?"},
    "interrupt": {"turn": 2},
    "playback_done": {"turn": 2, "interrupted": True, "heard": "Tomorrow you have"},
    "turn_metrics": {"turn": 2, "stages": {"first_audio_played": 830}},
    "welcome": {"protocol": 1, "brain": "ollama:qwen2.5:7b", "tts_rate": 22050},
    "state": {"state": "thinking", "turn": 1},
    "emotion": {"emotion": "happy", "turn": 1},
    "transcript": {"turn": 1, "text": "hello"},
    "sentence": {"turn": 1, "index": 0, "text": "Let me check.", "filler": True},
    "announce": {"turn": brain_turn_id(1), "text": "Your reminder: tea."},
    "turn_done": {"turn": 1, "text": "Hi.", "brain": "ollama", "stages": {"stt_done": 300}},
}


def test_every_message_type_has_a_sample():
    assert set(message_types()) == set(REQUIRED)


@pytest.mark.parametrize("mtype", sorted(REQUIRED))
def test_message_round_trip(mtype):
    msg = message_types()[mtype](**REQUIRED[mtype])
    wire = encode(msg)
    data = json.loads(wire)
    assert data["v"] == PROTOCOL_VERSION and data["type"] == mtype
    assert decode(wire) == msg


def test_unknown_fields_are_ignored_and_unknown_types_kept():
    wire = json.dumps({"v": 1, "type": "turn_start", "turn": 5, "future_field": [1]})
    assert decode(wire) == TurnStart(turn=5)
    other = decode(json.dumps({"v": 1, "type": "hologram_poke", "x": 3}))
    assert isinstance(other, Unknown)
    assert other.type == "hologram_poke" and other.data == {"x": 3}


@pytest.mark.parametrize(
    "wire",
    [
        "not json",
        "[1, 2]",
        json.dumps({"type": "ping", "id": 1, "ts": 0}),  # no version
        json.dumps({"v": 2, "type": "ping", "id": 1, "ts": 0}),  # future version
        json.dumps({"v": 1, "id": 1}),  # no type
        json.dumps({"v": 1, "type": "ping"}),  # missing required fields
    ],
)
def test_invalid_control_frames_raise(wire):
    with pytest.raises(ProtocolError):
        decode(wire)


def test_brain_turn_ids_never_collide_with_body_turns():
    assert is_brain_turn(brain_turn_id(1))
    assert not is_brain_turn(1)
    assert brain_turn_id(1) != 1


# --------------------------------------------------------------------------- audio
def test_audio_round_trip():
    pcm = (b"\x01\x00\xff\x7f") * 640  # 1280 samples = 80 ms at 16 kHz
    frame = AudioFrame(AudioKind.TTS, turn=brain_turn_id(7), seq=42, rate=16000, pcm=pcm,
                       sentence=3, flags=AudioFlag.END)
    data = encode_audio(frame)
    assert len(data) == 18 + len(pcm)
    back = decode_audio(data)
    assert back == frame
    assert back.end and back.duration_ms == pytest.approx(80.0)


@pytest.mark.parametrize(
    "data",
    [
        b"RA",  # too short
        b"XX" + bytes(16),  # bad magic
        b"RA\x02" + bytes(15),  # future version
        b"RA\x01\x09" + bytes(14),  # unknown kind
        encode_audio(AudioFrame(AudioKind.MIC, 1, 0, 16000, b"\x00\x00")) + b"\x00",  # odd
    ],
)
def test_invalid_audio_frames_raise(data):
    with pytest.raises(ProtocolError):
        decode_audio(data)


# --------------------------------------------------------------------------- transports
def test_memory_pipe_carries_messages_and_audio_both_ways():
    body, brain = memory_pipe()
    frame = AudioFrame(AudioKind.MIC, turn=1, seq=0, rate=16000, pcm=b"\x00\x01" * 10)
    body.send(TurnStart(turn=1))
    body.send(frame)
    assert brain.recv(timeout=1) == TurnStart(turn=1)
    assert brain.recv(timeout=1) == frame
    brain.send(Sentence(turn=1, index=0, text="Hi."))
    assert body.recv(timeout=1) == Sentence(turn=1, index=0, text="Hi.")
    assert body.recv(timeout=0.01) is None


def test_memory_pipe_close_wakes_both_sides():
    body, brain = memory_pipe()
    errors = []

    def wait():
        try:
            brain.recv()
        except ConnectionClosed:
            errors.append("closed")

    t = threading.Thread(target=wait)
    t.start()
    body.close()
    t.join(timeout=2)
    assert errors == ["closed"]
    with pytest.raises(ConnectionClosed):
        body.send(Interrupt(turn=1))
    body.close()  # twice is fine


def test_websocket_connection_over_localhost():
    from websockets.sync.client import connect
    from websockets.sync.server import serve

    received = []

    def handler(ws):
        conn = WebSocketConnection(ws)
        hello = conn.recv(timeout=5)
        received.append(hello)
        conn.send(TurnDone(turn=1, text="Hi.", brain="test"))
        conn.send(AudioFrame(AudioKind.TTS, 1, 0, 22050, b"\x00\x00" * 4, flags=AudioFlag.END))
        with contextlib.suppress(ConnectionClosed):
            conn.recv(timeout=5)

    with serve(handler, "127.0.0.1", 0) as server:
        port = server.socket.getsockname()[1]
        t = threading.Thread(target=server.serve_forever, daemon=True)
        t.start()
        try:
            with connect(f"ws://127.0.0.1:{port}") as ws:
                conn = WebSocketConnection(ws)
                conn.send(Hello(protocol=1, body_id="test", token="t"))
                assert conn.recv(timeout=5) == TurnDone(turn=1, text="Hi.", brain="test")
                audio = conn.recv(timeout=5)
                assert isinstance(audio, AudioFrame) and audio.end
                assert conn.recv(timeout=0.05) is None
                conn.close()
                assert conn.closed
                with pytest.raises(ConnectionClosed):
                    conn.send(Interrupt(turn=1))
        finally:
            server.shutdown()
            t.join(timeout=5)
    assert received == [Hello(protocol=1, body_id="test", token="t")]


# --------------------------------------------------------------------------- auth
def test_token_is_created_once_and_reused(tmp_path):
    path = tmp_path / "sub" / "link_token"
    token = load_or_create_token(path)
    assert len(token) >= 32
    assert load_or_create_token(path) == token
    assert token_matches(token, token)
    assert not token_matches(token, "wrong")
    assert not token_matches(token, None)
    assert not token_matches("", "")
