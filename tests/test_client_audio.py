"""VAD endpointer, AEC injection/engines, echo reference and the player."""

from __future__ import annotations

import sys

import pytest

from client_fakes import QUIET, SPEECH, frame
from roco.client.playback import Player
from roco.client.vad import EndpointSettings, EnergyVad, VadEndpointer, create_vad
from roco.voice import aec as aec_mod
from roco.voice.aec import AecFrameSource, AecUnavailable, NullAec, create_aec
from roco.voice.endpoint import Endpoint

# --------------------------------------------------------------------------- endpointer


def feed(ep, probs, ms=32.0):
    return [ep.feed([(p, ms)]) for p in probs]


def test_endpointer_starts_after_start_ms_and_ends_after_trailing_silence():
    ep = VadEndpointer(EndpointSettings(start_ms=96, trailing_ms=450))
    events = feed(ep, [0.9, 0.9])
    assert events == [Endpoint.CONTINUE, Endpoint.CONTINUE]
    assert feed(ep, [0.9]) == [Endpoint.SPEECH_START]
    assert ep.onset_ms == 0
    events = feed(ep, [0.0] * 14)  # 448 ms
    assert set(events) == {Endpoint.CONTINUE}
    assert feed(ep, [0.0]) == [Endpoint.SILENCE]
    assert ep.trailing_ms == 480


def test_endpointer_hysteresis_keeps_soft_speech():
    ep = VadEndpointer(EndpointSettings(start_ms=32, trailing_ms=96))
    feed(ep, [0.9])
    events = feed(ep, [0.4] * 10)  # above neg_threshold 0.35 while in speech
    assert Endpoint.SILENCE not in events
    assert ep.silence_ms == 0


def test_endpointer_no_speech_and_max_length():
    ep = VadEndpointer(EndpointSettings(start_timeout_s=0.32))
    assert feed(ep, [0.0] * 10)[-1] is Endpoint.NO_SPEECH
    ep = VadEndpointer(EndpointSettings(start_ms=32, max_s=0.64))
    assert feed(ep, [0.9] * 20)[-1] is Endpoint.MAX_LENGTH


def test_speech_start_mid_frame_still_counts_the_rest_of_the_frame():
    ep = VadEndpointer(EndpointSettings(start_ms=32, trailing_ms=64))
    assert ep.feed([(0.9, 32), (0.0, 32), (0.0, 32)]) is Endpoint.SPEECH_START
    assert ep.silence_ms == 64
    assert ep.feed([]) is Endpoint.SILENCE


def test_energy_vad_and_fallback_when_silero_model_missing(tmp_path):
    vad = create_vad("silero", tmp_path / "missing.onnx")
    assert isinstance(vad, EnergyVad)
    assert vad.scores(frame(SPEECH)) == [(1.0, 80.0)]
    assert vad.scores(frame(QUIET)) == [(0.0, 80.0)]
    with pytest.raises(ValueError):
        create_vad("magic", None)


def test_silero_vad_scores_32ms_windows():
    np = pytest.importorskip("numpy")
    pytest.importorskip("onnxruntime")
    from roco.client.vad import SileroVad
    from roco.voice.models import silero_vad_path

    path = silero_vad_path()
    if not path.is_file():
        pytest.skip("Silero model not downloaded")
    vad = SileroVad(path)
    out = vad.scores(np.zeros(1280, dtype=np.int16))
    assert len(out) == 2 and all(p < 0.2 for p, _ in out)
    assert len(vad.scores(np.zeros(1280, dtype=np.int16))) == 3  # 256 + 1280 carried


# --------------------------------------------------------------------------- AEC injection


class FakeAec:
    name = "fake"

    def __init__(self, fail=False):
        self.calls = []
        self.fail = fail
        self.closed = False

    def process(self, near, far):
        if self.fail:
            raise RuntimeError("boom")
        self.calls.append((near, far))
        return "clean"

    def close(self):
        self.closed = True


class FakeReference:
    def take(self, n):
        return f"far{n}"


class FakeMonitor:
    def __init__(self):
        self.updates = []

    def update(self, raw, clean, far):
        self.updates.append((raw, clean, far))


class FakeMic:
    def __init__(self, frames):
        self.frames = list(frames)
        self.closed = False

    def read(self, timeout):
        return self.frames.pop(0) if self.frames else None

    def flush(self):
        pass

    def close(self):
        self.closed = True


def test_aec_source_cleans_every_frame_with_the_reference():
    aec, mon = FakeAec(), FakeMonitor()
    src = AecFrameSource(FakeMic(["f1"]), aec, FakeReference(), mon)
    assert src.read(0.1) == "clean"
    assert aec.calls == [("f1", "far2")]
    assert mon.updates == [("f1", "clean", "far2")]
    assert src.read(0.1) is None
    src.close()
    assert aec.closed and src.source.closed


def test_aec_failure_passes_the_mic_through():
    src = AecFrameSource(FakeMic(["f1"]), FakeAec(fail=True), FakeReference(), FakeMonitor())
    assert src.read(0.1) == "f1"


def test_null_aec_and_engine_selection(monkeypatch):
    assert isinstance(create_aec("none"), NullAec)
    assert create_aec("off").process("x", "y") == "x"
    with pytest.raises(ValueError):
        create_aec("magic")
    monkeypatch.setitem(sys.modules, "livekit", None)  # import fails
    monkeypatch.setitem(sys.modules, "livekit.rtc", None)
    monkeypatch.setitem(sys.modules, "pyaec", None)
    with pytest.raises(AecUnavailable):
        create_aec("webrtc")
    assert isinstance(create_aec("auto"), NullAec)


# --------------------------------------------------------------------------- numpy parts


def test_stream_resampler_has_no_seams():
    np = pytest.importorskip("numpy")
    x = (np.sin(np.arange(22050) * 2 * np.pi * 200 / 22050) * 10000).astype(np.int16)
    whole = aec_mod.StreamResampler(22050, 16000)(x)
    r = aec_mod.StreamResampler(22050, 16000)
    parts = np.concatenate([r(x[i : i + 441]) for i in range(0, len(x), 441)])
    assert abs(len(parts) - 16000) <= 1
    n = min(len(whole), len(parts))
    assert np.max(np.abs(whole[:n].astype(int) - parts[:n])) <= 1


def test_echo_reference_pads_and_caps():
    np = pytest.importorskip("numpy")
    ref = aec_mod.EchoReference(max_s=0.1)
    ref.push(np.ones(320, dtype=np.int16).tobytes(), 16000)
    got = ref.take(480)
    assert list(got[:320]) == [1] * 320 and not got[320:].any()
    assert ref.underruns == 1
    ref.push(np.ones(4000, dtype=np.int16), 16000)
    assert ref.buffered_ms <= 100


def test_echo_monitor_median_erle():
    np = pytest.importorskip("numpy")
    mon = aec_mod.EchoMonitor(far_rms_min=100)
    loud = np.full(1280, 1000, dtype=np.int16)
    quiet = np.full(1280, 10, dtype=np.int16)
    mon.update(loud, quiet, np.zeros(1280, dtype=np.int16))  # far silent: ignored
    assert mon.erle_db is None
    for _ in range(5):
        mon.update(loud, quiet, loud)
    assert mon.erle_db == pytest.approx(40, abs=0.1)


@pytest.mark.parametrize("engine", ["webrtc", "speex"])
def test_real_engine_removes_a_delayed_echo(engine):
    np = pytest.importorskip("numpy")
    try:
        aec = create_aec(engine)
    except AecUnavailable:
        pytest.skip(f"{engine} AEC not installed")
    rng = np.random.default_rng(0)
    far = (rng.standard_normal(16000 * 4) * 3000).astype(np.int16)
    echo = np.concatenate([np.zeros(480), far[:-480] * 0.5]).astype(np.int16)  # 30 ms
    out = np.concatenate([aec.process(echo[i : i + 1280], far[i : i + 1280])
                          for i in range(0, len(far), 1280)])
    tail = slice(16000 * 2, None)
    erle = 10 * np.log10(np.mean(echo[tail].astype(float) ** 2)
                         / max(np.mean(out[tail].astype(float) ** 2), 1))
    assert erle > 15


# --------------------------------------------------------------------------- player


class Events:
    def __init__(self):
        self.log = []

    def on_audio_started(self, turn, t):
        self.log.append(("started", turn))

    def on_turn_drained(self, turn):
        self.log.append(("drained", turn))

    def on_mouth(self, a):
        self.log.append(("mouth", round(a, 2)))


def test_player_plays_in_order_and_tracks_heard_sentences():
    ev = Events()
    p = Player(rate=16000, listener=ev)
    p.set_text(1, 0, "Hi.")
    p.set_text(1, 1, "Bye.")
    assert p.pull(4) == bytes(8)  # idle: silence
    p.feed(1, 0, b"\x01\x00" * 3, 16000, end=True)
    p.feed(1, 1, b"\x02\x00" * 3, 16000, end=True)
    p.finish(1)
    assert p.pull(4) == b"\x01\x00" * 3 + b"\x02\x00"
    assert p.heard(1) == "Hi."
    assert p.pull(4) == b"\x02\x00" * 2 + bytes(4)
    assert p.heard(1) == "Hi. Bye."
    p.dispatch_pending()
    names = [e for e in ev.log if e[0] != "mouth"]
    assert names == [("started", 1), ("drained", 1)]
    assert ev.log[-1][0] == "mouth"
    assert not p.active


def test_player_cut_drops_the_turn_and_later_frames():
    p = Player(rate=16000)
    p.set_text(1, 0, "One,")
    p.feed(1, 0, b"\x01\x00" * 2, 16000, end=True)
    p.feed(1, 1, b"\x01\x00" * 100, 16000, end=True)
    p.pull(2)
    assert p.cut(1) == "One,"
    assert not p.playing
    assert p.feed(1, 2, b"\x01\x00", 16000) is False
    assert p.pull(2) == bytes(4)


def test_empty_reply_drains_immediately():
    ev = Events()
    p = Player(rate=16000, listener=ev)
    p.finish(5)
    p.dispatch_pending()
    assert ("drained", 5) in ev.log


def test_player_spoken_context_is_around_the_playhead():
    p = Player(rate=16000)
    for i, text in enumerate(["One.", "Two.", "Three.", "Four.", "Five."]):
        p.set_text(1, i, text)
        p.feed(1, i, bytes([1, 0]) * 2, 16000, end=True)
    p.pull(6)  # sentences 0-2 heard, 3 next
    assert p.spoken_context(1) == ["Two.", "Three.", "Four.", "Five."]
    assert p.spoken_context(9) == []


def test_echo_monitor_gate_learns_the_residual_and_passes_louder_speech():
    np = pytest.importorskip("numpy")
    mon = aec_mod.EchoMonitor(far_rms_min=100)
    far = np.full(1280, 3000, dtype=np.int16)
    residual = np.full(1280, 100, dtype=np.int16)
    assert mon.near_end(6.0)  # nothing playing: no echo, anything may be Adam
    mon.update(residual * 30, residual, far)
    assert not mon.near_end(6.0)  # playback just started: not trusted yet
    for _ in range(10):
        mon.update(residual * 30, residual, far)
    assert mon.near_end_db() == pytest.approx(0, abs=0.1)
    assert not mon.near_end(6.0)
    adam = np.full(1280, 400, dtype=np.int16)  # +12 dB over the residual
    mon.update(adam * 10, adam, far)
    assert mon.near_end(6.0)
    assert mon.residual_floor == pytest.approx(100.0**2)  # Adam's frame was not learnt
    silent = np.zeros(1280, dtype=np.int16)
    for _ in range(5):
        mon.update(residual, residual, silent)  # playback stopped: hangover, then free
    assert mon.near_end_db() is None
