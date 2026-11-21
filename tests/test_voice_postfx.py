"""TTS post-processing, the Kokoro wrapper and engine selection in the voice factory."""

from __future__ import annotations

import sys
import types

import pytest

from roco.voice.postfx import PostFxSettings


def test_postfx_settings_from_config():
    assert PostFxSettings.from_config(None) is None
    assert PostFxSettings.from_config(False) is None
    assert PostFxSettings.from_config(True) == PostFxSettings()
    assert PostFxSettings.from_config({"enabled": False}) is None
    custom = PostFxSettings.from_config({"reverb": 0.0, "compress": False, "junk": 1})
    assert custom.reverb == 0.0 and not custom.compress and custom.eq


def _tone(np, rate=22050, seconds=1.0, freq=440.0, amp=0.5):
    t = np.arange(int(rate * seconds)) / rate
    return (np.sin(2 * np.pi * freq * t) * amp * 32767).astype(np.int16).tobytes()


def test_postfx_keeps_audio_sane():
    np = pytest.importorskip("numpy")
    from roco.voice.postfx import PostFx

    fx = PostFx()
    pcm = _tone(np)
    out = np.frombuffer(fx.process(pcm, 22050), dtype=np.int16)
    n_in = len(pcm) // 2
    assert n_in <= len(out) <= n_in + int(0.3 * 22050)  # at most the room tail added
    assert 0.1 < np.abs(out).max() / 32767 <= 1.0
    assert fx.process(b"", 22050) == b""


def test_eq_cuts_rumble_and_keeps_speech_band():
    np = pytest.importorskip("numpy")
    from roco.voice.postfx import eq_curve_db

    gains = eq_curve_db(np.array([30.0, 1000.0, 3000.0, 10000.0]))
    assert gains[0] < -10  # high-pass
    assert abs(gains[1]) < 1.5  # speech fundamentals untouched
    assert gains[2] > 1  # presence lift
    assert gains[3] < -1  # gentle top cut


def test_compressor_evens_out_loud_and_quiet():
    np = pytest.importorskip("numpy")
    from roco.voice.postfx import PostFx

    fx = PostFx(PostFxSettings(eq=False, reverb=0.0))

    def rms(pcm):
        x = np.frombuffer(pcm, dtype=np.int16) / 32768
        return float(np.sqrt(np.mean(x**2)))

    loud, quiet = _tone(np, amp=0.8), _tone(np, amp=0.1)
    ratio_in = rms(loud) / rms(quiet)
    ratio_out = rms(fx.process(loud, 22050)) / rms(fx.process(quiet, 22050))
    assert ratio_out < ratio_in * 0.8


def test_processed_tts_joins_sentence_and_passes_load():
    np = pytest.importorskip("numpy")
    from roco.voice.postfx import PostFx, ProcessedTTS

    class Inner:
        loaded = False

        def load(self):
            self.loaded = True

        def __call__(self, text):
            yield _tone(np, seconds=0.2), 22050
            yield _tone(np, seconds=0.2), 22050

    inner = Inner()
    tts = ProcessedTTS(inner, PostFx(PostFxSettings(reverb=0.0)))
    tts.load()
    chunks = list(tts("Hello."))
    assert inner.loaded and len(chunks) == 1 and chunks[0][1] == 22050
    assert len(chunks[0][0]) == 2 * 2 * int(0.2 * 22050)


def test_kokoro_tts_converts_float_audio(tmp_path, monkeypatch):
    np = pytest.importorskip("numpy")
    from roco.voice.kokoro_tts import KokoroTTS

    calls = {}

    class FakeKokoro:
        @classmethod
        def from_session(cls, session, voices_path):
            calls["voices"] = voices_path
            return cls()

        def create(self, text, voice, speed, lang):
            calls.update(text=text, voice=voice, speed=speed, lang=lang)
            return np.array([0.0, 0.5, -1.5], dtype=np.float32), 24000

    class FakeOrt:
        class SessionOptions:
            intra_op_num_threads = 0

        @staticmethod
        def InferenceSession(path, options, providers):  # noqa: N802 - onnxruntime's name
            calls["threads"] = options.intra_op_num_threads
            return object()

    monkeypatch.setitem(sys.modules, "kokoro_onnx", types.SimpleNamespace(Kokoro=FakeKokoro))
    monkeypatch.setitem(sys.modules, "onnxruntime", FakeOrt)
    model, voices = tmp_path / "m.onnx", tmp_path / "v.bin"
    model.write_bytes(b"x")
    voices.write_bytes(b"x")
    tts = KokoroTTS(model, voices, voice="bf_isabella", threads=4)
    [(pcm, rate)] = list(tts("Hi there."))
    assert rate == 24000
    assert list(np.frombuffer(pcm, dtype=np.int16)) == [0, 16383, -32767]
    assert calls["voice"] == "bf_isabella" and calls["lang"] == "en-gb" and calls["threads"] == 4


def test_kokoro_missing_files_say_how_to_fix(tmp_path):
    from roco.voice.kokoro_tts import KokoroTTS

    with pytest.raises(FileNotFoundError, match="download-models"):
        KokoroTTS(tmp_path / "nope.onnx", tmp_path / "nope.bin").load()


def test_factory_selects_engine_and_postfx(config, isolated_home):
    from roco.voice import factory, models
    from roco.voice.kokoro_tts import KokoroTTS
    from roco.voice.postfx import ProcessedTTS
    from roco.voice.tts import PiperTTS

    config["voice"]["tts"]["postfx"] = False
    assert isinstance(factory.create_tts(config), PiperTTS)
    config["voice"]["tts"]["postfx"] = True  # the default
    tts = factory.create_tts(config)
    assert isinstance(tts, ProcessedTTS) and isinstance(tts.inner, PiperTTS)
    config["voice"]["tts"].update(engine="kokoro", postfx=False)
    config["voice"]["tts"]["kokoro"]["voice"] = "bf_isabella"
    tts = factory.create_tts(config)
    assert isinstance(tts, KokoroTTS) and tts.voice == "bf_isabella"
    assert models.tts_model_files(config) == [
        isolated_home / "models" / "kokoro" / "kokoro-v1.0.onnx",
        isolated_home / "models" / "kokoro" / "voices-v1.0.bin",
    ]
    config["voice"]["tts"]["engine"] = "espeak"
    with pytest.raises(factory.VoiceUnavailable, match="piper or kokoro"):
        factory.create_tts(config)


def test_kokoro_download_skipped_for_piper(config):
    from roco.voice.models import download_kokoro

    assert download_kokoro(config).status == "present"
