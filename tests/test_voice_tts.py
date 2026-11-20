"""Piper API adapter against fake voice objects for every piper-tts generation."""

from __future__ import annotations

import wave
from dataclasses import dataclass
from types import SimpleNamespace

from roco.voice.tts import iter_piper_audio, piper_api

PCM = b"\x10\x00\x20\x00" * 50


class OldStreamVoice:
    """piper-tts <= 1.2: synthesize_stream_raw + synthesize(text, wav_file)."""

    def __init__(self):
        self.config = SimpleNamespace(sample_rate=22050)
        self.kwargs = None

    def synthesize_stream_raw(self, text, speaker_id=None, length_scale=None, noise_scale=None,
                              noise_w=None, sentence_silence=0.0):
        self.kwargs = {"length_scale": length_scale}
        yield PCM
        yield b""  # empty chunks are skipped
        yield PCM

    def synthesize(self, text, wav_file, length_scale=None):  # pragma: no cover - not chosen
        raise AssertionError("stream_raw should be preferred")


class OldWavOnlyVoice:
    """An old build with only synthesize(text, wav_file)."""

    def __init__(self):
        self.length_scale = None

    def synthesize(self, text, wav_file: wave.Wave_write, length_scale=None):
        self.length_scale = length_scale
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(16000)
        wav_file.writeframes(PCM)


@dataclass
class FakeChunk:
    sample_rate: int
    audio_int16_bytes: bytes


class NewVoice:
    """piper-tts >= 1.3: synthesize(text, syn_config) yields AudioChunk."""

    def __init__(self):
        self.syn_config = "unset"

    def synthesize(self, text, syn_config=None, include_alignments=False):
        self.syn_config = syn_config
        for _ in text.split("."):
            yield FakeChunk(22050, PCM)


def test_detects_each_api():
    assert piper_api(OldStreamVoice()) == "stream_raw"
    assert piper_api(OldWavOnlyVoice()) == "wav_file"
    assert piper_api(NewVoice()) == "chunks"


def test_old_stream_api():
    voice = OldStreamVoice()
    out = list(iter_piper_audio(voice, "Hi.", length_scale=1.2))
    assert out == [(PCM, 22050), (PCM, 22050)]
    assert voice.kwargs == {"length_scale": 1.2}


def test_old_wav_api_round_trips_through_wave():
    voice = OldWavOnlyVoice()
    assert list(iter_piper_audio(voice, "Hi.", length_scale=0.9)) == [(PCM, 16000)]
    assert voice.length_scale == 0.9


def test_new_chunk_api_passes_synthesis_config():
    voice = NewVoice()
    made = []

    def factory(length_scale):
        made.append(length_scale)
        return {"length_scale": length_scale}

    out = list(iter_piper_audio(voice, "One. Two", length_scale=1.1, syn_config_factory=factory))
    assert out == [(PCM, 22050), (PCM, 22050)]
    assert made == [1.1] and voice.syn_config == {"length_scale": 1.1}


def test_new_api_without_length_scale_uses_voice_defaults():
    voice = NewVoice()
    list(iter_piper_audio(voice, "Hi", length_scale=None))
    assert voice.syn_config is None
