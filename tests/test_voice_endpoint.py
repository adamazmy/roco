"""Silence detection and noise floor on synthetic RMS sequences (no numpy needed)."""

from __future__ import annotations

import pytest

from roco.voice.audio import frame_rms, iter_blocks, mouth_from_rms, resolve_device
from roco.voice.endpoint import Endpoint, NoiseFloor, RecordingSettings, SilenceDetector
from voice_fakes import frame

FRAME_S = 0.08


def run(det: SilenceDetector, levels: list[float]) -> list[Endpoint]:
    return [det.feed(x) for x in levels]


def settings(**kw) -> RecordingSettings:
    base = dict(silence_s=0.9, max_s=15.0, margin=2.5, min_threshold=300.0,
                start_frames=2, start_timeout_s=4.0)
    base.update(kw)
    return RecordingSettings(**base)


def frames_for(seconds: float) -> int:
    return round(seconds / FRAME_S)


def test_frame_rms_matches_level_without_numpy():
    assert frame_rms(frame(1234)) == pytest.approx(1234)
    assert frame_rms(frame(500).tobytes()) == pytest.approx(500)
    assert frame_rms([]) == 0.0


def test_noise_floor_learns_room_and_ignores_speech():
    floor = NoiseFloor(initial=1000)
    floor.update(100)
    assert floor.level == 100  # drops at once to a quieter frame
    for _ in range(40):
        floor.update(100)
    # Two seconds of speech with natural gaps between words barely move it.
    for _ in range(5):
        for level in [3000, 3500, 2800, 400]:
            floor.update(level)
    assert floor.level == pytest.approx(100)
    for _ in range(100):
        floor.update(800)  # a fan switched on: learnt after a window (~3 s) or so
    assert floor.level == pytest.approx(800, rel=0.05)


def test_ends_after_configured_silence():
    det = SilenceDetector(settings(), NoiseFloor(100), FRAME_S)
    events = run(det, [100] * 3 + [3000] * 10 + [100] * 20)
    assert events[4] is Endpoint.SPEECH_START
    end = events.index(Endpoint.SILENCE)
    silent_frames = end - 12  # index of the first silent frame after speech is 13
    assert silent_frames * FRAME_S == pytest.approx(0.9, abs=FRAME_S)
    assert det.speech_end_s == pytest.approx(13 * FRAME_S)


def test_short_pause_mid_sentence_does_not_end_recording():
    det = SilenceDetector(settings(), NoiseFloor(100), FRAME_S)
    events = run(det, [3000] * 5 + [100] * frames_for(0.5) + [3000] * 5)
    assert Endpoint.SILENCE not in events


def test_single_click_is_not_speech():
    det = SilenceDetector(settings(start_timeout_s=1.0), NoiseFloor(100), FRAME_S)
    events = run(det, [100, 5000, 100, 100] + [100] * 20)
    assert Endpoint.SPEECH_START not in events
    assert events[-1] is Endpoint.NO_SPEECH or Endpoint.NO_SPEECH in events


def test_no_speech_times_out_at_start_timeout():
    det = SilenceDetector(settings(start_timeout_s=5.0), NoiseFloor(100), FRAME_S)
    events = run(det, [100] * 100)
    first = events.index(Endpoint.NO_SPEECH)
    assert (first + 1) * FRAME_S == pytest.approx(5.0, abs=FRAME_S)


def test_recording_is_capped_at_max_length():
    det = SilenceDetector(settings(max_s=2.0), NoiseFloor(100), FRAME_S)
    events = run(det, [3000] * 100)
    assert events.index(Endpoint.MAX_LENGTH) + 1 == frames_for(2.0)


def test_threshold_adapts_to_noisy_room():
    floor = NoiseFloor(100)
    for _ in range(100):
        floor.update(800)  # learnt while idle (the listener feeds every idle frame)
    det = SilenceDetector(settings(), floor, FRAME_S)
    assert det.threshold == pytest.approx(800 * 2.5, rel=0.05)
    assert run(det, [800] * 20)[-1] is Endpoint.CONTINUE  # the fan is not speech
    assert Endpoint.SPEECH_START in run(det, [4000] * 3)
    quiet = SilenceDetector(settings(), NoiseFloor(100), FRAME_S)
    assert quiet.threshold == 300.0  # min_threshold wins in a silent room


def test_speech_level_frames_do_not_teach_the_floor():
    floor = NoiseFloor(100)
    det = SilenceDetector(settings(max_s=100), floor, FRAME_S)
    run(det, [3000] * 100)
    assert floor.level == pytest.approx(100)


def test_settings_from_config_reads_voice_recording(config):
    s = RecordingSettings.from_config(config.get_path("voice.recording"), start_timeout_s=5)
    assert (s.silence_s, s.max_s, s.margin, s.start_timeout_s) == (0.9, 15.0, 2.5, 5)


def test_mouth_mapping_and_blocks():
    assert mouth_from_rms(0) == 0.0
    assert 0 < mouth_from_rms(1000) < mouth_from_rms(4000) < 1.0
    assert mouth_from_rms(30000) == 1.0
    blocks = list(iter_blocks(b"\x01\x00" * 250, 100))
    assert [len(b) for b in blocks] == [200, 200, 100]


DEVICES = [
    {"name": "Microsoft Sound Mapper - Input", "max_input_channels": 2, "max_output_channels": 0},
    {"name": "Microphone (USB Audio)", "max_input_channels": 1, "max_output_channels": 0},
    {"name": "Speakers (Realtek)", "max_input_channels": 0, "max_output_channels": 2},
    {"name": "Microphone (USB Audio)", "max_input_channels": 1, "max_output_channels": 0},
]


@pytest.mark.parametrize(
    "spec,kind,expected",
    [(None, "input", None), ("", "output", None), (1, "input", 1), ("2", "output", 2),
     ("usb", "input", 1), ("REALTEK", "output", 2)],
)
def test_resolve_device(spec, kind, expected):
    assert resolve_device(spec, kind, DEVICES) == expected


@pytest.mark.parametrize("spec,kind", [("usb", "output"), (2, "input"), (9, "input"),
                                       ("nope", "input")])
def test_resolve_device_errors_list_candidates(spec, kind):
    with pytest.raises(ValueError, match="Available"):
        resolve_device(spec, kind, DEVICES)
