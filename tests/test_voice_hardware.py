"""Real audio-device tests. Skipped in CI (`-m "not hardware"`) and without the voice extra.

Run locally with ``pytest -m hardware tests/test_voice_hardware.py`` after
``roco download-models``. They print numbers; the listening ones need a human.
"""

from __future__ import annotations

import time

import pytest

pytestmark = pytest.mark.hardware

pytest.importorskip("sounddevice")
pytest.importorskip("numpy")


@pytest.fixture
def real_home(monkeypatch):
    """Use the real ~/.roco so the downloaded models are found."""
    monkeypatch.delenv("ROCO_HOME", raising=False)


def test_mic_delivers_80ms_frames():
    from roco.voice.audio import FRAME_SAMPLES
    from roco.voice.mic import MicStream

    mic = MicStream(None)
    try:
        mic.flush()
        frames = [mic.read(1.0) for _ in range(12)]
    finally:
        mic.close()
    assert all(f is not None and len(f) == FRAME_SAMPLES for f in frames)


def test_piper_plays_on_speaker(real_home, config, bus):
    from roco.voice.factory import create_speaker

    mouth: list[float] = []
    bus.subscribe("mouth", mouth.append)
    speaker = create_speaker(config, bus)
    try:
        t0 = time.monotonic()
        speaker.say("This is a hardware test of ROCO's voice.")
        assert speaker.wait_done(30)
        assert time.monotonic() - t0 > 1.0  # real-time playback, not instant
    finally:
        speaker.close()
    assert max(mouth) > 0.3 and mouth[-1] == 0.0
