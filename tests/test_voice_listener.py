"""Listener logic with fake mic/wake/STT: wake flow, mute, follow-up timing."""

from __future__ import annotations

import pytest

from roco.bus import State
from roco.voice.endpoint import RecordingSettings
from roco.voice.listener import VoiceListener
from voice_fakes import SILENCE, SPEECH, FakeSource, FakeTranscriber, FakeWake

WAKE = 2222.0  # frames at this RMS make FakeWake fire
GUARD = 4  # round(0.3 / 0.08): frames skipped after (re)starting to listen


@pytest.fixture
def states(bus):
    seen: list = []
    bus.subscribe("state", seen.append)
    return seen


def make(bus, source, wake=None, text="what time is it", **kw):
    listener = VoiceListener(
        source, wake or FakeWake(WAKE), FakeTranscriber(text), bus,
        recording=RecordingSettings(silence_s=0.4, max_s=15, start_timeout_s=4.0), **kw,
    )

    def close_when_exhausted(_reads):
        if not source.frames:
            listener.close()

    source.on_read = close_when_exhausted
    return listener


def test_wake_then_record_then_transcribe(bus, states):
    src = FakeSource([(SILENCE, GUARD + 10), (WAKE, 1), (SILENCE, 2), (SPEECH, 10),
                      (SILENCE, 10)])
    lis = make(bus, src)
    assert lis.listen() == "what time is it"
    assert states == [State.LISTENING, State.THINKING]
    assert src.flushes == 1  # stale audio dropped before listening
    # speech frames + pre-roll + 2 trailing, not the whole trailing silence
    n = lis.transcriber.calls[0]
    assert 10 <= n <= 10 + 4 + 2
    t = lis.last_timing
    assert t.stt_start <= t.stt_done and t.speech_end <= t.stt_start


def test_question_said_without_pause_after_wake_word_is_kept(bus):
    src = FakeSource([(SILENCE, GUARD), (WAKE, 1), (SPEECH, 12), (SILENCE, 10)])
    lis = make(bus, src)
    assert lis.listen() == "what time is it"
    assert lis.transcriber.calls[0] >= 12 - 3  # at most the 0.24 s tail is skipped


def test_wake_word_tail_is_not_mistaken_for_the_question(bus):
    # "...vis" is still loud right after detection, then a pause, then the question.
    src = FakeSource([(SILENCE, GUARD), (WAKE, 1), (SPEECH, 3), (SILENCE, 3), (SPEECH, 8),
                      (SILENCE, 10)])
    lis = make(bus, src)
    assert lis.listen() == "what time is it"
    assert lis.transcriber.calls[0] >= 8


def test_no_speech_after_wake_returns_none_and_idle(bus, states):
    src = FakeSource([(SILENCE, GUARD), (WAKE, 1), (SILENCE, 80)])
    lis = make(bus, src)
    assert lis.listen() is None
    assert states == [State.LISTENING, State.IDLE]
    assert lis.transcriber.calls == []


def test_unintelligible_transcript_returns_none(bus, states):
    src = FakeSource([(SILENCE, GUARD), (WAKE, 1), (SILENCE, 1), (SPEECH, 6), (SILENCE, 10)])
    lis = make(bus, src, text=" ... ")
    assert lis.listen() is None
    assert states[-1] == State.IDLE


def test_frames_heard_while_muted_are_dropped(bus):
    src = FakeSource([(SILENCE, GUARD), (WAKE, 3), (SILENCE, 3)])
    wake = FakeWake(WAKE)
    lis = make(bus, src, wake)
    lis.set_muted(True)
    assert lis.listen() is None  # ROCO saying "jarvis" must not wake itself
    assert wake.seen == 0


def test_unmute_resets_wake_word_and_skips_echo_guard(bus):
    src = FakeSource([(SILENCE, 3)])
    wake = FakeWake(WAKE)
    lis = make(bus, src, wake)
    lis.set_muted(True)

    def unmute_after_muted_frames(reads):
        if reads == 4:  # the 3 muted frames are gone
            lis.set_muted(False)
            # echo of ROCO's last word, then the room
            src.add(WAKE, GUARD).add(SILENCE, 5)
        if not src.frames:
            lis.close()

    src.on_read = unmute_after_muted_frames
    resets_before = wake.resets
    assert lis.listen() is None
    assert wake.resets > resets_before
    assert wake.seen == 5  # the echo frames inside the guard never reached the detector


def test_is_speaking_mutes_automatically(bus):
    speaking = {"on": True}
    src = FakeSource([(WAKE, 5)])
    wake = FakeWake(WAKE)
    lis = make(bus, src, wake, is_speaking=lambda: speaking["on"])
    assert lis.listen() is None
    assert wake.seen == 0


def test_followup_times_out_after_window(bus, states):
    src = FakeSource([(SILENCE, 200)])
    lis = make(bus, src)
    assert lis.listen_followup(5.0) is None
    consumed = 200 - len(src.frames)
    # 5.0 s of 80 ms frames (63rd frame crosses 5 s) after the echo guard
    assert consumed == GUARD + 63
    assert states == [State.LISTENING, State.IDLE]


def test_followup_hears_speech_without_wake_word(bus, states):
    src = FakeSource([(SILENCE, GUARD + 20), (SPEECH, 6), (SILENCE, 10)])
    wake = FakeWake(WAKE)
    lis = make(bus, src, wake)
    assert lis.listen_followup(5.0) == "what time is it"
    assert wake.seen == 0
    assert states == [State.LISTENING, State.THINKING]
    assert lis.transcriber.calls[0] >= 6 + 1  # includes pre-roll


def test_close_unblocks_listen(bus):
    src = FakeSource()
    lis = make(bus, src)
    lis.close()
    assert lis.listen() is None
    assert lis.listen_followup(1.0) is None
    assert src.closed


def test_stt_crash_is_contained(bus, states):
    src = FakeSource([(SILENCE, GUARD), (WAKE, 1), (SILENCE, 1), (SPEECH, 6), (SILENCE, 10)])
    lis = make(bus, src)

    def boom(frames):
        raise RuntimeError("model exploded")

    lis.transcriber = boom
    assert lis.listen() is None
    assert states[-1] == State.IDLE
