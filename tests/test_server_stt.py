"""roco.server.stt.StreamingSTT: speculative final decode and committed prefix."""

from __future__ import annotations

from roco.server.stt import StreamingSettings, StreamingSTT, frame_rms
from server_fakes import FRAME_SAMPLES, FakeEngine, InlineExecutor, silence_frame, word_frame

FRAME_BYTES = FRAME_SAMPLES * 2


def words(n: int, start: int = 0) -> str:
    return " ".join(f"w{i}" for i in range(start, start + n))


def test_frame_rms():
    assert frame_rms(silence_frame()) == 0
    assert round(frame_rms(word_frame(0))) == 1000
    assert frame_rms(b"") == 0


def test_speculative_decode_is_used_when_speech_really_ended():
    engine = FakeEngine()
    stt = StreamingSTT(engine, InlineExecutor(), settings=StreamingSettings(partial_min_s=0))
    for i in range(8):
        stt.feed(word_frame(i))
    for _ in range(4):  # 320 ms of quiet: the speculative decode runs here
        stt.feed(silence_frame())
    assert stt.stats["speculative"] == 1
    for _ in range(2):  # the body's VAD keeps waiting a bit longer
        stt.feed(silence_frame())
    speech_end = stt.total_bytes - 6 * FRAME_BYTES
    assert stt.finish(speech_end) == words(8)
    assert stt.stats["speculative_used"] == 1
    assert stt.stats["final"] == 0
    assert len(engine.calls) == 1  # nothing decoded after end of speech


def test_speculation_is_discarded_when_speech_resumes():
    engine = FakeEngine()
    stt = StreamingSTT(engine, InlineExecutor(), settings=StreamingSettings(partial_min_s=0))
    for i in range(3):
        stt.feed(word_frame(i))
    for _ in range(4):
        stt.feed(silence_frame())  # a pause mid-sentence: speculative decode
    for i in range(3, 6):
        stt.feed(word_frame(i))  # ... and Adam carries on
    stt.feed(silence_frame())
    speech_end = stt.total_bytes - FRAME_BYTES
    assert stt.finish(speech_end) == words(6)
    assert stt.stats["speculative_used"] == 0
    assert stt.stats["final"] == 1


def test_long_utterance_commits_a_prefix_and_decodes_only_the_tail():
    engine = FakeEngine(per_segment=5)
    partials: list[str] = []
    settings = StreamingSettings(partial_min_s=1.0, partial_every_s=0.4, keep_tail_s=0.5,
                                 speculative_quiet_ms=10_000)
    stt = StreamingSTT(engine, InlineExecutor(), settings=settings, on_partial=partials.append)
    for i in range(40):  # 3.2 s of continuous speech
        stt.feed(word_frame(i))
    stt.feed(silence_frame())
    assert stt.stats["partials"] >= 3
    assert partials and partials[-1].startswith("w0 w1")
    text = stt.finish(stt.total_bytes - FRAME_BYTES)
    assert text == words(40)
    assert engine.calls[-1] < 41 * FRAME_BYTES / 2  # the final decode skipped the prefix


def test_finish_without_background_work_decodes_everything():
    engine = FakeEngine()
    stt = StreamingSTT(engine, InlineExecutor(),
                       settings=StreamingSettings(partial_min_s=0, speculative_quiet_ms=10_000))
    for i in range(5):
        stt.feed(word_frame(i))
    assert stt.finish() == words(5)
    stt.feed(word_frame(9))  # ignored once finished
    assert stt.total_bytes == 5 * FRAME_BYTES


def test_plain_callable_engine():
    stt = StreamingSTT(lambda pcm, rate: "hello there", InlineExecutor(),
                       settings=StreamingSettings(partial_min_s=0))
    stt.feed(word_frame(0))
    assert stt.finish() == "hello there"


def test_settings_from_config_ignores_unknown_keys_and_casts():
    s = StreamingSettings.from_config({"partial_min_s": "2", "speculative_quiet_ms": 300.0,
                                       "bogus": 1})
    assert s.partial_min_s == 2.0
    assert s.speculative_quiet_ms == 300


def level_frame(value: int) -> bytes:
    import struct

    return struct.pack("<h", value) * FRAME_SAMPLES


def test_speculation_still_counts_when_only_near_silence_follows():
    engine = FakeEngine()
    stt = StreamingSTT(engine, InlineExecutor(), settings=StreamingSettings(partial_min_s=0))
    for i in range(5):
        stt.feed(word_frame(i))
    for _ in range(4):
        stt.feed(silence_frame())  # speculative decode here
    for _ in range(2):
        stt.feed(level_frame(60))  # breath / room noise: the body's VAD called it speech
    assert stt.finish(stt.total_bytes) == words(5)
    assert stt.stats["speculative_used"] == 1


def test_a_quiet_word_after_a_speculation_triggers_another():
    engine = FakeEngine()
    stt = StreamingSTT(engine, InlineExecutor(), settings=StreamingSettings(partial_min_s=0))
    for i in range(5):
        stt.feed(word_frame(i))
    for _ in range(4):
        stt.feed(silence_frame())
    stt.feed(level_frame(200))  # too quiet to be "speech" here, but maybe a word
    stt.feed(silence_frame())
    stt.finish(stt.total_bytes - FRAME_BYTES)
    assert stt.stats["speculative"] == 2  # the first guess stopped before that frame
    assert stt.stats["speculative_used"] == 1
    assert stt.stats["final"] == 0
