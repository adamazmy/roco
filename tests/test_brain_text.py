"""Sentence splitter, emotion-tag parser and history."""

from __future__ import annotations

import json

import pytest

from roco.brain.emotion import EmotionTagParser
from roco.brain.history import History
from roco.brain.sentences import SentenceSplitter
from roco.bus import Emotion


def split(text: str, chunk: int | None = None) -> list[str]:
    splitter = SentenceSplitter()
    out: list[str] = []
    step = chunk or len(text) or 1
    for i in range(0, len(text), step):
        out += splitter.feed(text[i : i + step])
    rest = splitter.flush()
    return out + ([rest] if rest else [])


@pytest.mark.parametrize(
    "text, expected",
    [
        ("Hello there. How are you?", ["Hello there.", "How are you?"]),
        ("Wow! Really? Yes.", ["Wow!", "Really?", "Yes."]),
        ("It's 3.5 degrees. Nice.", ["It's 3.5 degrees.", "Nice."]),
        ("Meet at 10.30 tomorrow. Okay?", ["Meet at 10.30 tomorrow.", "Okay?"]),
        ("Mr. Smith and Dr. Who are here. Hi.", ["Mr. Smith and Dr. Who are here.", "Hi."]),
        ("Fruit, e.g. Apples, are good. Yes.", ["Fruit, e.g. Apples, are good.", "Yes."]),
        ("That is, i.e. the end. Fine.", ["That is, i.e. the end.", "Fine."]),
        ("Hmm... let me think. Right.", ["Hmm... let me think.", "Right."]),
        ("Well... Maybe. Sure.", ["Well...", "Maybe.", "Sure."]),
        ("Wait… What? No.", ["Wait…", "What?", "No."]),
        ("He said \"Stop.\" Then left.", ["He said \"Stop.\"", "Then left."]),
        ("J. K. Rowling wrote it. Yes.", ["J. K. Rowling wrote it.", "Yes."]),
        ("At 3 p.m. tomorrow. Done.", ["At 3 p.m. tomorrow.", "Done."]),
        ("It costs £4.99. Cheap!", ["It costs £4.99.", "Cheap!"]),
        ("One\nTwo\n\nThree", ["One", "Two", "Three"]),
        ("No punctuation at the end", ["No punctuation at the end"]),
        ("", []),
        ("   ", []),
    ],
)
@pytest.mark.parametrize("chunk", [None, 1, 4])
def test_splitter(text, expected, chunk):
    assert split(text, chunk) == expected


def test_splitter_releases_sentence_once_next_word_starts():
    s = SentenceSplitter()
    assert s.feed("Hello there.") == []
    assert s.feed(" ") == []
    assert s.feed("H") == ["Hello there."]
    assert s.flush() == "H"
    assert s.flush() is None


def test_splitter_strips_inline_emotion_tags():
    assert split("[happy] Hi. [Excited] Wow.") == ["Hi.", "Wow."]


# ---------------------------------------------------------------------- emotion tag


def feed_all(chunks):
    p = EmotionTagParser()
    out = "".join(p.feed(c) for c in chunks) + p.flush()
    return p.emotion, out


@pytest.mark.parametrize(
    "chunks, emotion, text",
    [
        (["[happy] Hi"], Emotion.HAPPY, "Hi"),
        (["[", "SA", "D", "]", "Oh no"], Emotion.SAD, "Oh no"),
        (["\n [ curious ]  Hm"], Emotion.CURIOUS, "Hm"),
        (["Hi there"], None, "Hi there"),
        (["[unknown] Hi"], Emotion.NEUTRAL, "Hi"),
        (["[this is not a tag at all, it is just a long bracket"], None,
         "[this is not a tag at all, it is just a long bracket"),
        (["[hap"], None, "[hap"),  # stream ended inside a "tag"
        ([], None, ""),
    ],
)
def test_emotion_parser(chunks, emotion, text):
    assert feed_all(chunks) == (emotion, text)


def test_emotion_parser_holds_back_until_decided():
    p = EmotionTagParser()
    assert p.feed("[ha") == "" and not p.decided
    assert p.feed("ppy] Hello") == "Hello" and p.decided
    assert p.feed(" [sad] again") == " [sad] again"  # only the leading tag is parsed


# ---------------------------------------------------------------------- history


def test_history_trims_to_max_turns():
    h = History(max_turns=2)
    for i in range(5):
        h.add_turn(f"u{i}", f"a{i}")
    assert [m["content"] for m in h.messages] == ["u3", "a3", "u4", "a4"]


def test_history_zero_turns_keeps_nothing():
    h = History(max_turns=0)
    h.add_turn("u", "a")
    assert h.messages == []


def test_history_persists_and_reloads(tmp_path):
    path = tmp_path / "sub" / "history.json"
    h = History(3, path)
    h.add_turn("hello", "hi")
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["version"] == 1
    assert History(3, path).messages == h.messages
    # Reloading with a smaller limit trims.
    h.add_turn("again", "yes")
    assert [m["content"] for m in History(1, path).messages] == ["again", "yes"]


@pytest.mark.parametrize(
    "raw",
    ["{not json", "[1, 2]", '{"messages": "nope"}', '{"messages": [{"role": "system"}]}'],
)
def test_corrupt_history_starts_fresh_with_warning(tmp_path, caplog, raw):
    path = tmp_path / "history.json"
    path.write_text(raw, encoding="utf-8")
    h = History(3, path)
    assert h.messages == []
    assert "starting fresh" in caplog.text
    h.add_turn("u", "a")  # and it can be overwritten
    assert len(History(3, path)) == 2


def test_history_trim_never_starts_with_assistant(tmp_path):
    path = tmp_path / "history.json"
    msgs = [{"role": "assistant", "content": "x"}, {"role": "user", "content": "u"},
            {"role": "assistant", "content": "a"}]
    path.write_text(json.dumps({"messages": msgs}), encoding="utf-8")
    assert [m["role"] for m in History(5, path).messages] == ["user", "assistant"]


def test_history_clear_deletes_file(tmp_path):
    path = tmp_path / "history.json"
    h = History(3, path)
    h.add_turn("u", "a")
    h.clear()
    assert h.messages == [] and not path.exists()


@pytest.mark.parametrize(
    "hm, expected",
    [
        ((0, 0), "midnight"),
        ((12, 0), "midday"),
        ((14, 5), "five minutes past two in the afternoon"),
        ((19, 54), "six minutes to eight in the evening"),
        ((7, 30), "half past seven in the morning"),
        ((8, 45), "quarter to nine in the morning"),
        ((11, 59), "one minute to twelve"),
        ((23, 40), "twenty minutes to twelve at night"),
        ((0, 15), "quarter past twelve at night"),
        ((9, 0), "nine o'clock in the morning"),
        ((22, 25), "twenty-five minutes past ten at night"),
    ],
)
def test_spoken_time(hm, expected):
    from datetime import datetime

    from roco.tools.clock import spoken_time

    assert spoken_time(datetime(2026, 11, 20, *hm)) == expected


def test_history_for_model_adds_tag_and_tool_trace(tmp_path):
    path = tmp_path / "history.json"
    trace = [
        {"role": "assistant", "content": "", "tool_calls": [{"function": {"name": "t"}}]},
        {"role": "tool", "content": "ok", "tool_name": "t"},
    ]
    h = History(3, path)
    h.add_turn("do it", "Done.", "happy", trace)
    h.add_turn("thanks", "Any time.")
    # The plain view stays tag-free and tool-free (Brain contract).
    assert h.messages == [
        {"role": "user", "content": "do it"},
        {"role": "assistant", "content": "Done."},
        {"role": "user", "content": "thanks"},
        {"role": "assistant", "content": "Any time."},
    ]
    expected = [
        {"role": "user", "content": "do it"},
        *trace,
        {"role": "assistant", "content": "[happy] Done."},
        {"role": "user", "content": "thanks"},
        {"role": "assistant", "content": "Any time."},
    ]
    assert h.for_model() == expected
    assert History(3, path).for_model() == expected  # survives a restart
    assert History(3, path).for_model(tags=False, tools=False) == h.messages


def _trace(result: str) -> list[dict]:
    return [
        {"role": "assistant", "content": "", "tool_calls": [{"function": {"name": "t"}}]},
        {"role": "tool", "content": result, "tool_name": "t"},
    ]


def test_history_caps_tool_results_and_stamps_their_time(tmp_path):
    from datetime import datetime
    from zoneinfo import ZoneInfo

    tz = ZoneInfo("Europe/London")
    h = History(5, tmp_path / "h.json")
    h.add_turn("mail?", "Two emails.", "neutral", _trace("x" * 2000),
               at=datetime(2026, 11, 19, 9, 12, tzinfo=tz))
    h.add_turn("again", "Still two.", "neutral", _trace("short"),
               at=datetime(2026, 11, 20, 20, 5, tzinfo=tz))
    tools = [m for m in h.for_model(now=datetime(2026, 11, 20, 21, 0, tzinfo=tz))
             if m["role"] == "tool"]
    assert "fetched yesterday." in tools[0]["content"]
    assert tools[0]["content"].endswith("... (cut short)")
    assert len(tools[0]["content"]) < 600
    assert "fetched about 55 minutes ago." in tools[1]["content"]
    assert tools[1]["content"].endswith(") short")


def test_history_squeezes_tool_results_of_older_turns():
    h = History(10)
    for i in range(5):
        h.add_turn(f"u{i}", f"a{i}", "happy", _trace(f"r{i}-" + "x" * 300))
    sent = h.for_model(tool_turns=3)
    results = [m["content"] for m in sent if m["role"] == "tool"]
    # Every turn keeps its tool call (no "action without a call" to imitate)...
    assert len(results) == 5
    assert sum(m["role"] == "assistant" and "tool_calls" in m for m in sent) == 5
    # ...but only the last three keep long results.
    assert all(len(r) < 120 for r in results[:2])
    assert all(len(r) > 300 for r in results[2:])
    assert sent[3] == {"role": "assistant", "content": "[happy] a0"}


@pytest.mark.parametrize(
    "text, expected",
    [
        ("Because they make up everything! [chuckling]", ["Because they make up everything!"]),
        ("[laughs softly] Oh, that's good.", ["Oh, that's good."]),
        ("I *really* like owls.", ["I really like owls."]),
        ("Just checking... [get_current_time]", ["Just checking..."]),
        ("Item [1] stays.", ["Item [1] stays."]),  # not a stage direction
    ],
)
def test_splitter_drops_stage_directions_and_asterisks(text, expected):
    assert split(text) == expected


def test_history_stamps_never_contain_a_clock_time(tmp_path):
    import re
    from datetime import datetime, timedelta
    from zoneinfo import ZoneInfo

    tz = ZoneInfo("Europe/London")
    now = datetime(2026, 11, 21, 11, 4, tzinfo=tz)
    h = History(10, tmp_path / "h.json")
    for ago in (timedelta(seconds=30), timedelta(minutes=7), timedelta(hours=3),
                timedelta(days=4)):
        h.add_turn("q", "a", "neutral", _trace("result"), at=now - ago)
    contents = [m["content"] for m in h.for_model(now=now) if m["role"] == "tool"]
    assert [c.split(".")[0] for c in contents] == [
        "(Old result, fetched just now",
        "(Old result, fetched about 7 minutes ago",
        "(Old result, fetched about 3 hours ago",
        "(Old result, fetched on Tuesday 17 November",
    ]
    assert not any(re.search(r"\d{1,2}:\d{2}", c) for c in contents)


def test_old_clock_answers_are_not_replayed(tmp_path):
    from datetime import datetime
    from zoneinfo import ZoneInfo

    tz = ZoneInfo("Europe/London")
    h = History(5, tmp_path / "h.json")
    trace = [
        {"role": "assistant", "content": "", "tool_calls": [
            {"function": {"name": "get_current_time", "arguments": {}}}]},
        {"role": "tool", "tool_name": "get_current_time",
         "content": "four minutes past eleven in the morning on Saturday (11:04)"},
    ]
    h.add_turn("time?", "It's four minutes past eleven.", "neutral", trace,
               at=datetime(2026, 11, 21, 11, 4, tzinfo=tz))
    tool = [m for m in h.for_model(now=datetime(2026, 11, 21, 11, 20, tzinfo=tz))
            if m["role"] == "tool"][0]
    assert "11:04" not in tool["content"] and "eleven" not in tool["content"]
    assert "call get_current_time again" in tool["content"]


def test_part_of_day_boundaries():
    from datetime import datetime

    from roco.tools.clock import part_of_day

    def at(h):
        return part_of_day(datetime(2026, 11, 21, h, 30))

    assert [at(h) for h in (4, 5, 11, 12, 16, 17, 21, 22, 23)] == [
        "night", "morning", "morning", "afternoon", "afternoon", "evening", "evening",
        "night", "night",
    ]
