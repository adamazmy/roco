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
