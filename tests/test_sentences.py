"""Clause-level release of long sentences (SentenceSplitter.clause_min_chars)."""

from __future__ import annotations

from roco.brain.sentences import SentenceSplitter

LIST_REPLY = (
    "Tomorrow you have the dentist at ten o'clock in the morning, lunch with Sam at one, "
    "and the weekly review at six in the evening."
)


def test_long_sentence_is_released_at_the_first_comma_after_the_threshold():
    splitter = SentenceSplitter()
    out = []
    for ch in LIST_REPLY:  # token by token
        out += splitter.feed(ch)
    out.append(splitter.flush())
    # The remainder is shorter than the threshold again, so it stays one chunk
    # (splitting at every comma would make the voice choppy).
    assert out == [
        "Tomorrow you have the dentist at ten o'clock in the morning,",
        "lunch with Sam at one, and the weekly review at six in the evening.",
    ]
    assert " ".join(out) == LIST_REPLY


def test_first_clause_arrives_before_the_sentence_ends():
    splitter = SentenceSplitter()
    released = splitter.feed(LIST_REPLY[:70])
    assert released == ["Tomorrow you have the dentist at ten o'clock in the morning,"]


def test_short_sentences_and_number_commas_are_not_split():
    splitter = SentenceSplitter()
    text = (
        "Sure, that's fine. "
        "You have 10,000 unread emails, which is rather a lot of post to read."
    )
    out = splitter.feed(text) + [splitter.flush()]
    assert out == [
        "Sure, that's fine.",
        "You have 10,000 unread emails, which is rather a lot of post to read.",
    ]


def test_clause_splitting_can_be_turned_off():
    splitter = SentenceSplitter(clause_min_chars=None)
    assert splitter.feed(LIST_REPLY[:-1]) == []
    assert splitter.flush() == LIST_REPLY[:-1]


# --------------------------------------------------------------------------- cleanup
def test_inline_dash_bullets_and_email_ids_are_not_read_aloud():
    from roco.brain.sentences import clean_sentence

    raw = "Here are your unread emails: - From Kiwi.com: cheap flights [id o1]"
    assert clean_sentence(raw) == "Here are your unread emails: From Kiwi.com: cheap flights"
    assert clean_sentence("- From UNiDAYS, a better deal.") == "From UNiDAYS, a better deal."
    assert clean_sentence("1. Buy milk.") == "Buy milk."


def test_hyphens_between_words_and_numbers_stay():
    from roco.brain.sentences import clean_sentence

    walk = "It's a ten - fifteen minute walk."
    assert clean_sentence(walk) == walk
    cold = "It's minus 5, or -5 degrees."
    assert clean_sentence(cold) == cold


def test_first_chunk_may_be_a_short_clause():
    splitter = SentenceSplitter(first_clause_min_chars=20)
    out = splitter.feed("Today you have a gym session, from eight to nine, and then lunch")
    assert out == ["Today you have a gym session,"]  # later clauses wait for 60 chars
    assert splitter.feed(" with Sam at one") == []  # under 60 chars: no early clause now
    assert splitter.flush() == "from eight to nine, and then lunch with Sam at one"


# --------------------------------------------------------------------------- first chunk
def _first_chunks(text: str) -> list[str]:
    from roco.brain.sentences import FIRST_CLAUSE_MIN_CHARS

    splitter = SentenceSplitter(first_clause_min_chars=FIRST_CLAUSE_MIN_CHARS)
    out = []
    for ch in text:  # token by token
        out += splitter.feed(ch)
    rest = splitter.flush()
    return out + ([rest] if rest else [])


def test_long_first_sentence_without_comma_starts_on_a_short_phrase():
    text = "It's twenty-four minutes past ten at night on a chilly Saturday. Bed soon."
    assert _first_chunks(text) == [
        "It's twenty-four minutes past ten",
        "at night on a chilly Saturday.",
        "Bed soon.",
    ]


def test_first_short_clause_is_released_at_its_comma():
    assert _first_chunks("Did you know, octopuses have three hearts and blue blood?") == [
        "Did you know,",
        "octopuses have three hearts and blue blood?",
    ]


def test_joining_word_split_waits_for_the_whole_word():
    from roco.brain.sentences import FIRST_CLAUSE_MIN_CHARS

    splitter = SentenceSplitter(first_clause_min_chars=FIRST_CLAUSE_MIN_CHARS)
    # "an" must not be taken for "and" before the next token arrives.
    assert splitter.feed("Your next meeting is on Tuesday an") == []
    assert splitter.feed("d then") == ["Your next meeting is on Tuesday"]


def test_only_the_first_chunk_is_split_on_joining_words():
    text = "Sure thing. Your next meeting is tomorrow at ten with the design team."
    assert _first_chunks(text) == [
        "Sure thing.",
        "Your next meeting is tomorrow at ten with the design team.",
    ]
