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
