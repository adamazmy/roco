"""Fixes from Adam's first spoken Phase 3 test (2026-11-21)."""

from __future__ import annotations

import pytest

from roco.personality.commands import parse_command
from roco.personality.core import _is_example


@pytest.mark.parametrize(
    "heard, preset",
    [
        ("No, no, no, go into unhinged form.", "unhinged"),
        ("Can you go on on hinged mode?", "unhinged"),  # Whisper's "unhinged"
        ("and drive us behave.", "chill"),  # garbled start, ends in "behave"
        ("Hey Jarvis, behave.", "chill"),
    ],
)
def test_loosely_phrased_commands_are_recognised(heard, preset):
    command = parse_command(heard)
    assert command is not None and command.value == preset


@pytest.mark.parametrize(
    "text",
    [
        "you never behave when I ask you to",
        "Tell me how to behave at a wedding",
        "I was in unhinged mode last night",
        "What do you think of my sleep schedule?",
    ],
)
def test_ordinary_sentences_are_not_commands(text):
    assert parse_command(text) is None


def test_copied_example_lines_are_dropped_even_in_pieces():
    assert _is_example("Your sleep schedule is a crime scene")
    assert _is_example("and your commit history is the evidence.")
    assert not _is_example("Your sleep schedule is a mess.")
    assert not _is_example("Phone.")  # too short to tell


def test_filter_sentence_drops_an_example_line():
    from roco.config import load_config
    from roco.personality import Personality

    p = Personality.from_config(load_config(overrides={"personality": {"file": None}}))
    assert p.filter_sentence("Your sleep schedule is a crime scene.") == ""
    assert p.filter_sentence("Bed. Now.") == "Bed. Now."


# ---- 2026-11-22: Whisper noise around the key phrase (spoken retest on integration/phase3)
@pytest.mark.parametrize(
    "heard, preset",
    [
        ("No, can you go into unhinged mode?", "unhinged"),
        ("of this unhinged mode.", "unhinged"),  # clipped start after the wake word
        ("this behavior.", "chill"),  # "behave" heard as "behavior"
        ("Behaviour.", "chill"),
        ("uh the chill mode", "chill"),
        ("okay so savage mode now", "unhinged"),
        ("No, professional mode.", "professional"),
    ],
)
def test_noisy_mode_commands_are_recognised(heard, preset):
    command = parse_command(heard)
    assert command is not None and command.value == preset, heard


@pytest.mark.parametrize(
    "text",
    [
        "I was in unhinged mode last night",
        "Tell me how to behave at a wedding",
        "you never behave when I ask you to",
        "Are you in unhinged mode?",
        "What is unhinged mode?",
        "Don't go unhinged.",
        "I like your behaviour",
        "Explain unhinged mode to Goose",
        "My mate's dog has a really unhinged mode of walking about the park",  # too long
    ],
)
def test_noisy_matching_leaves_ordinary_speech_alone(text):
    assert parse_command(text) is None, text


@pytest.mark.parametrize(("text", "value"), [("No flirting.", 0), ("no roasting", 0)])
def test_leading_no_still_means_no(text, value):
    command = parse_command(text)
    assert command is not None and command.kind == "set" and command.value == value


def test_prompt_tells_adam_the_exact_command():
    from roco.brain.persona import system_prompt

    assert '"Say: ROCO, unhinged mode"' in system_prompt()


@pytest.mark.parametrize(
    ("text", "key", "step"),
    [
        ("Could you maybe get a bit more unhinged today?", "roast", 25),
        ("I want you to be way more savage with me from now on.", "roast", 25),
        ("Be less chatty.", "chattiness", -25),
    ],
)
def test_more_or_less_requests_adjust_instead_of_reaching_the_model(text, key, step):
    command = parse_command(text)
    assert command is not None and (command.kind, command.key, command.value) == (
        "adjust", key, step,
    )


@pytest.mark.parametrize(
    "text", ["My brother got more savage last year.", "You are getting less flirty, I like it"]
)
def test_more_or_less_in_ordinary_speech_is_left_alone(text):
    assert parse_command(text) is None
