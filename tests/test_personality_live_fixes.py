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
