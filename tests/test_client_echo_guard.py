"""Echo guard: barge-ins that were ROCO hearing itself (hardware test 2026-11-22)."""

from __future__ import annotations

import pytest

from roco.client.echo_guard import is_echo, phonetic, words

TIME = ["It's just gone half past three on Sunday afternoon.", "Anything else you need?"]
STORY = [
    "Ollie was known for his love of adventure and his ability to solve puzzles.",
    "One day, Ollie stumbled upon a treasure map hidden in the shadows of a sunken ship.",
]


# What ROCO said -> what the brain transcribed from the leaked echo (Adam's laptop).
@pytest.mark.parametrize("heard", ["on the song.", "Just go.", "House three.", "I've passed."])
def test_the_echo_pairs_from_the_spoken_test_are_caught(heard):
    verdict = is_echo(heard, TIME)
    assert verdict.echo, verdict


@pytest.mark.parametrize("heard", ["a treasure map", "and his ability", "Holly was known",
                                   "sunk in ship"])
def test_garbled_echo_of_a_story_is_caught(heard):
    assert is_echo(heard, STORY).echo


@pytest.mark.parametrize("heard", [
    "Stop, do I have any unread emails?", "Wait a sec", "What time is the dentist?",
    "never mind", "Do I have meetings tomorrow", "set a reminder", "turn the lights on",
    "tell me a joke", "louder please", "that's wrong", "go into unhinged mode",
    "drop any hundred emails.", "Great.", "Okay thanks", "Hey Jarvis",
])
@pytest.mark.parametrize("reference", [TIME, STORY], ids=["time", "story"])
def test_real_interruptions_are_kept(heard, reference):
    assert not is_echo(heard, reference).echo


def test_empty_and_long_transcripts_are_not_judged_echo():
    assert is_echo("", TIME).reason == "empty"
    long = "one day ollie stumbled upon a treasure map hidden in the shadows"
    assert is_echo(long, STORY).reason == "long"


def test_normalisation_helpers():
    assert words("It's 3 o'clock!") == ["it's", "three", "o'clock"]
    assert phonetic("sunday") == phonetic("sundae")
    assert phonetic("past") == phonetic("passed")
