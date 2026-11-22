"""ROCO's personality (spec §2, §2.1, §7): settings, presets, per-person overrides,
voice commands and the system prompt they render to.

- :mod:`.settings`: the sliders, switches and presets;
- :mod:`.people`: person profiles (the Yousaf rule lives here);
- :mod:`.render`: settings -> concrete behaviour text for the 7B model;
- :mod:`.commands`: deterministic voice-command matching ("ROCO, unhinged mode");
- :mod:`.core`: :class:`Personality`, the live state the brain owns.
"""

from roco.personality.commands import Command, parse_command
from roco.personality.core import CommandResult, Personality
from roco.personality.people import PeopleStore, Person
from roco.personality.render import Company
from roco.personality.settings import PRESETS, PersonalitySettings, preset

__all__ = [
    "PRESETS",
    "Command",
    "CommandResult",
    "Company",
    "PeopleStore",
    "Person",
    "Personality",
    "PersonalitySettings",
    "parse_command",
    "preset",
]
