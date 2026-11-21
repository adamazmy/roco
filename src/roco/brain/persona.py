"""ROCO's persona / system prompt. Shared by every brain implementation.

The prompt is rendered from the personality settings by :mod:`roco.personality.render`;
:func:`system_prompt` gives it for the default settings. A brain with a live
:class:`roco.personality.Personality` calls ``personality.system_prompt(...)`` instead.
"""

from __future__ import annotations

from roco.personality.render import EMOTION_TAGS, PromptParts, render_system_prompt
from roco.personality.settings import PersonalitySettings

__all__ = ["EMOTION_TAGS", "system_prompt"]


def system_prompt(
    name: str = "ROCO", user: str = "Adam", settings: PersonalitySettings | None = None
) -> str:
    return render_system_prompt(
        PromptParts(name=name, user=user, settings=settings or PersonalitySettings())
    )
