"""Personality settings (spec §2.1): six 0-100 sliders, three switches and four presets.

:class:`PersonalitySettings` is an immutable value. Anything that changes it builds a new
one (:meth:`PersonalitySettings.with_values`), so a snapshot handed to another thread
never changes under it.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass, fields, replace
from typing import Any

SLIDERS: tuple[str, ...] = ("roast", "flirt", "sass", "chattiness", "proactivity", "energy")
SWITCHES: tuple[str, ...] = ("sensitive_topics_ok", "roast_guests", "tone_down_with_guests")
SETTING_KEYS: tuple[str, ...] = SLIDERS + SWITCHES

DEFAULT_PRESET = "default"


@dataclass(frozen=True)
class PersonalitySettings:
    roast: int = 50
    flirt: int = 50
    sass: int = 50
    chattiness: int = 50
    proactivity: int = 50
    energy: int = 50
    sensitive_topics_ok: bool = False
    roast_guests: bool = True
    tone_down_with_guests: bool = True

    def __post_init__(self) -> None:
        for name in SLIDERS:
            object.__setattr__(self, name, clamp(getattr(self, name)))
        for name in SWITCHES:
            object.__setattr__(self, name, bool(getattr(self, name)))

    def with_values(self, values: Mapping[str, Any]) -> PersonalitySettings:
        """A copy with the known keys of ``values`` applied (others are ignored)."""
        known = {k: v for k, v in values.items() if k in SETTING_KEYS and v is not None}
        for key in SWITCHES:
            if key in known:
                known[key] = to_bool(known[key])
        return replace(self, **known) if known else self

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def diff(self, other: PersonalitySettings) -> dict[str, Any]:
        """Keys whose value in ``self`` differs from ``other``."""
        return {
            f.name: getattr(self, f.name)
            for f in fields(self)
            if getattr(self, f.name) != getattr(other, f.name)
        }


def preset(name: str) -> PersonalitySettings:
    try:
        return PRESETS[name.strip().lower()]
    except KeyError:
        raise ValueError(
            f"Unknown personality preset {name!r}; choose from {', '.join(PRESETS)}"
        ) from None


def matching_preset(settings: PersonalitySettings) -> str | None:
    """The preset ``settings`` equals exactly, if any."""
    return next((name for name, value in PRESETS.items() if value == settings), None)


def clamp(value: Any) -> int:
    try:
        number = round(float(value))
    except (TypeError, ValueError):
        raise ValueError(
            f"Personality level must be a number from 0 to 100, got {value!r}"
        ) from None
    return max(0, min(100, int(number)))


def to_bool(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def settings_from_config(section: Mapping[str, Any]) -> PersonalitySettings:
    """``personality`` config section -> settings: the preset, then explicit keys over it."""
    base = preset(str(section.get("preset") or DEFAULT_PRESET))
    return base.with_values(section)


PRESETS: dict[str, PersonalitySettings] = {
    "default": PersonalitySettings(),
    "chill": PersonalitySettings(
        roast=20, flirt=25, sass=25, chattiness=40, proactivity=30, energy=25
    ),
    "unhinged": PersonalitySettings(
        roast=100, flirt=80, sass=100, chattiness=60, proactivity=60, energy=85,
        sensitive_topics_ok=True,
    ),
    "professional": PersonalitySettings(
        roast=0, flirt=0, sass=10, chattiness=35, proactivity=30, energy=40,
        roast_guests=False,
    ),
}
