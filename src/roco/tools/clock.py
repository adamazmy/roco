"""Time tools. Reference implementation of the tool-module convention.

Owner: workstream A (brain + text mode).
"""

from __future__ import annotations

import logging
from datetime import datetime, tzinfo
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from roco.config import Config
from roco.tools import ToolContext
from roco.tools.registry import ToolRegistry

log = logging.getLogger(__name__)

DEFAULT_TIMEZONE = "Europe/London"


def local_timezone(cfg: Config | None) -> tzinfo:
    """``assistant.timezone`` from the config, falling back to Europe/London if unknown."""
    name = cfg.get_path("assistant.timezone", DEFAULT_TIMEZONE) if cfg else DEFAULT_TIMEZONE
    try:
        return ZoneInfo(str(name or DEFAULT_TIMEZONE))
    except (ZoneInfoNotFoundError, ValueError):
        log.warning("Unknown timezone %r in config; using %s", name, DEFAULT_TIMEZONE)
        return ZoneInfo(DEFAULT_TIMEZONE)


_NUMBERS = (
    "twelve", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
    "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen",
    "nineteen", "twenty",
)


def _number(n: int) -> str:
    if n <= 20:
        return _NUMBERS[n] if n else "zero"
    tens, units = divmod(n, 10)
    word = {2: "twenty", 3: "thirty", 4: "forty", 5: "fifty"}[tens]
    return f"{word}-{_NUMBERS[units]}" if units else word


def spoken_time(t: datetime) -> str:
    """How a British person says it: "five past two in the afternoon", "midnight"."""
    h, m = t.hour, t.minute
    if m == 0 and h in (0, 12):
        return "midnight" if h == 0 else "midday"
    target = h if m <= 30 else (h + 1) % 24
    hour_word = _NUMBERS[target % 12] if target % 12 else "twelve"
    if m == 0:
        phrase = f"{hour_word} o'clock"
    elif m in (15, 30, 45):
        phrase = {15: "quarter past", 30: "half past", 45: "quarter to"}[m] + f" {hour_word}"
    else:
        mins = m if m < 30 else 60 - m
        unit = "minute" if mins == 1 else "minutes"
        rel = "past" if m < 30 else "to"
        phrase = f"{_number(mins)} {unit} {rel} {hour_word}"
    if target == 0:
        return f"{phrase} at night" if m else "midnight"
    if target == 12:
        return phrase if m else "midday"
    part = "in the morning" if target < 12 else "in the afternoon" if target < 18 else (
        "in the evening" if target < 22 else "at night"
    )
    return f"{phrase} {part}"


def describe_date(now: datetime) -> str:
    """e.g. ``Friday 20 November 2026``."""
    return f"{now:%A} {now.day} {now:%B %Y}"


def describe_now(now: datetime) -> str:
    """e.g. ``five minutes past two in the afternoon on Friday 20 November 2026 (14:05)``.

    The spoken form comes first because small models tend to echo tool results verbatim.
    """
    return f"{spoken_time(now)} on {now:%A} {now.day} {now:%B %Y} ({now:%H:%M})"


def register(registry: ToolRegistry, ctx: ToolContext) -> None:
    tz = local_timezone(ctx.config)

    @registry.tool(
        description="Get the current local date and time.",
        parameters={"type": "object", "properties": {}, "required": []},
    )
    def get_current_time() -> str:
        return describe_now(datetime.now(tz))
