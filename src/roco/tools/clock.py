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


def describe_now(now: datetime) -> str:
    """e.g. ``Friday 20 November 2026, 14:05``."""
    return f"{now:%A} {now.day} {now:%B %Y, %H:%M}"


def register(registry: ToolRegistry, ctx: ToolContext) -> None:
    tz = local_timezone(ctx.config)

    @registry.tool(
        description="Get the current local date and time.",
        parameters={"type": "object", "properties": {}, "required": []},
    )
    def get_current_time() -> str:
        return describe_now(datetime.now(tz))
