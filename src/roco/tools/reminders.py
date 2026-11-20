"""Reminder tools + background scheduler that publishes ``announce`` events.

Owner: workstream A. To implement:
- ``set_reminder`` (minutes from now OR a clock time), ``list_reminders``, ``cancel_reminder``
- a scheduler thread persisted to ``config["reminders"]["file"]`` (in ~/.roco)
  that calls ``ctx.bus.announce(text)`` when a reminder is due.
"""

from __future__ import annotations

from roco.tools import ToolContext
from roco.tools.registry import ToolRegistry


def register(registry: ToolRegistry, ctx: ToolContext) -> None:
    """Not implemented yet: registers nothing (workstream A)."""
