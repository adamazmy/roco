"""Time tools. Reference implementation of the tool-module convention.

Owner: workstream A (brain + text mode).
"""

from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

from roco.tools import ToolContext
from roco.tools.registry import ToolRegistry


def register(registry: ToolRegistry, ctx: ToolContext) -> None:
    tz = ZoneInfo(ctx.config.get_path("assistant.timezone", "Europe/London"))

    @registry.tool(
        description="Get the current local date and time.",
        parameters={"type": "object", "properties": {}, "required": []},
    )
    def get_current_time() -> str:
        now = datetime.now(tz)
        return now.strftime("%A %d %B %Y, %H:%M")
