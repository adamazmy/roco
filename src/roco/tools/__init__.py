"""Tools ROCO can call, and the plumbing that decides which ones are available.

Each tool module exposes ``register(registry, ctx) -> None``. It decides for
itself whether to register anything (e.g. the Google module registers nothing
unless Google is enabled and authenticated). :func:`build_registry` imports the
modules listed in :data:`TOOL_MODULES`; a module whose optional dependencies are
missing is skipped with a log line instead of breaking text mode.
"""

from __future__ import annotations

import importlib
import logging
from dataclasses import dataclass

from roco.bus import EventBus
from roco.config import Config
from roco.tools.registry import Tool, ToolRegistry

log = logging.getLogger(__name__)

__all__ = ["TOOL_MODULES", "Tool", "ToolContext", "ToolRegistry", "build_registry"]

# Order matters only for the order schemas are sent to the model.
TOOL_MODULES: tuple[str, ...] = (
    "roco.tools.clock",  # workstream A
    "roco.tools.reminders",  # workstream A
    "roco.tools.google",  # workstream D
)


@dataclass
class ToolContext:
    """What a tool module may need at registration time."""

    config: Config
    bus: EventBus


def build_registry(ctx: ToolContext, modules: tuple[str, ...] = TOOL_MODULES) -> ToolRegistry:
    registry = ToolRegistry()
    for module_name in modules:
        try:
            module = importlib.import_module(module_name)
        except ImportError as exc:
            log.info("Skipping tools from %s (%s)", module_name, exc)
            continue
        register = getattr(module, "register", None)
        if register is None:
            log.warning("Tool module %s has no register(registry, ctx)", module_name)
            continue
        try:
            register(registry, ctx)
        except Exception:
            log.exception("Tool module %s failed to register", module_name)
    return registry
