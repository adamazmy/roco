"""Decorator-based tool registry with explicit JSON schemas.

Tools are plain Python functions. The schema is written by hand (small local
models do better with short, explicit schemas than with inferred ones)::

    registry = ToolRegistry()

    @registry.tool(
        description="Get the current local date and time.",
        parameters={"type": "object", "properties": {}, "required": []},
    )
    def get_current_time() -> str:
        ...

:meth:`ToolRegistry.call` never raises: unknown tools, bad arguments and
exceptions inside the tool all come back as an ``"Error: ..."`` string so the
model can recover (apologise, retry with other arguments) instead of crashing
the conversation.
"""

from __future__ import annotations

import inspect
import json
import logging
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from typing import Any

log = logging.getLogger(__name__)

EMPTY_PARAMETERS: dict[str, Any] = {"type": "object", "properties": {}, "required": []}


@dataclass(frozen=True)
class Tool:
    name: str
    description: str
    func: Callable[..., Any]
    parameters: dict[str, Any] = field(default_factory=lambda: dict(EMPTY_PARAMETERS))

    def schema(self) -> dict[str, Any]:
        """Ollama ``/api/chat`` tool format (same shape as OpenAI function tools)."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    # ----------------------------------------------------------------- registration
    def register(
        self,
        func: Callable[..., Any],
        *,
        name: str | None = None,
        description: str | None = None,
        parameters: dict[str, Any] | None = None,
    ) -> Tool:
        tool_name = name or func.__name__
        if tool_name in self._tools:
            raise ValueError(f"Tool {tool_name!r} is already registered")
        desc = description or inspect.getdoc(func) or ""
        if not desc:
            raise ValueError(f"Tool {tool_name!r} needs a description")
        params = parameters if parameters is not None else dict(EMPTY_PARAMETERS)
        if params.get("type") != "object":
            raise ValueError(f"Tool {tool_name!r}: parameters must be a JSON schema object")
        tool = Tool(name=tool_name, description=desc.strip(), func=func, parameters=params)
        self._tools[tool_name] = tool
        return tool

    def tool(
        self,
        name: str | None = None,
        *,
        description: str | None = None,
        parameters: dict[str, Any] | None = None,
    ) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        """Decorator form of :meth:`register`. Returns the function unchanged."""

        def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
            self.register(func, name=name, description=description, parameters=parameters)
            return func

        return decorator

    def unregister(self, name: str) -> None:
        self._tools.pop(name, None)

    # ----------------------------------------------------------------- lookup
    def __contains__(self, name: object) -> bool:
        return name in self._tools

    def __len__(self) -> int:
        return len(self._tools)

    def __iter__(self) -> Iterator[Tool]:
        return iter(self._tools.values())

    def names(self) -> list[str]:
        return list(self._tools)

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def schemas(self) -> list[dict[str, Any]]:
        return [t.schema() for t in self._tools.values()]

    # ----------------------------------------------------------------- execution
    def call(self, name: str, arguments: dict[str, Any] | str | None = None) -> str:
        """Run a tool and return its result as text. Never raises."""
        tool = self._tools.get(name)
        if tool is None:
            available = ", ".join(self._tools) or "none"
            return f"Error: unknown tool {name!r}. Available tools: {available}."

        try:
            args = _parse_arguments(arguments)
        except ValueError as exc:
            return f"Error: invalid arguments for {name}: {exc}"

        required = tool.parameters.get("required", [])
        missing = [r for r in required if r not in args]
        if missing:
            return f"Error: {name} is missing required argument(s): {', '.join(missing)}."
        known = tool.parameters.get("properties", {})
        if known:
            # Small models sometimes invent extra arguments; drop them rather than fail.
            dropped = [k for k in args if k not in known]
            if dropped:
                log.debug("Dropping unknown arguments for %s: %s", name, dropped)
                args = {k: v for k, v in args.items() if k in known}

        try:
            inspect.signature(tool.func).bind(**args)
        except TypeError as exc:
            log.warning("Tool %s called with bad arguments %r: %s", name, args, exc)
            return f"Error: bad arguments for {name}: {exc}"

        try:
            result = tool.func(**args)
        except Exception as exc:
            log.exception("Tool %s failed", name)
            return f"Error: {name} failed: {exc}"
        return _to_text(result)


def _parse_arguments(arguments: dict[str, Any] | str | None) -> dict[str, Any]:
    if arguments is None or arguments == "":
        return {}
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except json.JSONDecodeError as exc:
            raise ValueError(f"not valid JSON ({exc.msg})") from exc
    if not isinstance(arguments, dict):
        raise ValueError("expected an object of named arguments")
    return arguments


def _to_text(result: Any) -> str:
    if result is None:
        return "Done."
    if isinstance(result, str):
        return result
    try:
        return json.dumps(result, ensure_ascii=False, default=str)
    except (TypeError, ValueError):
        return str(result)
