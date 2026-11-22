"""The Ollama backend of the brain: qwen2.5 on the PC, streamed, with native tools.

Everything model-independent (persona, history, voice commands, time prefetch, reminder
repair, barge-in) lives in :class:`roco.brain.core.ToolBrain`. This class runs the tool
loop against Ollama: while the model asks for tools (at most ``llm.max_tool_rounds``
rounds) it runs them and sends the results back as ``tool`` messages, a last round
without tools forces a spoken answer, and the streamed text goes through
:class:`EmotionTagParser` and the sentence splitter. Text tool calls written into the
reply as JSON (a qwen2.5 habit) are recovered instead of being spoken.

On barge-in the stream is closed at the next chunk (Ollama stops generating). Network or
model failures become :class:`OllamaError`; on its own the brain turns them into a short
British apology, and under the :class:`~roco.brain.router.BrainRouter` they are raised so
the router can answer on the other brain or say the offline line.
"""

from __future__ import annotations

import json
import logging
import re
import time
from collections.abc import Callable
from datetime import datetime
from typing import Any

from roco.brain.core import (
    ACTION_CLAIM,
    APOLOGY_GENERIC,
    APOLOGY_TANGLED,
    REMINDER_REPAIR_FAILED,
    REMINDER_REPAIR_NOTE,
    REMINDER_REQUEST,
    TIME_QUESTION,
    Cancelled,
    Reply,
    ToolBrain,
    TurnControl,
)
from roco.brain.emotion import EmotionTagParser
from roco.brain.history import History
from roco.brain.ollama import OllamaClient, OllamaError
from roco.bus import EventBus
from roco.config import Config
from roco.personality import Personality
from roco.tools.registry import ToolRegistry

log = logging.getLogger(__name__)

__all__ = [
    "ACTION_CLAIM", "APOLOGY_GENERIC", "APOLOGY_TANGLED", "APOLOGY_TIMEOUT",
    "APOLOGY_UNREACHABLE", "REMINDER_REPAIR_FAILED", "REMINDER_REPAIR_NOTE",
    "REMINDER_REQUEST", "TIME_QUESTION", "OllamaBrain", "apology_for", "create_brain",
]

APOLOGY_UNREACHABLE = "Sorry, I can't reach my brain at the moment. Is Ollama running?"
APOLOGY_TIMEOUT = "Sorry, I'm thinking far too slowly just now. Could you try again in a moment?"


def apology_for(error: OllamaError, model: str) -> str:
    if error.kind == "unreachable":
        return APOLOGY_UNREACHABLE
    if error.kind == "timeout":
        return APOLOGY_TIMEOUT
    if error.kind == "model_missing":
        spoken = model.replace(":", " ")
        return f"Sorry, I can't find the {spoken} model. It needs pulling in Ollama first."
    return APOLOGY_GENERIC


class OllamaBrain(ToolBrain):
    backend_errors = (OllamaError,)

    def __init__(
        self,
        config: Config,
        bus: EventBus,
        registry: ToolRegistry | None = None,
        *,
        client: OllamaClient | None = None,
        history: History | None = None,
        clock: Callable[[], datetime] | None = None,
        history_tags: bool = True,
        history_tools: bool = True,
        personality: Personality | None = None,
        raise_backend_errors: bool = False,
    ) -> None:
        super().__init__(
            config, bus, registry, history=history, clock=clock, history_tags=history_tags,
            history_tools=history_tools, personality=personality,
            raise_backend_errors=raise_backend_errors,
        )
        llm = config.section("llm")
        self.model = str(llm.get("model", "qwen2.5:7b"))
        self.client = client or OllamaClient(
            str(llm.get("host", "http://localhost:11434")),
            self.model,
            temperature=llm.get("temperature"),
            num_ctx=llm.get("num_ctx"),
            keep_alive=llm.get("keep_alive"),
            extra_options=llm.get("options") or None,
            timeout_s=float(llm.get("timeout_s", 120)),
        )

    @property
    def name(self) -> str:
        return f"ollama:{self.model}"

    def apology_for(self, error: BaseException) -> str:
        if isinstance(error, OllamaError):
            return apology_for(error, self.model)
        return APOLOGY_GENERIC

    def warm_up(self) -> bool:
        """Send the real system prompt and tools once, before Adam's first question.

        Measured on the RTX 3060 laptop: right after a load, the first chat took 4.6 s to
        its first token, later ones ~1 s (prompt processing of ~1,700 tokens + GPU warm-up).
        Ollama then reuses the cached prompt prefix. Best effort: False on failure.
        """
        tools = self.allowed_schemas() or None
        messages = [self.system_message(), {"role": "user", "content": "Hi."}]
        try:
            t0 = time.monotonic()
            self.client.warm_up(messages, tools)
        except OllamaError as exc:
            log.warning("LLM warm-up failed: %s", exc)
            return False
        log.info("LLM warmed up in %d ms", (time.monotonic() - t0) * 1000)
        return True

    def close(self) -> None:
        self.client.close()

    def _converse(
        self,
        messages: list[dict[str, Any]],
        reply: Reply,
        turn: int,
        ctl: TurnControl | None = None,
        trace_from: int | None = None,
    ) -> list[dict[str, Any]]:
        """Run the tool loop. Returns this turn's tool-call and tool-result messages
        (from ``trace_from``, default: what this call adds)."""
        ctl = ctl or TurnControl(None, None)
        start = len(messages) if trace_from is None else trace_from
        # Tools that read Adam's data aloud are hidden while guests are in the room.
        schemas = self.allowed_schemas() or None
        retried = False
        round_no = 0
        while round_no <= self.max_tool_rounds:
            # The last round goes without tools so the model has to answer in words.
            tools = schemas if round_no < self.max_tool_rounds else None
            content, tool_calls = self._stream_round(messages, tools, reply, ctl)
            if not tool_calls and not reply.sentences and not retried:
                # qwen2.5 occasionally ends a turn with no text and no tool call (or a bare
                # tag); one retry nearly always gets a real answer.
                log.warning("[turn %d] Empty answer (raw %r); retrying once", turn, content[:80])
                retried = True
                continue
            round_no += 1
            if not tool_calls:
                break
            if tools is None:
                log.warning("[turn %d] Model asked for tools after the last round; ignored", turn)
                break
            messages.append(
                {
                    "role": "assistant",
                    "content": content,
                    "tool_calls": [_sanitize_call(c) for c in tool_calls],
                }
            )
            ctl.tools([_call_parts(c)[0] or "" for c in tool_calls])
            for call in tool_calls:
                name, arguments = _call_parts(call)
                if ctl.cancelled:
                    # Every call needs a result, or the replayed history confuses the model.
                    messages.append(
                        {"role": "tool", "content": "Cancelled: not run.", "tool_name": name or ""}
                    )
                    continue
                result = self.run_tool(name, arguments)
                log.debug("[turn %d] tool %s(%s) -> %.200s", turn, name, _short(arguments), result)
                messages.append({"role": "tool", "content": result, "tool_name": name or ""})
            ctl.check()
        return messages[start:]

    def _stream_round(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
        reply: Reply,
        ctl: TurnControl | None = None,
    ) -> tuple[str, list[Any]]:
        ctl = ctl or TurnControl(None, None)
        parser = EmotionTagParser()
        raw: list[str] = []
        tool_calls: list[Any] = []
        # Text from the first "{" / "<tool_call>" on is held back: qwen2.5 sometimes writes
        # a tool call as JSON in its reply instead of a native call, and it must not be spoken.
        held: list[str] = []

        def push(text: str) -> None:
            if parser.decided and (text.strip() or parser.emotion is not None):
                reply.set_emotion(parser.emotion)
            if held:
                held.append(text)
                return
            cut = _json_start(text) if tools else -1
            if cut >= 0:
                held.append(text[cut:])
                text = text[:cut]
            reply.feed(text)

        stream = self.client.stream_chat(messages, tools)
        try:
            for chunk in stream:
                if ctl.cancelled:
                    break
                message = chunk.get("message") or {}
                piece = message.get("content") or ""
                if piece:
                    ctl.token()
                    raw.append(piece)
                    was_decided = parser.decided
                    out = parser.feed(piece)
                    if out or (parser.decided and not was_decided):
                        push(out)
                calls = message.get("tool_calls")
                if isinstance(calls, list):
                    tool_calls.extend(calls)
        finally:
            close = getattr(stream, "close", None)
            if callable(close):
                close()  # closes the HTTP response early: Ollama stops generating
        if ctl.cancelled:
            reply.drop_partial()
            raise Cancelled
        if not parser.decided:
            push(parser.flush())
        if held:
            text_calls, leftover = _text_tool_calls("".join(held), set(self.registry.names()))
            if text_calls and not tool_calls:
                log.info("Recovered %d tool call(s) written as text", len(text_calls))
                tool_calls = text_calls
            reply.feed(leftover)
        # Text spoken before a tool call ("Let me check.") is delivered now, in order.
        reply.flush()
        return "".join(raw), tool_calls



_TOOL_CALL_TAGS = re.compile(r"</?tool_call>", re.IGNORECASE)


def _json_start(text: str) -> int:
    """Index where a tool call written as text may begin, or -1.

    Any "<" counts, since a "<tool_call>" tag can arrive split across chunks; if the
    held text turns out not to be a tool call it is spoken anyway.
    """
    hits = [i for i in (text.find("{"), text.find("<")) if i >= 0]
    return min(hits) if hits else -1


def _text_tool_calls(text: str, known: set[str]) -> tuple[list[dict[str, Any]], str]:
    """Pull ``{"name": <known tool>, "arguments": {...}}`` objects out of reply text.

    Returns (tool calls, the text that is left to speak). Text that is not a
    tool call comes back unchanged with no calls, so it is spoken as normal.
    """
    cleaned = _TOOL_CALL_TAGS.sub(" ", text)
    decoder = json.JSONDecoder()
    calls: list[dict[str, Any]] = []
    spans: list[tuple[int, int]] = []
    i = cleaned.find("{")
    while i >= 0:
        try:
            obj, end = decoder.raw_decode(cleaned, i)
        except json.JSONDecodeError:
            i = cleaned.find("{", i + 1)
            continue
        if isinstance(obj, dict) and obj.get("name") in known:
            arguments = obj.get("arguments", obj.get("parameters", {}))
            calls.append({"function": {"name": obj["name"], "arguments": arguments}})
            spans.append((i, end))
        i = cleaned.find("{", end)
    if not calls:
        return [], text
    leftover, last = [], 0
    for start, end in spans:
        leftover.append(cleaned[last:start])
        last = end
    leftover.append(cleaned[last:])
    return calls, " ".join(part.strip() for part in leftover if part.strip())


def _call_parts(call: Any) -> tuple[str | None, dict[str, Any] | str | None]:
    function = call.get("function") if isinstance(call, dict) else None
    if not isinstance(function, dict):
        return None, None
    name = function.get("name")
    if not isinstance(name, str) or not name.strip():
        return None, None
    return name.strip(), function.get("arguments")


def _sanitize_call(call: Any) -> dict[str, Any]:
    """Echo a tool call back to Ollama in a shape it accepts (arguments must be an object)."""
    name, arguments = _call_parts(call)
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except json.JSONDecodeError:
            arguments = None
    if not isinstance(arguments, dict):
        arguments = {}
    return {"function": {"name": name or "unknown", "arguments": arguments}}


def _short(arguments: Any) -> str:
    try:
        return json.dumps(arguments, ensure_ascii=False)[:120]
    except (TypeError, ValueError):
        return repr(arguments)[:120]


def create_brain(
    config: Config, bus: EventBus, registry: ToolRegistry | None = None
) -> OllamaBrain:
    """The brain ``roco chat`` and ``roco run`` use.

    Without ``registry``, the standard tools are built (and the reminder scheduler started).
    """
    if registry is None:
        from roco.tools import ToolContext, build_registry

        registry = build_registry(ToolContext(config=config, bus=bus))
    return OllamaBrain(config, bus, registry)
