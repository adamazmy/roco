"""The Claude backend of the brain (Claude API from the PC), with native tool use.

The default brain since Adam's 2026-11-22 decision (docs/ARCHITECTURE.md): Claude Haiku
for normal turns, a larger Claude model when the :class:`~roco.brain.router.BrainRouter`
escalates a hard question. Everything model-independent (persona, history, voice
commands, time prefetch, reminder repair, barge-in) comes from
:class:`~roco.brain.core.ToolBrain`; this class only runs the tool loop on the
Messages API through workstream D's :class:`~roco.fallback.claude.ClaudeClient`
(streaming, wall-clock deadlines, usage logging to ``~/.roco/cloud_usage.jsonl``).

Request layout, ordered so the prompt cache hits (render order is tools -> system ->
messages; any byte change invalidates everything after it):

1. ``tools``: the registry's schemas in registry order, ``cache_control`` on the last one;
2. ``system``: the persona (roco.personality's render + date), ``cache_control`` on it;
3. ``messages``: the history (converted from the neutral shape, tool calls replayed as
   ``tool_use`` / ``tool_result`` with ids derived from their position, so the same
   history always gives the same bytes), then the new user turn, ``cache_control`` on the
   last block so the next tool round and the next turn read the whole prefix from cache.

Within a turn, each round's assistant content (thinking blocks included) is sent back
unchanged, as the API requires; earlier turns are replayed from the history without
their thinking blocks.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable
from datetime import datetime
from typing import Any

from roco.brain.core import Cancelled, Reply, ToolBrain, TurnControl
from roco.brain.emotion import EmotionTagParser
from roco.brain.history import History
from roco.bus import EventBus
from roco.config import Config
from roco.fallback.claude import ClaudeClient, CloudError, CloudPermanentError
from roco.personality import Personality
from roco.tools.registry import ToolRegistry

log = logging.getLogger(__name__)

DEFAULT_MODEL = "claude-haiku-5-5"
DEFAULT_ESCALATE_MODEL = "claude-sonnet-5-5"
CACHE = {"type": "ephemeral"}
APOLOGY_CLOUD = "Sorry, my cloud brain isn't answering just now. Try me again in a moment?"
APOLOGY_CLOUD_KEY = "Sorry, my cloud brain won't let me in. The API key needs sorting out."
REFUSAL_LINE = "I'll pass on that one, if it's all the same to you."


class ClaudeBrain(ToolBrain):
    backend_errors = (CloudError,)
    mask_old_times = False  # Claude reads an old time with its age correctly

    def __init__(
        self,
        config: Config,
        bus: EventBus,
        registry: ToolRegistry | None = None,
        *,
        client: ClaudeClient,
        history: History | None = None,
        clock: Callable[[], datetime] | None = None,
        personality: Personality | None = None,
        raise_backend_errors: bool = False,
    ) -> None:
        super().__init__(
            config, bus, registry, history=history, clock=clock, personality=personality,
            raise_backend_errors=raise_backend_errors,
        )
        self.client = client
        c = config.get_path("brain.claude", {}) or {}
        self.model = str(c.get("model") or DEFAULT_MODEL)
        self.escalate_model = str(c.get("escalate_model") or DEFAULT_ESCALATE_MODEL)
        self.effort = c.get("effort") or None
        self.escalate_effort = c.get("escalate_effort") or None
        self.thinking = str(c.get("thinking") or "adaptive")
        self.max_tokens = int(c.get("max_tokens", 1024))
        self.escalate_max_tokens = int(c.get("escalate_max_tokens", 4096))
        self.first_text_timeout_s = float(c.get("first_text_timeout_s", 6.0))
        # Set by the router for one turn: answer on the larger model.
        self.escalated = False
        self.last_usage: list[dict[str, Any]] = []  # per request of the last turn (tests, logs)

    @property
    def name(self) -> str:
        return f"claude:{self.current_model}"

    @property
    def current_model(self) -> str:
        return self.escalate_model if self.escalated else self.model

    def apology_for(self, error: BaseException) -> str:
        return APOLOGY_CLOUD_KEY if isinstance(error, CloudPermanentError) else APOLOGY_CLOUD

    def respond_turn(self, text, on_sentence=None, *, cancel=None, hooks=None) -> str:
        self.last_usage = []
        return super().respond_turn(text, on_sentence, cancel=cancel, hooks=hooks)

    # ------------------------------------------------------------------ request
    def request_params(
        self, messages: list[dict[str, Any]], *, final_round: bool = False
    ) -> dict[str, Any]:
        """Messages API parameters for the neutral ``messages`` (system first)."""
        schemas = self.allowed_schemas()
        tools = anthropic_tools(schemas)
        system, convo = to_anthropic(messages, with_tools=bool(tools))
        if convo:
            _mark_last_block(convo[-1])
        params: dict[str, Any] = {
            "model": self.current_model,
            "max_tokens": self.escalate_max_tokens if self.escalated else self.max_tokens,
            "system": [{"type": "text", "text": system, "cache_control": CACHE}],
            "messages": convo,
        }
        if tools:
            params["tools"] = tools
            if final_round:
                # tool_use blocks in the history need the tools defined; "none" forces words.
                params["tool_choice"] = {"type": "none"}
        effort = self.escalate_effort if self.escalated else self.effort
        if effort:
            params["output_config"] = {"effort": effort}
        if self.thinking in ("adaptive", "disabled"):
            params["thinking"] = {"type": self.thinking}
        return params

    def _converse(
        self,
        messages: list[dict[str, Any]],
        reply: Reply,
        turn: int,
        ctl: TurnControl | None = None,
        trace_from: int | None = None,
    ) -> list[dict[str, Any]]:
        ctl = ctl or TurnControl(None, None)
        start = len(messages) if trace_from is None else trace_from
        # This turn's rounds in the API's own shape (thinking blocks kept, sent back
        # unchanged); `messages` gets the neutral copy for the history.
        rounds: list[dict[str, Any]] = []
        head = len(messages)  # system, history, user turn (+ prefetched time): sent as is
        round_no = 0
        retried = False
        while round_no <= self.max_tool_rounds:
            final_round = round_no >= self.max_tool_rounds
            params = self.request_params(messages[:head], final_round=final_round)
            if rounds:
                params["messages"] = _unmark(params["messages"] + rounds)
                _mark_last_block(params["messages"][-1])
            result = self._stream_round(params, reply, ctl)
            message = result.message
            blocks = [_block_dict(b) for b in (getattr(message, "content", None) or [])]
            calls = [b for b in blocks if b.get("type") == "tool_use"]
            said = "".join(b.get("text", "") for b in blocks if b.get("type") == "text")
            if getattr(message, "stop_reason", None) == "refusal" and not reply.sentences:
                reply.say(REFUSAL_LINE)
                break
            if not calls and not reply.sentences and not retried:
                log.warning("[turn %d] Empty Claude answer; retrying once", turn)
                retried = True
                continue
            round_no += 1
            if not calls or final_round:
                break
            rounds.append({"role": "assistant", "content": blocks})
            neutral_calls = [{"function": {"name": c["name"], "arguments": c.get("input") or {}}}
                             for c in calls]
            messages.append({"role": "assistant", "content": said, "tool_calls": neutral_calls})
            ctl.tools([c["name"] for c in calls])
            results = []
            for call in calls:
                if ctl.cancelled:
                    output = "Cancelled: not run."
                else:
                    output = self.run_tool(call.get("name"), call.get("input") or {})
                    log.debug("[turn %d] tool %s -> %.200s", turn, call.get("name"), output)
                results.append({"type": "tool_result", "tool_use_id": call["id"],
                                "content": output})
                messages.append({"role": "tool", "content": output,
                                 "tool_name": call.get("name") or ""})
            rounds.append({"role": "user", "content": results})
            ctl.check()
        return messages[start:]

    def _stream_round(self, params: dict[str, Any], reply: Reply, ctl: TurnControl) -> Any:
        parser = EmotionTagParser()

        def push(text: str) -> None:
            # The tag (or its absence) is published before the first words.
            if parser.decided and (text.strip() or parser.emotion is not None):
                reply.set_emotion(parser.emotion)
            reply.feed(text)

        def on_text(piece: str) -> None:
            ctl.token()
            was_decided = parser.decided
            out = parser.feed(piece)
            if out or (parser.decided and not was_decided):
                push(out)

        cancel = ctl.cancel or threading.Event()
        result = self.client.stream_message(
            params, on_text, cancel, purpose="brain",
            first_text_timeout_s=self.first_text_timeout_s,
        )
        self.last_usage.append({"model": params["model"], **vars(result.usage)})
        if result.cancelled or ctl.cancelled:
            reply.drop_partial()
            raise Cancelled
        if not parser.decided:
            push(parser.flush())
        reply.flush()
        return result


# --------------------------------------------------------------------------- conversion


def anthropic_tools(schemas: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Ollama/OpenAI-style function schemas -> Anthropic tools, cache mark on the last."""
    tools = []
    for s in schemas:
        f = s.get("function", s)
        tools.append({
            "name": f["name"],
            "description": f.get("description", ""),
            "input_schema": f.get("parameters") or {"type": "object", "properties": {}},
        })
    if tools:
        tools[-1] = {**tools[-1], "cache_control": CACHE}
    return tools


def to_anthropic(
    messages: list[dict[str, Any]], *, with_tools: bool = True, id_prefix: str = "h"
) -> tuple[str, list[dict[str, Any]]]:
    """Neutral messages -> (system text, Anthropic ``messages``).

    Tool calls get ids from their position (``toolu_h0003``), so converting the same
    history twice gives identical bytes. A call with no result gets "Cancelled: not run."
    (the API rejects a ``tool_use`` without its ``tool_result``). Without tools, the tool
    exchanges are folded into plain text (the API needs tools defined to replay them).
    """
    system_parts: list[str] = []
    out: list[dict[str, Any]] = []
    pending: list[tuple[str, str]] = []  # (id, name) of calls waiting for a result
    counter = 0

    def add(role: str, blocks: list[dict[str, Any]]) -> None:
        if not blocks:
            return
        if out and out[-1]["role"] == role:
            out[-1]["content"].extend(blocks)
        else:
            out.append({"role": role, "content": list(blocks)})

    def close_pending() -> None:
        if pending:
            add("user", [{"type": "tool_result", "tool_use_id": i, "content": "Cancelled: not run."}
                         for i, _ in pending])
            pending.clear()

    for m in messages:
        role = m.get("role")
        content = m.get("content") or ""
        if role == "system":
            system_parts.append(str(content))
            continue
        if role == "tool":
            if not with_tools:
                add("user", [_text(f"(Result of {m.get('tool_name') or 'a tool'}: {content})")])
                continue
            name = m.get("tool_name") or ""
            index = next((k for k, (_, n) in enumerate(pending) if n == name), 0 if pending else -1)
            if index < 0:
                add("user", [_text(f"(Result of {name or 'a tool'}: {content})")])
                continue
            call_id, _ = pending.pop(index)
            add("user", [{"type": "tool_result", "tool_use_id": call_id,
                          "content": str(content) or "(empty)"}])
            continue
        close_pending()
        if role == "assistant":
            blocks = [_text(content)] if str(content).strip() else []
            for call in m.get("tool_calls") or []:
                fn = call.get("function") if isinstance(call, dict) else None
                if not isinstance(fn, dict) or not fn.get("name"):
                    continue
                args = fn.get("arguments")
                if not isinstance(args, dict):
                    args = {}
                if not with_tools:
                    blocks.append(_text(f"(Called {fn['name']}.)"))
                    continue
                call_id = f"toolu_{id_prefix}{counter:04d}"
                counter += 1
                pending.append((call_id, str(fn["name"])))
                blocks.append({"type": "tool_use", "id": call_id, "name": fn["name"],
                               "input": args})
            add("assistant", blocks)
        elif role == "user" and str(content).strip():
            add("user", [_text(content)])
    close_pending()
    while out and out[0]["role"] != "user":
        out.pop(0)
    return "\n\n".join(system_parts), out


def _text(text: Any) -> dict[str, Any]:
    return {"type": "text", "text": str(text)}


def _mark_last_block(message: dict[str, Any]) -> None:
    content = message.get("content")
    if isinstance(content, str):
        message["content"] = [{"type": "text", "text": content, "cache_control": CACHE}]
    elif isinstance(content, list) and content:
        content[-1] = {**content[-1], "cache_control": CACHE}


def _unmark(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Copy of ``messages`` with no cache marks (at most 4 are allowed per request)."""
    out = []
    for m in messages:
        content = m.get("content")
        if isinstance(content, list):
            content = [{k: v for k, v in b.items() if k != "cache_control"} for b in content]
        out.append({**m, "content": content})
    return out


def _block_dict(block: Any) -> dict[str, Any]:
    """An SDK content block (or a dict) as a plain dict to send back unchanged."""
    if isinstance(block, dict):
        return dict(block)
    for attr in ("to_dict", "model_dump"):
        fn = getattr(block, attr, None)
        if callable(fn):
            try:
                data = fn(exclude_none=True) if attr == "model_dump" else fn()
            except TypeError:
                data = fn()
            if isinstance(data, dict):
                return {k: v for k, v in data.items() if v is not None}
    return {k: v for k, v in vars(block).items() if not k.startswith("_") and v is not None}


def create_claude_client(cfg: Config, api_key: str | None, sdk: Any = None) -> ClaudeClient:
    """D's client with the brain's deadlines (``brain.claude``) and the shared usage log."""
    from roco.fallback.usage import UsageLog

    cloud = cfg.section("cloud")
    c = cfg.get_path("brain.claude", {}) or {}
    return ClaudeClient(
        api_key,
        model=str(c.get("model") or DEFAULT_MODEL),
        effort=None,
        max_tokens=int(c.get("max_tokens", 1024)),
        connect_timeout_s=float(cloud.get("connect_timeout_s", 3.0)),
        read_timeout_s=float(cloud.get("read_timeout_s", 8.0)),
        first_text_timeout_s=float(c.get("first_text_timeout_s", 6.0)),
        total_timeout_s=float(c.get("total_timeout_s", 30.0)),
        max_retries=int(cloud.get("max_retries", 1)),
        usage_log=UsageLog.from_config(cfg),
        client=sdk,
    )
