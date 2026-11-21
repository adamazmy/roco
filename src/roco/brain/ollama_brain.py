"""The Ollama-backed :class:`~roco.brain.base.Brain`: persona, history and the tool-call loop.

One call to :meth:`OllamaBrain.respond`:

1. publishes ``State.THINKING``;
2. sends system prompt + history + the user turn to Ollama (streamed, native tools);
3. while the model asks for tools (at most ``llm.max_tool_rounds`` rounds), runs
   them through :meth:`ToolRegistry.call` and sends the results back as ``tool``
   messages; a last round without tools forces a spoken answer if the limit is hit;
4. streams the answer through :class:`EmotionTagParser` (tag -> ``bus.set_emotion``)
   and :class:`SentenceSplitter` (sentences -> ``on_sentence``);
5. stores the user turn and the tag-free answer in the history.

Any network or model failure becomes a short British apology instead of an exception.

:meth:`OllamaBrain.respond_turn` adds barge-in support: a set ``cancel`` event stops the
stream at the next chunk (closing the HTTP stream makes Ollama stop generating), skips
the remaining tools and sentences, and stores what was said so far, so
:meth:`OllamaBrain.amend_last_reply` can replace it with what was actually heard.
"""

from __future__ import annotations

import itertools
import json
import logging
import re
import threading
from collections.abc import Callable
from datetime import datetime
from typing import Any

from roco.brain.base import Brain, SentenceCallback, TurnHooks
from roco.brain.emotion import EmotionTagParser
from roco.brain.history import History
from roco.brain.ollama import OllamaClient, OllamaError
from roco.brain.persona import system_prompt
from roco.brain.sentences import FIRST_CLAUSE_MIN_CHARS, SentenceSplitter
from roco.bus import Emotion, EventBus, State
from roco.config import Config
from roco.paths import resolve
from roco.tools.clock import describe_date, local_timezone
from roco.tools.registry import ToolRegistry

log = logging.getLogger(__name__)

_turn_ids = itertools.count(1)

APOLOGY_UNREACHABLE = "Sorry, I can't reach my brain at the moment. Is Ollama running?"
APOLOGY_TIMEOUT = "Sorry, I'm thinking far too slowly just now. Could you try again in a moment?"
APOLOGY_GENERIC = "Sorry, something went wrong in my head. Could you try that again?"

# "What time is it?", "what's the time now", "tell me the time": a plain question about
# the current time, answered from a prefetched get_current_time result.
TIME_QUESTION = re.compile(
    r"\b(what\s+time\s+is\s+it|what(?:'s|\s+is)\s+the\s+time|tell\s+me\s+the\s+time"
    r"|the\s+time\s+now|current\s+time)\b",
    re.IGNORECASE,
)
APOLOGY_TANGLED = "Sorry, I got a bit tangled up there. Could you ask me that again?"


def apology_for(error: OllamaError, model: str) -> str:
    if error.kind == "unreachable":
        return APOLOGY_UNREACHABLE
    if error.kind == "timeout":
        return APOLOGY_TIMEOUT
    if error.kind == "model_missing":
        spoken = model.replace(":", " ")
        return f"Sorry, I can't find the {spoken} model. It needs pulling in Ollama first."
    return APOLOGY_GENERIC


def _splitter(first: bool = True) -> SentenceSplitter:
    """A short first chunk gets the voice going sooner (TTS time grows with length)."""
    return SentenceSplitter(first_clause_min_chars=FIRST_CLAUSE_MIN_CHARS if first else None)


class _Cancelled(Exception):
    """The turn's cancel event was set (barge-in)."""


_NO_HOOKS = TurnHooks()


class _Turn:
    """Per-call cancel event and hooks, passed down the tool loop."""

    def __init__(self, cancel: threading.Event | None, hooks: TurnHooks | None) -> None:
        self.cancel = cancel
        self.hooks = hooks or _NO_HOOKS

    @property
    def cancelled(self) -> bool:
        return self.cancel is not None and self.cancel.is_set()

    def check(self) -> None:
        if self.cancelled:
            raise _Cancelled

    def token(self) -> None:
        if self.hooks.on_token is not None:
            try:
                self.hooks.on_token()
            except Exception:
                log.exception("on_token hook failed")

    def tools(self, names: list[str]) -> None:
        if self.hooks.on_tools is not None:
            try:
                self.hooks.on_tools(names)
            except Exception:
                log.exception("on_tools hook failed")


class _Reply:
    """Collects one reply: publishes the emotion first, then delivers sentences."""

    def __init__(
        self,
        bus: EventBus,
        on_sentence: SentenceCallback | None,
        cancel: threading.Event | None = None,
    ) -> None:
        self._bus = bus
        self._on_sentence = on_sentence
        self._cancel = cancel
        self._splitter = _splitter()
        self._emotion: Emotion | None = None
        self.first_emotion: Emotion | None = None
        self.sentences: list[str] = []

    @property
    def text(self) -> str:
        return " ".join(self.sentences)

    def set_emotion(self, emotion: Emotion | None) -> None:
        """Publish ``emotion``; a missing tag means neutral, unless one is already shown."""
        if emotion is None:
            if self._emotion is not None:
                return
            emotion = Emotion.NEUTRAL
        if emotion != self._emotion:
            self._emotion = emotion
            self.first_emotion = self.first_emotion or emotion
            self._bus.set_emotion(emotion)

    def feed(self, text: str) -> None:
        for sentence in self._splitter.feed(text):
            self._deliver(sentence)

    def flush(self) -> None:
        rest = self._splitter.flush()
        if rest:
            self._deliver(rest)

    def drop_partial(self) -> None:
        """Forget an unfinished sentence (the stream broke mid-way)."""
        self._splitter = _splitter(first=not self.sentences)

    def say(self, sentence: str) -> None:
        """Deliver a whole sentence of our own (apologies)."""
        self._deliver(sentence)

    def _deliver(self, sentence: str) -> None:
        if self._cancel is not None and self._cancel.is_set():
            return  # barge-in: nothing more is said or stored
        if self._emotion is None:
            self.set_emotion(None)
        self.sentences.append(sentence)
        if self._on_sentence is None:
            return
        try:
            self._on_sentence(sentence)
        except Exception:
            log.exception("on_sentence callback failed")


class OllamaBrain(Brain):
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
    ) -> None:
        self.config = config
        self.bus = bus
        self.registry = registry or ToolRegistry()
        llm = config.section("llm")
        self.model = str(llm.get("model", "qwen2.5:7b"))
        self.max_tool_rounds = max(0, int(llm.get("max_tool_rounds", 5)))
        self.client = client or OllamaClient(
            str(llm.get("host", "http://localhost:11434")),
            self.model,
            temperature=llm.get("temperature"),
            num_ctx=llm.get("num_ctx"),
            keep_alive=llm.get("keep_alive"),
            extra_options=llm.get("options") or None,
            timeout_s=float(llm.get("timeout_s", 120)),
        )
        if history is None:
            hist_cfg = config.section("history")
            persist = bool(hist_cfg.get("persist"))
            path = resolve(hist_cfg.get("file", "history.json")) if persist else None
            history = History(int(hist_cfg.get("max_turns", 12)), path)
        self.history = history
        self._tz = local_timezone(config)
        self._clock = clock or (lambda: datetime.now(self._tz))
        # Send past replies with their tag and tool trace (see roco.brain.history).
        self.history_tags = history_tags
        self.history_tools = history_tools
        self._lock = threading.Lock()

    # ------------------------------------------------------------------ Brain API
    def respond(self, text: str, on_sentence: SentenceCallback | None = None) -> str:
        return self.respond_turn(text, on_sentence)

    def respond_turn(
        self,
        text: str,
        on_sentence: SentenceCallback | None = None,
        *,
        cancel: threading.Event | None = None,
        hooks: TurnHooks | None = None,
    ) -> str:
        ctl = _Turn(cancel, hooks)
        with self._lock:
            turn = next(_turn_ids)
            self.bus.set_state(State.THINKING)
            reply = _Reply(self.bus, on_sentence, cancel)
            text = (text or "").strip()
            if not text:
                return ""
            started = self._clock()
            past = self.history.for_model(
                tags=self.history_tags, tools=self.history_tools, now=started
            )
            messages = [self._system_message(), *past]
            # No current time is added here: tried after the user text, it made qwen2.5 skip
            # tools far more often (2/4 vs 4/4 scenario runs). Time comes from the tool.
            messages.append({"role": "user", "content": text})
            base = len(messages)
            messages.extend(self._prefetch_time(text, turn))
            try:
                ctl.check()
                trace = self._converse(messages, reply, turn, ctl, trace_from=base)
            except _Cancelled:
                # Keep what was said and the tool calls already made; the server then
                # replaces the reply with what was heard (amend_last_reply).
                log.info("[turn %d] cancelled after %d sentence(s)", turn, len(reply.sentences))
                emotion = reply.first_emotion.value if reply.first_emotion else None
                self.history.add_turn(text, reply.text, emotion, messages[base:], at=started)
                return reply.text
            except OllamaError as exc:
                log.warning("[turn %d] Ollama failed (%s): %s", turn, exc.kind, exc)
                return self._apologise(reply, apology_for(exc, self.model))
            except Exception:
                log.exception("[turn %d] Unexpected brain failure", turn)
                return self._apologise(reply, APOLOGY_GENERIC)

            if not reply.sentences:
                log.warning("[turn %d] Model returned no text", turn)
                return self._apologise(reply, APOLOGY_TANGLED)
            emotion = reply.first_emotion.value if reply.first_emotion else None
            self.history.add_turn(text, reply.text, emotion, trace, at=started)
            return reply.text

    def amend_last_reply(self, heard: str) -> None:
        with self._lock:
            self.history.amend_last_reply((heard or "").strip())

    def reset(self) -> None:
        with self._lock:
            self.history.clear()

    def close(self) -> None:
        self.client.close()

    # ------------------------------------------------------------------ internals
    def _prefetch_time(self, text: str, turn: int) -> list[dict[str, Any]]:
        """For a plain "what time is it?", call get_current_time before the model does.

        Measured on qwen2.5:7b: later in a conversation it often works the time out from
        old reminder results ("at 11:58, in 20 minutes") and states a wrong one instead of
        calling the tool. Putting the time in the prompt fixed that but made it skip
        other tools ("I've set a reminder" with no set_reminder call). A real tool
        exchange added here does neither, and saves a model round.
        """
        if not TIME_QUESTION.search(text) or "get_current_time" not in self.registry:
            return []
        result = self.registry.call("get_current_time", {})
        log.debug("[turn %d] prefetched get_current_time -> %s", turn, result)
        call = {"function": {"name": "get_current_time", "arguments": {}}}
        return [
            {"role": "assistant", "content": "", "tool_calls": [call]},
            {"role": "tool", "content": result, "tool_name": "get_current_time"},
        ]

    def _system_message(self) -> dict[str, str]:
        assistant = self.config.section("assistant")
        prompt = system_prompt(
            str(assistant.get("name", "ROCO")), str(assistant.get("user_name", "Adam"))
        )
        now = self._clock()
        # Date only: the exact time comes from get_current_time, whose answer is spoken-style
        # (a 7B model misreads "19:58" as "half past seven" when it is in the prompt).
        prompt += f"\n\nToday is {describe_date(now)}."
        return {"role": "system", "content": prompt}

    def _apologise(self, reply: _Reply, apology: str) -> str:
        reply.drop_partial()
        self.bus.set_emotion(Emotion.SAD)
        reply.say(apology)
        return reply.text

    def _converse(
        self,
        messages: list[dict[str, Any]],
        reply: _Reply,
        turn: int,
        ctl: _Turn | None = None,
        trace_from: int | None = None,
    ) -> list[dict[str, Any]]:
        """Run the tool loop. Returns this turn's tool-call and tool-result messages
        (from ``trace_from``, default: what this call adds)."""
        ctl = ctl or _Turn(None, None)
        start = len(messages) if trace_from is None else trace_from
        schemas = self.registry.schemas() or None
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
                if name is None:
                    result = "Error: malformed tool call (no function name)."
                else:
                    result = self.registry.call(name, arguments)
                log.debug("[turn %d] tool %s(%s) -> %.200s", turn, name, _short(arguments), result)
                messages.append({"role": "tool", "content": result, "tool_name": name or ""})
            ctl.check()
        return messages[start:]

    def _stream_round(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
        reply: _Reply,
        ctl: _Turn | None = None,
    ) -> tuple[str, list[Any]]:
        ctl = ctl or _Turn(None, None)
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
            raise _Cancelled
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
