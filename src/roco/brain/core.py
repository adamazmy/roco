"""The model-independent half of ROCO's brain: everything that is not "talk to a model".

:class:`ToolBrain` holds persona, history, voice commands, barge-in and the safety nets
that keep a model honest; a backend subclass only runs the tool loop against its model
(:meth:`ToolBrain._converse`). Two backends exist: :class:`~roco.brain.ollama_brain.OllamaBrain`
(local qwen2.5) and :class:`~roco.brain.claude_brain.ClaudeBrain` (the Claude API). The
safety nets, shared by both:

- personality voice commands ("ROCO, unhinged mode") answered without a model call;
- the persona and personality settings rendered into the system prompt;
- the time prefetch: a plain "what time is it?" gets a real get_current_time result first;
- the reminder-claim repair: "I've set it" with no set_reminder call is made true or owned up;
- the example-line filter (personality) on every sentence;
- emotion tag parsing (the tag is published before the first sentence);
- sentence streaming, and cancel on barge-in with the history amended to what was heard.

Messages between this layer and a backend use one neutral shape (Ollama's, which is also
what :class:`~roco.brain.history.History` stores), so the history survives a switch from
one backend to the other mid-conversation::

    {"role": "system" | "user" | "assistant", "content": "..."}
    {"role": "assistant", "content": "", "tool_calls": [{"function": {"name", "arguments"}}]}
    {"role": "tool", "content": "...", "tool_name": "get_events"}

A backend failure (network, timeout, bad key) is apologised for in character, or, with
``raise_backend_errors=True`` (the :class:`~roco.brain.router.BrainRouter` sets it), raised
as :class:`BackendFailed` with nothing stored, so the router can answer on the other brain.
"""

from __future__ import annotations

import itertools
import logging
import re
import threading
from abc import abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from roco.brain.base import Brain, SentenceCallback, TurnHooks
from roco.brain.history import History
from roco.brain.sentences import FIRST_CLAUSE_MIN_CHARS, SentenceSplitter
from roco.bus import Emotion, EventBus, State
from roco.config import Config
from roco.paths import resolve
from roco.personality import Personality
from roco.tools.clock import describe_date, local_timezone, part_of_day
from roco.tools.registry import ToolRegistry

log = logging.getLogger(__name__)

_turn_ids = itertools.count(1)

APOLOGY_GENERIC = "Sorry, something went wrong in my head. Could you try that again?"
APOLOGY_TANGLED = "Sorry, I got a bit tangled up there. Could you ask me that again?"

# "What time is it?", "what's the time now", "tell me the time": a plain question about
# the current time, answered from a prefetched get_current_time result.
TIME_QUESTION = re.compile(
    r"\b(what\s+time\s+is\s+it|what(?:'s|\s+is)\s+the\s+time|tell\s+me\s+the\s+time"
    r"|the\s+time\s+now|current\s+time)\b",
    re.IGNORECASE,
)
# A reminder request answered with "I'll set it" but no set_reminder call: qwen2.5 does
# this in savage/unhinged mode (the joke crowds out the tool call). Spec §2 hard rule:
# never claim an action that didn't happen, so the brain repairs it (see _repair_reminder).
REMINDER_REQUEST = re.compile(r"\b(remind\s+me|reminder)\b", re.IGNORECASE)
ACTION_CLAIM = re.compile(
    r"\b(i'?ve|i\s+have|i'?ll|i\s+will|all)\s+(set|added|created|scheduled|sorted|done)\b"
    r"|\breminder(?:'s|\s+is)?\s+(set|added|created|scheduled)\b|\bdone\b"
    r"|\bi'?ll\s+remind\s+you\b|\bgot\s+it\b|\bconsider\s+it\b",
    re.IGNORECASE,
)
REMINDER_REPAIR_NOTE = (
    "(Note from the system, not from {user}: you said you'd set that reminder but you "
    "didn't call set_reminder. Call set_reminder now with the details {user} gave: "
    "minutes_from_now for \"in ten minutes\", at for a clock time like \"at six\".)"
)
REMINDER_REPAIR_FAILED = "Actually, I didn't manage to set that reminder. Ask me again?"
# "Cancel the garage one." answered "Cancelled, all done." with no cancel_reminder call.
CANCEL_REQUEST = re.compile(r"\b(cancel|delete|remove|scrap|get\s+rid\s+of|clear)\b",
                            re.IGNORECASE)
CANCEL_CLAIM = re.compile(
    r"\b(cancell?ed|deleted|removed|scrapped|cleared|gone|done|sorted)\b", re.IGNORECASE
)
CANCEL_REPAIR_NOTE = (
    "(Note from the system, not from {user}: you said it was cancelled but you didn't call "
    "cancel_reminder. If {user} meant a reminder, call list_reminders and then "
    "cancel_reminder with its number now; otherwise say plainly you can't cancel that.)"
)
CANCEL_REPAIR_FAILED = "Actually, I didn't manage to cancel that. Ask me again?"


@dataclass(frozen=True)
class ActionCheck:
    """A request whose spoken claim must be backed by a successful ``tool`` call."""

    tool: str
    request: Callable[[str], bool]
    claim: re.Pattern[str]
    note: str
    failed: str


ACTION_CHECKS = (
    ActionCheck(
        "set_reminder",
        lambda t: bool(REMINDER_REQUEST.search(t)) and not CANCEL_REQUEST.search(t),
        ACTION_CLAIM, REMINDER_REPAIR_NOTE, REMINDER_REPAIR_FAILED,
    ),
    ActionCheck(
        "cancel_reminder", lambda t: bool(CANCEL_REQUEST.search(t)),
        CANCEL_CLAIM, CANCEL_REPAIR_NOTE, CANCEL_REPAIR_FAILED,
    ),
)

# Questions about Adam's data that must be answered from a tool, never from memory: after
# a run of chatter in the history, qwen2.5 answered "any unread emails?" with an invented
# "no new emails" and read out a made-up calendar (2026-11-22 spoken retest).
_DAY = (r"(?:today|tomorrow|tonight|this|next|on|at|for|monday|tuesday|wednesday|thursday"
        r"|friday|saturday|sunday)")
DATA_QUESTIONS: tuple[tuple[re.Pattern[str], tuple[str, ...]], ...] = (
    (re.compile(r"\b(e-?mails?|inbox)\b", re.IGNORECASE),
     ("get_unread_emails", "search_emails", "read_email")),
    (re.compile(r"\b(calendar|diary|schedule|agenda|appointments?|meetings?)\b"
                rf"|\b(?:anything|what'?s|what\s+is)\s+on\s+{_DAY}\b|\bam\s+i\s+(?:free|busy)\b",
                re.IGNORECASE),
     ("get_events",)),
    (re.compile(r"\b(what|which|any|list|how\s+many|read\s+me|tell\s+me)\b[^.?!]{0,30}"
                r"\breminders?\b", re.IGNORECASE),
     ("list_reminders",)),
)
# Requests that change data: answered by asking for confirmation or by an action tool.
_CHANGE_REQUEST = re.compile(
    r"\b(add|create|book|put|move|make|set|cancel|delete|remove|send|reply)\b", re.IGNORECASE
)
FETCH_REPAIR_NOTE = (
    "(Note from the system, not from {user}: you answered without calling {tool}, so that "
    "answer may be made up. Call {tool} now and answer {user}'s question from its result.)"
)
FETCH_FAILED = "Sorry, I couldn't check that just now. Ask me again in a moment?"


class BackendFailed(Exception):
    """A brain's model could not answer (raised only with ``raise_backend_errors``).

    ``said`` is what was already delivered to ``on_sentence`` before the failure; the
    turn is NOT in the history. ``cause`` is the backend's own error.
    """

    def __init__(self, cause: BaseException, said: str = "") -> None:
        super().__init__(str(cause))
        self.cause = cause
        self.said = said


def splitter(first: bool = True) -> SentenceSplitter:
    """A short first chunk gets the voice going sooner (TTS time grows with length)."""
    return SentenceSplitter(first_clause_min_chars=FIRST_CLAUSE_MIN_CHARS if first else None)


class Cancelled(Exception):
    """The turn's cancel event was set (barge-in)."""


_NO_HOOKS = TurnHooks()


class TurnControl:
    """Per-call cancel event and hooks, passed down the tool loop."""

    def __init__(self, cancel: threading.Event | None, hooks: TurnHooks | None) -> None:
        self.cancel = cancel
        self.hooks = hooks or _NO_HOOKS
        # Called once, just before the first tool runs (releases a held reply).
        self.before_tools: Callable[[], None] | None = None

    @property
    def cancelled(self) -> bool:
        return self.cancel is not None and self.cancel.is_set()

    def check(self) -> None:
        if self.cancelled:
            raise Cancelled

    def token(self) -> None:
        if self.hooks.on_token is not None:
            try:
                self.hooks.on_token()
            except Exception:
                log.exception("on_token hook failed")

    def tools(self, names: list[str]) -> None:
        if self.before_tools is not None:
            release, self.before_tools = self.before_tools, None
            release()
        if self.hooks.on_tools is not None:
            try:
                self.hooks.on_tools(names)
            except Exception:
                log.exception("on_tools hook failed")


class Reply:
    """Collects one reply: publishes the emotion first, then delivers sentences."""

    def __init__(
        self,
        bus: EventBus,
        on_sentence: SentenceCallback | None,
        cancel: threading.Event | None = None,
        transform: Callable[[str], str] | None = None,
    ) -> None:
        self._bus = bus
        self._on_sentence = on_sentence
        self._cancel = cancel
        self._transform = transform
        self._splitter = splitter()
        self._emotion: Emotion | None = None
        self.first_emotion: Emotion | None = None
        self.sentences: list[str] = []
        # While holding, sentences are kept (and counted) but not spoken yet.
        self.holding = False
        self._held: list[str] = []

    def hold(self) -> None:
        self.holding = True

    def release(self) -> None:
        """Speak the held sentences now and stop holding."""
        self.holding = False
        held, self._held = self._held, []
        for sentence in held:
            self._emit(sentence)

    def discard(self) -> str:
        """Forget the held sentences (never spoken); returns their text."""
        self.holding = False
        held, self._held = self._held, []
        if held:
            del self.sentences[-len(held):]
        self._splitter = splitter(first=not self.sentences)
        return " ".join(held)

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
        self._splitter = splitter(first=not self.sentences)

    def say(self, sentence: str) -> None:
        """Deliver a whole sentence of our own (apologies)."""
        self._deliver(sentence)

    def _deliver(self, sentence: str) -> None:
        if self._cancel is not None and self._cancel.is_set():
            return  # barge-in: nothing more is said or stored
        if self._transform is not None:
            sentence = self._transform(sentence)
            if not sentence.strip():
                return  # the personality filter dropped it (e.g. a copied example line)
        if self._emotion is None:
            self.set_emotion(None)
        self.sentences.append(sentence)
        if self.holding:
            self._held.append(sentence)
            return
        self._emit(sentence)

    def _emit(self, sentence: str) -> None:
        if self._cancel is not None and self._cancel.is_set():
            return
        if self._on_sentence is None:
            return
        try:
            self._on_sentence(sentence)
        except Exception:
            log.exception("on_sentence callback failed")


class ToolBrain(Brain):
    """Shared brain logic; subclasses implement :meth:`_converse` for their model.

    Subclasses set :attr:`backend_errors` (the exception types their model raises) and
    may override :meth:`apology_for`.
    """

    backend_errors: tuple[type[BaseException], ...] = ()
    # Replace old get_current_time results in the history (see History.for_model).
    mask_old_times = True

    def __init__(
        self,
        config: Config,
        bus: EventBus,
        registry: ToolRegistry | None = None,
        *,
        history: History | None = None,
        clock: Callable[[], datetime] | None = None,
        history_tags: bool = True,
        history_tools: bool = True,
        personality: Personality | None = None,
        raise_backend_errors: bool = False,
    ) -> None:
        self.config = config
        self.bus = bus
        self.registry = registry or ToolRegistry()
        self.max_tool_rounds = max(0, int(config.get_path("llm.max_tool_rounds", 5)))
        self.history = history if history is not None else history_from_config(config)
        self._tz = local_timezone(config)
        self._clock = clock or (lambda: datetime.now(self._tz))
        # Send past replies with their tag and tool trace (see roco.brain.history).
        self.history_tags = history_tags
        self.history_tools = history_tools
        # Settings, company and voice commands (roco.personality); renders the prompt.
        self.personality = personality or Personality.from_config(config)
        self.raise_backend_errors = raise_backend_errors
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
        ctl = TurnControl(cancel, hooks)
        with self._lock:
            turn = next(_turn_ids)
            self.bus.set_state(State.THINKING)
            reply = Reply(self.bus, on_sentence, cancel, self.personality.filter_sentence)
            text = (text or "").strip()
            if not text:
                return ""
            started = self._clock()
            # "ROCO, unhinged mode", "Yousaf's here": applied at once, no model call.
            command = self.personality.handle_command(text, self.user_name)
            if command is not None:
                reply.set_emotion(command.emotion)
                reply.say(command.reply)
                self.history.add_turn(text, reply.text, command.emotion.value, [], at=started)
                return reply.text
            past = self.history.for_model(
                tags=self.history_tags, tools=self.history_tools, now=started,
                mask_time=self.mask_old_times,
            )
            messages = [self.system_message(text), *past]
            # No current time is added here: tried after the user text, it made qwen2.5 skip
            # tools far more often (2/4 vs 4/4 scenario runs). Time comes from the tool.
            messages.append({"role": "user", "content": text})
            base = len(messages)
            messages.extend(self._prefetch_time(text, turn))
            fetch = self._fetch_tools(text)
            if fetch:
                # Hold the words until a tool runs: an answer with no tool call is dropped
                # unspoken and the data is fetched (see _repair_fetch).
                reply.hold()
                ctl.before_tools = reply.release
            try:
                ctl.check()
                trace = self._converse(messages, reply, turn, ctl, trace_from=base)
                if fetch and reply.holding:
                    trace = trace + self._repair_fetch(messages, reply, turn, ctl, fetch)
            except Cancelled:
                # Keep what was said and the tool calls already made; the server then
                # replaces the reply with what was heard (amend_last_reply).
                log.info("[turn %d] cancelled after %d sentence(s)", turn, len(reply.sentences))
                emotion = reply.first_emotion.value if reply.first_emotion else None
                self.history.add_turn(text, reply.text, emotion, messages[base:], at=started)
                return reply.text
            except self.backend_errors as exc:
                log.warning("[turn %d] %s failed: %s", turn, self.name, exc)
                if self.raise_backend_errors:
                    reply.drop_partial()
                    raise BackendFailed(exc, reply.text) from exc
                return self._apologise(reply, self.apology_for(exc))
            except Exception:
                log.exception("[turn %d] Unexpected brain failure", turn)
                return self._apologise(reply, APOLOGY_GENERIC)

            if not reply.sentences:
                log.warning("[turn %d] Model returned no text", turn)
                return self._apologise(reply, self.empty_reply_line())
            check = self._unbacked_claim(text, reply.text, trace)
            if check is not None:
                try:
                    trace = trace + self._repair_action(messages, reply, turn, ctl, check)
                except Cancelled:
                    pass
                except self.backend_errors as exc:
                    log.warning("[turn %d] %s repair failed: %s", turn, check.tool, exc)
                    reply.say(check.failed)
            emotion = reply.first_emotion.value if reply.first_emotion else None
            self.history.add_turn(text, reply.text, emotion, trace, at=started)
            return reply.text

    def amend_last_reply(self, heard: str) -> None:
        with self._lock:
            self.history.amend_last_reply((heard or "").strip())

    def forget_turn(self, user_text: str) -> None:
        with self._lock:
            self.history.forget_exchange((user_text or "").strip())

    def reset(self) -> None:
        with self._lock:
            self.history.clear()

    def close(self) -> None:
        """Release the model client (default: nothing to release)."""

    # ------------------------------------------------------------------ for subclasses
    @property
    def name(self) -> str:
        """``backend:model`` for logs and the protocol (e.g. ``ollama:qwen2.5:7b``)."""
        return type(self).__name__

    @abstractmethod
    def _converse(
        self,
        messages: list[dict[str, Any]],
        reply: Reply,
        turn: int,
        ctl: TurnControl | None = None,
        trace_from: int | None = None,
    ) -> list[dict[str, Any]]:
        """Run the tool loop on ``messages`` (neutral shape, system first), streaming the
        answer into ``reply``. Append this turn's tool calls/results to ``messages`` in the
        neutral shape and return ``messages[trace_from:]`` (default: what this call added).

        Raise :class:`Cancelled` once ``ctl`` is cancelled, and one of
        :attr:`backend_errors` when the model cannot be reached.
        """

    def apology_for(self, error: BaseException) -> str:
        return APOLOGY_GENERIC

    def empty_reply_line(self) -> str:
        return APOLOGY_TANGLED

    def allowed_schemas(self) -> list[dict[str, Any]]:
        """Tool schemas (Ollama format) the personality allows right now."""
        allowed = self.personality.allowed_tool
        return [s for s in self.registry.schemas() if allowed(s["function"]["name"])]

    def run_tool(self, name: str | None, arguments: Any) -> str:
        """Run one tool call; never raises (errors come back as text for the model)."""
        if name is None:
            return "Error: malformed tool call (no function name)."
        if not self.personality.allowed_tool(name):
            return "Error: not available while other people are in the room."
        return self.registry.call(name, arguments)

    def system_message(self, text: str = "") -> dict[str, str]:
        assistant = self.config.section("assistant")
        # The user text only switches the prompt to "no jokes" on genuine distress.
        prompt = self.personality.system_prompt(
            str(assistant.get("name", "ROCO")), self.user_name, text
        )
        now = self._clock()
        # Date and part of day only: the exact time comes from get_current_time, whose answer
        # is spoken-style (a 7B model misreads "19:58" as "half past seven" when it is in the
        # prompt). The part of day stops "Morning, Adam" at half ten at night and invented
        # "it's three in the morning" lines.
        prompt += (
            f"\n\nToday is {describe_date(now)}, and it's {part_of_day(now)}. Match greetings "
            "and your mood to that. For the exact time, call get_current_time."
        )
        return {"role": "system", "content": prompt}

    _system_message = system_message  # the name before the shared layer existed

    @property
    def user_name(self) -> str:
        return str(self.config.section("assistant").get("user_name", "Adam"))

    # ------------------------------------------------------------------ safety nets
    def _unbacked_claim(
        self, text: str, said: str, trace: list[dict[str, Any]]
    ) -> ActionCheck | None:
        """The action check a reply fails: it claims the action but the tool never ran."""
        for check in ACTION_CHECKS:
            if (check.tool in self.registry and check.request(text)
                    and check.claim.search(said) and not tool_succeeded(trace, check.tool)):
                return check
        return None

    def _claims_unset_reminder(
        self, text: str, said: str, trace: list[dict[str, Any]]
    ) -> bool:
        check = self._unbacked_claim(text, said, trace)
        return check is not None and check.tool == "set_reminder"

    def _repair_action(
        self, messages: list[dict[str, Any]], reply: Reply, turn: int, ctl: TurnControl,
        check: ActionCheck,
    ) -> list[dict[str, Any]]:
        """Make an "I've set it" / "cancelled" true: one silent tool round asking for it.

        Nothing from the repair round is spoken; if the action still didn't happen, ROCO
        owns up. Only the tool calls and their results are kept for the history.
        """
        log.warning("[turn %d] claimed %s without calling it; repairing", turn, check.tool)
        repair = [
            *messages,
            {"role": "assistant", "content": reply.text},
            {"role": "user", "content": check.note.format(user=self.user_name)},
        ]
        silent = Reply(self.bus, None, ctl.cancel)
        done = self._converse(repair, silent, turn, ctl)
        kept = [m for m in done if m.get("tool_calls") or m.get("role") == "tool"]
        if not tool_succeeded(kept, check.tool):
            reply.say(check.failed)
        return kept

    def _repair_reminder(
        self, messages: list[dict[str, Any]], reply: Reply, turn: int, ctl: TurnControl
    ) -> list[dict[str, Any]]:
        return self._repair_action(messages, reply, turn, ctl, ACTION_CHECKS[0])

    def _fetch_tools(self, text: str) -> tuple[str, ...]:
        """For a question about Adam's data, the tools (available now) that answer it."""
        if _CHANGE_REQUEST.search(text):
            return ()
        allowed = self.personality.allowed_tool
        for pattern, tools in DATA_QUESTIONS:
            if pattern.search(text):
                found = tuple(t for t in tools if t in self.registry and allowed(t))
                if found:
                    return found
        return ()

    def _repair_fetch(
        self, messages: list[dict[str, Any]], reply: Reply, turn: int, ctl: TurnControl,
        tools: tuple[str, ...],
    ) -> list[dict[str, Any]]:
        """The model answered a data question without a tool: drop that answer (it was
        never spoken) and ask once more, this time for the tool. If it still won't call
        one, say we couldn't check instead of guessing."""
        dropped = reply.discard()
        log.warning("[turn %d] answered a %s question without calling it (%r); fetching",
                    turn, tools[0], dropped[:80])
        repair = [*messages]
        if dropped:
            repair.append({"role": "assistant", "content": dropped})
        repair.append({"role": "user",
                       "content": FETCH_REPAIR_NOTE.format(user=self.user_name, tool=tools[0])})
        reply.hold()
        ctl.before_tools = reply.release
        done = self._converse(repair, reply, turn, ctl)
        ctl.before_tools = None
        if reply.holding:
            reply.discard()
            reply.say(FETCH_FAILED)
        return [m for m in done if m.get("tool_calls") or m.get("role") == "tool"]

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

    def _apologise(self, reply: Reply, apology: str) -> str:
        reply.drop_partial()
        self.bus.set_emotion(Emotion.SAD)
        reply.say(apology)
        return reply.text


def history_from_config(config: Config) -> History:
    hist_cfg = config.section("history")
    persist = bool(hist_cfg.get("persist"))
    path = resolve(hist_cfg.get("file", "history.json")) if persist else None
    return History(int(hist_cfg.get("max_turns", 12)), path)


def tool_succeeded(messages: list[dict[str, Any]], name: str) -> bool:
    return any(
        m.get("role") == "tool"
        and m.get("tool_name") == name
        and not str(m.get("content", "")).startswith(("Error", "Cancelled"))
        for m in messages
    )
