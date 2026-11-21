"""Conversation history: the last N user/assistant turns, optionally persisted as JSON.

The stored text is the user turn and the tag-free assistant reply. Assistant
entries may also carry the turn's ``emotion`` and a compact ``tools`` trace (the
tool-call/tool-result messages of that turn)::

    {"version": 1, "messages": [
        {"role": "user", "content": "Remind me to call Mum at 9pm."},
        {"role": "assistant", "content": "Done, nine tonight.", "emotion": "happy",
         "at": "2026-11-20T20:05:00+00:00",
         "tools": [{"role": "assistant", "content": "", "tool_calls": [...]},
                   {"role": "tool", "content": "Reminder 1 set...", "tool_name": "set_reminder"}]}
    ]}

:attr:`History.messages` is the plain user/assistant view. :meth:`History.for_model`
is what the brain sends: small models copy the shape of their own past turns, so
without the tag and the tool trace they soon stop tagging replies and start
claiming actions ("I've cancelled it") without calling the tool. To keep that
cheap and honest, tool results are capped at 400 characters, only the last 3
turns replay their tools, and each replayed result says when it was fetched.
"""

from __future__ import annotations

import json
import logging
import os
import threading
from datetime import datetime
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

FORMAT_VERSION = 1
_ROLES = ("user", "assistant")
# A replayed `read_email` or a week of events must not push older turns out of context.
MAX_TOOL_RESULT_CHARS = 400
# The last this-many turns replay full (capped) results; older turns replay their tool
# calls with results squeezed to OLD_TOOL_RESULT_CHARS.
DEFAULT_TOOL_TURNS = 3
OLD_TOOL_RESULT_CHARS = 80


def _cap_tool_message(
    message: dict[str, Any], limit: int = MAX_TOOL_RESULT_CHARS
) -> dict[str, Any]:
    message = dict(message)
    content = message.get("content")
    too_long = isinstance(content, str) and len(content) > limit
    if message.get("role") == "tool" and too_long:
        message["content"] = content[:limit].rstrip() + "... (cut short)"
    return message


def _fetched_when(at: Any, now: datetime | None) -> str:
    """How long ago a tool result was fetched, WITHOUT a clock time: "just now",
    "about 5 minutes ago", "about 2 hours ago", "yesterday", "on Thursday 19 November".

    A clock time here ("fetched at 11:04") is the first thing a 7B model grabs when
    asked "what time is it?" later in the conversation, instead of calling the tool.
    """
    if not isinstance(at, str):
        return ""
    try:
        stamp = datetime.fromisoformat(at)
    except ValueError:
        return ""
    if now is None:
        return f"on {stamp:%A} {stamp.day} {stamp:%B}"
    if now.tzinfo is not None and stamp.tzinfo is not None:
        stamp = stamp.astimezone(now.tzinfo)
    minutes = int((now - stamp).total_seconds() // 60)
    if stamp.date() == now.date():
        if minutes < 2:
            return "just now"
        if minutes < 60:
            return f"about {minutes} minutes ago"
        hours = round(minutes / 60)
        return f"about {hours} hour{'s' if hours != 1 else ''} ago"
    if (now.date() - stamp.date()).days == 1:
        return "yesterday"
    return f"on {stamp:%A} {stamp.day} {stamp:%B}"


# Tools whose old results are only "what the time was": replaying their text invites the
# model to repeat a stale time, so only the reminder to call them again is kept.
TIME_ONLY_TOOLS = frozenset({"get_current_time"})


class History:
    """Keeps at most ``max_turns`` user+assistant exchanges."""

    def __init__(self, max_turns: int = 12, path: Path | None = None) -> None:
        self.max_turns = max(0, int(max_turns))
        self.path = path
        self._messages: list[dict[str, Any]] = []
        self._lock = threading.Lock()
        if path is not None:
            self._messages = self._trim(self._load(path))

    @property
    def messages(self) -> list[dict[str, Any]]:
        """The plain user/assistant messages (tag-free text), oldest first."""
        with self._lock:
            return [{"role": m["role"], "content": m["content"]} for m in self._messages]

    def for_model(
        self,
        tags: bool = True,
        tools: bool = True,
        tool_turns: int = DEFAULT_TOOL_TURNS,
        now: datetime | None = None,
    ) -> list[dict[str, Any]]:
        """Messages to send to the model: past replies with their tag and tool trace.

        Every turn that used tools replays its tool calls: a turn that shows an
        action with no call teaches the model to bluff. Only the last ``tool_turns``
        turns keep results of up to 400 characters; older results are squeezed to
        :data:`OLD_TOOL_RESULT_CHARS`. Each replayed result says when it was fetched,
        so the model can tell stale calendar/email data from fresh.
        """
        out: list[dict[str, Any]] = []
        with self._lock:
            entries = [dict(m) for m in self._messages]
        replies = [i for i, m in enumerate(entries) if m["role"] == "assistant"]
        recent = set(replies[-tool_turns:]) if tool_turns > 0 else set()
        for index, m in enumerate(entries):
            if m["role"] == "assistant":
                if tools and m.get("tools"):
                    when = _fetched_when(m.get("at"), now)
                    limit = MAX_TOOL_RESULT_CHARS if index in recent else OLD_TOOL_RESULT_CHARS
                    for t in m["tools"]:
                        t = _cap_tool_message(t, limit)
                        if t.get("role") == "tool" and t.get("tool_name") in TIME_ONLY_TOOLS:
                            t["content"] = (
                                "(Old result: the time back then. To know the time now, "
                                "call get_current_time again.)"
                            )
                        elif t.get("role") == "tool" and when:
                            t["content"] = (
                                f"(Old result, fetched {when}. It may be out of date: "
                                f"call the tool again for current data.) {t.get('content', '')}"
                            )
                        out.append(t)
                content = m["content"]
                if tags and m.get("emotion"):
                    content = f"[{m['emotion']}] {content}"
                out.append({"role": "assistant", "content": content})
            else:
                out.append({"role": m["role"], "content": m["content"]})
        return out

    def __len__(self) -> int:
        return len(self._messages)

    def add_turn(
        self,
        user: str,
        assistant: str,
        emotion: str | None = None,
        tools: list[dict[str, Any]] | None = None,
        at: datetime | None = None,
    ) -> None:
        """Store one exchange. Tool results are capped at :data:`MAX_TOOL_RESULT_CHARS`."""
        reply: dict[str, Any] = {"role": "assistant", "content": assistant}
        if emotion:
            reply["emotion"] = str(emotion)
        if tools:
            reply["tools"] = [_cap_tool_message(t) for t in tools]
        if at is not None:
            reply["at"] = at.isoformat(timespec="seconds")
        with self._lock:
            self._messages += [{"role": "user", "content": user}, reply]
            self._messages = self._trim(self._messages)
            snapshot = list(self._messages)
        self._save(snapshot)

    def clear(self) -> None:
        with self._lock:
            self._messages = []
        if self.path is not None:
            try:
                self.path.unlink(missing_ok=True)
            except OSError as exc:
                log.warning("Could not delete history file %s: %s", self.path, exc)

    # ------------------------------------------------------------------ internals
    def _trim(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        keep = self.max_turns * 2
        if keep == 0:
            return []
        messages = messages[-keep:]
        # Never start with an orphan assistant message.
        while messages and messages[0]["role"] != "user":
            messages = messages[1:]
        return messages

    @staticmethod
    def _load(path: Path) -> list[dict[str, Any]]:
        if not path.exists():
            return []
        try:
            data: Any = json.loads(path.read_text(encoding="utf-8"))
            raw = data["messages"] if isinstance(data, dict) else data
            if not isinstance(raw, list):
                raise ValueError("'messages' is not a list")
            messages = []
            for m in raw:
                if not (isinstance(m, dict) and m.get("role") in _ROLES):
                    raise ValueError(f"bad message entry: {m!r:.80}")
                if not isinstance(m.get("content"), str):
                    raise ValueError(f"bad message content: {m!r:.80}")
                entry: dict[str, Any] = {"role": m["role"], "content": m["content"]}
                if isinstance(m.get("emotion"), str):
                    entry["emotion"] = m["emotion"]
                if isinstance(m.get("tools"), list) and all(
                    isinstance(t, dict) and t.get("role") in ("assistant", "tool")
                    for t in m["tools"]
                ):
                    entry["tools"] = m["tools"]
                if isinstance(m.get("at"), str):
                    entry["at"] = m["at"]
                messages.append(entry)
            return messages
        except (OSError, ValueError, KeyError, TypeError) as exc:
            log.warning("History file %s is unreadable (%s); starting fresh.", path, exc)
            return []

    def _save(self, messages: list[dict[str, Any]]) -> None:
        if self.path is None:
            return
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_suffix(self.path.suffix + ".tmp")
            payload = {"version": FORMAT_VERSION, "messages": messages}
            tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
            os.replace(tmp, self.path)
        except OSError as exc:
            log.warning("Could not save history to %s: %s", self.path, exc)
