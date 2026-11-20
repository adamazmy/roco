"""Conversation history: the last N user/assistant turns, optionally persisted as JSON.

Only plain user and assistant messages are kept (tool-call messages live for the
current turn only), so the persisted file stays small and model-agnostic::

    {"version": 1, "messages": [{"role": "user", "content": "..."}, ...]}
"""

from __future__ import annotations

import json
import logging
import os
import threading
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

FORMAT_VERSION = 1
_ROLES = ("user", "assistant")


class History:
    """Keeps at most ``max_turns`` user+assistant exchanges."""

    def __init__(self, max_turns: int = 12, path: Path | None = None) -> None:
        self.max_turns = max(0, int(max_turns))
        self.path = path
        self._messages: list[dict[str, str]] = []
        self._lock = threading.Lock()
        if path is not None:
            self._messages = self._trim(self._load(path))

    @property
    def messages(self) -> list[dict[str, str]]:
        """A copy of the stored messages, oldest first."""
        with self._lock:
            return [dict(m) for m in self._messages]

    def __len__(self) -> int:
        return len(self._messages)

    def add_turn(self, user: str, assistant: str) -> None:
        with self._lock:
            self._messages += [
                {"role": "user", "content": user},
                {"role": "assistant", "content": assistant},
            ]
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
    def _trim(self, messages: list[dict[str, str]]) -> list[dict[str, str]]:
        keep = self.max_turns * 2
        if keep == 0:
            return []
        messages = messages[-keep:]
        # Never start with an orphan assistant message.
        while messages and messages[0]["role"] != "user":
            messages = messages[1:]
        return messages

    @staticmethod
    def _load(path: Path) -> list[dict[str, str]]:
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
                messages.append({"role": m["role"], "content": m["content"]})
            return messages
        except (OSError, ValueError, KeyError, TypeError) as exc:
            log.warning("History file %s is unreadable (%s); starting fresh.", path, exc)
            return []

    def _save(self, messages: list[dict[str, str]]) -> None:
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
