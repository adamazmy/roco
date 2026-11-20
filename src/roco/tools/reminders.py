"""Reminder tools + background scheduler that publishes ``announce`` events.

Owner: workstream A.

- ``set_reminder(text, minutes_from_now | at)``: ``at`` is a clock time ("15:30",
  "3:30pm"); if that time already passed today it means tomorrow.
- ``list_reminders()`` and ``cancel_reminder(id)``.

Reminders are persisted to ``reminders.file`` (in ~/.roco) so they survive a
restart. A daemon thread sleeps until the next one is due and calls
``bus.announce(text)``. Reminders that fell due while ROCO was off fire once at
start-up, worded as missed. The clock is injectable so tests never sleep.
"""

from __future__ import annotations

import json
import logging
import os
import re
import threading
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, tzinfo
from pathlib import Path
from typing import Any

from roco.bus import EventBus
from roco.paths import resolve
from roco.tools import ToolContext
from roco.tools.clock import local_timezone
from roco.tools.registry import ToolRegistry

log = logging.getLogger(__name__)

MAX_AHEAD = timedelta(days=7)
# A reminder this late at start-up is still announced normally rather than as "missed".
MISSED_GRACE = timedelta(minutes=1)
# Upper bound on one scheduler sleep, so a changed system clock is noticed.
MAX_SLEEP_S = 30.0

Clock = Callable[[], datetime]

_AT = re.compile(r"^\s*(\d{1,2})(?:[:.h ](\d{2}))?\s*(am|pm|a\.m\.|p\.m\.)?\s*$", re.IGNORECASE)


@dataclass
class Reminder:
    id: int
    text: str
    due: datetime

    def to_json(self) -> dict[str, Any]:
        data = asdict(self)
        data["due"] = self.due.isoformat()
        return data


def parse_clock_time(value: str) -> tuple[int, int]:
    """"15:30", "9.05", "7pm", "7:15 a.m." -> (hour, minute). Raises ValueError."""
    match = _AT.match(value or "")
    if not match:
        raise ValueError(f"{value!r} is not a clock time like 15:30")
    hour, minute = int(match.group(1)), int(match.group(2) or 0)
    suffix = (match.group(3) or "").lower().replace(".", "")
    if suffix:
        if not 1 <= hour <= 12:
            raise ValueError(f"{value!r} is not a valid time")
        hour = hour % 12 + (12 if suffix == "pm" else 0)
    if hour > 23 or minute > 59:
        raise ValueError(f"{value!r} is not a valid time")
    return hour, minute


def spoken_when(due: datetime, now: datetime) -> str:
    """"at 15:30 today", "at 08:00 tomorrow", "at 09:00 on Monday"."""
    days = (due.date() - now.date()).days
    if days == 0:
        day = "today"
    elif days == 1:
        day = "tomorrow"
    else:
        day = f"on {due:%A}"
    return f"at {due:%H:%M} {day}"


class ReminderService:
    """Owns the reminders, their file and the scheduler thread."""

    def __init__(
        self,
        bus: EventBus,
        path: Path | None,
        tz: tzinfo,
        clock: Clock | None = None,
        max_sleep_s: float = MAX_SLEEP_S,
    ) -> None:
        self.bus = bus
        self.path = path
        self.tz = tz
        self._clock = clock or (lambda: datetime.now(tz))
        self._max_sleep_s = max_sleep_s
        self._lock = threading.RLock()
        self._wake = threading.Event()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._reminders: dict[int, Reminder] = {}
        self._next_id = 1
        self._load()

    # ------------------------------------------------------------------ public API
    def now(self) -> datetime:
        return self._clock().astimezone(self.tz)

    def add(self, text: str, due: datetime) -> Reminder:
        with self._lock:
            reminder = Reminder(self._next_id, text, due.astimezone(self.tz))
            self._reminders[reminder.id] = reminder
            self._next_id += 1
            self._save()
        self._wake.set()
        return reminder

    def cancel(self, reminder_id: int) -> Reminder | None:
        with self._lock:
            reminder = self._reminders.pop(reminder_id, None)
            if reminder is not None:
                self._save()
        self._wake.set()
        return reminder

    def pending(self) -> list[Reminder]:
        with self._lock:
            return sorted(self._reminders.values(), key=lambda r: (r.due, r.id))

    def fire_missed(self) -> int:
        """Announce reminders that fell due while ROCO was off. Returns how many fired."""
        cutoff = self.now() - MISSED_GRACE
        return self._fire(lambda r: r.due <= cutoff, missed=True)

    def fire_due(self) -> int:
        """Announce every reminder that is due now. Returns how many fired."""
        now = self.now()
        return self._fire(lambda r: r.due <= now, missed=False)

    def start(self) -> None:
        """Fire missed reminders, then start the scheduler thread (idempotent)."""
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self.fire_missed()
        self._thread = threading.Thread(target=self._run, name="roco-reminders", daemon=True)
        self._thread.start()

    def stop(self, timeout: float = 2.0) -> None:
        self._stop.set()
        self._wake.set()
        if self._thread is not None:
            self._thread.join(timeout)
            self._thread = None

    # ------------------------------------------------------------------ scheduler
    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                self.fire_due()
            except Exception:
                log.exception("Reminder scheduler failed")
            self._wake.wait(self._seconds_to_next())
            self._wake.clear()

    def _seconds_to_next(self) -> float:
        upcoming = self.pending()
        if not upcoming:
            return self._max_sleep_s
        delta = (upcoming[0].due - self.now()).total_seconds()
        return min(self._max_sleep_s, max(0.0, delta))

    def _fire(self, is_due: Callable[[Reminder], bool], missed: bool) -> int:
        with self._lock:
            due = [r for r in self.pending() if is_due(r)]
            for reminder in due:
                del self._reminders[reminder.id]
            if due:
                self._save()
        for reminder in due:
            if missed:
                text = f"While I was switched off, you missed a reminder: {reminder.text}"
            else:
                text = f"Reminder: {reminder.text}"
            log.info("Reminder %d due: %s", reminder.id, reminder.text)
            self.bus.announce(_end_sentence(text))
        return len(due)

    # ------------------------------------------------------------------ persistence
    def _load(self) -> None:
        if self.path is None or not self.path.exists():
            return
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            reminders = {}
            for item in data.get("reminders", []):
                due = datetime.fromisoformat(item["due"])
                if due.tzinfo is None:
                    due = due.replace(tzinfo=self.tz)
                reminder = Reminder(int(item["id"]), str(item["text"]), due)
                reminders[reminder.id] = reminder
            self._reminders = reminders
            self._next_id = max([int(data.get("next_id", 1)), *[i + 1 for i in reminders]])
        except (OSError, ValueError, KeyError, TypeError, AttributeError) as exc:
            log.warning("Reminders file %s is unreadable (%s); starting empty.", self.path, exc)
            self._reminders, self._next_id = {}, 1

    def _save(self) -> None:
        if self.path is None:
            return
        payload = {
            "version": 1,
            "next_id": self._next_id,
            "reminders": [r.to_json() for r in self.pending()],
        }
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_suffix(self.path.suffix + ".tmp")
            tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
            os.replace(tmp, self.path)
        except OSError as exc:
            log.warning("Could not save reminders to %s: %s", self.path, exc)


# ---------------------------------------------------------------------- tool functions


def _end_sentence(text: str) -> str:
    text = text.strip()
    return text if text[-1:] in ".!?" else text + "."


def _is_blank(value: Any) -> bool:
    return value is None or (isinstance(value, str) and not value.strip())


def set_reminder(
    service: ReminderService,
    text: str,
    minutes_from_now: float | str | None = None,
    at: str | None = None,
) -> str:
    text = (text or "").strip().rstrip(".")
    if not text:
        return "Error: the reminder needs some text."
    now = service.now()
    has_minutes = not _is_blank(minutes_from_now)
    has_at = not _is_blank(at)
    if has_minutes and has_at:
        # Small models often fill both; a zero/placeholder minutes value means "use at".
        try:
            has_minutes = float(minutes_from_now) > 0  # type: ignore[arg-type]
        except (TypeError, ValueError):
            has_minutes = False
        if has_minutes:
            return "Error: give either minutes_from_now or at, not both."
    if has_minutes:
        try:
            minutes = float(minutes_from_now)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return f"Error: minutes_from_now must be a number, not {minutes_from_now!r}."
        if minutes <= 0:
            return "Error: minutes_from_now must be more than zero."
        due = now + timedelta(minutes=minutes)
        if due - now > MAX_AHEAD:
            return "Error: I can only set reminders up to a week ahead."
    elif has_at:
        try:
            hour, minute = parse_clock_time(str(at))
        except ValueError as exc:
            return f"Error: {exc}."
        due = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
        if due <= now:
            due += timedelta(days=1)
    else:
        return "Error: say when, with minutes_from_now or at (a time like 15:30)."
    reminder = service.add(text, due)
    return f"Reminder {reminder.id} set {spoken_when(reminder.due, now)}: {reminder.text}."


def list_reminders(service: ReminderService) -> str:
    pending = service.pending()
    if not pending:
        return "You have no reminders."
    now = service.now()
    items = [f"number {r.id} {spoken_when(r.due, now)}, {r.text}" for r in pending]
    noun = "reminder" if len(items) == 1 else "reminders"
    return f"You have {len(items)} {noun}: " + "; ".join(items) + "."


def cancel_reminder(service: ReminderService, id: int | str) -> str:  # noqa: A002
    try:
        reminder_id = int(str(id).strip().lstrip("#"))
    except ValueError:
        return f"Error: {id!r} is not a reminder number."
    reminder = service.cancel(reminder_id)
    if reminder is None:
        return f"There's no reminder number {reminder_id}."
    return f"Cancelled reminder {reminder.id}: {reminder.text}."


# ---------------------------------------------------------------------- registration

_services: dict[Path | None, ReminderService] = {}
_services_lock = threading.Lock()


def start_service(bus: EventBus, path: Path | None, tz: tzinfo) -> ReminderService:
    """Start the scheduler for ``path``, replacing any earlier one on the same file."""
    with _services_lock:
        old = _services.pop(path, None)
        if old is not None:
            old.stop()
        service = ReminderService(bus, path, tz)
        service.start()
        _services[path] = service
        return service


def register(registry: ToolRegistry, ctx: ToolContext) -> None:
    path = resolve(ctx.config.get_path("reminders.file", "reminders.json"))
    service = start_service(ctx.bus, path, local_timezone(ctx.config))
    register_tools(registry, service)


def register_tools(registry: ToolRegistry, service: ReminderService) -> None:
    @registry.tool(
        "set_reminder",
        description=(
            "Set a reminder. Give the text and EITHER minutes_from_now (e.g. 10) "
            "OR at, a 24-hour clock time like \"15:30\" (the next time it comes round)."
        ),
        parameters={
            "type": "object",
            "properties": {
                "text": {"type": "string", "description": "What to remind Adam about."},
                "minutes_from_now": {"type": "number", "description": "Minutes from now."},
                "at": {"type": "string", "description": "Clock time, HH:MM 24-hour."},
            },
            "required": ["text"],
        },
    )
    def set_reminder_tool(
        text: str, minutes_from_now: float | None = None, at: str | None = None
    ) -> str:
        return set_reminder(service, text, minutes_from_now, at)

    @registry.tool(
        "list_reminders",
        description="List the reminders that are still to come.",
        parameters={"type": "object", "properties": {}, "required": []},
    )
    def list_reminders_tool() -> str:
        return list_reminders(service)

    @registry.tool(
        "cancel_reminder",
        description="Cancel a reminder by its number (from list_reminders).",
        parameters={
            "type": "object",
            "properties": {"id": {"type": "integer", "description": "Reminder number."}},
            "required": ["id"],
        },
    )
    def cancel_reminder_tool(id: int) -> str:  # noqa: A002
        return cancel_reminder(service, id)
