"""Reminder tools and scheduler, with an injectable clock (no real waiting)."""

from __future__ import annotations

import json
import threading
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import pytest

from roco.bus import Topic
from roco.tools import ToolContext, build_registry
from roco.tools import reminders as rem
from roco.tools.reminders import ReminderService, parse_clock_time

TZ = ZoneInfo("Europe/London")


class FakeClock:
    def __init__(self, start: datetime) -> None:
        self.now = start

    def __call__(self) -> datetime:
        return self.now

    def advance(self, **kwargs) -> None:
        self.now += timedelta(**kwargs)


@pytest.fixture
def clock():
    return FakeClock(datetime(2026, 11, 20, 14, 0, tzinfo=TZ))


@pytest.fixture
def announced(bus):
    said: list[str] = []
    bus.subscribe(Topic.ANNOUNCE, said.append)
    return said


@pytest.fixture
def path(tmp_path):
    return tmp_path / "reminders.json"


@pytest.fixture
def service(bus, path, clock):
    return ReminderService(bus, path, TZ, clock=clock)


@pytest.mark.parametrize(
    "value, expected",
    [
        ("15:30", (15, 30)),
        ("9:05", (9, 5)),
        ("09.05", (9, 5)),
        ("7pm", (19, 0)),
        ("7:15 a.m.", (7, 15)),
        ("12am", (0, 0)),
        ("12:30 PM", (12, 30)),
        ("18", (18, 0)),
    ],
)
def test_parse_clock_time(value, expected):
    assert parse_clock_time(value) == expected


@pytest.mark.parametrize("value", ["25:00", "12:75", "13pm", "soon", ""])
def test_parse_clock_time_rejects(value):
    with pytest.raises(ValueError):
        parse_clock_time(value)


def test_set_in_minutes(service):
    result = rem.set_reminder(service, "check the oven.", minutes_from_now=10)
    assert result == "Reminder 1 set at 14:10 today: check the oven."
    assert service.pending()[0].due == datetime(2026, 11, 20, 14, 10, tzinfo=TZ)


def test_minutes_as_string(service):
    assert "14:05" in rem.set_reminder(service, "tea", minutes_from_now="5")


def test_set_at_later_today_and_past_means_tomorrow(service):
    assert rem.set_reminder(service, "call Mum", at="18:30").endswith("at 18:30 today: call Mum.")
    assert "at 09:00 tomorrow" in rem.set_reminder(service, "bins", at="09:00")
    assert "at 14:00 tomorrow" in rem.set_reminder(service, "exactly now", at="14:00")


@pytest.mark.parametrize(
    "kwargs, error",
    [
        ({"text": "", "minutes_from_now": 5}, "needs some text"),
        ({"text": "x"}, "say when"),
        ({"text": "x", "minutes_from_now": 5, "at": "15:00"}, "not both"),
        ({"text": "x", "minutes_from_now": -1}, "more than zero"),
        ({"text": "x", "minutes_from_now": "lots"}, "must be a number"),
        ({"text": "x", "minutes_from_now": 60 * 24 * 8}, "a week ahead"),
        ({"text": "x", "at": "half three"}, "not a clock time"),
    ],
)
def test_set_reminder_errors(service, kwargs, error):
    result = rem.set_reminder(service, **kwargs)
    assert result.startswith("Error:") and error in result
    assert service.pending() == []


def test_zero_minutes_with_at_uses_at(service):
    # Small models often send every argument; a placeholder 0 must not win.
    assert "at 16:00 today" in rem.set_reminder(service, "x", minutes_from_now=0, at="16:00")


def test_list_and_cancel(service):
    assert rem.list_reminders(service) == "You have no reminders."
    rem.set_reminder(service, "bins", at="09:00")
    rem.set_reminder(service, "tea", minutes_from_now=5)
    assert rem.list_reminders(service) == (
        "You have 2 reminders: number 2 at 14:05 today, tea; number 1 at 09:00 tomorrow, bins."
    )
    assert rem.cancel_reminder(service, "#1") == "Cancelled reminder 1: bins."
    assert rem.cancel_reminder(service, 1) == "There's no reminder number 1."
    assert rem.cancel_reminder(service, "one").startswith("Error:")
    assert rem.list_reminders(service).startswith("You have 1 reminder:")


def test_fire_due_announces_once(service, clock, announced):
    rem.set_reminder(service, "stretch", minutes_from_now=1)
    assert service.fire_due() == 0
    clock.advance(minutes=1)
    assert service.fire_due() == 1
    assert service.fire_due() == 0
    assert announced == ["Reminder: stretch."]
    assert service.pending() == []


def test_persisted_across_restart(bus, path, clock, service):
    rem.set_reminder(service, "bins", at="09:00")
    rem.set_reminder(service, "tea", minutes_from_now=5)
    rem.cancel_reminder(service, 2)
    data = json.loads(path.read_text(encoding="utf-8"))
    assert [r["text"] for r in data["reminders"]] == ["bins"]

    again = ReminderService(bus, path, TZ, clock=clock)
    assert [r.text for r in again.pending()] == ["bins"]
    # Ids keep counting after a restart instead of reusing 2.
    assert rem.set_reminder(again, "x", minutes_from_now=1).startswith("Reminder 3 ")


def test_missed_while_off_fire_once_on_start(bus, path, clock, service, announced):
    rem.set_reminder(service, "take the pills", minutes_from_now=5)
    rem.set_reminder(service, "later", minutes_from_now=120)
    clock.advance(minutes=30)  # ROCO was off for half an hour

    again = ReminderService(bus, path, TZ, clock=clock)
    assert again.fire_missed() == 1
    assert announced == ["While I was switched off, you missed a reminder: take the pills."]
    assert [r.text for r in again.pending()] == ["later"]
    assert ReminderService(bus, path, TZ, clock=clock).fire_missed() == 0  # only once


def test_corrupt_file_starts_empty(bus, path, clock, caplog):
    path.write_text("{oops", encoding="utf-8")
    service = ReminderService(bus, path, TZ, clock=clock)
    assert service.pending() == []
    assert "starting empty" in caplog.text


def test_scheduler_thread_fires_when_due(bus, path, clock, announced):
    fired = threading.Event()
    bus.subscribe(Topic.ANNOUNCE, lambda _t: fired.set())
    service = ReminderService(bus, path, TZ, clock=clock, max_sleep_s=0.02)
    service.start()
    try:
        rem.set_reminder(service, "kettle", minutes_from_now=3)
        assert not fired.wait(0.1)
        clock.advance(minutes=3)
        assert fired.wait(1.0)
        assert announced == ["Reminder: kettle."]
    finally:
        service.stop()


def test_register_exposes_tools_and_starts_scheduler(config, bus):
    registry = build_registry(ToolContext(config=config, bus=bus))
    try:
        assert {"get_current_time", "set_reminder", "list_reminders", "cancel_reminder"} <= set(
            registry.names()
        )
        result = registry.call("set_reminder", {"text": "stand up", "minutes_from_now": 20})
        assert result.startswith("Reminder 1 set at ")
        assert registry.call("cancel_reminder", {"id": "1"}).startswith("Cancelled reminder 1")
        assert registry.call("list_reminders") == "You have no reminders."
    finally:
        rem.stop_all()
