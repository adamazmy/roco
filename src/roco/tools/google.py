"""Google Calendar + Gmail tools (Gmail is read-only: ROCO never sends email).

Owner: workstream D.

:func:`register` adds nothing unless ``google.enabled`` is true AND a valid (or
refreshable) token exists in ``~/.roco``. With no calendar tool, the system
prompt makes ROCO say Google isn't connected (see :mod:`roco.brain.persona`).

The tool logic lives in :class:`GoogleTools`, which only needs two objects shaped
like the ``googleapiclient`` discovery resources (``calendar.events().list(...)
.execute()``, ``gmail.users().messages().get(...).execute()``). Tests inject fakes,
so this module imports no Google library at module level and works without the
``google`` extra.

Results are written for a small local model: short lines, local times in
``assistant.timezone``, email ids included so ``read_email`` can be called next.
"""

from __future__ import annotations

import base64
import binascii
import html
import json
import logging
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta, tzinfo
from email.utils import parseaddr
from typing import Any
from zoneinfo import ZoneInfo

from roco.integrations import google_auth, text
from roco.integrations.text import (
    BODY_CHARS,
    BODY_LABEL,
    ToolInputError,
    clamp_int,
    clean_text,
    clip,
    day_label,
    html_to_text,
    speakable,
    truncate,
)
from roco.tools import ToolContext
from roco.tools.clock import spoken_time
from roco.tools.registry import ToolRegistry

log = logging.getLogger(__name__)

MAX_DAYS = 14
MAX_EVENTS = 50
MAX_EMAILS = 10
HTTP_TIMEOUT_S = 20

# --------------------------------------------------------------------------- registration


def register(registry: ToolRegistry, ctx: ToolContext) -> None:
    cfg = ctx.config
    if not cfg.get_path("google.enabled", False):
        log.info("Google tools not registered: google.enabled is false in the config")
        return
    try:
        creds = google_auth.load_credentials(cfg)
    except google_auth.GoogleAuthError as exc:
        log.warning("Google tools not registered: %s", exc)
        return
    try:
        calendar, gmail = _build_services(creds)
    except ImportError as exc:
        log.warning("Google tools not registered: %s. %s", exc, google_auth.EXTRA_HINT)
        return
    tools = GoogleTools(
        calendar=calendar,
        gmail=gmail,
        tz=ZoneInfo(str(cfg.get_path("assistant.timezone", "Europe/London"))),
        calendar_id=str(cfg.get_path("google.calendar_id", "primary")),
    )
    provider = str(cfg.get_path("email.provider", "google")).lower()
    register_tools(registry, tools, email=provider == "google")
    log.info(
        "Google tools registered (calendar %s; Gmail %s)",
        tools.calendar_id,
        "on" if provider == "google" else f"off, email.provider is {provider}",
    )


def _build_services(creds: Any) -> tuple[Any, Any]:
    """Build the Calendar v3 and Gmail v1 clients with a network timeout."""
    import google_auth_httplib2
    import httplib2
    from googleapiclient.discovery import build

    def authorized_http() -> Any:
        http = httplib2.Http(timeout=HTTP_TIMEOUT_S)
        return google_auth_httplib2.AuthorizedHttp(creds, http=http)

    calendar = build("calendar", "v3", http=authorized_http(), cache_discovery=False)
    gmail = build("gmail", "v1", http=authorized_http(), cache_discovery=False)
    return calendar, gmail


def register_tools(registry: ToolRegistry, tools: GoogleTools, *, email: bool = True) -> None:
    """Expose a :class:`GoogleTools` instance's methods as model tools.

    ``email=False`` registers only the calendar tools (``email.provider`` is not google).
    """

    @registry.tool(
        description=(
            "List calendar events. date is the first day as YYYY-MM-DD (default today; "
            "'today' and 'tomorrow' also work); days is how many days to cover (default 1, "
            "max 14). Times are local."
        ),
        parameters={
            "type": "object",
            "properties": {
                "date": {"type": "string", "description": "First day, YYYY-MM-DD. Default today."},
                "days": {"type": "integer", "description": "Number of days, 1-14. Default 1."},
            },
            "required": [],
        },
    )
    def get_events(date: str | None = None, days: int = 1) -> str:
        return tools.get_events(date, days)

    @registry.tool(
        description=(
            "Create a calendar event. Only call this after the user has clearly confirmed "
            "the title, day, time and length. start_iso is the start like 2026-11-21T09:30; "
            "without a UTC offset it is local time."
        ),
        parameters={
            "type": "object",
            "properties": {
                "title": {"type": "string", "description": "Event title."},
                "start_iso": {
                    "type": "string",
                    "description": "Start, YYYY-MM-DDTHH:MM, local time unless it has an offset.",
                },
                "duration_minutes": {"type": "integer", "description": "Length in minutes."},
                "description": {"type": "string", "description": "Optional notes."},
            },
            "required": ["title", "start_iso", "duration_minutes"],
        },
    )
    def create_event(
        title: str, start_iso: str, duration_minutes: int, description: str | None = None
    ) -> str:
        return tools.create_event(title, start_iso, duration_minutes, description)

    if not email:
        return

    @registry.tool(
        description=(
            "List unread emails in the inbox, newest first: sender, subject, a short "
            "snippet and the id to pass to read_email."
        ),
        parameters={
            "type": "object",
            "properties": {
                "max": {"type": "integer", "description": "How many, 1-10. Default 5."},
            },
            "required": [],
        },
    )
    def get_unread_emails(max: int = 5) -> str:
        return tools.get_unread_emails(max)

    @registry.tool(
        description=(
            "Search emails with Gmail search syntax, e.g. 'from:alice', 'subject:invoice', "
            "'newer_than:7d', 'is:unread'. Returns sender, subject, snippet and id."
        ),
        parameters={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Gmail search query."},
                "max": {"type": "integer", "description": "How many, 1-10. Default 5."},
            },
            "required": ["query"],
        },
    )
    def search_emails(query: str, max: int = 5) -> str:
        return tools.search_emails(query, max)

    @registry.tool(
        description=(
            "Read one email by its id (from get_unread_emails or search_emails). Returns the "
            "sender, subject, date and plain-text body (shortened if long)."
        ),
        parameters={
            "type": "object",
            "properties": {"id": {"type": "string", "description": "Email id."}},
            "required": ["id"],
        },
    )
    def read_email(id: str) -> str:
        return tools.read_email(id)


# --------------------------------------------------------------------------- errors


def friendly_errors(action: str) -> Callable[[Callable[..., str]], Callable[..., str]]:
    """Turn Google/network exceptions into one short readable ``Error: ...`` line."""

    return text.friendly_errors(action, lambda exc: describe_error(exc), log)


AUTH_EXPIRED = (
    "The Google sign-in has expired or was revoked. Adam needs to run 'roco auth google' again."
)


def describe_error(exc: BaseException) -> str:
    """Classify without importing Google libraries (duck typing on the exception)."""
    name = type(exc).__name__
    if name == "RefreshError":
        return AUTH_EXPIRED
    status = _http_status(exc)
    if status is not None:
        reason = _http_reason(exc)
        if status == 401:
            return AUTH_EXPIRED
        if "mail service not enabled" in reason.lower():
            return "This Google account has no Gmail mailbox, so there is no email to read."
        if status == 403:
            text = (reason + " " + str(exc)).lower()
            if "accessnotconfigured" in text or "has not been used" in text or "disabled" in text:
                return "That Google API isn't enabled in the Google Cloud project."
            if "insufficient" in text or "scope" in text:
                return "ROCO lacks permission for that. Adam needs to run 'roco auth google' again."
            return "Google refused access (403)."
        if status == 404:
            return "Google couldn't find it (404). Check the id."
        if status == 429:
            return "Google is rate-limiting requests. Try again in a minute."
        if status >= 500:
            return "Google is having trouble right now. Try again shortly."
        return f"Google rejected the request ({status}{': ' + reason if reason else ''})."
    if isinstance(exc, (OSError, TimeoutError)) or name in {
        "ServerNotFoundError",
        "HttpLib2Error",
        "TransportError",
        "RelativeURIError",
    }:
        return "Couldn't reach Google. Check the internet connection."
    return f"Unexpected problem: {exc.__class__.__name__}."


def _http_status(exc: BaseException) -> int | None:
    status = getattr(exc, "status_code", None)
    if status is None:
        status = getattr(getattr(exc, "resp", None), "status", None)
    try:
        return int(status) if status is not None else None
    except (TypeError, ValueError):
        return None


def _http_reason(exc: BaseException) -> str:
    content = getattr(exc, "content", b"")
    try:
        if isinstance(content, bytes):
            content = content.decode("utf-8", "replace")
        error = json.loads(content).get("error", {})
        message = error.get("message", "") if isinstance(error, dict) else str(error)
    except (ValueError, AttributeError):
        message = ""
    return clip(" ".join(str(message).split()), 120)


# --------------------------------------------------------------------------- tools


@dataclass
class GoogleTools:
    calendar: Any
    gmail: Any
    tz: tzinfo
    calendar_id: str = "primary"
    now: Callable[[], datetime] = field(default=lambda: datetime.now().astimezone())

    def _now(self) -> datetime:
        return self.now().astimezone(self.tz)

    # ----------------------------------------------------------------- calendar
    @friendly_errors("read the calendar")
    def get_events(self, day: str | None = None, days: int | str = 1) -> str:
        first = self._parse_day(day)
        n_days = clamp_int(days, 1, MAX_DAYS, "days")
        start = datetime.combine(first, time(), self.tz)
        end = datetime.combine(first + timedelta(days=n_days), time(), self.tz)
        resp = (
            self.calendar.events()
            .list(
                calendarId=self.calendar_id,
                timeMin=start.isoformat(),
                timeMax=end.isoformat(),
                singleEvents=True,
                orderBy="startTime",
                maxResults=MAX_EVENTS,
            )
            .execute()
        )
        text = format_events(resp.get("items", []), self.tz, first, n_days)
        if resp.get("nextPageToken"):
            text += f"\n(Only the first {MAX_EVENTS} events are shown.)"
        return text

    @friendly_errors("create the event")
    def create_event(
        self,
        title: str,
        start_iso: str,
        duration_minutes: int | str,
        description: str | None = None,
    ) -> str:
        title = str(title or "").strip()
        if not title:
            raise ToolInputError("title must not be empty.")
        start = self._parse_start(start_iso)
        minutes = clamp_int(duration_minutes, 1, 24 * 60, "duration_minutes", clamp=False)
        end = start + timedelta(minutes=minutes)
        body = build_event_body(title, start, end, self.tz, description)
        created = (
            self.calendar.events().insert(calendarId=self.calendar_id, body=body).execute()
        )
        when = _format_timed(start, end, start.date())
        return f"Created: {created.get('summary', title)}, {day_label(start.date())}, {when}."

    def _parse_day(self, value: str | None) -> date:
        text = str(value or "").strip().lower()
        today = self._now().date()
        if text in ("", "today", "none", "null"):
            return today
        if text == "tomorrow":
            return today + timedelta(days=1)
        if text == "yesterday":
            return today - timedelta(days=1)
        try:
            return date.fromisoformat(text[:10])
        except ValueError:
            raise ToolInputError(f"date must be YYYY-MM-DD, got {value!r}.") from None

    def _parse_start(self, value: str) -> datetime:
        text = str(value or "").strip().replace(" ", "T", 1)
        try:
            start = datetime.fromisoformat(text)
        except ValueError:
            raise ToolInputError(
                f"start_iso must look like 2026-11-21T09:30, got {value!r}."
            ) from None
        if start.tzinfo is None:
            return start.replace(tzinfo=self.tz)
        return start.astimezone(self.tz)

    # ----------------------------------------------------------------- gmail
    @friendly_errors("check email")
    def get_unread_emails(self, max: int | str = 5) -> str:
        n = clamp_int(max, 1, MAX_EMAILS, "max")
        return self._list_messages("is:unread in:inbox", n, empty="No unread emails.")

    @friendly_errors("search email")
    def search_emails(self, query: str, max: int | str = 5) -> str:
        query = str(query or "").strip()
        if not query:
            raise ToolInputError("query must not be empty.")
        n = clamp_int(max, 1, MAX_EMAILS, "max")
        return self._list_messages(query, n, empty="No emails match.")

    @friendly_errors("read the email")
    def read_email(self, id: str) -> str:
        msg_id = str(id or "").strip()
        if not msg_id:
            raise ToolInputError("id must not be empty.")
        msg = (
            self.gmail.users().messages().get(userId="me", id=msg_id, format="full").execute()
        )
        return format_email(msg, self.tz)

    def _list_messages(self, query: str, n: int, *, empty: str) -> str:
        messages = self.gmail.users().messages()
        resp = messages.list(userId="me", q=query, maxResults=n).execute()
        refs = resp.get("messages") or []
        if not refs:
            return empty
        lines = []
        for ref in refs[:n]:
            msg = messages.get(
                userId="me",
                id=ref["id"],
                format="metadata",
                metadataHeaders=["From", "Subject", "Date"],
            ).execute()
            lines.append(format_email_line(msg, self.tz))
        return text.listing(lines, resp.get("resultSizeEstimate"))


# --------------------------------------------------------------------------- calendar formatting


def build_event_body(
    title: str, start: datetime, end: datetime, tz: tzinfo, description: str | None = None
) -> dict[str, Any]:
    tz_name = getattr(tz, "key", None) or str(tz)
    body: dict[str, Any] = {
        "summary": title,
        "start": {"dateTime": start.isoformat(), "timeZone": tz_name},
        "end": {"dateTime": end.isoformat(), "timeZone": tz_name},
    }
    if description and str(description).strip():
        body["description"] = str(description).strip()
    return body


def format_events(items: list[dict[str, Any]], tz: tzinfo, first: date, days: int) -> str:
    """One line per day that has events, times in words so they are spoken naturally, e.g.
    ``Tue 21 Nov: Dentist, half past nine in the morning to ten o'clock in the morning;
    all day: Mum's birthday``.

    An event that started before the window is listed on the window's first day.
    """
    last = first + timedelta(days=days - 1)
    by_day: dict[date, tuple[list[tuple[datetime, str]], list[str]]] = {}
    for ev in items:
        if ev.get("status") == "cancelled" or _declined(ev):
            continue
        parsed = _event_span(ev, tz)
        if parsed is None:
            continue
        start, end, all_day = parsed
        start_day = start if all_day else start.date()
        day = max(start_day, first)
        if day > last:
            continue
        timed, whole = by_day.setdefault(day, ([], []))
        title = _event_title(ev)
        if all_day:
            span = ""
            if end > start:  # inclusive last day
                span = f" ({day_label(start)} to {day_label(end)})"
            whole.append(f"all day: {title}{span}")
        else:
            timed.append((start, f"{title}, {_format_timed(start, end, day)}"))
    if not by_day:
        return "No events."
    lines = []
    for day in sorted(by_day):
        timed, whole = by_day[day]
        parts = [text for _, text in sorted(timed, key=lambda t: t[0])] + whole
        lines.append(f"{day_label(day)}: " + "; ".join(parts))
    return "\n".join(lines)


def _event_span(ev: dict[str, Any], tz: tzinfo) -> tuple[Any, Any, bool] | None:
    """(start, end, all_day). All-day: dates with an INCLUSIVE end. Timed: local datetimes."""
    s, e = ev.get("start") or {}, ev.get("end") or {}
    try:
        if "date" in s:
            start = date.fromisoformat(s["date"])
            end_excl = date.fromisoformat(e.get("date", s["date"]))
            return start, max(start, end_excl - timedelta(days=1)), True
        if "dateTime" in s:
            start_dt = _parse_dt(s["dateTime"], s.get("timeZone"), tz)
            end_dt = start_dt
            if "dateTime" in e:
                end_dt = _parse_dt(e["dateTime"], e.get("timeZone"), tz)
            return start_dt, end_dt, False
    except (ValueError, KeyError):
        log.debug("Skipping event with unreadable times: %r", ev.get("id"))
    return None


def _parse_dt(value: str, event_tz: str | None, tz: tzinfo) -> datetime:
    dt = datetime.fromisoformat(value)
    if dt.tzinfo is None:  # the API always sends an offset, but be lenient
        dt = dt.replace(tzinfo=ZoneInfo(event_tz) if event_tz else tz)
    return dt.astimezone(tz)


def _format_timed(start: datetime, end: datetime, day: date) -> str:
    def label(dt: datetime, ref: date) -> str:
        said = spoken_time(dt)
        return said if dt.date() == ref else f"{day_label(dt.date())} {said}"

    start_s = label(start, day)
    if end <= start:
        return start_s
    end_s = label(end, start.date())
    # "seven o'clock to eight o'clock in the morning", not "... in the morning" twice.
    for part in _DAY_PARTS:
        if start_s.endswith(part) and end_s.endswith(part) and start.date() == end.date():
            start_s = start_s[: -len(part)]
            break
    return f"{start_s} to {end_s}"


_DAY_PARTS = (" in the morning", " in the afternoon", " in the evening", " at night")


def _event_title(ev: dict[str, Any]) -> str:
    title = speakable(str(ev.get("summary") or "")) or "(no title)"
    location = speakable(str(ev.get("location") or ""))
    if re.match(r"(https?://|www\.)", location, re.I):
        title += " (online)"  # a meeting link is useless read aloud
    elif location:
        title += f" at {clip(location, 40)}"
    return title


def _declined(ev: dict[str, Any]) -> bool:
    return any(
        a.get("self") and a.get("responseStatus") == "declined" for a in ev.get("attendees") or []
    )


# --------------------------------------------------------------------------- email formatting


def format_email_line(msg: dict[str, Any], tz: tzinfo) -> str:
    headers = _headers(msg)
    return text.email_line(
        str(msg.get("id", "?")),
        _email_time(msg, tz),
        _sender_name(headers.get("from", "")),
        headers.get("subject", ""),
        html.unescape(msg.get("snippet", "")),
    )


def format_email(msg: dict[str, Any], tz: tzinfo, limit: int = BODY_CHARS) -> str:
    headers = _headers(msg)
    sender = headers.get("from", "") or "unknown"
    name, addr = parseaddr(sender)
    sender = f"{name} <{addr}>" if name and addr else (addr or name or sender)
    subject = " ".join(headers.get("subject", "").split()) or "(no subject)"
    body = extract_body(msg.get("payload") or {})
    if not body:
        body = html.unescape(msg.get("snippet", "")) or "(no text content)"
    lines = [f"From: {sender}", f"Subject: {subject}"]
    if when := _email_time(msg, tz):
        lines.append(f"Date: {when}")
    # Email content comes from third parties: label it so the model treats it as data.
    lines += [BODY_LABEL, truncate(body, limit)]
    return "\n".join(lines)


def _headers(msg: dict[str, Any]) -> dict[str, str]:
    payload = msg.get("payload") or {}
    return {
        str(h.get("name", "")).lower(): str(h.get("value", ""))
        for h in payload.get("headers") or []
    }


def _sender_name(value: str) -> str:
    name, addr = parseaddr(value)
    return speakable((name or addr or value).strip().strip('"')) or "unknown"


def _email_time(msg: dict[str, Any], tz: tzinfo) -> str:
    try:
        ms = int(msg["internalDate"])
    except (KeyError, TypeError, ValueError):
        return ""
    dt = datetime.fromtimestamp(ms / 1000, tz)
    return day_label(dt.date())  # a clock time per email is noise when read aloud


def extract_body(payload: dict[str, Any]) -> str:
    """Plain-text body of a Gmail ``payload``: text/plain if any, else stripped text/html."""
    plain: list[str] = []
    htmls: list[str] = []
    _collect_parts(payload, plain, htmls)
    text = "\n\n".join(t for t in plain if t.strip())
    if not text.strip() and htmls:
        text = "\n\n".join(html_to_text(h) for h in htmls)
    return clean_text(text)


def _collect_parts(part: dict[str, Any], plain: list[str], htmls: list[str]) -> None:
    mime = str(part.get("mimeType", "")).lower()
    for child in part.get("parts") or []:
        _collect_parts(child, plain, htmls)
    if part.get("filename"):  # attachment
        return
    data = (part.get("body") or {}).get("data")
    if not data or mime not in ("text/plain", "text/html"):
        return
    text = decode_body(data, _charset(part))
    (plain if mime == "text/plain" else htmls).append(text)


def _charset(part: dict[str, Any]) -> str:
    for h in part.get("headers") or []:
        if str(h.get("name", "")).lower() == "content-type":
            m = re.search(r'charset="?([\w.:-]+)', str(h.get("value", "")), re.I)
            if m:
                return m.group(1)
    return "utf-8"


def decode_body(data: str, charset: str = "utf-8") -> str:
    """Decode Gmail's base64url body data (padding optional)."""
    raw_str = data.strip()
    raw_str += "=" * (-len(raw_str) % 4)
    try:
        raw = base64.urlsafe_b64decode(raw_str)
    except (binascii.Error, ValueError):
        return ""
    try:
        return raw.decode(charset, errors="replace")
    except LookupError:
        return raw.decode("utf-8", errors="replace")
