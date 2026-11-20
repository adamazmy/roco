"""Outlook.com / Hotmail email tools via Microsoft Graph. Read-only: never sends email.

Owner: workstream D.

:func:`register` adds the email tools only when ``email.provider`` is ``outlook`` and
a signed-in account exists (``roco auth outlook``). They use the same names as the
Gmail tools (``get_unread_emails``, ``search_emails``, ``read_email``), so the model
always sees one email tool set; Google then provides only the calendar tools.

Graph message ids are ~150 characters, which a small model copies badly, so the
listings hand out short aliases (``o1``, ``o2``...) that :meth:`OutlookTools.read_email`
maps back. Graph is called with ``httpx`` (a core dependency); only the sign-in needs
the ``outlook`` extra (``msal``).
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, tzinfo
from typing import Any
from urllib.parse import quote
from zoneinfo import ZoneInfo

import httpx

from roco.integrations import outlook_auth, text
from roco.integrations.text import (
    BODY_CHARS,
    BODY_LABEL,
    ToolInputError,
    clamp_int,
    clean_text,
    day_label,
    html_to_text,
    truncate,
)
from roco.tools import ToolContext
from roco.tools.registry import ToolRegistry

log = logging.getLogger(__name__)

GRAPH_URL = "https://graph.microsoft.com/v1.0"
HTTP_TIMEOUT_S = 20
MAX_EMAILS = 10
LIST_FIELDS = "id,subject,from,receivedDateTime,bodyPreview"

# --------------------------------------------------------------------------- registration


def register(registry: ToolRegistry, ctx: ToolContext) -> None:
    cfg = ctx.config
    provider = str(cfg.get_path("email.provider", "google")).lower()
    if provider != "outlook":
        log.info("Outlook tools not registered: email.provider is %s", provider)
        return
    try:
        session = outlook_auth.load_session(cfg)
        session.access_token()  # refreshes if needed; fails if the sign-in is dead
    except outlook_auth.OutlookAuthError as exc:
        log.warning("Outlook tools not registered: %s", exc)
        return
    except OSError as exc:  # offline at start-up: tools will report the network error
        log.warning("Could not reach Microsoft to check the Outlook sign-in: %s", exc)
    graph = GraphClient(session.access_token)
    tz = ZoneInfo(str(cfg.get_path("assistant.timezone", "Europe/London")))
    register_tools(registry, OutlookTools(graph=graph, tz=tz))
    log.info("Outlook email tools registered")


def register_tools(registry: ToolRegistry, tools: OutlookTools) -> None:
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
            "Search emails. Use plain words or 'from:alice', 'subject:invoice'. Returns "
            "sender, subject, snippet and id."
        ),
        parameters={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search words."},
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
            "properties": {"id": {"type": "string", "description": "Email id, e.g. o1."}},
            "required": ["id"],
        },
    )
    def read_email(id: str) -> str:
        return tools.read_email(id)


# --------------------------------------------------------------------------- Graph client


class GraphError(Exception):
    def __init__(self, status: int, code: str = "", message: str = "") -> None:
        self.status, self.code, self.message = status, code, message
        super().__init__(f"Graph {status} {code}: {message}")


class GraphClient:
    """Minimal authenticated GET client for Microsoft Graph."""

    def __init__(
        self,
        token_provider: Callable[[], str],
        *,
        timeout: float = HTTP_TIMEOUT_S,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self._token = token_provider
        self._http = httpx.Client(base_url=GRAPH_URL, timeout=timeout, transport=transport)

    def get(
        self, path: str, params: dict[str, Any] | None = None, headers: dict[str, str] | None = None
    ) -> dict[str, Any]:
        all_headers = {"Authorization": f"Bearer {self._token()}", **(headers or {})}
        resp = self._http.get(path, params=params, headers=all_headers)
        if resp.status_code >= 400:
            try:
                err = resp.json().get("error") or {}
            except ValueError:
                err = {}
            code, message = str(err.get("code", "")), str(err.get("message", ""))
            raise GraphError(resp.status_code, code, message)
        return resp.json()


# --------------------------------------------------------------------------- errors

AUTH_EXPIRED = (
    "The Outlook sign-in has expired or was revoked. Adam needs to run 'roco auth outlook' again."
)


def describe_error(exc: BaseException) -> str:
    if isinstance(exc, outlook_auth.OutlookAuthError):
        return AUTH_EXPIRED
    if isinstance(exc, GraphError):
        code = exc.code.lower()
        if exc.status == 401:
            return AUTH_EXPIRED
        if exc.status == 404 or "itemnotfound" in code or "idmalformed" in code:
            return "That email wasn't found. Use an id from get_unread_emails or search_emails."
        if exc.status == 403:
            return "Outlook refused access (403). Adam may need to run 'roco auth outlook' again."
        if exc.status == 429:
            return "Outlook is rate-limiting requests. Try again in a minute."
        if exc.status >= 500:
            return "Outlook is having trouble right now. Try again shortly."
        reason = text.clip(" ".join(exc.message.split()), 120)
        return f"Outlook rejected the request ({exc.status}{': ' + reason if reason else ''})."
    if isinstance(exc, (httpx.TransportError, OSError, TimeoutError)):
        return "Couldn't reach Outlook. Check the internet connection."
    return f"Unexpected problem: {exc.__class__.__name__}."


def friendly_errors(action: str) -> Callable[[Callable[..., str]], Callable[..., str]]:
    return text.friendly_errors(action, describe_error, log)


# --------------------------------------------------------------------------- tools


@dataclass
class OutlookTools:
    graph: Any  # anything with .get(path, params=None, headers=None) -> dict
    tz: tzinfo
    _ids: dict[str, str] = field(default_factory=dict)  # alias -> Graph id
    _aliases: dict[str, str] = field(default_factory=dict)  # Graph id -> alias

    @friendly_errors("check email")
    def get_unread_emails(self, max: int | str = 5) -> str:
        n = clamp_int(max, 1, MAX_EMAILS, "max")
        folder = self.graph.get("/me/mailFolders/inbox", params={"$select": "unreadItemCount"})
        # Graph only sorts a filtered list if the $orderby property is filtered on first.
        resp = self.graph.get(
            "/me/mailFolders/inbox/messages",
            params={
                "$filter": "receivedDateTime ge 1900-01-01T00:00:00Z and isRead eq false",
                "$orderby": "receivedDateTime desc",
                "$top": n,
                "$select": LIST_FIELDS,
            },
        )
        messages = resp.get("value") or []
        if not messages:
            return "No unread emails."
        return text.listing([self._line(m) for m in messages[:n]], folder.get("unreadItemCount"))

    @friendly_errors("search email")
    def search_emails(self, query: str, max: int | str = 5) -> str:
        words = " ".join(str(query or "").replace('"', " ").split())
        if not words:
            raise ToolInputError("query must not be empty.")
        n = clamp_int(max, 1, MAX_EMAILS, "max")
        resp = self.graph.get(
            "/me/messages", params={"$search": f'"{words}"', "$top": n, "$select": LIST_FIELDS}
        )
        messages = resp.get("value") or []
        if not messages:
            return "No emails match."
        return text.listing([self._line(m) for m in messages[:n]])

    @friendly_errors("read the email")
    def read_email(self, id: str) -> str:
        key = str(id or "").strip()
        if not key:
            raise ToolInputError("id must not be empty.")
        graph_id = self._ids.get(key.lower(), key)
        if graph_id == key and len(key) < 20:  # an alias we never handed out
            raise ToolInputError(
                f"unknown email id {key!r}. Call get_unread_emails or search_emails first."
            )
        msg = self.graph.get(
            f"/me/messages/{quote(graph_id, safe='')}",
            params={"$select": "subject,from,receivedDateTime,body,bodyPreview"},
            headers={"Prefer": 'outlook.body-content-type="text"'},
        )
        return format_email(msg, self.tz)

    def _line(self, msg: dict[str, Any]) -> str:
        return text.email_line(
            self._alias(str(msg.get("id", ""))),
            _received(msg, self.tz),
            _sender(msg, with_address=False),
            str(msg.get("subject") or ""),
            str(msg.get("bodyPreview") or ""),
        )

    def _alias(self, graph_id: str) -> str:
        if graph_id not in self._aliases:
            alias = f"o{len(self._aliases) + 1}"
            self._aliases[graph_id] = alias
            self._ids[alias] = graph_id
        return self._aliases[graph_id]


# --------------------------------------------------------------------------- formatting


def format_email(msg: dict[str, Any], tz: tzinfo, limit: int = BODY_CHARS) -> str:
    body = msg.get("body") or {}
    content = str(body.get("content") or "")
    if str(body.get("contentType", "")).lower() == "html":
        content = html_to_text(content)
    content = clean_text(content) or clean_text(str(msg.get("bodyPreview") or ""))
    lines = [
        f"From: {_sender(msg, with_address=True)}",
        f"Subject: {' '.join(str(msg.get('subject') or '').split()) or '(no subject)'}",
    ]
    if when := _received(msg, tz):
        lines.append(f"Date: {when}")
    lines += [BODY_LABEL, truncate(content or "(no text content)", limit)]
    return "\n".join(lines)


def _sender(msg: dict[str, Any], *, with_address: bool) -> str:
    addr = (msg.get("from") or {}).get("emailAddress") or {}
    name, email = str(addr.get("name") or "").strip(), str(addr.get("address") or "").strip()
    if with_address and name and email and name != email:
        return f"{name} <{email}>"
    return name or email or "unknown"


def _received(msg: dict[str, Any], tz: tzinfo) -> str:
    try:
        dt = datetime.fromisoformat(str(msg["receivedDateTime"])).astimezone(tz)
    except (KeyError, ValueError):
        return ""
    return f"{day_label(dt.date())} {dt:%H:%M}"
