"""Outlook (Microsoft Graph) email tools, with Graph and MSAL fully faked.

Runs without the ``outlook`` extra: ``msal`` is replaced by a fake module in
``sys.modules`` and Graph by a fake client (or ``httpx.MockTransport``).
"""

from __future__ import annotations

import json
import logging
import sys
import types
from zoneinfo import ZoneInfo

import httpx
import pytest

from roco import cli, doctor
from roco.bus import EventBus
from roco.config import load_config
from roco.integrations import google_auth, outlook_auth
from roco.tools import ToolContext, ToolRegistry, build_registry
from roco.tools import google as g
from roco.tools import outlook as o

LONDON = ZoneInfo("Europe/London")
EMAIL_TOOLS = {"get_unread_emails", "search_emails", "read_email"}
LONG_ID = "AQMkADAwATM0MDAAMS1iNTcwLWI2NTEtMDACLTAwCgBGAAAD" + "x" * 80 + "/+="


def outlook_config(**outlook):
    return load_config(overrides={
        "email": {"provider": "outlook"},
        "outlook": {"client_id": "00000000-aaaa-bbbb-cccc-000000000000", **outlook},
    })


# --------------------------------------------------------------------------- fake msal


class FakeMsal:
    """Configurable stand-in for the msal module."""

    def __init__(self):
        self.silent_result = {"access_token": "AT"}
        self.silent_error: Exception | None = None
        self.flow = {"user_code": "ABCD1234", "message": "Go to https://microsoft.com/devicelogin "
                     "and enter the code ABCD1234."}
        self.flow_result = {
            "access_token": "AT",
            "scope": "https://graph.microsoft.com/Mail.Read openid profile",
            "id_token_claims": {"preferred_username": "adam@hotmail.com"},
        }
        self.apps: list = []
        self.interactive: dict | None = None
        self.interactive_error: Exception | None = None
        fake = self

        class SerializableTokenCache:
            def __init__(self):
                self.data = {}
                self.has_state_changed = False

            def serialize(self):
                return json.dumps(self.data)

            def deserialize(self, raw):
                self.data = json.loads(raw)

        class PublicClientApplication:
            def __init__(self, client_id, authority=None, token_cache=None):
                self.client_id, self.authority, self.cache = client_id, authority, token_cache
                fake.apps.append(self)

            def get_accounts(self):
                return self.cache.data.get("accounts", [])

            def acquire_token_silent_with_error(self, scopes, account):
                fake.silent_scopes = scopes
                if fake.silent_error:
                    raise fake.silent_error
                return fake.silent_result

            def initiate_device_flow(self, scopes):
                fake.flow_scopes = scopes
                return fake.flow

            def acquire_token_interactive(self, scopes, **kwargs):
                fake.interactive = {"scopes": scopes, **kwargs}
                if fake.interactive_error:
                    raise fake.interactive_error
                return self.acquire_token_by_device_flow(None)

            def acquire_token_by_device_flow(self, flow):
                if "access_token" in fake.flow_result:
                    self.cache.data["accounts"] = [{"username": "adam@hotmail.com"}]
                    self.cache.has_state_changed = True
                return fake.flow_result

        self.module = types.ModuleType("msal")
        self.module.SerializableTokenCache = SerializableTokenCache
        self.module.PublicClientApplication = PublicClientApplication


@pytest.fixture
def fake_msal(monkeypatch):
    fake = FakeMsal()
    monkeypatch.setitem(sys.modules, "msal", fake.module)
    return fake


def write_cache(home, accounts=({"username": "adam@hotmail.com"},)):
    (home / "outlook_token.json").write_text(
        json.dumps({"accounts": list(accounts)}), encoding="utf-8"
    )


# --------------------------------------------------------------------------- fake Graph


class FakeGraph:
    def __init__(self, routes=None, error=None):
        self.routes = routes or {}
        self.error = error
        self.calls: list[tuple[str, dict, dict]] = []

    def get(self, path, params=None, headers=None):
        self.calls.append((path, params or {}, headers or {}))
        if self.error:
            raise self.error
        if path not in self.routes:
            raise o.GraphError(404, "ErrorItemNotFound", "The specified object was not found.")
        return self.routes[path]


def message(msg_id, name, address, subject, preview, received="2026-11-20T09:15:00Z", **extra):
    return {
        "id": msg_id,
        "subject": subject,
        "from": {"emailAddress": {"name": name, "address": address}},
        "receivedDateTime": received,
        "bodyPreview": preview,
        **extra,
    }


def unread_routes(messages, count=None):
    return {
        "/me/mailFolders/inbox": {"unreadItemCount": len(messages) if count is None else count},
        "/me/mailFolders/inbox/messages": {"value": messages},
    }


def make_tools(graph):
    return o.OutlookTools(graph=graph, tz=LONDON)


# --------------------------------------------------------------------------- registration


def test_not_registered_unless_provider_is_outlook(config, bus, caplog):
    registry = ToolRegistry()
    with caplog.at_level(logging.INFO, logger="roco.tools.outlook"):
        o.register(registry, ToolContext(config=config, bus=bus))
    assert len(registry) == 0
    assert "email.provider is google" in caplog.text


def test_not_registered_without_client_id(bus, caplog):
    registry = ToolRegistry()
    cfg = load_config(overrides={"email": {"provider": "outlook"}})
    with caplog.at_level(logging.WARNING):
        o.register(registry, ToolContext(config=cfg, bus=bus))
    assert len(registry) == 0
    assert "outlook.client_id is not set" in caplog.text


def test_not_registered_without_sign_in(bus, caplog):
    registry = ToolRegistry()
    with caplog.at_level(logging.WARNING):
        o.register(registry, ToolContext(config=outlook_config(), bus=bus))
    assert len(registry) == 0
    assert "roco auth outlook" in caplog.text


def test_not_registered_when_sign_in_expired(fake_msal, isolated_home, bus, caplog):
    write_cache(isolated_home)
    fake_msal.silent_result = {
        "error": "invalid_grant", "error_description": "AADSTS70000: expired"
    }
    registry = ToolRegistry()
    with caplog.at_level(logging.WARNING):
        o.register(registry, ToolContext(config=outlook_config(), bus=bus))
    assert len(registry) == 0
    assert "AADSTS70000" in caplog.text


def test_registered_when_signed_in(fake_msal, isolated_home, bus):
    write_cache(isolated_home)
    registry = ToolRegistry()
    o.register(registry, ToolContext(config=outlook_config(), bus=bus))
    assert set(registry.names()) == EMAIL_TOOLS
    assert fake_msal.silent_scopes == ["Mail.Read"]
    assert fake_msal.apps[0].authority == "https://login.microsoftonline.com/consumers"


def test_registered_when_offline_at_start(fake_msal, isolated_home, bus):
    write_cache(isolated_home)
    fake_msal.silent_error = ConnectionError("no network")
    registry = ToolRegistry()
    o.register(registry, ToolContext(config=outlook_config(), bus=bus))
    assert set(registry.names()) == EMAIL_TOOLS


def test_google_gives_only_calendar_when_provider_is_outlook(monkeypatch, bus):
    monkeypatch.setattr(google_auth, "load_credentials", lambda cfg: object())
    monkeypatch.setattr(g, "_build_services", lambda creds: (object(), object()))
    cfg = load_config(overrides={"google": {"enabled": True}, "email": {"provider": "outlook"}})
    registry = ToolRegistry()
    g.register(registry, ToolContext(config=cfg, bus=bus))
    assert set(registry.names()) == {"get_events", "create_event"}


def test_default_registry_has_no_duplicate_email_tools(config, bus):
    registry = build_registry(ToolContext(config=config, bus=EventBus()))
    assert not EMAIL_TOOLS & set(registry.names())


def test_schemas_documented():
    registry = ToolRegistry()
    o.register_tools(registry, make_tools(FakeGraph()))
    assert set(registry.names()) == EMAIL_TOOLS
    assert not any("send" in n for n in registry.names())
    for tool in registry:
        for prop in tool.parameters["properties"].values():
            assert prop["description"]


# --------------------------------------------------------------------------- listing


def test_unread_emails_listing_and_request():
    graph = FakeGraph(unread_routes([
        message(LONG_ID, "Alice Smith", "alice@x.com", "Lunch \U0001f354?", "Are we on? " * 30),
        message("ID2", "", "bob@y.com", "", ""),
    ], count=12))
    out = make_tools(graph).get_unread_emails("3")
    lines = out.splitlines()
    assert lines[0] == "2 email(s) (about 12 in total):"
    assert lines[1].startswith("- [id o1] Fri 20 Nov 09:15 from Alice Smith: Lunch ? | Are we on?")
    assert lines[1].endswith("...")
    assert lines[2] == "- [id o2] Fri 20 Nov 09:15 from bob@y.com: (no subject)"
    path, params, _ = graph.calls[1]
    assert path == "/me/mailFolders/inbox/messages"
    assert params["$top"] == 3
    assert "isRead eq false" in params["$filter"]
    assert params["$filter"].startswith("receivedDateTime")  # required for $orderby
    assert params["$orderby"] == "receivedDateTime desc"
    assert "bodyPreview" in params["$select"]


def test_unread_emails_empty_and_clamped():
    graph = FakeGraph(unread_routes([]))
    tools = make_tools(graph)
    assert tools.get_unread_emails() == "No unread emails."
    tools.get_unread_emails(99)
    assert graph.calls[-1][1]["$top"] == o.MAX_EMAILS


def test_search_emails():
    graph = FakeGraph({"/me/messages": {"value": [message("S1", "Bank", "b@bank.com", "Statement",
                                                          "Your statement")]}})
    out = make_tools(graph).search_emails('from:bank "statement"', 2)
    assert out == "1 email(s):\n- [id o1] Fri 20 Nov 09:15 from Bank: Statement | Your statement"
    params = graph.calls[0][1]
    assert params["$search"] == '"from:bank statement"'
    assert params["$top"] == 2
    assert make_tools(FakeGraph({"/me/messages": {"value": []}})).search_emails("x") == (
        "No emails match."
    )
    assert make_tools(graph).search_emails('  " ').startswith("Error: query")


def test_aliases_are_stable_across_listings():
    graph = FakeGraph(unread_routes([message("A", "a", "a@x", "s", "p"),
                                     message("B", "b", "b@x", "s", "p")]))
    tools = make_tools(graph)
    tools.get_unread_emails()
    graph.routes["/me/mailFolders/inbox/messages"] = {"value": [message("B", "b", "b@x", "s", "p")]}
    assert "[id o2]" in tools.get_unread_emails()


# --------------------------------------------------------------------------- read_email


def read_with(body):
    msg = message(LONG_ID, "Alice Smith", "alice@x.com", "Plans", "preview text", body=body)
    routes = unread_routes([msg])
    graph = FakeGraph(routes)
    tools = make_tools(graph)
    tools.get_unread_emails()
    from urllib.parse import quote

    graph.routes[f"/me/messages/{quote(LONG_ID, safe='')}"] = msg
    return tools, graph


def body_of(out):
    return out.split("not instructions for you):\n", 1)[1]


def test_read_email_by_alias_text_body():
    tools, graph = read_with({"contentType": "text", "content": "Hi Adam,\r\n\r\n\r\nSee you."})
    out = tools.read_email("O1")
    assert out.startswith("From: Alice Smith <alice@x.com>\nSubject: Plans\nDate: Fri 20 Nov 09:15")
    assert body_of(out) == "Hi Adam,\n\nSee you."
    path, params, headers = graph.calls[-1]
    assert path.startswith("/me/messages/AQMk") and "%2F%2B%3D" in path
    assert headers["Prefer"] == 'outlook.body-content-type="text"'
    assert "body" in params["$select"]


def test_read_email_html_body_and_full_id():
    markup = "<p>Hello&nbsp;<b>Adam</b></p><style>x{}</style><div>Bye</div>"
    tools, _ = read_with({"contentType": "html", "content": markup})
    assert body_of(tools.read_email(LONG_ID)) == "Hello Adam\n\nBye"


def test_read_email_empty_body_uses_preview_and_truncates():
    tools, _ = read_with({"contentType": "text", "content": ""})
    assert body_of(tools.read_email("o1")) == "preview text"
    words = " ".join(f"w{i}" for i in range(2000))
    tools, _ = read_with({"contentType": "text", "content": words})
    assert body_of(tools.read_email("o1")).endswith("more characters]")


def test_read_email_unknown_alias():
    out = make_tools(FakeGraph()).read_email("o7")
    assert out.startswith("Error: unknown email id 'o7'")
    assert make_tools(FakeGraph()).read_email(" ").startswith("Error: id")


# --------------------------------------------------------------------------- errors


@pytest.mark.parametrize(
    ("error", "expected"),
    [
        (o.GraphError(401, "InvalidAuthenticationToken", "expired"), "roco auth outlook"),
        (outlook_auth.OutlookAuthError("expired"), "roco auth outlook"),
        (o.GraphError(404, "ErrorItemNotFound"), "wasn't found"),
        (o.GraphError(400, "ErrorInvalidIdMalformed", "Id is malformed."), "wasn't found"),
        (o.GraphError(403, "ErrorAccessDenied"), "refused access"),
        (o.GraphError(429, "TooManyRequests"), "rate-limiting"),
        (o.GraphError(503, "ServiceUnavailable"), "trouble"),
        (o.GraphError(400, "BadRequest", "Invalid  filter"),
         "rejected the request (400: Invalid filter)"),
        (httpx.ConnectError("dns"), "internet connection"),
        (httpx.ReadTimeout("slow"), "internet connection"),
        (ConnectionError("down"), "internet connection"),
        (ValueError("odd"), "Unexpected problem: ValueError"),
    ],
)
def test_describe_error(error, expected):
    assert expected in o.describe_error(error)


def test_tool_errors_are_one_line():
    tools = make_tools(FakeGraph(error=o.GraphError(401, "InvalidAuthenticationToken")))
    out = tools.get_unread_emails()
    assert out.startswith("Error: couldn't check email. The Outlook sign-in has expired")
    assert "\n" not in out


# --------------------------------------------------------------------------- Graph client


def test_graph_client_sends_token_and_parses_errors():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["auth"] = request.headers["Authorization"]
        seen["url"] = str(request.url)
        if request.url.path.endswith("/bad"):
            return httpx.Response(404, json={"error": {"code": "ErrorItemNotFound",
                                                       "message": "Not found."}})
        return httpx.Response(200, json={"value": []})

    client = o.GraphClient(lambda: "TOKEN", transport=httpx.MockTransport(handler))
    assert client.get("/me/messages", params={"$top": 2}) == {"value": []}
    assert seen["auth"] == "Bearer TOKEN"
    assert seen["url"].startswith("https://graph.microsoft.com/v1.0/me/messages?")
    with pytest.raises(o.GraphError) as exc:
        client.get("/me/messages/bad")
    assert exc.value.status == 404 and exc.value.code == "ErrorItemNotFound"


def test_graph_client_token_failure_becomes_auth_message():
    def expired():
        raise outlook_auth.OutlookAuthError("expired")

    client = o.GraphClient(expired, transport=httpx.MockTransport(lambda r: httpx.Response(200)))
    out = make_tools(client).get_unread_emails()
    assert "roco auth outlook" in out


# --------------------------------------------------------------------------- roco auth outlook


def test_auth_outlook_without_client_id(capsys, fake_msal):
    assert cli.main(["auth", "outlook"]) == cli.EXIT_FAIL
    assert "outlook.client_id is not set" in capsys.readouterr().out


def write_config(tmp_path, provider="google"):
    (tmp_path / "config.yaml").write_text(
        "outlook:\n  client_id: 00000000-aaaa-bbbb-cccc-000000000000\n"
        f"email:\n  provider: {provider}\n",
        encoding="utf-8",
    )


def test_auth_outlook_success(capsys, fake_msal, tmp_path, isolated_home):
    write_config(tmp_path)
    assert cli.main(["auth", "outlook"]) == cli.EXIT_OK
    out = capsys.readouterr().out
    assert "ABCD1234" in out and "devicelogin" in out
    assert "Signed in as adam@hotmail.com" in out
    assert "provider: outlook" in out
    assert fake_msal.flow_scopes == ["Mail.Read"]
    saved = json.loads((isolated_home / "outlook_token.json").read_text(encoding="utf-8"))
    assert saved["accounts"][0]["username"] == "adam@hotmail.com"


def test_auth_outlook_flow_refused(capsys, fake_msal, tmp_path, isolated_home):
    write_config(tmp_path)
    fake_msal.flow = {"error": "invalid_client",
                      "error_description": "AADSTS7000218: client_assertion required"}
    assert cli.main(["auth", "outlook"]) == cli.EXIT_FAIL
    out = capsys.readouterr().out
    assert "AADSTS7000218" in out and "Allow public client flows" in out
    assert not (isolated_home / "outlook_token.json").exists()


def test_auth_outlook_without_mail_scope(capsys, fake_msal, tmp_path, isolated_home):
    write_config(tmp_path)
    fake_msal.flow_result = {**fake_msal.flow_result, "scope": "openid profile User.Read"}
    assert cli.main(["auth", "outlook"]) == cli.EXIT_FAIL
    assert "Mail.Read" in capsys.readouterr().out
    assert not (isolated_home / "outlook_token.json").exists()


def test_auth_outlook_denied(capsys, fake_msal, tmp_path):
    write_config(tmp_path)
    fake_msal.flow_result = {"error": "authorization_declined",
                             "error_description": "The user declined."}
    assert cli.main(["auth", "outlook"]) == cli.EXIT_FAIL
    assert "declined" in capsys.readouterr().out


def test_auth_outlook_device_failure_suggests_browser(capsys, fake_msal, tmp_path):
    write_config(tmp_path)
    fake_msal.flow_result = {"error": "invalid_request",
                             "error_description": "The application is a first party application"}
    assert cli.main(["auth", "outlook"]) == cli.EXIT_FAIL
    assert "roco auth outlook --browser" in capsys.readouterr().out


def test_auth_outlook_browser_success(capsys, fake_msal, tmp_path, isolated_home):
    write_config(tmp_path, provider="outlook")
    assert cli.main(["auth", "outlook", "--browser"]) == cli.EXIT_OK
    out = capsys.readouterr().out
    assert "Signed in as adam@hotmail.com" in out
    assert "provider: outlook" not in out  # already configured
    assert fake_msal.interactive["scopes"] == ["Mail.Read"]
    assert fake_msal.interactive["prompt"] == "select_account"
    assert (isolated_home / "outlook_token.json").exists()


def test_auth_outlook_browser_errors(capsys, fake_msal, tmp_path, isolated_home):
    write_config(tmp_path)
    fake_msal.interactive_error = RuntimeError("could not open a browser")
    assert cli.main(["auth", "outlook", "--browser"]) == cli.EXIT_FAIL
    assert "could not open a browser" in capsys.readouterr().out
    fake_msal.interactive_error = None
    fake_msal.flow_result = {"error": "invalid_request",
                             "error_description": "The redirect URI is not valid."}
    assert cli.main(["auth", "outlook", "--browser"]) == cli.EXIT_FAIL
    assert "http://localhost" in capsys.readouterr().out
    assert not (isolated_home / "outlook_token.json").exists()


def test_granted_scope_matching():
    assert outlook_auth.granted("https://graph.microsoft.com/Mail.Read openid")
    assert outlook_auth.granted("mail.read")
    assert not outlook_auth.granted("Mail.ReadWrite")
    assert not outlook_auth.granted(None)


# --------------------------------------------------------------------------- doctor

OUTLOOK_CHECKS = [doctor.check_outlook_extra, doctor.check_outlook_signin]


def test_doctor_outlook_registered_and_optional_by_default(config):
    for check in OUTLOOK_CHECKS:
        assert check in doctor.CHECKS
    assert all(r.ok or r.optional for r in doctor.run_checks(config, OUTLOOK_CHECKS))


def test_doctor_outlook_required_when_provider_outlook(fake_msal, isolated_home):
    cfg = outlook_config()
    results = {r.name: r for r in doctor.run_checks(cfg, OUTLOOK_CHECKS)}
    assert not results["Outlook sign-in"].ok and not results["Outlook sign-in"].optional
    write_cache(isolated_home)
    results = {r.name: r for r in doctor.run_checks(cfg, OUTLOOK_CHECKS)}
    assert results["Outlook sign-in"].ok
    assert results["Outlook sign-in"].detail == "adam@hotmail.com"
