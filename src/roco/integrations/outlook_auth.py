"""Microsoft (Outlook.com / Hotmail) sign-in for ROCO's read-only email tools.

ROCO is a *public client*: the user registers their own app in Microsoft Entra
(no client secret), puts its Application (client) ID in ``outlook.client_id``, and
``roco auth outlook`` signs in with the **device code** flow: it prints a short code
to enter at https://microsoft.com/devicelogin in any browser. That works on a
headless Raspberry Pi too.

The MSAL token cache (refresh token included) is stored in
``~/.roco/outlook_token.json`` and is never committed. ``msal`` is imported inside
functions so that text mode and the tests work without the ``outlook`` extra.
"""

from __future__ import annotations

import contextlib
import logging
import os
import threading
from pathlib import Path
from typing import Any

from roco.config import Config
from roco.paths import resolve

log = logging.getLogger(__name__)

# Read-only mail. MSAL adds openid/profile/offline_access (refresh token) itself and
# refuses them if passed explicitly.
SCOPES: tuple[str, ...] = ("Mail.Read",)

EXTRA_HINT = 'Install the Outlook extra: pip install -e ".[outlook]"'
REAUTH_HINT = 'Run "roco auth outlook" to sign in again.'
SETUP_HINT = 'See "Outlook setup" in the README.'


class OutlookAuthError(RuntimeError):
    """Outlook cannot be used: not configured, not signed in, or the sign-in expired."""


def token_cache_path(cfg: Config) -> Path:
    return resolve(str(cfg.get_path("outlook.token_cache_file", "outlook_token.json")))


def client_id(cfg: Config) -> str:
    value = str(cfg.get_path("outlook.client_id", "") or "").strip()
    if not value:
        raise OutlookAuthError(
            "outlook.client_id is not set. Register an app in Microsoft Entra and put its "
            f"Application (client) ID in config.yaml. {SETUP_HINT}"
        )
    return value


def authority(cfg: Config) -> str:
    return str(
        cfg.get_path("outlook.authority", "") or "https://login.microsoftonline.com/consumers"
    )


def granted(scope_text: str | None) -> bool:
    """True when the token response's ``scope`` includes every scope in :data:`SCOPES`."""
    given = {s.rsplit("/", 1)[-1].lower() for s in (scope_text or "").split()}
    return all(s.lower() in given for s in SCOPES)


class OutlookSession:
    """An MSAL public client bound to the on-disk token cache."""

    def __init__(self, cfg: Config) -> None:
        try:
            import msal
        except ImportError as exc:
            raise OutlookAuthError(f"Microsoft sign-in library missing. {EXTRA_HINT}") from exc
        self.path = token_cache_path(cfg)
        self.cache = msal.SerializableTokenCache()
        if self.path.is_file():
            try:
                self.cache.deserialize(self.path.read_text(encoding="utf-8"))
            except (OSError, ValueError) as exc:
                raise OutlookAuthError(
                    f"Outlook token cache {self.path} is unreadable ({exc}). {REAUTH_HINT}"
                ) from exc
        self.app = msal.PublicClientApplication(
            client_id(cfg), authority=authority(cfg), token_cache=self.cache
        )
        self._lock = threading.Lock()

    def account(self) -> dict[str, Any] | None:
        accounts = self.app.get_accounts()
        return accounts[0] if accounts else None

    def access_token(self) -> str:
        """A valid Graph access token, refreshed silently when needed.

        Raises :class:`OutlookAuthError` when there is no usable sign-in. Network
        failures propagate (``requests`` exceptions are ``OSError``s).
        """
        with self._lock:
            account = self.account()
            if account is None:
                raise OutlookAuthError(f"Not signed in to Outlook. {REAUTH_HINT}")
            result = self.app.acquire_token_silent_with_error(list(SCOPES), account=account)
            self.save()
        if result and "access_token" in result:
            return str(result["access_token"])
        detail = (result or {}).get("error_description") or "no refresh token"
        raise OutlookAuthError(
            f"The Outlook sign-in has expired or was revoked ({_first_line(detail)}). "
            + REAUTH_HINT
        )

    def save(self) -> None:
        if not self.cache.has_state_changed:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_name(self.path.name + ".tmp")
        tmp.write_text(self.cache.serialize(), encoding="utf-8")
        with contextlib.suppress(OSError):  # best effort; Windows ignores most modes
            os.chmod(tmp, 0o600)
        os.replace(tmp, self.path)
        self.cache.has_state_changed = False


def load_session(cfg: Config) -> OutlookSession:
    """A session with a signed-in account, or :class:`OutlookAuthError` (no network)."""
    client_id(cfg)
    path = token_cache_path(cfg)
    if not path.is_file():
        raise OutlookAuthError(f"Not signed in to Outlook (no token at {path}). {REAUTH_HINT}")
    session = OutlookSession(cfg)
    if session.account() is None:
        raise OutlookAuthError(f"No Outlook account in {path}. {REAUTH_HINT}")
    return session


def _first_line(text: str, limit: int = 160) -> str:
    line = str(text).strip().splitlines()[0] if str(text).strip() else ""
    return line if len(line) <= limit else line[: limit - 3] + "..."


# --------------------------------------------------------------------------- roco auth outlook


def run_device_flow(cfg: Config) -> int:
    """``roco auth outlook``: device code sign-in, then save the token cache."""
    try:
        session = OutlookSession(cfg)
    except OutlookAuthError as exc:
        print(exc)
        return 1

    flow = session.app.initiate_device_flow(scopes=list(SCOPES))
    if "user_code" not in flow:
        print("Microsoft refused to start the sign-in: "
              + _first_line(flow.get("error_description") or flow.get("error") or str(flow)))
        print('Check the app registration: "Allow public client flows" must be Yes and it '
              f"must accept personal Microsoft accounts. {SETUP_HINT}")
        return 1

    print("ROCO asks for read-only access to your Outlook mail (Mail.Read).")
    print(flow["message"])  # "To sign in, use a web browser to open ... and enter the code ..."
    print("Waiting for you to finish in the browser...")
    result = session.app.acquire_token_by_device_flow(flow)  # blocks until done or expired
    if "access_token" not in result:
        print("Outlook sign-in failed: "
              + _first_line(result.get("error_description") or result.get("error") or "unknown"))
        return 1
    if not granted(result.get("scope")):
        print(f"Microsoft didn't grant Mail.Read (got: {result.get('scope')}). "
              "Add the Mail.Read delegated permission to the app registration and try again.")
        return 1

    session.save()
    who = (result.get("id_token_claims") or {}).get("preferred_username") or "your account"
    print(f"Signed in as {who}. Token saved to {session.path}")
    if str(cfg.get_path("email.provider", "google")).lower() != "outlook":
        print("Next: set `email: {provider: outlook}` in your config.yaml so ROCO reads it.")
    print("Check everything with: roco doctor")
    return 0
