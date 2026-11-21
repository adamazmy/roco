"""Google OAuth for ROCO: the ``roco auth google`` flow and loading the saved token.

ROCO is an "installed app": the user creates their own Desktop OAuth client in a
Google Cloud project, saves its JSON as ``~/.roco/google_credentials.json``, and
``roco auth google`` opens the browser to sign in. The resulting token (with a
refresh token) is stored in ``~/.roco/google_token.json``. Neither file is ever
committed.

The Google libraries are imported inside functions so that this module, and the
checks built on it, work without the ``google`` extra installed.
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
from pathlib import Path
from typing import Any

from roco.config import Config
from roco.paths import resolve

log = logging.getLogger(__name__)

# Calendar: read + create events. Gmail: read only. ROCO never sends email.
SCOPES: tuple[str, ...] = (
    "https://www.googleapis.com/auth/calendar.events",
    "https://www.googleapis.com/auth/gmail.readonly",
)

EXTRA_HINT = 'Install the Google extra: pip install -e ".[google]"'
REAUTH_HINT = 'Run "roco auth google" to sign in again.'


class GoogleAuthError(RuntimeError):
    """Google cannot be used: no token, a bad token, or a missing dependency."""


def credentials_path(cfg: Config) -> Path:
    """The OAuth client secrets JSON downloaded from Google Cloud."""
    return resolve(str(cfg.get_path("google.credentials_file", "google_credentials.json")))


def token_path(cfg: Config) -> Path:
    """Where the signed-in user's token is stored."""
    return resolve(str(cfg.get_path("google.token_file", "google_token.json")))


# --------------------------------------------------------------------------- files


def read_client_secrets(path: Path) -> dict[str, Any]:
    """Validate the client secrets file; raise :class:`GoogleAuthError` with a hint."""
    if not path.is_file():
        raise GoogleAuthError(
            f"Google client secrets not found at {path}. Create a Desktop app OAuth client "
            "in Google Cloud, download its JSON and save it there "
            '(see "Google setup" in the README).'
        )
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise GoogleAuthError(f"{path} is not valid JSON ({exc}).") from exc
    if not isinstance(data, dict):
        raise GoogleAuthError(f"{path} is not an OAuth client secrets file.")
    if "installed" not in data:
        kind = "a Web application" if "web" in data else "not a Desktop app"
        raise GoogleAuthError(
            f"{path} is {kind} OAuth client. Create an OAuth client of type "
            '"Desktop app" and download its JSON instead.'
        )
    return data


def read_token_info(path: Path) -> dict[str, Any]:
    """Read and sanity-check the saved token JSON (no network, no Google libraries)."""
    if not path.is_file():
        raise GoogleAuthError(f"Not signed in to Google (no token at {path}). {REAUTH_HINT}")
    try:
        info = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise GoogleAuthError(f"Google token {path} is unreadable ({exc}). {REAUTH_HINT}") from exc
    if not isinstance(info, dict) or not info.get("refresh_token"):
        raise GoogleAuthError(f"Google token {path} has no refresh token. {REAUTH_HINT}")
    granted = info.get("scopes")
    if isinstance(granted, str):
        granted = granted.split()
    if granted is not None:
        missing = [s for s in SCOPES if s not in granted]
        if missing:
            raise GoogleAuthError(
                f"Google token is missing permission(s): {', '.join(missing)}. {REAUTH_HINT}"
            )
    return info


def save_token(creds: Any, path: Path) -> None:
    """Write the token atomically, readable only by the current user where supported."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(creds.to_json(), encoding="utf-8")
    with contextlib.suppress(OSError):  # best effort; Windows ignores most modes
        os.chmod(tmp, 0o600)
    os.replace(tmp, path)


# --------------------------------------------------------------------------- credentials


def load_credentials(cfg: Config, *, refresh: bool = True) -> Any:
    """Return usable ``google.oauth2.credentials.Credentials`` or raise :class:`GoogleAuthError`.

    An expired access token is refreshed (and the token file updated). If the
    refresh is rejected by Google (revoked, or the 7-day expiry of apps in
    "Testing" mode) this raises; if Google simply can't be reached, the
    credentials are returned anyway so the tools can report a network error later.
    """
    path = token_path(cfg)
    info = read_token_info(path)
    try:
        from google.auth.exceptions import RefreshError, TransportError
        from google.auth.transport.requests import Request
        from google.oauth2.credentials import Credentials
    except ImportError as exc:
        raise GoogleAuthError(f"Google libraries not installed ({exc}). {EXTRA_HINT}") from exc

    try:
        creds = Credentials.from_authorized_user_info(info)
    except ValueError as exc:
        raise GoogleAuthError(f"Google token {path} is invalid ({exc}). {REAUTH_HINT}") from exc

    if creds.valid or not refresh:
        return creds
    try:
        creds.refresh(Request())
    except RefreshError as exc:
        raise GoogleAuthError(
            f"Google sign-in has expired or was revoked ({_short(exc)}). {REAUTH_HINT}"
        ) from exc
    except (TransportError, OSError) as exc:
        log.warning("Could not reach Google to refresh the token: %s", _short(exc))
        return creds
    save_token(creds, path)
    log.debug("Refreshed Google access token")
    return creds


def _short(exc: BaseException, limit: int = 160) -> str:
    text = str(exc).strip() or exc.__class__.__name__
    return text if len(text) <= limit else text[: limit - 3] + "..."


# --------------------------------------------------------------------------- roco auth google


def run_auth_flow(cfg: Config, *, open_browser: bool = True) -> int:
    """``roco auth google``: browser sign-in, then save the token. Returns an exit code."""
    secrets = credentials_path(cfg)
    try:
        read_client_secrets(secrets)
    except GoogleAuthError as exc:
        print(exc)
        return 1
    try:
        from google_auth_oauthlib.flow import InstalledAppFlow
    except ImportError:
        print(f"The Google libraries are not installed. {EXTRA_HINT}")
        return 1

    print("Opening your browser to sign in to Google.")
    print("ROCO asks for: your calendar events (read + create) and Gmail (read only).")
    print("If the browser doesn't open, copy the link printed below into it.")
    # Google lets the user untick permissions; without this oauthlib raises a bare
    # "Scope has changed" warning. We check the granted scopes ourselves below.
    os.environ.setdefault("OAUTHLIB_RELAX_TOKEN_SCOPE", "1")
    flow = InstalledAppFlow.from_client_secrets_file(str(secrets), scopes=list(SCOPES))
    try:
        # prompt=consent makes Google return a refresh token even on a repeat sign-in.
        creds = flow.run_local_server(
            port=0,
            open_browser=open_browser,
            prompt="consent",
            authorization_prompt_message="Sign-in link: {url}",
            success_message="ROCO is connected to Google. You can close this tab.",
        )
    except Exception as exc:  # oauthlib raises many types (denied, state mismatch...)
        print(f"Google sign-in failed: {_short(exc, 300)}")
        return 1

    granted = getattr(creds, "granted_scopes", None) or creds.scopes or SCOPES
    if isinstance(granted, str):
        granted = granted.split()
    missing = [s for s in SCOPES if s not in granted]
    if missing:
        print(
            "Google didn't grant every permission (missing: " + ", ".join(missing) + "). "
            "Run 'roco auth google' again and tick all the boxes."
        )
        return 1

    path = token_path(cfg)
    save_token(creds, path)
    print(f"Signed in. Token saved to {path}")
    if not cfg.get_path("google.enabled", False):
        print("Next: set `google: {enabled: true}` in your config.yaml so ROCO uses it.")
    print("Check everything with: roco doctor")
    return 0
