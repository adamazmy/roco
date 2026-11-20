"""``roco doctor``: environment checks with actionable hints.

Each check returns a :class:`CheckResult`. Workstreams add their own checks by
appending to :data:`CHECKS` (B: mic/speaker devices, voice files; D: Google token).
"""

from __future__ import annotations

import importlib.util
import sys
from collections.abc import Callable
from dataclasses import dataclass

import httpx

from roco.config import Config


@dataclass
class CheckResult:
    name: str
    ok: bool
    detail: str = ""
    hint: str = ""
    optional: bool = False  # a failing optional check does not fail `roco doctor`


Check = Callable[[Config], CheckResult]


def check_python(cfg: Config) -> CheckResult:
    v = sys.version_info
    return CheckResult(
        "Python >= 3.11", v >= (3, 11), f"{v.major}.{v.minor}.{v.micro}",
        hint="Install Python 3.11 or newer.",
    )


def check_config(cfg: Config) -> CheckResult:
    src = str(cfg.source) if cfg.source else "built-in defaults (no config.yaml found)"
    return CheckResult("Config", True, src)


def check_ollama(cfg: Config) -> CheckResult:
    host = str(cfg.get_path("llm.host")).rstrip("/")
    model = str(cfg.get_path("llm.model"))
    try:
        resp = httpx.get(f"{host}/api/tags", timeout=3.0)
        resp.raise_for_status()
        names = {m.get("name", "") for m in resp.json().get("models", [])}
    except (httpx.HTTPError, ValueError) as exc:
        return CheckResult(
            "Ollama reachable", False, f"{host}: {exc.__class__.__name__}",
            hint="Install Ollama from https://ollama.com and make sure it is running.",
        )
    pulled = model in names or f"{model}:latest" in names
    return CheckResult(
        f"Model {model} pulled", pulled, host,
        hint=f"Run: ollama pull {model}",
    )


def _extra_check(name: str, modules: tuple[str, ...], extra: str) -> Check:
    def check(cfg: Config) -> CheckResult:
        missing = [m for m in modules if importlib.util.find_spec(m) is None]
        return CheckResult(
            f"Extra [{extra}] installed", not missing,
            "missing: " + ", ".join(missing) if missing else "ok",
            hint=f'pip install -e ".[{extra}]"', optional=True,
        )

    check.__name__ = name
    return check


CHECKS: list[Check] = [
    check_python,
    check_config,
    check_ollama,
    _extra_check("check_voice_extra",
                 ("sounddevice", "openwakeword", "faster_whisper", "piper"), "voice"),
    _extra_check("check_face_extra", ("pygame",), "face"),
    # Google checks (workstream D) are appended below.
]


# --------------------------------------------------------------------------- Google (D)
# All optional while google.enabled is false, so a text-only setup stays green.

_GOOGLE_MODULES = ("googleapiclient", "google_auth_oauthlib", "google_auth_httplib2")


def _google_enabled(cfg: Config) -> bool:
    return bool(cfg.get_path("google.enabled", False))


def check_google_enabled(cfg: Config) -> CheckResult:
    on = _google_enabled(cfg)
    return CheckResult(
        "Google enabled", on, "google.enabled is " + ("true" if on else "false"),
        hint="After 'roco auth google', set google.enabled: true in config.yaml.",
        optional=True,
    )


def check_google_extra(cfg: Config) -> CheckResult:
    result = _extra_check("check_google_extra", _GOOGLE_MODULES, "google")(cfg)
    result.optional = not _google_enabled(cfg)
    return result


def check_google_client_secrets(cfg: Config) -> CheckResult:
    from roco.integrations import google_auth

    path = google_auth.credentials_path(cfg)
    try:
        google_auth.read_client_secrets(path)
    except google_auth.GoogleAuthError as exc:
        return CheckResult(
            "Google client secrets", False, str(exc),
            hint='See "Google setup" in the README.', optional=not _google_enabled(cfg),
        )
    return CheckResult("Google client secrets", True, str(path))


def check_google_token(cfg: Config) -> CheckResult:
    from roco.integrations import google_auth

    optional = not _google_enabled(cfg)
    path = google_auth.token_path(cfg)
    try:
        google_auth.read_token_info(path)
        if all(importlib.util.find_spec(m) is not None for m in _GOOGLE_MODULES):
            # Refreshes an expired access token: catches revoked / 7-day-expired tokens.
            google_auth.load_credentials(cfg)
    except google_auth.GoogleAuthError as exc:
        return CheckResult(
            "Google token", False, str(exc), hint="Run: roco auth google", optional=optional
        )
    return CheckResult("Google token", True, str(path), optional=optional)


CHECKS += [check_google_enabled, check_google_extra, check_google_client_secrets,
           check_google_token]


# --------------------------------------------------------------------------- Outlook (D)
# Only required when email.provider is outlook.


def _outlook_active(cfg: Config) -> bool:
    return str(cfg.get_path("email.provider", "google")).lower() == "outlook"


def check_outlook_extra(cfg: Config) -> CheckResult:
    result = _extra_check("check_outlook_extra", ("msal",), "outlook")(cfg)
    result.optional = not _outlook_active(cfg)
    return result


def check_outlook_signin(cfg: Config) -> CheckResult:
    from roco.integrations import outlook_auth

    optional = not _outlook_active(cfg)
    try:
        session = outlook_auth.load_session(cfg)
        session.access_token()  # refreshes: catches an expired or revoked sign-in
    except outlook_auth.OutlookAuthError as exc:
        return CheckResult(
            "Outlook sign-in", False, str(exc),
            hint="Set outlook.client_id, then run: roco auth outlook", optional=optional,
        )
    except OSError as exc:
        return CheckResult(
            "Outlook sign-in", False, f"couldn't reach Microsoft ({exc.__class__.__name__})",
            hint="Check the internet connection.", optional=True,
        )
    who = (session.account() or {}).get("username", "")
    return CheckResult("Outlook sign-in", True, who or str(session.path), optional=optional)


CHECKS += [check_outlook_extra, check_outlook_signin]


def run_checks(cfg: Config, checks: list[Check] | None = None) -> list[CheckResult]:
    results = []
    for check in checks or CHECKS:
        try:
            results.append(check(cfg))
        except Exception as exc:  # a broken check must not hide the others
            results.append(CheckResult(getattr(check, "__name__", "check"), False, repr(exc)))
    return results


def print_report(results: list[CheckResult]) -> bool:
    """Print results; return True if every non-optional check passed."""
    all_ok = True
    for r in results:
        mark = "OK  " if r.ok else ("WARN" if r.optional else "FAIL")
        line = f"[{mark}] {r.name}"
        if r.detail:
            line += f" - {r.detail}"
        print(line)
        if not r.ok and r.hint:
            print(f"       -> {r.hint}")
        all_ok &= r.ok or r.optional
    return all_ok
