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
    _extra_check("check_google_extra", ("googleapiclient", "google_auth_oauthlib"), "google"),
]

# Workstream B: microphone, speaker, wake word and Piper voice files.
from roco.voice.checks import VOICE_CHECKS  # noqa: E402  (imports CheckResult from here)

CHECKS += VOICE_CHECKS


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
