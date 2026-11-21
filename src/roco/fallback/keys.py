"""The Anthropic API key: where it is read from and how it is stored.

Lookup order: the ``ANTHROPIC_API_KEY`` environment variable, then the file named by
``cloud.api_key_file`` (default ``~/.roco/anthropic_key``). The key is never written
into the repository or the config file. ``save_api_key`` writes it with owner-only
permissions on Linux (the Pi); on Windows the file inherits the user profile's ACL,
which only that user and administrators can read.
"""

from __future__ import annotations

import logging
import os
import stat
from pathlib import Path

from roco.config import Config
from roco.paths import resolve

log = logging.getLogger(__name__)

ENV_VAR = "ANTHROPIC_API_KEY"
DEFAULT_KEY_FILE = "anthropic_key"


def key_file(cfg: Config) -> Path:
    return resolve(str(cfg.get_path("cloud.api_key_file", DEFAULT_KEY_FILE) or DEFAULT_KEY_FILE))


def _clean(value: str | None) -> str | None:
    value = (value or "").strip()
    return value or None


def load_api_key(cfg: Config) -> str | None:
    """The API key, or None when there is none. Never raises."""
    if key := _clean(os.environ.get(ENV_VAR)):
        return key
    path = key_file(cfg)
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return None
    except OSError as exc:
        log.warning("Cannot read the Anthropic key file %s: %s", path, exc)
        return None
    if os.name == "posix":
        try:
            mode = path.stat().st_mode
            if mode & (stat.S_IRWXG | stat.S_IRWXO):
                log.warning("%s is readable by other users; run: chmod 600 %s", path, path)
        except OSError:
            pass
    return _clean(text.splitlines()[0] if text.strip() else "")


def key_source(cfg: Config) -> str:
    """Where the key would come from, for messages: "env", the file path, or "none"."""
    if _clean(os.environ.get(ENV_VAR)):
        return f"environment variable {ENV_VAR}"
    path = key_file(cfg)
    return str(path) if load_api_key(cfg) else "none"


def save_api_key(cfg: Config, key: str) -> Path:
    """Write the key to the key file with owner-only permissions; returns the path."""
    key = (key or "").strip()
    if not key:
        raise ValueError("empty API key")
    if any(ch.isspace() for ch in key):
        raise ValueError("an API key has no spaces or line breaks")
    path = key_file(cfg)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Create with 0600 from the start so the key is never world-readable, even briefly.
    flags = os.O_WRONLY | os.O_CREAT | os.O_TRUNC
    fd = os.open(path, flags, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        fh.write(key + "\n")
    if os.name == "posix":
        os.chmod(path, 0o600)
    return path


def mask(key: str | None) -> str:
    """``sk-ant-...wxyz`` for display; never print a whole key."""
    if not key:
        return "(none)"
    return f"{key[:7]}...{key[-4:]}" if len(key) > 14 else "***"
