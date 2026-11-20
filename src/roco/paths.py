"""Where ROCO keeps its per-user state (history, reminders, tokens, models).

Everything lives under ``~/.roco`` unless the ``ROCO_HOME`` environment variable
points elsewhere (tests use this to stay out of the real home directory).
Nothing in here is ever committed to the repository.
"""

from __future__ import annotations

import os
from pathlib import Path


def roco_home() -> Path:
    """Return the ROCO state directory, creating it if needed."""
    override = os.environ.get("ROCO_HOME")
    home = Path(override).expanduser() if override else Path.home() / ".roco"
    home.mkdir(parents=True, exist_ok=True)
    return home


def models_dir() -> Path:
    """Directory for downloaded wake-word and voice models."""
    path = roco_home() / "models"
    path.mkdir(parents=True, exist_ok=True)
    return path


def resolve(path: str | os.PathLike[str]) -> Path:
    """Resolve a config path: absolute paths stay as-is, relative ones live in ROCO home."""
    p = Path(path).expanduser()
    return p if p.is_absolute() else roco_home() / p
