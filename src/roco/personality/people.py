"""Person profiles (spec §7), minimal local store for Phase 3.

Recognition (face/voice embeddings, "ROCO, this is <name>") is Phase 5. For now a profile
is a name, nicknames, how ROCO treats that person (per-person setting overrides) and
whether Adam's own data may be discussed in front of them. Who is in the room is set
by voice ("Yousaf's here", "guest mode"), see :mod:`roco.personality.commands`.

Profiles come from built-in defaults (Adam, Yousaf), then ``~/.roco/people.json``
(Phase 5 writes it), then ``personality.people`` in ``config.yaml``; later sources win
field by field.
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import re
import tempfile
import threading
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from roco.personality.settings import SETTING_KEYS

log = logging.getLogger(__name__)

OWNER = "adam"


@dataclass(frozen=True)
class Person:
    key: str
    name: str
    nicknames: tuple[str, ...] = ()
    # Other ways the speech-to-text may write the name, so "Yusuf's here" still matches.
    aliases: tuple[str, ...] = ()
    # False: ROCO never says ``name``; it uses the nicknames (the Yousaf rule).
    use_name: bool = True
    relationship: str = ""
    # How ROCO treats THIS person, e.g. {"roast": 100}. Keys from SETTING_KEYS.
    overrides: Mapping[str, Any] = field(default_factory=dict)
    # May Adam's calendar, email and other personal data be read out in front of them?
    private_ok: bool = False
    notes: str = ""

    @property
    def is_owner(self) -> bool:
        return self.key == OWNER

    def spoken_names(self) -> tuple[str, ...]:
        """Names ROCO may say out loud."""
        if self.use_name or not self.nicknames:
            return (self.name, *self.nicknames)
        return self.nicknames

    def all_names(self) -> tuple[str, ...]:
        return (self.name, *self.nicknames, *self.aliases)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["nicknames"] = list(self.nicknames)
        data["aliases"] = list(self.aliases)
        data["overrides"] = dict(self.overrides)
        return data

    @classmethod
    def from_dict(cls, key: str, data: Mapping[str, Any]) -> Person:
        overrides = {
            k: v for k, v in dict(data.get("overrides") or {}).items() if k in SETTING_KEYS
        }
        return cls(
            key=normalise_key(key),
            name=str(data.get("name") or key.title()),
            nicknames=tuple(str(n) for n in data.get("nicknames") or ()),
            aliases=tuple(str(n) for n in data.get("aliases") or ()),
            use_name=bool(data.get("use_name", True)),
            relationship=str(data.get("relationship") or ""),
            overrides=overrides,
            private_ok=bool(data.get("private_ok", False)),
            notes=str(data.get("notes") or ""),
        )


BUILTIN_PEOPLE: dict[str, dict[str, Any]] = {
    OWNER: {
        "name": "Adam",
        "relationship": "owner",
        "private_ok": True,
    },
    "yousaf": {
        "name": "Yousaf",
        "nicknames": ["Goose", "Bum"],
        "aliases": ["Yusuf", "Yousef", "Yusef", "Youssef", "Yousuf"],
        "use_name": False,
        "relationship": "Adam's friend",
        "overrides": {"roast": 100},
        "private_ok": False,
        "notes": "Never call him Yousaf. Call him Goose or Bum, mix them up, sometimes both.",
    },
}


def normalise_key(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.strip().lower()).strip("_")


def _merge(base: dict[str, Any], extra: Mapping[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in extra.items():
        if key == "overrides" and isinstance(value, Mapping):
            merged["overrides"] = {**dict(base.get("overrides") or {}), **value}
        else:
            merged[key] = value
    return merged


class PeopleStore:
    """Profiles by key, with lookup by any name, nickname or alias. Thread-safe."""

    def __init__(
        self,
        path: Path | None = None,
        extra: Mapping[str, Mapping[str, Any]] | None = None,
    ) -> None:
        self.path = path
        self._lock = threading.Lock()
        raw: dict[str, dict[str, Any]] = {k: dict(v) for k, v in BUILTIN_PEOPLE.items()}
        for source in (self._read_file(), extra or {}):
            for key, data in source.items():
                if isinstance(data, Mapping):
                    k = normalise_key(key)
                    raw[k] = _merge(raw.get(k, {}), data)
        self._people = {k: Person.from_dict(k, v) for k, v in raw.items()}

    def _read_file(self) -> dict[str, Any]:
        if self.path is None or not self.path.is_file():
            return {}
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            log.warning("Ignoring unreadable people file %s: %s", self.path, exc)
            return {}
        people = data.get("people", data) if isinstance(data, dict) else {}
        return people if isinstance(people, dict) else {}

    def get(self, key: str) -> Person | None:
        with self._lock:
            return self._people.get(normalise_key(key))

    @property
    def owner(self) -> Person:
        person = self.get(OWNER)
        assert person is not None
        return person

    def find(self, name: str) -> Person | None:
        """The profile one of whose names matches ``name`` (case-insensitive)."""
        wanted = normalise_key(name)
        if not wanted:
            return None
        with self._lock:
            for person in self._people.values():
                if any(normalise_key(n) == wanted for n in person.all_names()):
                    return person
        return None

    def all(self) -> list[Person]:
        with self._lock:
            return list(self._people.values())

    def upsert(self, person: Person) -> None:
        with self._lock:
            self._people[person.key] = person
        self.save()

    def forget(self, key: str) -> bool:
        """Delete a profile ("ROCO, forget me"). The owner cannot be forgotten."""
        key = normalise_key(key)
        if key == OWNER:
            return False
        with self._lock:
            removed = self._people.pop(key, None) is not None
        if removed:
            self.save()
        return removed

    def save(self) -> None:
        if self.path is None:
            return
        with self._lock:
            data = {"people": {k: p.to_dict() for k, p in self._people.items()}}
        write_json_atomic(self.path, data)


def write_json_atomic(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=path.name, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2, ensure_ascii=False)
        os.replace(tmp, path)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(tmp)
        raise

