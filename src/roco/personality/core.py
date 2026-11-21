"""The live personality: settings, who is in the room, voice commands and the system prompt.

One :class:`Personality` lives in the brain. Sources of the settings, later wins:

1. the preset named in ``personality.preset`` (``config.yaml``),
2. explicit levels in the ``personality`` config section,
3. changes made by voice or by a ``settings`` protocol message, saved to
   ``~/.roco/personality.json`` (``personality.file``) so they survive a restart.
   "ROCO, back to normal" deletes them.

Who is in the room (``Company``) is session state, never saved.

Protocol: :meth:`snapshot` is the ``{"personality": {...}}`` shape of a ``settings``
message (docs/PROTOCOL.md); :meth:`apply_settings` takes one; :meth:`on_change`
listeners get the new snapshot after every change, so the server can broadcast it.
"""

from __future__ import annotations

import contextlib
import json
import logging
import random
import re
import threading
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from roco.bus import Emotion
from roco.personality.commands import Command, parse_command
from roco.personality.people import PeopleStore, Person, write_json_atomic
from roco.personality.render import (
    Company,
    PromptParts,
    band,
    render_system_prompt,
)
from roco.personality.settings import (
    SETTING_KEYS,
    SLIDERS,
    PersonalitySettings,
    matching_preset,
    preset,
    settings_from_config,
)

log = logging.getLogger(__name__)

ChangeListener = Callable[[dict[str, Any]], None]

# Tools that read the owner's personal data out loud. Hidden from the model while
# someone who may not hear it is in the room (spec §5 Guest, §7 Yousaf, §10).
PRIVATE_TOOL = re.compile(r"^(?:get|list|search|read)_.*(?:event|email|mail|calendar)")

# Clear signs the owner is genuinely upset (spec §2 hard rule: the bit drops). The brain
# then renders that turn with no jokes, whatever the settings. Measured on qwen2.5:7b: the
# prompt rule alone still let unhinged mode joke about a hospital visit about 1 time in 9.
DISTRESS = re.compile(
    r"\b(?:scared|terrified|frightened|panick?ing|panic attack|anxious|anxiety|depressed|"
    r"depression|can'?t cope|falling apart|crying|in tears|heartbroken|devastated|grieving|"
    r"passed away|died|dying|funeral|(?:in|to) (?:the )?hospital|a ?& ?e|ambulance|"
    r"diagnosed|cancer|broke up|breaking up|dumped me|lost my job|got fired|suicid\w*|"
    r"kill myself|hurt myself|self[- ]harm|(?:really|so|very) (?:upset|sad|worried|stressed|"
    r"down|low)|worried sick|not okay|not ok|miss (?:him|her) so much)\b",
    re.IGNORECASE,
)


def is_distressed(text: str) -> bool:
    return bool(DISTRESS.search(text or ""))


BAND_WORDS = ("off", "low", "medium", "high", "max")
LABELS = {
    "roast": "Roast", "flirt": "Flirting", "sass": "Sass", "chattiness": "Chattiness",
    "proactivity": "Proactivity", "energy": "Energy",
}


@dataclass(frozen=True)
class CommandResult:
    """What a matched voice command did. ``reply`` is spoken as-is, ``changed`` is the
    snapshot when something changed (None for a status question)."""

    command: Command
    reply: str
    emotion: Emotion
    changed: dict[str, Any] | None


class Personality:
    def __init__(
        self,
        base: PersonalitySettings | None = None,
        *,
        people: PeopleStore | None = None,
        state_path: Path | None = None,
        rng: random.Random | None = None,
    ) -> None:
        self._config_settings = base or PersonalitySettings()
        self.people = people or PeopleStore()
        self.state_path = state_path
        self._rng = rng or random.Random()
        self._lock = threading.RLock()
        self._listeners: list[ChangeListener] = []
        self._voice_values: dict[str, Any] = self._load_state()
        self._settings = self._config_settings.with_values(self._voice_values)
        self._company = Company()

    @classmethod
    def from_config(cls, config: Any, rng: random.Random | None = None) -> Personality:
        from roco.paths import resolve

        section = config.section("personality") if hasattr(config, "section") else {}
        state_file = section.get("file", "personality.json")
        people_file = section.get("people_file", "people.json")
        try:
            base = settings_from_config(section)
        except ValueError as exc:
            log.warning("Bad personality config (%s); using the default preset", exc)
            base = PersonalitySettings()
        people = PeopleStore(
            resolve(people_file) if people_file else None, section.get("people") or {}
        )
        return cls(base, people=people, state_path=resolve(state_file) if state_file else None,
                   rng=rng)

    # ------------------------------------------------------------------ state
    @property
    def settings(self) -> PersonalitySettings:
        """The owner's settings, before any guest tone-down."""
        with self._lock:
            return self._settings

    @property
    def company(self) -> Company:
        with self._lock:
            return self._company

    def snapshot(self) -> dict[str, Any]:
        """``{"personality": {...}}``: the ``values`` of a protocol ``settings`` message."""
        with self._lock:
            data: dict[str, Any] = self._settings.to_dict()
            data["preset"] = matching_preset(self._settings)
            data["company"] = {
                "people": [p.key for p in self._company.people],
                "guests": list(self._company.guests),
                "guest_mode": self._company.guest_mode,
            }
        return {"personality": data}

    def on_change(self, listener: ChangeListener) -> Callable[[], None]:
        """Call ``listener(snapshot)`` after every change. Returns an unsubscribe function."""
        with self._lock:
            self._listeners.append(listener)

        def unsubscribe() -> None:
            with self._lock, contextlib.suppress(ValueError):
                self._listeners.remove(listener)

        return unsubscribe

    # ------------------------------------------------------------------ changes
    def apply_settings(self, values: Mapping[str, Any]) -> dict[str, Any]:
        """Apply a protocol ``settings`` payload; returns the new snapshot.

        Accepts ``{"personality": {...}}`` or the inner dict: setting keys, ``"preset"``,
        ``"reset": true`` and ``"company": {"people": [...], "guests": [...],
        "guest_mode": bool}``. Unknown keys are ignored; bad levels raise ``ValueError``.
        """
        inner = values.get("personality", values) if isinstance(values, Mapping) else {}
        if not isinstance(inner, Mapping):
            raise ValueError("personality settings must be an object")
        with self._lock:
            if inner.get("reset"):
                self._reset_locked()
            if inner.get("preset"):
                self._set_locked(preset(str(inner["preset"])).to_dict())
            updates = {k: v for k, v in inner.items() if k in SETTING_KEYS}
            if updates:
                self._set_locked(updates)
            company = inner.get("company")
            if isinstance(company, Mapping):
                self._company = self._company_from(company)
        return self._changed()

    def set_preset(self, name: str) -> dict[str, Any]:
        return self.apply_settings({"preset": name})

    def set_company(self, company: Company) -> dict[str, Any]:
        with self._lock:
            self._company = company
        return self._changed()

    def _set_locked(self, values: Mapping[str, Any]) -> None:
        new = self._settings.with_values(values)
        self._voice_values.update(new.diff(self._config_settings))
        for key in list(self._voice_values):
            if getattr(new, key) == getattr(self._config_settings, key):
                del self._voice_values[key]
        self._settings = new
        self._save_state()

    def _reset_locked(self) -> None:
        self._voice_values = {}
        self._settings = self._config_settings
        self._save_state()

    def _company_from(self, data: Mapping[str, Any]) -> Company:
        people = []
        for key in data.get("people") or ():
            person = self.people.get(str(key)) or self.people.find(str(key))
            if person is not None and not person.is_owner:
                people.append(person)
        return Company(
            people=tuple(people),
            guests=tuple(str(g) for g in data.get("guests") or ()),
            guest_mode=bool(data.get("guest_mode", False)),
        )

    def _changed(self) -> dict[str, Any]:
        snap = self.snapshot()
        with self._lock:
            listeners = list(self._listeners)
        for listener in listeners:
            try:
                listener(snap)
            except Exception:
                log.exception("personality change listener failed")
        return snap

    # ------------------------------------------------------------------ persistence
    def _load_state(self) -> dict[str, Any]:
        if self.state_path is None or not self.state_path.is_file():
            return {}
        try:
            data = json.loads(self.state_path.read_text(encoding="utf-8"))
            values = data.get("settings", {}) if isinstance(data, dict) else {}
            self._config_settings.with_values(values)  # validates
            return {k: v for k, v in values.items() if k in SETTING_KEYS}
        except (OSError, ValueError, TypeError) as exc:
            log.warning("Ignoring unreadable personality file %s: %s", self.state_path, exc)
            return {}

    def _save_state(self) -> None:
        if self.state_path is None:
            return
        try:
            if self._voice_values:
                write_json_atomic(self.state_path, {"settings": dict(self._voice_values)})
            elif self.state_path.is_file():
                self.state_path.unlink()
        except OSError as exc:
            log.warning("Could not save personality settings to %s: %s", self.state_path, exc)

    # ------------------------------------------------------------------ brain hooks
    def system_prompt(self, name: str = "ROCO", user: str = "Adam", message: str = "") -> str:
        """The system prompt; ``message`` is this turn's user text (distress check)."""
        serious = is_distressed(message)
        with self._lock:
            parts = PromptParts(name=name, user=user, settings=self._settings,
                                company=self._company, serious=serious)
        return render_system_prompt(parts)

    def allowed_tool(self, tool_name: str) -> bool:
        """False for tools that read the owner's data out loud while it isn't private."""
        return self.company.private_ok or not PRIVATE_TOOL.match(tool_name)

    def filter_sentence(self, sentence: str) -> str:
        """Enforce name rules on spoken text: a person with ``use_name`` off is never named."""
        for person in self.people.all():
            if person.use_name or not person.nicknames:
                continue
            names = "|".join(re.escape(n) for n in (person.name, *person.aliases))
            sentence = re.sub(
                rf"\b(?:{names})\b",
                lambda _m, p=person: self._rng.choice(p.nicknames),
                sentence,
                flags=re.IGNORECASE,
            )
        return sentence

    def handle_command(self, text: str, user: str = "Adam") -> CommandResult | None:
        """If ``text`` is a personality command, apply it and return the spoken confirmation."""
        command = parse_command(text, self.people)
        if command is None:
            return None
        try:
            result = self._run(command, user)
        except ValueError as exc:
            log.warning("Personality command %s failed: %s", command, exc)
            return None
        if result is not None:
            log.info("Personality command %s -> %r", command, result.reply)
        return result

    # ------------------------------------------------------------------ commands
    def _run(self, cmd: Command, user: str) -> CommandResult | None:
        say = self._rng.choice
        before = self.settings
        if cmd.kind == "preset":
            snap = self.set_preset(cmd.value)
            line = say(PRESET_LINES[cmd.value]).format(user=user)
            return CommandResult(cmd, line, PRESET_EMOTION[cmd.value], snap)
        if cmd.kind == "reset":
            snap = self.apply_settings({"reset": True})
            line = say(("Back to my usual self.", "Factory settings. Well, my settings."))
            return CommandResult(cmd, line, Emotion.HAPPY, snap)
        if cmd.kind == "status":
            return CommandResult(cmd, self._status_line(cmd.key), Emotion.NEUTRAL, None)
        if cmd.kind in {"set", "adjust"}:
            current = getattr(before, cmd.key)
            if cmd.kind == "adjust":
                value = current + cmd.value
            elif cmd.value == -1:  # "roast on"
                value = max(current, 50)
            else:
                value = cmd.value
            value = max(0, min(100, value))
            if value == current:
                line = say(ALREADY_LINES).format(
                    user=user, what=f"{LABELS[cmd.key]} is already {BAND_WORDS[band(value)]}"
                )
                return CommandResult(cmd, line, Emotion.NEUTRAL, None)
            snap = self.apply_settings({cmd.key: value})
            return CommandResult(cmd, self._level_line(cmd.key, current, value, user),
                                 Emotion.EXCITED if value > current else Emotion.NEUTRAL, snap)
        if cmd.kind == "switch":
            if getattr(before, cmd.key) == cmd.value:
                line = say(ALREADY_LINES).format(user=user, what="Already done")
                return CommandResult(cmd, line, Emotion.NEUTRAL, None)
            snap = self.apply_settings({cmd.key: cmd.value})
            line = say(SWITCH_LINES[(cmd.key, cmd.value)]).format(user=user)
            return CommandResult(cmd, line, Emotion.HAPPY, snap)
        return self._run_company(cmd, user)

    def _run_company(self, cmd: Command, user: str) -> CommandResult | None:
        say = self._rng.choice
        company = self.company
        if cmd.kind == "guests":
            snap = self.set_company(replace(company, guest_mode=bool(cmd.value)))
            if cmd.value:
                return CommandResult(cmd, say(GUEST_ON_LINES), Emotion.HAPPY, snap)
            return CommandResult(cmd, say(ALONE_LINES).format(user=user), Emotion.HAPPY, snap)
        if cmd.kind == "alone":
            snap = self.set_company(Company())
            return CommandResult(cmd, say(ALONE_LINES).format(user=user), Emotion.HAPPY, snap)
        if cmd.kind == "arrive":
            if cmd.key == "known":
                person = self.people.get(cmd.value)
                assert person is not None
                people = tuple(p for p in company.people if p.key != person.key) + (person,)
                snap = self.set_company(replace(company, people=people))
                return CommandResult(cmd, self._arrival_line(person), Emotion.EXCITED, snap)
            guests = tuple(g for g in company.guests if g != cmd.value) + (cmd.value,)
            snap = self.set_company(replace(company, guests=guests))
            line = say(GUEST_HELLO_LINES).format(name=cmd.value)
            return CommandResult(cmd, line, Emotion.HAPPY, snap)
        if cmd.kind == "leave":
            if cmd.key == "known":
                person = self.people.get(cmd.value)
                if person is None or person not in company.people:
                    return None
                people = tuple(p for p in company.people if p.key != person.key)
                snap = self.set_company(replace(company, people=people))
                nick = say(person.spoken_names())
                return CommandResult(cmd, say(BYE_LINES).format(name=nick), Emotion.NEUTRAL, snap)
            if cmd.value not in company.guests:
                return None
            guests = tuple(g for g in company.guests if g != cmd.value)
            snap = self.set_company(replace(company, guests=guests))
            return CommandResult(cmd, say(BYE_LINES).format(name=cmd.value), Emotion.NEUTRAL,
                                 snap)
        return None

    def _arrival_line(self, person: Person) -> str:
        if not person.use_name and person.nicknames:
            nicks = person.nicknames
            a, b = nicks[0], nicks[-1]
            lines = (f"Oh look, it's {a}.", f"{b}'s back.", f"{a}! Or should I say {b}.",
                     f"Well, well. {a}-{b} has arrived. Hide the snacks.")
            return self._rng.choice(lines)
        return self._rng.choice((f"Oh hello, {person.name}.", f"{person.name}! Good to see you."))

    def _level_line(self, key: str, old: int, new: int, user: str) -> str:
        say = self._rng.choice
        if (key, new) in EXTREME_LINES:
            return say(EXTREME_LINES[(key, new)]).format(user=user)
        word = BAND_WORDS[band(new)]
        if new > old:
            return say(UP_LINES).format(label=LABELS[key], level=word, user=user)
        return say(DOWN_LINES).format(label=LABELS[key], level=word, user=user)

    def _status_line(self, key: str | None) -> str:
        s = self.settings
        if key:
            return f"{LABELS[key]} is on {BAND_WORDS[band(getattr(s, key))]}."
        name = matching_preset(s)
        levels = ", ".join(
            f"{LABELS[k].lower()} {BAND_WORDS[band(getattr(s, k))]}" for k in SLIDERS[:3]
        )
        if name:
            return f"I'm in {name} mode: {levels}."
        return f"Custom settings: {levels}."


# --------------------------------------------------------------------------- lines
# In-character confirmations (spec §2.1: "Oh, you asked for this."). Never "Yousaf".

PRESET_LINES = {
    "unhinged": ("Oh, you asked for this.", "Unhinged mode. Gloves off, {user}. Don't cry.",
                 "Right. No filter. You were warned."),
    "professional": ("Professional mode. Best behaviour, I promise.",
                     "Understood. Polished, polite and terribly well behaved."),
    "chill": ("Fine, I'll behave. Mostly.", "Chill mode. Deep breaths, everyone.",
              "Alright, alright. Toning it down."),
    "default": ("Back to my usual self.", "Default mode. The classic."),
}
PRESET_EMOTION = {
    "unhinged": Emotion.EXCITED, "professional": Emotion.NEUTRAL, "chill": Emotion.HAPPY,
    "default": Emotion.HAPPY,
}
EXTREME_LINES = {
    ("roast", 100): ("Roast level max. Oh, you asked for this.",
                     "Maximum roast. I hope your ego has a seatbelt."),
    ("roast", 0): ("No more roasts. You're safe. For now.", "Fine. I'll be nice. It'll be weird."),
    ("flirt", 100): ("Maximum charm. Try to keep up, {user}.",),
    ("flirt", 0): ("Strictly business, then.", "No flirting. Your loss."),
    ("sass", 100): ("Sass on max. Oh, this is going to be fun.",),
    ("sass", 0): ("Fine. I'll agree with everything. How dull.",),
    ("chattiness", 0): ("Short answers. Got it.",),
    ("chattiness", 100): ("Oh, you want me to talk more? Careful what you wish for.",),
    ("energy", 100): ("Energy on max! Let's go!",),
    ("energy", 0): ("Calm mode. Very zen.",),
}
UP_LINES = ("{label} up to {level}. Your funeral.", "{label} on {level}. Brace yourself.",
            "{label} turned up to {level}.")
DOWN_LINES = ("{label} down to {level}.", "Fine, {label} on {level}. I'll go easy on you.")
ALREADY_LINES = ("{what}, {user}.", "{what}. Keep up.")
SWITCH_LINES = {
    ("sensitive_topics_ok", True): ("Nothing's off-limits now. Interesting choice.",),
    ("sensitive_topics_ok", False): ("Sensitive topics are off the table.",),
    ("roast_guests", True): ("Guests get roasted too. Noted.",),
    ("roast_guests", False): ("Fine, I'll leave the guests alone.",),
    ("tone_down_with_guests", True): ("I'll tone it down when we've got company.",),
    ("tone_down_with_guests", False): ("No toning down for guests. They'll cope.",),
}
GUEST_ON_LINES = ("Guest mode on. Best behaviour.", "Company! I'll be on my best behaviour.")
GUEST_HELLO_LINES = ("Hello, {name}! Lovely to meet you.", "Oh, hi {name}. Welcome.")
ALONE_LINES = ("Just the two of us again.", "Alone at last, {user}.")
BYE_LINES = ("Bye, {name}.", "See you, {name}.")
