"""Personality voice commands, matched deterministically before the LLM sees the turn.

"ROCO, roast level max", "unhinged mode", "behave", "professional mode", "Yousaf's here",
"guest mode", "it's just us now"... :func:`parse_command` turns a transcript into a
:class:`Command` or ``None``. It only matches when the WHOLE utterance is a command (after
dropping the robot's name and polite filler), so "I was in unhinged mode last night" still
goes to the model.

Applying a command and the in-character confirmation live in
:class:`roco.personality.core.Personality`.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from roco.personality.people import PeopleStore

STEP = 25  # "more roasting" / "less sass"


@dataclass(frozen=True)
class Command:
    """kind: preset | reset | set | adjust | switch | arrive | leave | guests | alone | status"""

    kind: str
    key: str | None = None
    value: Any = None


# Spoken names of each slider, and the adjective that means "more of it".
SLIDER_WORDS: dict[str, tuple[str, ...]] = {
    "roast": ("roast", "roasts", "roasting", "roaster", "savagery", "savageness"),
    "flirt": ("flirt", "flirting", "flirtiness", "flirtation", "charm"),
    "sass": ("sass", "sassiness", "attitude", "cheek", "cheekiness"),
    "chattiness": ("chattiness", "chat", "chatter", "talking", "talkativeness", "rambling"),
    "proactivity": ("proactivity", "proactiveness", "nudges", "suggestions"),
    "energy": ("energy", "hype", "enthusiasm"),
}
ADJECTIVES: dict[str, tuple[str, ...]] = {
    "roast": ("savage", "mean", "brutal", "roasty"),
    "flirt": ("flirty", "flirtatious", "charming"),
    "sass": ("sassy", "cheeky", "smug"),
    "chattiness": ("chatty", "talkative"),
    "proactivity": ("proactive",),
    "energy": ("energetic", "hyper", "lively", "excited"),
}
WORD_TO_SLIDER = {w: k for k, words in SLIDER_WORDS.items() for w in words}
ADJ_TO_SLIDER = {w: k for k, words in ADJECTIVES.items() for w in words}

LEVEL_WORDS: dict[str, int] = {
    "max": 100, "maximum": 100, "maxed": 100, "full": 100, "full blast": 100, "eleven": 100,
    "a hundred": 100, "one hundred": 100, "hundred": 100, "high": 75, "medium": 50,
    "mid": 50, "normal": 50, "default": 50, "half": 50, "low": 25, "min": 0, "minimum": 0,
    "zero": 0, "off": 0, "none": 0, "nothing": 0, "on": -1,  # -1: "on" = at least medium
    "ten": 10, "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60,
    "seventy": 70, "eighty": 80, "ninety": 90,
}

_NAMES = r"(?:hey |ok |okay |oi |yo )?(?:roco|rocco|roko|rocko|robo|roco's|jarvis|hey jarvis)"
_LEAD = re.compile(rf"^(?:{_NAMES}\s+)+")
_TRAIL = re.compile(rf"(?:\s+{_NAMES})+$")
_FILLER = re.compile(
    r"^(?:(?:please|can you|could you|would you|will you|right|okay|ok|now|so|and|just|"
    r"i want|i'd like|let's|lets|go ahead and|time for|time to)\s+)+"
)
_TAIL_FILLER = re.compile(r"(?:\s+(?:please|now|then|thanks|thank you|for me|mate|again))+$")


def normalise(text: str) -> str:
    """Lowercase, plain spaces, no punctuation except apostrophes, no name or filler."""
    t = text.lower().replace("’", "'").replace("%", " percent")
    t = re.sub(r"[^a-z0-9' ]+", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    t = _LEAD.sub("", t)
    t = _TRAIL.sub("", t)
    t = _FILLER.sub("", t)
    t = _TAIL_FILLER.sub("", t)
    return t.strip()


def parse_level(words: str) -> int | None:
    words = words.strip().removesuffix(" percent").removesuffix(" per cent").strip()
    if words.isdigit():
        return max(0, min(100, int(words)))
    return LEVEL_WORDS.get(words)


_SLIDER = "|".join(sorted(WORD_TO_SLIDER, key=len, reverse=True))
_ADJ = "|".join(sorted(ADJ_TO_SLIDER, key=len, reverse=True))
_LEVEL = r"(?:\d{1,3}(?: percent| per cent)?|" + "|".join(
    sorted((re.escape(w) for w in LEVEL_WORDS), key=len, reverse=True)
) + ")"
_THE = r"(?:(?:the|your|ur|my) )?"

PRESET_ALIASES = {
    "unhinged": "unhinged", "savage": "unhinged", "chaos": "unhinged", "feral": "unhinged",
    "chill": "chill", "chilled": "chill", "relaxed": "chill",
    "professional": "professional", "work": "professional", "demo": "professional",
    "polite": "professional", "default": "default", "normal": "default",
    "standard": "default",
}
_PRESET = "|".join(PRESET_ALIASES)

_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    # presets
    (re.compile(rf"(?:switch to |go into |go |enter |activate |turn on |use |be )?"
                rf"{_THE}(?P<preset>{_PRESET}) (?:mode|preset|personality|setting)(?: on)?"),
     "preset"),
    (re.compile(r"(?:go|get|be|going|getting) (?P<preset>unhinged|feral|savage)"), "preset"),
    (re.compile(r"(?:be (?P<preset>professional|chill|polite|normal))"), "preset"),
    (re.compile(r"(?:behave|behave yourself|chill out|tone it down|calm it down|"
                r"dial it back|easy tiger)"), "behave"),
    (re.compile(r"(?:back to normal|be yourself|reset (?:your )?personality|"
                r"reset (?:your )?settings|personality reset|go back to normal)"), "reset"),
    # status
    (re.compile(rf"(?:what(?:'s| is) {_THE}(?P<key>{_SLIDER}) (?:level|setting)|"
                rf"what level is {_THE}(?P<key2>{_SLIDER}) (?:on|at))"), "status"),
    (re.compile(r"(?:what mode are you in|what(?:'s| is) your (?:personality|mode|setting)s?"
                r"(?: set to| on)?|personality (?:status|check))"), "status"),
    # switches
    (re.compile(r"(?:(?:turn |switch )?sensitive (?:topics|stuff|subjects) (?P<on>on|off)|"
                r"(?P<on2>allow|enable|disable|block|no) sensitive (?:topics|stuff|subjects)|"
                r"nothing(?:'s| is) off limits)"), "sensitive"),
    (re.compile(r"(?P<neg>don't |dont |do not |stop |no )?roast(?:ing)? (?:the |my )?"
                r"guests(?: too)?"),
     "roast_guests"),
    # company
    (re.compile(r"(?:guest mode(?: (?P<on>on|off))?|(?:i've |i )?(?:got|have) (?:company|guests|"
                r"people over|friends over|a guest|a friend over)|we(?:'ve| have) (?:got )?"
                r"(?:company|guests)|someone(?:'s| is) here|there(?:'s| is) someone here)"),
     "guests"),
    (re.compile(r"(?:it's just (?:us|me|the two of us)|it is just (?:us|me)|we're alone|"
                r"we are alone|everyone(?:'s| has| is) (?:gone|left)|"
                r"(?:the )?guests? (?:have |has |'s |are |is )?(?:gone|left)|they've (?:gone|left)|"
                r"they (?:have )?(?:gone|left)|just me(?: now)?)"), "alone"),
    # levels: "roast level max", "set the sass to 80", "flirting off", "max roast", "no flirting"
    (re.compile(rf"(?:set |put |turn |make )?{_THE}(?P<key>{_SLIDER})(?: level| levels| setting|"
                rf" mode| meter)?(?: to| at| on| is)? (?P<level>{_LEVEL})"), "set"),
    (re.compile(rf"(?P<level>max|maximum|full|zero|no|minimum|min) (?P<key>{_SLIDER})"), "set"),
    (re.compile(r"(?:stop|quit|no more) (?P<key>roasting|flirting|roasts|sass|rambling)"
                r"(?: me)?"), "stop"),
    # adjust: "more roasting", "turn the sass up", "tone down the flirting", "be more savage"
    (re.compile(rf"(?:more|crank up|turn up|dial up|up|bump up|ramp up)(?: the| your)? "
                rf"(?P<key>{_SLIDER})"), "up"),
    (re.compile(rf"(?:turn |crank |dial |bump |ramp )?{_THE}(?P<key>{_SLIDER}) up"), "up"),
    (re.compile(rf"(?:less|tone down|turn down|dial down|ease off|ease up on|ease off on)"
                rf"(?: the| your)? (?P<key>{_SLIDER})"), "down"),
    (re.compile(rf"(?:turn |tone |dial |ease )?{_THE}(?P<key>{_SLIDER}) down"), "down"),
    (re.compile(rf"be (?:a (?:bit|lot|little) )?more (?P<adj>{_ADJ})"), "up"),
    (re.compile(rf"be (?:a (?:bit|lot|little) )?less (?P<adj>{_ADJ})"), "down"),
    (re.compile(r"(?:talk|say) (?P<dir>more|less)"), "talk"),
]

_ARRIVE = re.compile(
    r"(?:(?:my )?(?:friend|mate|pal|buddy) )?(?P<name>[a-z][a-z']*(?: [a-z][a-z']*)?)"
    r"(?:'s| is| has) (?:here|back|in the room|over|arrived|come over|joining us|with me|"
    r"in|just arrived)"
)
_LEAVE = re.compile(
    r"(?P<name>[a-z][a-z']*(?: (?!(?:has|is|just)\b)[a-z][a-z']*)?)(?:'s| is| has| just)?"
    r" (?:left|gone|gone home|"
    r"out|leaving|heading off|off)"
)
# Words that are not someone's name ("it's here", "dinner's here", "the package is here").
_NOT_NAME_WORDS = (
    "it this that he she they there here what who where everyone everybody someone somebody "
    "nobody dinner lunch breakfast food pizza tea coffee package parcel delivery post mail "
    "the a an time winter summer friday weekend it's that's my your our their i i'm im we "
    "we're you you're me us roco"
)
NOT_NAMES = frozenset(_NOT_NAME_WORDS.split())


def _slider(match: re.Match[str]) -> str | None:
    groups = match.groupdict()
    key = groups.get("key") or groups.get("key2")
    if key:
        return WORD_TO_SLIDER.get(key)
    adj = groups.get("adj")
    return ADJ_TO_SLIDER.get(adj) if adj else None


def _person(name: str, original: str, people: PeopleStore | None) -> tuple[str, str] | None:
    """("known", key) or ("guest", Display name) for a spoken name, or None."""
    first = name.split()[0]
    if first in NOT_NAMES or name in NOT_NAMES:
        return None
    if people is not None:
        known = people.find(name)
        if known is not None and not known.is_owner:
            return ("known", known.key)
        if known is not None:
            return None  # "Adam's here": that's the owner, nothing to change
    # Unknown: only a single word the speech-to-text wrote with a capital letter.
    if " " in name:
        return None
    capitalised = re.search(rf"\b{re.escape(first.capitalize())}\b", original)
    if not capitalised:
        return None
    return ("guest", first.capitalize())


def parse_command(text: str, people: PeopleStore | None = None) -> Command | None:
    t = normalise(text)
    if not t or len(t.split()) > 9:
        return None
    for pattern, kind in _PATTERNS:
        m = pattern.fullmatch(t)
        if not m:
            continue
        g = m.groupdict()
        if kind == "preset":
            return Command("preset", value=PRESET_ALIASES[g["preset"]])
        if kind == "behave":
            return Command("preset", value="chill")
        if kind == "reset":
            return Command("reset")
        if kind == "status":
            return Command("status", key=_slider(m))
        if kind == "sensitive":
            word = g.get("on") or g.get("on2") or "on"
            return Command("switch", "sensitive_topics_ok", word in {"on", "allow", "enable"})
        if kind == "roast_guests":
            return Command("switch", "roast_guests", not g.get("neg"))
        if kind == "guests":
            return Command("guests", value=g.get("on") != "off")
        if kind == "alone":
            return Command("alone")
        if kind == "set":
            key, level = _slider(m), g["level"]
            value = 0 if level == "no" else parse_level(level)
            if key is None or value is None:
                return None
            return Command("set", key, value)
        if kind == "stop":
            return Command("set", WORD_TO_SLIDER[g["key"]], 0)
        if kind in {"up", "down"}:
            key = _slider(m)
            return Command("adjust", key, STEP if kind == "up" else -STEP) if key else None
        if kind == "talk":
            return Command("adjust", "chattiness", STEP if g["dir"] == "more" else -STEP)
    m = _ARRIVE.fullmatch(t)
    if m:
        who = _person(m["name"], text, people)
        if who:
            return Command("arrive", who[0], who[1])
    m = _LEAVE.fullmatch(t)
    if m:
        who = _person(m["name"], text, people)
        if who:
            return Command("leave", who[0], who[1])
    return None
